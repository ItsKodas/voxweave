"""Calibration of live mode to the person speaking.

The voice models only learn the target voice, so nothing here is trained. A short recording of the speaker (a few
seconds of quiet, then some reading) is measured instead: how high they speak, how loud, and how noisy the
microphone is when nobody talks. Live mode then uses three things from it:

- a pitch shift that moves the speaker's usual pitch onto the target voice's usual pitch,
- a quiet gate: while the microphone only carries room noise, live mode outputs silence instead of converting
  the noise into voice-like sound,
- an input gain that brings a quiet speaker up to an ordinary speaking level.

The gate and the gain are applied by wrapping the voice engine, so the audio thread, the processor and the router
are untouched, and anything that goes wrong inside still ends in silence there.
"""
import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from aivoice.audiofile import read_wav, write_wav
from aivoice.config import SAMPLE_RATE
from aivoice.dsp import remove_rumble
from aivoice.engines.f0 import extract_f0

QUIET_SECONDS = 3.0  # the start of the recording, during which the speaker says nothing
TEST_SECONDS = 4.0  # then "testing, one, two, three": the microphone level the input boost is set from
# Then reading a paragraph aloud (the pitch, and the reading deep calibration converts), for as long as it takes:
# the speaker presses Done once at least MIN_READ_SECONDS have been read, and the take ends by itself at MAX_SECONDS.
MIN_READ_SECONDS = 8.0
MAX_SECONDS = 75.0
TEST_PHRASE = "Testing, one, two, three"
MIN_VOICED_SECONDS = 3.0
TARGET_LEVEL_DB = -20.0  # the level of ordinary close speech (RMS of voiced sound)
# Up to 24 dB of boost: a quiet microphone is brought up to an ordinary level (never so far that its loudest
# moment clips). It was 12 dB, which left quiet microphones too quiet for the model.
MIN_GAIN, MAX_GAIN = 0.5, 16.0
# Speech must stand this far above the room (its usual level against the room's), or the gate cannot tell them
# apart. It compared the quietest speech before, which refused quiet microphones whose hiss is relatively loud.
MIN_SPEECH_ABOVE_ROOM_DB = 10.0
GATE_MARGIN_DB = 6.0
GATE_FLOOR_DB = -70.0
RANGE_SECONDS = 120.0  # of a voice's clips measured for its pitch range
# The range was the 5th to 95th percentile before; the edges of a voice's recordings (up to the 99th) are pitches
# its model still knows. A range remembered by an older version is measured again.
RANGE_VERSION = 2
HANGOVER_SECONDS = 0.3  # speech that ended this recently still opens the gate
FRAME_SECONDS = 0.02
MAX_SHIFT = 24
ABANDONED_AFTER = 90.0  # a recording nobody finished is given up after this long
# One of these is read at each calibration, never the same one twice in a row: everyday sentences with questions,
# numbers and a range of sounds, each about 20 seconds aloud.
READINGS = (
    "The ferry leaves the harbour at seven each morning. If the wind is calm, you can see the far shore long before "
    "the engines slow, and the gulls follow the boat the whole way across. My grandfather kept a small workshop behind "
    "the house. He fixed clocks, radios and bicycles there, and he never once threw away a screw that might be useful "
    "again.",
    "Have you ever tried to bake bread on a cold day? The dough sulks in the corner and refuses to rise, no matter how "
    "long you wait. My neighbour swears by a warm oven with the light switched on, and honestly, she might be right. "
    "Last week her loaf came out golden, crisp and nearly twice the size of mine.",
    "We missed the last bus by about thirty seconds, so we walked home through the park instead. It was quieter than "
    "I expected: just the hum of the streetlights, a dog barking somewhere far away, and our shoes crunching on the "
    "gravel. By the time we reached the gate, it had started to rain, gently at first, then all at once.",
    "Before you start, check that the batteries are charged and the lens is clean. Hold the camera with both hands, "
    "keep your elbows tucked in, and breathe out slowly as you press the button. Most blurry photos come from rushing "
    "that last moment. Take three or four shots of the same scene, then choose the sharpest one later.",
    "Honestly, I thought the quiz would be easy. The first question was about the capital of Australia, and half the "
    "room shouted Sydney. It's Canberra, by the way. Then came a round on famous inventions, and nobody knew who "
    "built the first working television. We finished fourth out of nine teams, which still felt like a win.",
    "The old lighthouse stands at the end of a narrow path, surrounded by wild grass and yellow flowers. Every summer, "
    "visitors climb the hundred and twelve steps to the top, where the view stretches for miles across the bay. On a "
    "clear evening you can watch the fishing boats drift home as the sky turns orange, then pink, then deep blue.",
)
READING = READINGS[0]


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    pitch_hz: float  # the speaker's usual pitch (median of voiced sound)
    pitch_low_hz: float
    pitch_high_hz: float
    level_db: float  # RMS of voiced sound, dB relative to full scale
    noise_db: float  # how loud the microphone is when nobody speaks
    gate_db: float
    input_gain: float
    device: str
    created: float
    test_level_db: float | None = None  # the level of "testing, one, two, three", which set the input boost


def _frame_db(audio: np.ndarray, sr: int, seconds: float = FRAME_SECONDS) -> np.ndarray:
    n = max(1, int(sr * seconds))
    frames = len(audio) // n
    if frames == 0:
        return np.zeros(0, dtype=np.float64)
    x = np.asarray(audio[: frames * n], dtype=np.float64).reshape(frames, n)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(x * x, axis=1)), 1e-10))


def pitch_of(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """The pitch of each 10 ms of audio (0 where there is no voice) and the 16 kHz audio it was measured on."""
    wav16 = soxr.resample(np.nan_to_num(np.asarray(audio, dtype=np.float32)), sr, 16000).astype(np.float32)
    return extract_f0(wav16, "dio", max(1, len(wav16) // 160)), wav16


def _usual_pitch(f0: np.ndarray) -> tuple[float, float, float] | None:
    voiced = f0[f0 > 0]
    if len(voiced) < int(MIN_VOICED_SECONDS * 100):
        return None
    low, mid, high = np.percentile(voiced, [5, 50, 95])
    return float(mid), float(low), float(high)


def measure_speech(audio: np.ndarray, sr: int) -> dict | None:
    """The speaker's usual pitch, pitch range, speaking level and the input boost for it, from speech alone (live
    learning's pool of speech, which has no quiet part). None when too little of it has a pitch."""
    audio = remove_rumble(np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1)), sr)
    if len(audio) < int(MIN_VOICED_SECONDS * sr):
        return None
    f0, wav16 = pitch_of(audio, sr)
    usual = _usual_pitch(f0)
    if usual is None:
        return None
    frames = _frame_db(wav16, 16000, 0.01)
    voiced = frames[: len(f0)][f0[: len(frames)] > 0]
    if len(voiced) == 0:
        return None
    level = float(10.0 * np.log10(np.mean(10.0 ** (voiced / 10.0))))
    peak = float(np.percentile(np.abs(audio), 99.9))
    gain = 10.0 ** ((TARGET_LEVEL_DB - level) / 20.0)
    if peak > 0:
        gain = min(gain, 0.98 / peak)
    pitch, low, high = usual
    return {"pitch_hz": round(float(pitch), 1), "low_hz": round(float(low), 1), "high_hz": round(float(high), 1),
            "level_db": round(level, 1), "input_gain": round(float(min(MAX_GAIN, max(MIN_GAIN, gain))), 3)}


def _voiced_level(audio: np.ndarray, sr: int) -> tuple[float, np.ndarray, np.ndarray] | None:
    """(level of the voiced sound in dB, the voiced frames' levels, the pitch track), or None with no voice.
    The pitch is tracked on a copy brought to an ordinary level, so a quiet microphone is heard as well as a loud one."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 1e-6:
        return None
    f0, _ = pitch_of(audio * np.float32(0.5 / peak), sr)
    wav16 = soxr.resample(audio, sr, 16000).astype(np.float32)
    frames = _frame_db(wav16, 16000, 0.01)
    voiced = frames[: len(f0)][f0[: len(frames)] > 0]
    if len(voiced) == 0:
        return None
    return float(10.0 * np.log10(np.mean(10.0 ** (voiced / 10.0)))), voiced, f0


def review_check(audio: np.ndarray, sr: int) -> dict:
    """The mic check alone (QUIET_SECONDS of quiet, then the test phrase): the room's level, the phrase's level and
    peak as the microphone picked it up, and the boost suggested for it. Raises CalibrationError when the phrase
    was not heard or is barely louder than the room."""
    audio = remove_rumble(np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1)), sr)
    split, end = int(QUIET_SECONDS * sr), int((QUIET_SECONDS + TEST_SECONDS) * sr)
    quiet = _frame_db(audio[int(0.3 * sr): split - int(0.2 * sr)], sr)
    noise = float(np.percentile(quiet, 95)) if len(quiet) else GATE_FLOOR_DB
    test = _voiced_level(audio[split:end], sr)
    if test is None or len(test[1]) < 30:
        raise CalibrationError(f'The test phrase was not heard. Say "{TEST_PHRASE.lower()}" in your normal voice, at '
                               "your usual distance from the microphone.")
    if float(np.median(test[1])) - noise < MIN_SPEECH_ABOVE_ROOM_DB:
        raise CalibrationError("The room is too loud for your microphone, or you spoke during the quiet part: your "
                               "speech is barely louder than the quiet. Try again somewhere quieter, or move closer.")
    peak = float(np.max(np.abs(audio[split:end])))
    gain = 10.0 ** ((TARGET_LEVEL_DB - test[0]) / 20.0)
    if peak > 0:
        gain = min(gain, 0.98 / peak)
    return {"noise_db": round(noise, 1), "test_level_db": round(test[0], 1), "test_peak": round(peak, 4),
            "suggested_gain": round(float(min(MAX_GAIN, max(MIN_GAIN, gain))), 2),
            "min_gain": MIN_GAIN, "max_gain": MAX_GAIN, "target_db": TARGET_LEVEL_DB}


def analyse(audio: np.ndarray, sr: int, device: str, now: float, gain: float | None = None) -> Profile:
    """Measures a calibration recording: QUIET_SECONDS of silence (the room), TEST_SECONDS of "testing, one, two,
    three" (the microphone level, which sets the input boost), then reading aloud (the pitch)."""
    audio = remove_rumble(np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1)), sr)
    split, read_from = int(QUIET_SECONDS * sr), int((QUIET_SECONDS + TEST_SECONDS) * sr)
    if len(audio) < read_from + int(MIN_VOICED_SECONDS * sr):
        raise CalibrationError("The recording is too short. Stay quiet, say the test phrase, then read the text aloud.")
    quiet = _frame_db(audio[int(0.3 * sr): split - int(0.2 * sr)], sr)  # edges trimmed: a click, an early start
    noise = float(np.percentile(quiet, 95)) if len(quiet) else GATE_FLOOR_DB
    test = _voiced_level(audio[split:read_from], sr)
    if test is None or len(test[1]) < 30:  # under 0.3 s of voice
        raise CalibrationError(f'The test phrase was not heard. When asked, say "{TEST_PHRASE.lower()}" in your '
                               "normal voice, at your usual distance from the microphone.")
    test_level = test[0]
    speech = audio[read_from:]
    reading = _voiced_level(speech, sr)
    usual = None if reading is None else _usual_pitch(reading[2])
    if usual is None:
        raise CalibrationError("Not enough speech was heard. Read the text aloud for the whole time.")
    pitch, low, high = usual
    level, voiced = reading[0], reading[1]
    if float(np.median(voiced)) - noise < MIN_SPEECH_ABOVE_ROOM_DB:
        raise CalibrationError("The room is too loud for your microphone, or you spoke during the quiet part: your "
                               "speech is barely louder than the quiet. Try again somewhere quieter, or move closer.")
    quiet_speech = float(np.percentile(voiced, 10))
    gate = max(GATE_FLOOR_DB, min(noise + GATE_MARGIN_DB, quiet_speech - GATE_MARGIN_DB))
    if gain is None:
        peak = float(np.percentile(np.abs(audio[split:]), 99.9))
        gain = 10.0 ** ((TARGET_LEVEL_DB - test_level) / 20.0)
        if peak > 0:
            gain = min(gain, 0.98 / peak)  # never into clipping
    gain = float(min(MAX_GAIN, max(MIN_GAIN, gain)))
    return Profile(round(pitch, 1), round(low, 1), round(high, 1), round(level, 1), round(noise, 1),
                   round(float(gate), 1), round(gain, 3), device[:120], now, round(test_level, 1))


def suggest_shift(speaker_hz: float, target_hz: float) -> int:
    """Whole semitones that move the speaker's usual pitch onto the target's."""
    if speaker_hz <= 0 or target_hz <= 0:
        raise CalibrationError("A pitch must be positive")
    return int(max(-MAX_SHIFT, min(MAX_SHIFT, round(12.0 * math.log2(target_hz / speaker_hz)))))


class GatedEngine:
    """A voice engine that outputs silence while the microphone is quiet and lifts a quiet speaker's level.

    The gate looks at the newest part of the window (what is about to be played, plus a short stretch before it)
    and opens when any 20 ms of it is louder than the threshold, so a soft word start or end inside a spoken
    chunk is never cut. Closed, it returns silence without running the model.
    """

    def __init__(self, inner, gate_db: float | None, gain: float = 1.0):
        self._inner = inner
        self._gate_db = gate_db
        self._gain = float(gain)

    def __getattr__(self, name):  # anything else the engine offers (its noise position) is passed through
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in ("_inner", "_gate_db", "_gain"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def convert(self, audio, sample_rate, pitch_shift, params):
        audio = remove_rumble(np.asarray(audio, dtype=np.float32), sample_rate)
        if self._gate_db is not None:
            recent = int(sample_rate * ((params.chunk_ms + params.crossfade_ms) / 1000.0 + HANGOVER_SECONDS))
            levels = _frame_db(np.nan_to_num(audio[-recent:]), sample_rate)
            if len(levels) and float(levels.max()) < self._gate_db:
                return np.zeros_like(audio)
        if self._gain == 1.0:
            return self._inner.convert(audio, sample_rate, pitch_shift, params)
        out = self._inner.convert(np.clip(audio * np.float32(self._gain), -1.0, 1.0), sample_rate, pitch_shift, params)
        if getattr(params, "match_input_level", True):
            # The boost lifted the speaker for the model; the voice comes out at the speaker's own level.
            out = (np.asarray(out, dtype=np.float32) / np.float32(self._gain)).astype(np.float32)
        return out


class Calibration:
    """The speaker's profile on disk, the recording that makes it, and the target voices' usual pitch."""

    def __init__(self, data: Path, recorder, list_devices: Callable, library=None,
                 clock: Callable[[], float] = time.time):
        self._folder = data / "calibration"
        self._recorder = recorder
        self._list_devices = list_devices
        self._library = library
        self._clock = clock
        self._lock = threading.Lock()
        self._started: float | None = None
        self._reading = 0
        # The calibration in two takes: the mic check, reviewed by the speaker (played back, boost set by ear),
        # then the reading. Between them nothing records.
        self._take: str | None = None  # "check" or "reading" while recording
        self._check_audio: np.ndarray | None = None
        self._review: dict | None = None
        self._chosen_gain: float | None = None
        self._device_index: int | None = None
        self._device = ""
        self._error: str | None = None

    # ---- the profile

    def profile(self) -> Profile | None:
        try:
            raw = json.loads((self._folder / "profile.json").read_text(encoding="utf-8"))
            profile = Profile(**raw)
            numbers = (profile.pitch_hz, profile.level_db, profile.noise_db, profile.gate_db, profile.input_gain)
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers):
                return None
            if not (profile.pitch_hz > 0 and MIN_GAIN <= profile.input_gain <= MAX_GAIN):
                return None
            return profile
        except (OSError, ValueError, TypeError):
            return None

    def forget(self) -> None:
        for name in ("profile.json", "sample.wav", "test.wav"):
            try:
                (self._folder / name).unlink()
            except FileNotFoundError:
                pass

    def save_learned(self, learned: dict, sample: np.ndarray | None) -> Profile:
        """Makes what live learning measured the calibration: the speaker's pitch, range, level and boost from how
        they talk live (the quiet gate and noise floor stay as calibrated), and recent speech as the reading deep
        calibration converts."""
        with self._lock:
            profile = self.profile()
            if profile is None:
                raise CalibrationError("Calibrate first: live learning refines a calibration")
            updated = Profile(learned["pitch_hz"], learned["low_hz"], learned["high_hz"], learned["level_db"],
                              profile.noise_db, profile.gate_db, learned["input_gain"], profile.device, self._clock())
            self._folder.mkdir(parents=True, exist_ok=True)
            part = self._folder / "profile.json.part"
            part.write_text(json.dumps(asdict(updated)), encoding="utf-8")
            part.replace(self._folder / "profile.json")
            if sample is not None and len(sample) >= int(MIN_VOICED_SECONDS * SAMPLE_RATE):
                write_wav(self._folder / "sample.wav.part", np.asarray(sample, dtype=np.float32), SAMPLE_RATE)
                (self._folder / "sample.wav.part").replace(self._folder / "sample.wav")
            return updated

    def test_audio(self, gain: float = 1.0) -> np.ndarray | None:
        """The "testing, one, two, three" of the mic check being reviewed, else of the last calibration, as the
        microphone picked it up times `gain` (clipped, as the boost would clip it)."""
        with self._lock:
            pending = self._check_audio
        if pending is not None:
            audio = pending[int(QUIET_SECONDS * SAMPLE_RATE): int((QUIET_SECONDS + TEST_SECONDS) * SAMPLE_RATE)]
        else:
            path = self._folder / "test.wav"
            if self.profile() is None or not path.is_file():
                return None
            try:
                audio, sr = read_wav(path)
            except (ValueError, OSError):
                return None
            if sr != SAMPLE_RATE:
                audio = soxr.resample(audio, sr, SAMPLE_RATE).astype(np.float32)
        return np.clip(np.asarray(audio, dtype=np.float32) * np.float32(gain), -1.0, 1.0)

    def sample(self) -> np.ndarray | None:
        """The speaker's calibration reading (the part after the quiet), at SAMPLE_RATE. None when there is none:
        calibrations made before the reading was kept have only their measurements."""
        path = self._folder / "sample.wav"
        if self.profile() is None or not path.is_file():
            return None
        try:
            audio, sr = read_wav(path)
        except (ValueError, OSError):
            return None
        if sr != SAMPLE_RATE:
            audio = soxr.resample(audio, sr, SAMPLE_RATE).astype(np.float32)
        return audio

    def wrap(self, engine):
        """The engine with the speaker's gate and gain, or the engine itself when there is no profile."""
        profile = self.profile()
        return engine if profile is None else GatedEngine(engine, profile.gate_db, profile.input_gain)

    # ---- the recording

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active_locked()

    def _active_locked(self) -> bool:
        if self._started is None:
            return False
        if self._clock() - self._started > ABANDONED_AFTER:  # the window was closed mid-recording
            self._recorder.cancel()
            self._started = None
            self._error = "The calibration recording was not finished and has been dropped."
            return False
        return True

    def start(self, device: int) -> None:
        with self._lock:
            if self._active_locked():
                raise CalibrationError("A calibration recording is already running")
            source = next((d for d in self._list_devices() if d.index == device), None)
            if source is None:
                raise CalibrationError(f"Audio device {device} is no longer available")
            if source.max_input < 1:
                raise CalibrationError(f"{source.name} is not an input device")
            self._recorder.start(device)
            self._take, self._device_index = "check", device
            self._check_audio = self._review = self._chosen_gain = None
            others = [i for i in range(len(READINGS)) if i != self._reading] or [0]
            self._reading = random.choice(others)  # a different paragraph from last time
            self._started, self._device, self._error = self._clock(), source.name, None

    def cancel(self) -> None:
        with self._lock:
            if self._started is not None:
                self._recorder.cancel()
                self._started = None
            self._take = None
            self._check_audio = self._review = self._chosen_gain = None

    def _keep_failed(self, audio) -> None:
        try:  # kept, so why it failed can be looked into; the next failure replaces it
            self._folder.mkdir(parents=True, exist_ok=True)
            write_wav(self._folder / "failed-attempt.wav",
                      np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1)), SAMPLE_RATE)
        except OSError:
            pass

    def check(self) -> dict:
        """Ends the mic check and measures it for the speaker to review: play it back, set the boost by ear."""
        with self._lock:
            if not self._active_locked() or self._take != "check":
                raise CalibrationError("No mic check is being recorded")
            audio = self._recorder.stop()
            self._started, self._take = None, None
            check = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))[
                : int((QUIET_SECONDS + TEST_SECONDS) * SAMPLE_RATE)]
            try:
                review = review_check(check, SAMPLE_RATE)
            except CalibrationError as exc:
                self._error = str(exc)
                self._keep_failed(check)
                raise
            self._check_audio, self._review, self._error = check, review, None
            return review

    def read(self, gain: float) -> None:
        """Starts the reading, with the boost the speaker chose for the reviewed mic check."""
        with self._lock:
            if self._review is None or self._check_audio is None:
                raise CalibrationError("Record the mic check first")
            if not (isinstance(gain, (int, float)) and MIN_GAIN <= float(gain) <= MAX_GAIN):
                raise CalibrationError(f"The boost must be between {MIN_GAIN:g}x and {MAX_GAIN:g}x")
            if self._device_index is None:
                raise CalibrationError("Record the mic check first")
            self._recorder.start(self._device_index)
            self._chosen_gain, self._take, self._started = float(gain), "reading", self._clock()

    def finish(self) -> Profile:
        with self._lock:
            if not self._active_locked():
                raise CalibrationError("No calibration recording is running")
            audio = self._recorder.stop()
            self._started = None
            chosen = None
            if self._take == "reading" and self._check_audio is not None:
                # The two takes put together: the reviewed mic check, then the reading, at the boost chosen by ear.
                audio = np.concatenate([self._check_audio,
                                        np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))])
                chosen = self._chosen_gain
            self._take = None
            self._check_audio = self._review = self._chosen_gain = None
            try:
                profile = analyse(audio, SAMPLE_RATE, self._device, self._clock(), gain=chosen)
                self._folder.mkdir(parents=True, exist_ok=True)
                part = self._folder / "profile.json.part"
                part.write_text(json.dumps(asdict(profile)), encoding="utf-8")
                part.replace(self._folder / "profile.json")
                # The reading itself is kept too: deep calibration converts it with each voice to compare settings.
                sample = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))[
                    int((QUIET_SECONDS + TEST_SECONDS) * SAMPLE_RATE):]
                write_wav(self._folder / "sample.wav.part", sample, SAMPLE_RATE)
                (self._folder / "sample.wav.part").replace(self._folder / "sample.wav")
                # And the mic check as the microphone picked it up, to play back.
                check = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))[
                    int(QUIET_SECONDS * SAMPLE_RATE): int((QUIET_SECONDS + TEST_SECONDS) * SAMPLE_RATE)]
                write_wav(self._folder / "test.wav.part", check, SAMPLE_RATE)
                (self._folder / "test.wav.part").replace(self._folder / "test.wav")
            except CalibrationError as exc:
                self._error = str(exc)
                self._keep_failed(audio)
                raise
            except OSError as exc:
                self._error = f"The calibration could not be saved: {exc}"
                raise CalibrationError(self._error) from exc
            return profile

    def status(self) -> dict:
        with self._lock:
            active = self._active_locked()
            recorded = self._recorder.status() if active else {"seconds": 0.0, "level": 0.0}
            seconds = float(recorded["seconds"])
            profile = self.profile()
            if active and self._take == "reading":
                phase, read_from = "reading", 0.0
            elif active:
                phase = ("quiet" if seconds < QUIET_SECONDS else "testing" if seconds < QUIET_SECONDS + TEST_SECONDS
                         else "reading")
                read_from = QUIET_SECONDS + TEST_SECONDS
            else:
                phase, read_from = ("review" if self._review is not None else None), 0.0
            return {"recording": active, "seconds": seconds, "level": float(recorded["level"]),
                    "phase": phase, "read_from": read_from, "review": self._review,
                    "quiet_seconds": QUIET_SECONDS, "test_seconds": TEST_SECONDS, "test_phrase": TEST_PHRASE,
                    "min_read_seconds": MIN_READ_SECONDS, "total_seconds": MAX_SECONDS,
                    "text": READINGS[self._reading], "error": self._error,
                    "has_sample": profile is not None and (self._folder / "sample.wav").is_file(),
                    "profile": None if profile is None else asdict(profile)}

    # ---- the target voices

    def voice_pitch(self, voice_id: str) -> float | None:
        """The usual pitch of a voice, measured from its clips (up to two minutes of them). None when it has none."""
        stats = self._voice_stats(voice_id)
        return None if stats is None else stats[0]

    def voice_pitch_range(self, voice_id: str) -> tuple[float, float] | None:
        """The pitch range the voice's recordings cover (1st to 99th percentile, Hz), which is what a model
        trained on them knows. None when the voice has no recordings to measure."""
        stats = self._voice_stats(voice_id)
        return None if stats is None else (stats[1], stats[2])

    def _voice_stats(self, voice_id: str) -> tuple[float, float, float] | None:
        """(median, low, high) pitch of a voice's clips, remembered until its clips change."""
        if self._library is None:
            return None
        try:
            voice = self._library.get(voice_id)
        except ValueError:
            return None  # an exported voice with no recordings in the library
        clips = sorted(voice.clips, key=lambda c: c.index)
        if not clips:
            return None
        key = f"{RANGE_VERSION}:{len(clips)}:{clips[-1].index}"
        cache = self._folder / "voices" / f"{voice.id}.json"
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            stats = (float(saved["pitch_hz"]), float(saved["low_hz"]), float(saved["high_hz"]))
            if saved.get("key") == key and 0 < stats[1] <= stats[0] <= stats[2]:
                return stats
        except (OSError, ValueError, KeyError, TypeError):
            pass
        pitches, seconds = [], 0.0
        for clip in clips:
            if seconds >= RANGE_SECONDS:
                break
            try:
                audio, sr = read_wav(self._library.clip_audio_path(voice.id, clip.index))
            except (ValueError, OSError):
                continue
            f0, _ = pitch_of(audio, sr)
            pitches.append(f0[f0 > 0])
            seconds += len(audio) / sr
        voiced = np.concatenate(pitches) if pitches else np.zeros(0)
        if len(voiced) < int(MIN_VOICED_SECONDS * 100):
            return None
        low, pitch, high = (round(float(v), 1) for v in np.percentile(voiced, [1, 50, 99]))
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": key, "pitch_hz": pitch, "low_hz": low, "high_hz": high}),
                             encoding="utf-8")
        except OSError:
            pass  # measured again next time
        return pitch, low, high

    def suggestion(self, voice_id: str) -> dict:
        """The pitch shift for speaking as this voice, with the reason when there is none."""
        profile = self.profile()
        if profile is None:
            return {"pitch_shift": None, "reason": "Calibrate your voice first."}
        from aivoice.pretrained import base_voice_id

        target = self.voice_pitch(base_voice_id(voice_id))
        if target is None:
            return {"pitch_shift": None, "speaker_hz": profile.pitch_hz,
                    "reason": "This voice has no recordings in the library to measure, so set the pitch by hand."}
        return {"pitch_shift": suggest_shift(profile.pitch_hz, target), "speaker_hz": profile.pitch_hz,
                "target_hz": target, "reason": None}
