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
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from aivoice.audiofile import read_wav
from aivoice.config import SAMPLE_RATE
from aivoice.engines.f0 import extract_f0

QUIET_SECONDS = 3.0  # the start of the recording, during which the speaker says nothing
READ_SECONDS = 15.0  # then this much reading aloud
MIN_VOICED_SECONDS = 3.0
TARGET_LEVEL_DB = -20.0  # the level of ordinary close speech (RMS of voiced sound)
MIN_GAIN, MAX_GAIN = 0.5, 4.0
GATE_MARGIN_DB = 6.0
GATE_FLOOR_DB = -70.0
HANGOVER_SECONDS = 0.3  # speech that ended this recently still opens the gate
FRAME_SECONDS = 0.02
MAX_SHIFT = 24
ABANDONED_AFTER = 90.0  # a recording nobody finished is given up after this long
READING = ("The ferry leaves the harbour at seven each morning. If the wind is calm, you can see the far shore "
           "long before the engines slow, and the gulls follow the boat the whole way across. My grandfather kept "
           "a small workshop behind the house. He fixed clocks, radios and bicycles there, and he never once "
           "threw away a screw that might be useful again.")


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


def analyse(audio: np.ndarray, sr: int, device: str, now: float) -> Profile:
    """Measures a calibration recording: QUIET_SECONDS of silence, then reading aloud."""
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))
    split = int(QUIET_SECONDS * sr)
    if len(audio) < split + int(MIN_VOICED_SECONDS * sr):
        raise CalibrationError("The recording is too short. Stay quiet for the first seconds, then read the text aloud.")
    quiet = _frame_db(audio[int(0.3 * sr): split - int(0.2 * sr)], sr)  # edges trimmed: a click, an early start
    speech = audio[split:]
    f0, wav16 = pitch_of(speech, sr)
    usual = _usual_pitch(f0)
    if usual is None:
        raise CalibrationError("Not enough speech was heard. Read the text aloud for the whole time, close to the microphone.")
    pitch, low, high = usual
    # The level of the voiced parts only, so pauses do not count as quiet speech.
    frames = _frame_db(wav16, 16000, 0.01)
    voiced = frames[: len(f0)][f0[: len(frames)] > 0]
    level = float(10.0 * np.log10(np.mean(10.0 ** (voiced / 10.0))))
    noise = float(np.percentile(quiet, 95)) if len(quiet) else GATE_FLOOR_DB
    quiet_speech = float(np.percentile(voiced, 10))
    if quiet_speech - noise < 2 * GATE_MARGIN_DB:
        raise CalibrationError("The microphone is too noisy, or you spoke during the quiet part: the quiet and your "
                               "speech are too close in level to tell apart. Try again in a quieter moment.")
    gate = max(GATE_FLOOR_DB, min(noise + GATE_MARGIN_DB, quiet_speech - GATE_MARGIN_DB))
    peak = float(np.percentile(np.abs(speech), 99.9))
    gain = 10.0 ** ((TARGET_LEVEL_DB - level) / 20.0)
    if peak > 0:
        gain = min(gain, 0.98 / peak)  # never into clipping
    gain = float(min(MAX_GAIN, max(MIN_GAIN, gain)))
    return Profile(round(pitch, 1), round(low, 1), round(high, 1), round(level, 1), round(noise, 1),
                   round(float(gate), 1), round(gain, 3), device[:120], now)


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
        audio = np.asarray(audio, dtype=np.float32)
        if self._gate_db is not None:
            recent = int(sample_rate * ((params.chunk_ms + params.crossfade_ms) / 1000.0 + HANGOVER_SECONDS))
            levels = _frame_db(np.nan_to_num(audio[-recent:]), sample_rate)
            if len(levels) and float(levels.max()) < self._gate_db:
                return np.zeros_like(audio)
        if self._gain != 1.0:
            audio = np.clip(audio * np.float32(self._gain), -1.0, 1.0)
        return self._inner.convert(audio, sample_rate, pitch_shift, params)


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
        try:
            (self._folder / "profile.json").unlink()
        except FileNotFoundError:
            pass

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
            self._started, self._device, self._error = self._clock(), source.name, None

    def cancel(self) -> None:
        with self._lock:
            if self._started is not None:
                self._recorder.cancel()
                self._started = None

    def finish(self) -> Profile:
        with self._lock:
            if not self._active_locked():
                raise CalibrationError("No calibration recording is running")
            audio = self._recorder.stop()
            self._started = None
            try:
                profile = analyse(audio, SAMPLE_RATE, self._device, self._clock())
                self._folder.mkdir(parents=True, exist_ok=True)
                part = self._folder / "profile.json.part"
                part.write_text(json.dumps(asdict(profile)), encoding="utf-8")
                part.replace(self._folder / "profile.json")
            except CalibrationError as exc:
                self._error = str(exc)
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
            return {"recording": active, "seconds": seconds, "level": float(recorded["level"]),
                    "phase": None if not active else ("quiet" if seconds < QUIET_SECONDS else "reading"),
                    "quiet_seconds": QUIET_SECONDS, "total_seconds": QUIET_SECONDS + READ_SECONDS,
                    "text": READING, "error": self._error,
                    "profile": None if profile is None else asdict(profile)}

    # ---- the target voices

    def voice_pitch(self, voice_id: str) -> float | None:
        """The usual pitch of a voice, measured from its clips (up to a minute of them). None when it has none."""
        stats = self._voice_stats(voice_id)
        return None if stats is None else stats[0]

    def voice_pitch_range(self, voice_id: str) -> tuple[float, float] | None:
        """The pitch range the voice's recordings cover (5th to 95th percentile, Hz), which is what a model
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
        key = f"{len(clips)}:{clips[-1].index}"
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
            if seconds >= 60.0:
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
        low, pitch, high = (round(float(v), 1) for v in np.percentile(voiced, [5, 50, 95]))
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
        target = self.voice_pitch(voice_id)
        if target is None:
            return {"pitch_shift": None, "speaker_hz": profile.pitch_hz,
                    "reason": "This voice has no recordings in the library to measure, so set the pitch by hand."}
        return {"pitch_shift": suggest_shift(profile.pitch_hz, target), "speaker_hz": profile.pitch_hz,
                "target_hz": target, "reason": None}
