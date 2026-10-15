"""Deep calibration: finding the pitch shift and likeness that make the speaker sound most like a voice.

The speaker's calibration reading is converted with the voice at a range of settings, the way live mode would
(same quality level, the calibration's gate and gain, the voice's pitch range). Each result is scored:

- likeness to the voice: a speaker-recognition model's fingerprint of the result against the fingerprint of the
  voice's own recordings (aivoice.speaker), the one score that follows how much the result sounds like the person;
- clarity: the result is transcribed and compared with the transcript of the reading itself, so settings that
  lose words (likeness at 100 percent does) are never chosen.

The search is in two steps: the pitch shift (two semitones either side of the current one), then the likeness at
the best shift. The most alike setting whose clarity is within CLARITY_MARGIN of the clearest wins, and only when
it beats the current setting by MIN_GAIN: smaller differences are within what the scores can tell apart.
Measured on the koda recordings with the Woman voice, the fingerprint similarity ran from 0.82 to 0.86 across
settings, peaking near the calibration's own pitch suggestion, and rising slowly with likeness until clarity
fell at 100 percent. How robotic a result sounds is not scored: that is still for the ear.
"""
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from aivoice.audiofile import read_wav
from aivoice.config import SAMPLE_RATE
from aivoice.presets import QUALITY_LEVELS, resolve
from aivoice.pretrained import list_voices as _list_voices
from aivoice.processor import StreamProcessor
from aivoice.speaker import similarity

PITCH_STEPS = (-2, -1, 0, 1, 2)
LIKENESS_STEPS = (0.1, 0.3, 0.5, 0.75, 1.0)
CLARITY_MARGIN = 0.05
MIN_GAIN = 0.01
VOICE_CLIPS = 30  # of the voice's speech clips, for its fingerprint


@dataclass(frozen=True)
class Candidate:
    pitch_shift: int
    likeness: float
    similarity: float
    clarity: float


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def word_error_rate(reference: str, heard: str) -> float:
    """Word edits needed to turn what was heard into the reference, per reference word."""
    ref, hyp = _words(reference), _words(heard)
    if not ref:
        return 0.0
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row = row, [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (r != h))
    return row[len(hyp)] / len(ref)


def choose(tried: list[Candidate], baseline: Candidate) -> Candidate:
    """The most alike setting that keeps the words, or the baseline when nothing beats it by MIN_GAIN."""
    floor = max(c.clarity for c in tried) - CLARITY_MARGIN
    clear = [c for c in tried if c.clarity >= floor]
    best = max(clear, key=lambda c: c.similarity)
    if baseline in clear and best.similarity - baseline.similarity < MIN_GAIN:
        return baseline
    return best


def _idle() -> dict:
    return {"active": False, "voice_id": None, "step": 0, "steps": 0, "message": None, "error": None, "result": None}


class Tuner:
    """Runs deep calibration for one voice at a time on a background thread."""

    def __init__(self, data: Path, library, calibration, engine_factory, transcriber,
                 speaker_factory: Callable[[], object], list_voices=_list_voices, sample_rate: int = SAMPLE_RATE):
        self._data = data
        self._library = library
        self._calibration = calibration
        self._engine_factory = engine_factory
        self._transcriber = transcriber
        self._speaker_factory = speaker_factory
        self._list_voices = list_voices
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._status = _idle()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._status["active"]

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _set(self, **changes) -> None:
        with self._lock:
            self._status.update(changes)

    def start(self, voice_id: str, quality: int, pitch_shift: int, likeness: float) -> None:
        voice = next((v for v in self._list_voices(self._data) if v.id == voice_id), None)
        if voice is None:
            raise ValueError(f"Unknown voice: {voice_id}")
        if not (isinstance(quality, int) and 0 <= quality < len(QUALITY_LEVELS)):
            raise ValueError("Unknown quality level")
        if not (isinstance(pitch_shift, int) and -24 <= pitch_shift <= 24) or not 0.0 <= float(likeness) <= 1.0:
            raise ValueError("The current pitch shift or likeness is out of range")
        sample = self._calibration.sample()
        if sample is None:
            raise ValueError("Calibrate your voice again first: deep calibration converts the reading from your "
                             "calibration, and calibrations made before this version kept only their measurements.")
        clips = self._voice_clips(voice_id)
        if not clips:
            raise ValueError("This voice has no recordings to compare with, so it cannot be calibrated this way.")
        with self._lock:
            if self._status["active"]:
                raise ValueError("Deep calibration is already running")
            steps = len(PITCH_STEPS) + len(set(LIKENESS_STEPS) | {float(likeness)}) - 1
            self._status = {**_idle(), "active": True, "voice_id": voice_id, "steps": steps,
                            "message": "Getting ready"}
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(voice, sample, clips, quality, pitch_shift, float(likeness), self._cancel),
                daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _voice_clips(self, voice_id: str) -> list[Path]:
        try:
            voice = self._library.get(voice_id)
        except ValueError:
            return []
        speech = [c for c in voice.clips if getattr(c, "kind", "speech") == "speech"]
        return [self._library.clip_audio_path(voice_id, c.index) for c in speech[:VOICE_CLIPS]]

    def _convert(self, engine, sample: np.ndarray, quality: int, pitch_shift: int, likeness: float) -> np.ndarray:
        """The reading as live mode would turn it out at these settings."""
        params = resolve("balanced", dict(QUALITY_LEVELS[quality], index_rate=likeness))
        processor = StreamProcessor(engine, self._sr, params)
        processor.engine = self._calibration.wrap(engine)
        processor.pitch_shift = pitch_shift
        padded = np.concatenate([sample, np.zeros(processor.chunk * 2 + processor.crossfade, np.float32)])
        out = np.concatenate([processor.process(padded[i:i + processor.chunk])
                              for i in range(0, len(padded) - processor.chunk + 1, processor.chunk)])
        return out[processor.crossfade: processor.crossfade + len(sample)]

    def _run(self, voice, sample, clips, quality, start_shift, start_likeness, cancel: threading.Event) -> None:
        try:
            self._set(message="Learning what the voice sounds like")
            speaker = self._speaker_factory()
            prints = []
            for path in clips:
                audio, sr = read_wav(path)
                try:
                    prints.append(speaker.embedding(audio, sr))
                except ValueError:
                    continue  # too short to recognise a speaker from
            if not prints:
                raise ValueError("The voice's recordings are too short to compare with.")
            target = np.mean(prints, axis=0)
            target = target / (np.linalg.norm(target) + 1e-12)

            engine = self._engine_factory(voice)
            if hasattr(engine, "pitch_range"):
                engine.pitch_range = self._calibration.voice_pitch_range(voice.id)
            profile = self._calibration.profile()
            if hasattr(engine, "speech_level_db") and profile is not None:
                engine.speech_level_db = profile.level_db + 20.0 * float(np.log10(profile.input_gain))
            gain = profile.input_gain if profile is not None else 1.0
            reference = self._transcriber.transcribe(
                soxr.resample(np.clip(sample * gain, -1, 1), self._sr, 16000).astype(np.float32))

            tried: dict[tuple[int, float], Candidate] = {}

            def score(shift: int, likeness: float) -> Candidate:
                key = (shift, likeness)
                if key not in tried:
                    if cancel.is_set():
                        raise _Stopped()
                    self._set(step=len(tried) + 1,
                              message=f"Trying pitch {shift:+d}, likeness {round(likeness * 100)}%")
                    out = self._convert(engine, sample, quality, shift, likeness)
                    heard = self._transcriber.transcribe(soxr.resample(out, self._sr, 16000).astype(np.float32))
                    tried[key] = Candidate(shift, likeness, round(similarity(speaker.embedding(out, self._sr), target), 4),
                                           round(max(0.0, 1.0 - word_error_rate(reference, heard)), 3))
                return tried[key]

            baseline = score(start_shift, start_likeness)
            shifts = [score(start_shift + d, start_likeness) for d in PITCH_STEPS]
            best_shift = choose(shifts, baseline)
            levels = [score(best_shift.pitch_shift, r) for r in sorted(set(LIKENESS_STEPS) | {start_likeness})]
            best = choose(levels, best_shift)
            result = {"voice_id": voice.id, "pitch_shift": best.pitch_shift, "likeness": best.likeness,
                      "similarity": best.similarity, "clarity": best.clarity, "quality": quality,
                      "baseline": asdict(baseline), "tried": [asdict(c) for c in tried.values()],
                      "created": time.time()}
            folder = self._data / "calibration" / "tuned"
            folder.mkdir(parents=True, exist_ok=True)
            part = folder / f"{voice.id}.json.part"
            part.write_text(json.dumps(result), encoding="utf-8")
            part.replace(folder / f"{voice.id}.json")
            self._set(active=False, step=len(tried), steps=len(tried), message="Done", result=result)
        except _Stopped:
            self._set(active=False, message="Stopped before it finished.")
        except Exception as exc:  # noqa: BLE001 - a background thread reports, it never dies with a traceback
            self._set(active=False, message=None, error=f"Deep calibration failed: {exc}")


class _Stopped(Exception):
    pass
