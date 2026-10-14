"""A feminized version of a voice, made by signal processing rather than by a model of someone else's voice.

The WORLD vocoder (pyworld) splits speech into three parts: the pitch, the spectral envelope (the resonances of
the vocal tract, the formants) and the aperiodicity (how breathy each band is). A voice is heard as more feminine
mostly from a higher pitch together with higher formants, as from a shorter vocal tract; raising the pitch alone
gives the "chipmunk" sound. So the transform raises the pitch, stretches the envelope upwards in frequency, adds a
little breathiness, and puts the voice back together.

It is used two ways: at its best quality, offline, to turn a voice's clips into the dataset of a feminized voice
that is then trained like any other (the model then smooths what the vocoder leaves rough); and in a lighter form
as a live preview that needs no training.
"""
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyworld
import soxr

from aivoice.audiofile import read_wav
from aivoice.config import SAMPLE_RATE

FRAME_MS = 5.0
F0_FLOOR, F0_CEIL = 60.0, 800.0
LIVE_RATE = 24000  # the live preview works at half rate: four times less work, and speech has little above 12 kHz
LIVE_CONTEXT_S = 0.2  # analysed before the part that is played, so the pitch tracker has something to go on
PREVIEW_ID = "@feminize"
PREVIEW_NAME = "My voice, feminized (preview, no training)"


@dataclass(frozen=True)
class FeminizeSettings:
    pitch_semitones: float = 5.0  # how much higher the voice is
    formant_shift: float = 1.15  # how much higher its resonances are (1.15 = 15 percent)
    breathiness: float = 0.15  # 0 leaves it as it is, 1 is a whisper

    def validated(self) -> "FeminizeSettings":
        for name, value, low, high in (("pitch_semitones", self.pitch_semitones, -12.0, 12.0),
                                       ("formant_shift", self.formant_shift, 0.8, 1.4),
                                       ("breathiness", self.breathiness, 0.0, 0.8)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) \
                    or not low <= value <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}")
        return FeminizeSettings(float(self.pitch_semitones), float(self.formant_shift), float(self.breathiness))


def _warp(envelope: np.ndarray, alpha: float) -> np.ndarray:
    """Moves every resonance of a spectral envelope up by the factor alpha: the new envelope at a frequency f is the
    old one at f / alpha. Above the old top the last value carries on."""
    bins = envelope.shape[1]
    source = np.arange(bins) / alpha
    grid = np.arange(bins)
    return np.stack([np.interp(source, grid, row) for row in envelope])


def _breathier(aperiodicity: np.ndarray, amount: float) -> np.ndarray:
    """More noise in every band, most in the upper ones, where breath is heard."""
    if amount <= 0:
        return aperiodicity
    tilt = np.linspace(0.4, 1.0, aperiodicity.shape[1])
    mix = np.clip(amount * tilt, 0.0, 1.0)
    return np.clip(aperiodicity + (1.0 - aperiodicity) * mix, 1e-6, 1.0)


def _transform(x: np.ndarray, fs: int, settings: FeminizeSettings, best: bool) -> np.ndarray:
    x = np.ascontiguousarray(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    if len(x) < int(0.05 * fs) or not np.any(np.abs(x) > 1e-5):
        return np.zeros(len(x), dtype=np.float32)
    if best:
        f0, times = pyworld.harvest(x, fs, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_MS)
    else:
        f0, times = pyworld.dio(x, fs, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_MS)
        f0 = pyworld.stonemask(x, f0, times, fs)
    envelope = pyworld.cheaptrick(x, f0, times, fs)
    aperiodicity = pyworld.d4c(x, f0, times, fs)
    f0 = f0 * 2.0 ** (settings.pitch_semitones / 12.0)
    if settings.formant_shift != 1.0:
        envelope = _warp(envelope, settings.formant_shift)
    y = pyworld.synthesize(f0, envelope, _breathier(aperiodicity, settings.breathiness), fs, FRAME_MS)
    y = np.nan_to_num(y[: len(x)], nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    # The same loudness as what came in, so a voice made from many clips keeps their levels.
    rms_in, rms_out = float(np.sqrt(np.mean(x * x))), float(np.sqrt(np.mean(y * y)))
    if rms_out > 1e-9:
        y = y * (rms_in / rms_out)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def feminize(audio: np.ndarray, sr: int, settings: FeminizeSettings) -> np.ndarray:
    """The best-quality transform, for clips that become a training dataset or a preview sample."""
    return _transform(np.asarray(audio, dtype=np.float64).reshape(-1), sr, settings.validated(), best=True)


class FeminizeEngine:
    """Live mode's preview: a voice engine that feminizes the speaker's own voice, with no model behind it.

    The processor only plays the newest chunk and crossfade of each window, so only that part (plus a little
    context for the pitch tracker) is processed, at half rate, with the faster pitch tracker.
    """

    def __init__(self, settings: FeminizeSettings):
        self._settings = settings.validated()

    def convert(self, audio, sample_rate, pitch_shift, params):
        audio = np.asarray(audio, dtype=np.float32)
        tail = int(sample_rate * ((params.chunk_ms + params.crossfade_ms) / 1000.0 + LIVE_CONTEXT_S))
        tail = min(tail, len(audio))
        piece = soxr.resample(audio[-tail:], sample_rate, LIVE_RATE)
        settings = FeminizeSettings(self._settings.pitch_semitones + float(pitch_shift),
                                    self._settings.formant_shift, self._settings.breathiness)
        shifted = soxr.resample(_transform(piece, LIVE_RATE, settings, best=False), LIVE_RATE, sample_rate)
        out = np.zeros(len(audio), dtype=np.float32)
        shifted = shifted[: tail]
        out[len(audio) - len(shifted):] = shifted
        return out


# ---- settings kept for the live preview


def settings_path(data: Path) -> Path:
    return data / "feminize.json"


def load_settings(data: Path) -> FeminizeSettings:
    try:
        raw = json.loads(settings_path(data).read_text(encoding="utf-8"))
        return FeminizeSettings(**{k: raw[k] for k in ("pitch_semitones", "formant_shift", "breathiness")}).validated()
    except (OSError, ValueError, KeyError, TypeError):
        return FeminizeSettings()


def save_settings(data: Path, settings: FeminizeSettings) -> FeminizeSettings:
    settings = settings.validated()
    data.mkdir(parents=True, exist_ok=True)
    part = settings_path(data).with_suffix(".json.part")
    part.write_text(json.dumps(asdict(settings)), encoding="utf-8")
    part.replace(settings_path(data))
    return settings


# ---- making a feminized voice from a voice's clips


class FeminizeJob:
    """Makes a new library voice whose clips are the feminized clips of another voice, with the same transcripts.
    One job at a time, on its own thread; the source voice's recordings are only read."""

    def __init__(self, library, clock: Callable[[], float] = time.time):
        self._library = library
        self._clock = clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._state = {"active": False, "done": 0, "total": 0, "source": None, "voice_id": None, "voice_name": None,
                       "error": None, "finished": False, "cancelled": False}

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, source_id: str, settings: FeminizeSettings, name: str | None = None) -> dict:
        settings = settings.validated()
        with self._lock:
            if self._state["active"]:
                raise ValueError("A feminized voice is already being made. Wait for it to finish.")
            source = self._library.get(source_id)
            clips = [c for c in source.clips if c.text.strip()]
            if not clips:
                raise ValueError("This voice has no clips with transcripts to make a feminized voice from")
            new_name = (name or f"{source.name} (feminized)").strip()[:60]
            voice = self._library.create(new_name)
            self._cancel.clear()
            self._state = {"active": True, "done": 0, "total": len(clips), "source": source_id, "voice_id": voice.id,
                           "voice_name": voice.name, "error": None, "finished": False, "cancelled": False}
            self._thread = threading.Thread(target=self._run, args=(source_id, voice.id, clips, settings),
                                            name="feminize", daemon=True)
            self._thread.start()
            return dict(self._state)

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, source_id, voice_id, clips, settings) -> None:
        error, cancelled, skipped = None, False, 0
        try:
            for clip in clips:
                if self._cancel.is_set():
                    cancelled = True
                    break
                try:
                    audio, sr = read_wav(self._library.clip_audio_path(source_id, clip.index))
                except ValueError:
                    skipped += 1  # a clip whose audio has gone is left out, the others go on
                else:
                    if sr != SAMPLE_RATE:
                        audio, sr = soxr.resample(audio, sr, SAMPLE_RATE).astype(np.float32), SAMPLE_RATE
                    changed = feminize(audio, sr, settings)
                    if np.any(np.abs(changed) > 1e-4):
                        self._library.add_clip(voice_id, changed, clip.text, "imported",
                                               origin=f"feminized from clip {clip.index}")
                    else:
                        skipped += 1
                with self._lock:
                    self._state["done"] += 1
        except Exception as exc:  # noqa: BLE001 - the job must end with a message, never die silently
            error = f"Making the feminized voice failed: {exc}"
        with self._lock:
            self._state.update(active=False, finished=True, cancelled=cancelled, error=error, skipped=skipped)
