"""Pitch tracking with RMVPE, and keeping a pitch inside what a voice model knows.

RMVPE is the neural pitch tracker RVC prepares its training data with. Live mode used the fast WORLD tracker
("dio"), which misses a good part of voiced speech: those frames reach the model with no pitch and come out buzzy or
broken. tools/export_rmvpe_onnx.py turns the trainer's RMVPE weights into data/pretrained/rmvpe.onnx, which runs here
on the GPU. The graph takes the STFT magnitude; the STFT is done in numpy exactly as RVC does it.
"""
from pathlib import Path

import numpy as np

N_FFT, HOP = 1024, 160
THRESHOLD = 0.03  # RVC's: salience below this is "no pitch"
_CENTS = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic hann


def rmvpe_path(data: Path) -> Path:
    return data / "pretrained" / "rmvpe.onnx"


def stft_magnitude(audio16k: np.ndarray) -> np.ndarray:
    """(1, 513, frames) magnitude with a centred hann window of 1024 and hop 160, as RVC computes it."""
    x = np.asarray(audio16k, dtype=np.float32)
    if len(x) <= N_FFT // 2:  # reflect padding needs more samples than the pad
        x = np.pad(x, (0, N_FFT // 2 + 1 - len(x)))
    padded = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = 1 + (len(padded) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(padded[index] * _WINDOW, axis=1)).T[None].astype(np.float32)


def decode(salience: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Pitch in Hz per frame from RMVPE's salience (frames, 360): a weighted average of the nine bins around the
    peak, 0 where the peak is weak."""
    salience = np.asarray(salience, dtype=np.float64)
    center = np.argmax(salience, axis=1) + 4
    padded = np.pad(salience, ((0, 0), (4, 4)))
    offsets = np.arange(-4, 5)
    cols = center[:, None] + offsets[None, :]
    rows = np.arange(len(salience))[:, None]
    weights = padded[rows, cols]
    cents = np.sum(weights * _CENTS[cols], axis=1) / np.maximum(np.sum(weights, axis=1), 1e-12)
    f0 = 10.0 * 2.0 ** (cents / 1200.0)
    f0[np.max(padded, axis=1) <= threshold] = 0.0
    return f0.astype(np.float32)


class Rmvpe:
    def __init__(self, session):
        self._session = session
        self._input = session.get_inputs()[0].name

    def f0(self, audio16k: np.ndarray) -> np.ndarray:
        """Pitch in Hz for every 10 ms of 16 kHz audio (0 where there is none)."""
        magnitude = stft_magnitude(audio16k)
        frames = magnitude.shape[2]
        padded = 32 * ((frames - 1) // 32 + 1)  # the network halves the length five times
        if padded > frames:
            magnitude = np.pad(magnitude, ((0, 0), (0, 0), (0, padded - frames)))
        salience = self._session.run(None, {self._input: magnitude})[0][0][:frames]
        return decode(salience)


# ---- keeping the pitch inside what the voice model knows

MAX_GAP_FRAMES = 4  # a pitch tracker dropping out for up to 40 ms inside a voiced stretch is filled in


def fill_gaps(f0: np.ndarray, max_gap: int = MAX_GAP_FRAMES) -> np.ndarray:
    """Short unvoiced gaps between voiced frames, where the tracker lost the pitch for a moment, are bridged
    (in the log domain). Real pauses, longer than max_gap frames, are left silent."""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    voiced = np.flatnonzero(f0 > 0)
    for a, b in zip(voiced[:-1], voiced[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            t = np.arange(1, gap + 1) / (gap + 1)
            f0[a + 1:b] = np.exp(np.log(f0[a]) * (1 - t) + np.log(f0[b]) * t)
    return f0


def fit_to_range(f0: np.ndarray, low: float, high: float) -> np.ndarray:
    """Pitches the voice model never heard are moved to the nearest pitch it knows.

    A pitch more than half an octave outside the range is first moved by whole octaves (a tracker's octave error,
    or a voice far from the model's); what is still above `high` or below `low` is then pulled to that edge, softly,
    so the rise and fall of the speech keeps its shape instead of flattening into one note."""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    if not (0 < low < high):
        return f0
    voiced = f0 > 0
    v = f0[voiced].astype(np.float64)
    for _ in range(4):
        v = np.where(v > high * 1.5, v / 2.0, v)
        v = np.where(v < low / 1.5, v * 2.0, v)
    # Beyond an edge, a tenth of the distance (in octaves) is kept: the contour survives, the range is respected.
    above, below = v > high, v < low
    v[above] = high * (v[above] / high) ** 0.1
    v[below] = low * (v[below] / low) ** 0.1
    f0[voiced] = v.astype(np.float32)
    return f0


# ---- creak at the end of long sentences

CREAK_MAX_FRAMES = 40  # held for up to 0.4 s; longer stretches are left as they are
LOW_BAND_HZ = 1000.0
VOICED_LOW_SHARE = 0.6  # a frame with at least this share of its energy below 1 kHz sounds like voice, not a hiss
LOUD_WITHIN_DB = 30.0  # and is within this many dB of the loud parts of the window
TOO_LOW = 0.7  # a pitch below 70 percent of the speaker's recent pitch is creak or an octave error


def voice_like_frames(audio16k: np.ndarray, frames: int) -> np.ndarray:
    """Per 10 ms frame: whether the sound is voice-like (loud enough, energy mostly low) rather than silence or a
    hissing consonant. Resized to `frames`."""
    x = np.asarray(audio16k, dtype=np.float32)
    n = 320
    count = max(1, len(x) // 160)
    padded = np.pad(x, (n // 2, n // 2 + 160))
    index = np.arange(n)[None, :] + 160 * np.arange(count)[:, None]
    spectrum = np.abs(np.fft.rfft(padded[index] * np.hanning(n).astype(np.float32), axis=1)) ** 2
    total = spectrum.sum(axis=1) + 1e-12
    low = spectrum[:, : int(LOW_BAND_HZ / 8000.0 * spectrum.shape[1])].sum(axis=1)
    level = 10.0 * np.log10(total)
    loud = level > np.percentile(level, 90) - LOUD_WITHIN_DB
    flags = loud & (low / total >= VOICED_LOW_SHARE)
    if len(flags) == frames:
        return flags
    return flags[np.round(np.linspace(0, len(flags) - 1, frames)).astype(int)]


def repair_creak(f0: np.ndarray, voice_like: np.ndarray, max_frames: int = CREAK_MAX_FRAMES) -> np.ndarray:
    """Where the sound is plainly voice but the tracker found no pitch, or a pitch far below the speaker's recent
    one (creaky voice when breath runs out, or an octave error), the last good pitch is held for up to max_frames
    instead of the model getting "no pitch" or a pitch that jumps down an octave, which is what sounds robotic."""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    voice_like = np.asarray(voice_like, dtype=bool)
    good = f0 > 0
    if good.sum() < 5:
        return f0
    recent: list[float] = []
    held = 0
    last_good = 0.0
    for i in range(len(f0)):
        reference = float(np.median(recent)) if recent else 0.0
        bad = voice_like[i] and (f0[i] <= 0 or (reference > 0 and f0[i] < TOO_LOW * reference))
        if bad and last_good > 0 and held < max_frames:
            f0[i] = last_good
            held += 1
            continue
        if f0[i] > 0 and not bad:
            last_good = float(f0[i])
            recent.append(last_good)
            if len(recent) > 50:
                recent.pop(0)
        held = 0 if f0[i] > 0 else held
    return f0
