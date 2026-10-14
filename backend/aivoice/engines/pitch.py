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
