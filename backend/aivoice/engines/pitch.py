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


# ---- octave errors, wobble and blips

OCTAVE_REACH = 15  # frames either side (150 ms) that make the speaker's pitch at a moment
OCTAVE_TOLERANCE_ST = 3.0  # an octave error lands within this many semitones of an octave away
SPIKE_ST = 5.0  # a short stretch this far from the pitch around it is a tracker error, not intonation
SPIKE_MAX_FRAMES = 8
SMOOTH_FRAMES = 5
MIN_VOICED_FRAMES = 4  # voiced stretches shorter than this, alone in a pause, are dropped


def _voiced_runs(f0: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], (f0 > 0).astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _local_median(x: np.ndarray, reach: int) -> np.ndarray:
    return np.array([np.median(x[max(0, i - reach):i + reach + 1]) for i in range(len(x))])


def fix_octave_errors(f0: np.ndarray, reach: int = OCTAVE_REACH) -> np.ndarray:
    """Within each voiced stretch, a frame about an octave (or two) away from the speaker's pitch around it is moved
    back by whole octaves, up or down; a short stretch still far from it afterwards (a tracker error that is not an
    octave) takes the pitch around it. Rises and falls that take the time real speech takes are left as they are."""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    for a, b in _voiced_runs(f0):
        if b - a < 5:
            continue
        st = 12.0 * np.log2(f0[a:b].astype(np.float64))
        off = st - _local_median(st, reach)
        octaves = np.round(off / 12.0)
        wrong = (octaves != 0) & (np.abs(off - 12.0 * octaves) <= OCTAVE_TOLERANCE_ST)
        st[wrong] -= 12.0 * octaves[wrong]
        reference = _local_median(st, reach)
        far = np.abs(st - reference) > SPIKE_ST
        for s, e in _voiced_runs(far.astype(np.float32)):
            if e - s <= SPIKE_MAX_FRAMES:
                st[s:e] = reference[s:e]
        f0[a:b] = (2.0 ** (st / 12.0)).astype(np.float32)
    return f0


def smooth_pitch(f0: np.ndarray, width: int = SMOOTH_FRAMES, min_frames: int = MIN_VOICED_FRAMES) -> np.ndarray:
    """A running median over each voiced stretch (in semitones), which takes out single-frame wobble and keeps
    glides; voiced blips too short to be a syllable, alone in a pause, are dropped, so the model does not chirp."""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    half = width // 2
    for a, b in _voiced_runs(f0):
        if b - a < min_frames:
            f0[a:b] = 0.0
            continue
        st = np.log2(f0[a:b].astype(np.float64))
        f0[a:b] = (2.0 ** _local_median(st, half)).astype(np.float32)
    return f0


# ---- joins between live windows

def blend_join(current: np.ndarray, previous: np.ndarray, offset: int, start: int, length: int) -> np.ndarray:
    """Live mode converts overlapping windows, and two windows can track a slightly different pitch for the same
    moment; heard at the join, that is a warble. Frame j of this window is frame j + offset of the previous one.
    Before `start` (audio already played) the previous window's pitch is used, and over the `length` frames from
    `start` (the crossfade) the pitch moves from it to this window's own. Unvoiced frames are left alone."""
    out = np.asarray(current, dtype=np.float32).copy()
    previous = np.asarray(previous, dtype=np.float32)
    if offset <= 0:
        return out
    end = min(len(out), len(previous) - offset, start + length)
    for j in range(end):
        p = previous[j + offset]
        if out[j] <= 0 or p <= 0:
            continue
        w = 0.0 if j < start else (j - start) / max(1, length - 1)
        out[j] = np.float32(np.exp(np.log(p) * (1.0 - w) + np.log(out[j]) * w))
    return out
