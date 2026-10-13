"""Post-processing of synthesised speech pieces: trim, finish, check, join. Pure numpy and soxr."""

import numpy as np
import soxr

SAMPLE_RATE = 48000

# A piece whose loudest 20 ms frame is below this is treated as nothing, however it compares to itself:
# the relative rule alone would keep a piece of pure room tone.
MIN_LEVEL_DB = -60.0
MIN_SECONDS = 0.15
MIN_PEAK = 0.005
SAFETY_MARGIN = 0.85
# The package sizes pieces for 22 s of reference plus generated audio, leaving room under the 30 s limit.
PACKAGE_BUDGET_SECONDS = 22.0
MIN_PIECE_BYTES = 20  # the smallest max_chars tts_text.split accepts


def _clean(audio) -> np.ndarray:
    """Mono float32 copy with NaN and infinity replaced by silence."""
    out = np.array(audio, dtype=np.float32).reshape(-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
    """RMS level in dBFS of each frame; the last, partial frame is measured over its own length."""
    count = -(-len(audio) // frame)
    padded = np.zeros(count * frame, dtype=np.float64)
    padded[: len(audio)] = audio
    sums = np.sum(padded.reshape(count, frame) ** 2, axis=1)
    lengths = np.full(count, frame, dtype=np.float64)
    lengths[-1] = len(audio) - (count - 1) * frame
    return 10.0 * np.log10(np.maximum(sums / lengths, 1e-20))


def trim_edges(audio, sr: int, rel_db: float = -30.0, frame_ms: int = 20, keep_ms: int = 30) -> np.ndarray:
    """Cut leading and trailing frames more than rel_db below the loudest frame, keeping keep_ms of context."""
    audio = _clean(audio)
    if len(audio) == 0:
        return audio
    frame = max(1, sr * frame_ms // 1000)
    db = _frame_db(audio, frame)
    top = float(db.max())
    if top < MIN_LEVEL_DB:
        return audio[:0]
    loud = np.flatnonzero(db >= top + rel_db)
    keep = sr * keep_ms // 1000
    start = max(0, int(loud[0]) * frame - keep)
    end = min(len(audio), (int(loud[-1]) + 1) * frame + keep)
    return audio[start:end].copy()


def finish(audio, sr: int, fade_ms: int = 10, peak: float = 0.97) -> np.ndarray:
    """Mono float32 at 48 kHz: resampled if needed, edges faded, scaled down only if over the peak limit."""
    audio = _clean(audio)
    if len(audio) == 0:
        return audio
    if sr != SAMPLE_RATE:
        audio = np.array(soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ"), dtype=np.float32)
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    if top > peak:
        audio = audio * np.float32(peak / top)
    fade = min(SAMPLE_RATE * fade_ms // 1000, len(audio) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return np.ascontiguousarray(audio, dtype=np.float32)


def is_silent(audio, sr: int) -> bool:
    """True for a piece that is effectively nothing: under 0.15 s long, or with a peak below 0.005."""
    audio = _clean(audio)
    if len(audio) < MIN_SECONDS * sr:
        return True
    return float(np.abs(audio).max()) < MIN_PEAK


def join(pieces: list[np.ndarray], pause_ms: int = 300, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Concatenate pieces with pause_ms of silence between them (empty pieces are skipped)."""
    parts = [np.asarray(p, dtype=np.float32).reshape(-1) for p in pieces if len(p)]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    pause = np.zeros(sr * pause_ms // 1000, dtype=np.float32)
    out: list[np.ndarray] = []
    for i, part in enumerate(parts):
        if i and len(pause):
            out.append(pause)
        out.append(part)
    return np.concatenate(out)


def max_chars_for(ref_text: str, ref_seconds: float, speed: float = 1.0, limit_seconds: float = 30.0) -> int:
    """Largest piece, in UTF-8 bytes, that keeps reference plus generated audio under the limit.

    The model gives a piece ref_seconds / ref_bytes * piece_bytes / speed seconds, so the F5-TTS package's own
    expression is ref_bytes / ref_seconds * (22 - ref_seconds) * speed. Its 22 s budget is kept (it is the
    package's own allowance below the 30 s limit) and lowered if limit_seconds is smaller; 15 percent is
    then taken off for the model's slack. ref_text is the text exactly as it is given to the model.
    """
    ref_bytes = len(ref_text.encode("utf-8"))
    if ref_bytes == 0:
        raise ValueError("The reference transcript is empty.")
    if ref_seconds <= 0 or speed <= 0:
        raise ValueError("The reference length and the speed must be positive.")
    budget = min(PACKAGE_BUDGET_SECONDS, limit_seconds)
    raw = ref_bytes / ref_seconds * (budget - ref_seconds) * speed
    result = int(raw * SAFETY_MARGIN)
    if result < MIN_PIECE_BYTES:
        raise ValueError("The reference clip is too long to leave room for speech. Use a shorter clip.")
    return result
