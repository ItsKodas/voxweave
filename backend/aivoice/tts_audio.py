"""Post-processing of synthesised speech pieces: trim, finish, check, join. Pure numpy and soxr."""

import numpy as np
import soxr

SAMPLE_RATE = 48000

# A piece whose loudest 20 ms frame is below this is treated as nothing, however it compares to itself:
# the relative rule alone would keep a piece of pure room tone.
MIN_LEVEL_DB = -60.0
MIN_SECONDS = 0.15
MIN_PEAK = 0.005
# The peak every played piece is brought to, about -1 dBFS: one level for all pieces of a request, with room
# left for the player's gain stage and the 16-bit file.
PIECE_PEAK = 0.89
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


def _sustained(loud: np.ndarray, frames: int) -> np.ndarray:
    """The loud frames that belong to a run of at least `frames` loud frames in a row."""
    out = np.zeros(len(loud), dtype=bool)
    run = 0
    for i, value in enumerate(loud):
        run = run + 1 if value else 0
        if run >= frames:
            out[i - run + 1:i + 1] = True
    return out


def trim_edges(audio, sr: int, rel_db: float = -30.0, frame_ms: int = 20, keep_ms: int = 30,
               onset_ms: int = 60) -> np.ndarray:
    """Cut leading and trailing frames more than rel_db below the loudest frame, keeping keep_ms of context.

    The start is where sound first lasts onset_ms: the model's lead-in is the reference clip's room tone, which
    can be only 30 dB below the speech, so single frames of it reach the threshold (first real run). If nothing
    lasts that long, the start is the first loud frame. The end is the last loud frame, however short: the
    release of a final consonant is short and belongs to the word."""
    audio = _clean(audio)
    if len(audio) == 0:
        return audio
    frame = max(1, sr * frame_ms // 1000)
    db = _frame_db(audio, frame)
    top = float(db.max())
    if top < MIN_LEVEL_DB:
        return audio[:0]
    loud = db >= top + rel_db
    lasting = np.flatnonzero(_sustained(loud, -(-onset_ms // frame_ms)))
    first = int(lasting[0]) if len(lasting) else int(np.argmax(loud))
    last = int(np.flatnonzero(loud)[-1])
    keep = sr * keep_ms // 1000
    start = max(0, first * frame - keep)
    end = min(len(audio), (last + 1) * frame + keep)
    return audio[start:end].copy()


def drop_reference_tail(audio, sr: int, rel_db: float = -30.0, frame_ms: int = 20, keep_ms: int = 30,
                        onset_ms: int = 60, start_ms: int = 60, burst_ms: int = 400,
                        gap_ms: int = 400) -> np.ndarray:
    """Cut what the model said of the reference clip off the start of a piece as the model made it.

    A piece continues the reference clip. Its start can hold the dying end of the reference's last word, or, at
    few sampling steps, a word said at full level that is not in the text; then comes the pause before the
    sentence. trim_edges would keep both the sound and the pause. The sound is recognised by all of: it begins
    within the first start_ms and lasts at least onset_ms, it has ended within burst_ms, at least gap_ms without
    lasting sound follow, and lasting sound comes after that. Everything before that later sound is cut, bar
    keep_ms of context: the stray word can be the loudest thing in the piece, so trim_edges, measuring what is
    left against a quieter peak, could no longer tell the pause from speech. Anything else is returned whole.
    Only for a piece straight from the model: a sentence that really begins with a word shorter than burst_ms
    followed by a pause longer than gap_ms would lose that word."""
    audio = _clean(audio)
    frame = max(1, sr * frame_ms // 1000)
    if len(audio) < frame:
        return audio
    db = _frame_db(audio, frame)
    top = float(db.max())
    if top < MIN_LEVEL_DB:
        return audio
    lasting = _sustained(db >= top + rel_db, -(-onset_ms // frame_ms))
    if not lasting.any() or int(np.argmax(lasting)) * frame_ms > start_ms:
        return audio
    gap = -(-gap_ms // frame_ms)
    quiet_run = 0
    for i in range(int(np.argmax(lasting)), min(len(lasting), burst_ms // frame_ms + gap + 1)):
        quiet_run = 0 if lasting[i] else quiet_run + 1
        if quiet_run == gap:
            if not lasting[i:].any():
                return audio
            sentence = i + int(np.argmax(lasting[i:]))
            return audio[max(0, sentence * frame - sr * keep_ms // 1000):]
    return audio


def finish(audio, sr: int, fade_ms: int = 10, peak: float = 0.97, level: float | None = None) -> np.ndarray:
    """Mono float32 at 48 kHz: resampled if needed, set to its level, edges faded.

    With a level, the piece's peak is brought to it, up or down, so that the pieces of a request sound equally
    loud: the worker hands them over at whatever peak the model gave them (and far lower for a quiet reference
    clip). What is_silent calls nothing is left as it is, never raised into noise. Without a level the piece is
    only scaled down, and only if it is over the peak limit."""
    audio = _clean(audio)
    if len(audio) == 0:
        return audio
    if sr != SAMPLE_RATE:
        audio = np.array(soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ"), dtype=np.float32)
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    if level is not None:
        if top >= MIN_PEAK:
            audio = audio * np.float32(level / top)
    elif top > peak:
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
