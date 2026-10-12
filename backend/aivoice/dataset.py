import numpy as np

FRAME_MS = 10
MIN_SECONDS = 1.0
MAX_RECORD_SECONDS = 20.0
QUIET_PEAK = 0.05
CLIP_LEVEL = 0.999

PROBLEM_MESSAGES = {
    "too_quiet": "Nothing was heard, or it was too quiet. Check the microphone and speak closer to it.",
    "too_short": "That take was too short. Read the whole sentence before pressing Stop.",
    "too_long": "That take was too long. Keep each take under 20 seconds.",
    "clipped": "The recording distorted. Move back from the microphone or lower its level, then try again.",
}


def _clean(audio) -> np.ndarray:
    return np.nan_to_num(np.asarray(audio, dtype=np.float32)).reshape(-1)


def _frame_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """Loudness of each whole 10 ms frame in dBFS."""
    n = sr * FRAME_MS // 1000
    frames = len(audio) // n
    if frames == 0:
        return np.empty(0)
    x = audio[: frames * n].reshape(frames, n).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(x * x, axis=1)), 1e-10))


def trim_silence(audio, sr: int, threshold_db: float = -40.0, pad_ms: int = 100) -> np.ndarray:
    audio = _clean(audio)
    loud = np.flatnonzero(_frame_db(audio, sr) > threshold_db)
    if len(loud) == 0:
        return audio[:0]
    n = sr * FRAME_MS // 1000
    pad = sr * pad_ms // 1000
    return audio[max(0, int(loud[0]) * n - pad) : min(len(audio), (int(loud[-1]) + 1) * n + pad)]


def level_problem(audio: np.ndarray) -> str | None:
    audio = _clean(audio)
    if len(audio) == 0 or float(np.max(np.abs(audio))) < QUIET_PEAK:
        return "too_quiet"
    clipped = int(np.count_nonzero(np.abs(audio) >= CLIP_LEVEL))
    if clipped > max(8, int(len(audio) * 0.0002)):
        return "clipped"
    return None


def clip_problem(audio: np.ndarray, sr: int) -> str | None:
    """Why an already trimmed take is unusable, or None when it is fine."""
    audio = _clean(audio)
    if len(audio) == 0:
        return "too_quiet"
    if len(audio) < MIN_SECONDS * sr:
        return "too_short"
    if len(audio) > MAX_RECORD_SECONDS * sr:
        return "too_long"
    return level_problem(audio)


def split_on_silence(audio, sr: int, max_seconds: float = 15.0, min_seconds: float = 1.0,
                     threshold_db: float = -40.0, min_silence_ms: int = 300, pad_ms: int = 100) -> list[np.ndarray]:
    """Cut audio into utterances at pauses. An utterance longer than max_seconds is cut at its
    quietest moment in the last 3 seconds of each window, never leaving a tail under min_seconds.
    Hard cutting loses no audio. Pieces under min_seconds are dropped. pad_ms must be at most half
    of min_silence_ms so padding of neighbouring pieces cannot overlap."""
    if pad_ms * 2 > min_silence_ms:
        raise ValueError("pad_ms must be at most half of min_silence_ms")
    audio = _clean(audio)
    db = _frame_db(audio, sr)
    n = sr * FRAME_MS // 1000
    gap = max(1, min_silence_ms // FRAME_MS)

    regions: list[tuple[int, int]] = []
    start = last = None
    for i in np.flatnonzero(db > threshold_db):
        i = int(i)
        if start is None:
            start = i
        elif i - last - 1 >= gap:
            regions.append((start, last + 1))
            start = i
        last = i
    if start is not None:
        regions.append((start, last + 1))

    pad = sr * pad_ms // 1000
    limit_frames = (int(max_seconds * sr) - 2 * pad) // n  # leaves room for padding at both ends
    min_frames = -(-int(min_seconds * sr) // n)
    if limit_frames < 2 * min_frames + 1:
        raise ValueError("max_seconds is too small to cut audio into pieces of at least min_seconds")
    search = min(limit_frames - 1, 3000 // FRAME_MS)

    pieces: list[np.ndarray] = []
    for first, end in regions:
        cuts = [first]
        while end - cuts[-1] > limit_frames:
            # Keep at least min_frames after the cut so the tail is not discarded as too short.
            window_end = min(cuts[-1] + limit_frames, end - min_frames)
            lo = max(window_end - search, cuts[-1] + min_frames)
            cuts.append(lo + int(np.argmin(db[lo:window_end])))
        cuts.append(end)
        for k in range(len(cuts) - 1):
            a = cuts[k] * n - (pad if k == 0 else 0)
            b = cuts[k + 1] * n + (pad if k == len(cuts) - 2 else 0)
            piece = audio[max(0, a) : min(len(audio), b)]
            if len(piece) >= min_seconds * sr:
                pieces.append(piece)
    return pieces
