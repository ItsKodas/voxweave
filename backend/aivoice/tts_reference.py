"""Choosing the clip that F5-TTS clones a voice from. Reads Voice and Clip objects and, to judge a clip's level,
the clip's own audio file. Nothing is ever written."""
import math
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from aivoice.audiofile import read_wav
from aivoice.nonverbal import usable_as_reference
from aivoice.library import Clip, Library, LibraryError, Voice

MIN_SECONDS = 3.0
MAX_SECONDS = 12.0
PREFERRED_SECONDS = 8.0
# The worker's floor (tts/tts_worker.py, MIN_REFERENCE_RMS; a test holds the two together): the engine raises a
# quieter clip to its working level and lowers what it generates by as much, so every piece would fail as silent.
MIN_RMS = 0.003
NO_CLIP = "This voice has no clip between 3 and 12 seconds with a transcript. Record a sentence for it first."
RECORD_LOUDER = "Record a louder clip of 5 to 10 seconds."
NAMED_AT_MOST = 3  # more clips than this are counted in the message, not listed
CACHED_LEVELS = 4096

_levels: dict[tuple, float] = {}  # (path, modification time, size) -> RMS, so a clip is read once, not per poll
_levels_lock = threading.Lock()


def _usable(clip: Clip) -> bool:
    # Only plain speech: a laugh, a cough or a clip with a bracketed sound in it would be copied into every sentence.
    return (bool(clip.text and clip.text.strip()) and MIN_SECONDS <= clip.seconds <= MAX_SECONDS
            and usable_as_reference(getattr(clip, "kind", "speech"), clip.text))


def too_quiet(indexes: list[int]) -> str:
    """What to tell the user when these clips, the only candidates, are all too quiet to copy a voice from."""
    if len(indexes) == 1:
        return f"Clip {indexes[0]} is too quiet to copy. {RECORD_LOUDER}"
    if len(indexes) <= NAMED_AT_MOST:
        names = ", ".join(str(i) for i in indexes[:-1]) + f" and {indexes[-1]}"
        return f"Clips {names} are too quiet to copy. {RECORD_LOUDER}"
    return (f"All {len(indexes)} clips of this voice that are between 3 and 12 seconds long are too quiet to copy. "
            f"{RECORD_LOUDER}")


def clip_level(path) -> float | None:
    """The RMS level of a clip's audio file, measured as the worker measures it (the whole clip, channels
    averaged). None if the file cannot be read. An unchanged file is read only once."""
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with _levels_lock:
            if key in _levels:
                return _levels[key]
        audio, _ = read_wav(path)
        level = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    except (OSError, ValueError):
        return None
    with _levels_lock:
        if len(_levels) >= CACHED_LEVELS:
            _levels.clear()
        _levels[key] = level
    return level


def library_levels(library: Library, voice_id: str) -> Callable[[Clip], float | None]:
    """level(clip) for the clips of one voice in the library: None when the clip's audio is missing or broken."""

    def level(clip: Clip) -> float | None:
        try:
            return clip_level(library.clip_audio_path(voice_id, clip.index))
        except LibraryError:
            return None

    return level


def choose(voice: Voice, level: Callable[[Clip], float | None] | None = None) -> Clip:
    """Recorded clips before imported ones, then the length closest to 8 s, then the lowest index.

    With level(clip) -> the clip's RMS or None, the first clip in that order that is loud enough to copy wins,
    and clips are only measured until one is found. A clip whose audio cannot be read (None) is chosen only if
    no clip is known to be loud enough, so that the caller reports the missing audio. If every candidate is
    known to be too quiet, the ValueError says which clips and what to do about it."""
    usable = sorted((c for c in voice.clips if _usable(c)),
                    key=lambda c: (c.source != "recorded", abs(c.seconds - PREFERRED_SECONDS), c.index))
    if not usable:
        raise ValueError(NO_CLIP)
    if level is None:
        return usable[0]
    unknown = None
    for clip in usable:
        rms = level(clip)
        if rms is None:
            unknown = unknown or clip
        elif math.isfinite(rms) and rms >= MIN_RMS:
            return clip
    if unknown is not None:
        return unknown
    raise ValueError(too_quiet(sorted(c.index for c in usable)))


def describe(voice: Voice, level: Callable[[Clip], float | None] | None = None) -> dict:
    try:
        clip = choose(voice, level)
    except ValueError as e:
        return {"ok": False, "clip": None, "seconds": None, "message": str(e)}
    return {"ok": True, "clip": clip.index, "seconds": clip.seconds,
            "message": f"Clip {clip.index} ({clip.seconds:.1f} seconds) will be used as the reference."}
