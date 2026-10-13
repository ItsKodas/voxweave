"""Choosing the clip that F5-TTS clones a voice from. Reads Voice and Clip objects only."""
from aivoice.library import Clip, Voice

MIN_SECONDS = 3.0
MAX_SECONDS = 12.0
PREFERRED_SECONDS = 8.0
NO_CLIP = "This voice has no clip between 3 and 12 seconds with a transcript. Record a sentence for it first."


def _usable(clip: Clip) -> bool:
    return bool(clip.text and clip.text.strip()) and MIN_SECONDS <= clip.seconds <= MAX_SECONDS


def choose(voice: Voice) -> Clip:
    """Recorded clips before imported ones, then the length closest to 8 s, then the lowest index."""
    usable = [c for c in voice.clips if _usable(c)]
    if not usable:
        raise ValueError(NO_CLIP)
    return min(usable, key=lambda c: (c.source != "recorded", abs(c.seconds - PREFERRED_SECONDS), c.index))


def describe(voice: Voice) -> dict:
    try:
        clip = choose(voice)
    except ValueError as e:
        return {"ok": False, "clip": None, "seconds": None, "message": str(e)}
    return {"ok": True, "clip": clip.index, "seconds": clip.seconds,
            "message": f"Clip {clip.index} ({clip.seconds:.1f} seconds) will be used as the reference."}
