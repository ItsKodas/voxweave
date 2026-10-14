"""Telling laughs, coughs and other sounds apart from speech in imported clips.

A voice model for live mode learns every sound in its clips, so laughs and coughs in the training data are what let
it laugh and cough in that voice. Text to speech, on the other hand, copies its voice from one clip together with
that clip's transcript, so it must only ever be given plain speech. Each clip therefore gets a kind:

- "speech": words (the default, and what text to speech may use);
- "laugh", "cough", "breath", "hum", "filler" ("um", "mm-hmm") and "noise" (anything else that is not words).

The labels come from two things that are already at hand when a clip is imported: what the speech recogniser wrote
(Whisper writes a laugh as "Hahaha" or "(laughs)", a hum as "Mmmm", and invents short phrases such as "Thank you."
for a cough), and the sound itself (how much of it is voiced, how long it is). They are guesses, and the Clips list
lets the person correct them.
"""
import re

import numpy as np

KINDS = ("speech", "laugh", "cough", "breath", "hum", "filler", "noise")
NONVERBAL = tuple(k for k in KINDS if k != "speech")

# Annotations Whisper writes when non-speech symbols are allowed: "(laughs)", "[coughing]", "*sigh*".
_ANNOTATION = re.compile(r"[\(\[\*]\s*([a-z ]{2,30}?)\s*[\)\]\*]", re.I)
_ANNOTATION_KINDS = (
    ("laugh", ("laugh", "chuckl", "giggl")),
    ("cough", ("cough", "throat", "sneez")),
    ("breath", ("breath", "sigh", "inhal", "exhal", "gasp", "pant", "sniff")),
    ("hum", ("hum", "sing", "music")),
)
_LAUGH_WORDS = re.compile(r"^(?:(?:a?ha|he|hah|heh|hehe|hihi|lol|lmao|haha)+[\s,.!?~-]*)+$", re.I)
_HUM_WORDS = re.compile(r"^(?:m{3,}|a{3,}h*|o{3,}h*|e{3,}h*|u{3,}h*)[\s,.!?~-]*$", re.I)
_FILLER_WORDS = re.compile(r"^(?:(?:u+m+|u+h+|e+r+m*|hm+|mm+-?hm+|uh-?huh|ah+|oh+|ooh+)[\s,.!?~-]*)+$", re.I)
# Short phrases Whisper is known to invent for a cough, a breath or noise.
_INVENTED = {"thank you.", "thank you", "thanks.", "you", "you.", "bye.", "bye", "okay.", "oh.", "uh.", "hmm.",
             "so", "so.", "i", "i.", ".", "...", ""}


def strip_annotations(text: str) -> tuple[str, list[str]]:
    """The transcript without "(laughs)"-style annotations, and the annotations it had."""
    found = [m.group(1).strip().lower() for m in _ANNOTATION.finditer(text or "")]
    cleaned = re.sub(r"\s{2,}", " ", _ANNOTATION.sub(" ", text or "")).strip()
    return cleaned, found


def _annotation_kind(annotations: list[str]) -> str | None:
    for kind, stems in _ANNOTATION_KINDS:
        if any(stem in a for a in annotations for stem in stems):
            return kind
    return None


def voiced_share(audio: np.ndarray, sr: int) -> float:
    """How much of the loud part of the clip has a pitch (0 to 1)."""
    import pyworld
    import soxr

    x = soxr.resample(np.nan_to_num(np.asarray(audio, dtype=np.float32)), sr, 16000).astype(np.float64)
    if len(x) < 1600 or not np.any(np.abs(x) > 1e-4):
        return 0.0
    f0, times = pyworld.dio(x, 16000, f0_floor=60, f0_ceil=900, frame_period=10)
    f0 = pyworld.stonemask(x, f0, times, 16000)
    frames = min(len(f0), len(x) // 160)
    f0 = f0[:frames]
    level = np.sqrt(np.mean(x[: frames * 160].reshape(frames, 160) ** 2, axis=1) + 1e-12)
    loud = 20 * np.log10(level) > 20 * np.log10(level.max()) - 25
    return float(np.mean(f0[loud] > 0)) if loud.any() else 0.0


def classify(text: str, audio: np.ndarray, sr: int, no_speech: float | None = None) -> tuple[str, str]:
    """(kind, transcript to keep) for one clip. Non-speech clips are given a transcript like "[laugh]"."""
    cleaned, annotations = strip_annotations(text)
    words = cleaned.strip()
    lowered = words.lower()
    annotated = _annotation_kind(annotations)
    seconds = len(audio) / sr if sr else 0.0

    if annotated and not re.search(r"[a-z]{2,}", re.sub(r"\b(ha|he|heh|hah|um|uh|mm)\b", "", lowered)):
        return annotated, f"[{annotated}]"
    if words and _LAUGH_WORDS.match(words):
        return "laugh", f"[laugh] {words}"
    if words and _HUM_WORDS.match(words):
        return "hum", f"[hum] {words}"
    if words and _FILLER_WORDS.match(words):
        return "filler", words  # "um" and "mm-hmm" keep their words: they are speech-like, just not for references
    if lowered in _INVENTED or (no_speech is not None and no_speech > 0.6 and len(words.split()) <= 3):
        # No real words: decide from the sound. A cough is short and mostly unvoiced; a breath is unvoiced; a
        # voiced sound with no words is hummed or vocalised.
        share = voiced_share(audio, sr)
        if share < 0.35:
            return ("cough" if seconds <= 1.5 else "breath"), "[cough]" if seconds <= 1.5 else "[breath]"
        if share > 0.6:
            return "hum", "[hum]"
        return "noise", "[noise]"
    if annotated:
        # Words with a laugh or a cough in among them: speech, but the annotation is remembered in the transcript
        # so that text to speech does not pick it as a reference.
        return "speech", f"{words} [{annotated}]"
    return "speech", words


def usable_as_reference(kind: str, text: str) -> bool:
    """Whether text to speech may copy a voice from this clip: plain speech, with no bracketed sound in it."""
    return kind == "speech" and "[" not in (text or "")
