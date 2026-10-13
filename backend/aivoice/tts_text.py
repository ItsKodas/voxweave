"""Text preparation for text to speech: cleaning and splitting into pieces the model can say.

The model fails silently on bad input (babble, dropped words, near-silence), so this module is a guard.
`split` never returns an empty piece, a piece longer than `max_chars` UTF-8 bytes, or a piece with no
letter or digit in it.

Rules
- A sentence ends at `.`, `!`, `?` or `...` (runs of them, optionally followed by closing quotes or brackets)
  when whitespace or the end of the text follows, and at a blank line. Single newlines are plain spaces.
- Because a full stop must be followed by whitespace to count, "3.5 metres", "1,000.25", "v2.0.1" and URLs
  are never split inside.
- Abbreviations are NOT handled specially: "Dr. Smith said hi." becomes "Dr." and "Smith said hi.", and
  "e.g. this" splits after "e.g.". That may cut a sentence early, which costs a little intonation but never
  loses a word. ("e.g.apples", without a space, stays whole.)
- A sentence longer than the limit is cut at `; : ,` followed by a space (clauses are packed together up to
  the limit), then at spaces, then, for a single word that is itself too long, at a character boundary.
  A cut never falls inside a UTF-8 sequence.
- Each piece is stripped. A piece that does not end in sentence punctuation gets a full stop (a trailing
  comma, semicolon or colon is replaced by it), since the model reads those more steadily. The full stop
  counts toward the limit.
- Pieces with nothing speakable (no letter or digit: only punctuation or emoji) are dropped, so `split`
  returns an empty list for such text. `clean` rejects text that is empty after cleaning.
"""

import re
import unicodedata

MAX_TEXT = 5000
MIN_CHARS = 20

_CLOSERS = "\"'’”)]}»"
_TERMINATORS = ".!?…"
_SENTENCE = re.compile(r"\S.*?(?:[.!?…]+[\"'’”)\]}»]*(?=\s|\Z)|\Z)", re.DOTALL)
_CLAUSE_BREAK = re.compile(r"(?<=[;:,])\s+")
_NEWLINE_RUN = re.compile(r" *\n[ \n]*")
_DROPPED_CATEGORIES = {"Cc", "Cf", "Cs"}


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def clean(text: str) -> str:
    """Normalise newlines and whitespace, strip control characters, and enforce 1 to MAX_TEXT characters."""
    if len(text) > 20 * MAX_TEXT:  # cheap guard before any per-character work
        raise ValueError(f"The text is too long. Keep it under {MAX_TEXT} characters.")
    chars = []
    for c in text.replace("\r\n", "\n"):
        if c == "\n":
            chars.append(c)
        elif c.isspace():
            chars.append(" ")
        elif unicodedata.category(c) not in _DROPPED_CATEGORIES:
            chars.append(c)
    out = re.sub(r" +", " ", "".join(chars))
    out = _NEWLINE_RUN.sub(lambda m: "\n\n" if m.group().count("\n") >= 2 else " ", out)
    out = re.sub(r" +", " ", out).strip()
    if not out:
        raise ValueError("There is no text to speak.")
    if len(out) > MAX_TEXT:
        raise ValueError(f"The text is too long ({len(out)} characters). Keep it under {MAX_TEXT} characters.")
    return out


def _is_terminated(piece: str) -> bool:
    body = piece.rstrip(_CLOSERS)
    return bool(body) and body[-1] in _TERMINATORS


def _hard_split(word: str, budget: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for c in word:
        n = _nbytes(c)
        if size + n > budget and cur:
            out.append("".join(cur))
            cur, size = [], 0
        cur.append(c)
        size += n
    if cur:
        out.append("".join(cur))
    return out


def _pack(parts: list[str], budget: int, oversize) -> list[str]:
    """Join parts with single spaces into pieces of at most budget bytes; oversize() splits a too-big part."""
    out: list[str] = []
    cur = ""
    for part in parts:
        if _nbytes(part) > budget:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(oversize(part, budget))
            continue
        cand = f"{cur} {part}" if cur else part
        if _nbytes(cand) <= budget:
            cur = cand
        else:
            out.append(cur)
            cur = part
    if cur:
        out.append(cur)
    return out


def _split_words(text: str, budget: int) -> list[str]:
    return _pack(text.split(), budget, _hard_split)


def _split_clauses(text: str, budget: int) -> list[str]:
    return _pack(_CLAUSE_BREAK.split(text), budget, _split_words)


def _finish_piece(piece: str) -> str | None:
    piece = piece.strip()
    if not any(c.isalnum() for c in piece):
        return None
    if not _is_terminated(piece):
        piece = piece.rstrip(" ,;:") + "."
    return piece


def split(text: str, max_chars: int) -> list[str]:
    """Split text into speakable pieces of at most max_chars UTF-8 bytes (full stop included)."""
    if max_chars < MIN_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_CHARS}")
    pieces: list[str] = []
    for paragraph in clean(text).split("\n\n"):
        for match in _SENTENCE.finditer(paragraph):
            sentence = match.group().strip()
            size = _nbytes(sentence)
            if size <= max_chars and (_is_terminated(sentence) or size < max_chars):
                candidates = [sentence]
            else:
                candidates = _split_clauses(sentence, max_chars - 1)
            for candidate in candidates:
                piece = _finish_piece(candidate)
                if piece is not None:
                    pieces.append(piece)
    return pieces
