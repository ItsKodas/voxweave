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

Numbers (`write_out_numbers`, applied by `split(..., numbers=True)`, which is what the manager uses). The model
reads most numbers correctly and two shapes wrongly (measured in the first real run), so only those two are
rewritten and every other digit is left exactly as typed:
- An integer written with thousands separators ("1,200", up to 999,999,999,999, with an optional decimal part)
  becomes words: "one thousand two hundred", the decimals digit by digit after "point".
- A phone number becomes its digits, one by one, with a comma between its groups: "555 0142" is "five five
  five, zero one four two". A run of digits counts as a phone number only if it has 7 to 15 digits and looks
  like one: see `_phone_groups` for what is, on purpose, left alone (dates, ranges, decimals, plain integers).
The length limit is on the text as typed: it is checked before the numbers are written out, and the written-out
text may be several times longer (at worst about eight times, for text that is nothing but twelve-digit
numbers). Nothing later limits the whole: the pieces are cut to the byte budget after the rewriting, so each one
still fits, there are just more of them.
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


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "- - twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ("", " thousand", " million", " billion")
# One to three digits, then one to three groups of exactly three after commas, then perhaps decimals; not part
# of a word, a longer number, a time, a fraction or a version.
_THOUSANDS = re.compile(r"(?<![\w.,:/])([1-9]\d{0,2}(?:,\d{3}){1,3})(\.\d+)?(?![\w:/]|[.,]\d)")
# Groups of digits joined by one space, hyphen or dot, with an area code in brackets and a leading plus sign
# allowed. Only a candidate: _phone_groups decides.
_OPENING = r"(?:\((?=\d+\)))?"
_PHONE = re.compile(r"(?<![\w.,:/$£€#+-])\+?" + _OPENING + r"\d+(?:(?:\) ?|[ .-]" + _OPENING
                    + r")\d+)*(?![\w%°]|[.,:/-]\d)")
_MIN_PHONE_DIGITS, _MAX_PHONE_DIGITS = 7, 15


def _under_a_thousand(n: int) -> str:
    words = []
    if n >= 100:
        words.append(_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else ""))
    elif n:
        words.append(_ONES[n])
    return " ".join(words)


def _digits(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _say_thousands(match: re.Match) -> str:
    groups = [int(g) for g in match.group(1).split(",")]
    words = [_under_a_thousand(g) + _SCALES[len(groups) - 1 - i] for i, g in enumerate(groups) if g]
    decimals = match.group(2)
    if decimals:
        words.append("point " + _digits(decimals[1:]))
    return " ".join(words)


def _looks_like_a_year(group: str) -> bool:
    return len(group) == 4 and group[0] in "12"


def _phone_groups(candidate: str) -> list[str] | None:
    """The digit groups of a phone number, or None for anything that might be something else. Wrongly reading a
    date or a quantity digit by digit is worse than leaving a phone number as it was, so when in doubt: None.

    Always left alone: fewer than 7 or more than 15 digits; a date (three groups shaped like day, month and
    year, either way round); two groups around a dot (a decimal); dotted groups shorter than three digits (a
    version, a machine's address). A leading plus sign, an area code in brackets, or a first group that starts
    with 0 marks a phone number, and then any grouping is accepted, an unbroken run too. Without such a mark an
    unbroken run is a plain integer, two groups must be three or four digits then four ("555 0142", "9555
    0142") and neither two years nor two round numbers (a range, a list), and three or more groups must end in
    a group of four ("555 123 4567"), which thousands grouped by spaces never do."""
    if candidate.count("(") != candidate.count(")"):
        return None
    groups = re.findall(r"\d+", candidate)
    total = sum(len(g) for g in groups)
    if not _MIN_PHONE_DIGITS <= total <= _MAX_PHONE_DIGITS:
        return None
    lengths = [len(g) for g in groups]
    bracketed = "(" in candidate
    plus = candidate.startswith("+")
    if not (plus or bracketed) and len(groups) == 3 and (
            (lengths[0] <= 2 and lengths[1] <= 2 and lengths[2] == 4)
            or (lengths[0] == 4 and lengths[1] <= 2 and lengths[2] <= 2)):
        return None
    if "." in candidate and (len(groups) == 2 or min(lengths) < 3):
        return None
    if plus or bracketed or (groups[0][0] == "0" and lengths[0] >= 2):
        return groups
    if len(groups) == 1:
        return None
    if len(groups) == 2:
        if lengths not in ([3, 4], [4, 4]):
            return None
        if all(_looks_like_a_year(g) for g in groups) or all(g[-1] == "0" for g in groups):
            return None
        return groups
    return groups if lengths[-1] == 4 else None


def _say_phone(match: re.Match) -> str:
    candidate = match.group()
    groups = _phone_groups(candidate)
    if groups is None:
        return candidate
    return ("plus " if candidate.startswith("+") else "") + ", ".join(_digits(g) for g in groups)


def write_out_numbers(text: str) -> str:
    """The text with the two kinds of number the model misreads written as words (see the module's notes).
    Everything else, including every other number, comes back exactly as it was."""
    return _PHONE.sub(_say_phone, _THOUSANDS.sub(_say_thousands, text))


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


def split(text: str, max_chars: int, numbers: bool = False) -> list[str]:
    """Split text into speakable pieces of at most max_chars UTF-8 bytes (full stop included). With numbers,
    the numbers the model misreads are written out first: after the length check, before the cutting."""
    if max_chars < MIN_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_CHARS}")
    pieces: list[str] = []
    cleaned = clean(text)
    if numbers:
        cleaned = write_out_numbers(cleaned)
    for paragraph in cleaned.split("\n\n"):
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
