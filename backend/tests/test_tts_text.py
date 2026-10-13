import random

import pytest

from aivoice import tts_text
from aivoice.tts_text import MAX_TEXT, clean, split


def speakable(text: str) -> str:
    return "".join(c for c in text if c.isalnum())


def check_pieces(pieces, max_chars, source):
    for p in pieces:
        assert p == p.strip()
        assert p
        assert len(p.encode("utf-8")) <= max_chars, p
        assert any(c.isalnum() for c in p), p
        assert p.rstrip("\"')]}»”’")[-1:] in ".!?…。！？", p
    assert speakable("".join(pieces)) == speakable(source)


# ---- clean ----

def test_clean_normalises_whitespace_and_newlines():
    assert clean("a  b\t\tc\r\nd\re") == "a b c d e"
    assert clean("one\n\n\n\ntwo") == "one\n\ntwo"
    assert clean("one \n \n two") == "one\n\ntwo"
    assert clean("  hi  ") == "hi"
    assert clean("a b c") == "a b c"


def test_clean_strips_control_and_invisible_characters():
    assert clean("a\x00b\x07c\x1b[0m") == "abc[0m"
    assert clean("zero​width﻿") == "zerowidth"
    assert clean("lone\ud800surrogate") == "lonesurrogate"


@pytest.mark.parametrize("bad", ["", "   ", "\n\n\t", "\x00\x01", "​"])
def test_clean_rejects_empty(bad):
    with pytest.raises(ValueError):
        clean(bad)


def test_clean_length_limit():
    assert len(clean("a" * MAX_TEXT)) == MAX_TEXT
    with pytest.raises(ValueError):
        clean("a" * (MAX_TEXT + 1))
    # whitespace that collapses away does not count
    assert clean("a" + " " * 9000 + "b") == "a b"


# ---- split: ordinary behaviour ----

def test_prose_one_piece_per_sentence():
    text = "It was a quiet morning. Nobody spoke! Why would they? The end."
    assert split(text, 200) == ["It was a quiet morning.", "Nobody spoke!", "Why would they?", "The end."]


def test_blank_lines_end_sentences_and_get_a_full_stop():
    assert split("Title line\n\nBody text here", 200) == ["Title line.", "Body text here."]
    assert split("Line one\nline two.", 200) == ["Line one line two."]


def test_multiple_blank_lines():
    assert split("a b\n\n\n\n\nc d", 200) == ["a b.", "c d."]


def test_combined_terminators_quotes_and_brackets():
    assert split("Wait... what?! Really.", 200) == ["Wait...", "what?!", "Really."]
    assert split('He said "stop." Then left.', 200) == ['He said "stop."', "Then left."]
    assert split("(It ended.) Next one (maybe?) Last", 200) == ["(It ended.)", "Next one (maybe?)", "Last."]
    assert split("Done.” And more.", 200) == ["Done.”", "And more."]


def test_quotes_and_brackets_after_final_stop_stay_with_sentence():
    assert split('She said, "Go home."', 200) == ['She said, "Go home."']
    assert split("Final words.)]}", 200) == ["Final words.)]}"]


def test_missing_final_stop_is_added_and_trailing_comma_replaced():
    assert split("no stop here", 200) == ["no stop here."]
    assert split("trailing comma,", 200) == ["trailing comma."]
    assert split("what about this?", 200) == ["what about this?"]


def test_never_splits_inside_numbers():
    assert split("It is 3.5 metres long. Fine.", 200) == ["It is 3.5 metres long.", "Fine."]
    assert split("It costs 1,000.25 dollars.", 200) == ["It costs 1,000.25 dollars."]
    assert split("Version 2.0.1 is out", 200) == ["Version 2.0.1 is out."]


def test_numbers_survive_a_forced_split():
    text = "alpha beta 3.5 gamma delta 1,000.25 epsilon zeta eta theta 3.5"
    pieces = split(text, 25)
    check_pieces(pieces, 25, text)
    assert any("3.5" in p for p in pieces) and any("1,000.25" in p for p in pieces)


def test_abbreviations_are_not_handled_specially():
    # Documented behaviour: "Dr." and "e.g." followed by a space end a sentence early.
    assert "abbreviations" in tts_text.__doc__.lower()
    assert split("Dr. Smith said hello.", 200) == ["Dr.", "Smith said hello."]
    assert split("Use fruit, e.g. apples.", 200) == ["Use fruit, e.g.", "apples."]
    # but dotted abbreviations without a following space are kept whole
    assert split("Use e.g.apples or i.e.pears.", 200) == ["Use e.g.apples or i.e.pears."]


def test_url_is_not_split_and_emoji_dropped():
    assert split("See https://example.com/a.b/c?d=1 now.", 200) == ["See https://example.com/a.b/c?d=1 now."]
    assert split("Hello \U0001F600 world.", 200) == ["Hello \U0001F600 world."]
    assert split("\U0001F600\U0001F680. Hi!", 200) == ["Hi!"]


def test_unspeakable_input_gives_no_pieces():
    for text in ["...", "?!?!", "- - -", "\U0001F600", "...\n\n!!!", "\"'()"]:
        assert split(text, 50) == []


def test_whitespace_only_is_rejected():
    for text in ["", " ", "\n\n\n", " \n \t "]:
        with pytest.raises(ValueError):
            split(text, 50)


def test_max_chars_minimum():
    with pytest.raises(ValueError):
        split("hello", 19)
    assert split("hello", 20) == ["hello."]


def test_over_long_text_rejected():
    with pytest.raises(ValueError):
        split("a " * 3000, 100)


# ---- split: long and awkward ----

def test_long_sentence_splits_at_clause_marks_first():
    text = "first clause here, second clause here; third clause here: fourth clause here."
    pieces = split(text, 40)
    check_pieces(pieces, 40, text)
    assert pieces[0] == "first clause here, second clause here."
    assert len(pieces) == 2


def test_long_sentence_splits_at_spaces_then_hard():
    text = " ".join(["word"] * 200)
    pieces = split(text, 50)
    check_pieces(pieces, 50, text)
    assert all(p.startswith("word") for p in pieces)
    assert split("x" * 100, 30) == ["x" * 29 + ".", "x" * 29 + ".", "x" * 29 + ".", "x" * 13 + "."]


def test_3000_chars_no_punctuation_no_spaces():
    text = "a" * 3000
    pieces = split(text, 178)
    check_pieces(pieces, 178, text)
    assert len(pieces) >= 17


def test_3000_chars_no_punctuation_with_spaces():
    text = ("lorem ipsum dolor " * 200)[:3000]
    pieces = split(text, 135)
    check_pieces(pieces, 135, text)


def test_long_runs_of_punctuation_are_handled():
    assert split("." * 5000, 100) == []
    assert split("." * 4999 + "x", 100) == ["x."] or speakable("".join(split("." * 4999 + "x", 100))) == "x"
    text = "a" + "!" * 4000 + "b"
    check_pieces(split(text, 100), 100, text)


def test_exactly_5000_characters():
    text = ("This is sentence number one. " * 200)[:5000]
    assert len(text) == 5000
    check_pieces(split(text, 100), 100, text)
    nodots = "a" * 5000
    check_pieces(split(nodots, 100), 100, nodots)


def test_multibyte_characters_never_split_inside_and_limit_is_in_bytes():
    text = "é" * 200  # 2 bytes each
    pieces = split(text, 21)
    check_pieces(pieces, 21, text)
    for p in pieces:
        p.encode("utf-8").decode("utf-8")
    text = "日本語" * 100  # 3 bytes each
    for limit in (20, 21, 22, 23, 50):
        check_pieces(split(text, limit), limit, text)
    text = "ab\U0001F600cd" * 60  # 4-byte emoji in the middle
    for limit in range(20, 40):
        check_pieces(split(text, limit), limit, text)


def test_max_chars_lands_inside_multibyte_character():
    text = "a" + "é" * 30
    for limit in range(20, 30):
        pieces = split(text, limit)
        check_pieces(pieces, limit, text)
        assert all("�" not in p for p in pieces)


def test_property_random_text():
    rng = random.Random(1234)
    alphabet = ["word", "a", "Hello", "3.5", "1,000.25", "e.g.", "Dr.", "x" * 40, "é" * 15, "日本", "\U0001F600",
                ".", "!", "?", "...", ",", ";", ":", '"', "(", ")", "-", " ", "  ", "\n", "\n\n", "\t"]
    for _ in range(300):
        text = "".join(rng.choice(alphabet) + rng.choice(["", " ", " "]) for _ in range(rng.randint(1, 60)))
        max_chars = rng.choice([20, 21, 25, 40, 80, 178])
        try:
            cleaned = clean(text)
        except ValueError:
            with pytest.raises(ValueError):
                split(text, max_chars)
            continue
        check_pieces(split(text, max_chars), max_chars, cleaned)
