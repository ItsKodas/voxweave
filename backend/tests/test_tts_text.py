import random

import pytest

from aivoice import tts_text
from aivoice.tts_text import MAX_TEXT, clean, split, write_out_numbers


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


# ---- numbers the model misreads (write_out_numbers; split(..., numbers=True) is what the manager uses) ----
# Measured in the first real run: "1,200" was never read correctly and a phone number was garbled about half the
# time, at every step count. Years, "3.5", "40" and "70 percent" were read correctly, so they are left alone.

@pytest.mark.parametrize("written, said", [
    ("1,200", "one thousand two hundred"),
    ("1,000", "one thousand"),
    ("1,001", "one thousand one"),
    ("2,015", "two thousand fifteen"),
    ("10,000", "ten thousand"),
    ("12,345", "twelve thousand three hundred forty-five"),
    ("999,999", "nine hundred ninety-nine thousand nine hundred ninety-nine"),
    ("1,000,000", "one million"),
    ("2,500,000", "two million five hundred thousand"),
    ("1,000,000,000", "one billion"),
    ("7,000,000,019", "seven billion nineteen"),
    ("999,999,999,999", "nine hundred ninety-nine billion nine hundred ninety-nine million nine hundred "
                        "ninety-nine thousand nine hundred ninety-nine"),
    ("1,200.5", "one thousand two hundred point five"),
    ("1,000.25", "one thousand point two five"),
    ("3,400.007", "three thousand four hundred point zero zero seven"),
])
def test_integers_with_thousands_separators_are_written_out(written, said):
    assert write_out_numbers(written) == said
    assert write_out_numbers(f"It costs {written} dollars.") == f"It costs {said} dollars."
    assert write_out_numbers(f"About {written}.") == f"About {said}."
    assert write_out_numbers(f"({written})") == f"({said})"


@pytest.mark.parametrize("text", [
    "1,200,000,000,000",  # a trillion and beyond: not rewritten
    "0,200", "01,200",  # not a number written that way
    "1,20", "1,2000", "12,34,567", "1,200,5",  # not groups of three
    "1.200,50",  # another country's way of writing it
    "v1,200", "1,200th", "x1,200", "1,200x", "1,200.5.3", "1,200:30", "3/1,200", "1,200/3", "2.1,200",
    "13, 2026", "1, 200",  # a comma between words
    "1200", "40", "3.5", "1987", "2004", "70 percent", "1200000", "0.25", "12:30", "5:05 pm",
    "2026-10-13", "13/10/2026", "13.10.2026", "October 13, 2026", "the 1990s", "100%", "No. 5", "v2.0.1",
])
def test_everything_else_with_digits_in_it_is_left_exactly_as_it_is(text):
    assert write_out_numbers(text) == text
    sentence = f"It says {text} here."
    assert write_out_numbers(sentence) == sentence


def test_a_currency_or_minus_sign_stays_where_it_is():
    # only the number is rewritten; how the model reads the sign is not something this decides
    assert write_out_numbers("$1,200") == "$one thousand two hundred"
    assert write_out_numbers("-1,200") == "-one thousand two hundred"
    assert write_out_numbers("1,000-2,000") == "one thousand-two thousand"


@pytest.mark.parametrize("written, said", [
    ("555 0142", "five five five, zero one four two"),  # the one measured
    ("555-0142", "five five five, zero one four two"),
    ("9555 0142", "nine five five five, zero one four two"),
    ("555-123-4567", "five five five, one two three, four five six seven"),
    ("555.123.4567", "five five five, one two three, four five six seven"),
    ("1800 123 456", "one eight zero zero, one two three, four five six"),
    ("1300 655 506", "one three zero zero, six five five, five zero six"),
    ("1800-123-456", "one eight zero zero, one two three, four five six"),
    ("555 123-4567", "five five five, one two three, four five six seven"),
    ("(555) 123-4567", "five five five, one two three, four five six seven"),
    ("(02) 9555 0142", "zero two, nine five five five, zero one four two"),
    ("0412 345 678", "zero four one two, three four five, six seven eight"),
    ("0412345678", "zero four one two three four five six seven eight"),
    ("01632 960123", "zero one six three two, nine six zero one two three"),
    ("01 23 45 67 89", "zero one, two three, four five, six seven, eight nine"),
    ("+61 412 345 678", "plus six one, four one two, three four five, six seven eight"),
    ("+61412345678", "plus six one four one two three four five six seven eight"),
    ("+1 (555) 123-4567", "plus one, five five five, one two three, four five six seven"),
    ("+44 (0)20 7946 0958", "plus four four, zero, two zero, seven nine four six, zero nine five eight"),
])
def test_phone_numbers_are_read_digit_by_digit_with_a_pause_between_groups(written, said):
    assert write_out_numbers(written) == said
    assert write_out_numbers(f"Call {written} today.") == f"Call {said} today."
    assert write_out_numbers(f"Call {written}.") == f"Call {said}."
    assert write_out_numbers(f"(call {written})") == f"(call {said})"
    assert "oh" not in said.replace(",", "").split() and "o" not in said.replace(",", "").split()


@pytest.mark.parametrize("text", [
    # plain integers, however long: only a leading zero or plus sign makes an unbroken run a phone number
    "5550142", "1500000", "12345678901",
    # fewer than seven digits, or more than fifteen
    "555 014", "12 34 56", "1234 5678 9012 3456", "+1234567890123456",
    # dates
    "2026-10-13", "13-10-2026", "10-13-2026", "01-02-2026", "2026 10 13", "13 10 2026", "13.10.2026",
    "2026.10.13",
    # two years, a range of years, ranges and lists of round quantities
    "1987 2004", "1987-2004", "1000-2000", "500-1000", "3000-5000", "250 1000",
    # decimals, versions, addresses of machines
    "555.0142", "1234.5678", "0.1234567", "3.14159265", "1.2.3.4567", "192.168.1.100", "10.0.0.1234",
    # groups set apart by spaces only, with nothing that marks a phone number: they were read correctly as typed
    "2023 4521", "500 3271", "101 2345", "2024 2025 2026", "10 20 30 4000", "2 500 1000", "15 000 0000",
    "555 123 4567", "1 800 555 0199", "9555 1142", "1900 123 456", "1800 1234 56", "555 0000",
    # a date first, whatever follows it
    "5-10-2026 1200", "1-10-2026 2-10-2026", "2026-10-13 1500", "01-02-2026 1234", "13.10.2026 5550142",
    # groups that are not a phone number's
    "100 200 300", "1 200 000", "12 345 678", "555 0142 12", "12345 67890", "55 501 42", "1 2 3 4 5 6 7",
    # stuck to something else
    "x5550142", "0412345678x", "#0412345678", "$555 0142", "555 0142%", "12:30 555:0142", "0412/345/678",
    "555 0142.5", "A-555 0142", "555) 0142 1234", "(555 0142) 1234",
])
def test_digits_that_are_not_clearly_a_phone_number_are_left_exactly_as_they_are(text):
    assert write_out_numbers(text) == text
    sentence = f"It says {text} here."
    assert write_out_numbers(sentence) == sentence


def test_a_phone_number_in_brackets_is_read_and_the_brackets_stay():
    assert write_out_numbers("(555 0142)") == "(five five five, zero one four two)"
    assert write_out_numbers("My number (0412 345 678) is new.") == (
        "My number (zero four one two, three four five, six seven eight) is new.")


def test_a_phone_number_next_to_a_time_is_still_found():
    assert write_out_numbers("At 12:30 call 555 0142, or 1,200 times.") == (
        "At 12:30 call five five five, zero one four two, or one thousand two hundred times.")


def test_text_without_numbers_is_unchanged_byte_for_byte():
    rng = random.Random(99)
    alphabet = ["word", "a", "Hello", "e.g.", "Dr.", "é" * 5, "日本", "\U0001F600", ".", "!", "?", "...", ",", ";",
                ":", '"', "(", ")", "-", "+", " ", "  ", "\n", "\n\n", "\t", "$", "%", "/"]
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        assert write_out_numbers(text) == text
        assert write_out_numbers(text).encode("utf-8") == text.encode("utf-8")
    assert write_out_numbers("") == ""


def test_the_numbers_that_were_read_correctly_are_unchanged_byte_for_byte():
    text = "In 1987 it was 3.5 metres, 40 wide, 70 percent full, at 12:30 on 13/10/2026. Version 2.0.1, 1200 units."
    assert write_out_numbers(text) == text


def test_split_writes_numbers_out_only_when_asked():
    text = "It costs 1,200 dollars. Call 555 0142."
    assert split(text, 200) == ["It costs 1,200 dollars.", "Call 555 0142."]
    assert split(text, 200, numbers=True) == ["It costs one thousand two hundred dollars.",
                                              "Call five five five, zero one four two."]


def test_the_length_limit_applies_to_the_text_as_typed_and_the_pieces_still_fit_the_byte_budget():
    typed = ("999,999,999,999 " * 313)[:MAX_TEXT].strip()  # under 5000 characters that become about 38,000
    assert len(typed) <= MAX_TEXT
    for limit in (20, 80, 151):
        pieces = split(typed, limit, numbers=True)
        assert sum(len(p) for p in pieces) > 6 * MAX_TEXT
        for piece in pieces:
            assert piece == piece.strip() and piece
            assert len(piece.encode("utf-8")) <= limit, piece
            assert any(c.isalnum() for c in piece)
        said = "".join(c for c in "".join(pieces) if c.isalnum())
        assert said == "".join(c for c in write_out_numbers(typed) if c.isalnum())
    with pytest.raises(ValueError, match="too long"):
        split("1,200 " * 1000, 80, numbers=True)  # 5999 characters as typed: refused, whatever they become


def test_property_random_text_with_numbers_written_out():
    rng = random.Random(4321)
    alphabet = ["word", "a", "Hello", "3.5", "1,000.25", "1,200", "555 0142", "+61 412 345 678", "e.g.", "Dr.",
                "x" * 40, "é" * 15, "日本", "\U0001F600", ".", "!", "?", "...", ",", ";", ":", '"', "(", ")", "-", " ",
                "  ", "\n", "\n\n", "\t"]
    for _ in range(300):
        text = "".join(rng.choice(alphabet) + rng.choice(["", " ", " "]) for _ in range(rng.randint(1, 60)))
        max_chars = rng.choice([20, 21, 25, 40, 80, 178])
        try:
            cleaned = clean(text)
        except ValueError:
            with pytest.raises(ValueError):
                split(text, max_chars, numbers=True)
            continue
        check_pieces(split(text, max_chars, numbers=True), max_chars, write_out_numbers(cleaned))


@pytest.mark.parametrize("sentence", [
    "In 2023 4521 people attended.",
    "Of the 500 3271 were counted.",
    "It is in room 101 2345 times a year.",
    "On 5-10-2026 1200 people came.",
    "From 1-10-2026 2-10-2026 is one day.",
    "By 2026-10-13 1500 had signed.",
    "In 2024 2025 2026 it rose.",
    "Count 10 20 30 4000 and stop.",
    "Between 2 500 1000 and more.",
    "About 15 000 0000 units.",
])
def test_numbers_that_stand_next_to_each_other_are_not_taken_for_a_phone_number(sentence):
    assert write_out_numbers(sentence) == sentence
    assert split(sentence, 200, numbers=True) == [sentence]


def test_the_measured_phone_number_is_still_read_digit_by_digit():
    # "555 0142" is what the model garbled in the first real run. Set apart by a space only, a last group of
    # four digits that begins with 0 is what marks it: no quantity, year or group of thousands is written so.
    assert write_out_numbers("Call 555 0142 before noon.") == "Call five five five, zero one four two before noon."
    assert write_out_numbers("Call 9555 0142.") == "Call nine five five five, zero one four two."
    assert write_out_numbers("Call 555 1142.") == "Call 555 1142."
