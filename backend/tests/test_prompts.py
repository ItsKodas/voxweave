import re
from collections import Counter

from aivoice.prompts import load_prompts, next_prompt

FIRST_TEN = [
    "The quiet river bends just north of the old stone bridge.",
    "She packed a thick wool jumper for the chilly evening hike.",
    "Which of these boxes should go up to the attic today?",
    "A young fox jumped over the fence and vanished in the hedge.",
    "Please measure the garage door before you order a new one.",
    "They usually share a pot of strong tea after the morning shift.",
    "The children enjoyed the noisy parade through the town square.",
    "He fixed the buzzing light with a ladder and a spare fuse.",
    "Your television is far too loud for this time of night.",
    "We watched the purple clouds drift slowly above the harbour.",
]

SOUNDS = ["th", "sh", "ch", "ng", "wh", "oo", "ee", "ou", "ow", "oi", "oy", "ar", "er", "or", "ai", "ay",
          "ea", "igh", "qu", "ck", "j", "v", "z", "x", "sion", "sure", "ure", "au", "aw", "ph"]


def test_two_hundred_unique_sentences():
    prompts = load_prompts()
    assert len(prompts) == 200
    assert len(set(p.lower() for p in prompts)) == 200
    assert list(prompts[:10]) == FIRST_TEN


def test_every_sentence_is_readable_aloud_in_one_breath():
    for p in load_prompts():
        words = p.split()
        assert 6 <= len(words) <= 14, p
        assert len(p) <= 90, p
        assert p[0].isupper() and p[-1] in ".?", p
        assert re.fullmatch(r"[A-Za-z ,.'?-]+", p), p
        assert "  " not in p and p == p.strip(), p


def test_the_set_covers_english_sounds():
    text = " ".join(load_prompts()).lower()
    letters = Counter(ch for ch in text if ch.isalpha())
    assert all(letters[ch] >= 10 for ch in "abcdefghijklmnopqrstuvwxyz"), sorted(letters.items(), key=lambda kv: kv[1])[:5]
    for sound in SOUNDS:
        assert text.count(sound) >= 5, sound


def test_no_word_dominates():
    words = Counter(w.strip(",.?'").lower() for p in load_prompts() for w in p.split())
    common = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "with", "is", "was", "her", "his", "at", "by", "from", "that", "this", "it", "you", "we", "they", "he", "she", "i", "my", "your", "our", "their"}
    worst = max((count for word, count in words.items() if word not in common), default=0)
    assert worst <= 8


def test_about_one_in_ten_is_a_question():
    questions = sum(p.endswith("?") for p in load_prompts())
    assert 10 <= questions <= 40


def test_next_prompt_starts_at_the_first_unrecorded():
    assert next_prompt(set(), 5) == 0
    assert next_prompt({0, 1}, 5) == 2


def test_next_prompt_after_skips_forward_and_wraps():
    assert next_prompt(set(), 5, after=1) == 2
    assert next_prompt({2, 3}, 5, after=1) == 4
    assert next_prompt({0}, 5, after=4) == 1


def test_skipping_the_only_one_left_returns_it_again():
    assert next_prompt({0, 1, 2, 4}, 5, after=3) == 3


def test_next_prompt_none_when_all_recorded():
    assert next_prompt({0, 1, 2}, 3) is None
    assert next_prompt({0, 1, 2}, 3, after=1) is None


def test_out_of_range_after_is_treated_as_start():
    assert next_prompt(set(), 3, after=99) == 0
    assert next_prompt(set(), 3, after=-5) == 0


def test_paragraphs_are_distinct_short_enough_to_read_in_one_take_and_fit_a_transcript():
    from aivoice.library import MAX_TEXT
    from aivoice.prompts import load_paragraphs

    paragraphs = load_paragraphs()
    assert len(paragraphs) == 40 and len(set(paragraphs)) == 40
    for p in paragraphs:
        words = len(p.split())
        # 25 to 38 words: about 10 to 15 seconds aloud, inside the 20 second limit on a take.
        assert 25 <= words <= 38, p
        assert len(p) <= MAX_TEXT and p[-1] in ".?!"
        assert sum(p.count(mark) for mark in ".?!") >= 2, p
        assert not any(ch.isdigit() for ch in p), p
    assert not set(paragraphs) & set(load_prompts())
