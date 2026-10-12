import numpy as np

from aivoice.dataset import (
    MAX_RECORD_SECONDS, PROBLEM_MESSAGES, clip_problem, level_problem, split_on_silence, trim_silence,
)

SR = 48000
RNG = np.random.default_rng(0)


def noise(seconds, amp=0.3):
    return RNG.uniform(-amp, amp, int(seconds * SR)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_trim_removes_leading_and_trailing_silence_keeping_padding():
    audio = np.concatenate([silence(1.0), noise(2.0), silence(1.5)])
    trimmed = trim_silence(audio, SR)
    assert abs(len(trimmed) / SR - 2.2) < 0.03


def test_trim_of_silence_is_empty():
    assert len(trim_silence(silence(2.0), SR)) == 0


def test_trim_of_empty_input_is_empty():
    assert len(trim_silence(np.zeros(0, dtype=np.float32), SR)) == 0


def test_trim_keeps_internal_pauses():
    audio = np.concatenate([noise(1.0), silence(1.0), noise(1.0)])
    assert abs(len(trim_silence(audio, SR)) / SR - 3.0) < 0.03


def test_trim_survives_nan():
    audio = noise(1.0)
    audio[100] = np.nan
    assert np.all(np.isfinite(trim_silence(audio, SR)))


def test_level_ok():
    assert level_problem(noise(2.0)) is None


def test_level_too_quiet():
    assert level_problem(noise(2.0, amp=0.01)) == "too_quiet"


def test_level_clipped():
    audio = noise(2.0)
    audio[1000:1400] = 1.0
    assert level_problem(audio) == "clipped"


def test_a_few_full_scale_samples_are_not_clipping():
    audio = noise(2.0)
    audio[10:14] = 1.0
    assert level_problem(audio) is None


def test_clip_problem_order():
    assert clip_problem(np.zeros(0, dtype=np.float32), SR) == "too_quiet"
    assert clip_problem(noise(0.5), SR) == "too_short"
    assert clip_problem(noise(MAX_RECORD_SECONDS + 1), SR) == "too_long"
    assert clip_problem(noise(3.0, amp=0.01), SR) == "too_quiet"
    assert clip_problem(noise(3.0), SR) is None


def test_every_problem_has_a_message():
    assert set(PROBLEM_MESSAGES) == {"too_quiet", "too_short", "too_long", "clipped"}
    assert all(len(m) > 20 for m in PROBLEM_MESSAGES.values())


def test_split_separates_utterances_at_pauses():
    audio = np.concatenate([silence(0.5), noise(2.0), silence(0.5), noise(2.0), silence(0.5), noise(2.0), silence(0.5)])
    pieces = split_on_silence(audio, SR)
    assert len(pieces) == 3
    assert all(abs(len(p) / SR - 2.2) < 0.03 for p in pieces)


def test_split_ignores_short_gaps():
    audio = np.concatenate([noise(2.0), silence(0.2), noise(2.0)])
    assert len(split_on_silence(audio, SR)) == 1


def test_split_drops_pieces_under_one_second():
    audio = np.concatenate([noise(0.5), silence(1.0), noise(2.0)])
    pieces = split_on_silence(audio, SR)
    assert len(pieces) == 1 and len(pieces[0]) / SR > 1.9


def test_split_of_silence_is_empty():
    assert split_on_silence(silence(5.0), SR) == []


def test_split_of_empty_input_is_empty():
    assert split_on_silence(np.zeros(0, dtype=np.float32), SR) == []


def test_continuous_audio_is_cut_into_pieces_of_at_most_fifteen_seconds_without_loss():
    audio = noise(40.0)
    pieces = split_on_silence(audio, SR)
    assert len(pieces) >= 3
    assert all(len(p) <= 15 * SR for p in pieces)
    assert np.array_equal(np.concatenate(pieces), audio)


def test_long_utterance_is_cut_at_a_quiet_moment():
    quiet = noise(0.2, amp=0.02)  # above the silence threshold, so it does not end the region
    audio = np.concatenate([noise(13.0), quiet, noise(10.0)])
    pieces = split_on_silence(audio, SR)
    assert len(pieces) == 2
    first_end = len(pieces[0]) / SR
    assert 13.0 <= first_end <= 13.3
