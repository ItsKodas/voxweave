import numpy as np
import pytest

from aivoice import tts_audio
from aivoice.tts_audio import (PIECE_PEAK, SAMPLE_RATE, drop_reference_tail, finish, is_silent, join,
                                max_chars_for, trim_edges)

SR = SAMPLE_RATE


def tone(seconds, amp=0.3, sr=SR, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds, amp, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)


# ---- drop_reference_tail ----
# The real model's piece begins where the reference clip ended. Measured on 282 pieces in the first real run of
# the manager: about a second of the reference's room tone comes before the sentence, and in a third of the
# pieces that second begins with sound, 60 to 300 ms long: the dying end of the reference's last word (12 to 30 dB
# below the piece's loudest frame) or, at 8 steps, a stray word as loud as the sentence or louder.

def piece_with(burst_seconds, burst_amp, gap_seconds=1.0, room=0.001, sentence=0.3):
    return np.concatenate([tone(burst_seconds, burst_amp, freq=330.0), noise(gap_seconds, room),
                           tone(1.0, sentence), noise(0.1, room, seed=1)])


@pytest.mark.parametrize("burst_seconds, burst_amp, gap_seconds, room", [
    (0.12, 0.075, 1.0, 0.001),  # the end of the reference's last word, 12 dB down
    (0.30, 0.3, 1.0, 0.001),  # a stray word as loud as the sentence
    (0.28, 0.3, 0.42, 0.001),  # and the shortest pause seen after one
    (0.30, 0.9, 1.0, 0.012),  # a stray word louder than the sentence, before room tone 25 dB below the sentence
])
def test_sound_at_the_very_start_of_a_piece_and_the_pause_after_it_are_dropped(burst_seconds, burst_amp,
                                                                               gap_seconds, room):
    piece = piece_with(burst_seconds, burst_amp, gap_seconds, room)
    assert len(trim_edges(piece, SR)) / SR > 1.0 + gap_seconds  # trimming alone keeps the sound and the pause
    out = drop_reference_tail(piece, SR)
    assert out.dtype == np.float32
    cut = len(piece) - len(out)
    sentence = round((burst_seconds + gap_seconds) * SR)
    assert sentence - 0.05 * SR <= cut <= sentence  # cut just before the sentence
    assert np.array_equal(out, piece[cut:])
    assert 1.0 <= len(trim_edges(out, SR)) / SR <= 1.2


def test_single_loud_frames_in_the_pause_do_not_hide_it():
    piece = piece_with(0.12, 0.075)
    at = int(0.6 * SR)
    piece[at:at + SR // 50] = tone(0.02, 0.012)  # one 20 ms frame of room tone 28 dB below the sentence
    assert 1.0 <= len(trim_edges(drop_reference_tail(piece, SR), SR)) / SR <= 1.2


def test_a_piece_with_a_quiet_lead_in_is_left_as_it_is():
    piece = np.concatenate([noise(1.0, 0.001), tone(1.0, 0.3)])
    assert np.array_equal(drop_reference_tail(piece, SR), piece)


@pytest.mark.parametrize("word, pause", [
    (0.5, 1.0),  # a first word longer than any stray sound seen, then a long pause
    (0.1, 0.3),  # a short first word and an ordinary pause
    (0.1, 0.0),  # speech from the first sample on
])
def test_a_first_word_is_not_taken_for_a_stray_sound(word, pause):
    piece = np.concatenate([tone(word, 0.2), noise(pause, 0.001), tone(1.0, 0.3)])
    assert np.array_equal(drop_reference_tail(piece, SR), piece)


def test_sound_that_starts_later_than_the_very_beginning_is_not_touched():
    piece = np.concatenate([noise(0.2, 0.001), tone(0.1, 0.1), noise(1.0, 0.001, seed=1), tone(1.0, 0.3)])
    assert np.array_equal(drop_reference_tail(piece, SR), piece)


def test_sound_with_nothing_after_it_is_all_there_is_and_is_kept():
    piece = np.concatenate([tone(0.1, 0.3), noise(1.0, 0.001)])
    assert np.array_equal(drop_reference_tail(piece, SR), piece)


def test_drop_reference_tail_copes_with_nothing_quiet_and_broken_audio():
    assert len(drop_reference_tail(np.zeros(0, np.float32), SR)) == 0
    assert len(drop_reference_tail(np.zeros(SR), SR)) == SR
    assert len(drop_reference_tail(tone(0.01), SR)) == int(0.01 * SR)
    broken = piece_with(0.1, 0.05)
    broken[10] = np.nan
    out = drop_reference_tail(broken, SR)
    assert np.isfinite(out).all() and len(out) < len(broken)


def test_drop_reference_tail_works_at_the_workers_sample_rate():
    sr = 24000
    piece = np.concatenate([tone(0.1, 0.05, sr=sr), noise(1.0, 0.001, sr=sr), tone(1.0, 0.3, sr=sr)])
    out = drop_reference_tail(piece, sr)
    assert 1.05 * sr <= len(piece) - len(out) <= 1.1 * sr


# ---- the limits of the trimming, pinned ----
# Both functions judge a piece by its shape alone, so real speech of the same shape as a lead-in is cut. These
# tests state the limits; they are not wishes. They were not hardened (final fix wave, D5): the measured lead-ins
# are a sound 12 to 30 dB below the loudest frame followed by 0.4 to 1.2 s of room tone that is itself only 30 to
# 45 dB down (the spike saw up to 2 s), so neither "the gap must be much quieter than the sound" nor "at most
# 1.5 s may be dropped" can be added without leaving the reference's tail in some real pieces.

def test_limit_a_short_first_word_followed_by_a_long_pause_is_cut():
    # "Well." said in a quarter of a second from the first sample on, a pause of 0.45 s, then the sentence
    piece = np.concatenate([tone(0.25, 0.3), noise(0.45, 0.001), tone(1.0, 0.3)])
    out = drop_reference_tail(piece, SR)
    assert 0.65 * SR <= len(piece) - len(out) <= 0.70 * SR  # the word and the pause are gone
    assert 1.0 <= len(out) / SR <= 1.05


@pytest.mark.parametrize("word, pause", [
    (0.25, 0.36),  # the same word before a pause shorter than 0.4 s
    (0.44, 0.45),  # a word longer than 0.4 s before the same pause
])
def test_limit_the_same_word_is_kept_when_the_pause_is_shorter_or_the_word_longer(word, pause):
    piece = np.concatenate([tone(word, 0.3), noise(pause, 0.001), tone(1.0, 0.3)])
    assert np.array_equal(drop_reference_tail(piece, SR), piece)


def test_limit_a_long_pause_after_a_short_first_word_does_not_save_the_word():
    # nothing limits how much is dropped: the pause before the sentence was up to 2 s in the spike
    piece = np.concatenate([tone(0.25, 0.3), noise(3.0, 0.001), tone(1.0, 0.3)])
    assert 1.0 <= len(drop_reference_tail(piece, SR)) / SR <= 1.05


def test_limit_a_first_sound_under_60_ms_followed_by_a_gap_is_dropped():
    # a click of a consonant, 40 ms, a tenth of a second before the rest: trim_edges starts where sound lasts
    piece = np.concatenate([noise(0.5, 0.0005), tone(0.04, 0.3), noise(0.1, 0.0005, seed=1), tone(1.0, 0.3)])
    out = trim_edges(piece, SR)
    assert 1.0 <= len(out) / SR <= 1.1
    kept_from = len(piece) - len(out)
    assert kept_from > int(0.54 * SR)  # the 40 ms sound lies before the cut


def test_limit_a_first_sound_of_60_ms_is_kept():
    piece = np.concatenate([noise(0.5, 0.0005), tone(0.06, 0.3), noise(0.1, 0.0005, seed=1), tone(1.0, 0.3)])
    assert 1.16 <= len(trim_edges(piece, SR)) / SR <= 1.26


# ---- trim_edges ----

def test_trim_removes_room_tone_before_speech():
    room = noise(1.5, 0.003)  # about -50 dBFS
    speech = tone(1.0, 0.3)
    tail = noise(0.2, 0.003, seed=1)
    out = trim_edges(np.concatenate([room, speech, tail]), SR)
    slack = 2 * (0.030 + 0.020)  # keep_ms on both sides plus up to one frame of slack
    assert 1.0 <= len(out) / SR <= 1.0 + slack


def test_trim_does_not_start_at_a_single_loud_frame_of_room_tone():
    # the real model's lead-in is room tone only about 30 dB below the speech: single frames reach the threshold
    room = noise(1.0, 0.001)
    for at in (0.1, 0.5):
        room[int(at * SR):int(at * SR) + SR // 50] = tone(0.02, 0.012)  # 28 dB below the speech
    out = trim_edges(np.concatenate([room, tone(1.0, 0.3), noise(0.2, 0.001, seed=1)]), SR)
    assert 1.0 <= len(out) / SR <= 1.1


def test_trim_keeps_a_short_last_sound():
    # the release of a final consonant: 20 ms, a tenth of a second after the rest
    x = np.concatenate([np.zeros(SR // 2), tone(1.0, 0.3), np.zeros(SR // 10), tone(0.02, 0.1), np.zeros(SR // 2)])
    out = trim_edges(x, SR)
    assert 1.12 <= len(out) / SR <= 1.2


def test_trim_starts_at_the_first_loud_frame_when_nothing_lasts():
    out = trim_edges(np.concatenate([np.zeros(SR // 2), tone(0.04, 0.3), np.zeros(SR // 2)]), SR)
    assert 0.04 <= len(out) / SR <= 0.12


def test_trim_keeps_quiet_speech_relative_to_loudest_frame():
    # a whole quiet piece at about -37 dBFS is still speech: trimming is relative, not absolute
    speech = tone(1.0, 0.02)
    out = trim_edges(np.concatenate([np.zeros(SR), speech, np.zeros(SR)]), SR)
    assert 1.0 <= len(out) / SR < 1.15


def test_trim_keeps_quieter_internal_words():
    loud, soft = tone(0.5, 0.3), tone(0.5, 0.02)  # soft is 23 dB below loud: inside 30 dB
    x = np.concatenate([np.zeros(SR // 2), loud, np.zeros(SR // 4), soft, np.zeros(SR // 2)])
    out = trim_edges(x, SR)
    assert len(out) / SR >= 1.25


def test_trim_all_quiet_returns_empty():
    assert len(trim_edges(np.zeros(SR), SR)) == 0
    assert len(trim_edges(noise(1.0, 1e-5), SR)) == 0
    assert len(trim_edges(np.zeros(0, np.float32), SR)) == 0


def test_trim_nan_and_inf():
    x = np.concatenate([np.zeros(SR // 2), tone(0.5), np.zeros(SR // 2)])
    x[100] = np.nan
    x[SR // 2 + 50] = np.inf
    out = trim_edges(x, SR)
    assert np.isfinite(out).all()
    assert len(out) > 0
    assert len(trim_edges(np.full(SR, np.nan, np.float32), SR)) == 0


def test_trim_shorter_than_one_frame():
    out = trim_edges(tone(0.01, 0.3), SR)
    assert 0 < len(out) <= int(0.01 * SR)


def test_trim_returns_a_copy():
    x = tone(0.3, 0.3)
    out = trim_edges(x, SR)
    assert out.dtype == np.float32 and len(out) <= len(x)
    out[:] = 0
    assert np.abs(x).max() > 0.2


# ---- finish ----

def test_finish_scales_down_loud_peak_only():
    x = tone(0.5, 1.37)
    out = finish(x, SR)
    assert out.dtype == np.float32
    assert abs(np.abs(out).max() - 0.97) < 1e-3


def test_finish_leaves_quiet_piece_unscaled():
    x = tone(0.5, 0.2)
    out = finish(x, SR)
    mid = len(x) // 2
    assert np.allclose(out[mid - 100 : mid + 100], x[mid - 100 : mid + 100])
    assert abs(np.abs(out).max() - 0.2) < 1e-3


def test_finish_resamples_24k_to_48k():
    x = tone(1.0, 0.3, sr=24000)
    out = finish(x, 24000)
    assert abs(len(out) - 48000) <= 2
    mid = out[10000:38000]
    assert 0.28 < np.abs(mid).max() < 0.32
    crossings = np.count_nonzero(np.diff(np.signbit(out)))  # 220 Hz for 1 s
    assert abs(crossings - 440) <= 4


def test_finish_fades_both_edges():
    x = np.full(SR, 0.5, np.float32)
    out = finish(x, SR, fade_ms=10)
    n = SR * 10 // 1000
    assert out[0] == 0 and out[-1] == 0
    assert np.all(np.diff(out[:n]) >= 0) and np.all(np.diff(out[-n:]) <= 0)
    assert out[n + 5] == pytest.approx(0.5)
    assert out[-n - 5] == pytest.approx(0.5)


def test_finish_short_input_fade_does_not_overlap():
    out = finish(np.full(100, 0.5, np.float32), SR, fade_ms=10)
    assert len(out) == 100
    assert np.isfinite(out).all() and out.max() <= 0.5


def test_finish_cleans_nan_and_inf():
    x = tone(0.2, 0.3)
    x[50] = np.nan
    x[60] = np.inf
    x[70] = -np.inf
    out = finish(x, SR)
    assert np.isfinite(out).all() and np.abs(out).max() <= 0.97 + 1e-6


def test_finish_empty_and_input_untouched():
    assert len(finish(np.zeros(0, np.float32), SR)) == 0
    x = tone(0.2, 1.3)
    before = x.copy()
    finish(x, SR)
    assert np.array_equal(x, before)


# ---- finish to one level (what the manager asks for) ----

def test_two_pieces_of_different_raw_peak_are_finished_at_the_same_level():
    # the worker scales a piece only above 1.0, so neighbours arrive anywhere between about 0.7 and 1.0
    quiet, loud = finish(tone(0.5, 0.7), SR, level=PIECE_PEAK), finish(tone(0.5, 1.0), SR, level=PIECE_PEAK)
    assert np.abs(quiet).max() == pytest.approx(PIECE_PEAK, abs=1e-3)
    assert np.abs(loud).max() == pytest.approx(PIECE_PEAK, abs=1e-3)
    assert PIECE_PEAK == 0.89


def test_a_piece_from_a_quiet_reference_is_raised_to_the_same_level():
    # the worker lowers what it generates by as much as it raised a quiet reference: up to 30 dB
    out = finish(tone(0.5, 0.03, sr=24000), 24000, level=PIECE_PEAK)
    assert np.abs(out).max() == pytest.approx(PIECE_PEAK, abs=2e-3)


def test_the_level_is_set_after_resampling_and_before_the_fade():
    out = finish(np.full(SR, 0.5, np.float32), SR, level=PIECE_PEAK)
    n = SR * 10 // 1000
    assert out[0] == 0 and out[-1] == 0
    assert out[n + 5] == pytest.approx(PIECE_PEAK) and out.max() == pytest.approx(PIECE_PEAK)


def test_what_is_silent_is_never_raised():
    # below the peak is_silent calls nothing: raising it would play the noise floor at full level
    x = tone(0.5, 0.004)
    out = finish(x, SR, level=PIECE_PEAK)
    assert np.abs(out).max() == pytest.approx(0.004, abs=1e-4)
    assert not np.abs(finish(np.zeros(SR, np.float32), SR, level=PIECE_PEAK)).any()
    assert is_silent(x, SR)  # the two rules use the same floor


def test_the_level_leaves_the_input_untouched_and_copes_with_broken_audio():
    x = tone(0.2, 0.3)
    x[50] = np.nan
    before = x.copy()
    out = finish(x, SR, level=PIECE_PEAK)
    assert np.array_equal(x, before, equal_nan=True)
    assert np.isfinite(out).all() and np.abs(out).max() == pytest.approx(PIECE_PEAK, abs=1e-3)
    assert len(finish(np.zeros(0, np.float32), SR, level=PIECE_PEAK)) == 0


# ---- is_silent ----

def test_is_silent():
    assert is_silent(np.zeros(0, np.float32), SR)
    assert is_silent(np.zeros(SR), SR)
    assert is_silent(tone(0.14, 0.5), SR)  # too short
    assert not is_silent(tone(0.16, 0.5), SR)
    assert is_silent(tone(1.0, 0.004), SR)  # peak below 0.005
    assert not is_silent(tone(1.0, 0.006), SR)
    assert is_silent(np.full(SR, np.nan, np.float32), SR)
    assert is_silent(tone(0.07, 0.5, sr=24000), 24000)  # the limit is in seconds
    assert not is_silent(tone(0.2, 0.5, sr=24000), 24000)


# ---- join ----

def test_join_lengths_and_pauses():
    a, b, c = tone(0.5), tone(0.25), tone(0.1)
    out = join([a, b, c], pause_ms=300)
    pause = int(0.3 * SR)
    assert len(out) == len(a) + len(b) + len(c) + 2 * pause
    assert np.array_equal(out[len(a) : len(a) + pause], np.zeros(pause, np.float32))
    assert out.dtype == np.float32


def test_join_edge_cases():
    assert len(join([])) == 0
    a = tone(0.5)
    assert np.array_equal(join([a]), a)
    assert len(join([a, np.zeros(0, np.float32), a], pause_ms=100)) == 2 * len(a) + int(0.1 * SR)
    assert len(join([a, a], pause_ms=0)) == 2 * len(a)
    assert len(join([tone(0.1, sr=24000), tone(0.1, sr=24000)], pause_ms=500, sr=24000)) == 2400 * 2 + 12000


# ---- max_chars_for ----

# The shipped reference clip: 51 bytes of text (with the trailing space the package adds) over about 4.89 s
# that the package sees after its own preprocessing gives 178 before our safety margin.
REF_TEXT = "Some call me nature, others call me mother nature. "


def package_formula(text, secs, speed=1.0):
    return int(len(text.encode("utf-8")) / secs * (22 - secs) * speed)


def test_matches_package_formula_before_margin():
    assert package_formula(REF_TEXT, 4.89) == 178
    assert max_chars_for(REF_TEXT, 4.89) == int(len(REF_TEXT.encode()) / 4.89 * (22 - 4.89) * 0.85) == 151


@pytest.mark.parametrize("secs", [3.0, 5.33, 8.0, 10.0, 12.0])
@pytest.mark.parametrize("speed", [0.8, 1.0, 1.25])
def test_margin_applied_to_package_formula(secs, speed):
    raw = len(REF_TEXT.encode()) / secs * (22 - secs) * speed
    assert max_chars_for(REF_TEXT, secs, speed) == int(raw * 0.85)


def test_monotonic():
    values = [max_chars_for(REF_TEXT, s) for s in (3, 4, 5, 6, 8, 10, 12)]
    assert values == sorted(values, reverse=True)
    speeds = [max_chars_for(REF_TEXT, 5.0, speed=s) for s in (0.5, 0.8, 1.0, 1.2, 1.5)]
    assert speeds == sorted(speeds)
    longer_text = [max_chars_for("x" * n, 5.0) for n in (30, 60, 120)]
    assert longer_text == sorted(longer_text)


def test_generated_audio_fits_under_limit():
    for secs in (3.0, 8.0, 12.0):
        n = max_chars_for(REF_TEXT, secs)
        generated = n / (len(REF_TEXT.encode()) / secs)  # the model's duration for n bytes at speed 1
        assert secs + generated < 30.0 * 0.9


def test_limit_seconds_caps_budget():
    default = max_chars_for(REF_TEXT, 5.0)
    assert max_chars_for(REF_TEXT, 5.0, limit_seconds=30.0) == default
    assert max_chars_for(REF_TEXT, 5.0, limit_seconds=15.0) < default


def test_reference_too_long_is_a_clear_error():
    with pytest.raises(ValueError):
        max_chars_for(REF_TEXT, 30.0)  # the package formula would give a negative number
    with pytest.raises(ValueError):
        max_chars_for(REF_TEXT, 0.0)
    with pytest.raises(ValueError):
        max_chars_for(REF_TEXT, 5.0, speed=0.0)
    with pytest.raises(ValueError):
        max_chars_for("", 5.0)


def test_module_constant():
    assert tts_audio.SAMPLE_RATE == 48000
