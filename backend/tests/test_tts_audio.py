import numpy as np
import pytest

from aivoice import tts_audio
from aivoice.tts_audio import SAMPLE_RATE, finish, is_silent, join, max_chars_for, trim_edges

SR = SAMPLE_RATE


def tone(seconds, amp=0.3, sr=SR, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds, amp, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)


# ---- trim_edges ----

def test_trim_removes_room_tone_before_speech():
    room = noise(1.5, 0.003)  # about -50 dBFS
    speech = tone(1.0, 0.3)
    tail = noise(0.2, 0.003, seed=1)
    out = trim_edges(np.concatenate([room, speech, tail]), SR)
    slack = 2 * (0.030 + 0.020)  # keep_ms on both sides plus up to one frame of slack
    assert 1.0 <= len(out) / SR <= 1.0 + slack


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
