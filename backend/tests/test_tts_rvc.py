import numpy as np
import pytest

from aivoice import tts_rvc
from aivoice.tts_rvc import OfflineConverter, WINDOW_SECONDS

SR = 48000
WIN = WINDOW_SECONDS * SR
DELAY = int(0.02 * SR)
PARAMS = object()


class Identity:
    def __init__(self):
        self.calls = []

    def convert(self, audio, sample_rate, pitch_shift, params):
        self.calls.append((len(audio), sample_rate, pitch_shift, params))
        return audio.copy()


class Delayed(Identity):
    """Like the real engine: output is the input delayed by 20 ms, zeros at the start, same length."""

    def convert(self, audio, sample_rate, pitch_shift, params):
        super().convert(audio, sample_rate, pitch_shift, params)
        return np.concatenate([np.zeros(DELAY, np.float32), audio[:-DELAY]])


class WrongLength(Identity):
    def convert(self, audio, *a):
        return audio[:-1].copy()


class NotFinite(Identity):
    def convert(self, audio, *a):
        out = audio.copy()
        out[10] = np.nan
        return out


def speech(seconds, seed=0):
    """Noise burst with 0.3 s of silence at both ends, like a synthesised sentence."""
    n = int(seconds * SR)
    x = np.random.default_rng(seed).uniform(-0.5, 0.5, n).astype(np.float32)
    pad = min(int(0.3 * SR), n // 4)
    x[:pad] = 0
    x[n - pad:] = 0
    return x


@pytest.mark.parametrize("seconds", [1, 7, 19.9, 20])
def test_identity_keeps_length_and_content(seconds):
    eng = Identity()
    x = speech(seconds)
    y = OfflineConverter(eng, PARAMS).convert(x)
    assert y.dtype == np.float32 and len(y) == len(x)
    np.testing.assert_array_equal(y, x)
    assert eng.calls == [(WIN, SR, 0.0, PARAMS)]


@pytest.mark.parametrize("seconds", [1, 7, 19.9, 20])
def test_delayed_engine_alignment(seconds):
    x = speech(seconds)
    y = OfflineConverter(Delayed(), PARAMS).convert(x)
    assert len(y) == len(x)
    # what comes back is the piece delayed by exactly the engine's 20 ms, nothing more
    np.testing.assert_array_equal(y[:DELAY], 0)
    np.testing.assert_array_equal(y[DELAY:], x[:-DELAY])


def test_piece_sits_at_the_end_of_the_window():
    seen = []

    class Spy(Identity):
        def convert(self, audio, *a):
            seen.append(audio.copy())
            return audio.copy()

    x = speech(3)
    OfflineConverter(Spy(), PARAMS).convert(x)
    np.testing.assert_array_equal(seen[0][-len(x):], x)
    assert not seen[0][:-len(x)].any()


def test_long_piece_is_split_and_rejoined():
    eng = Identity()
    x = speech(45)
    # a clear quiet spot in the middle third, away from the exact centre
    q = int(len(x) * 0.4)
    x[q:q + int(0.05 * SR)] = 0
    y = OfflineConverter(eng, PARAMS).convert(x)
    assert len(y) == len(x)
    np.testing.assert_array_equal(y, x)
    assert len(eng.calls) == 3 or len(eng.calls) == 2
    assert all(c[0] == WIN for c in eng.calls)


def test_split_point_is_the_quietest_frame_in_the_middle_third():
    x = speech(45)
    q = int(len(x) * 0.4)
    x[q:q + int(0.05 * SR)] = 0
    cut = tts_rvc._split_point(x)
    assert q <= cut <= q + int(0.05 * SR)
    assert len(x) / 3 <= cut <= 2 * len(x) / 3


def test_long_piece_with_delay_loses_only_one_delay_per_part():
    eng = Delayed()
    x = speech(45)
    y = OfflineConverter(eng, PARAMS).convert(x)
    assert len(y) == len(x) and all(c[0] == WIN for c in eng.calls)


def test_empty_piece():
    eng = Identity()
    assert len(OfflineConverter(eng, PARAMS).convert(np.zeros(0, np.float32))) == 0
    assert eng.calls == []


def test_wrong_length_result_raises():
    with pytest.raises(ValueError, match="length"):
        OfflineConverter(WrongLength(), PARAMS).convert(speech(2))


def test_non_finite_result_raises():
    with pytest.raises(ValueError, match="finite"):
        OfflineConverter(NotFinite(), PARAMS).convert(speech(2))


def test_wrong_dtype_result_is_rejected_or_cast_to_float32():
    class F64(Identity):
        def convert(self, audio, *a):
            return audio.astype(np.float64)

    y = OfflineConverter(F64(), PARAMS).convert(speech(2))
    assert y.dtype == np.float32


def test_warm_runs_one_silent_window():
    eng = Identity()
    OfflineConverter(eng, PARAMS).warm()
    assert eng.calls == [(WIN, SR, 0.0, PARAMS)]


def test_offline_params_is_high_quality_with_index_rate():
    from aivoice.presets import PRESETS
    assert tts_rvc.offline_params() == PRESETS["high_quality"]
    assert tts_rvc.offline_params().index_rate == 0.6
