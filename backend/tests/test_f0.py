import numpy as np
import pytest

from aivoice.engines.f0 import extract_f0, shift_f0, to_coarse


def sine(freq, seconds=1.0, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def voice_like(freq, seconds=1.0, sr=16000):
    """A voice-like tone: a fundamental with decaying harmonics and a slight 5 Hz vibrato.

    Harvest reports a bare sine as unvoiced, which real speech never is.
    """
    t = np.arange(int(sr * seconds)) / sr
    phase = 2 * np.pi * freq * t + 0.5 * np.sin(2 * np.pi * 5 * t)
    x = sum(0.6**k * np.sin((k + 1) * phase) for k in range(6))
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


@pytest.mark.parametrize("method", ["dio", "harvest"])
def test_detects_220hz(method):
    f0 = extract_f0(voice_like(220), method, 100)
    assert f0.shape == (100,) and f0.dtype == np.float32
    voiced = f0[f0 > 0]
    assert len(voiced) > 50
    assert abs(np.median(voiced) - 220) < 5


@pytest.mark.parametrize("method", ["dio", "harvest"])
def test_silence_is_unvoiced_and_finite(method):
    f0 = extract_f0(np.zeros(16000, dtype=np.float32), method, 100)
    assert f0.shape == (100,)
    assert np.all(np.isfinite(f0)) and np.all(f0 == 0)


def test_frame_count_is_respected():
    assert extract_f0(sine(220), "dio", 37).shape == (37,)


def test_unknown_method():
    with pytest.raises(ValueError):
        extract_f0(sine(220), "magic", 10)


def test_shift_octave_doubles_and_keeps_unvoiced():
    f0 = np.array([0.0, 110.0, 220.0], dtype=np.float32)
    assert shift_f0(f0, 12).tolist() == pytest.approx([0.0, 220.0, 440.0])


def test_coarse_range():
    coarse = to_coarse(np.array([0.0, 50.0, 220.0, 1100.0, 5000.0], dtype=np.float32))
    assert coarse.dtype == np.int64
    assert coarse[0] == 1 and coarse[1] == 1 and coarse[3] == 255 and coarse[4] == 255
    assert 1 < coarse[2] < 255
