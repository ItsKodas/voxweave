import numpy as np
import pytest

from aivoice.engines.rvc import RvcEngine, blend_index
from aivoice.presets import LiveParams

SR = 48000
PARAMS = LiveParams(100, 20, 200, "dio", 0.0)


class In:
    def __init__(self, name):
        self.name = name


class FakeVec:
    def get_inputs(self):
        return [In("source")]

    def run(self, _, feed):
        frames = feed["source"].shape[-1] // 320
        return [np.zeros((1, frames, 768), dtype=np.float32)]


class FakeNet:
    def __init__(self, model_sr):
        self.model_sr = model_sr
        self.feeds = []

    def get_inputs(self):
        return [In(n) for n in ("phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd")]

    def run(self, _, feed):
        self.feeds.append(feed)
        frames = feed["phone"].shape[1]
        return [np.full((1, 1, frames * self.model_sr // 100), 0.25, dtype=np.float32)]


def make(model_sr=40000):
    net = FakeNet(model_sr)
    engine = RvcEngine(
        "model.onnx", "vec.onnx", model_sr,
        session_factory=lambda path: FakeVec() if "vec" in path else net,
    )
    return engine, net


def sine(freq, n=SR):
    return (0.5 * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def test_output_matches_input_length_and_dtype():
    engine, _ = make()
    for n in (48000, 40321, 26400):
        out = engine.convert(sine(220, n), SR, 0.0, PARAMS)
        assert out.shape == (n,) and out.dtype == np.float32


def test_48k_model_also_fits_length():
    engine, _ = make(model_sr=48000)
    assert engine.convert(sine(220), SR, 0.0, PARAMS).shape == (48000,)


def test_feed_shapes_and_types():
    engine, net = make()
    engine.convert(sine(220), SR, 0.0, PARAMS)
    feed = net.feeds[0]
    frames = feed["phone"].shape[1]
    assert feed["phone"].shape == (1, frames, 768) and feed["phone"].dtype == np.float32
    assert frames == 2 * (16000 // 320)
    assert feed["phone_lengths"].tolist() == [frames] and feed["phone_lengths"].dtype == np.int64
    assert feed["pitch"].shape == (1, frames) and feed["pitch"].dtype == np.int64
    assert feed["pitchf"].shape == (1, frames) and feed["pitchf"].dtype == np.float32
    assert feed["ds"].tolist() == [0]
    assert feed["rnd"].shape == (1, 192, frames)


def test_pitch_shift_of_an_octave_doubles_pitch():
    engine, net = make()
    engine.convert(sine(220), SR, 0.0, PARAMS)
    engine.convert(sine(220), SR, 12.0, PARAMS)
    base, shifted = (f["pitchf"][0] for f in net.feeds)
    assert np.median(shifted[shifted > 0]) / np.median(base[base > 0]) == pytest.approx(2.0, rel=1e-4)


def test_silent_input_gives_finite_output():
    engine, net = make()
    out = engine.convert(np.zeros(SR, dtype=np.float32), SR, 0.0, PARAMS)
    assert np.all(np.isfinite(out))
    assert np.all(net.feeds[0]["pitch"] == 1)


def test_no_index_ignores_index_rate():
    engine, _ = make()
    params = LiveParams(100, 20, 200, "dio", 0.8)
    assert engine.convert(sine(220), SR, 0.0, params).shape == (SR,)


class FakeIndex:
    def search(self, x, k):
        return np.ones((len(x), k), dtype=np.float32), np.zeros((len(x), k), dtype=np.int64)


def test_blend_index_rate_zero_is_unchanged():
    feats = np.random.default_rng(0).standard_normal((5, 768)).astype(np.float32)
    big = np.ones((3, 768), dtype=np.float32)
    assert np.array_equal(blend_index(feats, FakeIndex(), big, 0.0), feats)


def test_blend_index_rate_one_is_neighbours():
    feats = np.zeros((5, 768), dtype=np.float32)
    big = np.full((3, 768), 2.0, dtype=np.float32)
    assert np.allclose(blend_index(feats, FakeIndex(), big, 1.0), 2.0)


def test_blend_index_handles_zero_distance():
    class Exact(FakeIndex):
        def search(self, x, k):
            return np.zeros((len(x), k), dtype=np.float32), np.zeros((len(x), k), dtype=np.int64)

    out = blend_index(np.zeros((2, 768), dtype=np.float32), Exact(), np.ones((1, 768), dtype=np.float32), 0.5)
    assert np.all(np.isfinite(out))
