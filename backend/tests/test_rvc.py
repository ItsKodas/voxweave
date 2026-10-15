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


def test_make_session_refuses_cpu_fallback(monkeypatch):
    import onnxruntime as ort

    from aivoice.engines.rvc import make_session

    class CpuOnly:
        def __init__(self, *args, **kwargs):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", CpuOnly)
    with pytest.raises(RuntimeError, match="DirectML is not available"):
        make_session("model.onnx")


def _rms(x):
    return float(np.sqrt(np.mean(x * x)))


def test_output_loudness_follows_the_input():
    # The model's own loudness can dip where the speaker's does not; the output follows the input's envelope.
    engine, _ = make()
    audio = np.concatenate([sine(220, SR // 2), 0.2 * sine(220, SR // 2)])
    out = engine.convert(audio, SR, 0.0, PARAMS)
    loud, quiet = out[SR // 8: 3 * SR // 8], out[5 * SR // 8: 7 * SR // 8]
    expected = (0.2 ** PARAMS.volume_follow)
    assert _rms(quiet) / _rms(loud) == pytest.approx(expected, rel=0.1)


def test_volume_follow_zero_keeps_the_models_loudness():
    from dataclasses import replace

    engine, _ = make()
    audio = np.concatenate([sine(220, SR // 2), 0.2 * sine(220, SR // 2)])
    out = engine.convert(audio, SR, 0.0, replace(PARAMS, volume_follow=0.0))
    assert np.allclose(out[SR // 10: -SR // 10], 0.25, atol=1e-3)  # resampling rings only at the edges


def test_volume_follow_never_lifts_a_near_silent_output_far():
    from aivoice.engines.rvc import follow_envelope

    out = np.full(SR, 1e-4, dtype=np.float32)
    lifted = follow_envelope(out, sine(220), SR, 1.0)
    assert np.max(np.abs(lifted)) <= 1e-4 * 4.0 + 1e-7


def test_live_windows_carry_their_pitch_into_the_next_join():
    engine, net = make()
    chunk_ms, crossfade_ms, context_ms = PARAMS.chunk_ms, PARAMS.crossfade_ms, PARAMS.context_ms
    n = SR * (chunk_ms + crossfade_ms + context_ms) // 1000
    step = SR * chunk_ms // 1000
    engine.noise_position = 0
    engine.convert(sine(200, n), SR, 0.0, PARAMS)
    engine.noise_position = step
    engine.convert(sine(240, n), SR, 0.0, PARAMS)
    first, second = net.feeds[0]["pitchf"][0], net.feeds[1]["pitchf"][0]
    head = len(second) - chunk_ms // 10 - round((crossfade_ms + 12) / 10)  # 100 frames a second
    # where this window's played audio begins, it starts from the pitch the last window used for that moment
    assert second[head] == pytest.approx(first[head + chunk_ms // 10], rel=0.03)
    assert second[-1] == pytest.approx(240.0, rel=0.03)


def _quiet_after_speech(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    speech = sine(220, SR // 2)
    pause = (0.002 * rng.standard_normal(SR // 2)).astype(np.float32)  # room sound: no pitch, far below speech
    return np.concatenate([speech, pause])


def test_background_sound_in_a_pause_is_cut():
    from dataclasses import replace

    engine, _ = make()
    engine.speech_level_db = -9.0  # a 0.5 sine
    params = replace(PARAMS, volume_follow=0.0)
    out = engine.convert(_quiet_after_speech(), SR, 0.0, params)
    speech, pause = out[SR // 8: 3 * SR // 8], out[3 * SR // 4:]
    assert 20 * np.log10(_rms(pause) / _rms(speech)) < -25


def test_the_end_of_a_word_just_after_speech_is_kept():
    from dataclasses import replace

    engine, _ = make()
    engine.speech_level_db = -9.0
    out = engine.convert(_quiet_after_speech(), SR, 0.0, replace(PARAMS, volume_follow=0.0))
    just_after = out[SR // 2 + SR // 50: SR // 2 + SR // 10]  # 20 to 100 ms after the speech stops
    assert _rms(just_after) > 0.2


def test_background_cut_can_be_switched_off():
    from dataclasses import replace

    engine, _ = make()
    engine.speech_level_db = -9.0
    out = engine.convert(_quiet_after_speech(), SR, 0.0, replace(PARAMS, volume_follow=0.0, background_cut_db=0.0))
    assert np.allclose(out[SR // 10: -SR // 10], 0.25, atol=1e-3)


def test_each_conversion_leaves_what_it_did_for_the_newest_chunk():
    engine, _ = make()
    engine.speech_level_db = -9.0
    engine.convert(sine(220, SR // 2), SR, 12.0, PARAMS)
    insight = engine.insight
    frames = PARAMS.chunk_ms // 10
    assert insight["frames_per_s"] == pytest.approx(100, rel=0.05)
    assert len(insight["heard"]) == len(insight["sent"]) == len(insight["corrected"]) == len(insight["gated"]) == frames
    assert insight["number"] == 1
    voiced = [(h, s) for h, s in zip(insight["heard"], insight["sent"]) if h > 0 and s > 0]
    assert voiced and all(s / h == pytest.approx(2.0, rel=0.05) for h, s in voiced)  # an octave up, as asked
    assert not any(insight["gated"])
    pause = np.concatenate([sine(220, SR // 4), np.zeros(SR // 2, np.float32)])  # speech, then a long pause
    engine.convert(pause, SR, 12.0, PARAMS)
    assert engine.insight["number"] == 2 and all(engine.insight["gated"])  # background: turned down


def test_each_extra_can_be_switched_off(monkeypatch):
    from dataclasses import replace

    import aivoice.engines.rvc as rvc

    called = []
    monkeypatch.setattr(rvc, "repair_creak", lambda f0, v: called.append("creak") or f0)
    monkeypatch.setattr(rvc, "fix_octave_errors", lambda f0: called.append("octave") or f0)
    monkeypatch.setattr(rvc, "smooth_pitch", lambda f0: called.append("smooth") or f0)
    monkeypatch.setattr(rvc, "natural_variation", lambda f0, start, amount_st: called.append(("waver", amount_st)) or f0)
    engine, _ = make()
    engine.convert(sine(220), SR, 0.0, PARAMS)
    assert called == ["creak", "octave", "smooth", ("waver", 0.25)]
    called.clear()
    off = replace(PARAMS, creak_repair=False, pitch_cleanup=False, natural_variation_st=0.0)
    engine.convert(sine(220), SR, 0.0, off)
    assert called == [("waver", 0.0)]


def test_join_smoothing_can_be_switched_off():
    from dataclasses import replace

    engine, net = make()
    params = replace(PARAMS, join_smoothing=False)
    n = SR * (PARAMS.chunk_ms + PARAMS.crossfade_ms + PARAMS.context_ms) // 1000
    engine.noise_position = 0
    engine.convert(sine(200, n), SR, 0.0, params)
    engine.noise_position = SR * PARAMS.chunk_ms // 1000
    engine.convert(sine(240, n), SR, 0.0, params)
    second = net.feeds[1]["pitchf"][0]
    head = len(second) - PARAMS.chunk_ms // 10 - round((PARAMS.crossfade_ms + 12) / 10)
    assert second[head] == pytest.approx(240.0, rel=0.03)  # its own pitch, not the last window's
