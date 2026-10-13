import numpy as np
import pytest

from aivoice.engine_guard import DamagedModel, GuardedEngine, check
from aivoice.presets import PRESETS
from aivoice.processor import EngineFailure, StreamProcessor

SR = 48000


def speech_like(seconds=1.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_ordinary_and_loud_speech_and_silence_pass():
    check(speech_like())
    check(speech_like(amp=0.95))  # a shout that peaks near full scale is still far below noise on average
    check(np.zeros(SR, dtype=np.float32))
    check(np.zeros(0, dtype=np.float32))


def test_the_noise_of_a_damaged_model_is_refused():
    # Measured on a real model whose training had turned to NaN: RMS at full scale, peaks near 1.9.
    noise = np.random.default_rng(0).uniform(-1.9, 1.9, SR).astype(np.float32)
    with pytest.raises(DamagedModel, match="train this voice again"):
        check(noise)
    with pytest.raises(DamagedModel):
        check(np.full(SR, 0.9, dtype=np.float32))
    spike = speech_like()
    spike[100] = 3.0
    with pytest.raises(DamagedModel):
        check(spike)


def test_nan_and_infinity_do_not_get_past_or_crash_the_check():
    bad = speech_like()
    bad[:10] = np.nan
    check(bad)  # the processor has its own refusal for non-finite output
    bad[:10] = np.inf
    check(bad)


class Noisy:
    loaded = "yes"

    def convert(self, audio, sample_rate, pitch_shift, params):
        return np.random.default_rng(1).uniform(-1.9, 1.9, len(audio)).astype(np.float32)


class Fine:
    def convert(self, audio, sample_rate, pitch_shift, params):
        return np.asarray(audio, dtype=np.float32) * 0.5


def test_a_guarded_engine_passes_good_output_through_and_other_attributes_too():
    audio = speech_like()
    assert np.array_equal(GuardedEngine(Fine()).convert(audio, SR, 0.0, PRESETS["balanced"]), audio * 0.5)
    assert GuardedEngine(Noisy()).loaded == "yes"


def test_live_mode_plays_silence_for_a_damaged_model_and_then_stops_with_the_reason():
    params = PRESETS["balanced"]
    processor = StreamProcessor(GuardedEngine(Noisy()), SR, params)
    block = speech_like(params.chunk_ms / 1000)[: processor.chunk]
    for _ in range(4):
        assert not processor.process(block).any()
    with pytest.raises(EngineFailure, match="loud noise instead of a voice"):
        processor.process(block)


def test_the_engine_live_mode_and_text_to_speech_build_is_guarded(tmp_path, monkeypatch):
    import aivoice.engines.rvc as rvc
    from aivoice.manager import default_engine_factory
    from aivoice.pretrained import PretrainedVoice

    monkeypatch.setattr(rvc, "RvcEngine", lambda *a, **k: Noisy())
    engine = default_engine_factory(tmp_path)(PretrainedVoice("v", "V", tmp_path / "m.onnx", None, 40000))
    assert isinstance(engine, GuardedEngine)
    with pytest.raises(DamagedModel):
        engine.convert(speech_like(), SR, 0.0, PRESETS["balanced"])
