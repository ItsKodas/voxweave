import numpy as np
import pytest

from aivoice.presets import LiveParams
from aivoice.processor import EngineFailure, StreamProcessor

SR = 48000
PARAMS = LiveParams(chunk_ms=100, crossfade_ms=20, context_ms=200, f0_method="dio", index_rate=0.0)


class Identity:
    def convert(self, audio, sample_rate, pitch_shift, params):
        return audio.copy()


class Raises:
    def convert(self, audio, sample_rate, pitch_shift, params):
        raise RuntimeError("boom")


class Flaky:
    def __init__(self, fail_on):
        self.calls = 0
        self.fail_on = fail_on

    def convert(self, audio, sample_rate, pitch_shift, params):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("boom")
        return audio.copy()


def run(proc, signal):
    blocks = signal.reshape(-1, proc.chunk)
    return np.concatenate([proc.process(b) for b in blocks])


def noise(chunks, proc):
    return np.random.default_rng(0).standard_normal(chunks * proc.chunk).astype(np.float32)


def test_sizes():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    assert (proc.chunk, proc.crossfade, proc.window) == (4800, 960, 9600 + 4800 + 960)


def test_identity_engine_is_input_delayed_by_crossfade():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    x = noise(10, proc)
    y = run(proc, x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.allclose(y[: proc.crossfade], 0.0)
    assert np.allclose(y[proc.crossfade :], x[: -proc.crossfade], atol=1e-5)


def test_engine_error_emits_silence_never_input():
    proc = StreamProcessor(Raises(), SR, PARAMS, max_failures=100)
    y = run(proc, noise(5, proc) + 1.0)
    assert np.all(y == 0.0)


def test_repeated_failures_raise():
    proc = StreamProcessor(Raises(), SR, PARAMS, max_failures=3)
    block = np.ones(proc.chunk, dtype=np.float32)
    proc.process(block)
    proc.process(block)
    with pytest.raises(EngineFailure):
        proc.process(block)


def test_failure_counter_resets_on_success():
    proc = StreamProcessor(Flaky({1, 2, 4, 5}), SR, PARAMS, max_failures=3)
    block = np.ones(proc.chunk, dtype=np.float32)
    for _ in range(6):
        proc.process(block)


def test_wrong_length_engine_output_is_treated_as_failure():
    class Short:
        def convert(self, audio, sample_rate, pitch_shift, params):
            return audio[:10]

    proc = StreamProcessor(Short(), SR, PARAMS, max_failures=100)
    assert np.all(proc.process(np.ones(proc.chunk, dtype=np.float32)) == 0.0)


def test_nan_engine_output_is_treated_as_failure():
    class Nan:
        def convert(self, audio, sample_rate, pitch_shift, params):
            return np.full_like(audio, np.nan)

    proc = StreamProcessor(Nan(), SR, PARAMS, max_failures=100)
    assert np.all(proc.process(np.ones(proc.chunk, dtype=np.float32)) == 0.0)


def test_muted_emits_silence_without_calling_engine():
    engine = Flaky(set())
    proc = StreamProcessor(engine, SR, PARAMS)
    proc.mode = "muted"
    assert np.all(proc.process(np.ones(proc.chunk, dtype=np.float32)) == 0.0)
    assert engine.calls == 0


def test_unmute_fades_in_from_silence():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    ones = np.ones(proc.chunk, dtype=np.float32)
    proc.process(ones)
    proc.process(ones)
    proc.mode = "muted"
    proc.process(ones)
    proc.mode = "converted"
    y = proc.process(ones)
    assert y[0] == pytest.approx(0.0, abs=1e-3)
    assert y[-1] == pytest.approx(1.0, abs=1e-5)


def test_wrong_block_size_rejected():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    with pytest.raises(ValueError):
        proc.process(np.zeros(proc.chunk + 1, dtype=np.float32))


def test_pitch_shift_and_params_reach_engine():
    seen = {}

    class Spy:
        def convert(self, audio, sample_rate, pitch_shift, params):
            seen.update(sr=sample_rate, pitch=pitch_shift, params=params, n=len(audio))
            return audio

    proc = StreamProcessor(Spy(), SR, PARAMS)
    proc.pitch_shift = 7.0
    proc.process(np.zeros(proc.chunk, dtype=np.float32))
    assert seen == {"sr": SR, "pitch": 7.0, "params": PARAMS, "n": proc.window}
