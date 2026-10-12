import time

import numpy as np

from aivoice.engines.base import VoiceEngine
from aivoice.presets import LiveParams


class EngineFailure(RuntimeError):
    pass


class StreamProcessor:
    def __init__(self, engine: VoiceEngine, sample_rate: int, params: LiveParams, max_failures: int = 5):
        self.engine = engine
        self.sample_rate = sample_rate
        self.params = params
        self.chunk = sample_rate * params.chunk_ms // 1000
        self.crossfade = sample_rate * params.crossfade_ms // 1000
        self.window = sample_rate * params.context_ms // 1000 + self.chunk + self.crossfade
        self.mode = "converted"
        self.pitch_shift = 0.0
        self.infer_ms = 0.0
        self.slow_chunks = 0
        self._max_failures = max_failures
        self._failures = 0
        self._input = np.zeros(self.window, dtype=np.float32)
        self._tail = np.zeros(self.crossfade, dtype=np.float32)
        t = np.linspace(0.0, 1.0, self.crossfade, endpoint=False, dtype=np.float32)
        self._fade_in = np.sin(0.5 * np.pi * t) ** 2
        self._fade_out = 1.0 - self._fade_in

    def _silence(self) -> np.ndarray:
        self._tail[:] = 0.0
        return np.zeros(self.chunk, dtype=np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        if len(block) != self.chunk:
            raise ValueError(f"expected {self.chunk} samples, got {len(block)}")
        self._input = np.concatenate([self._input[self.chunk :], block.astype(np.float32)])
        if self.mode == "muted":
            return self._silence()
        try:
            start = time.perf_counter()
            out = self.engine.convert(self._input, self.sample_rate, self.pitch_shift, self.params)
            self.infer_ms = (time.perf_counter() - start) * 1000.0
            if out.shape != self._input.shape or not np.all(np.isfinite(out)):
                raise ValueError("engine returned invalid audio")
        except Exception as exc:  # noqa: BLE001 - any engine fault must fail closed
            self._failures += 1
            if self._failures >= self._max_failures:
                raise EngineFailure(f"Voice conversion failed {self._failures} times in a row: {exc}") from exc
            return self._silence()
        self._failures = 0
        if self.infer_ms > self.params.chunk_ms:
            self.slow_chunks += 1
        segment = out[-(self.chunk + self.crossfade) :].astype(np.float32)
        head = segment[: self.crossfade] * self._fade_in + self._tail * self._fade_out
        emitted = np.concatenate([head, segment[self.crossfade : self.chunk]])
        self._tail = segment[self.chunk :].copy()
        return emitted
