import time

import numpy as np

from aivoice.engines.base import VoiceEngine
from aivoice.presets import LiveParams


class EngineFailure(RuntimeError):
    pass


SOLA_SEARCH_S = 0.012  # how far a join may move to line the new chunk up with the end of the last one


def sola_offset(tail: np.ndarray, candidate: np.ndarray, search: int) -> int:
    """Where in candidate the stretch most like tail begins (0..search), by normalised cross-correlation.
    `search` itself is where the join falls without any moving, and is the answer when there is nothing to match.

    Each chunk is generated anew, so its waveform near the join is not in step with the end of the previous one;
    a plain crossfade then partly cancels the two and the join is heard as a dip or a click. Moving the join by a
    few milliseconds to where the waveforms agree (synchronous overlap-add, as RVC's own real-time tool does)
    removes that."""
    n = len(tail)
    if n == 0 or search <= 0 or not np.any(tail):
        return max(search, 0)
    region = candidate[: n + search].astype(np.float64)
    numerator = np.correlate(region, tail.astype(np.float64), mode="valid")
    energy = np.convolve(region * region, np.ones(n), mode="valid")
    score = numerator / np.sqrt(energy + 1e-8)
    return int(np.argmax(score[: search + 1]))


class StreamProcessor:
    def __init__(self, engine: VoiceEngine, sample_rate: int, params: LiveParams, max_failures: int = 5):
        self.engine = engine
        self.sample_rate = sample_rate
        self.params = params
        self.chunk = sample_rate * params.chunk_ms // 1000
        self.crossfade = sample_rate * params.crossfade_ms // 1000
        self.window = sample_rate * params.context_ms // 1000 + self.chunk + self.crossfade
        self.lookahead = sample_rate * params.lookahead_ms // 1000
        # The join may only move within audio the window really has before the newest chunk and crossfade.
        self.search = max(0, min(int(sample_rate * SOLA_SEARCH_S),
                                 self.window - self.chunk - self.crossfade - self.lookahead))
        self._consumed = 0  # samples of input taken in so far: where each window starts in time
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

    def _record_timing(self, start: float) -> None:
        self.infer_ms = (time.perf_counter() - start) * 1000.0
        if self.infer_ms > self.params.chunk_ms:
            self.slow_chunks += 1

    def process(self, block: np.ndarray) -> np.ndarray:
        if len(block) != self.chunk:
            raise ValueError(f"expected {self.chunk} samples, got {len(block)}")
        incoming = np.zeros(self.chunk, dtype=np.float32) if self.mode == "muted" else block.astype(np.float32)
        self._input = np.concatenate([self._input[self.chunk :], incoming])
        self._consumed += self.chunk
        if self.mode == "muted":
            return self._silence()
        start = time.perf_counter()
        try:
            if hasattr(self.engine, "noise_position"):
                # The engine draws its noise from one long sequence at the window's place in time, so two windows
                # covering the same moment use the same noise for it, and agree about it at the join.
                self.engine.noise_position = self._consumed - self.window
            out = self.engine.convert(self._input, self.sample_rate, self.pitch_shift, self.params)
            with np.errstate(over="ignore"):
                out = np.asarray(out, dtype=np.float32)
            if out.shape != self._input.shape or not np.all(np.isfinite(out)):
                raise ValueError("engine returned invalid audio")
        except Exception as exc:  # noqa: BLE001 - any engine fault must fail closed
            self._record_timing(start)
            self._failures += 1
            self._silence()
            if self._failures >= self._max_failures:
                raise EngineFailure(f"Voice conversion failed {self._failures} times in a row: {exc}") from exc
            return np.zeros(self.chunk, dtype=np.float32)
        self._record_timing(start)
        self._failures = 0
        end = len(out) - self.lookahead
        candidate = out[end - (self.chunk + self.crossfade + self.search) : end]
        offset = sola_offset(self._tail, candidate, self.search)
        segment = candidate[offset : offset + self.chunk + self.crossfade]
        head = segment[: self.crossfade] * self._fade_in + self._tail * self._fade_out
        emitted = np.concatenate([head, segment[self.crossfade : self.chunk]])
        self._tail = segment[self.chunk :].copy()
        return np.clip(emitted, -1.0, 1.0)
