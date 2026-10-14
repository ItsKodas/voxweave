import threading
import time

import numpy as np

from aivoice.processor import StreamProcessor
from aivoice.ringbuffer import RingBuffer
from aivoice.router import Router

SLOW_CHUNK_LIMIT = 10
JOIN_TIMEOUT_S = 2.0


class LiveSession:
    """Wires device streams to a StreamProcessor.

    Fail-closed: microphone samples only ever go into the private input buffer. The router
    receives processor output, or silence after any fault.

    Each start primes every output with one chunk of silence, and the devices are handed plain
    silence (leaving that chunk untouched) until the first converted chunk arrives. From then on
    the output runs one chunk ahead of the device, so an inference that is late by less than a
    chunk neither inserts a gap nor counts as a dropout.
    """

    def __init__(self, processor: StreamProcessor, router: Router, sample_rate: int, stream_factory,
                 input_device: int, output_devices: dict[str, int]):
        self.processor = processor
        self.router = router
        self.sample_rate = sample_rate
        self.running = False
        self.error: str | None = None
        self._factory = stream_factory
        self._input_device = input_device
        self._output_devices = output_devices
        self._inbuf = RingBuffer(self._input_capacity())
        self._streams: list = []
        self._thread: threading.Thread | None = None
        self._failed = False
        self._flowing = False
        self._in_level = 0.0
        self._out_level = 0.0
        # start()/stop() are serialized by _lifecycle. The worker thread never takes it, so stop()
        # can hold it while joining. _streams_lock only guards swapping the stream list, which makes
        # _close_streams idempotent when the worker and stop() both try to close.
        self._lifecycle = threading.RLock()
        self._streams_lock = threading.Lock()

    def _input_capacity(self) -> int:
        """Microphone audio waiting to be converted: a chunk being collected while the previous one is converted,
        with room to spare. Four seconds, or more for long chunks."""
        return max(self.sample_rate * 4, 3 * self.processor.chunk + self.sample_rate)

    def _fail(self, message: str) -> None:
        if self.error is None:
            self.error = message
        self._failed = True

    def feed(self, indata: np.ndarray) -> None:
        """Audio input callback. Never raises."""
        try:
            if self._failed:
                return
            mono = indata if indata.ndim == 1 else indata.mean(axis=1)
            self._in_level = float(np.max(np.abs(mono))) if len(mono) else 0.0
            self._inbuf.write(mono)
        except Exception as exc:  # noqa: BLE001 - must not propagate into the audio thread
            self._fail(f"Audio input failed: {exc}")

    def pump(self) -> bool:
        if self._failed or self._inbuf.available < self.processor.chunk:
            return False
        try:
            block = self._inbuf.read(self.processor.chunk)
            out = self.processor.process(block)
            self._out_level = float(np.max(np.abs(out)))
            self.router.push(out)
            self._flowing = True
            return True
        except Exception as exc:  # noqa: BLE001 - any fault must fail closed, never reach the worker
            self._fail(str(exc) or type(exc).__name__)
            try:
                self.router.push(np.zeros(self.processor.chunk, dtype=np.float32))
            except Exception:  # noqa: BLE001
                pass
            return False

    def _fill(self, name: str, n: int) -> np.ndarray:
        """Audio output callback. Silence until converted audio flows, then the router's output."""
        if not self._flowing:
            return np.zeros(n, dtype=np.float32)
        return self.router.read(name, n)

    def check_devices(self) -> bool:
        if all(s.active for s in list(self._streams)):
            return True
        self._fail("An audio device was disconnected")
        return False

    def start(self) -> None:
        with self._lifecycle:
            if self.running and not self._failed:
                return
            old = self._thread
            if old is not None and old is not threading.current_thread():
                old.join(timeout=JOIN_TIMEOUT_S)
                if old.is_alive():
                    raise RuntimeError("Previous live session is still shutting down")
            self._thread = None
            self._close_streams()
            self.running = False
            self.error = None
            self._failed = False
            self._in_level = self._out_level = 0.0
            self._inbuf = RingBuffer(self._input_capacity())
            self._flowing = False
            self.router.prime(self.processor.chunk)
            streams: list = []
            with self._streams_lock:
                self._streams = streams  # visible early so a failed open can close what was opened
            try:
                streams.append(self._factory.input(self._input_device, self.sample_rate, self.feed))
                for name, device in self._output_devices.items():
                    streams.append(
                        self._factory.output(device, self.sample_rate, lambda n, name=name: self._fill(name, n))
                    )
                for stream in streams:
                    stream.start()
            except Exception:
                self._close_streams()
                raise
            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        last_check = time.monotonic()
        try:
            while self.running and not self._failed:
                if not self.pump():
                    time.sleep(0.002)
                if time.monotonic() - last_check > 0.5:
                    last_check = time.monotonic()
                    self.check_devices()
        except Exception as exc:  # noqa: BLE001 - the worker must never die silently
            self._fail(str(exc) or type(exc).__name__)
        if self._failed:
            self._close_streams()
            self.running = False

    def _close_streams(self) -> None:
        with self._streams_lock:
            streams, self._streams = self._streams, []
        for stream in streams:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001 - closing a dead device must not raise
                pass
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        with self._lifecycle:
            self.running = False
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=JOIN_TIMEOUT_S)
                if not thread.is_alive():
                    self._thread = None
            self._close_streams()

    def stats(self) -> dict:
        p = self.processor
        streams = list(self._streams)
        stream_ms = sum(getattr(s, "latency", 0.0) for s in streams[:2]) * 1000.0
        # Latency is reported for the virtual mic; a listen-only session has just the monitor.
        output = next((name for name in ("mic", "monitor") if self.router.has(name)), None)
        buffered_ms = 1000.0 * self.router.buffered(output) / self.sample_rate if output else 0.0
        return {
            "running": self.running,
            "mode": p.mode,
            "error": self.error,
            "infer_ms": round(p.infer_ms, 1),
            "latency_ms": round(p.params.chunk_ms + p.params.crossfade_ms + p.infer_ms + stream_ms + buffered_ms, 1),
            "in_level": self._in_level,
            "out_level": self._out_level,
            "underruns": self.router.underruns(),
            "slow_chunks": p.slow_chunks,
            "suggest_faster": p.slow_chunks >= SLOW_CHUNK_LIMIT,
        }
