import time

import numpy as np

from aivoice.live import LiveSession
from aivoice.presets import LiveParams
from aivoice.processor import StreamProcessor
from aivoice.router import Router

SR = 48000
PARAMS = LiveParams(50, 10, 100, "dio", 0.0)


class Identity:
    def convert(self, audio, sample_rate, pitch_shift, params):
        return audio.copy()


class Raises:
    def convert(self, audio, sample_rate, pitch_shift, params):
        raise RuntimeError("boom")


class FakeStream:
    def __init__(self, kind, device):
        self.kind, self.device = kind, device
        self.active = False
        self.closed = False
        self.close_calls = 0
        self.latency = 0.01

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True
        self.close_calls += 1


class FakeStreams:
    def __init__(self, fail_on=None):
        self.streams = []
        self.fail_on = fail_on

    def input(self, device, samplerate, on_audio):
        return self._make("in", device)

    def output(self, device, samplerate, fill):
        return self._make("out", device)

    def _make(self, kind, device):
        if device == self.fail_on:
            raise RuntimeError("device busy")
        stream = FakeStream(kind, device)
        self.streams.append(stream)
        return stream


def make(engine=None, streams=None, outputs=None, max_failures=5):
    proc = StreamProcessor(engine or Identity(), SR, PARAMS, max_failures=max_failures)
    router = Router(SR * 2)
    outputs = outputs or {"mic": 3}
    for name in outputs:
        router.add_output(name)
    streams = streams or FakeStreams()
    return LiveSession(proc, router, SR, streams, 1, outputs), proc, router, streams


def drain(session):
    while session.pump():
        pass


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_arbitrary_block_sizes_lose_no_samples():
    session, proc, router, _ = make()
    x = np.random.default_rng(1).uniform(-0.9, 0.9, proc.chunk * 6).astype(np.float32)
    sizes = [1000, 37, 2048, 480, 5000, 1]
    pos = 0
    while pos < len(x):
        size = sizes[pos % len(sizes)]
        session.feed(x[pos : pos + size])
        pos += size
        drain(session)
    y = router.read("mic", len(x))
    assert np.allclose(y[proc.crossfade :], x[: -proc.crossfade], atol=1e-5)


def test_pump_without_enough_audio_does_nothing():
    session, proc, router, _ = make()
    session.feed(np.ones(proc.chunk - 1, dtype=np.float32))
    assert session.pump() is False
    assert router.buffered("mic") == 0


def test_stereo_input_is_downmixed():
    session, proc, router, _ = make()
    stereo = np.stack([np.ones(proc.chunk * 3), -np.ones(proc.chunk * 3)], axis=1).astype(np.float32)
    session.feed(stereo)
    drain(session)
    assert np.allclose(router.read("mic", proc.chunk * 3), 0.0)


def test_single_channel_2d_input_accepted():
    session, proc, router, _ = make()
    session.feed(np.ones((proc.chunk, 1), dtype=np.float32))
    assert session.pump() is True


def test_engine_failure_stops_session_with_error_and_silence():
    session, proc, router, _ = make(engine=Raises(), max_failures=2)
    session.feed(np.ones(proc.chunk * 4, dtype=np.float32))
    drain(session)
    assert session.error and "failed" in session.error
    assert np.all(router.read("mic", proc.chunk * 4) == 0.0)
    assert session.pump() is False


class RaisingProcessor:
    """Processor stub whose process() raises something other than EngineFailure."""

    chunk = 2400
    mode = "converted"
    infer_ms = 0.0
    slow_chunks = 0
    params = PARAMS

    def process(self, block):
        raise RuntimeError("unexpected")


def test_unexpected_processor_exception_fails_closed():
    router = Router(SR * 2)
    router.add_output("mic")
    session = LiveSession(RaisingProcessor(), router, SR, FakeStreams(), 1, {"mic": 3})
    session.feed(np.ones(RaisingProcessor.chunk * 2, dtype=np.float32))
    assert session.pump() is False
    assert session.error and "unexpected" in session.error
    assert np.all(router.read("mic", RaisingProcessor.chunk * 2) == 0.0)
    assert session.pump() is False


def test_no_microphone_sample_reaches_router_on_failure():
    session, proc, router, _ = make(engine=Raises(), max_failures=1)
    session.feed(np.full(proc.chunk * 3, 0.7, dtype=np.float32))
    drain(session)
    assert np.all(router.read("mic", proc.chunk * 3) == 0.0)


def test_feed_exception_is_contained_and_fails_session():
    session, proc, router, _ = make()
    session.feed(None)  # not an array: must not raise on the audio callback thread
    assert session.error
    assert session.pump() is False
    assert router.buffered("mic") == 0


def test_start_opens_and_starts_all_streams():
    session, _, _, streams = make(outputs={"mic": 3, "monitor": 2})
    session.start()
    try:
        assert [(s.kind, s.device) for s in streams.streams] == [("in", 1), ("out", 3), ("out", 2)]
        assert all(s.active for s in streams.streams)
        assert session.running
    finally:
        session.stop()


def test_start_twice_does_not_open_more_streams():
    session, _, _, streams = make()
    session.start()
    session.start()
    try:
        assert len(streams.streams) == 2
    finally:
        session.stop()


def test_stop_closes_streams_and_is_idempotent():
    session, _, _, streams = make()
    session.start()
    session.stop()
    session.stop()
    assert all(s.closed and not s.active for s in streams.streams)
    assert not session.running


def test_stop_before_start_is_safe():
    session, _, _, _ = make()
    session.stop()


def test_failed_open_closes_already_opened_streams():
    streams = FakeStreams(fail_on=3)
    session, _, _, _ = make(streams=streams)
    try:
        session.start()
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert all(s.closed for s in streams.streams)
    assert not session.running


def test_disconnected_device_is_detected():
    session, _, _, streams = make()
    session.start()
    try:
        streams.streams[0].active = False
        assert session.check_devices() is False
        assert "disconnected" in session.error
    finally:
        session.stop()


def test_worker_failure_closes_streams_once_even_with_concurrent_stop():
    session, proc, router, streams = make(engine=Raises(), max_failures=1)
    session.start()
    session.feed(np.ones(proc.chunk * 2, dtype=np.float32))
    assert wait_until(lambda: session.error is not None)
    session.stop()
    assert wait_until(lambda: not session.running)
    assert all(s.closed and not s.active for s in streams.streams)
    assert all(s.close_calls == 1 for s in streams.streams)


def test_start_after_failure_resets_state_and_runs_again():
    session, proc, router, streams = make(engine=Raises(), max_failures=1)
    session.start()
    session.feed(np.ones(proc.chunk, dtype=np.float32))
    assert wait_until(lambda: session.error is not None and not session.running)
    session.processor.engine = Identity()
    session.start()
    try:
        assert session.running and session.error is None
        assert len(streams.streams) == 4
        session.feed(np.full(proc.chunk * 2, 0.5, dtype=np.float32))
        assert wait_until(lambda: router.buffered("mic") >= proc.chunk * 2)
        assert session.error is None and session.running
    finally:
        session.stop()


def test_stats_shape():
    session, proc, _, _ = make()
    session.feed(np.full(proc.chunk, 0.5, dtype=np.float32))
    session.pump()
    stats = session.stats()
    assert set(stats) == {"running", "mode", "error", "infer_ms", "latency_ms", "in_level", "out_level",
                          "underruns", "slow_chunks", "suggest_faster"}
    assert stats["in_level"] == 0.5 and stats["mode"] == "converted"
    assert stats["latency_ms"] >= PARAMS.chunk_ms + PARAMS.crossfade_ms
