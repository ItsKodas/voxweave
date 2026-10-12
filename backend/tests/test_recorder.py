import numpy as np
import pytest

from aivoice.recorder import Recorder, RecorderError
from tests.test_live import FakeStream

SR = 48000


class CapturingStreams:
    """Stream factory double that hands the test the recorder's audio callback."""

    def __init__(self, fail=False):
        self.streams = []
        self.on_audio = None
        self.fail = fail

    def input(self, device, samplerate, on_audio):
        if self.fail:
            raise RuntimeError("device busy")
        self.on_audio = on_audio
        stream = FakeStream("in", device)
        self.streams.append(stream)
        return stream


def block(seconds, value=0.25):
    return np.full((int(seconds * SR), 1), value, dtype=np.float32)


def test_records_what_the_microphone_delivers():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(7)
    assert rec.recording and streams.streams[0].device == 7 and streams.streams[0].active
    streams.on_audio(block(0.5, 0.25))
    streams.on_audio(block(0.25, -0.5))
    status = rec.status()
    assert status == {"recording": True, "level": 0.5, "seconds": 0.75, "full": False}
    audio = rec.stop()
    assert audio.dtype == np.float32 and audio.shape == (int(0.75 * SR),)
    assert audio[0] == 0.25 and audio[-1] == -0.5
    assert not rec.recording and streams.streams[0].closed


def test_stereo_blocks_are_downmixed():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(1)
    streams.on_audio(np.stack([np.ones(480), -np.ones(480)], axis=1).astype(np.float32))
    assert np.allclose(rec.stop(), 0.0)


def test_stop_without_audio_returns_empty():
    rec = Recorder(CapturingStreams())
    rec.start(1)
    assert rec.stop().shape == (0,)


def test_recording_stops_growing_at_the_limit():
    streams = CapturingStreams()
    rec = Recorder(streams, max_seconds=1.0)
    rec.start(1)
    for _ in range(5):
        streams.on_audio(block(0.5))
    assert rec.status()["full"] is True
    assert len(rec.stop()) == int(1.0 * SR)


def test_start_twice_is_refused():
    rec = Recorder(CapturingStreams())
    rec.start(1)
    with pytest.raises(RecorderError, match="Already recording"):
        rec.start(1)
    rec.cancel()


def test_stop_when_idle_is_refused():
    with pytest.raises(RecorderError, match="Not recording"):
        Recorder(CapturingStreams()).stop()


def test_cancel_discards_and_is_safe_when_idle():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.cancel()
    rec.start(1)
    streams.on_audio(block(0.5))
    rec.cancel()
    assert not rec.recording and streams.streams[0].closed
    assert rec.status() == {"recording": False, "level": 0.0, "seconds": 0.0, "full": False}


def test_a_device_that_will_not_open_is_a_clear_error():
    rec = Recorder(CapturingStreams(fail=True))
    with pytest.raises(RecorderError, match="Could not open the microphone"):
        rec.start(1)
    assert not rec.recording


def test_late_callback_after_stop_is_ignored():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(1)
    rec.stop()
    streams.on_audio(block(0.5))
    assert rec.status()["seconds"] == 0.0


def test_bad_block_from_the_driver_does_not_raise():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(1)
    streams.on_audio(None)
    streams.on_audio(block(0.1))
    assert len(rec.stop()) == int(0.1 * SR)


class DyingStream(FakeStream):
    """A device that was unplugged: stop and close raise."""

    def stop(self):
        raise OSError("device unplugged")

    def close(self):
        raise OSError("device unplugged")


class DyingStreams(CapturingStreams):
    def input(self, device, samplerate, on_audio):
        self.on_audio = on_audio
        stream = DyingStream("in", device)
        self.streams.append(stream)
        return stream


def test_stream_that_dies_mid_take_still_returns_what_was_captured():
    streams = DyingStreams()
    rec = Recorder(streams)
    rec.start(1)
    streams.on_audio(block(0.5))
    audio = rec.stop()
    assert len(audio) == int(0.5 * SR)
    assert not rec.recording
    rec.start(1)  # and a new take can begin afterwards
    rec.cancel()


def test_callback_never_raises_on_odd_blocks():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(1)
    for bad in (None, "x", np.zeros((4, 2, 2), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)):
        streams.on_audio(bad)
    streams.on_audio(np.full((480, 1), np.nan, dtype=np.float32))
    audio = rec.stop()
    assert np.all(np.isfinite(audio))


def test_callback_after_the_stream_failed_to_start_is_ignored():
    streams = CapturingStreams()

    class NoStart(FakeStream):
        def start(self):
            raise RuntimeError("cannot start")

    def make(device, samplerate, on_audio):
        streams.on_audio = on_audio
        return NoStart("in", device)

    streams.input = make
    rec = Recorder(streams)
    with pytest.raises(RecorderError, match="Could not open the microphone"):
        rec.start(1)
    streams.on_audio(block(0.5))
    assert rec.status()["seconds"] == 0.0
