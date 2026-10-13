import threading

import numpy as np
import pytest

from aivoice.audiofile import read_wav
from aivoice.playback import Player, save_wav
from tests.test_live import FakeStreams

SR = 48000


class CapturingStreams(FakeStreams):
    def __init__(self, fail_on=None):
        super().__init__(fail_on)
        self.fills = {}

    def output(self, device, samplerate, fill):
        stream = super().output(device, samplerate, fill)
        self.fills[device] = fill
        return stream


def ramp(start, n):
    return np.arange(start, start + n, dtype=np.float32) / 1e6


def pull(fill, total, chunk=997):
    parts = []
    got = 0
    while got < total:
        n = min(chunk, total - got)
        out = fill(n)
        assert out.shape == (n,) and out.dtype == np.float32
        parts.append(out)
        got += n
    return np.concatenate(parts)


def make(devices=None, **kw):
    streams = CapturingStreams(**kw)
    player = Player(streams)
    player.start(devices or {"mic": 3, "monitor": 4})
    return player, streams


def test_pieces_play_in_order_with_pause_on_both_outputs():
    player, streams = make()
    a, b, c = ramp(1, 1000), ramp(5000, 777), ramp(9000, 50)
    for piece in (a, b, c):
        player.enqueue(piece, pause_ms=10)
    player.finish()
    pause = np.zeros(480, dtype=np.float32)
    expected = np.concatenate([a, pause, b, pause, c])
    for device in (3, 4):
        got = pull(streams.fills[device], len(expected))
        np.testing.assert_array_equal(got, expected)
    st = player.status()
    assert st["playing"] is False and st["error"] is None
    assert st["queued_seconds"] == 0 and st["played_seconds"] == pytest.approx(len(expected) / SR)
    assert all(s.closed for s in streams.streams)


def test_silence_when_empty_and_audio_arrives_later():
    player, streams = make({"mic": 3})
    fill = streams.fills[3]
    assert not fill(100).any()
    player.enqueue(ramp(1, 200), pause_ms=0)
    np.testing.assert_array_equal(fill(300)[:200], ramp(1, 200))
    assert player.status()["playing"] is True
    assert not any(s.closed for s in streams.streams)


def test_slow_output_does_not_delay_the_other_and_audio_is_released():
    player, streams = make()
    piece = ramp(1, 4000)
    player.enqueue(piece, pause_ms=0)
    player.enqueue(ramp(9000, 4000), pause_ms=0)
    fast = pull(streams.fills[3], 8000, chunk=313)
    np.testing.assert_array_equal(fast[:4000], piece)
    st = player.status()
    assert st["played_seconds"] == 0 and st["queued_seconds"] == pytest.approx(8000 / SR)
    slow = pull(streams.fills[4], 1000)
    np.testing.assert_array_equal(slow, piece[:1000])
    assert player.status()["played_seconds"] == pytest.approx(1000 / SR)
    session = player._session
    pull(streams.fills[4], 7000)
    assert len(session._segs) == 0  # both consumed everything, nothing retained


def test_stop_mid_piece_gives_silence_and_closes():
    player, streams = make()
    player.enqueue(ramp(1, 5000), pause_ms=0)
    assert streams.fills[3](100).any()
    player.stop()
    for device in (3, 4):
        assert not streams.fills[device](100).any()
    assert all(s.closed for s in streams.streams)
    st = player.status()
    assert st["playing"] is False and st["queued_seconds"] == 0
    player.enqueue(ramp(1, 5000))
    assert player.status()["queued_seconds"] == 0
    assert not streams.fills[3](100).any()


def test_stop_from_another_thread_while_callbacks_run():
    player, streams = make({"mic": 3})
    player.enqueue(ramp(1, 2_000_000), pause_ms=0)
    fill = streams.fills[3]
    started = threading.Event()

    def audio_thread():
        while not player._session._stopped:
            fill(256)
            started.set()

    t = threading.Thread(target=audio_thread)
    t.start()
    assert started.wait(3)
    player.stop()
    t.join(timeout=3)
    assert not t.is_alive()
    assert not fill(64).any()
    assert streams.streams[0].closed


def test_inactive_stream_sets_error_and_ends_without_raising():
    player, streams = make()
    player.enqueue(ramp(1, 5000), pause_ms=0)
    streams.streams[1].active = False
    st = player.status()
    assert st["playing"] is False
    assert st["error"] == "An audio device was disconnected"
    assert not streams.fills[3](100).any()
    assert all(s.closed for s in streams.streams)


def test_unwatched_session_closes_by_itself():
    player, streams = make({"mic": 3})
    player.enqueue(ramp(1, 100), pause_ms=0)
    player.finish()
    streams.fills[3](100)
    assert player.wait(3.0) is True
    assert streams.streams[0].closed and streams.streams[0].close_calls == 1


def test_finish_with_nothing_queued_ends():
    player, streams = make({"mic": 3})
    player.finish()
    assert player.wait(3.0) is True
    assert streams.streams[0].closed


def test_gains_apply_per_output():
    streams = CapturingStreams()
    player = Player(streams)
    player.start({"mic": 3, "monitor": 4}, gains={"monitor": 0.5})
    player.enqueue(np.full(10, 0.8, dtype=np.float32))
    np.testing.assert_allclose(streams.fills[3](10), 0.8)
    np.testing.assert_allclose(streams.fills[4](10), 0.4)


def test_start_twice_replaces_first_session():
    streams = CapturingStreams()
    player = Player(streams)
    player.start({"mic": 3})
    first = streams.streams[0]
    old_fill = streams.fills[3]
    player.enqueue(ramp(1, 100))
    player.start({"mic": 5})
    assert first.closed and first.close_calls == 1
    assert not old_fill(10).any()
    player.enqueue(ramp(1, 100), pause_ms=0)
    np.testing.assert_array_equal(streams.fills[5](100), ramp(1, 100))
    assert player.status()["playing"] is True


def test_unopenable_device_raises_and_leaves_nothing_open():
    streams = CapturingStreams(fail_on=4)
    player = Player(streams)
    with pytest.raises(ValueError):
        player.start({"mic": 3, "monitor": 4})
    assert streams.streams and all(s.closed for s in streams.streams)
    assert player.status()["playing"] is False
    player.enqueue(ramp(1, 10))  # ignored, no session


def test_no_devices_is_refused():
    with pytest.raises(ValueError):
        Player(CapturingStreams()).start({})


def test_bad_audio_shape_is_refused():
    player, _ = make({"mic": 3})
    with pytest.raises(ValueError):
        player.enqueue(np.zeros((10, 2), dtype=np.float32))


def test_save_wav_joins_with_pauses(tmp_path):
    path = tmp_path / "out.wav"
    a, b = np.full(480, 0.5, dtype=np.float32), np.full(240, -0.5, dtype=np.float32)
    save_wav(path, [a, np.zeros(0, dtype=np.float32), b], pause_ms=10)
    audio, sr = read_wav(path)
    assert sr == SR and len(audio) == 480 + 480 + 240
    assert audio[:480].min() > 0.49 and not audio[480:960].any() and audio[960:].max() < -0.49
    with pytest.raises(ValueError):
        save_wav(tmp_path / "none.wav", [])
