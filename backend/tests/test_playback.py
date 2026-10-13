import threading

import numpy as np
import pytest

from aivoice.audiofile import read_wav
from aivoice.playback import Player, save_wav
from tests.test_live import FakeStream, FakeStreams

SR = 48000


class EventStream(FakeStream):
    """FakeStream with hooks: `active` can raise, start can fail, close can block, closing is observable."""

    def __init__(self, kind, device):
        self.raise_active = False
        self.on_active_read = None
        self.start_fails = False
        self.close_gate = None
        self.close_entered = threading.Event()
        self.closed_event = threading.Event()
        super().__init__(kind, device)

    @property
    def active(self):
        hook, self.on_active_read = self.on_active_read, None
        if hook is not None:
            hook()
        if self.raise_active:
            raise RuntimeError("PortAudio stream is invalid")
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    def start(self):
        if self.start_fails:
            raise RuntimeError("cannot start")
        super().start()

    def close(self):
        self.close_entered.set()
        if self.close_gate is not None:
            assert self.close_gate.wait(5)
        super().close()
        self.closed_event.set()


class CapturingStreams(FakeStreams):
    def __init__(self, fail_on=None, start_fail_on=None):
        super().__init__(fail_on)
        self.fills = {}
        self.start_fail_on = start_fail_on

    def _make(self, kind, device):
        if device == self.fail_on:
            raise RuntimeError("device busy")
        stream = EventStream(kind, device)
        stream.start_fails = device == self.start_fail_on
        self.streams.append(stream)
        return stream

    def output(self, device, samplerate, fill):
        stream = self._make("out", device)
        stream.fill = fill
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


def make(devices=None, clock=None, **kw):
    streams = CapturingStreams(**kw)
    player = Player(streams, clock=clock) if clock else Player(streams)
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
    out = fill(300)
    np.testing.assert_array_equal(out[:200], ramp(1, 200))
    assert not out[200:].any()
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
    stop_returned = threading.Event()
    quit_audio = threading.Event()
    started = threading.Event()
    leaked = []

    def audio_thread():
        while not quit_audio.is_set():
            after_stop = stop_returned.is_set()
            out = fill(256)
            started.set()
            if after_stop and out.any():
                leaked.append(out)

    def stopper():
        player.stop()
        stop_returned.set()

    audio = threading.Thread(target=audio_thread, daemon=True)
    audio.start()
    assert started.wait(3)
    stopper_thread = threading.Thread(target=stopper, daemon=True)
    stopper_thread.start()
    stopper_thread.join(timeout=3)
    if stopper_thread.is_alive():  # a deadlock fails here instead of hanging the suite
        quit_audio.set()
        pytest.fail("stop() did not return within 3 seconds")
    for _ in range(50):  # make sure callbacks ran after stop returned
        fill(64)
    quit_audio.set()
    audio.join(timeout=3)
    assert not leaked
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


def test_unwatched_session_closes_by_itself_without_any_player_call():
    player, streams = make({"mic": 3})
    player.enqueue(ramp(1, 100), pause_ms=0)
    player.finish()
    streams.fills[3](100)
    # No Player method is called from here on: only the watcher can close the stream.
    assert streams.streams[0].closed_event.wait(3)


def test_enqueue_after_stop_is_not_queued():
    player, streams = make({"mic": 3})
    session = player._session
    player.enqueue(ramp(1, 100), pause_ms=0)
    player.stop()
    total = session._total
    player.enqueue(ramp(1, 5000), pause_ms=0)
    assert len(session._segs) == 0 and session._total == total


def test_enqueue_after_finish_is_not_queued():
    player, streams = make({"mic": 3})
    session = player._session
    player.enqueue(ramp(1, 100), pause_ms=0)
    player.finish()
    total, segs = session._total, len(session._segs)
    player.enqueue(ramp(1, 5000), pause_ms=0)
    assert session._total == total and len(session._segs) == segs
    np.testing.assert_array_equal(streams.fills[3](200)[:100], ramp(1, 100))


def test_finished_waits_for_every_output():
    player, streams = make()
    piece = ramp(1, 1000)
    player.enqueue(piece, pause_ms=0)
    pull(streams.fills[3], 1000)
    player.finish()
    assert player.status()["playing"] is True
    assert player.wait(0) is False
    assert not any(s.closed for s in streams.streams)
    pull(streams.fills[4], 1000)
    assert player.wait(3) is True
    assert all(s.closed for s in streams.streams)


@pytest.mark.parametrize("mode", ["inactive", "raises"])
def test_watcher_detects_lost_device_when_nobody_polls(mode):
    player, streams = make()
    player.enqueue(ramp(1, 5000), pause_ms=0)
    if mode == "raises":
        streams.streams[1].raise_active = True
    else:
        streams.streams[1].active = False
    for stream in streams.streams:
        assert stream.closed_event.wait(3)
    st = player.status()
    assert st["playing"] is False and st["error"] == "An audio device was disconnected"


def test_active_that_raises_is_a_disconnect_and_status_never_raises():
    player, streams = make()
    player.enqueue(ramp(1, 5000), pause_ms=0)
    streams.streams[0].raise_active = True
    st = player.status()
    assert st == {"playing": False, "queued_seconds": 0.0, "played_seconds": 0.0, "error": "An audio device was disconnected"}
    assert player.wait(0) is True
    assert all(s.closed for s in streams.streams)
    assert not streams.fills[3](50).any()


def test_stream_that_fails_to_start_leaves_nothing_open():
    streams = CapturingStreams(start_fail_on=4)
    player = Player(streams)
    with pytest.raises(ValueError):
        player.start({"mic": 3, "monitor": 4})
    assert len(streams.streams) == 2 and all(s.closed for s in streams.streams)
    assert player.status()["playing"] is False


def test_output_that_never_pulls_retains_only_queued_audio_and_stop_releases_it():
    player, streams = make()
    session = player._session
    for n in (1000, 2000, 3000):
        player.enqueue(ramp(1, n), pause_ms=0)
    pull(streams.fills[3], 6000)
    assert sum(len(seg) for seg in session._segs) == 6000
    assert player.status()["played_seconds"] == 0
    player.stop()
    assert len(session._segs) == 0


def test_normal_stop_records_no_disconnect_error():
    player, streams = make({"mic": 3})
    # The device reports inactive exactly as stop() runs: the check must not count it as a disconnect.
    streams.streams[0].on_active_read = player.stop
    player.enqueue(ramp(1, 100), pause_ms=0)
    session = player._session
    session.poll()
    assert session.error is None
    assert player.status()["error"] is None
    assert player.wait(3) is True


def test_finished_is_reported_only_after_streams_are_closed():
    player, streams = make({"mic": 3})
    gate = threading.Event()
    streams.streams[0].close_gate = gate
    stopper = threading.Thread(target=player.stop)
    stopper.start()
    assert streams.streams[0].close_entered.wait(3)
    try:
        assert player.status()["playing"] is True
        assert player.wait(0) is False
    finally:
        gate.set()
        stopper.join(timeout=3)
    assert player.wait(3) is True
    assert player.status()["playing"] is False


def test_nan_inf_and_overs_are_cleaned():
    player, streams = make({"mic": 3})
    player.enqueue(np.array([np.nan, np.inf, -np.inf, 2.0, -3.0, 0.5], dtype=np.float32), pause_ms=0)
    np.testing.assert_array_equal(streams.fills[3](6), np.array([0, 0, 0, 1, -1, 0.5], dtype=np.float32))


def test_stalled_output_ends_playback_with_error():
    now = [100.0]
    player, streams = make({"mic": 3}, clock=lambda: now[0])
    now[0] = 104.9
    assert player.status()["playing"] is True
    now[0] = 105.1
    st = player.status()
    assert st["playing"] is False and st["error"] == "An audio device stopped responding"
    assert all(s.closed for s in streams.streams)


def test_pulling_resets_the_stall_timer():
    now = [0.0]
    player, streams = make({"mic": 3}, clock=lambda: now[0])
    now[0] = 4.0
    streams.fills[3](10)
    now[0] = 8.0
    assert player.status()["error"] is None
    now[0] = 9.5
    assert player.status()["error"] == "An audio device stopped responding"


def test_no_extra_pause_when_the_queue_ran_dry():
    player, streams = make({"mic": 3})
    fill = streams.fills[3]
    a, b = ramp(1, 200), ramp(5000, 100)
    player.enqueue(a, pause_ms=10)
    fill(300)  # underrun: the gap already was the pause
    player.enqueue(b, pause_ms=10)
    np.testing.assert_array_equal(fill(100), b)


def test_pause_is_kept_when_the_queue_did_not_run_dry():
    player, streams = make({"mic": 3})
    fill = streams.fills[3]
    a, b = ramp(1, 200), ramp(5000, 100)
    player.enqueue(a, pause_ms=10)
    fill(200)  # exactly drained, no underrun
    player.enqueue(b, pause_ms=10)
    got = fill(580)
    assert not got[:480].any()
    np.testing.assert_array_equal(got[480:], b)


def test_late_audio_for_an_old_session_is_dropped():
    streams = CapturingStreams()
    player = Player(streams)
    old = player.start({"mic": 3})
    new = player.start({"mic": 5})
    assert old is not new
    player.enqueue(ramp(1, 100), pause_ms=0, session=old)
    assert player.status()["queued_seconds"] == 0
    player.enqueue(ramp(1, 100), pause_ms=0, session=new)
    assert player.status()["queued_seconds"] == pytest.approx(100 / SR)


def test_a_device_that_is_slow_to_open_is_not_taken_for_one_that_stopped_responding():
    # The stall timer starts when the streams have been started, not when the session is created: opening a
    # device can take seconds (and the computer can sleep in between).
    now = [100.0]

    class SlowStreams(CapturingStreams):
        def output(self, device, samplerate, fill):
            now[0] += 6.0
            return super().output(device, samplerate, fill)

    streams = SlowStreams()
    player = Player(streams, clock=lambda: now[0])
    player.start({"mic": 3, "monitor": 4})  # 12 seconds pass while the two devices open
    st = player.status()
    assert st["playing"] is True and st["error"] is None
    now[0] += 4.9
    assert player.status()["error"] is None
    now[0] += 0.2
    assert player.status()["error"] == "An audio device stopped responding"


class WatchedEvent(threading.Event):
    """An Event that shows when somebody has begun to wait for it."""

    def __init__(self):
        super().__init__()
        self.awaited = threading.Event()

    def wait(self, timeout=None):
        self.awaited.set()
        return super().wait(timeout)


def closing_on_the_watcher(devices=None):
    """A session whose watcher thread is held inside the close of its stream, as after playback has ended."""
    player, streams = make(devices or {"mic": 3})
    session = player._session
    session.done = WatchedEvent()
    gate = threading.Event()
    streams.streams[0].close_gate = gate
    player.finish()  # nothing queued: the watcher closes the stream, and is held inside close()
    assert streams.streams[0].close_entered.wait(3)
    return player, streams, session, gate


def test_stop_waits_for_streams_another_thread_is_still_closing():
    # stop() must not return, and so the next session must not open the device, before they are closed.
    player, streams, session, gate = closing_on_the_watcher()
    returned = threading.Event()
    threading.Thread(target=lambda: (player.stop(), returned.set()), daemon=True).start()
    try:
        assert session.done.awaited.wait(3), "stop() returned without waiting for the close"
        assert not returned.is_set() and not streams.streams[0].closed
    finally:
        gate.set()
    assert returned.wait(3) and streams.streams[0].closed and session.done.is_set()


def test_start_waits_for_the_previous_sessions_streams_to_be_closed():
    player, streams, session, gate = closing_on_the_watcher()
    first = streams.streams[0]
    opened_while_open = []
    real_output = streams.output

    def output(device, samplerate, fill):
        opened_while_open.append(not first.closed)
        return real_output(device, samplerate, fill)

    streams.output = output
    starter = threading.Thread(target=lambda: player.start({"mic": 3}), daemon=True)
    starter.start()
    try:
        assert session.done.awaited.wait(3), "start() did not wait for the old stream to be closed"
        assert opened_while_open == []
    finally:
        gate.set()
    starter.join(3)
    assert not starter.is_alive() and opened_while_open == [False]
    player.stop()


def test_stop_does_not_wait_forever_for_a_close_that_hangs(monkeypatch):
    from aivoice import playback

    monkeypatch.setattr(playback, "CLOSE_WAIT", 0.0)
    player, streams = make({"mic": 3})
    gate = threading.Event()
    streams.streams[0].close_gate = gate
    player.finish()
    assert streams.streams[0].close_entered.wait(3)
    try:
        player.stop()  # returns although the stream is still being closed
        assert player.status()["playing"] is True
    finally:
        gate.set()
    assert player.wait(3) is True
