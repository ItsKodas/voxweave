import numpy as np
import pytest

from aivoice.devices import Device
from aivoice.manager import LiveManager, StartConfig, default_engine_factory
from aivoice.pretrained import list_voices
from tests.test_live import FakeStreams, Identity, wait_until
from tests.test_pretrained import add_voice

DEVICES = [
    Device(1, "Headset Mic", 1, 0),
    Device(2, "Headset", 0, 2),
    Device(3, "VoiceMeeter Input", 0, 8),
]


@pytest.fixture
def setup(tmp_path):
    add_voice(tmp_path, "demo")
    streams = FakeStreams()
    built = []

    def engine_factory(voice):
        built.append(voice.id)
        return Identity()

    manager = LiveManager(tmp_path, lambda: DEVICES, engine_factory, streams)
    yield manager, streams, built
    manager.stop()


def cfg(**kw):
    base = dict(input_device=1, mic_device=3, voice="demo")
    base.update(kw)
    return StartConfig(**base)


def test_start_and_stop(setup):
    manager, streams, built = setup
    manager.start(cfg())
    assert manager.stats()["running"] is True
    assert built == ["demo"]
    manager.stop()
    assert manager.stats() == {
        "running": False, "mode": "off", "error": None, "infer_ms": 0.0, "latency_ms": 0.0,
        "in_level": 0.0, "out_level": 0.0, "underruns": 0, "slow_chunks": 0, "suggest_faster": False,
    }
    assert all(s.closed for s in streams.streams)


def test_monitor_adds_second_output(setup):
    manager, streams, _ = setup
    manager.start(cfg(monitor_device=2, monitor_gain=0.5))
    assert [(s.kind, s.device) for s in streams.streams] == [("in", 1), ("out", 3), ("out", 2)]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(input_device=2), "not an input"),
        (dict(mic_device=1), "not an output"),
        (dict(monitor_device=1), "not an output"),
        (dict(monitor_device=3), "must be different"),
        (dict(mic_device=None), "Choose a virtual mic or a listen device"),
        (dict(mic_device=None, monitor_device=1), "not an output"),
        (dict(input_device=99), "no longer available"),
        (dict(voice="ghost"), "Unknown voice"),
        (dict(preset="nope"), "Unknown preset"),
        (dict(overrides={"chunk_ms": 1}), "chunk_ms"),
        (dict(pitch_shift=40), "pitch_shift"),
        (dict(monitor_gain=9), "monitor_gain"),
    ],
)
def test_bad_start_requests_open_nothing(setup, kwargs, message):
    manager, streams, _ = setup
    with pytest.raises(ValueError, match=message):
        manager.start(cfg(**kwargs))
    assert streams.streams == []
    assert manager.stats()["running"] is False


def test_start_twice_restarts_cleanly(setup):
    manager, streams, _ = setup
    manager.start(cfg())
    first = list(streams.streams)
    manager.start(cfg(preset="low_latency"))
    assert all(s.closed for s in first)
    assert sum(1 for s in streams.streams if s.active) == 2


def test_stop_when_stopped_is_safe(setup):
    manager, _, _ = setup
    manager.stop()
    manager.stop()


def test_mode_change(setup):
    manager, _, _ = setup
    manager.start(cfg())
    manager.set_mode("muted")
    assert manager.stats()["mode"] == "muted"
    manager.set_mode("converted")
    assert manager.stats()["mode"] == "converted"
    with pytest.raises(ValueError, match="mode"):
        manager.set_mode("passthrough")


def test_mode_change_when_stopped_is_a_clear_error(setup):
    manager, _, _ = setup
    with pytest.raises(ValueError, match="not running"):
        manager.set_mode("muted")


def test_update_when_stopped_is_a_clear_error(setup):
    manager, _, _ = setup
    with pytest.raises(ValueError, match="not running"):
        manager.update(pitch_shift=2)


def test_update_pitch_and_monitor(setup):
    manager, _, _ = setup
    manager.start(cfg(monitor_device=2))
    manager.update(pitch_shift=5, monitor_enabled=False, monitor_gain=0.2)
    with pytest.raises(ValueError, match="pitch_shift"):
        manager.update(pitch_shift=100)


def test_engine_that_fails_to_load_reports_and_opens_nothing(tmp_path):
    add_voice(tmp_path, "demo")
    streams = FakeStreams()

    def broken(voice):
        raise RuntimeError("corrupt file")

    manager = LiveManager(tmp_path, lambda: DEVICES, broken, streams)
    with pytest.raises(ValueError, match="Could not load voice"):
        manager.start(cfg())
    assert streams.streams == []


def test_device_open_failure_is_a_clear_error(tmp_path):
    add_voice(tmp_path, "demo")
    streams = FakeStreams(fail_on=3)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), streams)
    with pytest.raises(ValueError, match="Could not open audio devices"):
        manager.start(cfg())
    assert manager.stats()["running"] is False


def test_engine_is_warmed_up_before_streams_open(tmp_path):
    add_voice(tmp_path, "demo")
    order = []

    class Warm:
        def convert(self, audio, sample_rate, pitch_shift, params):
            order.append("convert")
            return audio

    class Streams(FakeStreams):
        def _make(self, kind, device):
            order.append("open")
            return super()._make(kind, device)

    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Warm(), Streams())
    manager.start(cfg())
    manager.stop()
    assert order[0] == "convert"
    assert np.all(np.array(order[1:3]) == "open")


@pytest.mark.parametrize("raw", ['"gpu"', "true", "1.5", "[0]", "null"])
def test_default_engine_factory_rejects_bad_device_id_only_when_an_engine_is_built(tmp_path, raw):
    add_voice(tmp_path, "demo")
    (tmp_path / "settings.json").write_text('{"dml_device_id": %s}' % raw)
    build = default_engine_factory(tmp_path)  # must not raise: the backend has to come up
    with pytest.raises(ValueError, match="dml_device_id in settings.json must be an integer"):
        build(list_voices(tmp_path)[0])


def test_bad_device_id_is_a_clear_start_error_and_opens_nothing(tmp_path):
    add_voice(tmp_path, "demo")
    (tmp_path / "settings.json").write_text('{"dml_device_id": "gpu"}')
    streams = FakeStreams()
    manager = LiveManager(tmp_path, lambda: DEVICES, default_engine_factory(tmp_path), streams)
    with pytest.raises(ValueError, match="dml_device_id in settings.json must be an integer"):
        manager.start(cfg())
    assert streams.streams == []


def test_default_engine_factory_accepts_integer_device_id(tmp_path):
    (tmp_path / "settings.json").write_text('{"dml_device_id": 1}')
    assert callable(default_engine_factory(tmp_path))


def test_monitor_only_start_opens_input_and_one_output(setup):
    manager, streams, _ = setup
    manager.start(cfg(mic_device=None, monitor_device=2))
    assert [(s.kind, s.device) for s in streams.streams] == [("in", 1), ("out", 2)]
    stats = manager.stats()
    assert stats["running"] is True and stats["latency_ms"] > 0
    manager.update(monitor_gain=0.5, monitor_enabled=False)
    manager.set_mode("muted")
    manager.stop()
    assert all(s.closed for s in streams.streams)


def test_refresh_devices_calls_the_refresher_when_idle(tmp_path):
    calls = []
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams(),
                          refresh_devices=lambda: calls.append("refresh"))
    manager.refresh_devices()
    assert calls == ["refresh"]


def test_refresh_devices_is_refused_while_running(tmp_path):
    add_voice(tmp_path, "demo")
    calls = []
    streams = FakeStreams()
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), streams,
                          refresh_devices=lambda: calls.append("refresh"))
    manager.start(cfg())
    try:
        with pytest.raises(ValueError, match="Stop live mode before refreshing devices"):
            manager.refresh_devices()
        assert calls == []
        assert manager.stats()["running"] is True
        assert all(s.active for s in streams.streams)
    finally:
        manager.stop()


def test_refresh_devices_closes_a_failed_session_first_and_keeps_its_error(tmp_path):
    add_voice(tmp_path, "demo")
    order = []
    streams = FakeStreams()
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), streams,
                          refresh_devices=lambda: order.append(all(s.closed for s in streams.streams)))
    manager.start(cfg())
    streams.streams[0].active = False  # unplugged: the worker notices and fails the session
    assert wait_until(lambda: manager.stats()["running"] is False)
    manager.refresh_devices()
    assert order == [True]
    assert "disconnected" in manager.stats()["error"]


def test_start_accepts_a_trained_library_voice(tmp_path):
    from tests.test_pretrained import add_library_voice
    voice = add_library_voice(tmp_path, "Mine", None)
    built = []
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: built.append(v) or Identity(), FakeStreams())
    try:
        manager.start(cfg(voice=voice.id))
        assert manager.stats()["running"] is True
        assert [(v.id, v.name, v.sample_rate) for v in built] == [(voice.id, "Mine", 40000)]
    finally:
        manager.stop()
