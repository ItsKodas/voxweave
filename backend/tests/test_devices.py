import threading

import numpy as np
import pytest

from aivoice.devices import (Device, DeviceError, SoundDeviceStreams, default_devices, find_virtual_mic, get_device,
                             list_devices, refresh_devices)


class FakeSd:
    def __init__(self, default_input=1, default_output=2):
        self.opened = []
        self.calls = []
        self.default_input, self.default_output = default_input, default_output

    def query_hostapis(self):
        return (
            {"name": "MME", "default_input_device": 0, "default_output_device": -1},
            {"name": "Windows WASAPI", "default_input_device": self.default_input,
             "default_output_device": self.default_output},
        )

    def _terminate(self):
        self.calls.append("terminate")

    def _initialize(self):
        self.calls.append("initialize")

    def query_devices(self):
        return [
            {"index": 0, "name": "Mic (MME)", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
            {"index": 1, "name": "Headset Mic", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0},
            {"index": 2, "name": "Headset", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
            {"index": 3, "name": "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", "hostapi": 1,
             "max_input_channels": 0, "max_output_channels": 8},
        ]

    def WasapiSettings(self, **kwargs):
        return ("wasapi", kwargs)

    def InputStream(self, **kwargs):
        self.opened.append(("in", kwargs))
        return kwargs

    def OutputStream(self, **kwargs):
        self.opened.append(("out", kwargs))
        return kwargs


def test_lists_only_wasapi_devices():
    devices = list_devices(FakeSd())
    assert [d.index for d in devices] == [1, 2, 3]
    assert devices[0].max_input == 1 and devices[1].max_output == 2


def test_finds_voicemeeter():
    assert find_virtual_mic(list_devices(FakeSd())).index == 3


def test_virtual_mic_absent():
    devices = [d for d in list_devices(FakeSd()) if d.index != 3]
    assert find_virtual_mic(devices) is None


def test_virtual_mic_must_be_an_output():
    sd = FakeSd()
    devices = [d for d in list_devices(sd) if d.max_output == 0]
    assert find_virtual_mic(devices) is None


def test_get_device_unknown_index():
    with pytest.raises(DeviceError, match="no longer available"):
        get_device(list_devices(FakeSd()), 99)


def test_input_stream_uses_mono_float32_and_auto_convert():
    sd = FakeSd()
    received = []
    stream = SoundDeviceStreams(sd).input(1, 48000, received.append)
    assert stream["device"] == 1 and stream["samplerate"] == 48000
    assert stream["channels"] == 1 and stream["dtype"] == "float32"
    assert stream["extra_settings"] == ("wasapi", {"auto_convert": True})
    stream["callback"](np.ones((4, 1), dtype=np.float32), 4, None, None)
    assert received[0].shape == (4, 1)


def test_output_stream_duplicates_mono_to_stereo():
    sd = FakeSd()
    stream = SoundDeviceStreams(sd).output(2, 48000, lambda n: np.full(n, 0.5, dtype=np.float32))
    assert stream["channels"] == 2
    assert stream["extra_settings"] == ("wasapi", {"auto_convert": True})
    out = np.zeros((3, 2), dtype=np.float32)
    stream["callback"](out, 3, None, None)
    assert out.tolist() == [[0.5, 0.5]] * 3


def _out(index, name):
    return Device(index, name, 0, 2)


def test_prefers_main_voicemeeter_input_over_aux_regardless_of_order():
    devices = [
        _out(26, "VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)"),
        _out(30, "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)"),
    ]
    assert find_virtual_mic(devices).index == 30


def test_falls_back_to_aux_when_only_aux_present():
    devices = [_out(26, "VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)")]
    assert find_virtual_mic(devices).index == 26


def _run_output(fill):
    stream = SoundDeviceStreams(FakeSd()).output(2, 48000, fill)
    out = np.ones((3, 2), dtype=np.float32)
    stream["callback"](out, 3, None, None)
    return out


def test_output_callback_emits_silence_when_fill_raises():
    def boom(n):
        raise RuntimeError("engine died")

    assert _run_output(boom).tolist() == [[0.0, 0.0]] * 3


def test_output_callback_emits_silence_on_wrong_length():
    assert _run_output(lambda n: np.full(n + 1, 0.5, dtype=np.float32)).tolist() == [[0.0, 0.0]] * 3


def test_default_devices_come_from_the_wasapi_host_api():
    assert default_devices(FakeSd()) == {"default_input": 1, "default_output": 2}


def test_default_devices_are_none_when_windows_has_no_default():
    assert default_devices(FakeSd(default_input=-1, default_output=-1)) == {
        "default_input": None, "default_output": None}


def test_default_devices_are_none_without_wasapi():
    class NoWasapi(FakeSd):
        def query_hostapis(self):
            return ({"name": "MME", "default_input_device": 0, "default_output_device": 0},)

    assert default_devices(NoWasapi()) == {"default_input": None, "default_output": None}


def test_refresh_devices_reinitialises_portaudio():
    sd = FakeSd()
    refresh_devices(sd)
    assert sd.calls == ["terminate", "initialize"]


class _StreamSd:
    """A sounddevice stand-in whose streams are objects with start(), as the real ones are."""

    def __init__(self, log):
        self.log = log

    def WasapiSettings(self, **kwargs):
        return kwargs

    def _stream(self, **kwargs):
        log = self.log

        class Stream:
            def start(self):
                log.append(("start", threading.get_ident()))

        return Stream()

    InputStream = OutputStream = _stream


def test_com_is_made_ready_on_the_thread_that_opens_and_on_the_thread_that_starts_a_stream():
    # Found by the first real launch: a microphone stream started from a request thread failed with
    # "Unanticipated host error" because COM was only initialised on the thread that loaded PortAudio.
    for kind in ("input", "output"):
        log = []
        streams = SoundDeviceStreams(_StreamSd(log), com_init=lambda: log.append(("com", threading.get_ident())))
        stream = getattr(streams, kind)(1, 48000, lambda *_: None)
        assert log == [("com", threading.get_ident())]
        starter = threading.Thread(target=stream.start)
        starter.start()
        starter.join(5)
        assert [what for what, _ in log] == ["com", "com", "start"]
        assert log[1][1] == log[2][1] == starter.ident != threading.get_ident()


def test_ensure_com_runs_once_per_thread_and_never_raises():
    from aivoice.devices import ensure_com

    ensure_com()
    ensure_com()
