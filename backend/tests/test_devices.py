import numpy as np
import pytest

from aivoice.devices import Device, DeviceError, SoundDeviceStreams, find_virtual_mic, get_device, list_devices


class FakeSd:
    def __init__(self):
        self.opened = []

    def query_hostapis(self):
        return ({"name": "MME"}, {"name": "Windows WASAPI"})

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
