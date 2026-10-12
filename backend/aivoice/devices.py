from dataclasses import dataclass
from typing import Callable

import numpy as np

VIRTUAL_MIC_NAMES = ("voicemeeter input", "voicemeeter aux input", "cable input")


class DeviceError(ValueError):
    pass


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    max_input: int
    max_output: int


def _sd(sd):
    if sd is None:
        import sounddevice

        return sounddevice
    return sd


def list_devices(sd=None) -> list[Device]:
    sd = _sd(sd)
    wasapi = next((i for i, api in enumerate(sd.query_hostapis()) if "WASAPI" in api["name"]), None)
    return [
        Device(d["index"], d["name"], d["max_input_channels"], d["max_output_channels"])
        for d in sd.query_devices()
        if d["hostapi"] == wasapi
    ]


def find_virtual_mic(devices: list[Device]) -> Device | None:
    for device in devices:
        if device.max_output > 0 and any(n in device.name.lower() for n in VIRTUAL_MIC_NAMES):
            return device
    return None


def get_device(devices: list[Device], index: int) -> Device:
    for device in devices:
        if device.index == index:
            return device
    raise DeviceError(f"Audio device {index} is no longer available")


class SoundDeviceStreams:
    """Opens WASAPI streams. auto_convert lets Windows resample devices that are not 48 kHz."""

    def __init__(self, sd=None):
        self._sd = _sd(sd)

    def input(self, device: int, samplerate: int, on_audio: Callable[[np.ndarray], None]):
        def callback(indata, frames, time, status):
            on_audio(indata.copy())

        return self._sd.InputStream(
            device=device, samplerate=samplerate, channels=1, dtype="float32", latency="low",
            callback=callback, extra_settings=self._sd.WasapiSettings(auto_convert=True),
        )

    def output(self, device: int, samplerate: int, fill: Callable[[int], np.ndarray]):
        def callback(outdata, frames, time, status):
            outdata[:] = fill(frames)[:, None]

        return self._sd.OutputStream(
            device=device, samplerate=samplerate, channels=2, dtype="float32", latency="low",
            callback=callback, extra_settings=self._sd.WasapiSettings(auto_convert=True),
        )
