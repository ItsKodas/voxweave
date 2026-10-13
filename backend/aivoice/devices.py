import sys
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

# A virtual cable (VB-CABLE) comes first: someone who installed one did it to have a microphone for this app,
# and it leaves VoiceMeeter's own routing alone.
VIRTUAL_MIC_NAMES = ("cable input", "voicemeeter input", "voicemeeter aux input")


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


def _wasapi(sd) -> tuple[int | None, dict | None]:
    return next(((i, api) for i, api in enumerate(sd.query_hostapis()) if "WASAPI" in api["name"]), (None, None))


def list_devices(sd=None) -> list[Device]:
    sd = _sd(sd)
    wasapi, _ = _wasapi(sd)
    return [
        Device(d["index"], d["name"], d["max_input_channels"], d["max_output_channels"])
        for d in sd.query_devices()
        if d["hostapi"] == wasapi
    ]


def default_devices(sd=None) -> dict[str, int | None]:
    """The Windows default recording and playback devices (WASAPI indices), None when there is none."""
    _, api = _wasapi(_sd(sd))

    def pick(key: str) -> int | None:
        index = api.get(key, -1) if api else -1
        return index if index >= 0 else None

    return {"default_input": pick("default_input_device"), "default_output": pick("default_output_device")}


def refresh_devices(sd=None) -> None:
    """Re-initialises PortAudio, which only enumerates devices when it starts.

    This invalidates every open stream, so it must only be called while no session is running.
    """
    sd = _sd(sd)
    sd._terminate()
    sd._initialize()


def find_virtual_mic(devices: list[Device]) -> Device | None:
    # VIRTUAL_MIC_NAMES is a priority order: try each fragment across all devices before the next.
    for fragment in VIRTUAL_MIC_NAMES:
        for device in devices:
            if device.max_output > 0 and fragment in device.name.lower():
                return device
    return None


def get_device(devices: list[Device], index: int) -> Device:
    for device in devices:
        if device.index == index:
            return device
    raise DeviceError(f"Audio device {index} is no longer available")


_com = threading.local()


def ensure_com() -> None:
    """Initialises COM on the calling thread, once. WASAPI needs it on every thread that opens or starts a
    stream; PortAudio only did it for the thread that loaded it, so a stream started from a request or worker
    thread failed with "Unanticipated host error" (-9999). COM stays initialised for the life of the thread."""
    if sys.platform != "win32" or getattr(_com, "done", False):
        return
    import ctypes

    ctypes.windll.ole32.CoInitializeEx(None, 0)  # multithreaded; "already initialised" answers are fine
    _com.done = True


class SoundDeviceStreams:
    """Opens WASAPI streams. auto_convert lets Windows resample devices that are not 48 kHz."""

    def __init__(self, sd=None, com_init: Callable[[], None] = ensure_com):
        self._sd = _sd(sd)
        self._com_init = com_init

    def _ready(self, stream):
        """The stream, with COM made ready on whichever thread starts it."""
        start = getattr(stream, "start", None)
        if start is not None:
            def start_with_com(*args, **kwargs):
                self._com_init()
                return start(*args, **kwargs)

            stream.start = start_with_com
        return stream

    def input(self, device: int, samplerate: int, on_audio: Callable[[np.ndarray], None]):
        def callback(indata, frames, time, status):
            on_audio(indata.copy())

        self._com_init()
        return self._ready(self._sd.InputStream(
            device=device, samplerate=samplerate, channels=1, dtype="float32", latency="low",
            callback=callback, extra_settings=self._sd.WasapiSettings(auto_convert=True),
        ))

    def output(self, device: int, samplerate: int, fill: Callable[[int], np.ndarray]):
        def callback(outdata, frames, time, status):
            # Fail closed: any failure in fill, or a wrong-length result, emits silence.
            try:
                audio = np.asarray(fill(frames), dtype=np.float32)
                if audio.shape != (frames,):
                    raise ValueError("fill returned the wrong number of samples")
                outdata[:] = audio[:, None]
            except Exception:
                outdata[:] = 0

        self._com_init()
        return self._ready(self._sd.OutputStream(
            device=device, samplerate=samplerate, channels=2, dtype="float32", latency="low",
            callback=callback, extra_settings=self._sd.WasapiSettings(auto_convert=True),
        ))
