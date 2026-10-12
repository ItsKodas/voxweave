import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from aivoice.config import SAMPLE_RATE, load_settings
from aivoice.devices import Device, get_device
from aivoice.live import LiveSession
from aivoice.presets import resolve
from aivoice.pretrained import PretrainedVoice, list_voices, vec_model_path
from aivoice.processor import StreamProcessor
from aivoice.router import Router

MODES = ("converted", "muted")
_IDLE = {
    "running": False, "mode": "off", "error": None, "infer_ms": 0.0, "latency_ms": 0.0,
    "in_level": 0.0, "out_level": 0.0, "underruns": 0, "slow_chunks": 0, "suggest_faster": False,
}


@dataclass
class StartConfig:
    input_device: int
    mic_device: int
    voice: str
    monitor_device: int | None = None
    monitor_gain: float = 1.0
    preset: str = "balanced"
    overrides: dict | None = None
    pitch_shift: float = 0.0


def _check_pitch(value: float) -> None:
    if not -24 <= value <= 24:
        raise ValueError("pitch_shift must be between -24 and 24 semitones")


def _check_gain(value: float) -> None:
    if not 0 <= value <= 2:
        raise ValueError("monitor_gain must be between 0 and 2")


def default_engine_factory(data: Path) -> Callable[[PretrainedVoice], object]:
    from aivoice.engines.rvc import RvcEngine, make_session

    raw = load_settings(data)["dml_device_id"]
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise ValueError("dml_device_id in settings.json must be an integer")
    try:
        device_id = int(raw)
    except ValueError:
        raise ValueError("dml_device_id in settings.json must be an integer") from None

    def build(voice: PretrainedVoice):
        return RvcEngine(voice.model_path, vec_model_path(data), voice.sample_rate, voice.index_path,
                         session_factory=partial(make_session, device_id=device_id))

    return build


class LiveManager:
    def __init__(self, data: Path, list_devices: Callable[[], list[Device]], engine_factory, stream_factory,
                 sample_rate: int = SAMPLE_RATE):
        self._data = data
        self._list_devices = list_devices
        self._engine_factory = engine_factory
        self._stream_factory = stream_factory
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._session: LiveSession | None = None
        self._last_error: str | None = None

    def start(self, cfg: StartConfig) -> None:
        with self._lock:
            params = resolve(cfg.preset, cfg.overrides)
            _check_pitch(cfg.pitch_shift)
            _check_gain(cfg.monitor_gain)
            devices = self._list_devices()
            source = get_device(devices, cfg.input_device)
            if source.max_input < 1:
                raise ValueError(f"{source.name} is not an input device")
            outputs = {"mic": cfg.mic_device}
            if cfg.monitor_device is not None:
                if cfg.monitor_device == cfg.mic_device:
                    raise ValueError("Monitor and virtual mic must be different devices")
                outputs["monitor"] = cfg.monitor_device
            for index in outputs.values():
                device = get_device(devices, index)
                if device.max_output < 1:
                    raise ValueError(f"{device.name} is not an output device")
            voice = next((v for v in list_voices(self._data) if v.id == cfg.voice), None)
            if voice is None:
                raise ValueError(f"Unknown voice: {cfg.voice}")

            self._stop_locked()
            self._last_error = None
            try:
                engine = self._engine_factory(voice)
                processor = StreamProcessor(engine, self._sr, params)
                engine.convert(np.zeros(processor.window, dtype=np.float32), self._sr, 0.0, params)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Could not load voice {voice.name}: {exc}") from exc
            processor.pitch_shift = cfg.pitch_shift
            router = Router(self._sr * 2)
            router.add_output("mic")
            if "monitor" in outputs:
                router.add_output("monitor", cfg.monitor_gain)
            session = LiveSession(processor, router, self._sr, self._stream_factory, cfg.input_device, outputs)
            try:
                session.start()
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Could not open audio devices: {exc}") from exc
            self._session = session

    def _stop_locked(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()
            self._last_error = None

    def _require(self) -> LiveSession:
        if self._session is None or not self._session.running:
            raise ValueError("Live mode is not running")
        return self._session

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._require().processor.mode = mode

    def update(self, pitch_shift: float | None = None, monitor_enabled: bool | None = None,
               monitor_gain: float | None = None) -> None:
        session = self._require()
        if pitch_shift is not None:
            _check_pitch(pitch_shift)
            session.processor.pitch_shift = pitch_shift
        if session.router.has("monitor"):
            if monitor_gain is not None:
                _check_gain(monitor_gain)
                session.router.set_gain("monitor", monitor_gain)
            if monitor_enabled is not None:
                session.router.set_enabled("monitor", monitor_enabled)

    def stats(self) -> dict:
        session = self._session
        if session is None:
            return {**_IDLE, "error": self._last_error}
        stats = session.stats()
        if not stats["running"]:
            self._last_error = stats["error"]
            return {**_IDLE, "error": stats["error"]}
        return stats
