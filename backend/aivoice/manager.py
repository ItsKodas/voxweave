import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from aivoice import feminize
from aivoice.config import SAMPLE_RATE, dml_device_id
from aivoice.engine_guard import GuardedEngine
from aivoice.devices import Device, get_device
from aivoice.live import LiveSession
from aivoice.presets import resolve
from aivoice.pretrained import PretrainedVoice, base_voice_id, list_voices, vec_model_path
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
    mic_device: int | None  # None is listen-only: no virtual mic output at all
    voice: str
    monitor_device: int | None = None
    monitor_gain: float = 1.0
    preset: str = "balanced"
    overrides: dict | None = None
    pitch_shift: float = 0.0
    use_calibration: bool = True  # the speaker's quiet gate and input gain, when a calibration exists


def _check_pitch(value: float) -> None:
    if not -24 <= value <= 24:
        raise ValueError("pitch_shift must be between -24 and 24 semitones")


def _check_gain(value: float) -> None:
    if not 0 <= value <= 2:
        raise ValueError("monitor_gain must be between 0 and 2")


def buffer_samples(sample_rate: int, chunk: int) -> int:
    """How much converted audio each output may hold. The output runs one chunk ahead of the device, so up to two
    chunks wait there at once; a fixed two seconds was too little for chunks of a second or more, and the audio
    then cut out. Room for three chunks plus a second, and never less than the two seconds it always had."""
    return max(sample_rate * 2, 3 * chunk + sample_rate)


_RMVPE: dict = {}


def shared_rmvpe(data: Path, device_id: int):
    """The RMVPE pitch tracker, loaded once per GPU and shared by every engine; None when its model is missing,
    and the engines then use the faster, less accurate tracker."""
    from aivoice.engines.pitch import Rmvpe, rmvpe_path
    from aivoice.engines.rvc import make_session

    path = rmvpe_path(data)
    key = (str(path), device_id)
    if key not in _RMVPE:
        _RMVPE[key] = Rmvpe(make_session(str(path), device_id=device_id)) if path.is_file() else None
    return _RMVPE[key]


def default_engine_factory(data: Path) -> Callable[[PretrainedVoice], object]:
    from aivoice.engines.rvc import RvcEngine, make_session

    def build(voice: PretrainedVoice):
        # Read here, not at backend start: a bad settings.json must not stop the backend from
        # coming up, it has to surface as a Start error and in the setup check.
        device_id = dml_device_id(data)
        engine = RvcEngine(voice.model_path, vec_model_path(data), voice.sample_rate, voice.index_path,
                           session_factory=partial(make_session, device_id=device_id),
                           rmvpe=shared_rmvpe(data, device_id))
        # Live mode and text to speech both build their engine here, so neither can play a damaged model's noise.
        return GuardedEngine(engine)

    return build


class LiveManager:
    def __init__(self, data: Path, list_devices: Callable[[], list[Device]], engine_factory, stream_factory,
                 sample_rate: int = SAMPLE_RATE, refresh_devices: Callable[[], None] | None = None,
                 calibration=None):
        self._data = data
        self._list_devices = list_devices
        self._engine_factory = engine_factory
        self._stream_factory = stream_factory
        self._sr = sample_rate
        self._refresh_devices = refresh_devices
        self._calibration = calibration
        self._lock = threading.Lock()
        self._session: LiveSession | None = None
        self._last_error: str | None = None
        # The listen volume live mode was last given (at a start that succeeded, or by the slider while it ran).
        # Kept after a stop: text to speech plays to the same listen device and uses the same volume.
        self.monitor_gain = 1.0

    def start(self, cfg: StartConfig) -> None:
        with self._lock:
            params = resolve(cfg.preset, cfg.overrides)
            _check_pitch(cfg.pitch_shift)
            _check_gain(cfg.monitor_gain)
            devices = self._list_devices()
            source = get_device(devices, cfg.input_device)
            if source.max_input < 1:
                raise ValueError(f"{source.name} is not an input device")
            outputs: dict[str, int] = {}
            if cfg.mic_device is not None:
                outputs["mic"] = cfg.mic_device
            if cfg.monitor_device is not None:
                if cfg.monitor_device == cfg.mic_device:
                    raise ValueError("Monitor and virtual mic must be different devices")
                outputs["monitor"] = cfg.monitor_device
            if not outputs:
                raise ValueError("Choose a virtual mic or a listen device")
            for index in outputs.values():
                device = get_device(devices, index)
                if device.max_output < 1:
                    raise ValueError(f"{device.name} is not an output device")
            if cfg.voice == feminize.PREVIEW_ID:
                # No model: the speaker's own voice, feminized by signal processing, with the settings last saved.
                voice = None
            else:
                voice = next((v for v in list_voices(self._data) if v.id == cfg.voice), None)
                if voice is None:
                    raise ValueError(f"Unknown voice: {cfg.voice}")

            self._stop_locked()
            self._last_error = None
            try:
                if voice is None:
                    engine = GuardedEngine(feminize.FeminizeEngine(feminize.load_settings(self._data)))
                else:
                    engine = self._engine_factory(voice)
                processor = StreamProcessor(engine, self._sr, params)
                engine.convert(np.zeros(processor.window, dtype=np.float32), self._sr, 0.0, params)
            except Exception as exc:  # noqa: BLE001
                name = feminize.preview_name(feminize.load_settings(self._data)) if voice is None else voice.name
                raise ValueError(f"Could not load voice {name}: {exc}") from exc
            if voice is not None and self._calibration is not None and hasattr(engine, "pitch_range"):
                # Pitches the model never heard are moved to the nearest one its training voice used.
                try:
                    engine.pitch_range = self._calibration.voice_pitch_range(base_voice_id(voice.id))
                except Exception:  # noqa: BLE001 - without a range the engine works as before
                    engine.pitch_range = None
            if voice is not None and hasattr(engine, "speech_level_db"):
                # Background sound is judged against the speaker's calibrated level (after the calibration's gain);
                # without a calibration in use the engine learns it from the speech it hears.
                profile = self._calibration.profile() if cfg.use_calibration and self._calibration is not None else None
                engine.speech_level_db = None if profile is None else (
                    profile.level_db + 20.0 * float(np.log10(profile.input_gain)))
            if cfg.use_calibration and self._calibration is not None:
                # After the model has proved itself on the warm-up above: a closed gate would have hidden a
                # model that cannot run.
                processor.engine = self._calibration.wrap(engine)
            processor.pitch_shift = cfg.pitch_shift
            router = Router(buffer_samples(self._sr, processor.chunk))
            if "mic" in outputs:
                router.add_output("mic")
            if "monitor" in outputs:
                router.add_output("monitor", cfg.monitor_gain)
            session = LiveSession(processor, router, self._sr, self._stream_factory, cfg.input_device, outputs)
            try:
                session.start()
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Could not open audio devices: {exc}") from exc
            self._session = session
            self.monitor_gain = float(cfg.monitor_gain)

    def _stop_locked(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()
            self._last_error = None

    def refresh_devices(self) -> None:
        """Re-enumerates audio devices. Refused while live: it tears down every open stream."""
        with self._lock:
            session = self._session
            if session is not None:
                if session.running:
                    raise ValueError("Stop live mode before refreshing devices")
                # A failed session: make sure its streams are closed, but keep its error on show.
                self._last_error = session.error
                self._stop_locked()
            if self._refresh_devices is not None:
                self._refresh_devices()

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
                self.monitor_gain = float(monitor_gain)
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
