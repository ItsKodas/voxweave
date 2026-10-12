from dataclasses import dataclass
from pathlib import Path

from aivoice.config import dml_device_id
from aivoice.devices import Device, find_virtual_mic
from aivoice.pretrained import list_voices, vec_model_path


@dataclass(frozen=True)
class Check:
    id: str
    ok: bool
    message: str


def _settings_check(data: Path) -> Check:
    try:
        dml_device_id(data)
    except ValueError as exc:
        return Check("settings", False, f"{exc}. Fix or delete {data / 'settings.json'}; "
                                        "live mode cannot start until then.")
    return Check("settings", True, "Settings are valid.")


def run_checks(data: Path, devices: list[Device], providers: list[str]) -> list[Check]:
    dml = "DmlExecutionProvider" in providers
    mic = find_virtual_mic(devices)
    vec = vec_model_path(data)
    voices = list_voices(data)
    return [
        Check("directml", dml, "GPU acceleration (DirectML) is available." if dml
              else "DirectML is not available. Reinstall onnxruntime-directml in the backend environment."),
        Check("virtual_mic", mic is not None, f"Virtual mic found: {mic.name}." if mic
              else "No virtual mic found. Start VoiceMeeter, then choose 'VoiceMeeter Output' as the "
                   "microphone in your other apps. Monitor-only use still works."),
        Check("base_model", vec.is_file(), "Base speech model is installed." if vec.is_file()
              else f"Base speech model is missing. Download vec-768-layer-12.onnx into {vec.parent}."),
        Check("voice", bool(voices), f"{len(voices)} voice model(s) installed." if voices
              else "No voice models installed. Export one with tools/export_rvc_onnx.py (see README)."),
        _settings_check(data),
    ]
