from dataclasses import dataclass
from pathlib import Path

from aivoice.config import dml_device_id, installed_app
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


def _fixes(data: Path) -> dict:
    """How to fix each failing check. The installed app has no README, tools or backend folder to point to: its
    environment and base model are put right by the setup that runs when the app starts."""
    if installed_app():
        again = "Quit Voxweave from the tray menu and start Voxweave again: it repairs its own files."
        return {"directml": f"DirectML is not available. {again}",
                "base_model": f"Base speech model is missing. {again}",
                "voice": "No voice models yet. Bring in a trained voice as a voice pack: drop its .voxweave "
                         "file on the Voices tab. Training is not available in the installed app yet."}
    return {"directml": "DirectML is not available. Reinstall onnxruntime-directml in the backend environment.",
            "base_model": "Base speech model is missing. Download vec-768-layer-12.onnx into "
                          f"{vec_model_path(data).parent}.",
            "voice": "No voice models yet. Train one from a voice in the Library, or export one with "
                     "tools/export_rvc_onnx.py (see README)."}


def run_checks(data: Path, devices: list[Device], providers: list[str]) -> list[Check]:
    fixes = _fixes(data)
    dml = "DmlExecutionProvider" in providers
    mic = find_virtual_mic(devices)
    vec = vec_model_path(data)
    voices = list_voices(data)
    return [
        Check("directml", dml, "GPU acceleration (DirectML) is available." if dml
              else fixes["directml"]),
        Check("virtual_mic", mic is not None, f"Virtual mic found: {mic.name}." if mic
              else "No virtual mic found. Start VoiceMeeter, then choose 'VoiceMeeter Output' as the "
                   "microphone in your other apps. Monitor-only use still works."),
        Check("base_model", vec.is_file(), "Base speech model is installed." if vec.is_file()
              else fixes["base_model"]),
        Check("voice", bool(voices), f"{len(voices)} voice model(s) available." if voices
              else fixes["voice"]),
        _settings_check(data),
    ]
