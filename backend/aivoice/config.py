import json
import os
import re
from pathlib import Path

SAMPLE_RATE = 48000
_DEFAULTS = {"dml_device_id": 0}


def data_dir() -> Path:
    root = os.environ.get("AIVOICE_DATA")
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def installed_app() -> bool:
    """Whether this is the app put on the computer by the installer (the shell sets AIVOICE_INSTALLED=1), as
    opposed to a source checkout. The installed app has no training and no text-to-speech environment."""
    return os.environ.get("AIVOICE_INSTALLED") == "1"


def load_settings(data: Path) -> dict:
    settings = dict(_DEFAULTS)
    try:
        loaded = json.loads((data / "settings.json").read_text())
        if isinstance(loaded, dict):
            settings.update(loaded)
    except (OSError, ValueError):
        pass
    return settings


def dml_device_id(data: Path) -> int:
    """The DirectML device chosen in settings.json. Raises ValueError when it is not an integer."""
    raw = load_settings(data)["dml_device_id"]
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise ValueError("dml_device_id in settings.json must be an integer")
    try:
        return int(raw)
    except ValueError:
        raise ValueError("dml_device_id in settings.json must be an integer") from None


_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def whisper_model(data: Path) -> str:
    """The Whisper model used to transcribe imported audio. Raises ValueError for an odd value."""
    raw = load_settings(data).get("whisper_model", "small.en")
    if not isinstance(raw, str) or not _MODEL_NAME.match(raw):
        raise ValueError("whisper_model in settings.json must be a model name such as small.en")
    return raw
