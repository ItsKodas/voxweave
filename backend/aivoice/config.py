import json
import os
from pathlib import Path

SAMPLE_RATE = 48000
_DEFAULTS = {"dml_device_id": 0}


def data_dir() -> Path:
    root = os.environ.get("AIVOICE_DATA")
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
