import json
from dataclasses import dataclass
from pathlib import Path

VEC_FILE = "vec-768-layer-12.onnx"


@dataclass(frozen=True)
class PretrainedVoice:
    id: str
    name: str
    model_path: Path
    index_path: Path | None
    sample_rate: int


def vec_model_path(data: Path) -> Path:
    return data / "pretrained" / VEC_FILE


def list_voices(data: Path) -> list[PretrainedVoice]:
    root = data / "pretrained" / "rvc"
    if not root.is_dir():
        return []
    voices = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        model = folder / "model.onnx"
        try:
            meta = json.loads((folder / "voice.json").read_text())
            sample_rate = int(meta["sample_rate"])
            name = str(meta.get("name") or folder.name)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not model.is_file():
            continue
        index = folder / "model.index"
        voices.append(PretrainedVoice(folder.name, name, model, index if index.is_file() else None, sample_rate))
    return voices
