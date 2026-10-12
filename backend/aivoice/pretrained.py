import json
from dataclasses import dataclass
from pathlib import Path

from aivoice.library import Library, rvc_model_sample_rate

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
    """Exported voices first, then library voices that have a trained model (oldest first). Never raises."""
    voices = _exported_voices(data)
    taken = {v.id for v in voices}
    try:
        library_voices = Library(data).list()
    except Exception:  # one unreadable library must not take the exported voices down with it
        return voices
    for voice in library_voices:
        folder = data / "voices" / voice.id / "models" / "rvc"
        sample_rate = rvc_model_sample_rate(folder.parents[1])
        if "rvc" not in voice.models or sample_rate is None or voice.id in taken:
            continue
        index = folder / "model.index"
        voices.append(PretrainedVoice(voice.id, voice.name, folder / "model.onnx",
                                      index if index.is_file() else None, sample_rate))
    return voices


def _exported_voices(data: Path) -> list[PretrainedVoice]:
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
