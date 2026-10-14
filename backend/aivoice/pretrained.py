import json
from dataclasses import dataclass
from pathlib import Path

from aivoice.library import Library

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
        if voice.rvc_sample_rate is None or voice.id in taken:
            continue
        folder = data / "voices" / voice.id / "models" / "rvc"
        index = folder / "model.index"
        voices.append(PretrainedVoice(voice.id, voice.name, folder / "model.onnx",
                                      index if index.is_file() else None, voice.rvc_sample_rate))
        voices.extend(_snapshots(data / "voices" / voice.id / "models", voice.id, voice.name,
                                 index if index.is_file() else None, voice.rvc_sample_rate))
    return voices


SNAPSHOT = "@e"  # a version of a voice kept at an earlier epoch: "<voice id>@e100"


def base_voice_id(voice_id: str) -> str:
    """The library voice a live voice belongs to: "pokimane-8c240a@e100" belongs to "pokimane-8c240a"."""
    return voice_id.split(SNAPSHOT, 1)[0]


def _snapshots(models: Path, voice_id: str, name: str, index: Path | None, sample_rate: int) -> list[PretrainedVoice]:
    """The versions of a voice kept at earlier epochs of its training (models/rvc-e<epoch>/), lowest epoch first.
    They use the final model's search index."""
    found = []
    try:
        folders = list(models.glob("rvc-e*"))
    except OSError:
        return []
    for folder in folders:
        epoch = folder.name[len("rvc-e"):]
        if not epoch.isdigit() or not (folder / "model.onnx").is_file():
            continue
        found.append((int(epoch), PretrainedVoice(f"{voice_id}{SNAPSHOT}{int(epoch)}", f"{name} (epoch {int(epoch)})",
                                                  folder / "model.onnx", index, sample_rate)))
    return [voice for _, voice in sorted(found, key=lambda item: item[0])]


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
