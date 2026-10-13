"""Voice packs: one file that carries a trained voice from one computer to another.

A pack is a zip archive with the extension `.voxweave`. It holds the voice model, its index when there is one,
a small manifest, and, if the person exporting chose so, one short reference clip with its transcript so that
text to speech works on the other computer too. It never holds the rest of the recordings.

Importing treats the file as untrusted: only the member names listed here are ever read, each up to a fixed
size, nothing is unpacked by name onto disk, and the model must load as an ONNX graph before a voice is made.
ONNX files are data, not programs: loading one runs no code from the file.
"""
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

from aivoice.audiofile import read_wav, replace_file
from aivoice.config import SAMPLE_RATE
from aivoice.library import MAX_NAME, MAX_TEXT, Library, LibraryError
from aivoice.pretrained import list_voices

EXTENSION = ".voxweave"
FORMAT = "voxweave-voice"
VERSION = 1
MANIFEST, MODEL, INDEX, REFERENCE, REFERENCE_TEXT = "manifest.json", "model.onnx", "model.index", "reference.wav", "reference.txt"
LIMITS = {MANIFEST: 64 * 1024, MODEL: 1024 ** 3, INDEX: 4 * 1024 ** 3, REFERENCE: 32 * 1024 ** 2,
          REFERENCE_TEXT: 16 * 1024}
MAX_PACK_BYTES = 6 * 1024 ** 3
MAX_REFERENCE_SECONDS = 20.0
CHUNK = 1024 * 1024


class VoicePackError(ValueError):
    pass


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"


def loads_as_onnx(path: Path) -> bool:
    """Whether the file is an ONNX graph that the runtime accepts. Runs on the CPU; nothing from the file runs."""
    try:
        import onnxruntime

        onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return True
    except Exception:  # noqa: BLE001 - anything at all means "not a usable model"
        return False


def _free_name(folder: Path, stem: str) -> Path:
    for n in range(1, 1000):
        candidate = folder / (stem + ("" if n == 1 else f"-{n}") + EXTENSION)
        if not candidate.exists():
            return candidate
    raise VoicePackError("There are too many exports of this voice in the folder already")


def export_voice(data: Path, library: Library, voice_id: str, folder: Path, include_reference: bool = True,
                 choose_reference: Callable | None = None, now: Callable[[], float] = time.time) -> Path:
    """Writes the voice's model as a pack into folder and returns the file. Existing files are never replaced."""
    voice = next((v for v in list_voices(data) if v.id == voice_id), None)
    if voice is None or not voice.model_path.is_file():
        raise VoicePackError("This voice has no trained model to export. Train it first.")
    manifest = {"format": FORMAT, "version": VERSION, "name": voice.name, "sample_rate": voice.sample_rate,
                "has_index": voice.index_path is not None, "created": int(now()), "reference": None}
    reference = None
    if include_reference and choose_reference is not None:
        try:
            clip = choose_reference(library.get(voice_id))
            audio_path = library.clip_audio_path(voice_id, clip.index)
            if clip.text.strip() and clip.seconds <= MAX_REFERENCE_SECONDS:
                reference = (audio_path, clip.text.strip())
                manifest["reference"] = {"seconds": clip.seconds}
        except ValueError:
            reference = None  # no clip that would do: the pack simply has no reference
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target = _free_name(folder, _slug(voice.name))
        part = target.with_name(target.name + ".part")
        try:
            # Stored, not compressed: model weights do not compress, and a 400 MB index would only take longer.
            with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as pack:
                pack.writestr(MANIFEST, json.dumps(manifest, indent=2))
                pack.write(voice.model_path, MODEL)
                if voice.index_path is not None:
                    pack.write(voice.index_path, INDEX)
                if reference is not None:
                    pack.write(reference[0], REFERENCE)
                    pack.writestr(REFERENCE_TEXT, reference[1])
            replace_file(part, target)
        finally:
            part.unlink(missing_ok=True)
    except OSError as exc:
        raise VoicePackError(f"The voice could not be exported: {exc.strerror or exc}") from exc
    return target


def _read_small(pack: zipfile.ZipFile, name: str) -> bytes:
    info = pack.getinfo(name)
    if info.file_size > LIMITS[name]:
        raise VoicePackError("This file is not a voice pack this app can read")
    with pack.open(info) as source:
        data = source.read(LIMITS[name] + 1)
    if len(data) > LIMITS[name]:
        raise VoicePackError("This file is not a voice pack this app can read")
    return data


def _copy_member(pack: zipfile.ZipFile, name: str, target: Path) -> None:
    """Copies one member to target, stopping at its size limit whatever the archive claims about its size."""
    written = 0
    with pack.open(pack.getinfo(name)) as source, open(target, "wb") as out:
        while chunk := source.read(CHUNK):
            written += len(chunk)
            if written > LIMITS[name]:
                raise VoicePackError("This file is not a voice pack this app can read")
            out.write(chunk)


def _manifest(pack: zipfile.ZipFile) -> dict:
    try:
        manifest = json.loads(_read_small(pack, MANIFEST).decode("utf-8"))
    except (KeyError, ValueError):
        raise VoicePackError("This file is not a voice pack this app can read") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise VoicePackError("This file is not a voice pack this app can read")
    if manifest.get("version") != VERSION:
        raise VoicePackError("This voice pack was made by a newer version of the app. Update the app to import it.")
    name, rate = manifest.get("name"), manifest.get("sample_rate")
    if (not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_NAME
            or any(ord(ch) < 32 for ch in name)):
        raise VoicePackError("This voice pack has no usable name")
    if not isinstance(rate, int) or isinstance(rate, bool) or not 8000 <= rate <= 192000:
        raise VoicePackError("This voice pack does not say what sample rate its model has")
    return {"name": name.strip(), "sample_rate": rate}


def import_voice(data: Path, library: Library, path: Path, is_model: Callable[[Path], bool] = loads_as_onnx):
    """Makes a new voice in the library from a pack and returns it. On any failure nothing is left behind."""
    path = Path(path)
    try:
        if not path.is_file():
            raise VoicePackError("The voice pack was not found")
        if path.stat().st_size > MAX_PACK_BYTES:
            raise VoicePackError("This file is too large to be a voice pack")
        pack = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        raise VoicePackError("This file is not a voice pack this app can read") from None
    with pack:
        members = set(pack.namelist())
        if MODEL not in members:
            raise VoicePackError("This file is not a voice pack this app can read")
        manifest = _manifest(pack)
        try:
            voice = library.create(manifest["name"])
        except LibraryError as exc:
            raise VoicePackError(str(exc)) from exc
        folder = data / "voices" / voice.id / "models"
        staging = folder / "rvc.part"
        try:
            staging.mkdir(parents=True)
            _copy_member(pack, MODEL, staging / MODEL)
            if not is_model(staging / MODEL):
                raise VoicePackError("The model in this voice pack is not a voice model this app can load")
            if INDEX in members:
                _copy_member(pack, INDEX, staging / INDEX)
            (staging / "voice.json").write_text(
                json.dumps({"name": manifest["name"], "sample_rate": manifest["sample_rate"]}), encoding="utf-8")
            replace_file(staging, folder / "rvc")
            if REFERENCE in members and REFERENCE_TEXT in members:
                _add_reference(pack, library, voice.id, folder)
        except (VoicePackError, OSError, zipfile.BadZipFile, EOFError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            try:
                library.delete(voice.id)
            except Exception:  # noqa: BLE001 - the import already failed; say why, not this
                pass
            if isinstance(exc, VoicePackError):
                raise
            raise VoicePackError("The voice pack could not be read to the end. Is the file complete?") from exc
    return library.get(voice.id)


def _add_reference(pack: zipfile.ZipFile, library: Library, voice_id: str, scratch: Path) -> None:
    """Adds the pack's reference clip to the new voice. A reference that cannot be used is left out: the model is
    what the pack is for, and the voice works in live mode without it."""
    temp = scratch / "reference.part.wav"
    try:
        _copy_member(pack, REFERENCE, temp)
        text = _read_small(pack, REFERENCE_TEXT).decode("utf-8", "replace").strip()[:MAX_TEXT]
        audio, rate = read_wav(temp)
        if not text or rate != SAMPLE_RATE or not 1.0 <= len(audio) / rate <= MAX_REFERENCE_SECONDS:
            return
        if not np.all(np.isfinite(audio)):
            return
        library.add_clip(voice_id, audio, text, "imported", origin="voice pack")
    except (ValueError, OSError, KeyError):
        return
    finally:
        temp.unlink(missing_ok=True)

