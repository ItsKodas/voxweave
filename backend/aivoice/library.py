import json
import logging
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soxr

from aivoice.audiofile import replace_file, write_wav

SAMPLE_RATE = 48000
TRAIN_RATE = 16000
MAX_NAME = 60
MAX_TEXT = 500
log = logging.getLogger(__name__)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SOURCES = ("recorded", "imported")
CLIP_SUFFIXES = (".orig.wav", ".wav", ".txt")
# One lock for every Library in the process, so two instances on the same folder cannot interleave.
_LOCK = threading.RLock()


class LibraryError(ValueError):
    pass


@dataclass
class Clip:
    index: int
    text: str
    seconds: float
    source: str
    prompt: int | None = None
    origin: str | None = None


@dataclass
class Voice:
    id: str
    name: str
    created: str
    clips: list[Clip] = field(default_factory=list)
    next_index: int = 1  # persisted so a deleted clip's number is never handed out again
    rvc_sample_rate: int | None = None  # of the trained RVC model; derived from disk, never stored in voice.json

    @property
    def models(self) -> list[str]:
        return ["rvc"] if self.rvc_sample_rate is not None else []

    @property
    def minutes(self) -> float:
        return round(sum(c.seconds for c in self.clips) / 60.0, 2)


def _now() -> float:
    return time.time()


def _now_ns() -> int:
    return time.time_ns()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    replace_file(tmp, path)


atomic_json = _atomic_json  # the public name, for other modules


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    replace_file(tmp, path)


@contextmanager
def _readable(what: str):
    """A disk failure (full, read-only, file in use) inside the block becomes a LibraryError the user can read."""
    try:
        yield
    except OSError as exc:
        reason = exc.strerror or str(exc) or type(exc).__name__
        log.error("%s: %s", what, reason, exc_info=True)
        raise LibraryError(f"{what}: {reason}") from exc


def _damage(exc: Exception) -> str:
    """Why a voice.json could not be used, for the log. Never quotes the file: only where and what kind."""
    if isinstance(exc, json.JSONDecodeError):
        return f"voice.json is not valid JSON (line {exc.lineno}, column {exc.colno})"
    if isinstance(exc, OSError):
        return f"voice.json could not be read ({exc.strerror or type(exc).__name__})"
    return f"voice.json does not have the expected contents ({type(exc).__name__})"


def rvc_model_sample_rate(voice_folder: Path) -> int | None:
    """The sample rate of the trained RVC model in a voice folder, or None when it has no usable model.

    Whatever is in the model folder (missing, huge, nested, a directory, nonsense), this never raises."""
    folder = voice_folder / "models" / "rvc"
    try:
        if not (folder / "model.onnx").is_file():
            return None
        rate = json.loads((folder / "voice.json").read_text(encoding="utf-8"))["sample_rate"]
        if isinstance(rate, float) and rate.is_integer():
            rate = int(rate)
        if isinstance(rate, int) and not isinstance(rate, bool) and 8000 <= rate <= 192000:
            return rate
    except Exception:
        pass
    return None


class Library:
    """Voices and their clips on disk. Each voice's voice.json is the source of truth."""

    def __init__(self, data: Path):
        self._data = data
        self._root = data / "voices"
        self._lock = _LOCK
        self._reported: set[tuple[str, str]] = set()  # skipped folders already named in the log

    def _dir(self, voice_id) -> Path:
        if not isinstance(voice_id, str) or not ID_RE.match(voice_id):
            raise LibraryError("Unknown voice")
        return self._root / voice_id

    def _read(self, voice_id) -> Voice:
        """Raises LibraryError for an id that is not one, and the underlying error for a voice.json that is unusable."""
        folder = self._dir(voice_id)
        raw = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
        clips = [Clip(**c) for c in raw["clips"]]
        floor = max((c.index for c in clips), default=0) + 1
        return Voice(voice_id, str(raw["name"]), str(raw["created"]), clips,
                     max(int(raw.get("next_index", 1)), floor), rvc_model_sample_rate(folder))

    def _load(self, voice_id) -> Voice:
        try:
            return self._read(voice_id)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise LibraryError("Unknown voice") from None

    def _save(self, voice: Voice) -> None:
        _atomic_json(self._dir(voice.id) / "voice.json",
                     {"name": voice.name, "created": voice.created, "next_index": voice.next_index,
                      "clips": [asdict(c) for c in voice.clips]})
        self._write_summary()

    def _write_summary(self) -> None:
        # voices.json is only a derived summary: failing to refresh it must not fail (or undo) the real change.
        try:
            self._data.mkdir(parents=True, exist_ok=True)
            _atomic_json(self._data / "voices.json", [
                {"id": v.id, "name": v.name, "created": v.created, "minutes": v.minutes, "clips": len(v.clips), "models": v.models}
                for v in self.list()
            ])
        except OSError as exc:
            log.warning("Could not refresh voices.json: %s", exc)

    def list(self) -> list[Voice]:
        with self._lock:
            if not self._root.is_dir():
                return []
            voices = []
            for folder in self._root.iterdir():
                if not folder.is_dir():
                    continue
                # A damaged or foreign folder is left alone, and named in the log so it does not just vanish.
                try:
                    voices.append(self._read(folder.name))
                except LibraryError:
                    self._report_skipped(folder.name, "its name is not a voice id")
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    self._report_skipped(folder.name, _damage(exc))
            return sorted(voices, key=lambda v: (v.created, v.id))

    def _report_skipped(self, folder: str, reason: str) -> None:
        # Once per folder and reason: list() runs on every save, and a long import must not flood the log.
        if (folder, reason) not in self._reported:
            self._reported.add((folder, reason))
            log.warning("Skipping the folder %s in %s: %s. Nothing in it was changed.", folder, self._root, reason)

    def get(self, voice_id) -> Voice:
        with self._lock:
            return self._load(voice_id)

    def create(self, name: str) -> Voice:
        name = (name or "").strip()
        if not name or len(name) > MAX_NAME or any(ord(ch) < 32 for ch in name):
            raise LibraryError(f"A voice name must be 1 to {MAX_NAME} characters on one line")
        with self._lock, _readable("Could not create the voice"):
            self._root.mkdir(parents=True, exist_ok=True)
            while True:
                voice_id = f"{_slug(name)}-{secrets.token_hex(3)}"
                try:
                    (self._root / voice_id).mkdir()  # fails if the id is taken, so an existing voice is never reused
                    break
                except FileExistsError:
                    continue
            folder = self._root / voice_id
            seconds, fraction = divmod(_now_ns(), 1_000_000_000)
            created = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(seconds)) + f".{fraction:09d}"
            voice = Voice(voice_id, name, created)
            try:
                (folder / "clips").mkdir()
                self._save(voice)
            except OSError:
                self._discard_new_folder(folder)
                raise
            return voice

    @staticmethod
    def _discard_new_folder(folder: Path) -> None:
        """Removes the folder create() made a moment ago. rmdir only removes empty folders, so nothing can be lost."""
        for path, remove in ((folder / "voice.json.tmp", os.unlink), (folder / "clips", os.rmdir), (folder, os.rmdir)):
            try:
                remove(path)
            except OSError:
                pass

    def delete(self, voice_id) -> None:
        """Moves the voice to data/trash. Nothing is erased."""
        with self._lock:
            self._load(voice_id)
            trash = self._data / "trash"
            trash.mkdir(parents=True, exist_ok=True)
            target = trash / f"{voice_id}-{int(_now())}"
            for n in range(2, 1000):
                if not target.exists():
                    break
                target = trash / f"{voice_id}-{int(_now())}-{n}"
            try:
                # A same-drive rename, never shutil.move: a copy-then-delete fallback could half-erase the voice.
                os.rename(self._dir(voice_id), target)
            except OSError as exc:
                raise LibraryError("Could not move the voice to the trash; close anything using its files and try again") from exc
            self._write_summary()

    def add_clip(self, voice_id, audio: np.ndarray, text: str, source: str,
                 prompt: int | None = None, origin: str | None = None) -> Clip:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) == 0:
            raise LibraryError("There is no audio to save")
        if source not in SOURCES:
            raise LibraryError("A clip's source must be 'recorded' or 'imported'")
        with self._lock, _readable("Could not save the clip"):
            voice = self._load(voice_id)
            index = voice.next_index
            clips = self._dir(voice_id) / "clips"
            clips.mkdir(exist_ok=True)
            stem = f"{index:04d}"
            # Audio first, record last: a crash or a failed write leaves stray files that belong to no clip
            # (never listed, never served), and the next save, which gets the same number, overwrites them.
            write_wav(clips / f"{stem}.orig.wav", audio, SAMPLE_RATE)
            write_wav(clips / f"{stem}.wav", soxr.resample(audio, SAMPLE_RATE, TRAIN_RATE), TRAIN_RATE)
            text = (text or "").strip()[:MAX_TEXT]
            _atomic_text(clips / f"{stem}.txt", text)
            clip = Clip(index, text, round(len(audio) / SAMPLE_RATE, 2), source, prompt, origin)
            voice.clips.append(clip)
            voice.next_index = index + 1
            self._save(voice)
            return clip

    def _find(self, voice: Voice, index: int) -> Clip:
        clip = next((c for c in voice.clips if c.index == index), None)
        if clip is None:
            raise LibraryError("Unknown clip")
        return clip

    def update_text(self, voice_id, index: int, text: str) -> Clip:
        text = (text or "").strip()
        if len(text) > MAX_TEXT:
            raise LibraryError(f"A transcript can be at most {MAX_TEXT} characters")
        with self._lock:
            voice = self._load(voice_id)
            clip = self._find(voice, index)
            clip.text = text
            with _readable("Could not save the transcript"):
                self._save(voice)  # the record first: voice.json is the truth, the .txt is a convenience copy
            txt = self._dir(voice_id) / "clips" / f"{index:04d}.txt"
            try:
                _atomic_text(txt, text)
            except OSError as exc:
                log.warning("Saved the new transcript but could not update %s: %s", txt.name, exc)
            return clip

    def delete_clip(self, voice_id, index: int) -> None:
        """Takes the clip out of the voice and moves its files to data/trash/<voice-id>/clips. Nothing is erased."""
        with self._lock:
            voice = self._load(voice_id)
            voice.clips.remove(self._find(voice, index))
            with _readable("Could not delete the clip"):
                self._save(voice)
            self._trash_clip_files(voice_id, f"{index:04d}")

    def _trash_clip_files(self, voice_id, stem: str) -> None:
        # The record is already saved, so a file that cannot be moved is only a stray: it stays where it is,
        # whole, and is never listed, served or handed to another clip.
        clips = self._dir(voice_id) / "clips"
        trash = self._data / "trash" / voice_id / "clips"
        try:
            trash.mkdir(parents=True, exist_ok=True)
            # One name for all three files, and never one that is already in the trash.
            name = stem
            for n in range(1, 1000):
                if not any((trash / f"{name}{suffix}").exists() for suffix in CLIP_SUFFIXES):
                    break
                name = f"{stem}-{int(_now())}" + ("" if n == 1 else f"-{n}")
        except OSError as exc:
            log.warning("Could not use the trash folder %s, so the files of clip %s were left in the voice's "
                        "clips folder (no longer part of the voice): %s", trash, stem, exc)
            return
        for suffix in CLIP_SUFFIXES:
            source = clips / f"{stem}{suffix}"
            if not source.exists():
                continue
            try:
                # A same-drive rename, as for a voice: never a copy-then-delete that could lose half a file.
                # On Windows it also refuses to replace an existing file.
                os.rename(source, trash / f"{name}{suffix}")
            except OSError as exc:
                log.warning("Could not move %s to the trash (left behind, no longer part of the voice): %s",
                            source.name, exc)

    def clip_audio_path(self, voice_id, index: int) -> Path:
        with self._lock:
            self._find(self._load(voice_id), index)
            path = self._dir(voice_id) / "clips" / f"{index:04d}.orig.wav"
            if not path.is_file():
                raise LibraryError("The audio for this clip is missing")
            return path
