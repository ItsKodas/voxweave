import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soxr

from aivoice.audiofile import write_wav

SAMPLE_RATE = 48000
TRAIN_RATE = 16000
MAX_NAME = 60
MAX_TEXT = 500
log = logging.getLogger(__name__)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SOURCES = ("recorded", "imported")
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
    _replace(tmp, path)


def _replace(src: Path, dst: Path) -> None:
    """os.replace, retried briefly: Windows refuses while antivirus or an indexer has the file open."""
    for attempt in range(8):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace(tmp, path)


class Library:
    """Voices and their clips on disk. Each voice's voice.json is the source of truth."""

    def __init__(self, data: Path):
        self._data = data
        self._root = data / "voices"
        self._lock = _LOCK

    def _dir(self, voice_id) -> Path:
        if not isinstance(voice_id, str) or not ID_RE.match(voice_id):
            raise LibraryError("Unknown voice")
        return self._root / voice_id

    def _load(self, voice_id) -> Voice:
        folder = self._dir(voice_id)
        try:
            raw = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
            clips = [Clip(**c) for c in raw["clips"]]
            floor = max((c.index for c in clips), default=0) + 1
            return Voice(voice_id, str(raw["name"]), str(raw["created"]), clips, max(int(raw.get("next_index", 1)), floor))
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
                {"id": v.id, "name": v.name, "created": v.created, "minutes": v.minutes, "clips": len(v.clips), "models": []}
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
                try:
                    voices.append(self._load(folder.name))
                except LibraryError:
                    continue  # damaged or foreign folder: leave it alone
            return sorted(voices, key=lambda v: (v.created, v.id))

    def get(self, voice_id) -> Voice:
        with self._lock:
            return self._load(voice_id)

    def create(self, name: str) -> Voice:
        name = (name or "").strip()
        if not name or len(name) > MAX_NAME or any(ord(ch) < 32 for ch in name):
            raise LibraryError(f"A voice name must be 1 to {MAX_NAME} characters on one line")
        with self._lock:
            while True:
                voice_id = f"{_slug(name)}-{secrets.token_hex(3)}"
                try:
                    (self._root / voice_id / "clips").mkdir(parents=True)
                    break
                except FileExistsError:
                    continue
            seconds, fraction = divmod(_now_ns(), 1_000_000_000)
            created = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(seconds)) + f".{fraction:09d}"
            voice = Voice(voice_id, name, created)
            self._save(voice)
            return voice

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
        with self._lock:
            voice = self._load(voice_id)
            index = voice.next_index
            clips = self._dir(voice_id) / "clips"
            clips.mkdir(exist_ok=True)
            stem = f"{index:04d}"
            # Audio first, record last: a crash leaves stray files, which the next save overwrites.
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
            self._save(voice)  # the record first: voice.json is the truth, the .txt is a convenience copy
            txt = self._dir(voice_id) / "clips" / f"{index:04d}.txt"
            try:
                _atomic_text(txt, text)
            except OSError as exc:
                log.warning("Saved the new transcript but could not update %s: %s", txt.name, exc)
            return clip

    def delete_clip(self, voice_id, index: int) -> None:
        with self._lock:
            voice = self._load(voice_id)
            voice.clips.remove(self._find(voice, index))
            self._save(voice)
            for suffix in (".orig.wav", ".wav", ".txt"):
                path = self._dir(voice_id) / "clips" / f"{index:04d}{suffix}"
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("Could not remove %s (left behind, no longer part of the voice): %s", path.name, exc)

    def clip_audio_path(self, voice_id, index: int) -> Path:
        with self._lock:
            self._find(self._load(voice_id), index)
            path = self._dir(voice_id) / "clips" / f"{index:04d}.orig.wav"
            if not path.is_file():
                raise LibraryError("The audio for this clip is missing")
            return path
