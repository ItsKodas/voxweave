import threading
from pathlib import Path

import soxr

from aivoice.audiofile import AudioFileError, read_audio
from aivoice.dataset import level_problem, split_on_silence
from aivoice.library import SAMPLE_RATE, TRAIN_RATE, Library, LibraryError

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}
MAX_FILES = 200


class _StopJob(Exception):
    """Ends the whole job with one message, rather than failing every remaining file the same way."""


def _idle() -> dict:
    return {"active": False, "finished": False, "total_files": 0, "done_files": 0, "current": None,
            "clips_added": 0, "skipped": 0, "errors": []}


class _Job:
    """One import run. Each run owns its status and cancel flag, so a thread that is still
    finishing can never write into, or be cancelled by, the next run."""

    def __init__(self, total: int):
        self.status = {**_idle(), "active": True, "total_files": total}
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


class Importer:
    """Turns dropped audio files into transcribed clips on a background thread. One job at a time."""

    def __init__(self, library: Library, transcriber, read_audio=read_audio):
        self._library = library
        self._transcriber = transcriber
        self._read = read_audio
        self._lock = threading.Lock()  # guards _job and every job's status
        self._job: _Job | None = None

    def start(self, voice_id: str, paths: list[str]) -> None:
        self._library.get(voice_id)
        if not paths:
            raise ValueError("No files to import")
        if len(paths) > MAX_FILES:
            raise ValueError(f"Import at most {MAX_FILES} files at a time")
        if not all(isinstance(p, str) for p in paths):
            raise ValueError("Each file must be given as a path")
        with self._lock:
            # "Running" is the job's own flag, set false by the job as its last act, not
            # Thread.is_alive(), which stays true for a moment after the job is already finished.
            if self._job is not None and self._job.status["active"]:
                raise ValueError("An import is already running")
            job = _Job(len(paths))
            job.thread = threading.Thread(target=self._run, args=(job, voice_id, list(paths)), daemon=True)
            job.thread.start()
            self._job = job  # only once the thread runs, so a failed start cannot leave a stuck active job

    def cancel(self) -> None:
        with self._lock:
            job = self._job
        if job is not None:
            job.cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._job.thread if self._job is not None else None
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict:
        with self._lock:
            status = self._job.status if self._job is not None else _idle()
            return {**status, "errors": list(status["errors"])}

    def _set(self, job: _Job, **changes) -> None:
        with self._lock:
            job.status.update(changes)

    def _bump(self, job: _Job, key: str) -> None:
        with self._lock:
            job.status[key] += 1

    def _error(self, job: _Job, message: str) -> None:
        with self._lock:
            job.status["errors"].append(message)

    def _check_voice(self, voice_id: str) -> None:
        try:
            self._library.get(voice_id)
        except LibraryError:
            raise _StopJob("The voice was deleted during the import") from None

    def _import_file(self, job: _Job, voice_id: str, path: Path) -> None:
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise AudioFileError(f"{path.name} is not a supported audio file")
        if not path.is_file():
            raise AudioFileError(f"{path.name} was not found or is not a file")
        pieces = split_on_silence(self._read(path, SAMPLE_RATE), SAMPLE_RATE)
        if not pieces:
            raise AudioFileError(f"{path.name}: no speech was found")
        for piece in pieces:
            if job.cancel.is_set():
                return
            # Whisper invents words for near-silence, so quiet pieces are filtered by level
            # before transcribing, never by looking for an empty transcript afterwards.
            if level_problem(piece) == "too_quiet":
                self._bump(job, "skipped")
                continue
            try:
                text = self._transcriber.transcribe(soxr.resample(piece, SAMPLE_RATE, TRAIN_RATE))
            except Exception as exc:  # noqa: BLE001
                raise _StopJob(f"Transcription failed: {exc}") from exc
            try:
                self._library.add_clip(voice_id, piece, text, "imported", origin=path.name)
            except Exception:
                self._check_voice(voice_id)  # a deleted voice ends the job; any other cause is this file's
                raise
            self._bump(job, "clips_added")

    def _run(self, job: _Job, voice_id: str, paths: list[str]) -> None:
        try:
            for raw in paths:
                if job.cancel.is_set():
                    break
                path = Path(raw)
                self._set(job, current=path.name)
                self._check_voice(voice_id)
                try:
                    self._import_file(job, voice_id, path)
                except _StopJob:
                    raise
                except AudioFileError as exc:
                    self._error(job, str(exc))
                except Exception as exc:  # noqa: BLE001 - one bad file must not end the job
                    self._error(job, f"{path.name}: {exc}")
                if job.cancel.is_set():
                    break
                self._bump(job, "done_files")
        except _StopJob as exc:
            self._error(job, str(exc))
        except Exception as exc:  # noqa: BLE001 - a thread must never die with a traceback
            self._error(job, f"Import failed: {exc}")
        finally:
            self._set(job, active=False, finished=True, current=None)
