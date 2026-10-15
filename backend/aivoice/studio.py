import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from aivoice.dataset import PROBLEM_MESSAGES, clip_problem, level_problem, trim_silence
from aivoice.devices import Device, get_device
from aivoice.importer import Importer
from aivoice.library import SAMPLE_RATE, Library, LibraryError
from aivoice.audiofile import write_wav
from aivoice.prompts import load_pages, load_paragraphs, load_prompts, next_prompt
from aivoice.recorder import Recorder

log = logging.getLogger(__name__)

PAGE_MAX_SECONDS = 15 * 60  # a page takes two to four minutes to read
PAGE_MIN_SECONDS = 20.0
PAGE_WORDS_PER_MINUTE = 150  # reading aloud
KEEP_PAGE_RECORDINGS = 10  # the newest page recordings are kept, so a read can be imported again
PAGE_TOO_SHORT = "That was too short to be a page. Read the whole page aloud from the top, then press Stop."
PAGE_TOO_QUIET = "Almost nothing was heard. Check that the right microphone is chosen and read the page aloud."


class Studio:
    """The guided recording flow: which sentence is next, taking a take, and judging it."""

    def __init__(self, library: Library, recorder: Recorder, importer: Importer,
                 list_devices: Callable[[], list[Device]], prompts: tuple[str, ...] | None = None,
                 paragraphs: tuple[str, ...] | None = None,
                 pages: tuple[tuple[str, str], ...] | None = None, recordings: Path | None = None,
                 sample_rate: int = SAMPLE_RATE):
        self._library = library
        self._recorder = recorder
        self._importer = importer
        self._list_devices = list_devices
        # Sentences first, then the longer paragraphs: a clip remembers its prompt by position in this list, so
        # the sentences keep the positions they always had.
        if paragraphs is None:
            paragraphs = load_paragraphs() if prompts is None else ()
        sentences = prompts if prompts is not None else load_prompts()
        self._long_from = len(sentences)
        self._prompts = tuple(sentences) + tuple(paragraphs)
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._take: tuple[str, int] | None = None
        # Full pages, read in one take and sliced into clips by the importer. Their recordings are kept here.
        self._pages = tuple(pages) if pages is not None else load_pages()
        self._recordings = recordings
        self._page_take: tuple[str, int] | None = None

    @property
    def recording(self) -> bool:
        return self._recorder.recording

    def prompt(self, voice_id, after: int | None = None, kind: str | None = None) -> dict:
        """The next thing to read. kind "sentences" or "paragraphs" keeps to that part of the list and adds
        "kind" and "number" (the position within it, from 1); without a kind the whole list is one sequence."""
        voice = self._library.get(voice_id)
        recorded = {c.prompt for c in voice.clips if c.prompt is not None}
        if kind is None:
            index = next_prompt(recorded, len(self._prompts), after)
            return {"index": index, "text": None if index is None else self._prompts[index],
                    "total": len(self._prompts), "recorded": len(recorded & set(range(len(self._prompts))))}
        if kind not in ("sentences", "paragraphs"):
            raise ValueError("kind must be sentences or paragraphs")
        first, end = (0, self._long_from) if kind == "sentences" else (self._long_from, len(self._prompts))
        done = {i - first for i in recorded if first <= i < end}
        local = next_prompt(done, end - first, None if after is None else after - first)
        index = None if local is None else first + local
        return {"index": index, "text": None if index is None else self._prompts[index], "total": end - first,
                "recorded": len(done), "kind": kind, "number": None if local is None else local + 1}

    def pages(self) -> list[dict]:
        out = []
        for index, (title, text) in enumerate(self._pages):
            words = len(text.split())
            out.append({"index": index, "title": title, "text": text, "words": words,
                        "minutes": round(words / PAGE_WORDS_PER_MINUTE, 1)})
        return out

    def start_page(self, voice_id, device: int, page: int) -> None:
        """Starts reading a full page: one long take, sliced into clips by the importer when it stops."""
        with self._lock:
            self._library.get(voice_id)
            if not isinstance(page, int) or not 0 <= page < len(self._pages):
                raise ValueError("Unknown page")
            if self._importer.status().get("active"):
                raise ValueError("Wait for the import to finish before reading a page: the page is imported too")
            source = get_device(self._list_devices(), device)
            if source.max_input < 1:
                raise ValueError(f"{source.name} is not an input device")
            self._recorder.start(device, max_seconds=PAGE_MAX_SECONDS)
            self._take, self._page_take = None, (voice_id, page)

    def _finish_page(self, voice_id: str, page: int, audio) -> dict:
        trimmed = trim_silence(audio, self._sr)
        seconds = len(trimmed) / self._sr
        if len(trimmed) == 0 or level_problem(trimmed) == "too_quiet":
            return {"accepted": False, "reason": "too_quiet", "message": PAGE_TOO_QUIET, "clip": None}
        if seconds < PAGE_MIN_SECONDS:
            return {"accepted": False, "reason": "too_short", "message": PAGE_TOO_SHORT, "clip": None}
        if self._recordings is None:
            raise ValueError("Reading a page is not available")
        self._recordings.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self._recordings / f"{voice_id}-page{page + 1}-{stamp}.wav"
        write_wav(path, trimmed, self._sr)
        self._importer.start(voice_id, [str(path)])
        old = sorted(self._recordings.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
        for stale in old[KEEP_PAGE_RECORDINGS:]:
            try:
                stale.unlink()
            except OSError:
                pass  # removed next time
        return {"accepted": True, "imported": True, "seconds": round(seconds, 1), "reason": None, "message": None,
                "clip": None}

    def start_recording(self, voice_id, device: int, prompt: int) -> None:
        with self._lock:
            self._library.get(voice_id)
            if not isinstance(prompt, int) or not 0 <= prompt < len(self._prompts):
                raise ValueError("Unknown sentence")
            source = get_device(self._list_devices(), device)
            if source.max_input < 1:
                raise ValueError(f"{source.name} is not an input device")
            self._recorder.start(device)  # raises if already recording, before the current take is touched
            self._take, self._page_take = (voice_id, prompt), None

    def stop_recording(self) -> dict:
        with self._lock:
            take, self._take = self._take, None
            page_take, self._page_take = self._page_take, None
            # The recorder is idle once this returns, whatever happens below: nothing after it touches the recorder.
            audio = self._recorder.stop()
            if page_take is not None:
                return self._finish_page(*page_take, audio)
            if take is None:  # not reachable through the API: the recorder only starts via start_recording
                raise ValueError("Not recording")
            voice_id, prompt = take
            trimmed = trim_silence(audio, self._sr)
            problem = clip_problem(trimmed, self._sr)
            if problem is not None:
                return {"accepted": False, "reason": problem, "message": PROBLEM_MESSAGES[problem], "clip": None}
            earlier = [c.index for c in self._library.get(voice_id).clips if c.prompt == prompt]
            clip = self._library.add_clip(voice_id, trimmed, self._prompts[prompt], "recorded", prompt=prompt)
            for index in earlier:  # a redo replaces the earlier take, but only once the new one is saved
                try:
                    self._library.delete_clip(voice_id, index)
                except LibraryError as exc:  # the new take is saved, so the take is still accepted
                    log.warning("Saved the new take but could not remove the earlier take (clip %s): %s", index, exc)
            return {"accepted": True, "reason": None, "message": None, "clip": asdict(clip)}

    def cancel_recording(self) -> None:
        with self._lock:
            self._recorder.cancel()
            self._take = self._page_take = None

    def status(self) -> dict:
        return {**self._recorder.status(), "import": self._importer.status()}
