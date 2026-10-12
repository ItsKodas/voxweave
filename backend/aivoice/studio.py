import logging
import threading
from dataclasses import asdict
from typing import Callable

from aivoice.dataset import PROBLEM_MESSAGES, clip_problem, trim_silence
from aivoice.devices import Device, get_device
from aivoice.importer import Importer
from aivoice.library import SAMPLE_RATE, Library, LibraryError
from aivoice.prompts import load_prompts, next_prompt
from aivoice.recorder import Recorder

log = logging.getLogger(__name__)


class Studio:
    """The guided recording flow: which sentence is next, taking a take, and judging it."""

    def __init__(self, library: Library, recorder: Recorder, importer: Importer,
                 list_devices: Callable[[], list[Device]], prompts: tuple[str, ...] | None = None,
                 sample_rate: int = SAMPLE_RATE):
        self._library = library
        self._recorder = recorder
        self._importer = importer
        self._list_devices = list_devices
        self._prompts = prompts if prompts is not None else load_prompts()
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._take: tuple[str, int] | None = None

    @property
    def recording(self) -> bool:
        return self._recorder.recording

    def prompt(self, voice_id, after: int | None = None) -> dict:
        voice = self._library.get(voice_id)
        recorded = {c.prompt for c in voice.clips if c.prompt is not None}
        index = next_prompt(recorded, len(self._prompts), after)
        return {"index": index, "text": None if index is None else self._prompts[index],
                "total": len(self._prompts), "recorded": len(recorded)}

    def start_recording(self, voice_id, device: int, prompt: int) -> None:
        with self._lock:
            self._library.get(voice_id)
            if not isinstance(prompt, int) or not 0 <= prompt < len(self._prompts):
                raise ValueError("Unknown sentence")
            source = get_device(self._list_devices(), device)
            if source.max_input < 1:
                raise ValueError(f"{source.name} is not an input device")
            self._recorder.start(device)  # raises if already recording, before the current take is touched
            self._take = (voice_id, prompt)

    def stop_recording(self) -> dict:
        with self._lock:
            take, self._take = self._take, None
            # The recorder is idle once this returns, whatever happens below: nothing after it touches the recorder.
            audio = self._recorder.stop()
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
            self._take = None

    def status(self) -> dict:
        return {**self._recorder.status(), "import": self._importer.status()}
