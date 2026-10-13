"""The /tts routes, and text to speech's side of "one activity at a time".

Lock order, shared with /live, /library and /training: the audio lock, then the voices lock, then one manager's
own lock (training's or text to speech's, never one inside the other), then the studio's, then the library's. The
text-to-speech manager's lock is only ever taken last of these: the manager calls nothing of live mode, training
or the studio while it holds it.

Speak checks "nothing else is using the sound devices or the GPU" and starts the request under the audio lock,
which is where live start, recording start and training start make the opposite check, so the two can never pass
each other. Status, voices, environment and stop take neither shared lock.

Nothing that can end a process (the manager's status, stop and unload) is ever called while the audio lock or
the voices lock is held, with one exception no route can avoid: speak itself looks once more for a worker that
has just died, as training's start does for its own.
"""
import threading
from contextlib import AbstractContextManager
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from aivoice import tts_reference, tts_text
from aivoice.api_library import same_app_only_for
from aivoice.library import ID_RE, Library, LibraryError
from aivoice.tts import ALREADY, NOTHING_TO_SAY, STEPS, TtsManager

SPEAK_WHILE_LIVE = "Stop live mode before speaking"
SPEAK_WHILE_RECORDING = "Stop recording before speaking"
SPEAK_WHILE_TRAINING = "Training is running. Wait for it to finish or cancel it before speaking"
MISSING_AUDIO = "The audio of clip {}, this voice's reference clip, is missing. Delete that clip or record it again."


class SpeakBody(BaseModel):
    # Left untyped on purpose: the route and the manager judge them, so a bad value is a 400 with a readable
    # message, not a 422. Fields that are not listed are ignored, as in every other body of this app; there is
    # no speed setting (the manager speaks at one speed).
    voice_id: object = None
    text: object = None
    preset: object = None
    mic_device: object = None
    monitor_device: object = None
    save: object = False


def _device(value, name: str) -> int | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{name} must be a device number")
    return value


def tts_router(tts: TtsManager, library: Library | None, studio, training, live_running: Callable[[], bool],
               exclusive: AbstractContextManager, voices_lock: AbstractContextManager,
               training_starting: threading.Event, allowed_origins: list[str],
               monitor_gain: Callable[[], float] | None = None) -> APIRouter:
    """`exclusive` and `voices_lock` are the audio lock and the voices lock shared with /live, /library and
    /training, taken in that order and only by speak. `training_starting` is set, under the audio lock, while a
    training start is unloading the text-to-speech worker: nothing may be spoken from then on. `monitor_gain()`
    is the listen volume live mode was last given: speech is heard on the listen device at the same volume."""
    router = APIRouter(prefix="/tts", dependencies=[Depends(same_app_only_for(allowed_origins))])

    def refuse_if_busy() -> None:
        """A 409 saying what has to stop first. Reads that only take a manager's own lock: nothing here ever
        ends a process, so it may be asked under the audio lock."""
        if tts.is_active():
            reason = ALREADY
        elif live_running():
            reason = SPEAK_WHILE_LIVE
        elif studio is not None and studio.recording:
            reason = SPEAK_WHILE_RECORDING
        elif training is not None and (training.is_active() or training_starting.is_set()):
            reason = SPEAK_WHILE_TRAINING
        else:
            return
        raise HTTPException(status_code=409, detail=reason)

    @router.get("/status")
    def status() -> dict:
        return tts.status()

    @router.get("/environment")
    def environment(refresh: bool = False) -> dict:
        # The full check starts a GPU process. Beside a training run or live mode it could disturb them, so then
        # the answer is what is already known (the manager itself never probes beside its own request).
        # A training start that is unloading the worker is about to put training on the GPU: that counts too.
        if training is not None and (training.is_active() or training_starting.is_set()):
            return {**tts.environment(probe=False), "checked_while_training": True}
        if live_running():
            return {**tts.environment(probe=False), "checked_while_live": True}
        return tts.environment(refresh)

    @router.get("/voices")
    def voices() -> list[dict]:
        listed = []
        for voice in library.list() if library is not None else []:
            # The level of the clip counts: one too quiet to copy is passed over, and a voice with no other is
            # not usable, with the reason. Reads a clip's audio once; after that only its size and date.
            found = tts_reference.describe(voice, tts_reference.library_levels(library, voice.id))
            reference = {"ok": found["ok"], "seconds": found["seconds"], "message": found["message"]}
            if found["ok"]:
                try:  # speak refuses a reference clip whose audio file is gone, so it is not called usable here
                    library.clip_audio_path(voice.id, found["clip"])
                except LibraryError:
                    reference = {"ok": False, "seconds": None, "message": MISSING_AUDIO.format(found["clip"])}
            listed.append({"id": voice.id, "name": voice.name, "has_model": voice.rvc_sample_rate is not None,
                           "reference": reference})
        return listed

    def refuse_what_cannot_be_spoken(body: SpeakBody) -> None:
        """The refusals that need no look at the GPU or the devices, in the manager's own words and before the
        environment check can start a process for a request that could never be carried out. The manager judges
        all of it again, and the rest (the reference clip, the model, the devices), when the request starts."""
        text = tts_text.clean(body.text if isinstance(body.text, str) else "")
        if not tts_text.split(text, 4 * tts_text.MAX_TEXT):
            raise ValueError(NOTHING_TO_SAY)
        if not isinstance(body.preset, str):
            raise ValueError("Unknown preset. Choose fast, balanced or best.")
        if body.preset not in STEPS:
            raise ValueError(f"Unknown preset: {body.preset}. Choose fast, balanced or best.")
        if not isinstance(body.voice_id, str) or not ID_RE.match(body.voice_id):
            raise ValueError("Unknown voice")
        if library is not None:
            try:
                library.get(body.voice_id)
            except LibraryError:
                raise ValueError("Unknown voice") from None

    @router.post("/speak")
    def speak(body: SpeakBody) -> dict:
        refuse_what_cannot_be_spoken(body)
        mic = _device(body.mic_device, "mic_device")
        monitor = _device(body.monitor_device, "monitor_device")
        if not isinstance(body.save, bool):
            raise ValueError("save must be true or false")
        # Asked again under the audio lock below; here it keeps the checks that follow off a busy GPU.
        refuse_if_busy()
        # Ends a worker that died or sat idle too long, here where no lock is held, so that the manager finds
        # nothing to end when it looks again inside speak.
        tts.status()
        # The full environment check is a slow subprocess: run it before any lock is taken. A passing report is
        # remembered, so the manager's own check inside speak is then the quick one, or none with a worker loaded.
        report = tts.environment()
        # "unknown" means another request got in first: the refusal under the lock below says so.
        if not report["ok"] and not report.get("unknown"):
            failing = "; ".join(c["message"] for c in report["checks"] if not c["ok"])
            raise ValueError(f"Text to speech is not set up: {failing}")
        with exclusive:
            refuse_if_busy()
            with voices_lock:  # so a delete of this voice cannot slip in between its check and the start
                return tts.speak(body.voice_id, body.text, body.preset, mic_device=mic, monitor_device=monitor,
                                 save=body.save, monitor_gain=monitor_gain() if monitor_gain else 1.0)

    @router.post("/stop")
    def stop() -> dict:
        return tts.stop()

    return router
