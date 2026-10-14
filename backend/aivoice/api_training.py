import threading
from contextlib import AbstractContextManager
from typing import Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from aivoice.api_library import same_app_only_for
from aivoice.studio import Studio
from aivoice.training import TrainingManager


class TrainBody(BaseModel):
    voice_id: str
    # Left untyped on purpose: the manager judges them, so a bad value is a 400 with its message, not a 422.
    epochs: object = None
    batch_size: object = None
    resume: bool = False
    # Train to a time limit: the best clips and the epochs that fit in this many minutes (see aivoice.quicktrain).
    minutes: object = None


def training_router(training: TrainingManager, studio: Studio | None, live_running: Callable[[], bool],
                    exclusive: AbstractContextManager, voices_lock: AbstractContextManager,
                    allowed_origins: list[str], tts=None,
                    training_starting: threading.Event | None = None) -> APIRouter:
    """`exclusive` and `voices_lock` are the audio lock and voices lock shared with /live and /library, taken in
    that order. Only start takes them: status and environment are polled and must never queue behind a save.

    `tts` is the text-to-speech manager, whose worker shares the GPU. `training_starting` is set under the audio
    lock while a start is unloading that worker, and /tts/speak, which reads it under the same lock, is refused
    from then until the start has succeeded (training is active) or been refused."""
    training_starting = training_starting if training_starting is not None else threading.Event()

    def refuse_if_busy() -> None:
        """Audio lock held. Reads only: nothing here ever ends a process."""
        if live_running():
            raise ValueError("Stop live mode before training")
        if studio is not None and studio.recording:
            raise ValueError("Stop recording before training")
        if tts is not None and tts.is_active():
            raise ValueError("Stop the speech before training")

    router = APIRouter(prefix="/training", dependencies=[Depends(same_app_only_for(allowed_origins))])

    @router.get("/status")
    def status(voice_id: str | None = None) -> dict:
        return training.status(voice_id)

    @router.get("/environment")
    def environment(refresh: bool = False) -> dict:
        # The full check starts a GPU process. Beside a training run or live mode it could disturb them, so
        # then the answer is what is already known (the manager itself also never probes beside a job).
        if training.is_active():
            return {**training.environment(probe=False), "checked_while_training": True}
        if live_running():
            return {**training.environment(probe=False), "checked_while_live": True}
        if tts is not None and tts.is_active():  # a worker that is loaded but idle is no obstacle
            return {**training.environment(probe=False), "checked_while_speaking": True}
        return training.environment(refresh)

    @router.get("/plan")
    def time_plan(voice_id: str, minutes: float) -> dict:
        """What a run limited to this many minutes would train on: how much of the best audio, and how many epochs."""
        from aivoice import quicktrain

        return quicktrain.plan_for(training.data, training.library, voice_id, minutes)

    @router.post("/start")
    def start(body: TrainBody) -> dict:
        if not training.running:
            if live_running():  # asked again under the audio lock below; here it keeps the check off the GPU
                raise ValueError("Stop live mode before training")
            if tts is not None and tts.is_active():  # likewise
                raise ValueError("Stop the speech before training")
            # The check is a slow subprocess: run it before any lock is taken. A passing report is remembered, so
            # the manager's own check inside start is then the quick one. (While training runs, start is refused.)
            report = training.environment()
            # "unknown" means another start got in first: the manager's own refusal below says so.
            if not report["ok"] and not report.get("unknown"):
                failing = "; ".join(c["message"] for c in report["checks"] if not c["ok"])
                raise ValueError(f"Training is not set up: {failing}")
        epochs, clips = body.epochs, None
        if body.minutes is not None and not body.resume:
            # Worked out before any lock: scoring a voice's clips the first time reads every one of them.
            from aivoice import quicktrain

            if isinstance(body.minutes, bool) or not isinstance(body.minutes, (int, float)):
                raise ValueError("The time limit must be a number of minutes")
            chosen = quicktrain.plan_for(training.data, training.library, body.voice_id, float(body.minutes))
            epochs, clips = chosen["epochs"], chosen["clips"]
        # The text-to-speech worker holds GPU memory even when idle, so it is ended first. Ending a process can
        # take seconds and must not happen under the audio lock; instead the lock is taken twice, and in between
        # `training_starting` keeps anything from being spoken (which would load the worker again).
        with exclusive:
            refuse_if_busy()
            if training_starting.is_set():
                raise ValueError("Training is already starting")
            training_starting.set()
        try:
            if tts is not None:
                tts.unload()  # raises, with a message for the user, if a process could not be stopped
            with exclusive:
                refuse_if_busy()  # live mode or a recording may have started meanwhile; speech cannot have
                with voices_lock:  # so a delete of this voice cannot slip in between the check and the start
                    return training.start(body.voice_id, epochs, body.batch_size, body.resume, clips)
        finally:
            training_starting.clear()

    @router.post("/cancel")
    def cancel() -> dict:
        return training.cancel()

    return router
