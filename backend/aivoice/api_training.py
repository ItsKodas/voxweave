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


def training_router(training: TrainingManager, studio: Studio, live_running: Callable[[], bool],
                    exclusive: AbstractContextManager, voices_lock: AbstractContextManager,
                    allowed_origins: list[str]) -> APIRouter:
    """`exclusive` and `voices_lock` are the audio lock and voices lock shared with /live and /library, taken in
    that order. Only start takes them: status and environment are polled and must never queue behind a save."""
    router = APIRouter(prefix="/training", dependencies=[Depends(same_app_only_for(allowed_origins))])

    @router.get("/status")
    def status(voice_id: str | None = None) -> dict:
        return training.status(voice_id)

    @router.get("/environment")
    def environment(refresh: bool = False) -> dict:
        return training.environment(refresh)

    @router.post("/start")
    def start(body: TrainBody) -> dict:
        with exclusive:
            if live_running():
                raise ValueError("Stop live mode before training")
            if studio.recording:
                raise ValueError("Stop recording before training")
            with voices_lock:  # so a delete of this voice cannot slip in between the check and the start
                return training.start(body.voice_id, body.epochs, body.batch_size, body.resume)

    @router.post("/cancel")
    def cancel() -> dict:
        return training.cancel()

    return router
