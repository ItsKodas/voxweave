"""Routes for calibrating live mode to the person speaking (see aivoice.calibration)."""
from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from aivoice.api_library import same_app_only_for
from aivoice.calibration import Calibration


class CalibrateBody(BaseModel):
    input_device: int


def calibration_router(calibration: Calibration, studio, live_running: Callable[[], bool],
                       exclusive: AbstractContextManager, allowed_origins: list[str]) -> APIRouter:
    """`exclusive` is the audio lock: a calibration recording holds the microphone like a clip recording does, so
    it is checked and started under the same lock as live mode and recording."""
    router = APIRouter(prefix="/calibration", dependencies=[Depends(same_app_only_for(allowed_origins))])

    @router.get("/status")
    def status() -> dict:
        return calibration.status()

    @router.post("/start")
    def start(body: CalibrateBody) -> dict:
        with exclusive:
            if live_running():
                raise ValueError("Stop live mode before calibrating")
            if studio is not None and studio.recording:
                raise ValueError("Stop recording before calibrating")
            calibration.start(body.input_device)
        return calibration.status()

    @router.post("/finish")
    def finish() -> dict:
        with exclusive:  # the microphone is being closed: nothing else may open devices meanwhile
            profile = calibration.finish()
        return {**calibration.status(), "profile": asdict(profile)}

    @router.post("/cancel")
    def cancel() -> dict:
        with exclusive:
            calibration.cancel()
        return calibration.status()

    @router.delete("/profile")
    def forget() -> dict:
        calibration.forget()
        return calibration.status()

    @router.get("/suggestion")
    def suggestion(voice: str) -> dict:
        return calibration.suggestion(voice)

    return router
