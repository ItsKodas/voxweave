"""Routes for calibrating live mode to the person speaking (see aivoice.calibration)."""
from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Callable

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from aivoice.api_library import same_app_only_for
from aivoice.calibration import Calibration
from aivoice.config import SAMPLE_RATE


def _wav(audio, sample_rate: int):
    """16-bit mono WAV bytes as a response the page can play."""
    import io
    import wave

    from fastapi import Response

    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return Response(content=buffer.getvalue(), media_type="audio/wav")


class CalibrateBody(BaseModel):
    input_device: int


class ReadBody(BaseModel):
    gain: float


class TuneBody(BaseModel):
    voice_id: str
    quality: int
    pitch_shift: int
    likeness: float
    advanced: dict | None = None


def calibration_router(calibration: Calibration, studio, live_running: Callable[[], bool],
                       exclusive: AbstractContextManager, allowed_origins: list[str], tuner=None, data=None,
                       training=None, tts=None, importer=None) -> APIRouter:
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

    @router.post("/check")
    def check() -> dict:
        """Ends the mic check, for the speaker to play back and set the boost before reading."""
        with exclusive:
            calibration.check()
        return calibration.status()

    @router.post("/read")
    def read(body: ReadBody) -> dict:
        with exclusive:
            if live_running():
                raise ValueError("Stop live mode before calibrating")
            if studio is not None and studio.recording:
                raise ValueError("Stop recording before calibrating")
            calibration.read(body.gain)
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

    # ---- deep calibration (aivoice.deeptune), which needs the speaker model (aivoice.speaker)

    @router.get("/speaker-model")
    def speaker_model() -> dict:
        from aivoice import speaker

        return {"ready": data is not None and speaker.model_ready(data), "size_mb": round(speaker.MODEL_SIZE / 1e6, 1),
                "source": speaker.MODEL_URL, "licence": speaker.MODEL_LICENCE}

    @router.post("/speaker-model/download")
    def download_speaker_model() -> dict:
        """Fetches the speaker model, only ever when the person asked for it on screen."""
        from aivoice import speaker

        if data is None:
            raise ValueError("Deep calibration is not available")
        try:
            speaker.download_model(data)
        except OSError as exc:
            raise ValueError(f"The speaker model could not be downloaded: {exc}") from exc
        return speaker_model()

    @router.post("/tune")
    def tune(body: TuneBody) -> dict:
        from aivoice import speaker

        if tuner is None or data is None:
            raise ValueError("Deep calibration is not available")
        if not speaker.model_ready(data):
            raise ValueError("Download the speaker model first")
        with exclusive:  # the graphics card and the microphone: nothing else may be using them
            if live_running():
                raise ValueError("Stop live mode before deep calibration")
            if training is not None and training.is_active():
                raise ValueError("Training is running. Wait for it to finish or pause it before deep calibration")
            if tts is not None and tts.is_active():
                raise ValueError("Stop the speech before deep calibration")
            if calibration.active or (studio is not None and studio.recording):
                raise ValueError("Finish recording before deep calibration")
            if importer is not None and importer.status().get("active"):
                raise ValueError("Wait for the import to finish before deep calibration")
            tuner.start(body.voice_id, body.quality, body.pitch_shift, body.likeness, body.advanced)
        return tuner.status()

    @router.get("/test-audio")
    def test_audio(gain: float = 1.0):
        """The mic check being reviewed (else the last calibration's), times `gain`, as a WAV."""
        if not 0.0 < gain <= 64.0:
            raise ValueError("The boost must be above 0")
        audio = calibration.test_audio(gain)
        if audio is None:
            raise ValueError("Calibrate first: the mic check is kept from the next calibration on")
        return _wav(audio, SAMPLE_RATE)

    @router.post("/preview")
    def preview(body: TuneBody):
        """The calibration reading in a voice, as a WAV to play: hearing a voice before going live."""
        if tuner is None:
            raise ValueError("Previews are not available")
        with exclusive:  # the graphics card: live mode and training must not be using it
            if live_running():
                raise ValueError("Stop live mode to play a preview")
            if training is not None and training.is_active():
                raise ValueError("Training is running. Pause it to play a preview")
            if tts is not None and tts.is_active():
                raise ValueError("Stop the speech to play a preview")
        audio, sr = tuner.preview(body.voice_id, body.quality, body.pitch_shift, body.likeness, body.advanced)
        return _wav(audio, sr)

    @router.get("/tune/status")
    def tune_status() -> dict:
        return tuner.status() if tuner is not None else {"active": False}

    @router.post("/tune/cancel")
    def tune_cancel() -> dict:
        if tuner is not None:
            tuner.cancel()
        return tune_status()

    return router
