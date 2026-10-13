import logging
import threading
from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Callable
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from aivoice.api import log_refused
from aivoice.importer import Importer
from aivoice.library import Library, Voice
from aivoice.studio import Studio


log = logging.getLogger(__name__)


class VoiceBody(BaseModel):
    name: str


class TextBody(BaseModel):
    text: str


class RecordBody(BaseModel):
    voice_id: str
    input_device: int
    prompt: int


class ImportBody(BaseModel):
    paths: list[str]


def _summary(voice: Voice) -> dict:
    return {"id": voice.id, "name": voice.name, "created": voice.created, "minutes": voice.minutes,
            "clips": len(voice.clips), "models": voice.models}


def _origin_of(url: str) -> str | None:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else None


def same_app_only_for(allowed_origins: list[str]) -> Callable[[Request], None]:
    """The request-origin rule of /library and /training, as a dependency."""

    def same_app_only(request: Request) -> None:
        # Recordings are sensitive, and a GET is exempt from the app-wide Origin check, so any web page could
        # embed a clip in <audio>. Browsers name the page in Origin or Referer; refuse any that is not the app.
        # The app's own webview (an allowed origin) is cross-site to 127.0.0.1, so Sec-Fetch-Site alone cannot
        # tell it from a foreign page: cross-site is accepted only when Origin or Referer names the app.
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        named = [o for o in (origin, _origin_of(referer) if referer else None) if o is not None]
        if any(o not in allowed_origins for o in named):
            log_refused(request)
            raise HTTPException(status_code=403, detail="Origin not allowed")
        # A browser page (Sec-Fetch-Site other than "none") must name the app: a page on another localhost port
        # is same-site and may send neither header on a no-cors GET. Clients that send no Sec-Fetch-Site are not pages.
        site = request.headers.get("sec-fetch-site")
        if site is not None and site != "none" and not named:
            log_refused(request)
            raise HTTPException(status_code=403, detail="Origin not allowed")

    return same_app_only


def library_router(studio: Studio, library: Library, importer: Importer, live_running: Callable[[], bool],
                   exclusive: AbstractContextManager, allowed_origins: list[str], training=None,
                   voices_lock: AbstractContextManager | None = None, tts=None,
                   calibrating: Callable[[], bool] = lambda: False) -> APIRouter:
    """`exclusive` is the audio lock, shared with the live-start and device-refresh routes: checking
    "is live running?" and then starting a recording must not interleave with the reverse."""

    router = APIRouter(prefix="/library", dependencies=[Depends(same_app_only_for(allowed_origins))])
    # Only the pair that touches voices on disk: a delete must not queue behind a slow live start.
    voices_lock = voices_lock if voices_lock is not None else threading.Lock()

    @router.get("/voices")
    def voices() -> list[dict]:
        return [_summary(v) for v in library.list()]

    @router.post("/voices")
    def create_voice(body: VoiceBody) -> dict:
        return _summary(library.create(body.name))

    @router.delete("/voices/{voice_id}")
    def delete_voice(voice_id: str) -> dict:
        with voices_lock:
            if studio.recording:
                raise ValueError("Stop recording before deleting a voice")
            job = importer.status()
            if job["active"] and job["voice_id"] == voice_id:
                raise ValueError("Wait for the import to finish before deleting this voice")
            if training is not None:
                # Reads only: neither this nor forget below ever ends a process while the voices lock is held.
                training.require_not_training(voice_id)
            # Also a read only. Any voice, not just the one being spoken: the request reads its clip and its model
            # as it goes, and speak starts under this same lock.
            if tts is not None and tts.is_active():
                raise ValueError("Stop the speech before deleting a voice")
            library.delete(voice_id)
            if training is not None:
                try:
                    training.forget(voice_id)
                except Exception as exc:  # noqa: BLE001 - the voice is already gone; leftovers are only derived files
                    log.warning("Deleted voice %s but could not remove its training files: %s", voice_id, exc)
        return {"deleted": True}

    @router.get("/voices/{voice_id}/clips")
    def clips(voice_id: str) -> list[dict]:
        return [asdict(c) for c in library.get(voice_id).clips]

    @router.patch("/voices/{voice_id}/clips/{index}")
    def edit_clip(voice_id: str, index: int, body: TextBody) -> dict:
        return asdict(library.update_text(voice_id, index, body.text))

    @router.delete("/voices/{voice_id}/clips/{index}")
    def delete_clip(voice_id: str, index: int) -> dict:
        library.delete_clip(voice_id, index)
        return {"deleted": True}

    @router.get("/voices/{voice_id}/clips/{index}/audio")
    def clip_audio(voice_id: str, index: int) -> FileResponse:
        return FileResponse(library.clip_audio_path(voice_id, index), media_type="audio/wav")

    @router.get("/voices/{voice_id}/prompt")
    def prompt(voice_id: str, after: int | None = None, kind: str | None = None) -> dict:
        return studio.prompt(voice_id, after, kind)

    @router.post("/record/start")
    def record_start(body: RecordBody) -> dict:
        with exclusive:
            if live_running():
                raise ValueError("Stop live mode before recording")
            if training is not None and training.is_active():  # a read only: never a kill under the audio lock
                raise ValueError("Training is running. Wait for it to finish or cancel it before recording")
            if tts is not None and tts.is_active():  # also a read only; speak checks and starts under this lock
                raise ValueError("Stop the speech before recording")
            if calibrating():
                raise ValueError("Finish the calibration recording before recording a clip")
            with voices_lock:  # so a delete cannot check "not recording" and then lose the voice to a new take
                studio.start_recording(body.voice_id, body.input_device, body.prompt)
        return studio.status()

    @router.post("/record/stop")
    def record_stop() -> dict:
        with exclusive:  # the stream may still be closing after recording reads False: devices must wait
            with voices_lock:  # a delete must not pass its "not recording" check while the take is being saved
                return studio.stop_recording()

    @router.post("/record/cancel")
    def record_cancel() -> dict:
        with exclusive:
            studio.cancel_recording()
        return studio.status()

    @router.post("/voices/{voice_id}/import")
    def import_files(voice_id: str, body: ImportBody) -> dict:
        with voices_lock:  # so a delete of this voice cannot slip in between the check and the start
            importer.start(voice_id, body.paths)
        return studio.status()

    @router.post("/import/cancel")
    def import_cancel() -> dict:
        importer.cancel()
        return studio.status()

    @router.get("/status")
    def status() -> dict:
        return studio.status()

    return router
