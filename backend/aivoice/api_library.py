import threading
from dataclasses import asdict
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from aivoice.importer import Importer
from aivoice.library import Library, Voice
from aivoice.studio import Studio


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
            "clips": len(voice.clips)}


def library_router(studio: Studio, library: Library, importer: Importer, live_running: Callable[[], bool],
                   exclusive=None) -> APIRouter:
    """`exclusive` is shared with the live-start and device-refresh routes, so checking "is live running?"
    and then starting a recording cannot interleave with checking "is recording?" and then starting live."""
    router = APIRouter(prefix="/library")
    guard = exclusive if exclusive is not None else threading.Lock()
    importing: dict[str, str | None] = {"voice": None}  # the voice of the most recent import run

    @router.get("/voices")
    def voices() -> list[dict]:
        return [_summary(v) for v in library.list()]

    @router.post("/voices")
    def create_voice(body: VoiceBody) -> dict:
        return _summary(library.create(body.name))

    @router.delete("/voices/{voice_id}")
    def delete_voice(voice_id: str) -> dict:
        with guard:
            if studio.recording:
                raise ValueError("Stop recording before deleting a voice")
            if importing["voice"] == voice_id and importer.status()["active"]:
                raise ValueError("Wait for the import to finish before deleting this voice")
            library.delete(voice_id)
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
    def prompt(voice_id: str, after: int | None = None) -> dict:
        return studio.prompt(voice_id, after)

    @router.post("/record/start")
    def record_start(body: RecordBody) -> dict:
        with guard:
            if live_running():
                raise ValueError("Stop live mode before recording")
            studio.start_recording(body.voice_id, body.input_device, body.prompt)
        return studio.status()

    @router.post("/record/stop")
    def record_stop() -> dict:
        return studio.stop_recording()

    @router.post("/record/cancel")
    def record_cancel() -> dict:
        studio.cancel_recording()
        return studio.status()

    @router.post("/voices/{voice_id}/import")
    def import_files(voice_id: str, body: ImportBody) -> dict:
        with guard:  # so a delete of this voice cannot slip in between the start and the bookkeeping
            importer.start(voice_id, body.paths)
            importing["voice"] = voice_id
        return studio.status()

    @router.post("/import/cancel")
    def import_cancel() -> dict:
        importer.cancel()
        return studio.status()

    @router.get("/status")
    def status() -> dict:
        return studio.status()

    return router
