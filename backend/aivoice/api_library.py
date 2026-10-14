import logging
import threading
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from aivoice import feminize, tts_reference, voicepack
from aivoice.api import log_refused
from aivoice.audiofile import read_wav
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


class ExportBody(BaseModel):
    include_reference: bool = True


class PackBody(BaseModel):
    path: str


class FeminizeBody(BaseModel):
    pitch_semitones: float = 5.0
    formant_shift: float = 1.15
    breathiness: float = 0.15

    def settings(self) -> "feminize.FeminizeSettings":
        return feminize.FeminizeSettings(self.pitch_semitones, self.formant_shift, self.breathiness).validated()


class FeminizeStartBody(FeminizeBody):
    name: str | None = None


def wav_bytes(audio, sr: int) -> bytes:
    """16-bit PCM WAV in memory, for a sample the screen plays at once."""
    import io
    import wave

    import numpy as np

    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def downloads_folder() -> Path:
    """Where exported voice packs go: the Downloads folder, where people look for a file they just made."""
    return Path.home() / "Downloads"


def _has_reference(pack: Path) -> bool:
    import zipfile

    with zipfile.ZipFile(pack) as z:
        return voicepack.REFERENCE in z.namelist()


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
                   calibrating: Callable[[], bool] = lambda: False, data: Path | None = None,
                   export_dir: Callable[[], Path] | None = None) -> APIRouter:
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

    feminizer = feminize.FeminizeJob(library)

    def sample_clip(voice_id: str):
        """The clip a sample is made from: the one text to speech would choose, else the first of 3 to 12 seconds."""
        voice = library.get(voice_id)
        try:
            return tts_reference.choose(voice, tts_reference.library_levels(library, voice.id))
        except ValueError:
            for clip in voice.clips:
                if 3.0 <= clip.seconds <= 12.0:
                    return clip
        raise ValueError("This voice has no clip of 3 to 12 seconds to make a sample from")

    @router.post("/voices/{voice_id}/feminize/preview")
    def feminize_preview(voice_id: str, body: FeminizeBody) -> Response:
        """One of the voice's clips, feminized with these settings, as a WAV to listen to."""
        settings = body.settings()
        clip = sample_clip(voice_id)
        audio, sr = read_wav(library.clip_audio_path(voice_id, clip.index))
        return Response(content=wav_bytes(feminize.feminize(audio, sr, settings), sr), media_type="audio/wav",
                        headers={"X-Clip-Index": str(clip.index)})

    @router.post("/voices/{voice_id}/feminize")
    def feminize_start(voice_id: str, body: FeminizeStartBody) -> dict:
        """Starts making a new voice from this voice's clips, feminized. The new voice is then trained."""
        with voices_lock:
            return feminizer.start(voice_id, body.settings(), body.name)

    @router.get("/feminize/status")
    def feminize_status() -> dict:
        return feminizer.status()

    @router.post("/feminize/cancel")
    def feminize_cancel() -> dict:
        feminizer.cancel()
        return feminizer.status()

    @router.get("/feminize/settings")
    def feminize_settings() -> dict:
        if data is None:
            raise ValueError("Settings are not available")
        return asdict(feminize.load_settings(data))

    @router.put("/feminize/settings")
    def feminize_save(body: FeminizeBody) -> dict:
        """The settings the live feminized preview uses from its next start."""
        if data is None:
            raise ValueError("Settings are not available")
        return asdict(feminize.save_settings(data, body.settings()))

    @router.post("/voices/{voice_id}/export")
    def export_pack(voice_id: str, body: ExportBody) -> dict:
        """Writes the voice's trained model as one .voxweave file and says where it is."""
        if data is None:
            raise ValueError("Exporting is not available")
        if training is not None and training.is_active():  # the model files may be replaced at any moment
            raise ValueError("Training is running. Wait for it to finish or cancel it before exporting a voice")
        pack = voicepack.export_voice(
            data, library, voice_id, (export_dir or downloads_folder)(), body.include_reference,
            choose_reference=lambda voice: tts_reference.choose(voice, tts_reference.library_levels(library, voice.id)))
        return {"path": str(pack), "bytes": pack.stat().st_size, "has_reference": _has_reference(pack)}

    @router.post("/packs/import")
    def import_pack(body: PackBody) -> dict:
        """Makes a new voice from a .voxweave file on this computer."""
        if data is None:
            raise ValueError("Importing is not available")
        with voices_lock:
            return _summary(voicepack.import_voice(data, library, Path(body.path)))

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
