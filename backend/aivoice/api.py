import asyncio
import contextlib
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aivoice.devices import Device, find_virtual_mic
from aivoice.manager import LiveManager, StartConfig
from aivoice.presets import PRESETS
from aivoice.pretrained import list_voices
from aivoice.setup_check import run_checks

# The backend is only ever addressed as a loopback host. Anything else is DNS rebinding.
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
ALLOWED_ORIGINS = [
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
    "http://localhost:1420",
]
# How long the app's shutdown waits for the training worker and the text-to-speech worker to be stopped before it
# goes on. Not a hard limit on the exit: a stop still in progress then keeps running on its own thread, which the
# interpreter waits for when it exits, so the real limits are the managers' own (see their shutdown methods).
SHUTDOWN_WAIT = 10.0
log = logging.getLogger(__name__)


def log_refused(request: Request) -> None:
    """One line saying which request an origin rule refused and the headers it judged, so that a webview sending
    headers nobody expected can be diagnosed from data/backend.log. These headers are not secrets; each is cut
    short and shown with repr, so a hostile value can neither flood the log nor forge a line in it."""

    def shown(name: str) -> str:
        value = request.headers.get(name)
        return "None" if value is None else repr(value[:120])

    log.warning("Refused %s %s (origin not allowed): Origin=%s Referer=%s Sec-Fetch-Site=%s", request.method,
                request.url.path[:120], shown("origin"), shown("referer"), shown("sec-fetch-site"))


class StartBody(BaseModel):
    input_device: int
    mic_device: int | None = None
    voice: str
    monitor_device: int | None = None
    monitor_gain: float = 1.0
    preset: str = "balanced"
    overrides: dict | None = None
    pitch_shift: float = 0.0


class ModeBody(BaseModel):
    mode: str


class SettingsBody(BaseModel):
    pitch_shift: float | None = None
    monitor_enabled: bool | None = None
    monitor_gain: float | None = None


def create_app(manager: LiveManager, data: Path, list_devices: Callable[[], list[Device]],
               providers: list[str], default_devices: Callable[[], dict] | None = None,
               allowed_hosts: list[str] = ALLOWED_HOSTS, studio=None, library=None, importer=None,
               training=None, tts=None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

        async def bounded(shutdown) -> None:
            # The lifespan moves on after SHUTDOWN_WAIT, but the stop itself is not abandoned: it runs on in its
            # thread until the manager's own limits end it. A worker also ends itself once this process is gone.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(asyncio.to_thread(shutdown)), SHUTDOWN_WAIT)

        # Stops both workers so a normal exit leaves nothing for the watchdog. Training's is asked to stop, given
        # 5 seconds to reach a safe point, then killed; text to speech is silenced and its worker killed. The two
        # share no lock and neither takes the audio lock or the voices lock, so they are stopped side by side.
        await asyncio.gather(*(bounded(m.shutdown) for m in (tts, training) if m is not None))

    app = FastAPI(title="AI Voice", lifespan=lifespan)
    # Held while checking and then changing audio state, so recording and live mode cannot both win a race.
    audio_state = threading.Lock()
    app.state.audio_state = audio_state
    # Taken after the audio lock; shared by the voice delete and training start so neither can pass the other's check.
    voices_lock = threading.Lock()
    app.state.voices_lock = voices_lock
    # Set, under the audio lock, while a training start is unloading the text-to-speech worker (see api_tts).
    training_starting = threading.Event()

    # Middleware added later runs earlier: Host check, then CORS, then the Origin check below.
    @app.middleware("http")
    async def reject_foreign_origin(request: Request, call_next):
        # CORS only stops a page from reading a response; a "simple" cross-site POST is still
        # delivered. Browsers always send Origin on those, so refuse any origin we do not serve.
        origin = request.headers.get("origin")
        if request.method not in SAFE_METHODS and origin is not None and origin not in ALLOWED_ORIGINS:
            log_refused(request)
            return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    def device_payload() -> dict:
        found = list_devices()
        mic = find_virtual_mic(found)
        defaults = default_devices() if default_devices else {}
        known = {d.index for d in found}

        def default(key: str) -> int | None:
            return defaults.get(key) if defaults.get(key) in known else None

        return {"devices": [asdict(d) for d in found], "virtual_mic": mic.index if mic else None,
                "default_input": default("default_input"), "default_output": default("default_output")}

    @app.get("/devices")
    def devices() -> dict:
        return device_payload()

    @app.post("/devices/refresh")
    def refresh_devices() -> dict:
        with audio_state:
            if studio is not None and studio.recording:
                raise ValueError("Stop recording before refreshing devices")
            if tts is not None and tts.is_active():  # it would tear down the streams the speech is played on
                raise ValueError("Stop the speech before refreshing devices")
            manager.refresh_devices()
        return device_payload()

    @app.get("/voices")
    def voices() -> list[dict]:
        return [{"id": v.id, "name": v.name, "has_index": v.index_path is not None} for v in list_voices(data)]

    @app.get("/presets")
    def presets() -> dict:
        return {name: asdict(p) for name, p in PRESETS.items()}

    @app.get("/setup-check")
    def setup_check() -> list[dict]:
        return [asdict(c) for c in run_checks(data, list_devices(), providers)]

    @app.post("/live/start")
    def start(body: StartBody) -> dict:
        with audio_state:
            if studio is not None and studio.recording:
                raise ValueError("Stop recording before starting live mode")
            if training is not None and training.is_active():  # a read only: never a kill under the audio lock
                raise ValueError("Training is running. Wait for it to finish or cancel it before starting live mode")
            # Also a read only. Speak makes the opposite check and starts under this same lock. A worker that is
            # loaded but idle holds no sound device and is no obstacle.
            if tts is not None and tts.is_active():
                raise ValueError("Stop the speech before starting live mode")
            manager.start(StartConfig(**body.model_dump()))
        return manager.stats()

    @app.post("/live/stop")
    def stop() -> dict:
        # Live mode reads "not running" from the moment the stop begins, up to two seconds before its devices
        # are closed. Under the audio lock, nothing that checks "is live running?" (recording, training, text to
        # speech, another live start) can act on that answer until the devices really are free. Nothing the stop
        # calls, and nothing the live thread does on its way out, takes this lock.
        with audio_state:
            manager.stop()
        return manager.stats()

    @app.post("/live/mode")
    def mode(body: ModeBody) -> dict:
        manager.set_mode(body.mode)
        return manager.stats()

    @app.patch("/live/settings")
    def settings(body: SettingsBody) -> dict:
        manager.update(**body.model_dump())
        return manager.stats()

    @app.get("/live/stats")
    def stats() -> dict:
        return manager.stats()

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        # Browsers do not apply CORS to WebSockets, so check the Origin ourselves.
        origin = socket.headers.get("origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            while True:
                await socket.send_json(manager.stats())
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            pass

    if studio is not None:
        from aivoice.api_library import library_router

        app.include_router(library_router(studio, library, importer, lambda: bool(manager.stats()["running"]),
                                          audio_state, ALLOWED_ORIGINS, training, voices_lock, tts))
    if training is not None:
        from aivoice.api_training import training_router

        app.include_router(training_router(training, studio, lambda: bool(manager.stats()["running"]),
                                           audio_state, voices_lock, ALLOWED_ORIGINS, tts, training_starting))
    if tts is not None:
        from aivoice.api_tts import tts_router

        app.include_router(tts_router(tts, library, studio, training, lambda: bool(manager.stats()["running"]),
                                      audio_state, voices_lock, training_starting, ALLOWED_ORIGINS,
                                      monitor_gain=lambda: manager.monitor_gain))

    return app
