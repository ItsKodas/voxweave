import asyncio
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
               allowed_hosts: list[str] = ALLOWED_HOSTS) -> FastAPI:
    app = FastAPI(title="AI Voice")

    # Middleware added later runs earlier: Host check, then CORS, then the Origin check below.
    @app.middleware("http")
    async def reject_foreign_origin(request: Request, call_next):
        # CORS only stops a page from reading a response; a "simple" cross-site POST is still
        # delivered. Browsers always send Origin on those, so refuse any origin we do not serve.
        origin = request.headers.get("origin")
        if request.method not in SAFE_METHODS and origin is not None and origin not in ALLOWED_ORIGINS:
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
        manager.start(StartConfig(**body.model_dump()))
        return manager.stats()

    @app.post("/live/stop")
    def stop() -> dict:
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

    return app
