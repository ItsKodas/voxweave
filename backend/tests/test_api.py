import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from aivoice.api import create_app
from aivoice.manager import LiveManager
from aivoice.presets import PRESETS
from tests.test_live import FakeStreams, Identity
from tests.test_manager import DEVICES
from tests.test_pretrained import add_voice


def make_client(tmp_path, providers=("DmlExecutionProvider",)):
    add_voice(tmp_path, "demo", index=True)
    streams = FakeStreams()
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), streams)
    return TestClient(create_app(manager, tmp_path, lambda: DEVICES, list(providers))), streams


START = {"input_device": 1, "mic_device": 3, "voice": "demo"}


def test_devices(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/devices").json()
    assert body["virtual_mic"] == 3
    assert body["devices"][0] == {"index": 1, "name": "Headset Mic", "max_input": 1, "max_output": 0}


def test_voices(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/voices").json() == [{"id": "demo", "name": "Demo", "has_index": True}]


def test_presets(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/presets").json()
    assert set(body) == set(PRESETS)
    assert body["balanced"]["chunk_ms"] == PRESETS["balanced"].chunk_ms


def test_setup_check(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/setup-check").json()
    assert {c["id"] for c in body} == {"directml", "virtual_mic", "base_model", "voice"}
    assert next(c for c in body if c["id"] == "base_model")["ok"] is False


def test_start_mode_settings_stop(tmp_path):
    client, streams = make_client(tmp_path)
    assert client.post("/live/start", json={**START, "monitor_device": 2}).json()["running"] is True
    assert client.post("/live/mode", json={"mode": "muted"}).json()["mode"] == "muted"
    assert client.patch("/live/settings", json={"pitch_shift": 3, "monitor_gain": 0.5}).status_code == 200
    assert client.get("/live/stats").json()["running"] is True
    assert client.post("/live/stop").json()["running"] is False
    assert all(s.closed for s in streams.streams)


def test_bad_requests_return_400_with_message_and_open_nothing(tmp_path):
    client, streams = make_client(tmp_path)
    cases = [
        ({**START, "monitor_device": 3}, "must be different"),
        ({**START, "input_device": 2}, "not an input"),
        ({**START, "voice": "ghost"}, "Unknown voice"),
        ({**START, "overrides": {"chunk_ms": 1}}, "chunk_ms"),
    ]
    for body, message in cases:
        res = client.post("/live/start", json=body)
        assert res.status_code == 400 and message in res.json()["detail"]
    assert streams.streams == []


def test_missing_field_is_422(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/live/start", json={"voice": "demo"}).status_code == 422


def test_mode_when_stopped_is_400(tmp_path):
    client, _ = make_client(tmp_path)
    res = client.post("/live/mode", json={"mode": "muted"})
    assert res.status_code == 400 and "not running" in res.json()["detail"]


def test_stop_when_stopped_is_ok(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/live/stop").status_code == 200


def test_websocket_streams_stats(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["running"] is False and first["mode"] == "off"
        client.post("/live/start", json=START)
        seen_running = False
        for _ in range(20):
            if ws.receive_json()["running"]:
                seen_running = True
                break
        assert seen_running
    client.post("/live/stop")


def preflight(client, origin):
    return client.options("/live/start", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST"})


def test_cors_preflight_allows_tauri_origin(tmp_path):
    client, _ = make_client(tmp_path)
    res = preflight(client, "http://tauri.localhost")
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://tauri.localhost"


def test_cors_preflight_rejects_foreign_origin(tmp_path):
    client, _ = make_client(tmp_path)
    res = preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in res.headers


def test_websocket_rejects_foreign_origin(tmp_path):
    client, _ = make_client(tmp_path)
    with pytest.raises(WebSocketDisconnect) as info:
        with client.websocket_connect("/ws", headers={"Origin": "https://evil.example"}):
            pass
    assert info.value.code == 1008


def test_websocket_accepts_allowed_origin(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws", headers={"Origin": "http://tauri.localhost"}) as ws:
        assert ws.receive_json()["running"] is False
