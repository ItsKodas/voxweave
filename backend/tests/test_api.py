import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from aivoice.api import ALLOWED_HOSTS, create_app
from aivoice.manager import LiveManager
from aivoice.presets import PRESETS
from tests.test_live import FakeStreams, Identity
from tests.test_manager import DEVICES
from tests.test_pretrained import add_voice


DEFAULTS = {"default_input": 1, "default_output": 2}


def make_client(tmp_path, providers=("DmlExecutionProvider",), refreshed=None):
    add_voice(tmp_path, "demo", index=True)
    streams = FakeStreams()
    refresh = (lambda: refreshed.append("refresh")) if refreshed is not None else None
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), streams, refresh_devices=refresh)
    app = create_app(manager, tmp_path, lambda: DEVICES, list(providers), default_devices=lambda: DEFAULTS,
                     allowed_hosts=[*ALLOWED_HOSTS, "testserver"])
    return TestClient(app), streams


START = {"input_device": 1, "mic_device": 3, "voice": "demo"}


def test_devices(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/devices").json()
    assert body["virtual_mic"] == 3
    assert body["devices"][0] == {"index": 1, "name": "Headset Mic", "max_input": 1, "max_output": 0}
    assert body["default_input"] == 1 and body["default_output"] == 2


def test_devices_without_a_defaults_source_reports_none(tmp_path):
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    client = TestClient(create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=["testserver"]))
    body = client.get("/devices").json()
    assert body["default_input"] is None and body["default_output"] is None


def test_refresh_devices_reinitialises_and_returns_the_device_payload(tmp_path):
    refreshed = []
    client, _ = make_client(tmp_path, refreshed=refreshed)
    res = client.post("/devices/refresh")
    assert res.status_code == 200
    assert refreshed == ["refresh"]
    assert res.json() == client.get("/devices").json()


def test_refresh_devices_is_refused_while_live(tmp_path):
    refreshed = []
    client, _ = make_client(tmp_path, refreshed=refreshed)
    client.post("/live/start", json=START)
    res = client.post("/devices/refresh")
    assert res.status_code == 400
    assert res.json()["detail"] == "Stop live mode before refreshing devices"
    assert refreshed == []
    assert client.get("/live/stats").json()["running"] is True
    client.post("/live/stop")


def test_voices(tmp_path):
    client, _ = make_client(tmp_path)
    # The feminized preview of the speaker's own voice is always offered last, with no model behind it.
    assert client.get("/voices").json() == [{"id": "demo", "name": "Demo", "has_index": True},
                                            {"id": "@feminize", "name": "My voice, feminized (preview, no training)",
                                             "has_index": False}]


def test_presets(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/presets").json()
    assert set(body) == set(PRESETS)
    assert body["balanced"]["chunk_ms"] == PRESETS["balanced"].chunk_ms


def test_setup_check(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/setup-check").json()
    assert {c["id"] for c in body} == {"directml", "virtual_mic", "base_model", "voice", "settings"}
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


def test_monitor_only_start_needs_no_virtual_mic(tmp_path):
    client, streams = make_client(tmp_path)
    res = client.post("/live/start", json={"input_device": 1, "mic_device": None, "voice": "demo", "monitor_device": 2})
    assert res.status_code == 200 and res.json()["running"] is True
    assert [(s.kind, s.device) for s in streams.streams] == [("in", 1), ("out", 2)]
    client.post("/live/stop")


@pytest.mark.parametrize("body", [
    {"input_device": 1, "mic_device": None, "voice": "demo"},
    {"input_device": 1, "voice": "demo"},
])
def test_start_with_neither_output_is_400(tmp_path, body):
    client, streams = make_client(tmp_path)
    res = client.post("/live/start", json=body)
    assert res.status_code == 400
    assert res.json()["detail"] == "Choose a virtual mic or a listen device"
    assert streams.streams == []


def test_production_hosts_are_loopback_only():
    assert ALLOWED_HOSTS == ["127.0.0.1", "localhost"]


def test_wrong_host_is_rejected(tmp_path):
    client, streams = make_client(tmp_path)
    assert client.get("/health", headers={"Host": "evil.example"}).status_code == 400
    res = client.post("/live/start", json=START, headers={"Host": "evil.example:8000"})
    assert res.status_code == 400
    assert streams.streams == []


def test_loopback_hosts_are_accepted_with_any_port(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/health", headers={"Host": "127.0.0.1:51234"}).status_code == 200
    assert client.get("/health", headers={"Host": "localhost:51234"}).status_code == 200


def test_default_app_rejects_hosts_other_than_loopback(tmp_path):
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    client = TestClient(create_app(manager, tmp_path, lambda: DEVICES, []))
    assert client.get("/health").status_code == 400  # the test client's host is "testserver"
    assert client.get("/health", headers={"Host": "127.0.0.1:1"}).status_code == 200


def test_websocket_with_wrong_host_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"Host": "evil.example"}):
            pass


def test_foreign_origin_cannot_stop_live_mode(tmp_path):
    client, streams = make_client(tmp_path)
    client.post("/live/start", json=START)
    res = client.post("/live/stop", headers={"Origin": "https://evil.example"})
    assert res.status_code == 403
    assert client.get("/live/stats").json()["running"] is True
    assert all(s.active for s in streams.streams)
    client.post("/live/stop")


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://localhost:9999"])
def test_foreign_origin_cannot_start_or_change_anything(tmp_path, origin):
    client, streams = make_client(tmp_path)
    headers = {"Origin": origin}
    assert client.post("/live/start", json=START, headers=headers).status_code == 403
    assert client.post("/live/mode", json={"mode": "muted"}, headers=headers).status_code == 403
    assert client.patch("/live/settings", json={"pitch_shift": 1}, headers=headers).status_code == 403
    assert client.post("/devices/refresh", headers=headers).status_code == 403
    assert streams.streams == []


def test_allowed_origin_and_no_origin_can_stop(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/live/stop", headers={"Origin": "http://tauri.localhost"}).status_code == 200
    assert client.post("/live/stop").status_code == 200


def test_foreign_origin_can_still_read(tmp_path):
    client, _ = make_client(tmp_path)
    res = client.get("/health", headers={"Origin": "https://evil.example"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers


def test_quality_levels_are_offered_fastest_first(tmp_path):
    client, _ = make_client(tmp_path)
    levels = client.get("/quality-levels").json()
    assert levels[0]["chunk_ms"] == 120 and levels[-1]["chunk_ms"] == 2000
    assert all({"chunk_ms", "context_ms", "crossfade_ms"} == set(level) for level in levels)
