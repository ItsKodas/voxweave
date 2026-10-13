import json
import zipfile

from fastapi.testclient import TestClient

import aivoice.voicepack as voicepack
from aivoice.api import create_app
from aivoice.devices import Device
from aivoice.importer import Importer
from aivoice.library import Library
from aivoice.manager import LiveManager
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from tests.test_live import FakeStreams, Identity
from tests.test_recorder import CapturingStreams
from tests.test_voicepack import trained_voice

DEVICES = [Device(1, "Mic", 1, 0), Device(2, "Speakers", 0, 2)]
ORIGIN = {"Origin": "http://tauri.localhost"}


class Training:
    active = False

    def is_active(self):
        return self.active

    def shutdown(self):
        pass


def rig(tmp_path, monkeypatch, training=None):
    monkeypatch.setattr(voicepack, "loads_as_onnx", lambda path: True)
    monkeypatch.setattr(voicepack.import_voice, "__defaults__", (lambda path: True,))
    data = tmp_path / "data"
    library, voice = trained_voice(data)
    studio = Studio(library, Recorder(CapturingStreams()), Importer(library, None), lambda: DEVICES, prompts=("One.",))
    manager = LiveManager(data, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    app = create_app(manager, data, lambda: DEVICES, [], allowed_hosts=["testserver"], studio=studio, library=library,
                     importer=None, export_dir=lambda: tmp_path / "Downloads")
    if training is not None:
        from aivoice.api_library import library_router
        app = create_app(manager, data, lambda: DEVICES, [], allowed_hosts=["testserver"])
        app.include_router(library_router(studio, library, None, lambda: False, app.state.audio_state,
                                          ["http://tauri.localhost"], training, data=data,
                                          export_dir=lambda: tmp_path / "Downloads"))
    return TestClient(app), library, voice


def test_a_voice_is_exported_to_a_file_and_imported_back_as_a_new_voice(tmp_path, monkeypatch):
    client, library, voice = rig(tmp_path, monkeypatch)
    out = client.post(f"/library/voices/{voice.id}/export", json={"include_reference": True}, headers=ORIGIN)
    assert out.status_code == 200
    body = out.json()
    assert body["path"].endswith("ada.voxweave") and body["has_reference"] is True and body["bytes"] > 5000
    assert json.loads(zipfile.ZipFile(body["path"]).read("manifest.json"))["name"] == "Ada"
    back = client.post("/library/packs/import", json={"path": body["path"]}, headers=ORIGIN)
    assert back.status_code == 200
    assert back.json()["name"] == "Ada" and back.json()["models"] == ["rvc"] and back.json()["clips"] == 1
    assert back.json()["id"] != voice.id and len(client.get("/library/voices").json()) == 2
    assert {v["name"] for v in client.get("/voices").json()} == {"Ada"}  # both show up for live mode


def test_export_without_the_reference_clip(tmp_path, monkeypatch):
    client, _, voice = rig(tmp_path, monkeypatch)
    body = client.post(f"/library/voices/{voice.id}/export", json={"include_reference": False}, headers=ORIGIN).json()
    assert body["has_reference"] is False


def test_refusals_are_plain_messages(tmp_path, monkeypatch):
    client, library, voice = rig(tmp_path, monkeypatch)
    empty = library.create("Empty")
    res = client.post(f"/library/voices/{empty.id}/export", json={}, headers=ORIGIN)
    assert res.status_code == 400 and "no trained model" in res.json()["detail"]
    (tmp_path / "junk.voxweave").write_text("junk")
    res = client.post("/library/packs/import", json={"path": str(tmp_path / "junk.voxweave")}, headers=ORIGIN)
    assert res.status_code == 400 and "not a voice pack" in res.json()["detail"]
    res = client.post("/library/packs/import", json={"path": str(tmp_path / "missing.voxweave")}, headers=ORIGIN)
    assert res.status_code == 400
    assert len(client.get("/library/voices").json()) == 2


def test_a_foreign_page_can_neither_export_nor_import(tmp_path, monkeypatch):
    client, _, voice = rig(tmp_path, monkeypatch)
    evil = {"Origin": "https://evil.example"}
    assert client.post(f"/library/voices/{voice.id}/export", json={}, headers=evil).status_code == 403
    assert client.post("/library/packs/import", json={"path": "C:/x.voxweave"}, headers=evil).status_code == 403
    assert not (tmp_path / "Downloads").exists()


def test_exporting_waits_for_training_to_finish(tmp_path, monkeypatch):
    training = Training()
    training.active = True
    client, _, voice = rig(tmp_path, monkeypatch, training)
    res = client.post(f"/library/voices/{voice.id}/export", json={}, headers=ORIGIN)
    assert res.status_code == 400 and "Training is running" in res.json()["detail"]
