import shutil
import threading

import numpy as np
from fastapi.testclient import TestClient

from aivoice.api import ALLOWED_HOSTS, create_app
from aivoice.audiofile import write_wav
from aivoice.importer import Importer
from aivoice.library import Library
from aivoice.manager import LiveManager
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from tests.test_live import FakeStreams, Identity
from tests.test_manager import DEVICES
from tests.test_pretrained import add_voice
from tests.test_recorder import CapturingStreams
from tests.test_studio import PROMPTS, speak

SR = 48000


class Echo:
    def transcribe(self, audio16k):
        return "heard"


class Blocking:
    """A transcriber that holds the import open until the test releases it."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio16k):
        self.entered.set()
        self.release.wait(10)
        return "heard"


def make(tmp_path, refreshed=None, transcriber=None):
    add_voice(tmp_path, "demo")
    library = Library(tmp_path)
    mic = CapturingStreams()
    importer = Importer(library, transcriber or Echo())
    studio = Studio(library, Recorder(mic), importer, lambda: DEVICES, prompts=PROMPTS)
    refresh = (lambda: refreshed.append(1)) if refreshed is not None else None
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams(), refresh_devices=refresh)
    app = create_app(manager, tmp_path, lambda: DEVICES, ["DmlExecutionProvider"],
                     allowed_hosts=[*ALLOWED_HOSTS, "testserver"], studio=studio, library=library, importer=importer)
    return TestClient(app), mic, importer


def new_voice(client, name="V"):
    return client.post("/library/voices", json={"name": name}).json()


def record(client, mic, voice_id, prompt=0, **kw):
    assert client.post("/library/record/start", json={"voice_id": voice_id, "input_device": 1, "prompt": prompt}).status_code == 200
    speak(mic, **kw)
    return client.post("/library/record/stop").json()


def drop_file(tmp_path, name="talk.wav"):
    path = tmp_path / name
    write_wav(path, np.random.default_rng(2).uniform(-0.3, 0.3, 2 * SR).astype(np.float32), SR)
    return path


def test_create_list_delete_voice(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client, "My Voice")
    assert voice["name"] == "My Voice" and voice["minutes"] == 0 and voice["clips"] == 0
    assert [v["id"] for v in client.get("/library/voices").json()] == [voice["id"]]
    assert client.delete(f"/library/voices/{voice['id']}").json() == {"deleted": True}
    assert client.get("/library/voices").json() == []


def test_bad_voice_name_is_400(tmp_path):
    client, _, _ = make(tmp_path)
    res = client.post("/library/voices", json={"name": "   "})
    assert res.status_code == 400 and "name" in res.json()["detail"]


def test_unknown_voice_is_400_everywhere(tmp_path):
    client, _, _ = make(tmp_path)
    for method, path in [("get", "/library/voices/nope-000000/clips"), ("get", "/library/voices/nope-000000/prompt"),
                         ("delete", "/library/voices/nope-000000"), ("get", "/library/voices/..%2F..%2Fx/clips")]:
        assert getattr(client, method)(path).status_code in (400, 404)


def test_guided_recording_round_trip(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    assert client.get(f"/library/voices/{voice['id']}/prompt").json()["index"] == 0
    result = record(client, mic, voice["id"])
    assert result["accepted"] is True and result["clip"]["text"] == PROMPTS[0]
    clips = client.get(f"/library/voices/{voice['id']}/clips").json()
    assert [c["index"] for c in clips] == [1]
    assert client.get(f"/library/voices/{voice['id']}/prompt").json()["index"] == 1
    assert client.get(f"/library/voices/{voice['id']}/prompt", params={"after": 1}).json()["index"] == 2
    assert client.get("/library/voices").json()[0]["clips"] == 1


def test_rejected_take_returns_reason_and_message(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    result = record(client, mic, voice["id"], amp=0.01)
    assert result == {"accepted": False, "reason": "too_quiet", "message": result["message"], "clip": None}
    assert client.get(f"/library/voices/{voice['id']}/clips").json() == []


def test_clip_audio_edit_and_delete(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    record(client, mic, voice["id"])
    base = f"/library/voices/{voice['id']}/clips/1"
    audio = client.get(f"{base}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav" and audio.content[:4] == b"RIFF"
    assert client.patch(base, json={"text": "Fixed."}).json()["text"] == "Fixed."
    assert client.delete(base).json() == {"deleted": True}
    assert client.get(f"{base}/audio").status_code == 400


def test_audio_route_cannot_leave_the_clips_folder(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    vid = voice["id"]
    record(client, mic, vid)
    # Bait files a traversal would reach: beside the voice folder, in the data folder, in the clips folder.
    (tmp_path / "voices" / "secret.orig.wav").write_bytes(b"RIFF-secret-voices")
    (tmp_path / "secret.orig.wav").write_bytes(b"RIFF-secret-data")
    (tmp_path / "voices" / vid / "clips" / "9999.orig.wav").write_bytes(b"RIFF-secret-unlisted")
    own = (tmp_path / "voices" / vid / "clips" / "0001.orig.wav").read_bytes()

    bad_ids = ["..", "%2e%2e", "..%2f", "..%2F..%2Fsecret", "%2e%2e%2f%2e%2e%2fsecret", "..%5C..%5Csecret",
               "..%5c", "%2e%2e%5csecret", f"{vid}%2F..%2F..", "C:%5Cwindows", "%00", "nope-000000", "UPPER-000000"]
    for bad in bad_ids:
        res = client.get(f"/library/voices/{bad}/clips/1/audio")
        assert res.status_code in (400, 404, 422), bad
        assert b"secret" not in res.content and res.content != own, bad

    bad_indexes = ["-1", "0", "2", "9999", "999999999", "99999999999999999999999", "1.5", "abc",
                   "..%2F..%2Fsecret", "%2e%2e", "1%2F..%2F1", "-0001", "1e0", "%00"]
    for bad in bad_indexes:
        res = client.get(f"/library/voices/{vid}/clips/{bad}/audio")
        assert res.status_code in (400, 404, 422), bad
        assert b"secret" not in res.content and res.content != own, bad

    # a valid request serves exactly that voice's own file
    res = client.get(f"/library/voices/{vid}/clips/1/audio")
    assert res.status_code == 200 and res.content == own


def test_status_while_recording_and_cancel(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    speak(mic)
    status = client.get("/library/status").json()
    assert status["recording"] is True and status["seconds"] > 1 and status["import"]["active"] is False
    assert client.post("/library/record/cancel").json()["recording"] is False
    assert client.post("/library/record/stop").status_code == 400


def test_recording_is_refused_while_live_and_live_while_recording(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client)
    start_live = {"input_device": 1, "mic_device": 3, "voice": "demo"}
    record_body = {"voice_id": voice["id"], "input_device": 1, "prompt": 0}

    assert client.post("/live/start", json=start_live).status_code == 200
    res = client.post("/library/record/start", json=record_body)
    assert res.status_code == 400 and res.json()["detail"] == "Stop live mode before recording"
    assert client.get("/library/status").json()["recording"] is False
    client.post("/live/stop")

    assert client.post("/library/record/start", json=record_body).status_code == 200
    res = client.post("/live/start", json=start_live)
    assert res.status_code == 400 and res.json()["detail"] == "Stop recording before starting live mode"
    assert client.get("/live/stats").json()["running"] is False
    client.post("/library/record/cancel")


def test_device_refresh_is_refused_while_recording(tmp_path):
    refreshed = []
    client, _, _ = make(tmp_path, refreshed)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    res = client.post("/devices/refresh")
    assert res.status_code == 400 and res.json()["detail"] == "Stop recording before refreshing devices"
    assert refreshed == []
    client.post("/library/record/cancel")
    assert client.post("/devices/refresh").status_code == 200


def test_device_refresh_is_allowed_during_an_import(tmp_path):
    refreshed = []
    block = Blocking()
    client, _, importer = make(tmp_path, refreshed, block)
    voice = new_voice(client)
    path = drop_file(tmp_path)
    assert client.post(f"/library/voices/{voice['id']}/import", json={"paths": [str(path)]}).status_code == 200
    assert block.entered.wait(10)
    try:
        assert client.get("/library/status").json()["import"]["active"] is True
        assert client.post("/devices/refresh").status_code == 200 and refreshed == [1]
    finally:
        block.release.set()
        importer.wait(10)


def test_deleting_a_voice_during_its_import_is_refused(tmp_path):
    block = Blocking()
    client, _, importer = make(tmp_path, transcriber=block)
    voice = new_voice(client)
    other = new_voice(client, "Other")
    path = drop_file(tmp_path)
    client.post(f"/library/voices/{voice['id']}/import", json={"paths": [str(path)]})
    assert block.entered.wait(10)
    try:
        res = client.delete(f"/library/voices/{voice['id']}")
        assert res.status_code == 400
        assert res.json()["detail"] == "Wait for the import to finish before deleting this voice"
        assert voice["id"] in [v["id"] for v in client.get("/library/voices").json()]
        assert client.delete(f"/library/voices/{other['id']}").json() == {"deleted": True}  # an unrelated voice
    finally:
        block.release.set()
        importer.wait(10)
    status = client.get("/library/status").json()["import"]
    assert status["finished"] and status["errors"] == [] and status["clips_added"] == 1
    assert client.delete(f"/library/voices/{voice['id']}").json() == {"deleted": True}


def test_voice_cannot_be_deleted_while_recording(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    res = client.delete(f"/library/voices/{voice['id']}")
    assert res.status_code == 400 and res.json()["detail"] == "Stop recording before deleting a voice"
    client.post("/library/record/cancel")


def test_stop_after_the_voice_vanished_is_400_and_leaves_recorder_idle(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    speak(mic)
    shutil.rmtree(tmp_path / "voices" / voice["id"])
    assert client.post("/library/record/stop").status_code == 400
    assert client.get("/library/status").json()["recording"] is False


def test_import_files(tmp_path):
    client, _, importer = make(tmp_path)
    voice = new_voice(client)
    path = drop_file(tmp_path)
    res = client.post(f"/library/voices/{voice['id']}/import", json={"paths": [str(path)]})
    assert res.status_code == 200
    importer.wait(10)
    status = client.get("/library/status").json()["import"]
    assert status["finished"] and status["clips_added"] == 1
    assert client.get(f"/library/voices/{voice['id']}/clips").json()[0]["text"] == "heard"
    assert client.post("/library/import/cancel").status_code == 200


def test_import_with_no_files_is_400(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client)
    assert client.post(f"/library/voices/{voice['id']}/import", json={"paths": []}).status_code == 400


def test_library_routes_obey_the_origin_rule(tmp_path):
    client, _, _ = make(tmp_path)
    res = client.post("/library/voices", json={"name": "X"}, headers={"Origin": "https://evil.example"})
    assert res.status_code == 403
    assert client.get("/library/voices").json() == []


def test_library_routes_obey_the_host_rule(tmp_path):
    client, _, _ = make(tmp_path)
    assert client.get("/library/voices", headers={"Host": "evil.example"}).status_code == 400


def test_app_without_a_studio_has_no_library_routes(tmp_path):
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    client = TestClient(create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=["testserver"]))
    assert client.get("/library/voices").status_code == 404
