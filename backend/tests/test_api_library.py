import errno
import shutil
import threading

import numpy as np
import pytest
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


def test_voice_summary_lists_the_trained_models(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client)
    assert voice["models"] == []
    folder = tmp_path / "voices" / voice["id"] / "models" / "rvc"
    folder.mkdir(parents=True)
    (folder / "model.onnx").write_bytes(b"x")
    (folder / "voice.json").write_text('{"name": "V", "sample_rate": 40000}')
    assert client.get("/library/voices").json()[0]["models"] == ["rvc"]


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


class SlowStopStreams(CapturingStreams):
    """The microphone stream's stop() blocks until released, like a PortAudio stream that is slow to close."""

    def __init__(self):
        super().__init__()
        self.inside_stop = threading.Event()
        self.release = threading.Event()

    def input(self, device, samplerate, on_audio):
        stream = super().input(device, samplerate, on_audio)
        outer = self

        def slow_stop():
            outer.inside_stop.set()
            outer.release.wait(10)
            stream.active = False

        stream.stop = slow_stop
        return stream


def run_in_thread(fn):
    done = threading.Event()
    box = {}

    def target():
        box["result"] = fn()
        done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return done, box, thread


def make_slow_stop(tmp_path, refreshed):
    add_voice(tmp_path, "demo")
    library = Library(tmp_path)
    mic = SlowStopStreams()
    importer = Importer(library, Echo())
    studio = Studio(library, Recorder(mic), importer, lambda: DEVICES, prompts=PROMPTS)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams(),
                          refresh_devices=lambda: refreshed.append(1))
    app = create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=[*ALLOWED_HOSTS, "testserver"],
                     studio=studio, library=library, importer=importer)
    return TestClient(app), mic


@pytest.mark.parametrize("route", ["stop", "cancel"])
def test_device_changes_wait_for_a_closing_microphone_stream(tmp_path, route):
    refreshed = []
    client, mic = make_slow_stop(tmp_path, refreshed)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    speak(mic)
    stopping, _, stop_thread = run_in_thread(lambda: client.post(f"/library/record/{route}"))
    assert mic.inside_stop.wait(10)
    refresh_done, refresh_box, refresh_thread = run_in_thread(lambda: client.post("/devices/refresh"))
    live_done, live_box, live_thread = run_in_thread(
        lambda: client.post("/live/start", json={"input_device": 1, "mic_device": 3, "voice": "demo"}))
    assert not refresh_done.wait(0.3) and not live_done.wait(0.1)
    assert refreshed == []
    mic.release.set()
    for thread in (stop_thread, refresh_thread, live_thread):
        thread.join(10)
    assert stopping.is_set() and live_box["result"].status_code == 200
    # Both waited for the stream. Which of the two then takes the audio lock first is up to the scheduler, and
    # a refresh that comes second is correctly refused because live mode is running.
    if refresh_box["result"].status_code == 200:
        assert refreshed == [1]
    else:
        assert refresh_box["result"].status_code == 400 and refreshed == []
    client.post("/live/stop")


def test_voice_delete_and_import_do_not_queue_behind_the_audio_lock(tmp_path):
    client, _, importer = make(tmp_path)
    voice = new_voice(client)
    path = drop_file(tmp_path)
    with client.app.state.audio_state:  # as if a slow live start were in progress
        done, box, thread = run_in_thread(lambda: client.post(
            f"/library/voices/{voice['id']}/import", json={"paths": [str(path)]}))
        assert done.wait(10) and box["result"].status_code == 200
        importer.wait(10)
        done, box, thread = run_in_thread(lambda: client.delete(f"/library/voices/{voice['id']}"))
        assert done.wait(10) and box["result"].json() == {"deleted": True}


def test_a_take_over_20_seconds_is_too_long_over_http(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    result = record(client, mic, voice["id"], seconds=25.0)
    assert result["accepted"] is False and result["reason"] == "too_long"
    assert client.get(f"/library/voices/{voice['id']}/clips").json() == []


def test_clip_audio_is_not_served_to_other_sites(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    record(client, mic, voice["id"])
    url = f"/library/voices/{voice['id']}/clips/1/audio"
    assert client.get(url).status_code == 200
    assert client.get(url, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get(url, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get(url, headers={"Referer": "https://evil.example/page"}).status_code == 403
    assert client.get(url, headers={"Origin": "http://tauri.localhost"}).status_code == 200
    # the app's own webview is cross-site to 127.0.0.1 but names itself
    app_audio = {"Sec-Fetch-Site": "cross-site", "Referer": "http://tauri.localhost/"}
    assert client.get(url, headers=app_audio).status_code == 200
    assert client.get(url, headers={"Sec-Fetch-Site": "same-origin"}).status_code == 403  # a page, naming nobody
    assert client.get("/library/voices", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post("/library/voices", json={"name": "X"}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    # stage 1 routes are unchanged
    assert client.get("/devices", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


def test_library_requires_the_app_to_be_named_when_a_browser_page_asks(tmp_path):
    client, _, _ = make(tmp_path)
    url = "/library/voices"
    assert client.get(url, headers={"Sec-Fetch-Site": "same-site"}).status_code == 403
    assert client.get(url, headers={"Sec-Fetch-Site": "same-origin"}).status_code == 403
    assert client.get(url, headers={"Sec-Fetch-Site": "cross-site", "Origin": "http://tauri.localhost"}).status_code == 200
    assert client.get(url, headers={"Sec-Fetch-Site": "same-site", "Referer": "http://localhost:1420/"}).status_code == 200
    assert client.get(url, headers={"Sec-Fetch-Site": "none"}).status_code == 200
    assert client.get(url).status_code == 200
    assert client.get(url, headers={"Sec-Fetch-Site": "same-site", "Origin": "http://localhost:9999"}).status_code == 403
    assert client.get("/devices", headers={"Sec-Fetch-Site": "same-site"}).status_code == 200  # stage 1 unchanged


class SlowSaveLibrary(Library):
    """add_clip blocks until released, like a slow disk."""

    def __init__(self, data):
        super().__init__(data)
        self.entered = threading.Event()
        self.release = threading.Event()

    def add_clip(self, *args, **kwargs):
        self.entered.set()
        self.release.wait(10)
        return super().add_clip(*args, **kwargs)


def test_a_voice_delete_waits_for_a_take_that_is_being_saved(tmp_path):
    add_voice(tmp_path, "demo")
    library = SlowSaveLibrary(tmp_path)
    mic = CapturingStreams()
    importer = Importer(library, Echo())
    studio = Studio(library, Recorder(mic), importer, lambda: DEVICES, prompts=PROMPTS)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    app = create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=[*ALLOWED_HOSTS, "testserver"],
                     studio=studio, library=library, importer=importer)
    client = TestClient(app)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    speak(mic)
    stopped, stop_box, stop_thread = run_in_thread(lambda: client.post("/library/record/stop"))
    assert library.entered.wait(10)
    deleted, delete_box, delete_thread = run_in_thread(lambda: client.delete(f"/library/voices/{voice['id']}"))
    assert not deleted.wait(0.4)  # held back until the save finishes
    library.release.set()
    stop_thread.join(10)
    delete_thread.join(10)
    assert stop_box["result"].json()["accepted"] is True
    assert delete_box["result"].json() == {"deleted": True}
    assert not (tmp_path / "voices" / voice["id"]).exists()
    trashed = list((tmp_path / "trash").glob(f"{voice['id']}-*"))
    assert len(trashed) == 1 and (trashed[0] / "clips" / "0001.orig.wav").exists()


def test_a_full_disk_while_saving_a_take_is_a_readable_400_and_leaves_the_recorder_idle(tmp_path, monkeypatch):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)

    def no_space(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("aivoice.library.write_wav", no_space)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    speak(mic)
    res = client.post("/library/record/stop")
    monkeypatch.undo()
    assert res.status_code == 400
    assert res.json()["detail"] == "Could not save the clip: No space left on device"
    assert client.get("/library/status").json()["recording"] is False
    assert client.get(f"/library/voices/{voice['id']}/clips").json() == []
    assert record(client, mic, voice["id"])["clip"]["index"] == 1  # recording works again once there is room


def test_disk_failures_on_the_other_library_writes_are_readable_400s(tmp_path, monkeypatch):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    record(client, mic, voice["id"])

    def no_space(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("aivoice.library._atomic_json", no_space)
    created = client.post("/library/voices", json={"name": "Another"})
    edited = client.patch(f"/library/voices/{voice['id']}/clips/1", json={"text": "changed"})
    deleted = client.delete(f"/library/voices/{voice['id']}/clips/1")
    monkeypatch.undo()
    assert (created.status_code, created.json()["detail"]) == (400, "Could not create the voice: No space left on device")
    assert (edited.status_code, edited.json()["detail"]) == (400, "Could not save the transcript: No space left on device")
    assert (deleted.status_code, deleted.json()["detail"]) == (400, "Could not delete the clip: No space left on device")
    assert [v["id"] for v in client.get("/library/voices").json()] == [voice["id"]]
    assert len(client.get(f"/library/voices/{voice['id']}/clips").json()) == 1
    assert client.get(f"/library/voices/{voice['id']}/clips/1/audio").status_code == 200


def test_deleting_a_clip_over_http_moves_it_to_the_trash(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    record(client, mic, voice["id"])
    assert client.delete(f"/library/voices/{voice['id']}/clips/1").json() == {"deleted": True}
    trash = tmp_path / "trash" / voice["id"] / "clips"
    assert sorted(p.name for p in trash.iterdir()) == ["0001.orig.wav", "0001.txt", "0001.wav"]
    assert client.get(f"/library/voices/{voice['id']}/clips/1/audio").status_code == 400


def _refusals(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "Refused" in r.getMessage()]


def test_a_refused_library_request_is_logged_with_the_headers_that_caused_it(tmp_path, caplog):
    client, _, _ = make(tmp_path)
    headers = {"Origin": "https://evil.example", "Referer": "https://evil.example/page?q=1", "Sec-Fetch-Site": "cross-site"}
    with caplog.at_level("WARNING", logger="aivoice"):
        assert client.get("/library/voices", headers=headers).status_code == 403
    lines = _refusals(caplog)
    assert len(lines) == 1
    for part in ("GET", "/library/voices", "https://evil.example", "https://evil.example/page?q=1", "cross-site"):
        assert part in lines[0]


def test_a_refused_library_request_with_no_headers_named_says_so(tmp_path, caplog):
    client, _, _ = make(tmp_path)
    with caplog.at_level("WARNING", logger="aivoice"):
        assert client.get("/library/status", headers={"Sec-Fetch-Site": "same-site"}).status_code == 403
    lines = _refusals(caplog)
    assert len(lines) == 1 and "/library/status" in lines[0] and "same-site" in lines[0]
    assert "Origin=None" in lines[0] and "Referer=None" in lines[0]


def test_a_library_write_refused_by_the_app_wide_origin_check_is_logged_too(tmp_path, caplog):
    client, _, _ = make(tmp_path)
    with caplog.at_level("WARNING", logger="aivoice"):
        res = client.post("/library/voices", json={"name": "X"}, headers={"Origin": "http://unexpected.localhost"})
    assert res.status_code == 403
    lines = _refusals(caplog)
    assert len(lines) == 1
    assert "POST" in lines[0] and "/library/voices" in lines[0] and "http://unexpected.localhost" in lines[0]


def test_allowed_library_requests_log_no_refusal_and_long_headers_are_cut(tmp_path, caplog):
    client, _, _ = make(tmp_path)
    with caplog.at_level("WARNING", logger="aivoice"):
        assert client.get("/library/voices", headers={"Origin": "http://tauri.localhost"}).status_code == 200
        assert client.get("/library/voices").status_code == 200
        assert _refusals(caplog) == []
        assert client.get("/library/voices", headers={"Referer": "https://evil.example/" + "x" * 5000}).status_code == 403
    lines = _refusals(caplog)
    assert len(lines) == 1 and len(lines[0]) < 600


def test_pages_are_listed_and_a_page_read_is_guarded_like_a_take(tmp_path):
    client, mic, importer = make(tmp_path)
    voice = new_voice(client)
    pages = client.get("/library/pages").json()
    assert len(pages) >= 5 and {"index", "title", "text", "words", "minutes"} <= set(pages[0])
    body = {"voice_id": voice["id"], "input_device": 1, "page": 0}
    assert client.post("/live/start", json={"input_device": 1, "mic_device": 3, "voice": "demo"}).status_code == 200
    res = client.post("/library/record/page", json=body)
    assert res.status_code == 400 and res.json()["detail"] == "Stop live mode before recording"
    client.post("/live/stop")
    res = client.post("/library/record/page", json=body)
    assert res.status_code == 200 and res.json()["recording"] is True
    stopped = client.post("/library/record/stop").json()  # nothing was read: refused, nothing imported
    assert stopped["accepted"] is False
    assert client.get("/library/status").json()["recording"] is False
