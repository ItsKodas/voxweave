import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from aivoice.api import ALLOWED_HOSTS, create_app
from aivoice.importer import Importer
from aivoice.manager import LiveManager
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from tests.test_api_library import Echo
from tests.test_live import FakeStreams, Identity
from tests.test_manager import DEVICES
from tests.test_pretrained import add_voice
from tests.test_recorder import CapturingStreams
from tests.test_studio import PROMPTS
from tests.test_training import FAIL, PASS, Rig, line

START = {"input_device": 1, "mic_device": 3, "voice": "demo"}
LIVE_REFUSED = "Training is running. Wait for it to finish or cancel it before starting live mode"
RECORD_REFUSED = "Training is running. Wait for it to finish or cancel it before recording"


class Setup:
    def __init__(self, tmp_path, with_training=True):
        self.rig = Rig(tmp_path)
        self.rig.build()
        self.data = self.rig.data
        self.data.mkdir(parents=True, exist_ok=True)
        add_voice(self.data, "demo")
        self.refreshed = []
        self.mic = CapturingStreams()
        self.live_streams = FakeStreams()
        importer = Importer(self.rig.library, Echo())
        studio = Studio(self.rig.library, Recorder(self.mic), importer, lambda: DEVICES, prompts=PROMPTS)
        self.manager = LiveManager(self.data, lambda: DEVICES, lambda v: Identity(), self.live_streams,
                                   refresh_devices=lambda: self.refreshed.append(1))
        self.app = create_app(self.manager, self.data, lambda: DEVICES, ["DmlExecutionProvider"],
                              allowed_hosts=[*ALLOWED_HOSTS, "testserver"], studio=studio, library=self.rig.library,
                              importer=importer, training=self.rig.manager if with_training else None)
        self.client = TestClient(self.app)
        self.voice = self.rig.voice()

    def start_training(self, **body):
        return self.client.post("/training/start", json={"voice_id": self.voice, **body})


@pytest.fixture
def s(tmp_path):
    setup = Setup(tmp_path)
    yield setup
    setup.manager.stop()


def test_status_and_environment_routes(s):
    idle = s.client.get("/training/status").json()
    assert idle["state"] == "idle" and idle["voice_id"] is None
    assert s.client.get(f"/training/status?voice_id={s.voice}").json()["voice_id"] == s.voice
    report = s.client.get("/training/environment").json()
    assert report["ok"] is True and report["checks"] == PASS["checks"]
    assert s.rig.verify_calls == [False]
    s.client.get("/training/environment")
    assert s.rig.verify_calls == [False]  # a passing report is remembered
    assert s.client.get("/training/environment?refresh=true").json()["checks"] == PASS["checks"]
    assert s.rig.verify_calls == [False, False]


def test_start_status_cancel_happy_path(s):
    res = s.start_training(epochs=30, batch_size=4)
    assert res.status_code == 200 and res.json()["state"] == "running" and res.json()["voice_id"] == s.voice
    assert len(s.rig.launches) == 1
    assert "30" in s.rig.launches[0][0]
    s.rig.progress(s.voice, line(percent=40.0, epoch=3, epochs=30))
    status = s.client.get("/training/status").json()
    assert status["state"] == "running" and status["percent"] == 40.0
    cancelled = s.client.post("/training/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelling"


def test_start_validation_errors_are_400_with_the_managers_message(s):
    res = s.start_training(epochs=5)
    assert res.status_code == 400 and res.json()["detail"] == "Epochs must be between 20 and 1000"
    res = s.start_training(batch_size=3)
    assert res.status_code == 400 and res.json()["detail"] == "Batch size must be 4, 8 or 16"
    res = s.start_training(epochs="many")
    assert res.status_code == 400 and res.json()["detail"] == "Epochs must be between 20 and 1000"
    res = s.client.post("/training/start", json={"voice_id": "nope"})
    assert res.status_code == 400 and res.json()["detail"] == "Unknown voice"
    res = s.start_training(resume=True)
    assert res.status_code == 400 and "no interrupted training run" in res.json()["detail"]
    s.rig.report = FAIL
    res = s.start_training()
    assert res.status_code == 400 and "torch sees no GPU" in res.json()["detail"]
    assert s.rig.launches == []
    assert s.client.post("/training/cancel").json()["detail"] == "Training is not running"


def test_training_is_refused_while_live_mode_runs(s):
    assert s.client.post("/live/start", json=START).status_code == 200
    res = s.start_training()
    assert res.status_code == 400 and res.json()["detail"] == "Stop live mode before training"
    assert s.rig.launches == [] and s.rig.manager.status()["state"] == "idle"


def test_training_is_refused_while_recording(s):
    body = {"voice_id": s.voice, "input_device": 1, "prompt": 0}
    assert s.client.post("/library/record/start", json=body).status_code == 200
    res = s.start_training()
    assert res.status_code == 400 and res.json()["detail"] == "Stop recording before training"
    assert s.rig.launches == []


def test_live_start_is_refused_while_training_runs_or_cancels(s):
    assert s.start_training().status_code == 200
    res = s.client.post("/live/start", json=START)
    assert res.status_code == 400 and res.json()["detail"] == LIVE_REFUSED
    s.client.post("/training/cancel")
    res = s.client.post("/live/start", json=START)
    assert res.status_code == 400 and res.json()["detail"] == LIVE_REFUSED
    assert s.client.get("/live/stats").json()["running"] is False and s.live_streams.streams == []


def test_recording_start_is_refused_while_training_runs_or_cancels(s):
    body = {"voice_id": s.voice, "input_device": 1, "prompt": 0}
    assert s.start_training().status_code == 200
    res = s.client.post("/library/record/start", json=body)
    assert res.status_code == 400 and res.json()["detail"] == RECORD_REFUSED
    s.client.post("/training/cancel")
    assert s.client.post("/library/record/start", json=body).json()["detail"] == RECORD_REFUSED
    assert s.mic.streams == []


def test_live_works_again_once_training_ends(s):
    s.start_training()
    s.rig.finish(s.voice, 0, ok=True, percent=100.0)
    assert s.client.get("/training/status").json()["state"] == "done"
    assert s.client.post("/live/start", json=START).status_code == 200


def test_device_refresh_stays_allowed_during_training(s):
    s.start_training()
    assert s.client.post("/devices/refresh").status_code == 200
    assert s.refreshed == [1]


def test_delete_is_refused_for_the_training_voice_but_allowed_for_others(s):
    other = s.rig.voice("Other")
    s.rig.checkpoint(other)
    s.start_training()
    res = s.client.delete(f"/library/voices/{s.voice}")
    assert res.status_code == 400 and res.json()["detail"] == "Stop training before deleting this voice"
    assert (s.data / "voices" / s.voice).is_dir()
    assert s.client.delete(f"/library/voices/{other}").json() == {"deleted": True}
    assert not (s.data / "voices" / other).exists() and not s.rig.experiment_dir(other).exists()


def test_a_failure_to_forget_does_not_fail_the_delete(s, caplog):
    other = s.rig.voice("Other")

    def broken(voice_id):
        raise OSError("disk on fire")

    s.rig.manager.forget = broken
    with caplog.at_level(logging.WARNING):
        res = s.client.delete(f"/library/voices/{other}")
    assert res.status_code == 200 and not (s.data / "voices" / other).exists()
    assert "disk on fire" in caplog.text


def test_training_routes_refuse_foreign_origins(s):
    for method, url in (("get", "/training/status"), ("get", "/training/environment"),
                        ("post", "/training/cancel")):
        for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"},
                        {"Referer": "https://evil.example/p"}, {"Sec-Fetch-Site": "same-site"}):
            assert getattr(s.client, method)(url, headers=headers).status_code == 403, (url, headers)
    res = s.client.post("/training/start", json={"voice_id": s.voice}, headers={"Sec-Fetch-Site": "cross-site"})
    assert res.status_code == 403 and s.rig.launches == []
    app_headers = {"Sec-Fetch-Site": "cross-site", "Origin": "http://tauri.localhost"}
    assert s.client.get("/training/status", headers=app_headers).status_code == 200
    assert s.client.get("/training/status", headers={"Sec-Fetch-Site": "none"}).status_code == 200


def test_trained_voice_is_listed_by_voices_after_the_job_completes(s):
    assert [v["id"] for v in s.client.get("/voices").json() if v["id"] != "@feminize"] == ["demo"]
    s.start_training()
    models = s.data / "voices" / s.voice / "models" / "rvc"
    models.mkdir(parents=True)
    (models / "model.onnx").write_bytes(b"x")
    (models / "voice.json").write_text(json.dumps({"sample_rate": 40000}), encoding="utf-8")
    s.rig.finish(s.voice, 0, ok=True, percent=100.0)
    assert s.client.get("/training/status").json()["state"] == "done"
    assert [v["id"] for v in s.client.get("/voices").json() if v["id"] != "@feminize"] == ["demo", s.voice]


def test_without_a_training_manager_there_are_no_training_routes(tmp_path):
    s = Setup(tmp_path, with_training=False)
    assert s.client.get("/training/status").status_code == 404
    assert s.client.post("/live/start", json=START).status_code == 200
    s.manager.stop()
    other = s.rig.voice("Other")
    assert s.client.delete(f"/library/voices/{other}").json() == {"deleted": True}


def test_shutting_the_app_down_stops_the_worker(s):
    s.start_training()
    with s.client:
        pass
    assert s.rig.handles[0].kills == 1
    assert s.rig.manager.status()["state"] == "interrupted"


class SlowVerify:
    """Holds the environment check open until released."""

    def __init__(self, s):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.inner = s.rig.verify
        s.rig.manager._verify = self

    def __call__(self, fast):
        self.entered.set()
        assert self.release.wait(10)
        return self.inner(fast)


def test_refresh_does_not_probe_while_training_runs(s):
    assert s.client.get("/training/environment").json()["ok"] is True
    s.start_training()
    calls = len(s.rig.verify_calls)
    report = s.client.get("/training/environment?refresh=true").json()
    assert report["ok"] is True and report["checked_while_training"] is True
    assert len(s.rig.verify_calls) == calls
    s.client.post("/training/cancel")
    assert s.client.get("/training/environment?refresh=true").json()["checked_while_training"] is True
    assert len(s.rig.verify_calls) == calls


def test_refresh_probes_when_idle(s):
    s.client.get("/training/environment")
    report = s.client.get("/training/environment?refresh=true").json()
    assert "checked_while_training" not in report and len(s.rig.verify_calls) == 2


def test_a_slow_environment_check_blocks_only_the_start_request(s):
    slow = SlowVerify(s)
    result = {}
    thread = threading.Thread(target=lambda: result.update(res=s.start_training()))
    thread.start()
    try:
        assert slow.entered.wait(10)
        assert s.client.post("/live/start", json=START).status_code == 200  # the audio lock is free
        s.client.post("/live/stop")
        assert s.client.get("/training/status").json()["state"] == "idle"  # polling does not wait either
        assert s.client.delete(f"/library/voices/{s.rig.voice('Other')}").status_code == 200
    finally:
        slow.release.set()
        thread.join(10)
    assert "res" in result and result["res"].status_code in (200, 400)


def test_not_ready_environment_is_a_400_before_any_lock_or_launch(s):
    s.rig.report = FAIL
    with s.app.state.audio_state:  # even with the audio lock held elsewhere the refusal comes back
        res = s.start_training()
    assert res.status_code == 400
    assert res.json()["detail"] == "Training is not set up: torch sees no GPU; ffmpeg not found on PATH"
    assert s.rig.launches == []


def test_training_routes_exist_without_a_studio(tmp_path):
    rig = Rig(tmp_path)
    rig.build()
    rig.data.mkdir(parents=True, exist_ok=True)
    add_voice(rig.data, "demo")
    manager = LiveManager(rig.data, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    client = TestClient(create_app(manager, rig.data, lambda: DEVICES, [], allowed_hosts=[*ALLOWED_HOSTS, "testserver"],
                                   training=rig.manager))
    voice = rig.voice()
    assert client.get("/training/status").status_code == 200
    assert client.post("/training/start", json={"voice_id": voice}).status_code == 200
    assert client.post("/live/start", json=START).status_code == 400


# Final review, F1 and F5: nothing that holds the audio lock or the voices lock ever performs a kill


class BlockedKill:
    """A training job whose kill is due, and whose kill would hang until released."""

    def __init__(self, s):
        self.entered = threading.Event()
        self.release = threading.Event()
        assert s.start_training().status_code == 200
        s.rig.manager.cancel()
        s.rig.clock += 90  # the worker ignored the cancel: whoever services the job next has to kill it
        self.handle = s.rig.handles[0]
        self.handle.on_kill = self.hang

    def hang(self):
        self.entered.set()
        assert self.release.wait(10)


def test_live_start_recording_and_delete_never_wait_for_a_kill_nor_start_one(s):
    blocked = BlockedKill(s)
    other = s.rig.voice("Other")
    record = {"voice_id": s.voice, "input_device": 1, "prompt": 0}
    answers = {}

    def ask():
        answers["live"] = s.client.post("/live/start", json=START)
        answers["record"] = s.client.post("/library/record/start", json=record)
        answers["delete"] = s.client.delete(f"/library/voices/{s.voice}")
        answers["other"] = s.client.delete(f"/library/voices/{other}")
        answers["refresh"] = s.client.post("/devices/refresh")

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    try:
        thread.join(5)
        assert not thread.is_alive(), "a route is waiting for the kill"
    finally:
        blocked.release.set()
        thread.join(10)
    assert blocked.handle.kills == 0 and not blocked.entered.is_set()
    assert (answers["live"].status_code, answers["live"].json()["detail"]) == (400, LIVE_REFUSED)
    assert (answers["record"].status_code, answers["record"].json()["detail"]) == (400, RECORD_REFUSED)
    assert answers["delete"].json()["detail"] == "Stop training before deleting this voice"
    assert answers["other"].json() == {"deleted": True} and answers["refresh"].status_code == 200
    assert (s.data / "voices" / s.voice).is_dir() and s.live_streams.streams == [] and s.mic.streams == []
    # The once-a-second status poll is where the kill happens.
    assert s.client.get("/training/status").json()["state"] == "cancelled" and blocked.handle.kills == 1


def test_the_routes_answer_while_the_status_poll_is_busy_killing(s):
    blocked = BlockedKill(s)
    poll = threading.Thread(target=lambda: s.client.get("/training/status"), daemon=True)
    poll.start()
    try:
        assert blocked.entered.wait(10)
        res = s.client.post("/live/start", json=START)
        assert (res.status_code, res.json()["detail"]) == (400, LIVE_REFUSED)
        res = s.client.delete(f"/library/voices/{s.voice}")
        assert res.json()["detail"] == "Stop training before deleting this voice"
        assert s.client.delete(f"/library/voices/{s.rig.voice('Other')}").json() == {"deleted": True}
    finally:
        blocked.release.set()
        poll.join(10)
    assert blocked.handle.kills == 1


def test_delete_of_the_training_voice_is_refused_even_when_status_cannot_be_worked_out(s):
    s.start_training()

    def broken():
        raise RuntimeError("status is broken")

    s.rig.manager._service = broken
    assert s.rig.manager.status(s.voice)["state"] == "idle"  # status swallows the error: it must not be the guard
    res = s.client.delete(f"/library/voices/{s.voice}")
    assert res.status_code == 400 and res.json()["detail"] == "Stop training before deleting this voice"
    assert (s.data / "voices" / s.voice).is_dir()


# Final review, F8: the full environment check starts a GPU process, so it never runs beside live mode or a job


def test_environment_is_not_probed_while_live_mode_runs(s):
    assert s.client.post("/live/start", json=START).status_code == 200
    report = s.client.get("/training/environment").json()
    assert report == {"ok": False, "ready": False, "unknown": True, "checks": [], "checked_while_live": True}
    assert s.client.get("/training/environment?refresh=true").json()["unknown"] is True
    assert s.rig.verify_calls == []
    res = s.start_training()  # and a start is refused before its own check would run
    assert res.status_code == 400 and res.json()["detail"] == "Stop live mode before training"
    assert s.rig.verify_calls == [] and s.rig.launches == []

    s.client.post("/live/stop")
    assert s.client.get("/training/environment").json() == {**PASS, "ready": True}
    assert s.rig.verify_calls == [False]
    assert s.client.post("/live/start", json=START).status_code == 200
    s.rig.report = FAIL
    report = s.client.get("/training/environment?refresh=true").json()  # the last known report, and it is kept
    assert report == {**PASS, "ready": True, "checked_while_live": True}
    assert s.rig.verify_calls == [False]
    s.client.post("/live/stop")
    assert s.client.get("/training/environment").json()["ready"] is True and s.rig.verify_calls == [False]


def test_a_failing_report_is_what_is_known_while_live_mode_runs(s):
    s.rig.report = FAIL
    assert s.client.get("/training/environment").json()["ready"] is False
    assert s.client.post("/live/start", json=START).status_code == 200
    report = s.client.get("/training/environment").json()
    assert report == {**FAIL, "ready": False, "checked_while_live": True} and s.rig.verify_calls == [False]


def test_a_time_limited_run_is_planned_and_started_with_the_chosen_clips_and_epochs(s, monkeypatch):
    import aivoice.quicktrain as quicktrain

    chosen = {"clips": ["0001.orig.wav"], "epochs": 77, "audio_minutes": 5.0, "available_minutes": 9.0,
              "estimated_minutes": 10.0, "seconds_per_slice_epoch": 0.05, "clip_count": 1}
    asked = []
    monkeypatch.setattr(quicktrain, "plan_for", lambda data, library, voice_id, minutes: asked.append(minutes) or chosen)
    assert s.client.get(f"/training/plan?voice_id={s.voice}&minutes=10").json()["epochs"] == 77
    res = s.start_training(minutes=10)
    assert res.status_code == 200
    job = s.data / "voices" / s.voice / "training"
    assert (job / "selected-clips.json").read_text().count("0001.orig.wav") == 1
    assert json.loads((job / "job.json").read_text())["epochs"] == 77
    assert asked == [10.0, 10.0]


def test_a_time_limit_that_cannot_work_is_refused_before_anything_starts(s):
    res = s.start_training(minutes=10)  # this voice's clip has no audio file: no usable audio
    assert res.status_code == 400 and "usable audio" in res.json()["detail"]
    assert s.start_training(minutes="ten").status_code == 400
    assert s.client.get("/training/status").json()["state"] == "idle"
