import threading

import pytest
from fastapi.testclient import TestClient

from aivoice.api import ALLOWED_HOSTS, create_app
from aivoice.importer import Importer
from aivoice.manager import LiveManager
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from aivoice.tts_reference import NO_CLIP
from tests import test_training
from tests.test_api_library import Echo
from tests.test_live import FakeStreams, Identity
from tests.test_manager import DEVICES
from tests.test_pretrained import add_voice
from tests.test_recorder import CapturingStreams
from tests.test_studio import PROMPTS
from tests.test_training import FakeProc
from tests.test_tts import FAIL, PASS, TEXT, WAIT, Rig

START = {"input_device": 1, "mic_device": 3, "voice": "demo"}
ALREADY = "Already speaking. Stop it first."
SPEAK_WHILE_LIVE = "Stop live mode before speaking"
SPEAK_WHILE_RECORDING = "Stop recording before speaking"
SPEAK_WHILE_TRAINING = "Training is running. Wait for it to finish or cancel it before speaking"
LIVE_REFUSED = "Stop the speech before starting live mode"
RECORD_REFUSED = "Stop the speech before recording"
TRAINING_REFUSED = "Stop the speech before training"
REFRESH_REFUSED = "Stop the speech before refreshing devices"
DELETE_REFUSED = "Stop the speech before deleting a voice"
NOT_SET_UP = "Text to speech is not set up: torch sees no GPU; model files wrong"


class Setup:
    """The whole app on fakes: the text-to-speech rig and the training rig of their own tests on one data folder,
    live mode and the recorder on fake streams."""

    def __init__(self, tmp_path, with_tts=True, with_training=True):
        self.rig = Rig(tmp_path)
        self.rig.devices = list(DEVICES)
        self.rig.build()
        self.training = test_training.Rig(tmp_path)
        self.training.build()
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
                              importer=importer, training=self.training.manager if with_training else None,
                              tts=self.rig.manager if with_tts else None)
        self.client = TestClient(self.app)
        self.voice = self.rig.voice().id
        self.trainable = self.training.voice("Trainable")

    def speak(self, **body):
        return self.client.post("/tts/speak", json={"voice_id": self.voice, "text": TEXT, "preset": "fast",
                                                    "mic_device": 3, "monitor_device": 2, **body})

    def speaking(self):
        """A request in progress: the worker has been asked for the reference clip and has not answered."""
        self.rig.auto = False
        assert self.speak().status_code == 200
        worker = self.rig.worker()
        worker.expect("set_reference")
        assert self.rig.manager.is_active()
        return worker

    def loaded(self):
        """A request that has ended: the worker is loaded and idle."""
        assert self.speak().status_code == 200
        assert self.rig.finish()["error"] is None
        worker, = self.rig.workers
        assert self.client.get("/tts/status").json()["engine"] == "ready"
        return worker

    def start_training(self):
        return self.client.post("/training/start", json={"voice_id": self.trainable})

    def record(self):
        return self.client.post("/library/record/start",
                                json={"voice_id": self.voice, "input_device": 1, "prompt": 0})

    def nothing_spoken(self):
        assert self.rig.launches == [] and self.rig.player.starts == [] and self.rig.tmp_folders() == []
        assert not self.rig.manager.is_active()


@pytest.fixture
def s(tmp_path):
    setup = Setup(tmp_path)
    yield setup
    setup.manager.stop()
    for worker in setup.rig.workers:
        worker.die()
    setup.rig.manager.shutdown()


def refused(res, status, detail):
    assert (res.status_code, res.json()) == (status, {"detail": detail})


def in_thread(call):
    result = {}
    thread = threading.Thread(target=lambda: result.update(res=call()), daemon=True)
    thread.start()
    return thread, result


def answered(call):
    """The response to a request that must not wait for a lock the test is holding: made on another thread, so
    that a route which does wait fails the test instead of hanging it."""
    thread, result = in_thread(call)
    thread.join(WAIT)
    assert not thread.is_alive(), "the route is waiting for a lock"
    return result["res"]


# ---------------------------------------------------------------- the routes themselves

def test_status_when_nothing_has_been_spoken(s):
    assert s.client.get("/tts/status").json() == {
        "engine": "stopped", "state": "idle", "voice_id": None, "preset": None, "piece": 0, "pieces": 0,
        "error": None, "saved_path": None, "started": None}


def test_environment_route_remembers_a_passing_report_and_checks_again_on_refresh(s):
    assert s.client.get("/tts/environment").json() == {**PASS, "ready": True}
    assert s.rig.verify_calls == [False]
    s.client.get("/tts/environment")
    assert s.rig.verify_calls == [False]
    s.rig.report = FAIL
    assert s.client.get("/tts/environment?refresh=true").json() == {**FAIL, "ready": False}
    assert s.rig.verify_calls == [False, False]


def test_voices_lists_every_library_voice_with_its_model_and_reference_clip(s):
    trained = s.rig.voice("Trained", seconds=5.0, model=True)
    empty = s.rig.voice("Empty", seconds=0)
    listed = {v["id"]: v for v in s.client.get("/tts/voices").json()}
    assert set(listed) == {s.voice, trained.id, empty.id, s.trainable}
    assert listed[s.voice] == {"id": s.voice, "name": "Narrator", "has_model": False, "reference": {
        "ok": True, "seconds": 8.0, "message": "Clip 1 (8.0 seconds) will be used as the reference."}}
    assert listed[trained.id]["has_model"] is True and listed[trained.id]["reference"]["seconds"] == 5.0
    assert listed[empty.id] == {"id": empty.id, "name": "Empty", "has_model": False,
                                "reference": {"ok": False, "seconds": None, "message": NO_CLIP}}


def test_voices_does_not_offer_a_model_whose_file_is_gone(s):
    trained = s.rig.voice("Trained", model=True)
    assert {v["id"]: v["has_model"] for v in s.client.get("/tts/voices").json()}[trained.id] is True
    (s.data / "voices" / trained.id / "models" / "rvc" / "model.onnx").unlink()
    assert {v["id"]: v["has_model"] for v in s.client.get("/tts/voices").json()}[trained.id] is False
    refused(s.speak(voice_id=trained.id, preset="best"), 400,
            "Best needs this voice's trained model, and it has none. Train the voice first, or choose Fast or "
            "Balanced.")


def test_voices_does_not_call_a_voice_usable_when_its_reference_clip_lost_its_audio(s):
    (s.data / "voices" / s.voice / "clips" / "0001.orig.wav").unlink()
    listed = {v["id"]: v for v in s.client.get("/tts/voices").json()}
    message = "The audio of clip 1, this voice's reference clip, is missing. Delete that clip or record it again."
    assert listed[s.voice]["reference"] == {"ok": False, "seconds": None, "message": message}
    refused(s.speak(), 400, message)  # the same words speak refuses with


def test_speak_starts_the_request_and_status_follows_it_to_the_end(s):
    s.rig.auto = False
    res = s.speak()
    assert res.status_code == 200
    started = res.json()
    assert started["state"] in ("preparing", "speaking") and started["voice_id"] == s.voice
    assert started["preset"] == "fast" and started["pieces"] == 3 and started["error"] is None
    assert s.rig.player.starts == [{"mic": 3, "monitor": 2}]
    worker = s.rig.worker()
    worker.expect("set_reference")
    assert s.client.get("/tts/status").json()["state"] == "preparing"
    s.rig.auto = True
    worker.emit(event="reference", seconds=8.0)
    done = s.rig.finish()
    assert done["state"] == "idle" and done["error"] is None and done["piece"] == 3
    assert s.client.get("/tts/status").json() == done and done["engine"] == "ready"
    assert len(s.rig.player.played) == 3


def test_speak_without_a_listen_device_plays_to_the_virtual_mic_only(s):
    assert s.speak(monitor_device=None).status_code == 200
    s.rig.finish()
    assert s.rig.player.starts == [{"mic": 3}]


def test_speak_can_save_the_result(s):
    assert s.speak(save=True).status_code == 200
    saved = s.rig.finish()["saved_path"]
    assert saved is not None and saved.endswith(f"-{s.voice}.wav")
    assert s.client.get("/tts/status").json()["saved_path"] == saved


def test_stop_ends_the_request_and_keeps_the_worker(s):
    worker = s.speaking()
    s.rig.auto = True
    stopping = {}
    thread = threading.Thread(target=lambda: stopping.update(res=s.client.post("/tts/stop")))
    thread.start()
    assert s.rig.player.stopped.wait(WAIT)
    worker.emit(event="reference", seconds=8.0)  # the answer the stopped request was still owed
    thread.join(WAIT)
    res = stopping["res"]
    assert res.status_code == 200 and res.json()["state"] == "idle" and res.json()["error"] is None
    assert worker.kills == 0 and not s.rig.manager.is_active()
    assert s.client.post("/tts/stop").json()["state"] == "idle"  # harmless when nothing is spoken


@pytest.mark.parametrize("body, detail", [
    ({"voice_id": "nope"}, "Unknown voice"),
    ({"voice_id": "../x"}, "Unknown voice"),
    ({"voice_id": 7}, "Unknown voice"),
    ({"voice_id": None}, "Unknown voice"),
    ({"text": ""}, "There is no text to speak."),
    ({"text": None}, "There is no text to speak."),
    ({"text": ["a"]}, "There is no text to speak."),
    ({"text": "a" * 5001}, "The text is too long (5001 characters). Keep it under 5000 characters."),
    ({"text": "... !!!"}, "There is nothing to say in that text"),
    ({"preset": "turbo"}, "Unknown preset: turbo. Choose fast, balanced or best."),
    ({"preset": ["fast"]}, "Unknown preset. Choose fast, balanced or best."),
    ({"preset": None}, "Unknown preset. Choose fast, balanced or best."),
    ({"mic_device": "3"}, "mic_device must be a device number"),
    ({"monitor_device": True}, "monitor_device must be a device number"),
    ({"monitor_device": 2.5}, "monitor_device must be a device number"),
    ({"save": "yes"}, "save must be true or false"),
])
def test_what_needs_no_look_at_the_devices_is_refused_before_the_environment_is_checked(s, body, detail):
    refused(s.speak(**body), 400, detail)
    s.nothing_spoken()
    assert s.rig.verify_calls == []  # the first check starts a GPU process: not for a request that cannot be done


@pytest.mark.parametrize("body, detail", [
    ({"preset": "best"}, "Best needs this voice's trained model, and it has none. Train the voice first, or "
                         "choose Fast or Balanced."),
    ({"mic_device": None, "monitor_device": None}, "Choose a virtual mic or a listen device"),
    ({"mic_device": 2}, "Monitor and virtual mic must be different devices"),
    ({"mic_device": 1}, "Headset Mic is not an output device"),
    ({"mic_device": 99}, "Audio device 99 is no longer available"),
])
def test_what_cannot_be_spoken_is_a_400_with_a_readable_message_and_starts_nothing(s, body, detail):
    refused(s.speak(**body), 400, detail)
    s.nothing_spoken()


def test_five_thousand_characters_are_accepted(s):
    assert s.speak(text="a" * 5000).status_code == 200


def test_a_body_that_is_not_an_object_is_refused(s):
    assert s.client.post("/tts/speak", json=["x"]).status_code == 422
    refused(s.client.post("/tts/speak", json={}), 400, "There is no text to speak.")  # judged in the manager's order
    s.nothing_spoken()


def test_the_request_is_started_under_the_audio_lock_and_the_voices_lock(s):
    held = []
    inner = s.rig.manager.speak

    def speak(*args, **kwargs):
        held.append((s.app.state.audio_state.locked(), s.app.state.voices_lock.locked()))
        return inner(*args, **kwargs)

    s.rig.manager.speak = speak
    assert s.speak().status_code == 200
    assert held == [(True, True)]  # so neither a live start nor a delete of the voice can slip in beside it


def test_speak_is_refused_when_the_environment_is_not_set_up(s):
    s.rig.report = FAIL
    with s.app.state.audio_state:  # the check is made, and refused, before the audio lock is asked for
        refused(answered(s.speak), 400, NOT_SET_UP)
    s.nothing_spoken()


# ---------------------------------------------------------------- one activity at a time: speak is refused

def test_speak_is_refused_while_live_mode_runs(s):
    assert s.client.post("/live/start", json=START).status_code == 200
    refused(s.speak(), 409, SPEAK_WHILE_LIVE)
    s.nothing_spoken()
    assert s.rig.verify_calls == []  # not even the environment check: it would start a GPU process beside live
    s.client.post("/live/stop")
    assert s.speak().status_code == 200


def test_speak_is_refused_while_recording(s):
    assert s.record().status_code == 200
    refused(s.speak(), 409, SPEAK_WHILE_RECORDING)
    s.nothing_spoken()
    s.client.post("/library/record/cancel")
    assert s.speak().status_code == 200


def test_speak_is_refused_while_training_runs_or_cancels(s):
    assert s.start_training().status_code == 200
    refused(s.speak(), 409, SPEAK_WHILE_TRAINING)
    s.client.post("/training/cancel")
    refused(s.speak(), 409, SPEAK_WHILE_TRAINING)
    s.nothing_spoken()
    assert s.rig.verify_calls == []
    s.training.finish(s.trainable, 0, cancelled=True)
    assert s.client.get("/training/status").json()["state"] == "cancelled"
    assert s.speak().status_code == 200


def test_a_second_speak_is_refused_while_one_is_in_progress(s):
    s.speaking()
    refused(s.speak(), 409, ALREADY)
    assert len(s.rig.launches) == 1 and len(s.rig.player.starts) == 1


# ---------------------------------------------------------------- one activity at a time: the others are refused

def test_live_start_is_refused_while_speaking_and_allowed_again_afterwards(s):
    worker = s.speaking()
    refused(s.client.post("/live/start", json=START), 400, LIVE_REFUSED)
    assert s.client.get("/live/stats").json()["running"] is False and s.live_streams.streams == []
    assert s.rig.manager.is_active() and worker.kills == 0  # and the speech goes on
    s.rig.auto = True
    worker.emit(event="reference", seconds=8.0)
    s.rig.finish()
    assert s.client.post("/live/start", json=START).status_code == 200  # a loaded, idle worker is no obstacle
    assert worker.kills == 0


def test_recording_is_refused_while_speaking_and_allowed_again_afterwards(s):
    worker = s.speaking()
    refused(s.record(), 400, RECORD_REFUSED)
    assert s.mic.streams == [] and s.rig.manager.is_active()
    s.rig.auto = True
    worker.emit(event="reference", seconds=8.0)
    s.rig.finish()
    assert s.record().status_code == 200 and worker.kills == 0


def test_training_is_refused_while_speaking_and_the_speech_goes_on(s):
    worker = s.speaking()
    refused(s.start_training(), 400, TRAINING_REFUSED)
    assert s.training.launches == [] and s.training.verify_calls == []  # no GPU check beside the speech either
    assert s.rig.manager.is_active() and worker.kills == 0


def test_speech_that_began_after_the_first_look_still_refuses_the_training(s):
    def speak_meanwhile(fast):  # the training start has passed its first checks and has not taken the audio lock
        s.rig.auto = False
        s.rig.manager.speak(s.voice, TEXT, "fast", monitor_device=2)
        return test_training.PASS

    s.training.manager._verify = speak_meanwhile
    refused(s.start_training(), 400, TRAINING_REFUSED)
    assert s.training.launches == []
    assert s.rig.manager.is_active() and s.rig.worker().kills == 0  # and the speech was not unloaded under it


def test_device_refresh_is_refused_while_speaking(s):
    s.speaking()
    refused(s.client.post("/devices/refresh"), 400, REFRESH_REFUSED)
    assert s.refreshed == []
    s.rig.workers[0].die()
    s.rig.finish()
    assert s.client.post("/devices/refresh").status_code == 200 and s.refreshed == [1]


def test_deleting_a_voice_is_refused_while_speaking(s):
    other = s.rig.voice("Other").id
    worker = s.speaking()
    refused(s.client.delete(f"/library/voices/{s.voice}"), 400, DELETE_REFUSED)
    refused(s.client.delete(f"/library/voices/{other}"), 400, DELETE_REFUSED)
    assert (s.data / "voices" / s.voice).is_dir() and (s.data / "voices" / other).is_dir()
    worker.die()
    s.rig.finish()
    assert s.client.delete(f"/library/voices/{s.voice}").json() == {"deleted": True}


def test_a_reference_clip_deleted_between_requests_is_a_plain_refusal(s):
    s.loaded()
    assert s.client.delete(f"/library/voices/{s.voice}/clips/1").status_code == 200
    refused(s.speak(), 400, NO_CLIP)
    assert not s.rig.manager.is_active() and s.client.get("/tts/status").json()["state"] == "idle"


# ---------------------------------------------------------------- live mode that is still letting go of its devices

class SlowLiveStop:
    """Live mode running, with an output stream whose close hangs until released: POST /live/stop has made live
    mode read "not running" and is still closing the devices."""

    def __init__(self, s):
        assert s.client.post("/live/start", json=START).status_code == 200
        self.s = s
        self.entered = threading.Event()
        self.release = threading.Event()
        self.streams = list(s.live_streams.streams)
        output = self.streams[-1]
        closing = output.close

        def close():
            self.entered.set()
            assert self.release.wait(WAIT)
            closing()

        output.close = close
        self.thread, self.result = in_thread(lambda: s.client.post("/live/stop"))
        assert self.entered.wait(WAIT)
        assert s.client.get("/live/stats").json()["running"] is False  # which is all the busy checks can see

    def all_closed(self):
        return all(stream.closed for stream in self.streams)

    def finish(self, *threads):
        for thread in threads:
            thread.join(0.2)  # time for a request that does not wait for the stop to get to the devices
        self.release.set()
        for thread in (self.thread, *threads):
            thread.join(WAIT)
            assert not thread.is_alive()
        assert self.result["res"].status_code == 200


def test_speak_arriving_while_live_mode_is_stopping_never_opens_the_output_beside_it(s):
    stopping = SlowLiveStop(s)
    opened_after_close = []
    start = s.rig.player.start

    def recorded(devices, gains=None):
        opened_after_close.append(stopping.all_closed())
        return start(devices, gains)

    s.rig.player.start = recorded
    thread, result = in_thread(s.speak)
    stopping.finish(thread)
    assert result["res"].status_code in (200, 409)  # held until live mode let go, or refused; never beside it
    assert opened_after_close in ([True], [])
    if result["res"].status_code == 409:
        s.nothing_spoken()


def test_recording_arriving_while_live_mode_is_stopping_never_opens_the_microphone_beside_it(s):
    stopping = SlowLiveStop(s)
    opened_after_close = []
    opening = s.mic.input

    def recorded(device, samplerate, on_audio):
        opened_after_close.append(stopping.all_closed())
        return opening(device, samplerate, on_audio)

    s.mic.input = recorded
    thread, result = in_thread(s.record)
    stopping.finish(thread)
    assert result["res"].status_code in (200, 400)
    assert opened_after_close in ([True], [])


def test_training_arriving_while_live_mode_is_stopping_never_starts_beside_it(s):
    stopping = SlowLiveStop(s)
    launched_after_close = []
    launch = s.training.manager._launch

    def recorded(argv, log_path):
        launched_after_close.append(stopping.all_closed())
        return launch(argv, log_path)

    s.training.manager._launch = recorded
    thread, result = in_thread(s.start_training)
    stopping.finish(thread)
    assert result["res"].status_code in (200, 400)
    assert launched_after_close in ([True], [])


def test_live_stop_still_answers_and_reports_idle(s):
    assert s.client.post("/live/start", json=START).status_code == 200
    res = s.client.post("/live/stop")
    assert res.status_code == 200 and res.json()["running"] is False
    assert all(stream.closed for stream in s.live_streams.streams)
    assert s.client.post("/live/stop").status_code == 200  # harmless when it is not running
    assert s.speak().status_code == 200


# ---------------------------------------------------------------- training start and the loaded worker

def test_training_start_unloads_an_idle_worker_first(s):
    worker = s.loaded()
    res = s.start_training()
    assert res.status_code == 200 and res.json()["state"] == "running"
    assert worker.kills == 1 and s.client.get("/tts/status").json()["engine"] == "stopped"
    assert len(s.training.launches) == 1


def test_training_start_is_refused_when_the_worker_cannot_be_unloaded(s):
    s.rig.children(FakeProc(7006, killable=False))
    s.loaded()
    res = s.start_training()
    assert res.status_code == 400
    assert res.json()["detail"].startswith("The voice engine could not be stopped (process 7006 is still running)")
    assert s.training.launches == [] and s.client.get("/training/status").json()["state"] == "idle"
    refused(s.start_training(), 400, res.json()["detail"])  # and it stays refused, never "already starting"


class HangingKill:
    """An idle, loaded worker whose kill hangs until released."""

    def __init__(self, s):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.worker = s.loaded()
        self.worker.on_kill = self.hang

    def hang(self):
        self.entered.set()
        assert self.release.wait(WAIT)


def test_nothing_can_be_spoken_while_training_start_is_unloading_the_worker(s):
    hanging = HangingKill(s)
    thread, result = in_thread(s.start_training)
    try:
        assert hanging.entered.wait(WAIT)
        refused(s.speak(), 409, SPEAK_WHILE_TRAINING)
        refused(s.start_training(), 400, "Training is already starting")
        assert len(s.rig.launches) == 1 and s.training.launches == []
        # The worker is ended without the audio lock or the voices lock: these answer meanwhile.
        assert s.client.post("/devices/refresh").status_code == 200
        assert s.client.delete(f"/library/voices/{s.rig.voice('Other').id}").json() == {"deleted": True}
    finally:
        hanging.release.set()
        thread.join(WAIT)
    assert result["res"].status_code == 200 and len(s.training.launches) == 1
    assert s.client.get("/tts/status").json()["engine"] == "stopped" and len(s.rig.launches) == 1


def test_live_mode_started_while_the_worker_was_being_unloaded_still_refuses_the_training(s):
    hanging = HangingKill(s)
    thread, result = in_thread(s.start_training)
    try:
        assert hanging.entered.wait(WAIT)
        assert s.client.post("/live/start", json=START).status_code == 200
    finally:
        hanging.release.set()
        thread.join(WAIT)
    refused(result["res"], 400, "Stop live mode before training")
    assert s.training.launches == []
    s.client.post("/live/stop")
    assert s.speak().status_code == 200  # the start that was refused no longer holds anything back


def test_a_start_refused_by_the_training_manager_leaves_speaking_possible(s):
    refused(s.client.post("/training/start", json={"voice_id": "nope"}), 400, "Unknown voice")
    assert s.speak().status_code == 200


# ---------------------------------------------------------------- no kill and no slow check under a shared lock

def test_a_worker_due_to_be_ended_is_ended_before_speak_takes_the_audio_lock(s):
    hanging = HangingKill(s)
    s.rig.clock += 601  # idle for too long: whoever looks next has to end it
    thread, result = in_thread(s.speak)
    try:
        assert hanging.entered.wait(WAIT)
        assert s.client.post("/live/start", json=START).status_code == 200  # the audio lock is free
    finally:
        hanging.release.set()
        thread.join(WAIT)
    refused(result["res"], 409, SPEAK_WHILE_LIVE)  # and the check under the lock saw live mode
    assert len(s.rig.launches) == 1 and len(s.rig.player.starts) == 1


class SlowVerify:
    """Holds the environment check open until released."""

    def __init__(self, s):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.inner = s.rig.verify
        s.rig.manager._verify = self

    def __call__(self, fast):
        self.entered.set()
        assert self.release.wait(WAIT)
        return self.inner(fast)


def test_a_slow_environment_check_blocks_only_the_speak_request(s):
    slow = SlowVerify(s)
    thread, result = in_thread(s.speak)
    try:
        assert slow.entered.wait(WAIT)
        assert s.client.get("/tts/status").json()["state"] == "idle"
        assert s.client.get("/tts/voices").status_code == 200
        assert s.client.post("/devices/refresh").status_code == 200  # the audio lock is free
        assert s.client.delete(f"/library/voices/{s.rig.voice('Other').id}").status_code == 200  # the voices lock too
    finally:
        slow.release.set()
        thread.join(WAIT)
    assert result["res"].status_code == 200


def test_status_voices_and_stop_never_wait_for_the_audio_lock(s):
    s.speaking()
    s.rig.auto = True
    with s.app.state.audio_state:
        assert answered(lambda: s.client.get("/tts/status")).json()["state"] == "preparing"
        assert answered(lambda: s.client.get("/tts/voices")).status_code == 200
        assert answered(lambda: s.client.get("/tts/environment")).status_code == 200
        thread, result = in_thread(lambda: s.client.post("/tts/stop"))
        assert s.rig.player.stopped.wait(WAIT)
        s.rig.workers[0].emit(event="reference", seconds=8.0)
        thread.join(WAIT)
        assert not thread.is_alive() and result["res"].json()["state"] == "idle"


# ---------------------------------------------------------------- the GPU check never runs beside another activity

def test_the_environment_is_not_probed_while_live_mode_runs(s):
    assert s.client.post("/live/start", json=START).status_code == 200
    assert s.client.get("/tts/environment?refresh=true").json() == {
        "ok": False, "ready": False, "unknown": True, "checks": [], "checked_while_live": True}
    assert s.rig.verify_calls == []
    s.client.post("/live/stop")
    assert s.client.get("/tts/environment").json() == {**PASS, "ready": True}
    assert s.client.post("/live/start", json=START).status_code == 200
    s.rig.report = FAIL
    assert s.client.get("/tts/environment?refresh=true").json() == {**PASS, "ready": True, "checked_while_live": True}
    assert s.rig.verify_calls == [False]


def test_the_environment_is_not_probed_while_training_runs(s):
    assert s.start_training().status_code == 200
    report = s.client.get("/tts/environment?refresh=true").json()
    assert report["unknown"] is True and report["checked_while_training"] is True
    assert s.rig.verify_calls == []


def test_the_environment_is_not_probed_while_a_training_start_is_unloading_the_worker(s):
    hanging = HangingKill(s)
    thread, result = in_thread(s.start_training)
    try:
        assert hanging.entered.wait(WAIT)
        calls = len(s.rig.verify_calls)
        report = s.client.get("/tts/environment?refresh=true").json()
        assert report["ok"] is True and report["checked_while_training"] is True
        assert len(s.rig.verify_calls) == calls
    finally:
        hanging.release.set()
        thread.join(WAIT)
    assert result["res"].status_code == 200


def test_the_training_environment_is_not_probed_while_speaking(s):
    s.speaking()
    report = s.client.get("/training/environment?refresh=true").json()
    assert report == {"ok": False, "ready": False, "unknown": True, "checks": [], "checked_while_speaking": True}
    assert s.training.verify_calls == []
    s.rig.workers[0].die()
    s.rig.finish()
    assert s.client.get("/training/environment").json()["ok"] is True  # a loaded or failed worker is no obstacle


# ---------------------------------------------------------------- origin, absence, shutdown

def test_tts_routes_refuse_foreign_origins(s):
    for method, url in (("get", "/tts/status"), ("get", "/tts/environment"), ("get", "/tts/voices"),
                        ("post", "/tts/stop")):
        for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"},
                        {"Referer": "https://evil.example/p"}, {"Sec-Fetch-Site": "same-site"}):
            assert getattr(s.client, method)(url, headers=headers).status_code == 403, (url, headers)
    for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"}):
        res = s.client.post("/tts/speak", json={"voice_id": s.voice, "text": TEXT, "preset": "fast",
                                                "monitor_device": 2}, headers=headers)
        assert res.status_code == 403
    s.nothing_spoken()
    assert s.rig.verify_calls == []
    app_headers = {"Sec-Fetch-Site": "cross-site", "Origin": "http://tauri.localhost"}
    assert s.client.get("/tts/status", headers=app_headers).status_code == 200
    assert s.client.get("/tts/voices", headers={"Sec-Fetch-Site": "none"}).status_code == 200


def test_without_a_tts_manager_there_are_no_tts_routes_and_the_rest_works_as_before(tmp_path):
    s = Setup(tmp_path, with_tts=False)
    try:
        for url in ("/tts/status", "/tts/environment", "/tts/voices"):
            assert s.client.get(url).status_code == 404
        assert s.client.post("/tts/speak", json={}).status_code == 404
        assert s.client.post("/devices/refresh").status_code == 200
        assert s.record().status_code == 200
        s.client.post("/library/record/cancel")
        assert s.client.get("/training/environment").json()["ok"] is True
        assert s.start_training().status_code == 200
        s.client.post("/training/cancel")
        s.training.finish(s.trainable, 0, cancelled=True)
        assert s.client.get("/training/status").json()["state"] == "cancelled"
        assert s.client.post("/live/start", json=START).status_code == 200
        s.manager.stop()
        assert s.client.delete(f"/library/voices/{s.voice}").json() == {"deleted": True}
    finally:
        s.manager.stop()


def test_tts_routes_exist_without_a_studio_or_training(tmp_path):
    rig = Rig(tmp_path)
    rig.devices = list(DEVICES)
    rig.build()
    rig.data.mkdir(parents=True, exist_ok=True)
    add_voice(rig.data, "demo")
    manager = LiveManager(rig.data, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    client = TestClient(create_app(manager, rig.data, lambda: DEVICES, [], allowed_hosts=[*ALLOWED_HOSTS, "testserver"],
                                   library=rig.library, tts=rig.manager))
    voice = rig.voice().id
    try:
        assert [v["id"] for v in client.get("/tts/voices").json()] == [voice]
        rig.auto = False
        body = {"voice_id": voice, "text": TEXT, "preset": "fast", "monitor_device": 2}
        assert client.post("/tts/speak", json=body).status_code == 200
        refused(client.post("/live/start", json=START), 400, LIVE_REFUSED)
        rig.worker().die()
        rig.finish()
        assert client.post("/live/start", json=START).status_code == 200
        refused(client.post("/tts/speak", json=body), 409, SPEAK_WHILE_LIVE)
    finally:
        manager.stop()
        rig.manager.shutdown()


def test_shutting_the_app_down_stops_an_idle_worker(s):
    worker = s.loaded()
    with s.client:
        pass
    assert worker.kills == 1 and s.rig.manager.status()["engine"] == "stopped"
    with pytest.raises(ValueError, match="The app is closing"):
        s.rig.manager.speak(s.voice, TEXT, "fast", monitor_device=2)


def test_shutting_the_app_down_stops_a_request_in_progress_and_its_worker(s):
    worker = s.speaking()
    s.rig.auto = True  # the worker answers the cancel

    def answer():  # the answer the stopped request is still owed, once the shutdown has silenced the player
        s.rig.player.stopped.wait(WAIT)
        worker.emit(event="reference", seconds=8.0)

    threading.Thread(target=answer, daemon=True).start()
    with s.client:
        pass
    assert worker.kills == 1 and s.rig.player.stops >= 1
    status = s.rig.manager.status()
    assert status["state"] == "idle" and status["engine"] == "stopped" and s.rig.tmp_folders() == []


def test_shutting_the_app_down_stops_a_training_run_as_before(s):
    assert s.start_training().status_code == 200
    with s.client:
        pass
    assert s.training.handles[0].kills == 1 and s.training.manager.status()["state"] == "interrupted"
    assert s.rig.manager.status()["engine"] == "stopped"
