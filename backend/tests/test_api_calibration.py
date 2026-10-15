import pytest
import numpy as np
from fastapi.testclient import TestClient

from aivoice.api import create_app
from aivoice.calibration import Calibration, GatedEngine
from aivoice.devices import Device
from aivoice.manager import LiveManager, StartConfig
from tests.test_calibration import FakeRecorder
from tests.test_live import FakeStreams, Identity
from tests.test_pretrained import add_voice

DEVICES = [Device(1, "Headset Mic", 1, 0), Device(2, "Headset", 0, 2), Device(3, "VoiceMeeter Input", 0, 8)]
ORIGIN = {"Origin": "http://tauri.localhost"}


class Studio:
    recording = False


def rig(tmp_path, studio=None):
    add_voice(tmp_path, "demo")
    recorder = FakeRecorder()
    calibration = Calibration(tmp_path, recorder, lambda: DEVICES)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda voice: Identity(), FakeStreams(), calibration=calibration)
    app = create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=["testserver"], calibration=calibration)
    if studio is not None:  # the routes only read `.recording`
        from aivoice.api_calibration import calibration_router
        app = create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=["testserver"])
        app.include_router(calibration_router(calibration, studio, lambda: bool(manager.stats()["running"]),
                                              app.state.audio_state, ["http://tauri.localhost"]))
    return TestClient(app), manager, calibration, recorder


def test_a_calibration_is_recorded_measured_and_reported(tmp_path):
    client, manager, _, _ = rig(tmp_path)
    assert client.get("/calibration/status").json()["profile"] is None
    started = client.post("/calibration/start", json={"input_device": 1}, headers=ORIGIN).json()
    assert started["recording"] and started["text"] and started["total_seconds"] == 18.0
    done = client.post("/calibration/finish", headers=ORIGIN).json()
    assert not done["recording"] and 110 < done["profile"]["pitch_hz"] < 130
    assert client.get("/calibration/status").json()["profile"] == done["profile"]
    assert client.get("/calibration/suggestion", params={"voice": "demo"}).json()["pitch_shift"] is None
    assert client.delete("/calibration/profile", headers=ORIGIN).json()["profile"] is None
    manager.stop()


def test_calibrating_and_live_mode_and_device_refresh_keep_out_of_each_others_way(tmp_path):
    client, manager, calibration, _ = rig(tmp_path)
    client.post("/calibration/start", json={"input_device": 1}, headers=ORIGIN)
    live = client.post("/live/start", json={"input_device": 1, "mic_device": 3, "voice": "demo"}, headers=ORIGIN)
    assert live.status_code == 400 and "calibration" in live.json()["detail"]
    refresh = client.post("/devices/refresh", headers=ORIGIN)
    assert refresh.status_code == 400 and "calibration" in refresh.json()["detail"]
    assert client.post("/calibration/cancel", headers=ORIGIN).json()["recording"] is False
    assert client.post("/live/start", json={"input_device": 1, "mic_device": 3, "voice": "demo"},
                       headers=ORIGIN).status_code == 200
    refused = client.post("/calibration/start", json={"input_device": 1}, headers=ORIGIN)
    assert refused.status_code == 400 and "Stop live mode" in refused.json()["detail"] and not calibration.active
    manager.stop()


def test_calibration_is_refused_while_a_clip_is_being_recorded(tmp_path):
    studio = Studio()
    studio.recording = True
    client, manager, calibration, _ = rig(tmp_path, studio)
    refused = client.post("/calibration/start", json={"input_device": 1}, headers=ORIGIN)
    assert refused.status_code == 400 and "Stop recording" in refused.json()["detail"] and not calibration.active
    manager.stop()


def test_bad_requests_get_plain_messages_and_foreign_pages_get_nothing(tmp_path):
    client, manager, calibration, _ = rig(tmp_path)
    assert client.post("/calibration/finish", headers=ORIGIN).status_code == 400
    assert client.post("/calibration/start", json={"input_device": 2}, headers=ORIGIN).status_code == 400
    foreign = client.post("/calibration/start", json={"input_device": 1}, headers={"Origin": "https://evil.example"})
    assert foreign.status_code == 403 and not calibration.active
    manager.stop()


def test_live_mode_uses_the_calibration_only_when_there_is_one_and_it_is_wanted(tmp_path):
    client, manager, calibration, _ = rig(tmp_path)
    manager.start(StartConfig(input_device=1, mic_device=3, voice="demo"))
    assert isinstance(manager._session.processor.engine, Identity)  # no profile yet: the engine as it is
    manager.stop()
    calibration.start(1)
    calibration.finish()
    manager.start(StartConfig(input_device=1, mic_device=3, voice="demo"))
    engine = manager._session.processor.engine
    assert isinstance(engine, GatedEngine)
    window = np.zeros(manager._session.processor.window, dtype=np.float32)
    assert not engine.convert(window, 48000, 0.0, manager._session.processor.params).any()
    manager.start(StartConfig(input_device=1, mic_device=3, voice="demo", use_calibration=False))
    assert isinstance(manager._session.processor.engine, Identity)
    manager.stop()


def test_a_clip_cannot_be_recorded_during_a_calibration_recording_and_the_other_way_round(tmp_path):
    from aivoice.importer import Importer
    from aivoice.library import Library
    from aivoice.recorder import Recorder
    from aivoice.studio import Studio as RealStudio
    from tests.test_recorder import CapturingStreams

    add_voice(tmp_path, "demo")
    library = Library(tmp_path)
    studio = RealStudio(library, Recorder(CapturingStreams()), Importer(library, None), lambda: DEVICES,
                        prompts=("One sentence.",))
    calibration = Calibration(tmp_path, FakeRecorder(), lambda: DEVICES, library)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda voice: Identity(), FakeStreams(), calibration=calibration)
    client = TestClient(create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=["testserver"], studio=studio,
                                   library=library, importer=None, calibration=calibration))
    voice = client.post("/library/voices", json={"name": "V"}).json()
    take = {"voice_id": voice["id"], "input_device": 1, "prompt": 0}
    client.post("/calibration/start", json={"input_device": 1})
    refused = client.post("/library/record/start", json=take)
    assert refused.status_code == 400 and "calibration" in refused.json()["detail"] and not studio.recording
    client.post("/calibration/cancel")
    assert client.post("/library/record/start", json=take).status_code == 200
    refused = client.post("/calibration/start", json={"input_device": 1})
    assert refused.status_code == 400 and "Stop recording" in refused.json()["detail"]
    client.post("/library/record/cancel")


def test_live_mode_gives_the_engine_the_voices_pitch_range(tmp_path):
    from aivoice.library import Library
    from tests.test_calibration import voice_like

    library = Library(tmp_path)
    voice = library.create("Ranged")
    library.add_clip(voice.id, voice_like(4.0, 220.0), "words", "recorded")
    folder = tmp_path / "voices" / voice.id / "models" / "rvc"
    folder.mkdir(parents=True)
    (folder / "model.onnx").write_bytes(b"x")
    (folder / "voice.json").write_text('{"name": "Ranged", "sample_rate": 40000}')

    class Ranged(Identity):
        pitch_range = None

    engine = Ranged()
    calibration = Calibration(tmp_path, FakeRecorder(), lambda: DEVICES, library)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: engine, FakeStreams(), calibration=calibration)
    manager.start(StartConfig(input_device=1, mic_device=3, voice=voice.id))
    low, high = engine.pitch_range
    assert low == pytest.approx(220, rel=0.05) and high == pytest.approx(220, rel=0.05)
    manager.stop()


def test_live_mode_gives_the_engine_the_speakers_calibrated_level(tmp_path):
    import json

    add_voice(tmp_path, "demo")
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "profile.json").write_text(json.dumps(dict(
        pitch_hz=110, pitch_low_hz=90, pitch_high_hz=140, level_db=-37.0, noise_db=-70, gate_db=-64,
        input_gain=4.0, device="Headset Mic", created=5)))

    class Levelled(Identity):
        speech_level_db = None

    engine = Levelled()
    calibration = Calibration(tmp_path, FakeRecorder(), lambda: DEVICES)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: engine, FakeStreams(), calibration=calibration)
    manager.start(StartConfig(input_device=1, mic_device=3, voice="demo"))
    assert engine.speech_level_db == pytest.approx(-37.0 + 20 * np.log10(4.0))  # the level after the gain
    manager.stop()
    manager.start(StartConfig(input_device=1, mic_device=3, voice="demo", use_calibration=False))
    assert engine.speech_level_db is None  # learned from speech instead
    manager.stop()
