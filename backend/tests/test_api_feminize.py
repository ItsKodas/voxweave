import io
import time
import wave

import numpy as np
from fastapi.testclient import TestClient

from aivoice.api import create_app
from aivoice.devices import Device
from aivoice.feminize import FeminizeEngine
from aivoice.importer import Importer
from aivoice.library import Library
from aivoice.manager import LiveManager, StartConfig
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from tests.test_feminize import median_f0, vowel
from tests.test_live import FakeStreams, Identity
from tests.test_recorder import CapturingStreams

DEVICES = [Device(1, "Mic", 1, 0), Device(2, "Speakers", 0, 2), Device(3, "VoiceMeeter Input", 0, 8)]
ORIGIN = {"Origin": "http://tauri.localhost"}


def rig(tmp_path):
    data = tmp_path / "data"
    library = Library(data)
    voice = library.create("Kai")
    library.add_clip(voice.id, vowel(5.0), "A clip of five seconds.", "recorded")
    library.add_clip(voice.id, vowel(2.0, f0=130), "Short.", "recorded")
    voice = library.get(voice.id)
    studio = Studio(library, Recorder(CapturingStreams()), Importer(library, None), lambda: DEVICES, prompts=("One.",))
    manager = LiveManager(data, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    app = create_app(manager, data, lambda: DEVICES, [], allowed_hosts=["testserver"], studio=studio, library=library,
                     importer=None)
    return TestClient(app), library, voice, manager, data


def test_a_sample_is_a_feminized_clip_of_the_voice(tmp_path):
    client, library, voice, manager, _ = rig(tmp_path)
    res = client.post(f"/library/voices/{voice.id}/feminize/preview",
                      json={"pitch_semitones": 5, "formant_shift": 1.15, "breathiness": 0.0}, headers=ORIGIN)
    assert res.status_code == 200 and res.headers["content-type"] == "audio/wav"
    assert res.headers["x-clip-index"] == str(voice.clips[0].index)
    with wave.open(io.BytesIO(res.content)) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767
    assert len(audio) == 5 * 48000  # the five-second clip, not the short one
    assert abs(median_f0(audio) - 120 * 2 ** (5 / 12)) < 6


def test_bad_settings_and_a_voice_without_a_usable_clip_are_refused(tmp_path):
    client, library, voice, manager, _ = rig(tmp_path)
    res = client.post(f"/library/voices/{voice.id}/feminize/preview", json={"pitch_semitones": 40}, headers=ORIGIN)
    assert res.status_code == 400 and "pitch_semitones" in res.json()["detail"]
    empty = library.create("Empty")
    res = client.post(f"/library/voices/{empty.id}/feminize/preview", json={}, headers=ORIGIN)
    assert res.status_code == 400


def test_a_feminized_voice_is_made_and_reported(tmp_path):
    client, library, voice, manager, _ = rig(tmp_path)
    started = client.post(f"/library/voices/{voice.id}/feminize", json={"pitch_semitones": 4}, headers=ORIGIN).json()
    assert started["active"] and started["total"] == 2
    status = started
    for _ in range(600):
        status = client.get("/library/feminize/status").json()
        if status["finished"]:
            break
        time.sleep(0.05)
    assert status["finished"] and status["error"] is None and status["done"] == 2
    assert "Kai (feminized)" in [v["name"] for v in client.get("/library/voices").json()]


def test_the_live_preview_settings_are_saved_and_used_by_live_mode(tmp_path):
    client, library, voice, manager, data = rig(tmp_path)
    assert client.get("/library/feminize/settings").json() == {"pitch_semitones": 5.0, "formant_shift": 1.15,
                                                                 "breathiness": 0.15}
    saved = client.put("/library/feminize/settings", json={"pitch_semitones": 7, "formant_shift": 1.2,
                                                           "breathiness": 0.1}, headers=ORIGIN).json()
    assert saved == {"pitch_semitones": 7.0, "formant_shift": 1.2, "breathiness": 0.1}
    assert client.put("/library/feminize/settings", json={"formant_shift": 3}, headers=ORIGIN).status_code == 400
    assert any(v["id"] == "@feminize" for v in client.get("/voices").json())
    manager.start(StartConfig(input_device=1, mic_device=3, voice="@feminize"))
    engine = manager._session.processor.engine
    inner = getattr(engine, "_inner", engine)
    assert isinstance(inner, FeminizeEngine) and inner._settings.pitch_semitones == 7.0
    manager.stop()


def test_a_foreign_page_cannot_make_or_change_anything(tmp_path):
    client, library, voice, manager, _ = rig(tmp_path)
    evil = {"Origin": "https://evil.example"}
    assert client.post(f"/library/voices/{voice.id}/feminize", json={}, headers=evil).status_code == 403
    assert client.put("/library/feminize/settings", json={}, headers=evil).status_code == 403
    assert len(library.list()) == 1
