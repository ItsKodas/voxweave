import time

import numpy as np
import pytest

from aivoice.insight import FPS, WINDOW_SECONDS, Insight

SR = 48000
CHUNK = SR // 4  # 250 ms


def engine_insight(number, heard=110.0, sent=220.0, corrected=False, gated=False, frames=25):
    return {"number": number, "frames_per_s": 100.0, "heard": [heard] * frames, "sent": [sent] * frames,
            "corrected": [corrected] * frames, "gated": [gated] * frames}


def block(amp=0.3):
    return (amp * np.sin(np.arange(CHUNK) / 10)).astype(np.float32)


def test_a_chunk_adds_its_pitch_levels_and_timing():
    insight = Insight(SR)
    insight.add(block(), block(0.1), engine_insight(1, corrected=True), infer_ms=80.0, chunk_ms=250)
    snap = insight.snapshot()
    n = int(0.25 * FPS)
    assert snap["frames_per_s"] == FPS and len(snap["heard"]) == len(snap["sent"]) == n
    assert snap["heard"][0] == pytest.approx(110.0) and snap["sent"][0] == pytest.approx(220.0)
    assert all(snap["corrected"]) and not any(snap["gated"])
    assert snap["in_db"][0] == pytest.approx(20 * np.log10(0.3 / np.sqrt(2)), abs=1.0)
    assert snap["out_db"][0] < snap["in_db"][0]
    assert snap["chunks"] == [{"t": pytest.approx(0.25), "infer_ms": 80.0, "budget_ms": 250}]
    assert snap["seconds"] == pytest.approx(0.25)


def test_a_chunk_the_quiet_gate_closed_shows_as_gated_with_no_pitch():
    insight = Insight(SR)
    insight.add(block(), block(0.1), engine_insight(1), 50.0, 250)
    insight.add(block(0.001), np.zeros(CHUNK, np.float32), engine_insight(1), 5.0, 250)  # the engine did not run
    snap = insight.snapshot()
    n = int(0.25 * FPS)
    assert all(snap["gated"][n:]) and all(h == 0 for h in snap["heard"][n:])


def test_only_the_last_ten_seconds_are_kept():
    insight = Insight(SR)
    for i in range(60):  # 15 s
        insight.add(block(), block(), engine_insight(i + 1), 50.0, 250)
    snap = insight.snapshot()
    assert len(snap["heard"]) == WINDOW_SECONDS * FPS and snap["seconds"] == pytest.approx(15.0)
    assert len(snap["chunks"]) == WINDOW_SECONDS * 4


def test_voice_match_is_scored_about_every_second_from_recent_speech():
    scored = []

    def matcher(audio, sr):
        scored.append(len(audio) / sr)
        return 0.83

    insight = Insight(SR, matcher_factory=lambda: matcher, match_every=0.05)
    insight.start()
    try:
        for i in range(8):
            insight.add(block(), block(0.2), engine_insight(i + 1), 50.0, 250)
        deadline = time.time() + 5
        while not insight.snapshot()["match"] and time.time() < deadline:
            time.sleep(0.02)
    finally:
        insight.stop()
    snap = insight.snapshot()
    assert snap["match"] and snap["match"][-1]["value"] == pytest.approx(0.83) and snap["match_note"] is None
    assert max(scored) <= 3.01  # the last three seconds at most


def test_voice_match_waits_for_enough_speech():
    insight = Insight(SR, matcher_factory=lambda: (lambda a, sr: 0.9), match_every=0.05)
    insight.start()
    try:
        for i in range(8):
            insight.add(block(0.001), np.zeros(CHUNK, np.float32), engine_insight(i + 1, heard=0, sent=0, gated=True),
                        5.0, 250)
        time.sleep(0.3)
    finally:
        insight.stop()
    assert insight.snapshot()["match"] == []


def test_voice_match_explains_why_it_is_not_available():
    def unavailable():
        raise LookupError("Voice match needs the speaker model.")

    insight = Insight(SR, matcher_factory=unavailable, match_every=0.05)
    insight.start()
    try:
        deadline = time.time() + 5
        while insight.snapshot()["match_note"] is None and time.time() < deadline:
            time.sleep(0.02)
    finally:
        insight.stop()
    assert insight.snapshot()["match_note"] == "Voice match needs the speaker model."


# ---- inside live mode

class Reporting:
    """An engine that says what it did, like the real one."""
    def __init__(self):
        self.insight = None
        self.n = 0

    def convert(self, audio, sample_rate, pitch_shift, params):
        self.n += 1
        frames = params.chunk_ms // 10
        self.insight = {"number": self.n, "frames_per_s": 100.0, "heard": [120.0] * frames,
                        "sent": [240.0] * frames, "corrected": [False] * frames, "gated": [False] * frames}
        return audio.copy()


def test_live_mode_collects_insight_and_serves_it(tmp_path):
    from fastapi.testclient import TestClient

    from aivoice.api import create_app
    from aivoice.manager import LiveManager, StartConfig
    from tests.test_live import FakeStreams
    from tests.test_manager import DEVICES
    from tests.test_pretrained import add_voice

    add_voice(tmp_path, "demo")
    made = []

    def factory(voice):
        made.append(voice.id)
        return Insight(SR)

    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Reporting(), FakeStreams(), insight_factory=factory)
    client = TestClient(create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=["testserver"]))
    assert client.get("/live/insight").json() == {"running": False}
    manager.start(StartConfig(input_device=1, mic_device=3, voice="demo"))
    session = manager._session
    for _ in range(4):
        session.feed(np.full(session.processor.chunk, 0.1, np.float32))
    deadline = time.time() + 5
    while not client.get("/live/insight").json().get("chunks") and time.time() < deadline:
        time.sleep(0.02)
    data = client.get("/live/insight").json()
    assert data["running"] is True and made == ["demo"]
    assert data["chunks"] and data["heard"][-1] == 120.0 and data["sent"][-1] == 240.0
    manager.stop()
    assert client.get("/live/insight").json() == {"running": False}


def test_a_broken_insight_never_stops_the_voice(tmp_path):
    from tests.test_live import make as make_session

    class Broken:
        def add(self, *a, **k):
            raise RuntimeError("graphs broke")

        def start(self):
            pass

        def stop(self):
            pass

    session, *_ = make_session()
    session.insight = Broken()
    session.feed(np.full(session.processor.chunk, 0.1, np.float32))
    assert session.pump() is True and session.error is None
