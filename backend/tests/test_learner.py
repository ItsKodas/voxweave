import numpy as np
import pytest

from aivoice.learner import Learner, agreed_changes

SR = 48000
CHUNK = SR // 2  # 500 ms


def insight(number, voiced=True):
    frames = 50
    return {"number": number, "frames_per_s": 100.0, "heard": [110.0 if voiced else 0.0] * frames,
            "sent": [250.0 if voiced else 0.0] * frames, "corrected": [False] * frames,
            "gated": [not voiced] * frames}


def speech(amp=0.2):
    return (amp * np.sin(np.arange(CHUNK) / 12)).astype(np.float32)


START = {"pitch_shift": 12, "likeness": 0.3,
         "advanced": {"creak_repair": True, "protect": 0.33}, "input_gain": 4.0, "speech_level_db": -25.0}


class Rig:
    def __init__(self, tune_results=None):
        self.now = [0.0]
        self.applied = []
        self.tune_results = list(tune_results or [])
        self.tuned_on = []
        self.learner = Learner(SR, dict(START), self.measure, self.tune if tune_results is not None else None,
                               self.applied.append, clock=lambda: self.now[0], measure_every=10.0, pass_every=60.0)
        self.n = 0

    def measure(self, audio):
        return {"pitch_hz": 104.0, "low_hz": 88.0, "high_hz": 150.0, "level_db": -32.0, "input_gain": 2.0}

    def tune(self, audio, settings, progress):
        self.tuned_on.append((len(audio) / SR, dict(settings)))
        progress(1, "Trying")
        return self.tune_results.pop(0)

    def talk(self, seconds, voiced=True):
        for _ in range(int(seconds * 2)):
            self.n += 1
            self.learner.add(speech() if voiced else np.zeros(CHUNK, np.float32), insight(self.n, voiced))
            self.now[0] += 0.5

    def pause(self):
        self.talk(1.0, voiced=False)


def test_only_clear_speech_is_kept_up_to_two_minutes():
    rig = Rig()
    rig.talk(10)
    rig.talk(5, voiced=False)
    assert rig.learner.status()["speech_seconds"] == pytest.approx(10.0)
    rig.talk(200)
    assert rig.learner.status()["speech_seconds"] == pytest.approx(120.0)


def test_the_voice_is_re_measured_and_the_input_boost_moves_gradually_in_a_pause():
    rig = Rig()
    rig.talk(20)
    rig.learner.tick()
    assert rig.learner.status()["learned"]["pitch_hz"] == 104.0
    assert rig.applied == []  # still talking: nothing changes mid-word
    rig.pause()
    rig.learner.tick()
    gain = rig.applied[-1]["input_gain"]
    assert 2.0 < gain < 4.0  # part of the way from 4.0 toward the measured 2.0
    assert rig.applied[-1]["speech_level_db"] == pytest.approx(-32.0 + 20 * np.log10(gain), abs=0.01)


def test_a_tuning_pass_needs_45_seconds_of_speech_and_changes_only_what_two_passes_agree_on():
    better = {"pitch_shift": 14, "likeness": 0.6, "advanced": {"creak_repair": False, "protect": 0.33}}
    rig = Rig(tune_results=[better, better])
    rig.talk(30)
    rig.learner.tick()
    assert rig.tuned_on == []  # not enough speech yet
    rig.talk(30)
    rig.learner.tick()
    assert len(rig.tuned_on) == 1 and rig.tuned_on[0][1]["pitch_shift"] == 12
    rig.pause()
    rig.learner.tick()
    assert not any("pitch_shift" in a for a in rig.applied)  # one pass alone changes nothing
    rig.now[0] += 61
    rig.talk(5)
    rig.learner.tick()
    rig.pause()
    rig.learner.tick()
    change = next(a for a in rig.applied if "pitch_shift" in a)
    assert change["pitch_shift"] == 13  # a semitone at a time toward 14
    assert change["overrides"]["index_rate"] == pytest.approx(0.4)  # 10 percent at a time toward 60
    assert change["overrides"]["creak_repair"] is False
    status = rig.learner.status()
    assert status["passes"] == 2 and status["current"]["pitch_shift"] == 13


def test_agreed_changes():
    current = {"pitch_shift": 12, "likeness": 0.3, "advanced": {"f0_method": "rmvpe"}}
    up = {"pitch_shift": 13, "likeness": 0.5, "advanced": {"f0_method": "dio"}}
    down = {"pitch_shift": 11, "likeness": 0.1, "advanced": {"f0_method": "rmvpe"}}
    assert agreed_changes(current, None, up) == {}
    assert agreed_changes(current, down, up) == {}  # they disagree
    assert agreed_changes(current, up, up) == {"pitch_shift": 13, "overrides": {"index_rate": 0.4, "f0_method": "dio"}}


def test_turning_learning_off_puts_back_what_it_started_with_unless_saved():
    better = {"pitch_shift": 14, "likeness": 0.6, "advanced": {"creak_repair": False, "protect": 0.33}}
    rig = Rig(tune_results=[better, better])
    rig.talk(60)
    rig.learner.tick()
    rig.now[0] += 61
    rig.talk(2)
    rig.learner.tick()
    rig.pause()
    rig.learner.tick()
    rig.learner.revert()
    back = rig.applied[-1]
    assert back["pitch_shift"] == 12 and back["input_gain"] == 4.0 and back["speech_level_db"] == -25.0
    assert back["overrides"] == {"index_rate": 0.3, "creak_repair": True, "protect": 0.33}


def test_save_hands_over_the_learned_voice_settings_and_speech():
    rig = Rig()
    rig.talk(30)
    rig.learner.tick()
    saved = rig.learner.saved()
    assert saved["profile"]["pitch_hz"] == 104.0 and saved["settings"]["pitch_shift"] == 12
    assert 25 <= len(saved["sample"]) / SR <= 30


def test_learning_waits_while_live_mode_is_falling_behind():
    rig = Rig(tune_results=[{"pitch_shift": 12, "likeness": 0.3, "advanced": START["advanced"]}])
    behind = [True]
    rig.learner._behind = lambda: behind[0]
    rig.talk(60)
    rig.learner.tick()
    assert rig.tuned_on == [] and "behind" in rig.learner.status()["message"]
    behind[0] = False
    rig.learner.tick()
    assert len(rig.tuned_on) == 1
