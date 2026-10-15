import json
import threading

import numpy as np
import pytest

from aivoice.deeptune import CLARITY_MARGIN, MIN_GAIN, Candidate, Tuner, choose, word_error_rate

SR = 48000


def cand(shift, rate, sim, clarity):
    return Candidate(shift, rate, sim, clarity)


def test_word_error_rate():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0
    assert word_error_rate("the quick brown fox", "The quick, brown fox!") == 0  # case and punctuation do not count
    assert word_error_rate("the quick brown fox", "the brown fox") == pytest.approx(0.25)
    assert word_error_rate("", "anything") == 0


def test_choose_takes_the_most_alike_setting_that_keeps_the_words():
    baseline = cand(13, 0.3, 0.85, 0.95)
    tried = [baseline, cand(13, 0.5, 0.86, 0.94), cand(13, 0.75, 0.87, 0.95 - CLARITY_MARGIN - 0.01),
             cand(13, 1.0, 0.88, 0.80)]
    assert choose(tried, baseline) == tried[1]  # the more alike ones lose words


def test_choose_keeps_the_current_setting_unless_the_gain_is_real():
    baseline = cand(13, 0.3, 0.850, 0.95)
    tried = [baseline, cand(12, 0.3, 0.850 + MIN_GAIN / 2, 0.95)]
    assert choose(tried, baseline) == baseline
    tried.append(cand(14, 0.3, 0.850 + MIN_GAIN * 2, 0.95))
    assert choose(tried, baseline) == tried[-1]


def test_choose_leaves_a_current_setting_that_loses_words():
    baseline = cand(13, 1.0, 0.88, 0.70)
    tried = [baseline, cand(13, 0.5, 0.86, 0.95)]
    assert choose(tried, baseline) == tried[1]


# ---- the tuner, with stand-ins for the engine, the speaker model and the transcriber

class FakeEngine:
    """Remembers the settings of the conversion in progress; the 'voice' is the input made louder."""
    pitch_range = None
    speech_level_db = None
    noise_position = None

    def __init__(self):
        self.calls = []
        self.last = None

    def convert(self, audio, sample_rate, pitch_shift, params):
        self.last = (pitch_shift, params.index_rate)
        self.calls.append(self.last)
        return np.asarray(audio, dtype=np.float32) * 0.5


class FakeSpeaker:
    """Sounds most like the voice one semitone above the suggestion with likeness 0.75."""
    def __init__(self, engine, best=(8, 0.75)):
        self.engine, self.best = engine, best

    def embedding(self, audio, sample_rate):
        if self.engine.last is None:  # the voice's own clips
            return np.array([1.0, 0.0])
        shift, rate = self.engine.last
        miss = abs(shift - self.best[0]) * 0.1 + abs(rate - self.best[1]) * 0.5
        return np.array([1.0, miss])


class FakeTranscriber:
    """Hears every word, except with likeness 1.0."""
    def __init__(self, engine):
        self.engine = engine

    def transcribe(self, audio16k):
        if self.engine.last is not None and self.engine.last[1] >= 1.0:
            return "mumble"
        return "the ferry leaves the harbour at seven each morning"


class FakeCalibration:
    def __init__(self, sample=True):
        self._sample = np.sin(np.arange(SR * 6) / 20).astype(np.float32) * 0.2 if sample else None

    def sample(self):
        return self._sample

    def suggestion(self, voice_id):
        return {"pitch_shift": 7}

    def voice_pitch_range(self, voice_id):
        return (150.0, 300.0)

    def profile(self):
        return type("P", (), {"level_db": -30.0, "input_gain": 2.0})()

    def wrap(self, engine):
        return engine


class FakeLibrary:
    def __init__(self, tmp_path, clips=3):
        self.tmp_path = tmp_path
        self.voice = type("V", (), {"id": "woman", "clips": [type("C", (), {"index": i, "kind": "speech"})() for i in range(clips)]})()

    def get(self, voice_id):
        if voice_id != "woman":
            raise ValueError("Unknown voice")
        return self.voice

    def clip_audio_path(self, voice_id, index):
        from aivoice.audiofile import write_wav
        path = self.tmp_path / f"clip{index}.wav"
        if not path.exists():
            write_wav(path, np.sin(np.arange(SR * 2) / 15).astype(np.float32) * 0.2, SR)
        return path


def make(tmp_path, sample=True, clips=3):
    engine = FakeEngine()
    voice = type("PV", (), {"id": "woman", "name": "Woman"})()
    tuner = Tuner(tmp_path, FakeLibrary(tmp_path, clips), FakeCalibration(sample), lambda v: engine,
                  FakeTranscriber(engine), lambda: FakeSpeaker(engine), list_voices=lambda data: [voice])
    return tuner, engine


def test_the_tuner_finds_the_pitch_and_likeness_that_sound_most_like_the_voice(tmp_path):
    tuner, engine = make(tmp_path)
    tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3)
    tuner.wait(30)
    status = tuner.status()
    assert status["active"] is False and status["error"] is None
    result = status["result"]
    assert (result["pitch_shift"], result["likeness"]) == (8, 0.75)
    assert result["similarity"] > result["baseline"]["similarity"]
    assert result["clarity"] == pytest.approx(1.0)
    assert {(t["pitch_shift"], t["likeness"]) for t in result["tried"]} >= {(5, 0.3), (9, 0.3), (8, 1.0)}
    saved = json.loads((tmp_path / "calibration" / "tuned" / "woman.json").read_text(encoding="utf-8"))
    assert (saved["pitch_shift"], saved["likeness"]) == (8, 0.75)
    assert status["step"] == status["steps"]


def test_the_tuner_needs_a_kept_calibration_reading(tmp_path):
    tuner, _ = make(tmp_path, sample=False)
    with pytest.raises(ValueError, match="Calibrate"):
        tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3)


def test_the_tuner_needs_the_voices_recordings(tmp_path):
    tuner, _ = make(tmp_path, clips=0)
    with pytest.raises(ValueError, match="recordings"):
        tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3)


def test_the_tuner_refuses_an_unknown_voice_and_a_second_run(tmp_path):
    tuner, engine = make(tmp_path)
    with pytest.raises(ValueError):
        tuner.start("nobody", quality=1, pitch_shift=0, likeness=0.3)
    gate = threading.Event()
    original = engine.convert

    def slow(*a, **k):
        gate.wait(5)
        return original(*a, **k)

    engine.convert = slow
    tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3)
    with pytest.raises(ValueError, match="already"):
        tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3)
    assert tuner.active
    tuner.cancel()
    gate.set()
    tuner.wait(30)
    status = tuner.status()
    assert status["active"] is False and status["result"] is None and "stopped" in status["message"].lower()
