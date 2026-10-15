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
        from aivoice.presets import QUALITY_LEVELS

        quality = next(i for i, q in enumerate(QUALITY_LEVELS) if q["chunk_ms"] == params.chunk_ms)
        self.last = (pitch_shift, params.index_rate, quality)
        self.params = params
        self.calls.append(self.last)
        return np.asarray(audio, dtype=np.float32) * 0.5


class FakeSpeaker:
    """Sounds most like the voice one semitone above the suggestion, with likeness 0.75, at quality level 4."""
    def __init__(self, engine, best=(8, 0.75, 3)):
        self.engine, self.best = engine, best

    def embedding(self, audio, sample_rate):
        if self.engine.last is None:  # the voice's own clips
            return np.array([1.0, 0.0])
        shift, rate, quality = self.engine.last
        miss = abs(shift - self.best[0]) * 0.1 + abs(rate - self.best[1]) * 0.5 + abs(quality - self.best[2]) * 0.15
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
    assert (result["pitch_shift"], result["likeness"], result["quality"]) == (8, 0.75, 3)
    assert result["similarity"] > result["baseline"]["similarity"]
    assert result["clarity"] == pytest.approx(1.0)
    assert {(t["pitch_shift"], t["likeness"]) for t in result["tried"]} >= {(5, 0.3), (9, 0.3), (8, 1.0)}
    assert {t["quality"] for t in result["tried"]} == set(range(7))  # every quality level was tried
    # every advanced setting was tried on and off, and kept as it was since none made a difference
    assert result["advanced"] == {"f0_method": "rmvpe", "protect": 0.33, "volume_follow": 0.75, "natural_variation_st": 0.25,
                                  "pitch_cleanup": True, "creak_repair": True, "join_smoothing": True,
                                  "background_cut_db": 30.0}
    tried_adv = [t["advanced"] for t in result["tried"]]
    assert any(a["f0_method"] == "dio" for a in tried_adv) and any(a["creak_repair"] is False for a in tried_adv)
    assert any(a["background_cut_db"] == 0.0 for a in tried_adv) and any(a["protect"] == 0.5 for a in tried_adv)
    saved = json.loads((tmp_path / "calibration" / "tuned" / "woman.json").read_text(encoding="utf-8"))
    assert (saved["pitch_shift"], saved["likeness"], saved["quality"]) == (8, 0.75, 3)
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


def test_among_quality_levels_that_sound_as_alike_the_faster_one_wins():
    baseline = Candidate(13, 0.5, 0.80, 0.95, quality=1)
    tried = [baseline, Candidate(13, 0.5, 0.85, 0.95, quality=3), Candidate(13, 0.5, 0.855, 0.95, quality=6)]
    assert choose(tried, baseline, prefer_faster=True) == tried[1]  # level 7 is no better than level 4 and slower
    assert choose(tried, baseline) == tried[2]


def test_a_slower_quality_that_is_only_as_good_does_not_replace_the_current_one():
    baseline = Candidate(13, 0.5, 0.85, 0.95, quality=3)
    tried = [baseline, Candidate(13, 0.5, 0.852, 0.95, quality=6)]
    assert choose(tried, baseline, prefer_faster=True) == baseline


def test_an_advanced_setting_that_clearly_sounds_more_alike_is_chosen(tmp_path):
    tuner, engine = make(tmp_path)

    class LikesProtection(FakeSpeaker):
        def embedding(self, audio, sample_rate):
            emb = super().embedding(audio, sample_rate)
            if self.engine.last is not None and self.engine.params.protect != 0.5:
                emb = emb + np.array([0.0, 0.3])
            return emb

    tuner._speaker_factory = lambda: LikesProtection(engine)
    tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3, advanced={"protect": 0.33})
    tuner.wait(60)
    result = tuner.status()["result"]
    assert result["advanced"]["protect"] == 0.5


def test_the_tuner_checks_the_advanced_settings_it_starts_from(tmp_path):
    tuner, _ = make(tmp_path)
    with pytest.raises(ValueError, match="Unknown settings"):
        tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3, advanced={"nonsense": 1})
    with pytest.raises(ValueError, match="protect"):
        tuner.start("woman", quality=1, pitch_shift=7, likeness=0.3, advanced={"protect": 2.0})


def test_a_preview_turns_the_calibration_reading_out_in_the_voice(tmp_path):
    tuner, engine = make(tmp_path)
    audio, sr = tuner.preview("woman", quality=2, pitch_shift=9, likeness=0.5, advanced={"creak_repair": False})
    assert sr == SR and len(audio) == len(FakeCalibration().sample())
    assert engine.last[:2] == (9, 0.5) and engine.params.creak_repair is False and engine.params.chunk_ms == 500
    rms = float(np.sqrt(np.mean(audio ** 2)))
    assert 20 * np.log10(rms) == pytest.approx(-20.0, abs=1.0) and np.max(np.abs(audio)) <= 0.95  # comfortable to hear


def test_a_preview_needs_a_kept_reading(tmp_path):
    tuner, _ = make(tmp_path, sample=False)
    with pytest.raises(ValueError, match="Calibrate"):
        tuner.preview("woman", quality=1, pitch_shift=0, likeness=0.3)
