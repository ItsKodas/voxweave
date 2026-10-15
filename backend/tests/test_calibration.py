import json

import numpy as np
import pytest

from aivoice.calibration import (ABANDONED_AFTER, QUIET_SECONDS, TEST_SECONDS, Calibration, CalibrationError, GatedEngine,
                                 analyse, suggest_shift)
from aivoice.devices import Device
from aivoice.library import Library
from aivoice.presets import PRESETS

SR = 48000
DEVICES = [Device(1, "Headset Mic", 1, 0), Device(2, "Speakers", 0, 2)]


def voice_like(seconds: float, hz: float, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    """A buzzy tone with harmonics, which the pitch tracker hears as a voice at hz."""
    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 9))
    return (amp * wave / np.max(np.abs(wave))).astype(np.float32)


def noise(seconds: float, amp: float, seed: int = 0, sr: int = SR) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(seconds * sr))).astype(np.float32)


def take(hz: float = 120.0, amp: float = 0.2, room: float = 0.001, speech_seconds: float = 8.0,
         test_amp: float | None = None) -> np.ndarray:
    """A calibration recording: quiet, then "testing, one, two, three" (at test_amp, else amp), then the reading."""
    test = voice_like(TEST_SECONDS, hz, amp if test_amp is None else test_amp) + noise(TEST_SECONDS, room, 3)
    speech = voice_like(speech_seconds, hz, amp) + noise(speech_seconds, room, 1)
    return np.concatenate([noise(QUIET_SECONDS, room), test, speech])


def test_a_recording_gives_the_speakers_pitch_level_noise_gate_and_gain():
    profile = analyse(take(hz=120.0, amp=0.2, room=0.001), SR, "Headset Mic", 1000.0)
    assert profile.pitch_hz == pytest.approx(120.0, rel=0.03)
    assert profile.pitch_low_hz <= profile.pitch_hz <= profile.pitch_high_hz
    assert profile.noise_db == pytest.approx(-60.0, abs=3.0)
    # The gate sits above the room noise and well below the speech.
    assert profile.noise_db < profile.gate_db < profile.level_db - 6
    assert profile.device == "Headset Mic" and profile.created == 1000.0


def test_a_quiet_speaker_is_lifted_and_a_loud_one_is_never_pushed_into_clipping():
    quiet = analyse(take(amp=0.02, room=0.0002), SR, "m", 0.0)
    loud = analyse(take(amp=0.9, room=0.001), SR, "m", 0.0)
    assert 4.0 < quiet.input_gain <= 16.0  # a quiet microphone is brought up past the old 4x limit
    assert 0.5 <= loud.input_gain <= 0.98 / 0.9 + 1e-3


def test_too_short_too_little_speech_and_a_noisy_microphone_are_refused_with_reasons():
    with pytest.raises(CalibrationError, match="too short"):
        analyse(take(speech_seconds=1.0), SR, "m", 0.0)
    silent = np.concatenate([noise(QUIET_SECONDS + TEST_SECONDS, 0.001), noise(8.0, 0.001, 2)])
    with pytest.raises(CalibrationError, match="test phrase was not heard"):
        analyse(silent, SR, "m", 0.0)
    check_only = take()
    check_only[int((QUIET_SECONDS + TEST_SECONDS) * SR):] = noise(8.0, 0.001, 2)
    with pytest.raises(CalibrationError, match="Not enough speech"):
        analyse(check_only, SR, "m", 0.0)
    with pytest.raises(CalibrationError, match="too loud"):
        analyse(take(amp=0.05, room=0.03), SR, "m", 0.0)


def test_a_very_quiet_microphone_calibrates():
    profile = analyse(take(amp=0.004, room=0.00005), SR, "m", 0.0)  # speech near -55 dB
    assert profile.input_gain == 16.0 and profile.pitch_hz == pytest.approx(120.0, rel=0.03)


def test_nan_in_the_recording_does_not_break_the_measurement():
    audio = take()
    audio[1000:1100] = np.nan
    assert analyse(audio, SR, "m", 0.0).pitch_hz == pytest.approx(120.0, rel=0.03)


def test_the_pitch_shift_moves_the_speaker_onto_the_target_in_whole_semitones():
    assert suggest_shift(110.0, 220.0) == 12
    assert suggest_shift(220.0, 110.0) == -12
    assert suggest_shift(120.0, 120.0) == 0
    assert suggest_shift(100.0, 133.5) == 5
    assert suggest_shift(50.0, 1000.0) == 24  # limited to what live mode accepts
    with pytest.raises(CalibrationError):
        suggest_shift(0.0, 100.0)


class CountingEngine:
    def __init__(self):
        self.seen = []

    def convert(self, audio, sample_rate, pitch_shift, params):
        self.seen.append(np.array(audio))
        return np.full_like(audio, 0.5)


def window(params, tail: np.ndarray) -> np.ndarray:
    size = SR * params.context_ms // 1000 + SR * params.chunk_ms // 1000 + SR * params.crossfade_ms // 1000
    out = np.zeros(size, dtype=np.float32)
    out[-len(tail):] = tail[-size:]
    return out


def test_the_gate_outputs_silence_for_room_noise_without_running_the_model():
    params = PRESETS["balanced"]
    inner = CountingEngine()
    gated = GatedEngine(inner, gate_db=-50.0)
    out = gated.convert(window(params, noise(2.0, 0.0005)), SR, 0.0, params)
    assert inner.seen == [] and not out.any() and out.shape == window(params, noise(2.0, 0.0005)).shape


def test_the_gate_opens_for_speech_and_stays_open_just_after_it_ends():
    params = PRESETS["balanced"]
    inner = CountingEngine()
    gated = GatedEngine(inner, gate_db=-50.0)
    assert gated.convert(window(params, voice_like(1.0, 120.0)), SR, 0.0, params).max() == 0.5
    # Speech ended 0.4 s ago: inside the newest chunk plus the hangover, so the tail of the word is still converted.
    ended = np.concatenate([voice_like(1.0, 120.0), np.zeros(int(0.4 * SR), dtype=np.float32)])
    assert gated.convert(window(params, ended), SR, 0.0, params).max() == 0.5
    # A second later the gate has closed again.
    long_ago = np.concatenate([voice_like(1.0, 120.0), np.zeros(int(1.5 * SR), dtype=np.float32)])
    assert not gated.convert(window(params, long_ago), SR, 0.0, params).any()
    assert len(inner.seen) == 2


def test_the_gain_lifts_the_input_and_never_clips_it():
    params = PRESETS["balanced"]
    inner = CountingEngine()
    GatedEngine(inner, gate_db=None, gain=4.0).convert(window(params, voice_like(1.0, 120.0, amp=0.5)), SR, 0.0, params)
    assert inner.seen[0].max() == pytest.approx(1.0) and np.abs(inner.seen[0]).max() <= 1.0
    inner = CountingEngine()
    GatedEngine(inner, gate_db=None, gain=2.0).convert(window(params, voice_like(1.0, 120.0, amp=0.1)), SR, 0.0, params)
    assert inner.seen[0].max() == pytest.approx(0.2, rel=0.02)


class FakeRecorder:
    def __init__(self, audio=None):
        self.audio = audio if audio is not None else take()
        self.running = False
        self.cancelled = 0

    def start(self, device):
        self.running = True

    def stop(self):
        self.running = False
        return self.audio

    def cancel(self):
        self.running = False
        self.cancelled += 1

    def status(self):
        return {"recording": self.running, "level": 0.3, "seconds": 4.0, "full": False}


def make(tmp_path, recorder=None, library=None, now=None):
    clock = now if now is not None else [1000.0]
    return Calibration(tmp_path, recorder or FakeRecorder(), lambda: DEVICES, library, clock=lambda: clock[0]), clock


def test_a_finished_recording_is_saved_and_read_back(tmp_path):
    cal, _ = make(tmp_path)
    assert cal.profile() is None and cal.status()["profile"] is None
    cal.start(1)
    status = cal.status()
    assert status["recording"] and status["phase"] == "testing" and status["seconds"] == 4.0
    profile = cal.finish()
    assert not cal.active
    assert cal.profile() == profile and profile.device == "Headset Mic"
    assert make(tmp_path)[0].profile() == profile  # a new start of the app finds it
    assert cal.status()["profile"]["pitch_hz"] == profile.pitch_hz
    cal.forget()
    assert cal.profile() is None
    cal.forget()  # twice is fine


def test_start_refuses_a_missing_or_output_device_and_a_second_recording(tmp_path):
    cal, _ = make(tmp_path)
    with pytest.raises(CalibrationError, match="no longer available"):
        cal.start(9)
    with pytest.raises(CalibrationError, match="not an input device"):
        cal.start(2)
    cal.start(1)
    with pytest.raises(CalibrationError, match="already running"):
        cal.start(1)
    cal.cancel()
    assert not cal.active
    with pytest.raises(CalibrationError, match="No calibration recording"):
        cal.finish()


def test_a_failed_measurement_keeps_the_earlier_profile_and_says_why(tmp_path):
    cal, _ = make(tmp_path)
    cal.start(1)
    good = cal.finish()
    bad, _ = make(tmp_path, FakeRecorder(np.zeros(SR * 10, dtype=np.float32)))
    bad.start(1)
    with pytest.raises(CalibrationError, match="test phrase was not heard"):
        bad.finish()
    assert bad.profile() == good and "test phrase was not heard" in bad.status()["error"] and not bad.active


def test_a_recording_nobody_finished_is_dropped_so_it_cannot_block_live_mode_for_ever(tmp_path):
    recorder = FakeRecorder()
    cal, clock = make(tmp_path, recorder)
    cal.start(1)
    clock[0] += ABANDONED_AFTER + 1
    assert not cal.active and recorder.cancelled == 1
    assert "not finished" in cal.status()["error"]


def test_a_damaged_profile_file_counts_as_no_profile(tmp_path):
    cal, _ = make(tmp_path)
    folder = tmp_path / "calibration"
    folder.mkdir()
    for text in ("not json", "{}", json.dumps({"pitch_hz": "x"}),
                 json.dumps(dict(pitch_hz=float("nan"), pitch_low_hz=1, pitch_high_hz=2, level_db=-20, noise_db=-60,
                                 gate_db=-50, input_gain=1.0, device="m", created=0)),
                 json.dumps(dict(pitch_hz=120, pitch_low_hz=1, pitch_high_hz=2, level_db=-20, noise_db=-60,
                                 gate_db=-50, input_gain=50.0, device="m", created=0))):
        (folder / "profile.json").write_text(text)
        assert cal.profile() is None

    class Engine:
        pass

    engine = Engine()
    assert cal.wrap(engine) is engine  # nothing to apply


def test_wrap_applies_the_saved_gate_and_gain(tmp_path):
    cal, _ = make(tmp_path)
    cal.start(1)
    profile = cal.finish()
    inner = CountingEngine()
    wrapped = cal.wrap(inner)
    params = PRESETS["balanced"]
    assert not wrapped.convert(window(params, noise(2.0, 0.0003)), SR, 0.0, params).any()
    wrapped.convert(window(params, voice_like(1.0, 120.0, amp=0.05)), SR, 0.0, params)
    assert inner.seen[0].max() == pytest.approx(0.05 * profile.input_gain, rel=0.05)


def test_a_library_voices_usual_pitch_gives_the_suggested_shift(tmp_path):
    library = Library(tmp_path)
    voice = library.create("High")
    for _ in range(2):
        library.add_clip(voice.id, voice_like(3.0, 240.0), "hello", "recorded")
    cal, _ = make(tmp_path, library=library)
    assert cal.suggestion(voice.id) == {"pitch_shift": None, "reason": "Calibrate your voice first."}
    cal.start(1)
    cal.finish()  # a speaker at 120 Hz
    assert cal.voice_pitch(voice.id) == pytest.approx(240.0, rel=0.03)
    answer = cal.suggestion(voice.id)
    assert answer["pitch_shift"] == 12 and answer["reason"] is None
    assert answer["speaker_hz"] == pytest.approx(120.0, rel=0.03)
    # Measured once: the answer is remembered until the clips change.
    cached = json.loads((tmp_path / "calibration" / "voices" / f"{voice.id}.json").read_text())
    assert cached["pitch_hz"] == answer["target_hz"]
    library.add_clip(voice.id, voice_like(3.0, 240.0), "more", "recorded")
    assert cal.voice_pitch(voice.id) == pytest.approx(240.0, rel=0.03)
    assert json.loads((tmp_path / "calibration" / "voices" / f"{voice.id}.json").read_text())["key"] != cached["key"]


def test_a_voice_without_recordings_has_no_suggestion_and_says_so(tmp_path):
    library = Library(tmp_path)
    empty = library.create("Empty")
    cal, _ = make(tmp_path, library=library)
    cal.start(1)
    cal.finish()
    for voice_id in (empty.id, "bench", "../../etc"):
        answer = cal.suggestion(voice_id)
        assert answer["pitch_shift"] is None and "by hand" in answer["reason"]


def test_a_voices_pitch_range_is_measured_from_its_clips_and_remembered(tmp_path):
    library = Library(tmp_path)
    voice = library.create("Wide")
    for hz in (200.0, 250.0, 300.0):
        library.add_clip(voice.id, voice_like(2.0, hz), "words", "recorded")
    cal, _ = make(tmp_path, library=library)
    low, high = cal.voice_pitch_range(voice.id)
    assert low == pytest.approx(200, rel=0.05) and high == pytest.approx(300, rel=0.05)
    assert low <= cal.voice_pitch(voice.id) <= high
    assert cal.voice_pitch_range("nope") is None


def test_a_voices_range_measured_by_an_older_version_is_measured_again(tmp_path):
    library = Library(tmp_path)
    voice = library.create("Wide")
    for hz in (200.0, 250.0, 300.0):
        library.add_clip(voice.id, voice_like(2.0, hz), "words", "recorded")
    cache = tmp_path / "calibration" / "voices" / f"{voice.id}.json"
    cache.parent.mkdir(parents=True)
    # the key an older version wrote for these clips, with its narrower range
    last = max(c.index for c in library.get(voice.id).clips)
    cache.write_text(json.dumps({"key": f"3:{last}", "pitch_hz": 250.0, "low_hz": 240.0, "high_hz": 260.0}), encoding="utf-8")
    cal, _ = make(tmp_path, library=library)
    low, high = cal.voice_pitch_range(voice.id)
    assert low == pytest.approx(200, rel=0.05) and high == pytest.approx(300, rel=0.05)


def test_the_reading_is_kept_for_deep_calibration_and_forgotten_with_the_profile(tmp_path):
    from aivoice.calibration import QUIET_SECONDS

    cal, _ = make(tmp_path)
    assert cal.sample() is None and cal.status()["has_sample"] is False
    cal.start(1)
    cal.finish()
    sample = cal.sample()
    recorded = take()
    start = int((QUIET_SECONDS + TEST_SECONDS) * 48000)
    assert sample is not None and len(sample) == len(recorded) - start  # the reading only
    assert np.allclose(sample, recorded[start:], atol=1e-4)
    assert cal.status()["has_sample"] is True
    cal.forget()
    assert cal.sample() is None


def test_a_failed_calibration_keeps_no_reading(tmp_path):
    cal, _ = make(tmp_path, recorder=FakeRecorder(np.zeros(48000, np.float32)))
    cal.start(1)
    with pytest.raises(Exception):
        cal.finish()
    assert cal.sample() is None



def test_the_mic_check_phrase_sets_the_input_boost():
    quiet_check = analyse(take(amp=0.05, test_amp=0.02), SR, "m", 0.0)
    louder_check = analyse(take(amp=0.05, test_amp=0.04), SR, "m", 0.0)
    # the boost follows the level of "testing, one, two, three": a check twice as loud gets half the boost
    assert quiet_check.input_gain / louder_check.input_gain == pytest.approx(2.0, rel=0.1)
    assert quiet_check.test_level_db < louder_check.test_level_db


def test_the_boost_never_makes_the_loudest_moment_clip():
    profile = analyse(take(amp=0.6, test_amp=0.02), SR, "m", 0.0)  # a timid mic check, then loud reading
    assert profile.input_gain * 0.6 <= 0.99


def test_a_missing_mic_check_is_explained():
    audio = take()
    start, end = int(QUIET_SECONDS * SR), int((QUIET_SECONDS + TEST_SECONDS) * SR)
    audio[start:end] = noise(TEST_SECONDS, 0.001, 9)
    with pytest.raises(CalibrationError, match="testing, one, two, three"):
        analyse(audio, SR, "m", 0.0)


def test_the_status_walks_through_quiet_mic_check_and_reading(tmp_path):
    class At(FakeRecorder):
        def __init__(self, seconds):
            super().__init__()
            self.seconds = seconds

        def status(self):
            return {"recording": self.running, "level": 0.1, "seconds": self.seconds, "full": False}

    phases = []
    for seconds in (1.0, QUIET_SECONDS + 1.0, QUIET_SECONDS + TEST_SECONDS + 1.0):
        cal, _ = make(tmp_path, recorder=At(seconds))
        cal.start(1)
        phases.append(cal.status()["phase"])
        cal.cancel()
    assert phases == ["quiet", "testing", "reading"]
    status = cal.status()
    assert status["test_seconds"] == TEST_SECONDS and status["total_seconds"] == 75.0



class Echo:
    """A voice that comes out exactly as loud as what it is given."""
    def convert(self, audio, sample_rate, pitch_shift, params):
        return np.array(audio, dtype=np.float32)


def test_the_converted_voice_comes_out_as_loud_as_the_speaker_went_in():
    from dataclasses import replace

    params = PRESETS["balanced"]
    speech = window(params, voice_like(1.0, 120.0, amp=0.1))
    out = GatedEngine(Echo(), gate_db=None, gain=4.0).convert(speech, SR, 0.0, params)
    assert np.max(np.abs(out)) == pytest.approx(0.1, rel=0.01)  # the boost is for the model, not the listener
    louder = GatedEngine(Echo(), gate_db=None, gain=4.0).convert(speech, SR, 0.0, replace(params, match_input_level=False))
    assert np.max(np.abs(louder)) == pytest.approx(0.4, rel=0.01)


def test_each_calibration_reads_one_of_several_paragraphs_never_the_same_twice_running(tmp_path):
    from aivoice.calibration import READINGS

    assert len(READINGS) >= 5 and all(len(r.split()) >= 35 for r in READINGS)
    cal, _ = make(tmp_path)
    seen = []
    for _ in range(12):
        cal.start(1)
        seen.append(cal.status()["text"])
        cal.cancel()
    assert all(t in READINGS for t in seen)
    assert all(a != b for a, b in zip(seen, seen[1:]))
    assert len(set(seen)) >= 3


def test_the_reading_runs_until_done_with_a_minimum_and_a_safety_limit(tmp_path):
    from aivoice.calibration import MAX_SECONDS, MIN_READ_SECONDS

    cal, _ = make(tmp_path)
    cal.start(1)
    status = cal.status()
    assert status["min_read_seconds"] == MIN_READ_SECONDS
    assert status["total_seconds"] == MAX_SECONDS >= QUIET_SECONDS + TEST_SECONDS + 45
    cal.cancel()


def test_the_mic_check_is_kept_as_the_microphone_heard_it(tmp_path):
    cal, _ = make(tmp_path)
    assert cal.test_audio() is None
    cal.start(1)
    cal.finish()
    check = cal.test_audio()
    recorded = take()
    start, end = int(QUIET_SECONDS * SR), int((QUIET_SECONDS + TEST_SECONDS) * SR)
    assert check is not None and len(check) == end - start
    assert np.allclose(check, recorded[start:end], atol=1e-4)  # not boosted: what the microphone picked up
    cal.forget()
    assert cal.test_audio() is None


def test_a_failed_calibration_keeps_its_recording_to_look_into(tmp_path):
    from aivoice.audiofile import read_wav

    bad, _ = make(tmp_path, FakeRecorder(np.zeros(SR * 10, dtype=np.float32)))
    bad.start(1)
    with pytest.raises(CalibrationError):
        bad.finish()
    audio, sr = read_wav(tmp_path / "calibration" / "failed-attempt.wav")
    assert sr == SR and len(audio) == SR * 10
