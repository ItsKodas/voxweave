import json

import numpy as np
import pytest

from aivoice.calibration import (ABANDONED_AFTER, QUIET_SECONDS, Calibration, CalibrationError, GatedEngine,
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


def take(hz: float = 120.0, amp: float = 0.2, room: float = 0.001, speech_seconds: float = 8.0) -> np.ndarray:
    speech = voice_like(speech_seconds, hz, amp) + noise(speech_seconds, room, 1)
    return np.concatenate([noise(QUIET_SECONDS, room), speech])


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
    assert 1.5 < quiet.input_gain <= 4.0
    assert 0.5 <= loud.input_gain <= 0.98 / 0.9 + 1e-3


def test_too_short_too_little_speech_and_a_noisy_microphone_are_refused_with_reasons():
    with pytest.raises(CalibrationError, match="too short"):
        analyse(take(speech_seconds=1.0), SR, "m", 0.0)
    silent = np.concatenate([noise(QUIET_SECONDS, 0.001), noise(8.0, 0.001, 2)])
    with pytest.raises(CalibrationError, match="Not enough speech"):
        analyse(silent, SR, "m", 0.0)
    with pytest.raises(CalibrationError, match="too noisy"):
        analyse(take(amp=0.05, room=0.03), SR, "m", 0.0)


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
    assert status["recording"] and status["phase"] == "reading" and status["seconds"] == 4.0
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
    with pytest.raises(CalibrationError, match="Not enough speech"):
        bad.finish()
    assert bad.profile() == good and "Not enough speech" in bad.status()["error"] and not bad.active


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
