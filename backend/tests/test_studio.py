import numpy as np
import pytest

from aivoice.devices import Device
from aivoice.importer import Importer
from aivoice.library import Library, LibraryError
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from tests.test_recorder import CapturingStreams

SR = 48000
PROMPTS = ("First sentence here.", "Second sentence here.", "Third sentence here.")
DEVICES = [Device(1, "Headset Mic", 1, 0), Device(2, "Headset", 0, 2)]


class NoTranscriber:
    def transcribe(self, audio16k):
        return ""


def make(tmp_path):
    lib = Library(tmp_path)
    streams = CapturingStreams()
    studio = Studio(lib, Recorder(streams), Importer(lib, NoTranscriber()), lambda: DEVICES, prompts=PROMPTS)
    return studio, lib, streams, lib.create("V")


def speak(streams, seconds=2.0, amp=0.3, lead=0.5):
    rng = np.random.default_rng(5)
    streams.on_audio(np.zeros((int(lead * SR), 1), dtype=np.float32))
    streams.on_audio(rng.uniform(-amp, amp, (int(seconds * SR), 1)).astype(np.float32))
    streams.on_audio(np.zeros((int(lead * SR), 1), dtype=np.float32))


def test_prompt_for_a_new_voice(tmp_path):
    studio, _, _, voice = make(tmp_path)
    assert studio.prompt(voice.id) == {"index": 0, "text": "First sentence here.", "total": 3, "recorded": 0}


def test_accepted_take_is_trimmed_saved_and_advances_the_prompt(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    assert studio.recording and studio.status()["recording"] is True
    speak(streams)
    result = studio.stop_recording()
    assert result["accepted"] is True and result["reason"] is None and result["message"] is None
    clip = result["clip"]
    assert (clip["index"], clip["text"], clip["source"], clip["prompt"]) == (1, "First sentence here.", "recorded", 0)
    assert clip["seconds"] == pytest.approx(2.2, abs=0.05)
    assert not studio.recording
    assert studio.prompt(voice.id) == {"index": 1, "text": "Second sentence here.", "total": 3, "recorded": 1}
    assert len(lib.get(voice.id).clips) == 1


@pytest.mark.parametrize("seconds, amp, reason", [(0.0, 0.3, "too_quiet"), (0.3, 0.3, "too_short"), (2.0, 0.01, "too_quiet")])
def test_rejected_takes_write_nothing(tmp_path, seconds, amp, reason):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    if seconds:
        speak(streams, seconds, amp)
    result = studio.stop_recording()
    assert result["accepted"] is False and result["reason"] == reason and result["clip"] is None
    assert len(result["message"]) > 20
    assert lib.get(voice.id).clips == []
    assert list((tmp_path / "voices" / voice.id / "clips").iterdir()) == []


def test_stop_pressed_at_once_is_rejected_cleanly(tmp_path):
    studio, lib, _, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    assert studio.stop_recording()["reason"] == "too_quiet"
    assert lib.get(voice.id).clips == []


def test_redoing_a_sentence_replaces_the_earlier_take(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    for _ in range(2):
        studio.start_recording(voice.id, 1, 0)
        speak(streams)
        studio.stop_recording()
    clips = lib.get(voice.id).clips
    assert [(c.index, c.prompt) for c in clips] == [(2, 0)]


def test_a_rejected_redo_keeps_the_earlier_take(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    studio.stop_recording()
    studio.start_recording(voice.id, 1, 0)
    assert studio.stop_recording()["accepted"] is False
    assert [c.index for c in lib.get(voice.id).clips] == [1]


def test_skip_moves_to_the_next_unrecorded_sentence(tmp_path):
    studio, _, _, voice = make(tmp_path)
    assert studio.prompt(voice.id, after=0)["index"] == 1
    assert studio.prompt(voice.id, after=2)["index"] == 0


def test_all_sentences_recorded(tmp_path):
    studio, _, streams, voice = make(tmp_path)
    for i in range(3):
        studio.start_recording(voice.id, 1, i)
        speak(streams)
        studio.stop_recording()
    assert studio.prompt(voice.id) == {"index": None, "text": None, "total": 3, "recorded": 3}


def test_start_validation(tmp_path):
    studio, _, _, voice = make(tmp_path)
    with pytest.raises(LibraryError):
        studio.start_recording("nope-000000", 1, 0)
    with pytest.raises(ValueError, match="Unknown sentence"):
        studio.start_recording(voice.id, 1, 3)
    with pytest.raises(ValueError, match="not an input"):
        studio.start_recording(voice.id, 2, 0)
    with pytest.raises(ValueError, match="no longer available"):
        studio.start_recording(voice.id, 9, 0)
    assert not studio.recording


def test_stop_when_not_recording_is_a_clear_error(tmp_path):
    studio, _, _, _ = make(tmp_path)
    with pytest.raises(ValueError, match="Not recording"):
        studio.stop_recording()


def test_cancel_discards_the_take(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    studio.cancel_recording()
    assert not studio.recording and lib.get(voice.id).clips == []


def test_voice_deleted_mid_take_is_a_clear_error_and_leaves_the_recorder_idle(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    lib.delete(voice.id)
    with pytest.raises(LibraryError):
        studio.stop_recording()
    assert not studio.recording
    # and the next take starts cleanly: no stale take is left behind
    other = lib.create("Other")
    studio.start_recording(other.id, 1, 0)
    speak(streams)
    assert studio.stop_recording()["accepted"] is True


def test_a_save_failure_leaves_the_recorder_idle(tmp_path, monkeypatch):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(lib, "add_clip", boom)
    with pytest.raises(OSError):
        studio.stop_recording()
    assert not studio.recording
    with pytest.raises(ValueError, match="Not recording"):
        studio.stop_recording()


def test_starting_twice_keeps_the_first_take(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    with pytest.raises(ValueError, match="Already recording"):
        studio.start_recording(voice.id, 1, 1)
    speak(streams)
    assert studio.stop_recording()["clip"]["prompt"] == 0


def test_status_includes_import_progress(tmp_path):
    studio, _, _, _ = make(tmp_path)
    status = studio.status()
    assert status["recording"] is False and status["import"]["active"] is False
