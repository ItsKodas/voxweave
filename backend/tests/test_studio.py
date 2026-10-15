import errno

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


def test_a_take_over_20_seconds_is_rejected_as_too_long(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams, seconds=25.0)
    result = studio.stop_recording()
    assert result["accepted"] is False and result["reason"] == "too_long" and result["clip"] is None
    assert lib.get(voice.id).clips == []
    assert list((tmp_path / "voices" / voice.id / "clips").iterdir()) == []


def test_a_redo_stays_accepted_when_the_old_take_cannot_be_removed(tmp_path, monkeypatch, caplog):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    studio.stop_recording()

    def refuse(*a, **k):
        raise LibraryError("Unknown clip")

    monkeypatch.setattr(lib, "delete_clip", refuse)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    with caplog.at_level("WARNING"):
        result = studio.stop_recording()
    assert result["accepted"] is True and result["clip"]["index"] == 2
    assert not studio.recording
    assert any("earlier take" in r.message for r in caplog.records)


def test_redoing_a_sentence_moves_the_earlier_take_to_the_trash(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    studio.stop_recording()
    clips = tmp_path / "voices" / voice.id / "clips"
    first = {name: (clips / name).read_bytes() for name in ("0001.orig.wav", "0001.wav", "0001.txt")}
    studio.start_recording(voice.id, 1, 0)
    speak(streams, seconds=3.0)
    assert studio.stop_recording()["clip"]["index"] == 2
    trash = tmp_path / "trash" / voice.id / "clips"
    assert {p.name: p.read_bytes() for p in trash.iterdir()} == first
    assert sorted(p.name for p in clips.iterdir()) == ["0002.orig.wav", "0002.txt", "0002.wav"]


def test_a_take_that_cannot_be_saved_is_a_readable_error_and_keeps_the_earlier_take(tmp_path, monkeypatch):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    studio.stop_recording()

    def no_space(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("aivoice.library.write_wav", no_space)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    with pytest.raises(ValueError, match="Could not save the clip: No space left on device"):
        studio.stop_recording()
    monkeypatch.undo()
    assert not studio.recording
    assert [c.index for c in lib.get(voice.id).clips] == [1]  # the redo failed, so the earlier take stays
    assert not (tmp_path / "trash").exists()


def test_paragraphs_follow_the_sentences_and_are_asked_for_by_kind(tmp_path):
    lib = Library(tmp_path)
    studio = Studio(lib, Recorder(CapturingStreams()), Importer(lib, NoTranscriber()), lambda: DEVICES,
                    prompts=PROMPTS, paragraphs=("First paragraph. It is longer.", "Second paragraph. Also long."))
    voice = lib.create("Ada")
    assert studio.prompt(voice.id, kind="sentences") == {
        "index": 0, "text": "First sentence here.", "total": 3, "recorded": 0, "kind": "sentences", "number": 1}
    assert studio.prompt(voice.id, kind="paragraphs") == {
        "index": 3, "text": "First paragraph. It is longer.", "total": 2, "recorded": 0, "kind": "paragraphs",
        "number": 1}
    # Skip moves on within the kind and wraps round inside it.
    assert studio.prompt(voice.id, after=3, kind="paragraphs")["index"] == 4
    assert studio.prompt(voice.id, after=4, kind="paragraphs")["index"] == 3
    assert studio.prompt(voice.id, after=2, kind="sentences")["index"] == 0
    with pytest.raises(ValueError, match="sentences or paragraphs"):
        studio.prompt(voice.id, kind="songs")


def test_a_recorded_paragraph_counts_only_among_the_paragraphs(tmp_path):
    lib = Library(tmp_path)
    studio = Studio(lib, Recorder(CapturingStreams()), Importer(lib, NoTranscriber()), lambda: DEVICES,
                    prompts=PROMPTS, paragraphs=("First paragraph. It is longer.", "Second paragraph. Also long."))
    voice = lib.create("Ada")
    lib.add_clip(voice.id, np.zeros(48000, dtype=np.float32), "First paragraph. It is longer.", "recorded", prompt=3)
    assert studio.prompt(voice.id, kind="paragraphs")["index"] == 4
    assert studio.prompt(voice.id, kind="paragraphs")["recorded"] == 1
    assert studio.prompt(voice.id, kind="sentences")["recorded"] == 0
    lib.add_clip(voice.id, np.zeros(48000, dtype=np.float32), "Second paragraph. Also long.", "recorded", prompt=4)
    assert studio.prompt(voice.id, kind="paragraphs") == {
        "index": None, "text": None, "total": 2, "recorded": 2, "kind": "paragraphs", "number": None}


def test_the_real_studio_offers_the_shipped_paragraphs_after_the_200_sentences(tmp_path):
    lib = Library(tmp_path)
    studio = Studio(lib, Recorder(CapturingStreams()), Importer(lib, NoTranscriber()), lambda: DEVICES)
    voice = lib.create("Ada")
    first = studio.prompt(voice.id, kind="paragraphs")
    assert (first["index"], first["total"], first["number"]) == (200, 40, 1)
    assert studio.prompt(voice.id, kind="sentences")["total"] == 200


# ---- reading a full page

PAGES = (("A Page", "One two three. " * 40), ("Another Page", "Four five six. " * 40))


def make_pages(tmp_path):
    lib = Library(tmp_path)
    streams = CapturingStreams()
    importer = Importer(lib, NoTranscriber())
    studio = Studio(lib, Recorder(streams), importer, lambda: DEVICES, prompts=PROMPTS, pages=PAGES,
                    recordings=tmp_path / "page-recordings")
    return studio, lib, streams, importer, lib.create("V")


def talk_with_pauses(streams, sentences=12, seconds=2.5, amp=0.3):
    rng = np.random.default_rng(7)
    for _ in range(sentences):
        streams.on_audio(rng.uniform(-amp, amp, (int(seconds * SR), 1)).astype(np.float32))
        streams.on_audio(np.zeros((int(0.8 * SR), 1), dtype=np.float32))


def test_pages_are_listed_with_their_length(tmp_path):
    studio, *_ = make_pages(tmp_path)
    pages = studio.pages()
    assert [p["title"] for p in pages] == ["A Page", "Another Page"]
    assert pages[0]["index"] == 0 and pages[0]["words"] == 120 and pages[0]["minutes"] == pytest.approx(0.8, abs=0.1)
    assert pages[0]["text"].startswith("One two three.")


def test_a_read_page_is_kept_and_sliced_into_clips_by_the_importer(tmp_path):
    studio, lib, streams, importer, voice = make_pages(tmp_path)
    studio.start_page(voice.id, 1, 1)
    talk_with_pauses(streams)
    result = studio.stop_recording()
    assert result["accepted"] is True and result["imported"] is True and result["clip"] is None
    assert result["seconds"] > 35
    importer.wait(10)
    clips = lib.get(voice.id).clips
    assert len(clips) >= 5 and all(c.source == "imported" for c in clips)
    saved = list((tmp_path / "page-recordings").glob("*.wav"))
    assert len(saved) == 1 and clips[0].origin == saved[0].name


def test_a_page_take_can_run_much_longer_than_a_sentence(tmp_path):
    studio, _, streams, importer, voice = make_pages(tmp_path)
    studio.start_page(voice.id, 1, 0)
    talk_with_pauses(streams, sentences=12, seconds=2.5)  # about 40 s, past a sentence take's 30 s limit
    assert studio.status()["full"] is False
    studio.stop_recording()
    importer.wait(10)
    studio.start_recording(voice.id, 1, 0)  # and a sentence take afterwards has its usual limit again
    streams.on_audio(np.zeros((31 * SR, 1), dtype=np.float32))
    assert studio.status()["full"] is True
    studio.cancel_recording()


def test_a_page_read_too_short_is_rejected_and_nothing_is_kept(tmp_path):
    studio, lib, streams, _, voice = make_pages(tmp_path)
    studio.start_page(voice.id, 1, 0)
    talk_with_pauses(streams, sentences=2)
    result = studio.stop_recording()
    assert result["accepted"] is False and result["reason"] == "too_short" and "page" in result["message"]
    assert lib.get(voice.id).clips == [] and not list((tmp_path / "page-recordings").glob("*.wav"))


def test_page_start_validation(tmp_path):
    studio, _, _, importer, voice = make_pages(tmp_path)
    with pytest.raises(ValueError, match="Unknown page"):
        studio.start_page(voice.id, 1, 5)
    with pytest.raises(ValueError, match="not an input device"):
        studio.start_page(voice.id, 2, 0)
    assert not studio.recording


def test_a_page_cannot_start_while_an_import_runs(tmp_path, monkeypatch):
    studio, _, _, importer, voice = make_pages(tmp_path)
    monkeypatch.setattr(importer, "status", lambda: {"active": True})
    with pytest.raises(ValueError, match="import"):
        studio.start_page(voice.id, 1, 0)
    assert not studio.recording


def test_only_the_newest_page_recordings_are_kept(tmp_path):
    studio, _, streams, importer, voice = make_pages(tmp_path)
    folder = tmp_path / "page-recordings"
    folder.mkdir()
    for i in range(12):
        (folder / f"old-{i:02d}.wav").write_bytes(b"x")
    studio.start_page(voice.id, 1, 0)
    talk_with_pauses(streams)
    studio.stop_recording()
    importer.wait(10)
    assert len(list(folder.glob("*.wav"))) == 10


def test_the_real_studio_offers_the_shipped_pages(tmp_path):
    lib = Library(tmp_path)
    studio = Studio(lib, Recorder(CapturingStreams()), Importer(lib, NoTranscriber()), lambda: DEVICES)
    pages = studio.pages()
    assert len(pages) >= 5 and all(p["words"] >= 300 and p["title"] for p in pages)
