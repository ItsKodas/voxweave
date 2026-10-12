import threading
import time

import numpy as np
import pytest

from aivoice.audiofile import AudioFileError, write_wav
from aivoice.importer import AUDIO_EXTENSIONS, Importer
from aivoice.library import Library, LibraryError

SR = 48000
RNG = np.random.default_rng(3)


def noise(seconds, amp=0.3):
    return RNG.uniform(-amp, amp, int(seconds * SR)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


class FakeTranscriber:
    def __init__(self, fail=False):
        self.count = 0
        self.fail = fail

    def transcribe(self, audio16k):
        if self.fail:
            raise RuntimeError("model download failed")
        self.count += 1
        return f"line {self.count}"


def setup(tmp_path, transcriber=None):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")
    return lib, voice, Importer(lib, transcriber or FakeTranscriber())


def wav(tmp_path, name, audio):
    path = tmp_path / name
    write_wav(path, audio, SR)
    return str(path)


def run(importer, voice, paths):
    importer.start(voice.id, paths)
    importer.wait(10)
    return importer.status()


def test_imports_each_utterance_as_a_transcribed_clip(tmp_path):
    lib, voice, importer = setup(tmp_path)
    path = wav(tmp_path, "talk.wav", np.concatenate([noise(2.0), silence(0.6), noise(3.0)]))
    status = run(importer, voice, [path])
    assert status == {"active": False, "finished": True, "total_files": 1, "done_files": 1, "current": None,
                      "clips_added": 2, "skipped": 0, "errors": []}
    clips = lib.get(voice.id).clips
    assert [(c.text, c.source, c.origin) for c in clips] == [("line 1", "imported", "talk.wav"), ("line 2", "imported", "talk.wav")]


def test_bad_files_are_reported_by_name_and_the_rest_still_import(tmp_path):
    lib, voice, importer = setup(tmp_path)
    fake = tmp_path / "notes.mp3"
    fake.write_text("not audio")
    folder = tmp_path / "folder.wav"
    folder.mkdir()
    doc = tmp_path / "essay.docx"
    doc.write_text("words")
    good = wav(tmp_path, "good.wav", noise(2.0))
    status = run(importer, voice, [str(fake), str(folder), str(doc), str(tmp_path / "gone.wav"), good])
    assert status["clips_added"] == 1 and status["done_files"] == 5 and status["finished"]
    joined = " | ".join(status["errors"])
    assert len(status["errors"]) == 4
    for name in ("notes.mp3", "folder.wav", "essay.docx", "gone.wav"):
        assert name in joined
    assert len(lib.get(voice.id).clips) == 1


def test_a_file_with_no_speech_is_reported(tmp_path):
    _, voice, importer = setup(tmp_path)
    status = run(importer, voice, [wav(tmp_path, "quiet.wav", silence(3.0))])
    assert status["clips_added"] == 0
    assert status["errors"] == ["quiet.wav: no speech was found"]


def test_too_quiet_pieces_are_skipped_and_counted(tmp_path):
    _, voice, importer = setup(tmp_path)
    audio = np.concatenate([noise(2.0, amp=0.02), silence(0.6), noise(2.0)])
    status = run(importer, voice, [wav(tmp_path, "mixed.wav", audio)])
    assert (status["clips_added"], status["skipped"]) == (1, 1)


def test_transcription_failure_stops_the_job_with_one_clear_error(tmp_path):
    lib, voice, importer = setup(tmp_path, FakeTranscriber(fail=True))
    paths = [wav(tmp_path, "a.wav", noise(2.0)), wav(tmp_path, "b.wav", noise(2.0))]
    status = run(importer, voice, paths)
    assert status["finished"] and status["clips_added"] == 0
    assert status["errors"] == ["Transcription failed: model download failed"]
    assert lib.get(voice.id).clips == []


def test_unknown_voice_and_empty_or_oversized_lists_are_refused(tmp_path):
    _, voice, importer = setup(tmp_path)
    with pytest.raises(LibraryError):
        importer.start("nope-000000", ["a.wav"])
    with pytest.raises(ValueError, match="No files"):
        importer.start(voice.id, [])
    with pytest.raises(ValueError, match="at most"):
        importer.start(voice.id, ["a.wav"] * 201)


def test_a_second_import_is_refused_while_one_runs(tmp_path):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")

    gate = threading.Event()

    def slow_read(path, sample_rate=SR):
        gate.wait(5)
        return noise(2.0)

    importer = Importer(lib, FakeTranscriber(), read_audio=slow_read)
    real = wav(tmp_path, "a.wav", noise(2.0))
    importer.start(voice.id, [real])
    assert importer.status()["active"] is True
    with pytest.raises(ValueError, match="already running"):
        importer.start(voice.id, [real])
    gate.set()
    importer.wait(10)
    assert importer.status()["clips_added"] == 1


def test_cancel_stops_before_the_next_file(tmp_path):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")
    importer_box = {}

    def read_then_cancel(path, sample_rate=SR):
        importer_box["importer"].cancel()
        return noise(2.0)

    importer = Importer(lib, FakeTranscriber(), read_audio=read_then_cancel)
    importer_box["importer"] = importer
    paths = [wav(tmp_path, f"{n}.wav", noise(2.0)) for n in "abc"]
    status = run(importer, voice, paths)
    assert status["finished"] and status["done_files"] < 3
    assert status["clips_added"] == 0


def test_status_before_any_import():
    lib = Library.__new__(Library)
    assert Importer(lib, FakeTranscriber()).status() == {
        "active": False, "finished": False, "total_files": 0, "done_files": 0, "current": None,
        "clips_added": 0, "skipped": 0, "errors": []}


def test_common_formats_are_accepted():
    assert {".wav", ".mp3", ".flac", ".m4a", ".ogg"} <= AUDIO_EXTENSIONS
    assert AudioFileError


def test_a_voice_deleted_mid_import_ends_the_job_with_one_error(tmp_path):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")
    reads = []

    def read_then_delete(path, sample_rate=SR):
        reads.append(path)
        if len(reads) == 1:
            lib.delete(voice.id)
        return noise(2.0)

    importer = Importer(lib, FakeTranscriber(), read_audio=read_then_delete)
    paths = [wav(tmp_path, f"{n}.wav", noise(2.0)) for n in "abcd"]
    status = run(importer, voice, paths)
    assert status["finished"] and not status["active"]
    assert status["errors"] == ["The voice was deleted during the import"]
    assert status["clips_added"] == 0
    assert len(reads) == 1  # the remaining files were never decoded


def test_a_voice_deleted_before_the_job_reads_anything_is_one_error(tmp_path):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")
    gate = threading.Event()
    reads = []

    def slow_read(path, sample_rate=SR):
        reads.append(path)
        return noise(2.0)

    class GatedLibrary:
        """Lets the test delete the voice after start() has validated it but before the job runs."""

        def __getattr__(self, name):
            return getattr(lib, name)

        def get(self, voice_id):
            result = lib.get(voice_id)
            if not gate.is_set():
                gate.set()
                lib.delete(voice_id)
            return result

    importer = Importer(GatedLibrary(), FakeTranscriber(), read_audio=slow_read)
    status = run(importer, voice, [wav(tmp_path, "a.wav", noise(2.0)), wav(tmp_path, "b.wav", noise(2.0))])
    assert status["errors"] == ["The voice was deleted during the import"]
    assert reads == []


def test_a_new_import_can_start_as_soon_as_the_last_one_reports_finished(tmp_path):
    lib, voice, importer = setup(tmp_path)
    path = wav(tmp_path, "a.wav", noise(2.0))
    importer.start(voice.id, [path])
    deadline = time.monotonic() + 10
    while not importer.status()["finished"]:
        assert time.monotonic() < deadline
    importer.start(voice.id, [path])  # must not be refused although the old thread may still be exiting
    importer.wait(10)
    status = importer.status()
    assert status["finished"] and status["clips_added"] == 1  # the new job's own numbers, not the old job's


def test_cancel_with_nothing_running_does_not_affect_the_next_import(tmp_path):
    lib, voice, importer = setup(tmp_path)
    importer.cancel()
    status = run(importer, voice, [wav(tmp_path, "a.wav", noise(2.0))])
    assert status["clips_added"] == 1


def test_a_cancelled_job_cannot_cancel_or_alter_the_next_one(tmp_path):
    lib, voice, importer = setup(tmp_path)
    path = wav(tmp_path, "a.wav", noise(2.0))
    importer.start(voice.id, [path])
    importer.cancel()
    importer.wait(10)
    importer.start(voice.id, [path])
    importer.wait(10)
    assert importer.status()["clips_added"] == 1


def test_a_path_that_is_not_text_is_refused_up_front(tmp_path):
    _, voice, importer = setup(tmp_path)
    with pytest.raises(ValueError, match="path"):
        importer.start(voice.id, [None])


def test_an_unexpected_failure_while_reading_is_reported_not_raised(tmp_path):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")

    def broken(path, sample_rate=SR):
        raise MemoryError("out of memory")

    status = run(Importer(lib, FakeTranscriber(), read_audio=broken), voice, [wav(tmp_path, "a.wav", noise(2.0))])
    assert status["finished"] and status["errors"] == ["a.wav: out of memory"]
