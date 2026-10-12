import json
import os
import re
import threading

import numpy as np
import pytest

from aivoice import library as library_module
from aivoice.audiofile import read_wav
from aivoice.library import Library, LibraryError

SR = 48000


def audio(seconds=2.0):
    return np.random.default_rng(1).uniform(-0.3, 0.3, int(seconds * SR)).astype(np.float32)


def test_empty_library(tmp_path):
    assert Library(tmp_path).list() == []


def test_create_and_list(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("My Voice")
    assert voice.name == "My Voice" and voice.clips == [] and voice.minutes == 0
    assert re.fullmatch(r"my-voice-[0-9a-f]{6}", voice.id)
    assert [v.id for v in lib.list()] == [voice.id]
    assert (tmp_path / "voices" / voice.id / "clips").is_dir()


@pytest.mark.parametrize("name", ["../../etc", "a/b\\c", "🎤 Mic", "...", "   spaced   ", "Ünïcödé", "CON"])
def test_awkward_names_get_safe_ids_and_keep_their_display_name(tmp_path, name):
    lib = Library(tmp_path)
    voice = lib.create(name)
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*-[0-9a-f]{6}", voice.id)
    assert voice.name == name.strip()
    assert (tmp_path / "voices" / voice.id).parent == tmp_path / "voices"
    assert lib.get(voice.id).name == name.strip()


@pytest.mark.parametrize("name", ["", "   ", "x" * 61, "line\nbreak"])
def test_invalid_names_are_rejected(tmp_path, name):
    with pytest.raises(LibraryError):
        Library(tmp_path).create(name)


def test_duplicate_names_get_distinct_ids(tmp_path):
    lib = Library(tmp_path)
    a, b = lib.create("Same"), lib.create("Same")
    assert a.id != b.id and len(lib.list()) == 2


@pytest.mark.parametrize("bad", ["", "..", "../x", "a/b", "A", "nope-000000", None])
def test_unknown_or_unsafe_ids_are_rejected(tmp_path, bad):
    lib = Library(tmp_path)
    lib.create("Real")
    with pytest.raises(LibraryError):
        lib.get(bad)


def test_add_clip_writes_three_files_and_records_it(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    clip = lib.add_clip(voice.id, audio(2.0), "Hello there.", "recorded", prompt=4)
    assert (clip.index, clip.text, clip.source, clip.prompt, clip.origin) == (1, "Hello there.", "recorded", 4, None)
    assert clip.seconds == pytest.approx(2.0, abs=0.01)
    folder = tmp_path / "voices" / voice.id / "clips"
    assert sorted(p.name for p in folder.iterdir()) == ["0001.orig.wav", "0001.txt", "0001.wav"]
    orig, sr_orig = read_wav(folder / "0001.orig.wav")
    small, sr_small = read_wav(folder / "0001.wav")
    assert sr_orig == 48000 and abs(len(orig) - 96000) <= 2
    assert sr_small == 16000 and abs(len(small) - 32000) <= 2
    assert (folder / "0001.txt").read_text(encoding="utf-8") == "Hello there."
    assert lib.get(voice.id).clips == [clip]
    assert lib.get(voice.id).minutes == pytest.approx(2.0 / 60, abs=0.01)


def test_indices_are_never_reused(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "one", "recorded")
    second = lib.add_clip(voice.id, audio(), "two", "recorded")
    lib.delete_clip(voice.id, second.index)
    third = lib.add_clip(voice.id, audio(), "three", "imported", origin="file.mp3")
    assert third.index == 3 and third.origin == "file.mp3"
    assert [c.index for c in lib.get(voice.id).clips] == [1, 3]


def test_update_text(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "wrong", "imported")
    clip = lib.update_text(voice.id, 1, "  right  ")
    assert clip.text == "right"
    assert (tmp_path / "voices" / voice.id / "clips" / "0001.txt").read_text(encoding="utf-8") == "right"
    assert lib.get(voice.id).clips[0].text == "right"


def test_update_text_rejects_overlong_text_and_unknown_clip(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "imported")
    with pytest.raises(LibraryError):
        lib.update_text(voice.id, 1, "y" * 501)
    with pytest.raises(LibraryError):
        lib.update_text(voice.id, 9, "y")


def test_delete_clip_removes_files(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    lib.delete_clip(voice.id, 1)
    assert list((tmp_path / "voices" / voice.id / "clips").iterdir()) == []
    assert lib.get(voice.id).clips == []
    with pytest.raises(LibraryError):
        lib.delete_clip(voice.id, 1)


def test_clip_audio_path(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    assert lib.clip_audio_path(voice.id, 1).name == "0001.orig.wav"
    with pytest.raises(LibraryError):
        lib.clip_audio_path(voice.id, 2)


def test_delete_voice_moves_it_to_trash(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    lib.delete(voice.id)
    assert lib.list() == []
    trashed = list((tmp_path / "trash").iterdir())
    assert len(trashed) == 1 and trashed[0].name.startswith(voice.id)
    assert (trashed[0] / "clips" / "0001.orig.wav").is_file()


def test_summary_manifest_is_written(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(60.0), "x", "recorded")
    manifest = json.loads((tmp_path / "voices.json").read_text(encoding="utf-8"))
    assert manifest == [{"id": voice.id, "name": "V", "created": voice.created, "minutes": 1.0, "clips": 1, "models": []}]


def test_a_corrupt_voice_file_hides_only_that_voice_and_deletes_nothing(tmp_path):
    lib = Library(tmp_path)
    good, bad = lib.create("Good"), lib.create("Bad")
    lib.add_clip(bad.id, audio(), "x", "recorded")
    (tmp_path / "voices" / bad.id / "voice.json").write_text("{half written")
    assert [v.id for v in Library(tmp_path).list()] == [good.id]
    assert (tmp_path / "voices" / bad.id / "clips" / "0001.orig.wav").is_file()


def test_leftover_files_from_an_interrupted_save_do_not_block_the_next_save(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    folder = tmp_path / "voices" / voice.id / "clips"
    (folder / "0001.orig.wav").write_bytes(b"junk")
    (folder / "0001.wav.tmp").write_bytes(b"junk")
    clip = lib.add_clip(voice.id, audio(), "x", "recorded")
    assert clip.index == 1
    _, sr = read_wav(folder / "0001.orig.wav")
    assert sr == 48000


def test_a_stray_folder_without_voice_json_is_ignored(tmp_path):
    (tmp_path / "voices" / "stray-folder").mkdir(parents=True)
    assert Library(tmp_path).list() == []


def test_voices_are_listed_oldest_first(tmp_path):
    lib = Library(tmp_path)
    ids = [lib.create(n).id for n in ("b", "a", "c")]
    assert [v.id for v in lib.list()] == ids


def test_indices_are_not_reused_even_after_deleting_the_newest_clip(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "one", "recorded")
    lib.delete_clip(voice.id, 1)
    assert lib.add_clip(voice.id, audio(), "two", "recorded").index == 2
    assert Library(tmp_path).add_clip(voice.id, audio(), "three", "recorded").index == 3


def test_two_threads_adding_clips_never_collide(tmp_path):
    voice = Library(tmp_path).create("V")
    errors = []

    def work():
        try:
            lib = Library(tmp_path)  # separate instances, as separate requests might use
            for _ in range(5):
                lib.add_clip(voice.id, audio(1.0), "t", "recorded")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    clips = Library(tmp_path).get(voice.id).clips
    assert sorted(c.index for c in clips) == list(range(1, 21))


def test_delete_voice_twice_in_one_second_keeps_both_in_trash(tmp_path, monkeypatch):
    monkeypatch.setattr("aivoice.library._now", lambda: 1000.0)
    lib = Library(tmp_path)
    for _ in range(2):
        voice = lib.create("V")
        lib.add_clip(voice.id, audio(), "x", "recorded")
        lib.delete(voice.id)
    assert len(list((tmp_path / "trash").iterdir())) == 2


def test_failed_trash_move_keeps_the_voice_intact(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")

    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr("aivoice.library.os.rename", locked)
    with pytest.raises(LibraryError):
        lib.delete(voice.id)
    monkeypatch.undo()
    assert [v.id for v in lib.list()] == [voice.id]
    assert (tmp_path / "voices" / voice.id / "clips" / "0001.orig.wav").is_file()


def test_replace_is_retried_when_windows_has_the_file_open(tmp_path, monkeypatch):
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("in use")
        return real(src, dst)

    lib = Library(tmp_path)
    monkeypatch.setattr("aivoice.library.os.replace", flaky)
    voice = lib.create("V")
    assert lib.get(voice.id).name == "V"


def test_delete_clip_survives_a_locked_audio_file(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    real = type(tmp_path).unlink

    def locked(self, missing_ok=False):
        if self.name.endswith(".orig.wav"):
            raise PermissionError("in use")
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(type(tmp_path), "unlink", locked)
    lib.delete_clip(voice.id, 1)  # the record is gone; a stray file is harmless
    monkeypatch.undo()
    assert lib.get(voice.id).clips == []


def test_add_clip_rejects_empty_audio_and_bad_source_without_writing(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    with pytest.raises(LibraryError):
        lib.add_clip(voice.id, np.zeros(0, np.float32), "x", "recorded")
    with pytest.raises(LibraryError):
        lib.add_clip(voice.id, audio(), "x", "bogus")
    assert list((tmp_path / "voices" / voice.id / "clips").iterdir()) == []


def test_a_voice_json_of_the_wrong_shape_hides_only_that_voice(tmp_path):
    lib = Library(tmp_path)
    good, bad = lib.create("Good"), lib.create("Bad")
    (tmp_path / "voices" / bad.id / "voice.json").write_text("[1, 2]")
    assert [v.id for v in lib.list()] == [good.id]
    lib.add_clip(good.id, audio(), "x", "recorded")  # saving the good one still works


def _break_summary(monkeypatch):
    real = library_module._atomic_json

    def picky(path, obj):
        if path.name == "voices.json":
            raise OSError("disk full")
        return real(path, obj)

    monkeypatch.setattr(library_module, "_atomic_json", picky)


def test_an_unwritable_summary_never_fails_or_undoes_the_real_change(tmp_path, monkeypatch, caplog):
    _break_summary(monkeypatch)
    lib = Library(tmp_path)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        voice = lib.create("V")
        lib.add_clip(voice.id, audio(), "one", "recorded")
        lib.add_clip(voice.id, audio(), "two", "recorded")
        lib.update_text(voice.id, 1, "uno")
        lib.delete_clip(voice.id, 2)
        assert [(c.index, c.text) for c in Library(tmp_path).get(voice.id).clips] == [(1, "uno")]
        lib.delete(voice.id)
    assert lib.list() == []
    assert len(list((tmp_path / "trash").iterdir())) == 1
    assert any("voices.json" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "voices.json").exists()


def test_update_text_keeps_the_saved_record_when_the_txt_copy_fails(tmp_path, monkeypatch, caplog):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "old", "recorded")
    real = library_module._atomic_text

    def failing(path, text):
        raise OSError("locked")

    monkeypatch.setattr(library_module, "_atomic_text", failing)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        clip = lib.update_text(voice.id, 1, "new")
    monkeypatch.setattr(library_module, "_atomic_text", real)
    assert clip.text == "new"
    assert Library(tmp_path).get(voice.id).clips[0].text == "new"
    assert any("0001.txt" in r.getMessage() for r in caplog.records)


def test_create_retries_when_the_generated_id_is_taken(tmp_path, monkeypatch):
    values = iter(["aaaaaa", "aaaaaa", "bbbbbb"])
    monkeypatch.setattr(library_module.secrets, "token_hex", lambda n: next(values))
    lib = Library(tmp_path)
    first = lib.create("Same")
    lib.add_clip(first.id, audio(), "keep me", "recorded")
    second = lib.create("Same")
    assert (first.id, second.id) == ("same-aaaaaa", "same-bbbbbb")
    assert [c.text for c in lib.get(first.id).clips] == ["keep me"]
    assert (tmp_path / "voices" / first.id / "clips" / "0001.orig.wav").is_file()


def test_a_stranded_audio_file_is_never_served_or_reused(tmp_path, monkeypatch, caplog):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    real = type(tmp_path).unlink

    def locked(self, missing_ok=False):
        if self.name.endswith(".orig.wav"):
            raise PermissionError("in use")
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(type(tmp_path), "unlink", locked)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        lib.delete_clip(voice.id, 1)
    monkeypatch.undo()
    assert any("0001.orig.wav" in r.getMessage() for r in caplog.records)
    assert (tmp_path / "voices" / voice.id / "clips" / "0001.orig.wav").is_file()  # orphan on disk
    assert lib.get(voice.id).clips == []
    with pytest.raises(LibraryError):
        lib.clip_audio_path(voice.id, 1)
    assert lib.add_clip(voice.id, audio(), "y", "recorded").index == 2


def test_created_timestamps_order_across_a_second_boundary(tmp_path, monkeypatch):
    values = iter([1_700_000_000_999_999_999, 1_700_000_001_000_000_001])
    monkeypatch.setattr(library_module, "_now_ns", lambda: next(values))
    lib = Library(tmp_path)
    first, second = lib.create("a"), lib.create("b")
    assert first.created < second.created
    assert [v.id for v in lib.list()] == [first.id, second.id]
