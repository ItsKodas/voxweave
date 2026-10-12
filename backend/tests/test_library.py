import errno
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
    real = os.rename

    def locked(src, dst):
        if str(src).endswith(".orig.wav"):
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr("aivoice.library.os.rename", locked)
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
    assert len(list((tmp_path / "trash").glob(f"{voice.id}-*"))) == 1  # the voice itself
    assert (tmp_path / "trash" / voice.id / "clips" / "0002.orig.wav").is_file()  # and the clip deleted before it
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
    real = os.rename

    def locked(src, dst):
        if str(src).endswith(".orig.wav"):
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr("aivoice.library.os.rename", locked)
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


# --- deleted clips go to the trash -------------------------------------------------------------


def _clip_bytes(folder, stem="0001"):
    return {suffix: (folder / f"{stem}{suffix}").read_bytes() for suffix in (".orig.wav", ".wav", ".txt")}


def test_delete_clip_moves_its_three_files_to_the_trash_intact(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "keep me", "recorded")
    lib.add_clip(voice.id, audio(1.0), "the other", "recorded")
    clips = tmp_path / "voices" / voice.id / "clips"
    before = _clip_bytes(clips)
    lib.delete_clip(voice.id, 1)
    trash = tmp_path / "trash" / voice.id / "clips"
    assert sorted(p.name for p in trash.iterdir()) == ["0001.orig.wav", "0001.txt", "0001.wav"]  # and nothing else
    assert _clip_bytes(trash) == before
    assert [p.name for p in (tmp_path / "trash" / voice.id).iterdir()] == ["clips"]
    assert sorted(p.name for p in clips.iterdir()) == ["0002.orig.wav", "0002.txt", "0002.wav"]
    assert [c.index for c in lib.get(voice.id).clips] == [2]
    restored, sr = read_wav(trash / "0001.orig.wav")
    assert sr == 48000 and abs(len(restored) - 2 * SR) <= 2


def test_two_deleted_clips_with_the_same_file_names_are_both_kept_in_the_trash(tmp_path, monkeypatch):
    monkeypatch.setattr("aivoice.library._now", lambda: 1000.0)
    lib = Library(tmp_path)
    voice = lib.create("V")
    record = tmp_path / "voices" / voice.id / "voice.json"
    fresh = record.read_bytes()
    clips = tmp_path / "voices" / voice.id / "clips"
    kept = []
    for seconds in (1.0, 2.0, 3.0):
        # The record is put back to "no clips yet" (as after restoring an old copy), so number 1 is handed out again.
        record.write_bytes(fresh)
        assert lib.add_clip(voice.id, audio(seconds), f"{seconds} s", "recorded").index == 1
        kept.append(_clip_bytes(clips))
        lib.delete_clip(voice.id, 1)
    trash = tmp_path / "trash" / voice.id / "clips"
    names = sorted(p.name for p in trash.iterdir())
    assert len(names) == 9 and list(clips.iterdir()) == []
    stems = sorted({n.split(".")[0] for n in names})
    assert len(stems) == 3 and stems[0] == "0001"
    # Each clip's three files share one name, and every deleted clip's content is still there.
    found = [_clip_bytes(trash, stem) for stem in stems]
    assert all(clip in found for clip in kept) and len(found) == len(kept)


def test_a_clip_file_that_cannot_be_moved_to_the_trash_is_left_in_place_not_erased(tmp_path, monkeypatch, caplog):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    clips = tmp_path / "voices" / voice.id / "clips"
    before = _clip_bytes(clips)
    real = os.rename

    def locked(src, dst):
        if str(src).endswith(".orig.wav"):
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr("aivoice.library.os.rename", locked)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        lib.delete_clip(voice.id, 1)
    monkeypatch.undo()
    assert any("0001.orig.wav" in r.getMessage() for r in caplog.records)
    assert lib.get(voice.id).clips == []
    trash = tmp_path / "trash" / voice.id / "clips"
    assert (clips / "0001.orig.wav").read_bytes() == before[".orig.wav"]  # stranded, still whole
    assert (trash / "0001.wav").read_bytes() == before[".wav"]
    assert (trash / "0001.txt").read_bytes() == before[".txt"]


def test_delete_clip_never_overwrites_a_file_already_in_the_trash(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    trash = tmp_path / "trash" / voice.id / "clips"
    trash.mkdir(parents=True)
    (trash / "0001.wav").write_bytes(b"an earlier recording")
    lib.delete_clip(voice.id, 1)
    assert (trash / "0001.wav").read_bytes() == b"an earlier recording"
    assert len(list(trash.iterdir())) == 4
    assert list((tmp_path / "voices" / voice.id / "clips").iterdir()) == []


def test_delete_clip_with_an_unusable_trash_folder_keeps_the_files_where_they_are(tmp_path, caplog):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    (tmp_path / "trash").write_text("a file where the trash folder should be")
    with caplog.at_level("WARNING", logger="aivoice.library"):
        lib.delete_clip(voice.id, 1)
    assert lib.get(voice.id).clips == []
    clips = tmp_path / "voices" / voice.id / "clips"
    assert sorted(p.name for p in clips.iterdir()) == ["0001.orig.wav", "0001.txt", "0001.wav"]
    assert any("trash" in r.getMessage() for r in caplog.records)


# --- disk failures are readable ----------------------------------------------------------------


def _no_space(*args, **kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


def _record_writes_fail(monkeypatch):
    real = library_module._atomic_json

    def picky(path, obj):
        if path.name == "voice.json":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, obj)

    monkeypatch.setattr(library_module, "_atomic_json", picky)


def test_add_clip_on_a_full_disk_is_a_readable_error_and_registers_nothing(tmp_path, monkeypatch, caplog):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "first", "recorded")
    record = tmp_path / "voices" / voice.id / "voice.json"
    before = record.read_bytes()
    monkeypatch.setattr(library_module, "write_wav", _no_space)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        with pytest.raises(LibraryError, match="Could not save the clip: No space left on device") as caught:
            lib.add_clip(voice.id, audio(), "second", "recorded")
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, OSError) and caught.value.__cause__.errno == errno.ENOSPC
    assert any("No space left" in r.getMessage() for r in caplog.records)
    assert record.read_bytes() == before
    assert [c.text for c in Library(tmp_path).get(voice.id).clips] == ["first"]
    assert lib.add_clip(voice.id, audio(), "second again", "recorded").index == 2  # the number was not used up


def test_add_clip_failing_part_way_leaves_only_harmless_strays(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    voice = lib.create("V")
    record = tmp_path / "voices" / voice.id / "voice.json"
    before = record.read_bytes()
    real = library_module.write_wav

    def second_write_fails(path, samples, rate):
        if not str(path).endswith(".orig.wav"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, samples, rate)

    monkeypatch.setattr(library_module, "write_wav", second_write_fails)
    with pytest.raises(LibraryError, match="No space left"):
        lib.add_clip(voice.id, audio(), "x", "recorded")
    monkeypatch.undo()
    assert record.read_bytes() == before and lib.get(voice.id).clips == []
    with pytest.raises(LibraryError):
        lib.clip_audio_path(voice.id, 1)  # the stray audio is never served
    clip = lib.add_clip(voice.id, audio(1.0), "y", "recorded")  # and the next save replaces it
    assert clip.index == 1
    stored, _ = read_wav(tmp_path / "voices" / voice.id / "clips" / "0001.orig.wav")
    assert abs(len(stored) - SR) <= 2


def test_add_clip_whose_record_cannot_be_written_is_a_readable_error(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    voice = lib.create("V")
    record = tmp_path / "voices" / voice.id / "voice.json"
    before = record.read_bytes()
    _record_writes_fail(monkeypatch)
    with pytest.raises(LibraryError, match="Could not save the clip: No space left on device"):
        lib.add_clip(voice.id, audio(), "x", "recorded")
    monkeypatch.undo()
    assert record.read_bytes() == before and lib.get(voice.id).clips == []


def test_create_on_a_full_disk_is_a_readable_error_and_lists_no_half_made_voice(tmp_path, monkeypatch, caplog):
    lib = Library(tmp_path)
    kept = lib.create("Kept")
    _record_writes_fail(monkeypatch)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        with pytest.raises(LibraryError, match="Could not create the voice: No space left on device") as caught:
            lib.create("New")
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, OSError)
    assert any("No space left" in r.getMessage() for r in caplog.records)
    assert [v.id for v in lib.list()] == [kept.id]
    assert [p.name for p in (tmp_path / "voices").iterdir()] == [kept.id]  # the empty new folder was not left behind


def test_create_when_the_voices_folder_cannot_be_made_is_a_readable_error(tmp_path):
    (tmp_path / "voices").write_text("a file where the voices folder should be")
    with pytest.raises(LibraryError, match="Could not create the voice: "):
        Library(tmp_path).create("New")


def test_update_text_on_a_full_disk_is_a_readable_error_and_keeps_the_old_transcript(tmp_path, monkeypatch, caplog):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "old", "recorded")
    _record_writes_fail(monkeypatch)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        with pytest.raises(LibraryError, match="Could not save the transcript: No space left on device") as caught:
            lib.update_text(voice.id, 1, "new")
    monkeypatch.undo()
    assert isinstance(caught.value.__cause__, OSError)
    assert any("No space left" in r.getMessage() for r in caplog.records)
    assert Library(tmp_path).get(voice.id).clips[0].text == "old"
    assert (tmp_path / "voices" / voice.id / "clips" / "0001.txt").read_text(encoding="utf-8") == "old"


def test_delete_clip_on_a_full_disk_is_a_readable_error_and_keeps_the_clip(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    clips = tmp_path / "voices" / voice.id / "clips"
    before = _clip_bytes(clips)
    _record_writes_fail(monkeypatch)
    with pytest.raises(LibraryError, match="Could not delete the clip: No space left on device"):
        lib.delete_clip(voice.id, 1)
    monkeypatch.undo()
    assert [c.index for c in lib.get(voice.id).clips] == [1]
    assert _clip_bytes(clips) == before and not (tmp_path / "trash").exists()
    assert lib.clip_audio_path(voice.id, 1).is_file()


# --- damaged folders are reported --------------------------------------------------------------


def test_a_damaged_voice_folder_is_reported_in_the_log_without_its_contents(tmp_path, caplog):
    lib = Library(tmp_path)
    good, bad = lib.create("Good"), lib.create("Bad")
    (tmp_path / "voices" / bad.id / "voice.json").write_text('{"name": "private-name", half written')
    with caplog.at_level("WARNING", logger="aivoice.library"):
        assert [v.id for v in Library(tmp_path).list()] == [good.id]
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(messages) == 1
    assert bad.id in messages[0] and "JSON" in messages[0]
    assert "private-name" not in messages[0] and "half written" not in messages[0]


@pytest.mark.parametrize("content, reason", [
    ("[1, 2]", "TypeError"),
    ('{"name": "private-name", "created": "c"}', "KeyError"),
    ('{"name": "private-name", "created": "c", "clips": [], "next_index": "private-number"}', "ValueError"),
])
def test_a_voice_file_of_the_wrong_shape_is_reported_by_kind_only(tmp_path, caplog, content, reason):
    lib = Library(tmp_path)
    bad = lib.create("Bad")
    (tmp_path / "voices" / bad.id / "voice.json").write_text(content)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        assert lib.list() == []
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1 and bad.id in messages[0] and reason in messages[0]
    assert "private" not in messages[0]


def test_a_folder_without_a_voice_file_is_reported_once_not_on_every_listing(tmp_path, caplog):
    (tmp_path / "voices" / "stray-folder").mkdir(parents=True)
    (tmp_path / "voices" / "notes.txt").write_text("not a folder: ignored quietly")
    lib = Library(tmp_path)
    with caplog.at_level("WARNING", logger="aivoice.library"):
        for _ in range(3):
            assert lib.list() == []
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1 and "stray-folder" in messages[0]


def add_model(tmp_path, voice_id, meta='{"name": "Trained", "sample_rate": 40000}', model=True):
    folder = tmp_path / "voices" / voice_id / "models" / "rvc"
    folder.mkdir(parents=True)
    if model:
        (folder / "model.onnx").write_bytes(b"x")
    if meta is not None:
        (folder / "voice.json").write_text(meta)


def test_voice_models_reflect_the_files_on_disk(tmp_path):
    lib = Library(tmp_path)
    plain, trained = lib.create("Plain"), lib.create("Trained")
    assert lib.get(trained.id).models == []
    add_model(tmp_path, trained.id)
    assert lib.get(trained.id).models == ["rvc"]
    assert lib.get(plain.id).models == []
    assert {v.id: v.models for v in lib.list()} == {plain.id: [], trained.id: ["rvc"]}


@pytest.mark.parametrize("meta, model", [(None, True), ("{nope", True), ('{"name": "x"}', True),
                                         ('{"name": "x", "sample_rate": "fast"}', True),
                                         ('{"name": "x", "sample_rate": 40000}', False)])
def test_damaged_model_folder_means_no_model(tmp_path, meta, model):
    lib = Library(tmp_path)
    voice = lib.create("V")
    add_model(tmp_path, voice.id, meta=meta, model=model)
    assert lib.get(voice.id).models == []


def test_summary_carries_models_but_voice_json_does_not(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    add_model(tmp_path, voice.id)
    lib.add_clip(voice.id, audio(60.0), "x", "recorded")
    manifest = json.loads((tmp_path / "voices.json").read_text(encoding="utf-8"))
    assert manifest[0]["models"] == ["rvc"]
    assert "models" not in json.loads((tmp_path / "voices" / voice.id / "voice.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("rate, ok", [("0", False), ("-5", False), ("true", False), ('"40000"', False),
                                      ("40000.0", True), ("40000.5", False), ("1e999", False),
                                      ("7999", False), ("192001", False), ("48000", True)])
def test_only_a_sane_sample_rate_counts(tmp_path, rate, ok):
    lib = Library(tmp_path)
    voice = lib.create("V")
    add_model(tmp_path, voice.id, meta='{"name": "x", "sample_rate": %s}' % rate)
    assert lib.get(voice.id).models == (["rvc"] if ok else [])


@pytest.mark.parametrize("kind", ["inf", "nested", "directory"])
def test_a_damaged_model_file_never_affects_the_library(tmp_path, kind):
    lib = Library(tmp_path)
    voice = lib.create("V")
    add_model(tmp_path, voice.id, meta=None)
    target = tmp_path / "voices" / voice.id / "models" / "rvc" / "voice.json"
    if kind == "inf":
        target.write_text('{"sample_rate": 1e999}')
    elif kind == "nested":
        target.write_text("[" * 200000 + "]" * 200000)
    else:
        target.mkdir()
    assert [v.id for v in lib.list()] == [voice.id]
    assert lib.get(voice.id).models == []
    lib.add_clip(voice.id, audio(), "hello", "recorded")
    assert lib.update_text(voice.id, 1, "changed").text == "changed"
    assert lib.get(voice.id).models == []
