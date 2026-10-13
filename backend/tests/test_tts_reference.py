import pytest

from aivoice.library import Clip, Voice
from aivoice import tts_reference

NONE = "This voice has no clip between 3 and 12 seconds with a transcript. Record a sentence for it first."


def clip(index, seconds, source="recorded", text="Hello there."):
    return Clip(index=index, text=text, seconds=seconds, source=source)


def voice(*clips):
    return Voice(id="v", name="V", created="2026-01-01", clips=list(clips))


def test_closest_to_eight_seconds_wins():
    assert tts_reference.choose(voice(clip(1, 4), clip(2, 7.5), clip(3, 11))).index == 2


def test_recorded_before_imported_even_when_import_is_closer():
    v = voice(clip(1, 8.0, "imported"), clip(2, 4.0, "recorded"))
    assert tts_reference.choose(v).index == 2


def test_lowest_index_breaks_a_tie():
    v = voice(clip(5, 7.0), clip(2, 9.0), clip(9, 7.0))
    assert tts_reference.choose(v).index == 2


def test_limits_are_inclusive():
    assert tts_reference.choose(voice(clip(1, 3.0))).index == 1
    assert tts_reference.choose(voice(clip(1, 12.0))).index == 1


@pytest.mark.parametrize("c", [
    clip(1, 2.99), clip(1, 12.01), clip(1, 8, text=""), clip(1, 8, text="  \n "),
])
def test_exclusions(c):
    with pytest.raises(ValueError) as e:
        tts_reference.choose(voice(c))
    assert str(e.value) == NONE


def test_excluded_clips_are_skipped_not_fatal():
    v = voice(clip(1, 2), clip(2, 8, text=""), clip(3, 30), clip(4, 12, "imported"))
    assert tts_reference.choose(v).index == 4


def test_no_clips_at_all():
    with pytest.raises(ValueError, match="no clip between 3 and 12"):
        tts_reference.choose(voice())


def test_describe_ok_and_not_ok():
    d = tts_reference.describe(voice(clip(3, 7.25)))
    assert d["ok"] is True and d["clip"] == 3 and d["seconds"] == 7.25 and isinstance(d["message"], str) and d["message"]
    bad = tts_reference.describe(voice(clip(1, 1.0)))
    assert bad == {"ok": False, "clip": None, "seconds": None, "message": NONE}


# ---- the level of the clip (a clip too quiet to copy is refused by the worker, so it is not chosen) ----

def levels(**by_index):
    """level(clip) from a table; a clip that is not in it cannot be read."""
    asked = []

    def level(c):
        asked.append(c.index)
        return by_index.get(f"c{c.index}")

    level.asked = asked
    return level


def test_the_floor_is_the_one_the_worker_refuses_below():
    from tests.test_tts_worker import MODULE, load
    assert tts_reference.MIN_RMS == load(MODULE, "aivoice_tts_worker_levels").MIN_REFERENCE_RMS == 0.003


def test_a_clip_that_is_loud_enough_is_chosen_as_before_and_no_other_clip_is_read():
    level = levels(c1=0.05, c2=0.05, c3=0.05)
    assert tts_reference.choose(voice(clip(1, 4), clip(2, 7.5), clip(3, 11)), level).index == 2
    assert level.asked == [2]


def test_a_quiet_clip_is_passed_over_for_the_next_best_one():
    v = voice(clip(1, 4), clip(2, 7.5), clip(3, 11, "imported"))
    assert tts_reference.choose(v, levels(c1=0.05, c2=0.0029, c3=0.05)).index == 1
    assert tts_reference.choose(v, levels(c1=0.001, c2=0.0, c3=0.003)).index == 3  # the floor itself is enough
    described = tts_reference.describe(v, levels(c1=0.05, c2=0.0029, c3=0.05))
    assert described["ok"] is True and described["clip"] == 1 and described["seconds"] == 4
    assert described["message"] == "Clip 1 (4.0 seconds) will be used as the reference."


def test_a_level_that_is_not_a_number_counts_as_too_quiet():
    v = voice(clip(1, 8), clip(2, 5))
    assert tts_reference.choose(v, levels(c1=float("nan"), c2=0.05)).index == 2


def test_a_voice_whose_only_clip_is_too_quiet_is_not_usable_and_says_what_to_do():
    message = "Clip 3 is too quiet to copy. Record a louder clip of 5 to 10 seconds."
    v = voice(clip(1, 2.0), clip(3, 7.0))
    with pytest.raises(ValueError) as refused:
        tts_reference.choose(v, levels(c3=0.001))
    assert str(refused.value) == message == tts_reference.too_quiet([3])
    assert tts_reference.describe(v, levels(c3=0.001)) == {"ok": False, "clip": None, "seconds": None,
                                                         "message": message}


def test_several_clips_that_are_all_too_quiet_are_named_together():
    v = voice(clip(1, 4), clip(2, 7.5), clip(5, 11))
    with pytest.raises(ValueError) as refused:
        tts_reference.choose(v, levels(c1=0.001, c2=0.002, c5=0.0))
    assert str(refused.value) == ("Clips 1, 2 and 5 are too quiet to copy. Record a louder clip of 5 to 10 "
                                  "seconds.")
    many = voice(*(clip(i, 8) for i in range(1, 8)))
    with pytest.raises(ValueError) as refused:
        tts_reference.choose(many, lambda c: 0.0)
    assert str(refused.value) == ("All 7 clips of this voice that are between 3 and 12 seconds long are too quiet "
                                  "to copy. Record a louder clip of 5 to 10 seconds.")


def test_a_clip_whose_audio_cannot_be_read_is_chosen_only_when_nothing_is_known_to_be_good():
    # so that the caller can say that its audio is missing, as it did before levels were looked at
    v = voice(clip(1, 4), clip(2, 7.5), clip(3, 11))
    assert tts_reference.choose(v, levels(c1=0.05)).index == 1  # 2 cannot be read: 1 is known to be good
    assert tts_reference.choose(v, levels(c1=0.001)).index == 2  # nothing good: the best one that may be
    assert tts_reference.choose(v, levels()).index == 2


def test_a_voice_without_a_usable_clip_says_so_whatever_the_levels():
    with pytest.raises(ValueError) as refused:
        tts_reference.choose(voice(clip(1, 2.0), clip(2, 8, text="")), lambda c: 0.0)
    assert str(refused.value) == NONE


def test_clip_level_is_the_rms_of_the_file_as_the_worker_measures_it(tmp_path):
    import numpy as np
    from aivoice.audiofile import write_wav

    path = tmp_path / "0001.orig.wav"
    t = np.arange(48000 * 3) / 48000
    write_wav(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    assert tts_reference.clip_level(path) == pytest.approx(0.1 / 2 ** 0.5, rel=1e-3)
    write_wav(path, np.zeros(48000, np.float32), 48000)  # the file changed: it is measured again
    assert tts_reference.clip_level(path) == 0.0
    assert tts_reference.clip_level(tmp_path / "missing.wav") is None
    (tmp_path / "broken.wav").write_bytes(b"not a wav file")
    assert tts_reference.clip_level(tmp_path / "broken.wav") is None


def test_clip_level_reads_an_unchanged_file_only_once(tmp_path, monkeypatch):
    import numpy as np
    from aivoice.audiofile import write_wav

    path = tmp_path / "0002.orig.wav"
    write_wav(path, np.full(48000, 0.25, np.float32), 48000)
    reads = []
    real = tts_reference.read_wav
    monkeypatch.setattr(tts_reference, "read_wav", lambda p: (reads.append(p), real(p))[1])
    assert tts_reference.clip_level(path) == pytest.approx(0.25, rel=1e-3)
    assert tts_reference.clip_level(path) == pytest.approx(0.25, rel=1e-3)
    assert len(reads) == 1


def test_library_levels_reads_the_voices_own_clip_and_never_writes(tmp_path):
    import numpy as np
    from aivoice.library import Library

    library = Library(tmp_path / "data")
    created = library.create("Quiet")
    quiet = np.random.default_rng(0).uniform(-0.002, 0.002, 8 * 48000).astype(np.float32)
    loud = np.random.default_rng(1).uniform(-0.2, 0.2, 4 * 48000).astype(np.float32)
    library.add_clip(created.id, quiet, "This one is too quiet.", "recorded")
    library.add_clip(created.id, loud, "This one is fine.", "recorded")
    folder = tmp_path / "data" / "voices" / created.id / "clips"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    v = library.get(created.id)
    level = tts_reference.library_levels(library, v.id)
    assert level(v.clips[0]) < tts_reference.MIN_RMS < level(v.clips[1])
    assert tts_reference.choose(v, level).index == 2
    (folder / "0002.orig.wav").unlink()
    assert level(v.clips[1]) is None
    del before["0002.orig.wav"]
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
