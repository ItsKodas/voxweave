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
