import pytest

from aivoice.presets import PRESETS, LiveParams, resolve


def test_three_presets_exist():
    assert set(PRESETS) == {"low_latency", "balanced", "high_quality"}


def test_presets_order_by_chunk_size():
    assert PRESETS["low_latency"].chunk_ms < PRESETS["balanced"].chunk_ms < PRESETS["high_quality"].chunk_ms


def test_resolve_returns_preset():
    assert resolve("balanced") == PRESETS["balanced"]


def test_resolve_applies_overrides():
    params = resolve("balanced", {"chunk_ms": 300, "index_rate": 0.5})
    assert params.chunk_ms == 300 and params.index_rate == 0.5
    assert params.crossfade_ms == PRESETS["balanced"].crossfade_ms


def test_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve("nope")


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_ms": 10},
        {"chunk_ms": 5000},
        {"crossfade_ms": 2},
        {"chunk_ms": 100, "crossfade_ms": 200},
        {"context_ms": -1},
        {"context_ms": 9000},
        {"f0_method": "magic"},
        {"index_rate": 1.5},
        {"index_rate": -0.1},
        {"bogus": 1},
    ],
)
def test_invalid_overrides_rejected(overrides):
    with pytest.raises(ValueError):
        resolve("balanced", overrides)


def test_all_presets_are_valid():
    for name in PRESETS:
        assert isinstance(resolve(name), LiveParams)


def test_builtin_presets_use_rmvpe_and_protect_consonants():
    # RMVPE is the tracker the voices were trained with; harvest is slower than real time, so no preset uses it.
    # "harvest" and "dio" stay valid overrides.
    assert all(p.f0_method == "rmvpe" for p in PRESETS.values())
    assert all(p.protect == 0.33 for p in PRESETS.values())
    assert resolve("balanced", {"f0_method": "dio"}).f0_method == "dio"
    assert resolve("balanced", {"protect": 0.5}).protect == 0.5
    for bad in (-0.1, 0.6):
        with pytest.raises(ValueError, match="protect"):
            resolve("balanced", {"protect": bad})
    assert resolve("balanced", {"f0_method": "harvest"}).f0_method == "harvest"


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_volume_follow_must_be_between_0_and_1(value):
    with pytest.raises(ValueError, match="volume_follow"):
        resolve("balanced", {"volume_follow": value})


@pytest.mark.parametrize("value", [-1.0, 61.0])
def test_background_cut_must_be_between_0_and_60_db(value):
    with pytest.raises(ValueError, match="background_cut_db"):
        resolve("balanced", {"background_cut_db": value})


def test_every_voice_quality_extra_is_on_by_default_and_can_be_switched():
    p = resolve("balanced")
    assert (p.pitch_cleanup, p.creak_repair, p.join_smoothing) == (True, True, True)
    assert p.natural_variation_st == pytest.approx(0.25)
    off = resolve("balanced", {"pitch_cleanup": False, "creak_repair": False, "join_smoothing": False,
                               "natural_variation_st": 0.0})
    assert (off.pitch_cleanup, off.creak_repair, off.join_smoothing, off.natural_variation_st) == (False, False, False, 0.0)


@pytest.mark.parametrize("overrides, field", [({"pitch_cleanup": "yes"}, "pitch_cleanup"), ({"creak_repair": 1}, "creak_repair"),
                                              ({"join_smoothing": None}, "join_smoothing"),
                                              ({"natural_variation_st": 0.9}, "natural_variation_st")])
def test_extras_are_checked(overrides, field):
    with pytest.raises(ValueError, match=field):
        resolve("balanced", overrides)



def test_matching_the_speakers_volume_is_on_by_default_and_checked():
    assert resolve("balanced").match_input_level is True
    assert resolve("balanced", {"match_input_level": False}).match_input_level is False
    with pytest.raises(ValueError, match="match_input_level"):
        resolve("balanced", {"match_input_level": "no"})
