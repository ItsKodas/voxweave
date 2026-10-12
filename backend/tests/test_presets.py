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


def test_builtin_presets_use_dio():
    # Benchmarked on this machine: harvest pitch extraction is slower than real time,
    # so no built-in preset may use it. "harvest" stays a valid f0_method override.
    assert all(p.f0_method == "dio" for p in PRESETS.values())
    assert resolve("balanced", {"f0_method": "harvest"}).f0_method == "harvest"
