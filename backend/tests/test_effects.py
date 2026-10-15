from dataclasses import replace

import numpy as np
import pytest

from aivoice.effects import Effects
from aivoice.presets import PRESETS, resolve

SR = 48000
BASE = PRESETS["balanced"]


def run(effects, audio, params, block=SR // 20):
    """The audio through the effects in blocks, as live mode feeds them (50 ms each by default)."""
    return np.concatenate([effects.process(audio[i:i + block], params) for i in range(0, len(audio), block)])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def tone(hz, seconds=1.0, amp=0.3):
    return (amp * np.sin(2 * np.pi * hz * np.arange(int(seconds * SR)) / SR)).astype(np.float32)


def test_with_every_effect_off_the_voice_is_untouched():
    audio = tone(220)
    assert np.array_equal(run(Effects(SR), audio, BASE), audio)


def test_echo_repeats_the_voice_after_the_delay_even_when_it_is_longer_than_a_chunk():
    params = replace(BASE, fx_echo=0.5, fx_echo_ms=300)
    impulse = np.zeros(SR, np.float32)
    impulse[1000] = 1.0
    out = run(Effects(SR), impulse, params)  # 50 ms blocks, a 300 ms echo
    first_repeat = 1000 + int(0.3 * SR)
    around = out[first_repeat - 300: first_repeat + 300]
    assert np.max(np.abs(around)) == pytest.approx(0.5, rel=0.1)
    second = out[first_repeat + int(0.3 * SR) - 300: first_repeat + int(0.3 * SR) + 300]
    assert 0.1 < np.max(np.abs(second)) < np.max(np.abs(around))  # the echo fades with each repeat


def test_reverb_leaves_a_tail_that_lasts_longer_in_a_bigger_room():
    impulse = np.zeros(SR * 2, np.float32)
    impulse[100] = 1.0
    small = run(Effects(SR), impulse, replace(BASE, fx_reverb=0.6, fx_room=0.1))
    large = run(Effects(SR), impulse, replace(BASE, fx_reverb=0.6, fx_room=1.0))
    late = slice(int(0.5 * SR), int(1.0 * SR))
    assert rms(large[late]) > 3 * rms(small[late]) > 0
    assert rms(small[int(0.05 * SR):int(0.2 * SR)]) > 1e-4  # a tail, not silence


def test_lofi_cuts_the_highs_and_keeps_the_voice():
    params = replace(BASE, fx_lofi=1.0)
    high, low = tone(9000), tone(300)
    assert rms(run(Effects(SR), high, params)[SR // 4:]) < 0.15 * rms(high)
    assert rms(run(Effects(SR), low, params)[SR // 4:]) > 0.6 * rms(low)


def test_static_adds_hiss_and_crackle_that_grow_with_the_slider():
    silence = np.zeros(SR, np.float32)
    assert rms(run(Effects(SR), silence, replace(BASE, fx_static=0.0))) == 0.0
    some = rms(run(Effects(SR), silence, replace(BASE, fx_static=0.3)))
    lots = rms(run(Effects(SR), silence, replace(BASE, fx_static=1.0)))
    assert 0 < some < lots < 0.2


def test_the_output_never_goes_past_full_scale():
    loud = tone(220, amp=0.95)
    out = run(Effects(SR), loud, replace(BASE, fx_echo=1.0, fx_reverb=1.0, fx_static=1.0, fx_lofi=0.5))
    assert np.max(np.abs(out)) <= 1.0


@pytest.mark.parametrize("overrides, field", [({"fx_lofi": 1.5}, "fx_lofi"), ({"fx_static": -0.1}, "fx_static"),
                                              ({"fx_reverb": 2}, "fx_reverb"), ({"fx_room": -1}, "fx_room"),
                                              ({"fx_echo": 1.2}, "fx_echo"), ({"fx_echo_ms": 10}, "fx_echo_ms")])
def test_effect_settings_are_checked(overrides, field):
    with pytest.raises(ValueError, match=field):
        resolve("balanced", overrides)


def test_effects_are_off_by_default_and_can_change_while_live():
    from aivoice.presets import LIVE_CHANGEABLE

    p = resolve("balanced")
    assert (p.fx_lofi, p.fx_static, p.fx_reverb, p.fx_echo) == (0.0, 0.0, 0.0, 0.0)
    assert {"fx_lofi", "fx_static", "fx_reverb", "fx_room", "fx_echo", "fx_echo_ms"} <= set(LIVE_CHANGEABLE)


def test_live_mode_applies_the_effects_to_what_it_plays():
    from aivoice.processor import StreamProcessor
    from tests.test_live import Identity

    params = replace(resolve("balanced"), fx_static=1.0)
    proc = StreamProcessor(Identity(), SR, params)
    out = np.concatenate([proc.process(np.zeros(proc.chunk, np.float32)) for _ in range(6)])
    assert rms(out) > 0  # silence in, static out


def test_lofi_holds_samples_across_blocks_like_one_long_stream():
    params = replace(BASE, fx_lofi=1.0, fx_static=0.0)
    audio = tone(300, 0.5)
    whole = Effects(SR).process(audio, params)
    pieces = run(Effects(SR), audio, params, block=777)  # block edges in the middle of held groups
    assert np.allclose(whole, pieces, atol=1e-5)


def test_effects_are_quick_enough_for_live_mode():
    import time

    params = replace(BASE, fx_lofi=0.6, fx_static=0.5, fx_reverb=0.7, fx_room=1.0, fx_echo=0.5, fx_echo_ms=400)
    effects = Effects(SR)
    chunk = tone(220, 0.25)
    effects.process(chunk, params)  # builds the impulse response once
    start = time.perf_counter()
    for _ in range(8):
        effects.process(chunk, params)
    assert (time.perf_counter() - start) / 8 < 0.05  # well under a 250 ms chunk
