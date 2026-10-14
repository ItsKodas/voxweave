import numpy as np

from aivoice.presets import PRESETS, QUALITY_LEVELS, resolve
from aivoice.processor import StreamProcessor, sola_offset

SR = 48000


def test_the_offset_finds_where_the_waveform_continues():
    t = np.arange(4000) / SR
    wave = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    tail = wave[1000:1400]
    candidate = wave[1000 - 137:]
    assert sola_offset(tail, candidate, 300) == 137


def test_with_nothing_to_match_the_join_stays_where_it_was():
    assert sola_offset(np.zeros(100, np.float32), np.ones(1000, np.float32), 50) == 50
    assert sola_offset(np.zeros(0, np.float32), np.ones(1000, np.float32), 50) == 50


class FreshPhase:
    """Like a voice model: every call makes the same tone but starting at a different phase, so consecutive chunks
    are out of step at the join."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def convert(self, audio, sample_rate, pitch_shift, params):
        phase = self.rng.uniform(0, 2 * np.pi)
        t = np.arange(len(audio)) / sample_rate
        return (0.5 * np.sin(2 * np.pi * 180 * t + phase)).astype(np.float32)


def run(processor, chunks=12):
    return np.concatenate([processor.process(np.zeros(processor.chunk, np.float32)) for _ in range(chunks)])


def test_joins_of_out_of_step_chunks_keep_their_level_with_the_search_and_dip_without_it():
    params = PRESETS["balanced"]
    with_search = StreamProcessor(FreshPhase(), SR, params)
    without = StreamProcessor(FreshPhase(), SR, params)
    without.search = 0
    a, b = run(with_search)[with_search.chunk:], run(without)[without.chunk:]

    def worst_dip(y, chunk, crossfade):
        # The quietest stretch around each join, relative to the tone's steady level.
        level = np.sqrt(np.mean(y[: chunk // 2] ** 2))
        dips = []
        for j in range(chunk, len(y) - chunk, chunk):
            around = y[j - crossfade: j + crossfade]
            frames = around[: len(around) // 240 * 240].reshape(-1, 240)
            dips.append(np.sqrt(np.mean(frames ** 2, axis=1)).min() / level)
        return min(dips)

    assert worst_dip(a, with_search.chunk, with_search.crossfade) > 0.85
    assert worst_dip(b, without.chunk, without.crossfade) < 0.5


def test_quality_levels_are_valid_settings_that_only_get_longer():
    previous = None
    for level in QUALITY_LEVELS:
        params = resolve("balanced", level)
        if previous is not None:
            assert params.chunk_ms > previous.chunk_ms and params.context_ms >= previous.context_ms
        previous = params
    assert QUALITY_LEVELS[1] == {"chunk_ms": 250, "context_ms": 800, "crossfade_ms": 50}  # the Balanced preset


def test_a_window_too_short_for_a_full_search_searches_what_it_has():
    params = resolve("balanced", {"context_ms": 0})
    p = StreamProcessor(FreshPhase(), SR, params)
    assert p.search == 0
    assert len(p.process(np.zeros(p.chunk, np.float32))) == p.chunk


def simulate(level, capacity, infer_blocks=0):
    """A live session fed in 10 ms blocks, with each conversion finishing infer_blocks blocks after its input is
    complete. Returns the output buffer's overflows and its underruns once audio flows."""
    from aivoice.live import LiveSession
    from aivoice.router import Router

    class Half:
        def convert(self, a, sr, p, params):
            return np.asarray(a, np.float32) * 0.5

    params = resolve("balanced", QUALITY_LEVELS[level])
    proc = StreamProcessor(Half(), SR, params)
    router = Router(capacity)
    router.add_output("mic")
    session = LiveSession(proc, router, SR, None, 0, {"mic": 1})
    router.prime(proc.chunk)
    block = 480
    pending = []
    for i in range(int(30 * SR / block)):
        session.feed(np.full((block, 1), 0.3, np.float32))
        if session._inbuf.available >= proc.chunk:
            pending.append(i + infer_blocks)
        while pending and pending[0] <= i:
            pending.pop(0)
            session.pump()
        session._fill("mic", block)
    buffer = router._outputs["mic"].buffer
    return buffer.overflows, buffer.underruns


def test_every_quality_level_fits_its_output_buffer_whatever_the_conversion_time():
    from aivoice.manager import buffer_samples

    for level in range(len(QUALITY_LEVELS)):
        chunk = SR * QUALITY_LEVELS[level]["chunk_ms"] // 1000
        for infer_fraction in (0.0, 0.5, 0.9):
            blocks = int(chunk * infer_fraction / 480)
            overflows, underruns = simulate(level, buffer_samples(SR, chunk), blocks)
            assert (overflows, underruns) == (0, 0), (level + 1, infer_fraction)


def test_the_old_fixed_two_seconds_was_too_little_for_long_chunks():
    assert simulate(6, SR * 2)[0] > 0
