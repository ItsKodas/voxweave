from dataclasses import dataclass, fields, replace

F0_METHODS = ("dio", "harvest", "rmvpe")


@dataclass(frozen=True)
class LiveParams:
    chunk_ms: int
    crossfade_ms: int
    context_ms: int
    f0_method: str
    index_rate: float
    protect: float = 0.33  # consonants keep this much of the index's pull; 0.5 or more protects nothing
    lookahead_ms: int = 0  # the played audio ends this far before the newest input: the model has heard past it


# Every preset uses RMVPE, the neural pitch tracker the voices were trained with (it falls back to "dio" when its
# model is missing). "harvest" benchmarked slower than real time at every preset window.
PRESETS: dict[str, LiveParams] = {
    "low_latency": LiveParams(120, 30, 400, "rmvpe", 0.0),
    "balanced": LiveParams(250, 50, 800, "rmvpe", 0.3),
    "high_quality": LiveParams(500, 80, 1500, "rmvpe", 0.6),
}


# The quality slider on the Live tab: longer chunks with more context before them give the model and the pitch
# tracker more to go on, at the cost of delay. Level 1 is the fastest; the last level is the most the app allows.
# lookahead_ms is 0 at every step: playing audio from before the newest input was tried and measured, and added delay
# without making the joins any smoother.
QUALITY_LEVELS: list[dict] = [
    {"chunk_ms": 120, "context_ms": 400, "crossfade_ms": 30, "lookahead_ms": 0},
    {"chunk_ms": 250, "context_ms": 800, "crossfade_ms": 50, "lookahead_ms": 0},
    {"chunk_ms": 500, "context_ms": 1500, "crossfade_ms": 80, "lookahead_ms": 0},
    {"chunk_ms": 750, "context_ms": 2500, "crossfade_ms": 100, "lookahead_ms": 0},
    {"chunk_ms": 1000, "context_ms": 3000, "crossfade_ms": 120, "lookahead_ms": 0},
    {"chunk_ms": 1500, "context_ms": 4000, "crossfade_ms": 150, "lookahead_ms": 0},
    {"chunk_ms": 2000, "context_ms": 5000, "crossfade_ms": 200, "lookahead_ms": 0},
]


def _validate(p: LiveParams) -> None:
    if not 40 <= p.chunk_ms <= 2000:
        raise ValueError("chunk_ms must be between 40 and 2000")
    if not 5 <= p.crossfade_ms <= p.chunk_ms:
        raise ValueError("crossfade_ms must be between 5 and chunk_ms")
    if not 0 <= p.context_ms <= 5000:
        raise ValueError("context_ms must be between 0 and 5000")
    if p.f0_method not in F0_METHODS:
        raise ValueError(f"f0_method must be one of {F0_METHODS}")
    if not 0.0 <= p.index_rate <= 1.0:
        raise ValueError("index_rate must be between 0 and 1")
    if not 0.0 <= p.protect <= 0.5:
        raise ValueError("protect must be between 0 and 0.5")
    if not 0 <= p.lookahead_ms <= min(500, p.context_ms):
        raise ValueError("lookahead_ms must be between 0 and the smaller of 500 and context_ms")


def resolve(preset: str, overrides: dict | None = None) -> LiveParams:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    params = PRESETS[preset]
    if overrides:
        known = {f.name for f in fields(LiveParams)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        params = replace(params, **overrides)
    _validate(params)
    return params
