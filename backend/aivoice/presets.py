from dataclasses import dataclass, fields, replace

F0_METHODS = ("dio", "harvest")


@dataclass(frozen=True)
class LiveParams:
    chunk_ms: int
    crossfade_ms: int
    context_ms: int
    f0_method: str
    index_rate: float


# high_quality uses "dio" too: harvest pitch extraction benchmarked slower than real time here.
PRESETS: dict[str, LiveParams] = {
    "low_latency": LiveParams(120, 30, 400, "dio", 0.0),
    "balanced": LiveParams(250, 50, 800, "dio", 0.3),
    "high_quality": LiveParams(500, 80, 1500, "dio", 0.6),
}


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
