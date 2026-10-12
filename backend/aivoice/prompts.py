from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_prompts() -> tuple[str, ...]:
    text = Path(__file__).with_name("prompts_en.txt").read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def next_prompt(recorded: set[int], total: int, after: int | None = None) -> int | None:
    """The next sentence to read: the first one not yet recorded after `after`, wrapping round."""
    if after is None or not 0 <= after < total:
        order = range(total)
    else:
        order = [*range(after + 1, total), *range(0, after + 1)]
    return next((i for i in order if i not in recorded), None)
