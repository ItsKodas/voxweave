from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_prompts() -> tuple[str, ...]:
    text = Path(__file__).with_name("prompts_en.txt").read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


@lru_cache(maxsize=1)
def load_paragraphs() -> tuple[str, ...]:
    """Longer passages of two or three sentences, each short enough to read aloud in well under 20 seconds."""
    text = Path(__file__).with_name("prompts_en_long.txt").read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


@lru_cache(maxsize=1)
def load_pages() -> tuple[tuple[str, str], ...]:
    """Full pages to read in one go, as (title, text): each "# Title" line starts a page."""
    text = Path(__file__).with_name("pages_en.txt").read_text(encoding="utf-8")
    pages: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("# "):
            pages.append((line[2:].strip(), []))
        elif pages:
            pages[-1][1].append(line.rstrip())
    return tuple((title, "\n".join(lines).strip()) for title, lines in pages)


def next_prompt(recorded: set[int], total: int, after: int | None = None) -> int | None:
    """The next sentence to read: the first one not yet recorded after `after`, wrapping round."""
    if after is None or not 0 <= after < total:
        order = range(total)
    else:
        order = [*range(after + 1, total), *range(0, after + 1)]
    return next((i for i in order if i not in recorded), None)
