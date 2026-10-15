"""What live mode is doing, for the graphs on the Live tab: the last few seconds of the pitch heard in the speaker's
voice and the pitch the model was given, where the clean-up stepped in, where background sound was turned down,
the input and output levels, how long each chunk took to convert against the time it had, and, about once a second,
how much the converted voice sounds like the voice (the speaker-recognition model of deep calibration, comparing
the last three seconds of output with the voice's own recordings). Scoring runs on its own thread, so it never
holds up the conversion; the audio thread only appends.
"""
import threading
from collections import deque
from typing import Callable

import numpy as np

WINDOW_SECONDS = 10
FPS = 50  # points a second in the graphs
MATCH_SECONDS = 3.0
MATCH_MIN_VOICED = 1.0 / 3.0  # of those seconds must have had a pitch
MATCH_POINTS = 120


def _levels(audio: np.ndarray, count: int) -> list[float]:
    edges = np.linspace(0, len(audio), count + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        x = audio[a:b].astype(np.float64)
        out.append(round(float(10.0 * np.log10(np.mean(x * x) + 1e-10)), 1) if b > a else -100.0)
    return out


def _resample(values: list, count: int) -> list:
    if not values:
        return [0] * count
    index = np.minimum((np.arange(count) * len(values) / count).astype(int), len(values) - 1)
    return [values[i] for i in index]


class Insight:
    def __init__(self, sample_rate: int, matcher_factory: Callable[[], Callable] | None = None,
                 match_every: float = 1.0):
        self._sr = sample_rate
        self._matcher_factory = matcher_factory
        self._every = match_every
        self._lock = threading.Lock()
        size = WINDOW_SECONDS * FPS
        self._series = {name: deque(maxlen=size) for name in ("heard", "sent", "corrected", "gated", "in_db", "out_db")}
        self._chunks: deque = deque()
        self._match: deque = deque(maxlen=MATCH_POINTS)
        self._note: str | None = None
        self._audio: deque = deque()  # recent output blocks, MATCH_SECONDS of them
        self._audio_samples = 0
        self._voiced: deque = deque(maxlen=int(MATCH_SECONDS * FPS))
        self._samples = 0
        self._last_number = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, block: np.ndarray, out: np.ndarray, engine_insight: dict | None, infer_ms: float, chunk_ms: int) -> None:
        """One converted chunk. Called on the live thread, so it only stores."""
        count = max(1, round(len(block) / self._sr * FPS))
        fresh = engine_insight is not None and engine_insight.get("number") != self._last_number
        if fresh:
            self._last_number = engine_insight["number"]
            heard = _resample(engine_insight["heard"], count)
            sent = _resample(engine_insight["sent"], count)
            corrected = _resample(engine_insight["corrected"], count)
            gated = _resample(engine_insight["gated"], count)
        else:  # the quiet gate kept the engine from running: nothing was converted
            heard, sent, corrected, gated = [0.0] * count, [0.0] * count, [False] * count, [True] * count
        in_db, out_db = _levels(block, count), _levels(out, count)
        with self._lock:
            self._samples += len(block)
            now = self._samples / self._sr
            for name, values in (("heard", heard), ("sent", sent), ("corrected", corrected), ("gated", gated),
                                 ("in_db", in_db), ("out_db", out_db)):
                self._series[name].extend(values)
            self._chunks.append({"t": round(now, 3), "infer_ms": round(float(infer_ms), 1), "budget_ms": chunk_ms})
            while self._chunks and self._chunks[0]["t"] <= now - WINDOW_SECONDS:
                self._chunks.popleft()
            self._audio.append(np.asarray(out, dtype=np.float32).copy())
            self._audio_samples += len(out)
            while self._audio and self._audio_samples - len(self._audio[0]) >= MATCH_SECONDS * self._sr:
                self._audio_samples -= len(self._audio.popleft())
            self._voiced.extend(s > 0 and not g for s, g in zip(sent, gated))

    def snapshot(self) -> dict:
        with self._lock:
            return {"frames_per_s": FPS, "seconds": round(self._samples / self._sr, 3),
                    **{name: list(values) for name, values in self._series.items()},
                    "chunks": list(self._chunks), "match": list(self._match), "match_note": self._note}

    # ---- voice match, on its own thread

    def start(self) -> None:
        if self._matcher_factory is None or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(5)

    def _run(self) -> None:
        try:
            matcher = self._matcher_factory()
        except Exception as exc:  # noqa: BLE001 - shown on the Live tab instead of the graph
            with self._lock:
                self._note = str(exc) or "Voice match is not available."
            return
        while not self._stop.wait(self._every):
            with self._lock:
                voiced = sum(self._voiced) / max(1, self._voiced.maxlen)
                audio = np.concatenate(list(self._audio)) if self._audio else np.zeros(0, np.float32)
                now = self._samples / self._sr
            if voiced < MATCH_MIN_VOICED or len(audio) < self._sr:
                continue
            try:
                value = float(matcher(audio[-int(MATCH_SECONDS * self._sr):], self._sr))
            except Exception:  # noqa: BLE001 - one failed score leaves a gap in the graph, nothing more
                continue
            with self._lock:
                self._match.append({"t": round(now, 3), "value": round(value, 3)})
