"""Learning the speaker's voice while live mode runs ("Keep learning" on the Live tab).

While it is on, the chunks in which the speaker is clearly speaking (a pitch, not background) are kept in memory,
up to two minutes of speech. From them:

- every MEASURE_EVERY seconds the speaker's voice is measured again (usual pitch, pitch range, speaking level, and
  from the level the input boost), as a calibration would measure it but from how they really talk live; the
  input boost and the speaking level the background cut judges against move a quarter of the way to the new
  values at a time;
- every PASS_EVERY seconds, once there are MIN_PASS_SECONDS of speech, a tuning pass runs the deep-calibration
  search on the last PASS_SECONDS of it (on the processor, so it never takes the graphics card from live mode).
  A setting changes only when two passes in a row agree on it, and then by small steps: a semitone of pitch, ten
  points of likeness. Quality is not tuned: it sets the chunk size, which cannot change while live mode runs.

Changes are applied only in a pause in the speech, so nothing jumps mid-word. Nothing is written anywhere until
the speaker saves; turning learning off without saving puts back what live mode started with.
"""
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

POOL_SPEECH_SECONDS = 120.0
MIN_MEASURE_SECONDS = 15.0
MIN_PASS_SECONDS = 45.0
PASS_SECONDS = 30.0
MEASURE_EVERY = 10.0
PASS_EVERY = 180.0
SPEECH_SHARE = 0.4  # of a chunk's frames with a pitch, for the chunk to count as speech
GAIN_STEP = 0.25  # of the way to the newly measured boost, in the log domain, per change
MAX_PITCH_STEP = 1
MAX_LIKENESS_STEP = 0.1


def _sign(x: float) -> int:
    return (x > 1e-9) - (x < -1e-9)


def agreed_changes(current: dict, previous: dict | None, best: dict) -> dict:
    """What two passes in a row agree should change, in steps of at most a semitone and ten points of likeness."""
    if previous is None:
        return {}
    changes: dict = {}
    overrides: dict = {}
    way = _sign(best["pitch_shift"] - current["pitch_shift"])
    if way and way == _sign(previous["pitch_shift"] - current["pitch_shift"]):
        changes["pitch_shift"] = current["pitch_shift"] + way * MAX_PITCH_STEP
    gap = best["likeness"] - current["likeness"]
    if _sign(gap) and _sign(gap) == _sign(previous["likeness"] - current["likeness"]):
        overrides["index_rate"] = round(current["likeness"] + max(-MAX_LIKENESS_STEP, min(MAX_LIKENESS_STEP, gap)), 2)
    for name, value in best["advanced"].items():
        if value != current["advanced"].get(name) and previous["advanced"].get(name) == value:
            overrides[name] = value
    if overrides:
        changes["overrides"] = overrides
    return changes


class _Stopped(Exception):
    pass


class Learner:
    """`start`: pitch_shift, likeness, advanced (dict), input_gain, speech_level_db as live mode started.
    `measure_voice(audio)` returns the voice's measurements (pitch_hz, low_hz, high_hz, level_db, input_gain) or
    None; `tune(audio, settings, progress)` runs a tuning pass and returns the best pitch_shift, likeness and
    advanced, or is None for measuring only; `apply(changes)` hands changes to live mode."""

    def __init__(self, sample_rate: int, start: dict, measure_voice: Callable, tune: Callable | None,
                 apply: Callable[[dict], None], behind: Callable[[], bool] = lambda: False,
                 clock: Callable[[], float] = time.monotonic, measure_every: float = MEASURE_EVERY,
                 pass_every: float = PASS_EVERY):
        self._sr = sample_rate
        self._start = {**start, "advanced": dict(start["advanced"])}
        self._current = {**start, "advanced": dict(start["advanced"])}
        self._measure_voice = measure_voice
        self._tune = tune
        self._apply = apply
        self._behind = behind
        self._clock = clock
        self._measure_every = measure_every
        self._pass_every = pass_every
        self._lock = threading.Lock()
        self._pool: deque = deque()  # (audio, speech seconds)
        self._speech = 0.0
        self._paused = False
        self._last_number = None
        self._last_measure = -1e18
        self._last_pass = -1e18
        self._previous: dict | None = None
        self._pending: dict = {}
        self._learned: dict | None = None
        self._passes = 0
        self._message = "Listening to you speak"
        self._last_change: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- fed by the live thread

    def add(self, block: np.ndarray, engine_insight: dict | None) -> None:
        if engine_insight is None or engine_insight.get("number") == self._last_number:
            share = 0.0  # the quiet gate kept the engine from running: silence
        else:
            self._last_number = engine_insight["number"]
            sent, gated = engine_insight["sent"], engine_insight["gated"]
            share = sum(1 for s, g in zip(sent, gated) if s > 0 and not g) / max(1, len(sent))
        with self._lock:
            self._paused = share == 0.0
            if share >= SPEECH_SHARE:
                seconds = share * len(block) / self._sr
                self._pool.append((np.asarray(block, dtype=np.float32).copy(), seconds))
                self._speech += seconds
                while self._speech > POOL_SPEECH_SECONDS + 1e-9 and self._pool:
                    self._speech -= self._pool.popleft()[1]

    # ---- the background work, one step at a time

    def _audio(self, seconds: float | None = None) -> np.ndarray:
        with self._lock:
            blocks = [b for b, _ in self._pool]
        audio = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        return audio if seconds is None else audio[-int(seconds * self._sr):]

    def tick(self) -> None:
        now = self._clock()
        with self._lock:
            speech = self._speech
        if now - self._last_measure >= self._measure_every and speech >= MIN_MEASURE_SECONDS:
            self._last_measure = now
            learned = self._measure_voice(self._audio())
            if learned is not None:
                self._learned = learned
                current = self._current["input_gain"]
                gain = float(current * (learned["input_gain"] / current) ** GAIN_STEP)
                self._pending.update({"input_gain": round(gain, 3),
                                      "speech_level_db": round(learned["level_db"] + 20.0 * np.log10(gain), 2)})
        if self._tune is not None and now - self._last_pass >= self._pass_every and speech >= MIN_PASS_SECONDS:
            if self._behind():
                self._message = "Waiting: live mode is falling behind, so learning holds off"
            else:
                self._run_pass(now)
        with self._lock:
            paused = self._paused
        if self._pending and paused:
            self._apply_pending()

    def _run_pass(self, now: float) -> None:
        self._message = f"Tuning on your last {int(PASS_SECONDS)} seconds of speech"
        settings = {k: self._current[k] for k in ("pitch_shift", "likeness")}
        settings["advanced"] = dict(self._current["advanced"])

        def progress(step, message):
            if self._stop.is_set():
                raise _Stopped()
            while self._behind() and not self._stop.is_set():
                time.sleep(0.5)
            self._message = f"Tuning on your recent speech: {message}"

        try:
            best = self._tune(self._audio(PASS_SECONDS), settings, progress)
        except _Stopped:
            return
        except Exception as exc:  # noqa: BLE001 - one failed pass is reported, the next one tries again
            self._message = f"A tuning pass failed: {exc}"
            self._last_pass = now
            return
        self._last_pass = now
        self._passes += 1
        changes = agreed_changes(self._current, self._previous, best)
        self._previous = best
        if changes:
            overrides = {**self._pending.get("overrides", {}), **changes.pop("overrides", {})}
            self._pending.update(changes)
            if overrides:
                self._pending["overrides"] = overrides
        self._message = f"{self._passes} tuning pass{'es' if self._passes != 1 else ''} done"

    def _apply_pending(self) -> None:
        changes, self._pending = self._pending, {}
        self._apply(changes)
        described = []
        if "pitch_shift" in changes:
            described.append(f"pitch {self._current['pitch_shift']:+d} to {changes['pitch_shift']:+d}")
            self._current["pitch_shift"] = changes["pitch_shift"]
        for name, value in changes.get("overrides", {}).items():
            if name == "index_rate":
                described.append(f"likeness {round(self._current['likeness'] * 100)}% to {round(value * 100)}%")
                self._current["likeness"] = value
            else:
                described.append(f"{name.replace('_', ' ')} to {value}")
                self._current["advanced"][name] = value
        for name in ("input_gain", "speech_level_db"):
            if name in changes:
                self._current[name] = changes[name]
        if described:
            self._last_change = ", ".join(described)

    # ---- for live mode and the Live tab

    def revert(self) -> None:
        start = self._start
        self._apply({"pitch_shift": start["pitch_shift"],
                     "overrides": {"index_rate": start["likeness"], **start["advanced"]},
                     "input_gain": start["input_gain"], "speech_level_db": start["speech_level_db"]})

    def saved(self) -> dict:
        """What saving keeps: the voice as measured, the settings live mode has now, and recent speech."""
        return {"profile": self._learned, "sample": self._audio(PASS_SECONDS),
                "settings": {"pitch_shift": self._current["pitch_shift"], "likeness": self._current["likeness"],
                             "advanced": dict(self._current["advanced"])}}

    def status(self) -> dict:
        with self._lock:
            speech = self._speech
        return {"enabled": True, "speech_seconds": round(speech, 1), "passes": self._passes,
                "message": self._message, "last_change": self._last_change, "learned": self._learned,
                "current": {"pitch_shift": self._current["pitch_shift"], "likeness": self._current["likeness"],
                            "advanced": dict(self._current["advanced"])}}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(10)

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - learning must never take live mode down
                self._message = f"Learning hit a problem and will try again: {exc}"
