import threading
import time
from collections import deque

import numpy as np

from aivoice.audiofile import write_wav

SAMPLE_RATE = 48000
POLL_S = 0.1
STALL_S = 5.0
DISCONNECTED = "An audio device was disconnected"
STALLED = "An audio device stopped responding"


def _pause(pause_ms: int, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(sample_rate * pause_ms / 1000)), dtype=np.float32)


def _mono(audio) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Audio must be a mono (one dimensional) array")
    # NaN and infinities become silence (never full-scale garbage); everything else is clipped.
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def save_wav(path, pieces, pause_ms: int = 300) -> None:
    """Write the pieces, separated by pause_ms of silence, as one mono 48 kHz WAV file."""
    parts: list[np.ndarray] = []
    for piece in pieces:
        arr = _mono(piece)
        if not len(arr):
            continue
        if parts:
            parts.append(_pause(pause_ms, SAMPLE_RATE))
        parts.append(arr)
    if not parts:
        raise ValueError("There is no audio to save")
    write_wav(path, np.concatenate(parts), SAMPLE_RATE)


class _Cursor:
    """One output's read position: an absolute segment number and an offset inside it."""

    def __init__(self, gain: float, now: float):
        self.last_pull = now
        self.starved = False  # the last callback found the queue empty (a gap already happened)
        self.seg = 0
        self.off = 0
        self.played = 0
        self.gain = gain


class _Session:
    """One playback run. Pieces are queued once; every output reads them at its own pace.

    The fill callbacks run on audio threads: they hold the lock only for slicing and bookkeeping
    and never close streams. Streams are closed by poll(), called from a watcher thread, wait()
    or status(), so a session nobody polls still closes within POLL_S of ending.
    """

    def __init__(self, sample_rate: int, gains: dict[str, float], clock=time.monotonic):
        self.sample_rate = sample_rate
        self._clock = clock
        self.error: str | None = None
        self.done = threading.Event()
        self.streams: list = []
        self._outs = {name: _Cursor(gain, clock()) for name, gain in gains.items()}
        self._segs: deque[np.ndarray] = deque()
        self._base = 0  # absolute number of the first queued segment
        self._total = 0  # samples ever queued
        self._pieces = 0
        self._finished = False
        self._ended = False
        self._stopped = False
        self._closing = False
        self._lock = threading.Lock()
        self._streams_lock = threading.Lock()
        self._wake = threading.Event()

    # -- queue (non-audio threads)

    def enqueue(self, audio, pause_ms: int) -> None:
        arr = _mono(audio)
        with self._lock:
            if self._stopped or self._finished or not len(arr):
                return
            # If every output already ran dry, the gap before this piece was the pause.
            if self._pieces and not all(o.starved for o in self._outs.values()):
                pause = _pause(pause_ms, self.sample_rate)
                if len(pause):
                    self._segs.append(pause)
                    self._total += len(pause)
            self._segs.append(arr)
            self._total += len(arr)
            self._pieces += 1

    def finish(self) -> None:
        with self._lock:
            self._finished = True
            self._check_end()

    def _check_end(self) -> None:
        # Caller holds the lock.
        if self._finished and not self._ended and all(
            o.seg - self._base >= len(self._segs) for o in self._outs.values()
        ):
            self._ended = True
            self._wake.set()

    # -- audio thread

    def fill(self, name: str, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        try:
            if self._stopped:
                return out
            with self._lock:
                if self._stopped:
                    return out
                o = self._outs[name]
                pos = 0
                while pos < n:
                    idx = o.seg - self._base
                    if idx >= len(self._segs):
                        break
                    seg = self._segs[idx]
                    take = min(n - pos, len(seg) - o.off)
                    out[pos:pos + take] = seg[o.off:o.off + take]
                    pos += take
                    o.off += take
                    if o.off >= len(seg):
                        o.seg += 1
                        o.off = 0
                o.played += pos
                o.starved = pos < n
                o.last_pull = self._clock()
                lowest = min(c.seg for c in self._outs.values())
                while self._base < lowest:  # release audio every output has consumed
                    self._segs.popleft()
                    self._base += 1
                self._check_end()
                gain = o.gain
            if gain != 1.0:
                out *= gain
            return out
        except Exception:  # noqa: BLE001 - never raise into the audio thread
            return np.zeros(n, dtype=np.float32)

    # -- lifecycle (non-audio threads)

    def _close(self) -> None:
        """Close the streams once; done is set by the thread that closed them, after they are closed."""
        with self._streams_lock:
            if self._closing:
                return
            self._closing = True
            streams = list(self.streams)
        try:
            for stream in streams:
                try:
                    stream.stop()
                except Exception:  # noqa: BLE001 - closing a dead device must not raise
                    pass
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            with self._streams_lock:
                self.streams = []
            self.done.set()
            self._wake.set()

    def _problem(self) -> str | None:
        try:
            if not all(s.active for s in list(self.streams)):
                return DISCONNECTED
        except Exception:  # noqa: BLE001 - sounddevice raises on an invalidated (unplugged) stream
            return DISCONNECTED
        now = self._clock()
        if any(now - o.last_pull > STALL_S for o in self._outs.values()):
            return STALLED
        return None

    def poll(self) -> None:
        """Never raises: any failure ends playback with an error and closes the streams."""
        if self.done.is_set():
            return
        try:
            if not (self._stopped or self._ended):
                problem = self._problem()
                if problem is not None:
                    with self._lock:
                        # Re-check: a normal stop or finish also makes devices go inactive.
                        if not (self._stopped or self._ended):
                            if self.error is None:
                                self.error = problem
                            self._stopped = True
                            self._segs.clear()
            if self._stopped or self._ended:
                self._close()
        except Exception as exc:  # noqa: BLE001
            if self.error is None:
                self.error = str(exc) or type(exc).__name__
            self._stopped = True
            self._close()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._segs.clear()
        self._close()

    def watch(self) -> None:
        while not self.done.is_set():
            try:
                self._wake.wait(POLL_S)
                self._wake.clear()
                self.poll()
            except Exception:  # noqa: BLE001 - the watcher must never exit without closing
                self._stopped = True
                self._close()

    def status(self) -> dict:
        self.poll()
        with self._lock:
            played = min((o.played for o in self._outs.values()), default=0)
            queued = 0 if self._stopped else self._total - played
        return {
            "playing": not self.done.is_set(),
            "queued_seconds": queued / self.sample_rate,
            "played_seconds": played / self.sample_rate,
            "error": self.error,
        }


class Player:
    """Queues finished pieces and plays them on one or two output devices as audio arrives.

    enqueue() goes to the current session, so a caller must stop its producer before calling
    start() again. start() returns a session token; passing it to enqueue(session=token) makes
    audio that arrives late for a replaced session be dropped instead of joining the new one.
    """

    def __init__(self, stream_factory, sample_rate: int = SAMPLE_RATE, clock=time.monotonic):
        self._factory = stream_factory
        self._clock = clock
        self.sample_rate = sample_rate
        self._session: _Session | None = None
        self._lifecycle = threading.RLock()

    def start(self, devices: dict[str, int], gains: dict[str, float] | None = None):
        if not devices:
            raise ValueError("Choose at least one output device")
        gains = gains or {}
        with self._lifecycle:
            old, self._session = self._session, None
            if old is not None:
                old.stop()
            session = _Session(self.sample_rate, {name: float(gains.get(name, 1.0)) for name in devices}, self._clock)
            try:
                for name, device in devices.items():
                    session.streams.append(
                        self._factory.output(device, self.sample_rate, lambda n, name=name: session.fill(name, n))
                    )
                for stream in list(session.streams):
                    stream.start()
            except Exception as exc:
                session.stop()
                raise ValueError(f"Could not open the audio output: {exc}") from exc
            self._session = session
            threading.Thread(target=session.watch, daemon=True).start()
            return session

    def enqueue(self, audio: np.ndarray, pause_ms: int = 300, session=None) -> None:
        current = self._session
        if session is not None and session is not current:
            return
        session = current
        if session is not None:
            session.enqueue(audio, pause_ms)

    def finish(self) -> None:
        session = self._session
        if session is not None:
            session.finish()

    def stop(self) -> None:
        with self._lifecycle:
            if self._session is not None:
                self._session.stop()

    def wait(self, timeout: float | None = None) -> bool:
        session = self._session
        if session is None:
            return True
        session.poll()
        return session.done.wait(timeout)

    def status(self) -> dict:
        session = self._session
        if session is None:
            return {"playing": False, "queued_seconds": 0.0, "played_seconds": 0.0, "error": None}
        return session.status()
