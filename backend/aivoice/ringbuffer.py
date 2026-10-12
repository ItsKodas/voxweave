import threading

import numpy as np


class RingBuffer:
    """Thread-safe mono float32 FIFO. Reads never block: a short read is padded with silence."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._read = 0
        self._size = 0
        self._primed = False
        self._lock = threading.Lock()
        self.underruns = 0
        self.overflows = 0

    @property
    def available(self) -> int:
        with self._lock:
            return self._size

    def write(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32).ravel()
        with self._lock:
            overflowed = False
            if len(data) > self._cap:
                data = data[-self._cap :]
                overflowed = True
            excess = self._size + len(data) - self._cap
            if excess > 0:
                self._read = (self._read + excess) % self._cap
                self._size -= excess
                overflowed = True
            if overflowed:
                self.overflows += 1
            start = (self._read + self._size) % self._cap
            first = min(len(data), self._cap - start)
            self._buf[start : start + first] = data[:first]
            self._buf[: len(data) - first] = data[first:]
            self._size += len(data)
            self._primed = True

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            take = min(n, self._size)
            first = min(take, self._cap - self._read)
            out[:first] = self._buf[self._read : self._read + first]
            out[first:take] = self._buf[: take - first]
            self._read = (self._read + take) % self._cap
            self._size -= take
            if take < n and self._primed:
                self.underruns += 1
        return out
