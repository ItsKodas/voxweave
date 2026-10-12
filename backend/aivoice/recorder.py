import threading

import numpy as np

from aivoice.config import SAMPLE_RATE


class RecorderError(ValueError):
    pass


class Recorder:
    """Captures the microphone into memory. One take at a time."""

    def __init__(self, stream_factory, sample_rate: int = SAMPLE_RATE, max_seconds: float = 30.0):
        self._factory = stream_factory
        self._sr = sample_rate
        self._max = int(max_seconds * sample_rate)
        self._lock = threading.Lock()        # start / stop / cancel
        self._data_lock = threading.Lock()   # blocks shared with the audio callback
        self._stream = None
        self._blocks: list[np.ndarray] = []
        self._samples = 0
        self._level = 0.0

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def _on_audio(self, indata) -> None:
        try:
            block = np.asarray(indata, dtype=np.float32)
            if block.ndim == 2:
                mono = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1)
            elif block.ndim == 1:
                mono = block
            else:
                return
            mono = np.nan_to_num(mono)
            with self._data_lock:
                if self._stream is None:
                    return
                self._level = float(np.max(np.abs(mono))) if len(mono) else 0.0
                if self._samples < self._max:
                    self._blocks.append(mono.astype(np.float32, copy=True))
                    self._samples += len(mono)
        except Exception:  # noqa: BLE001 - never raise into the audio driver thread
            pass

    def _reset(self) -> None:
        with self._data_lock:
            self._blocks = []
            self._samples = 0
            self._level = 0.0

    def _close(self) -> None:
        stream = self._stream
        with self._data_lock:
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - an unplugged device must not block stopping
                pass

    def start(self, device: int) -> None:
        with self._lock:
            if self._stream is not None:
                raise RecorderError("Already recording")
            self._reset()
            try:
                stream = self._factory.input(device, self._sr, self._on_audio)
                with self._data_lock:
                    self._stream = stream
                stream.start()
            except Exception as exc:  # noqa: BLE001
                self._close()
                raise RecorderError(f"Could not open the microphone: {exc}") from exc

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._stream is None:
                raise RecorderError("Not recording")
            self._close()
            with self._data_lock:
                audio = np.concatenate(self._blocks) if self._blocks else np.zeros(0, dtype=np.float32)
            self._reset()
            return audio[: self._max].astype(np.float32)

    def cancel(self) -> None:
        with self._lock:
            self._close()
            self._reset()

    def status(self) -> dict:
        with self._data_lock:
            return {"recording": self._stream is not None, "level": self._level,
                    "seconds": round(self._samples / self._sr, 2), "full": self._samples >= self._max}
