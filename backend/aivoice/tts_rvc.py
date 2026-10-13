"""Offline pass of a synthesised piece through the voice's RVC engine.

Every engine call gets exactly WINDOW_SECONDS of audio: a DirectML session is fast only for the first input
shape it sees and fails above about 36 s. The engine returns the same length as its input, aligned at the END
(it pads the start with zeros), so the piece is placed at the end of the window and cut from the end.
"""
import numpy as np

from aivoice.presets import LiveParams, resolve

WINDOW_SECONDS = 20
FRAME_SECONDS = 0.02


def offline_params() -> LiveParams:
    """The high_quality preset; its index_rate is kept."""
    return resolve("high_quality")


def _split_point(piece: np.ndarray, sample_rate: int = 48000) -> int:
    """Start of the quietest 20 ms frame whose start lies in the middle third of the piece."""
    frame = int(FRAME_SECONDS * sample_rate)
    lo = len(piece) // 3
    hi = max(lo + 1, 2 * len(piece) // 3 - frame)
    starts = np.arange(lo, hi, frame)
    energy = [float(np.sum(np.square(piece[s:s + frame], dtype=np.float64))) for s in starts]
    return int(starts[int(np.argmin(energy))])


class OfflineConverter:
    def __init__(self, engine, params: LiveParams, sample_rate: int = 48000):
        self._engine = engine
        self._params = params
        self._sr = sample_rate
        self._window = WINDOW_SECONDS * sample_rate

    def warm(self) -> None:
        """One silent window, so the session's first (slow) call is not a real piece."""
        self._call(np.zeros(self._window, np.float32))

    def convert(self, piece: np.ndarray) -> np.ndarray:
        piece = np.asarray(piece, dtype=np.float32).reshape(-1)
        if len(piece) == 0:
            return piece.copy()
        if len(piece) > self._window:
            cut = _split_point(piece, self._sr)
            return np.concatenate([self.convert(piece[:cut]), self.convert(piece[cut:])])
        buffer = np.zeros(self._window, np.float32)
        buffer[self._window - len(piece):] = piece
        return self._call(buffer)[self._window - len(piece):]

    def _call(self, buffer: np.ndarray) -> np.ndarray:
        out = np.asarray(self._engine.convert(buffer, self._sr, 0.0, self._params))
        if out.shape != buffer.shape:
            raise ValueError(f"Voice conversion returned the wrong length ({out.size} samples, expected {buffer.size}).")
        out = out.astype(np.float32, copy=False)
        if not np.isfinite(out).all():
            raise ValueError("Voice conversion returned audio that is not finite.")
        return out
