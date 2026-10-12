from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from aivoice.engines.f0 import extract_f0, shift_f0, to_coarse
from aivoice.presets import LiveParams


def make_session(path: str, device_id: int = 0):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False  # required by the DirectML provider
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), options,
        providers=[("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
    )


def blend_index(feats: np.ndarray, index, big_npy: np.ndarray, rate: float) -> np.ndarray:
    """Pull features toward their nearest neighbours in the trained voice's index."""
    if rate <= 0:
        return feats
    score, ix = index.search(feats, 8)
    weight = np.square(1.0 / np.maximum(score, 1e-9))
    weight /= weight.sum(axis=1, keepdims=True)
    neighbours = np.sum(big_npy[ix] * weight[:, :, None], axis=1)
    return (neighbours * rate + feats * (1.0 - rate)).astype(np.float32)


def _fit(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or pad at the start so the end (the newest audio) stays aligned."""
    if len(audio) >= n:
        return audio[-n:]
    return np.concatenate([np.zeros(n - len(audio), dtype=audio.dtype), audio])


class RvcEngine:
    def __init__(self, model_path, vec_path, model_sample_rate: int, index_path=None,
                 session_factory: Callable[[str], object] | None = None):
        factory = session_factory or make_session
        self._vec = factory(str(vec_path))
        self._net = factory(str(model_path))
        self._vec_input = self._vec.get_inputs()[0].name
        self._model_sr = model_sample_rate
        self._index = None
        self._big_npy = None
        if index_path is not None and Path(index_path).exists():
            import faiss

            self._index = faiss.read_index(str(index_path))
            self._big_npy = self._index.reconstruct_n(0, self._index.ntotal)

    def convert(self, audio: np.ndarray, sample_rate: int, pitch_shift: float, params: LiveParams) -> np.ndarray:
        n = len(audio)
        wav16 = soxr.resample(audio.astype(np.float32), sample_rate, 16000).astype(np.float32)
        feats = self._vec.run(None, {self._vec_input: wav16[None, None, :]})[0][0]
        if self._index is not None:
            feats = blend_index(feats.astype(np.float32), self._index, self._big_npy, params.index_rate)
        phone = np.repeat(feats, 2, axis=0)[None].astype(np.float32)
        frames = phone.shape[1]
        pitchf = shift_f0(extract_f0(wav16, params.f0_method, frames), pitch_shift)
        feed = {
            "phone": phone,
            "phone_lengths": np.array([frames], dtype=np.int64),
            "pitch": to_coarse(pitchf)[None],
            "pitchf": pitchf[None],
            "ds": np.array([0], dtype=np.int64),
            "rnd": np.random.randn(1, 192, frames).astype(np.float32),
        }
        out = np.asarray(self._net.run(None, feed)[0], dtype=np.float32).reshape(-1)
        if self._model_sr != sample_rate:
            out = soxr.resample(out, self._model_sr, sample_rate).astype(np.float32)
        return _fit(out, n)
