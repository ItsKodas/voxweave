from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from aivoice.engines.f0 import extract_f0, shift_f0, to_coarse
from aivoice.engines.pitch import blend_join, fix_octave_errors, smooth_pitch, fill_gaps, fit_to_range, repair_creak, voice_like_frames
from aivoice.presets import LiveParams


def make_session(path: str, device_id: int = 0):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False  # required by the DirectML provider
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(path), options,
        providers=[("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
    )
    if "DmlExecutionProvider" not in session.get_providers():
        raise RuntimeError("DirectML is not available; refusing to run voice conversion on the CPU")
    return session


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


def protect_consonants(phone: np.ndarray, phone0: np.ndarray, pitchf: np.ndarray, protect: float) -> np.ndarray:
    """RVC's "protect": where there is no pitch (consonants, breath), keep part of the features as they were before
    the index pulled them towards the training voice, so s, t, f and sh stay clear instead of turning into noise.
    At 0.5 or more nothing is protected."""
    if protect >= 0.5:
        return phone
    mask = np.where(pitchf > 0, 1.0, protect).astype(np.float32)[None, :, None]
    return (phone * mask + phone0 * (1.0 - mask)).astype(np.float32)


SOLA_SEARCH_MS = 12  # the processor may move a join this far (aivoice.processor.SOLA_SEARCH_S)
ENVELOPE_FRAME_S = 0.04
ENVELOPE_HOP_S = 0.01
MAX_ENVELOPE_GAIN = 4.0  # before the follow strength: a near-silent output is never lifted into loud noise


def _envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS level around each sample (40 ms frames every 10 ms, interpolated)."""
    frame, hop = int(sample_rate * ENVELOPE_FRAME_S), int(sample_rate * ENVELOPE_HOP_S)
    padded = np.pad(audio.astype(np.float32), (frame // 2, frame // 2))
    squares = np.concatenate([[0.0], np.cumsum(padded.astype(np.float64) ** 2)])
    starts = np.arange(0, len(padded) - frame + 1, hop)
    rms = np.sqrt(np.maximum(squares[starts + frame] - squares[starts], 0.0) / frame)
    return np.interp(np.arange(len(audio)), starts, rms)


def follow_envelope(out: np.ndarray, audio: np.ndarray, sample_rate: int, follow: float) -> np.ndarray:
    """The output with its loudness moved toward the input's: `follow` 0 leaves it, 1 gives it the input's envelope.
    The change is limited to MAX_ENVELOPE_GAIN either way."""
    if follow <= 0.0:
        return out
    ratio = (_envelope(audio, sample_rate) + 1e-5) / (_envelope(out, sample_rate) + 1e-5)
    gain = np.clip(ratio, 1.0 / MAX_ENVELOPE_GAIN, MAX_ENVELOPE_GAIN) ** follow
    return (out * gain).astype(np.float32)


class RvcEngine:
    def __init__(self, model_path, vec_path, model_sample_rate: int, index_path=None,
                 session_factory: Callable[[str], object] | None = None, rmvpe=None):
        factory = session_factory or make_session
        self._rmvpe = rmvpe
        # The model's noise input, made once per length and reused: chunk after chunk the same input then gives
        # the same voice texture, instead of a slightly different one every few hundred milliseconds.
        self._noise: dict[int, np.ndarray] = {}
        # Set by live mode before each conversion: where the window starts, in samples since live mode started.
        # With it the noise comes from one long sequence at that place in time; without it (text to speech) the
        # same noise is used for every window of a length.
        self.noise_position: int | None = None
        # The training voice's usual pitch range (low, high) in Hz, when known: pitches outside it are moved inside.
        self.pitch_range: tuple[float, float] | None = None
        # The pitch the last live window used and where it started (in frames since live mode started), so the next
        # window's join starts from the same pitch.
        self._last_pitch: tuple[float, np.ndarray] | None = None
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

    _LONG_NOISE = np.random.default_rng(1234).standard_normal((1, 192, 6000)).astype(np.float32)  # 60 s, repeating

    def _noise_at(self, frames: int, sample_rate: int) -> np.ndarray:
        if self.noise_position is None:
            return self._noise_for(frames)
        start = (int(self.noise_position) * 100 // sample_rate) % self._LONG_NOISE.shape[2]  # 100 frames a second
        index = (start + np.arange(frames)) % self._LONG_NOISE.shape[2]
        return self._LONG_NOISE[:, :, index]

    def _noise_for(self, frames: int) -> np.ndarray:
        if frames not in self._noise:
            if len(self._noise) > 8:
                self._noise.clear()
            self._noise[frames] = np.random.default_rng(1234).standard_normal((1, 192, frames)).astype(np.float32)
        return self._noise[frames]

    def _join_previous(self, pitchf: np.ndarray, n: int, sample_rate: int, params: LiveParams) -> np.ndarray:
        """In live mode, the pitch where this window's played audio begins starts from what the last window used
        there (see blend_join). Text to speech, with no position in time, is left as it is."""
        if self.noise_position is None:
            self._last_pitch = None
            return pitchf
        rate = len(pitchf) * sample_rate / n  # frames a second
        position = self.noise_position * rate / sample_rate
        if self._last_pitch is not None:
            last_position, last = self._last_pitch
            join = round((params.crossfade_ms + SOLA_SEARCH_MS) * rate / 1000)
            start = len(pitchf) - round(params.chunk_ms * rate / 1000) - join
            pitchf = blend_join(pitchf, last, round(position - last_position), start, join)
        self._last_pitch = (position, pitchf.copy())
        return pitchf

    def convert(self, audio: np.ndarray, sample_rate: int, pitch_shift: float, params: LiveParams) -> np.ndarray:
        n = len(audio)
        wav16 = soxr.resample(audio.astype(np.float32), sample_rate, 16000).astype(np.float32)
        feats = self._vec.run(None, {self._vec_input: wav16[None, None, :]})[0][0].astype(np.float32)
        phone0 = np.repeat(feats, 2, axis=0)[None]
        if self._index is not None:
            feats = blend_index(feats, self._index, self._big_npy, params.index_rate)
        phone = np.repeat(feats, 2, axis=0)[None].astype(np.float32)
        frames = phone.shape[1]
        pitchf = fill_gaps(extract_f0(wav16, params.f0_method, frames, rmvpe=self._rmvpe))
        pitchf = repair_creak(pitchf, voice_like_frames(wav16, frames))
        pitchf = smooth_pitch(fix_octave_errors(pitchf))
        pitchf = shift_f0(pitchf, pitch_shift)
        if self.pitch_range is not None:
            pitchf = fit_to_range(pitchf, *self.pitch_range)
        pitchf = self._join_previous(pitchf, n, sample_rate, params)
        if self._index is not None:
            phone = protect_consonants(phone, phone0, pitchf, params.protect)
        feed = {
            "phone": phone,
            "phone_lengths": np.array([frames], dtype=np.int64),
            "pitch": to_coarse(pitchf)[None],
            "pitchf": pitchf[None],
            "ds": np.array([0], dtype=np.int64),
            "rnd": self._noise_at(frames, sample_rate),
        }
        out = np.asarray(self._net.run(None, feed)[0], dtype=np.float32).reshape(-1)
        if self._model_sr != sample_rate:
            out = soxr.resample(out, self._model_sr, sample_rate).astype(np.float32)
        return follow_envelope(_fit(out, n), audio, sample_rate, params.volume_follow)
