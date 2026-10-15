"""How alike two voices sound: a speaker-recognition model (WeSpeaker ResNet34, trained on the 5,994 speakers of
VoxCeleb2) turns speech into a 256-number "fingerprint", and the cosine similarity of two fingerprints says how
much the speakers sound alike. Deep calibration uses it to score converted speech against a voice's recordings.

The model is downloaded only when the person asks for it (it is not part of the first-run setup): a pinned
revision, checked for its size and SHA-256 before it is used.

Its input is Kaldi-style 80-band log mel filterbanks of 16 kHz audio at 16-bit scale, with the mean of each band
removed, computed here with numpy exactly as torchaudio.compliance.kaldi.fbank does for WeSpeaker (Hamming window,
no dither); tests/test_speaker.py checks it against values torchaudio produced.
"""
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr

MODEL_URL = ("https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/"
             "f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx")
MODEL_SIZE = 26530309
MODEL_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
MODEL_LICENCE = "WeSpeaker ResNet34 speaker model (VoxCeleb2), CC BY 4.0, https://github.com/wenet-e2e/wespeaker"
MIN_SECONDS = 1.0

_FRAME, _SHIFT, _FFT, _BANDS = 400, 160, 512, 80


def model_path(data: Path) -> Path:
    return data / "pretrained" / "speaker" / "voxceleb_resnet34_LM.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_ready(data: Path) -> bool:
    path = model_path(data)
    return path.is_file() and path.stat().st_size == MODEL_SIZE


def download_model(data: Path, opener=urllib.request.urlopen) -> Path:
    """Fetches the model to a .part file and keeps it only if its size and SHA-256 are the pinned ones."""
    path = model_path(data)
    if path.is_file() and path.stat().st_size == MODEL_SIZE and _sha256(path) == MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".onnx.part")
    try:
        with opener(MODEL_URL, timeout=60) as response, open(part, "wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                if out.tell() > MODEL_SIZE:
                    raise OSError("The speaker model download is larger than expected")
        if part.stat().st_size != MODEL_SIZE or _sha256(part) != MODEL_SHA256:
            raise OSError("The speaker model download did not match its checksum and was not kept")
        part.replace(path)
        return path
    finally:
        part.unlink(missing_ok=True)


def _mel_banks() -> np.ndarray:
    """Kaldi's triangular mel filters, 20 Hz to 8 kHz, as (bands, fft bins): the last (Nyquist) bin weighs 0."""
    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    low, high = mel(20.0), mel(8000.0)
    step = (high - low) / (_BANDS + 1)
    bins = mel(np.arange(_FFT // 2) * 16000.0 / _FFT)
    banks = np.zeros((_BANDS, _FFT // 2 + 1))
    for b in range(_BANDS):
        left, centre, right = low + b * step, low + (b + 1) * step, low + (b + 2) * step
        up = (bins - left) / (centre - left)
        down = (right - bins) / (right - centre)
        banks[b, : _FFT // 2] = np.maximum(0.0, np.minimum(up, down))
    return banks


_BANKS = _mel_banks()
_WINDOW = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(_FRAME) / (_FRAME - 1))


def fbank(audio16k: np.ndarray) -> np.ndarray:
    """80-band log mel filterbanks (frames of 25 ms every 10 ms), one row per frame, before mean removal."""
    x = np.asarray(audio16k, dtype=np.float64) * 32768.0
    count = 1 + (len(x) - _FRAME) // _SHIFT
    if count < 1:
        return np.zeros((0, _BANDS), dtype=np.float32)
    frames = x[np.arange(_FRAME)[None, :] + _SHIFT * np.arange(count)[:, None]]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = np.concatenate([frames[:, :1] * (1 - 0.97), frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=_FFT, axis=1)) ** 2
    return np.log(np.maximum(power @ _BANKS.T, np.finfo(np.float32).eps)).astype(np.float32)


@dataclass
class SpeakerModel:
    """Fingerprints of speech, from the ONNX model in a session made by the caller."""
    session: object

    def embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio16k = soxr.resample(np.asarray(audio, dtype=np.float32), sample_rate, 16000) if sample_rate != 16000 \
            else np.asarray(audio, dtype=np.float32)
        if len(audio16k) < MIN_SECONDS * 16000:
            raise ValueError("Too little audio to recognise a speaker from")
        feats = fbank(audio16k)
        feats = feats - feats.mean(axis=0, keepdims=True)
        name = self.session.get_inputs()[0].name
        emb = np.asarray(self.session.run(None, {name: feats[None].astype(np.float32)})[0][0], dtype=np.float64)
        return emb / (np.linalg.norm(emb) + 1e-12)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two fingerprints (1 is the same voice; different people are usually below 0.3)."""
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))




def voice_fingerprint(model: SpeakerModel, clip_paths: list) -> np.ndarray | None:
    """The average fingerprint of a voice's recordings (unit length), or None when none was long enough."""
    from aivoice.audiofile import read_wav

    prints = []
    for path in clip_paths:
        try:
            audio, sr = read_wav(path)
            prints.append(model.embedding(audio, sr))
        except (ValueError, OSError):
            continue  # too short, or unreadable
    if not prints:
        return None
    mean = np.mean(prints, axis=0)
    return mean / (np.linalg.norm(mean) + 1e-12)


def speech_clip_paths(library, voice_id: str, limit: int = 30) -> list:
    """Up to `limit` of a voice's speech clips (not laughs or other sounds)."""
    try:
        voice = library.get(voice_id)
    except ValueError:
        return []
    speech = [c for c in voice.clips if getattr(c, "kind", "speech") == "speech"]
    return [library.clip_audio_path(voice_id, c.index) for c in speech[:limit]]


def voice_matcher(data: Path, library, voice_id: str, session_factory):
    """A function scoring how much audio sounds like the voice (0 to 1), for live mode's voice match graph.
    Raises LookupError with the reason when it cannot be made."""
    if not model_ready(data):
        raise LookupError("Voice match needs the speaker model: press Find best settings once to download it.")
    model = SpeakerModel(session_factory(model_path(data)))
    target = voice_fingerprint(model, speech_clip_paths(library, voice_id))
    if target is None:
        raise LookupError("This voice has no recordings in the library to compare with, so there is no voice match.")
    return lambda audio, sample_rate: max(0.0, similarity(model.embedding(audio, sample_rate), target))
