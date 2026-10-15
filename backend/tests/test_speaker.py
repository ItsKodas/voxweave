import io
from pathlib import Path

import numpy as np
import pytest

from aivoice import speaker
from aivoice.speaker import SpeakerModel, download_model, fbank, model_path, model_ready, similarity

REFERENCE = Path(__file__).parent / "fixtures" / "kaldi_fbank_reference.npz"


def test_filterbanks_match_torchaudios_kaldi_fbank():
    # Made with torchaudio.compliance.kaldi.fbank(x * 32768, num_mel_bins=80, frame_length=25, frame_shift=10,
    # sample_frequency=16000, window_type="hamming", dither=0.0), as WeSpeaker computes them for this model.
    ref = np.load(REFERENCE)
    ours = fbank(ref["audio"])
    assert ours.shape == ref["fbank"].shape
    assert np.abs(ours - ref["fbank"]).max() < 1e-3


def test_too_little_audio_gives_no_frames():
    assert fbank(np.zeros(100, np.float32)).shape == (0, 80)


class FakeSession:
    def __init__(self):
        self.shapes = []

    def get_inputs(self):
        return [type("I", (), {"name": "feats"})()]

    def run(self, _, feed):
        feats = feed["feats"]
        self.shapes.append(feats.shape)
        assert abs(float(feats[0].mean(axis=0).mean())) < 1e-4  # each band's mean removed
        return [np.full((1, 256), 3.0, np.float32)]


def test_embeddings_are_unit_length_from_16k_mean_removed_filterbanks():
    session = FakeSession()
    model = SpeakerModel(session)
    t = np.arange(48000 * 2) / 48000
    emb = model.embedding((0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 48000)
    assert np.linalg.norm(emb) == pytest.approx(1.0)
    assert session.shapes[0][0] == 1 and session.shapes[0][2] == 80 and 190 <= session.shapes[0][1] <= 200  # 2 s
    with pytest.raises(ValueError, match="Too little audio"):
        model.embedding(np.zeros(4000, np.float32), 48000)


def test_similarity_of_fingerprints():
    assert similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_download_is_kept_only_when_its_size_and_checksum_match(tmp_path, monkeypatch):
    payload = b"model bytes"
    import hashlib

    monkeypatch.setattr(speaker, "MODEL_SIZE", len(payload))
    monkeypatch.setattr(speaker, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    opened = []

    def opener(url, timeout):
        opened.append(url)
        return FakeResponse(payload)

    assert not model_ready(tmp_path)
    path = download_model(tmp_path, opener=opener)
    assert path == model_path(tmp_path) and path.read_bytes() == payload and model_ready(tmp_path)
    assert opened == [speaker.MODEL_URL] and "f0c48c298fd835726c27956a5d617bad7115627e" in opened[0]  # pinned revision
    download_model(tmp_path, opener=opener)
    assert len(opened) == 1  # already there and intact: not fetched again


def test_a_tampered_download_is_thrown_away(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker, "MODEL_SIZE", 11)
    monkeypatch.setattr(speaker, "MODEL_SHA256", "0" * 64)
    with pytest.raises(OSError, match="checksum"):
        download_model(tmp_path, opener=lambda url, timeout: FakeResponse(b"model bytes"))
    assert not model_path(tmp_path).exists() and not list(model_path(tmp_path).parent.glob("*.part"))


def test_a_download_larger_than_expected_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker, "MODEL_SIZE", 4)
    with pytest.raises(OSError, match="larger"):
        download_model(tmp_path, opener=lambda url, timeout: FakeResponse(b"x" * (2 << 20)))
    assert not model_path(tmp_path).exists()
