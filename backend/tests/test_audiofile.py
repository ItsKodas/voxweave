import wave

import numpy as np
import pytest

from aivoice.audiofile import AudioFileError, read_audio, read_wav, write_wav


def tone(seconds, sr, freq=220.0, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_stereo_wav(path, left, right, sr):
    data = np.stack([left, right], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767).astype("<i2").tobytes())


def test_wav_round_trip(tmp_path):
    audio = tone(1.0, 48000)
    write_wav(tmp_path / "a.wav", audio, 48000)
    back, sr = read_wav(tmp_path / "a.wav")
    assert sr == 48000 and back.dtype == np.float32
    assert np.allclose(back, audio, atol=1e-4)


def test_write_wav_clips_out_of_range_values(tmp_path):
    write_wav(tmp_path / "a.wav", np.array([2.0, -2.0, 0.0], dtype=np.float32), 16000)
    back, _ = read_wav(tmp_path / "a.wav")
    assert back[0] == pytest.approx(1.0, abs=1e-3) and back[1] == pytest.approx(-1.0, abs=1e-3)


def test_write_wav_leaves_no_temp_file(tmp_path):
    write_wav(tmp_path / "a.wav", tone(0.1, 16000), 16000)
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_read_audio_matches_written_wav(tmp_path):
    audio = tone(1.0, 48000)
    write_wav(tmp_path / "a.wav", audio, 48000)
    back = read_audio(tmp_path / "a.wav")
    assert back.dtype == np.float32 and back.ndim == 1
    assert abs(len(back) - 48000) <= 2
    assert np.allclose(back[:47000], audio[:47000], atol=1e-3)


def test_read_audio_resamples_to_requested_rate(tmp_path):
    write_wav(tmp_path / "a.wav", tone(1.0, 44100), 44100)
    back = read_audio(tmp_path / "a.wav", 48000)
    assert abs(len(back) - 48000) <= 64


def test_read_audio_downmixes_stereo(tmp_path):
    left = tone(0.5, 48000, amp=0.5)
    write_stereo_wav(tmp_path / "s.wav", left, -left, 48000)
    back = read_audio(tmp_path / "s.wav")
    assert back.ndim == 1
    assert np.max(np.abs(back)) < 0.01


def test_text_file_with_audio_extension_is_a_clear_error(tmp_path):
    fake = tmp_path / "notes.mp3"
    fake.write_text("this is not audio")
    with pytest.raises(AudioFileError, match="notes.mp3"):
        read_audio(fake)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(AudioFileError, match="gone.wav"):
        read_audio(tmp_path / "gone.wav")


def test_directory_is_a_clear_error(tmp_path):
    folder = tmp_path / "folder.wav"
    folder.mkdir()
    with pytest.raises(AudioFileError, match="folder.wav"):
        read_audio(folder)


def test_empty_audio_file_is_a_clear_error(tmp_path):
    write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32), 48000)
    with pytest.raises(AudioFileError, match="empty.wav"):
        read_audio(tmp_path / "empty.wav")


def test_too_long_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("aivoice.audiofile.MAX_SECONDS", 1)
    write_wav(tmp_path / "long.wav", tone(3.0, 16000), 16000)
    with pytest.raises(AudioFileError, match="long.wav"):
        read_audio(tmp_path / "long.wav", 16000)
