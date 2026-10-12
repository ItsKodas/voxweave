import os
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


def test_non_file_strings_never_reach_ffmpeg(tmp_path, monkeypatch):
    import av

    real_open = av.open

    def guarded(file, *a, **k):
        assert not isinstance(file, str), "a string reached FFmpeg's protocol parser"
        return real_open(file, *a, **k)

    monkeypatch.setattr(av, "open", guarded)
    monkeypatch.chdir(tmp_path)
    for bad in ("http://127.0.0.1:9/x.wav", "concat:a.wav|b.wav", "subfile,,start,0,end,1,,:x.wav"):
        with pytest.raises(AudioFileError):
            read_audio(bad)


def test_real_file_is_opened_via_file_object(tmp_path, monkeypatch):
    import av

    real_open = av.open
    seen = []

    def spy(file, *a, **k):
        seen.append(type(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(av, "open", spy)
    write_wav(tmp_path / "a.wav", tone(0.2, 16000), 16000)
    read_audio(tmp_path / "a.wav", 16000)
    assert seen and all(t is not str for t in seen)


def test_read_wav_garbage_and_truncated_are_clear_errors(tmp_path):
    (tmp_path / "junk.wav").write_bytes(b"not a wav at all")
    with pytest.raises(AudioFileError, match="junk.wav"):
        read_wav(tmp_path / "junk.wav")
    write_wav(tmp_path / "ok.wav", tone(0.5, 16000), 16000)
    data = (tmp_path / "ok.wav").read_bytes()
    (tmp_path / "cut.wav").write_bytes(data[:20])
    with pytest.raises(AudioFileError, match="cut.wav"):
        read_wav(tmp_path / "cut.wav")
    with pytest.raises(AudioFileError, match="gone.wav"):
        read_wav(tmp_path / "gone.wav")


def test_read_wav_rejects_8_bit(tmp_path):
    with wave.open(str(tmp_path / "eight.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(bytes(range(100)))
    with pytest.raises(AudioFileError, match="eight.wav"):
        read_wav(tmp_path / "eight.wav")


def test_write_wav_failure_cleans_tmp_and_keeps_original(tmp_path, monkeypatch):
    target = tmp_path / "a.wav"
    write_wav(target, tone(0.1, 16000), 16000)
    before = target.read_bytes()

    def boom(*a, **k):
        raise OSError("disk says no")

    monkeypatch.setattr("aivoice.audiofile.os.replace", boom)
    with pytest.raises(OSError):
        write_wav(target, tone(0.2, 16000), 16000)
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]
    assert target.read_bytes() == before


def test_write_wav_retries_when_windows_has_the_file_open(tmp_path, monkeypatch):
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr("aivoice.audiofile.os.replace", flaky)
    monkeypatch.setattr("aivoice.audiofile.time.sleep", lambda s: None)
    audio = tone(0.1, 16000)
    write_wav(tmp_path / "a.wav", audio, 16000)
    monkeypatch.undo()
    assert calls["n"] == 3
    back, sr = read_wav(tmp_path / "a.wav")
    assert sr == 16000 and np.allclose(back, audio, atol=1e-4)
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_write_wav_gives_up_when_the_file_stays_in_use_and_leaves_no_temp_file(tmp_path, monkeypatch):
    calls = {"n": 0}

    def locked(src, dst):
        calls["n"] += 1
        raise PermissionError("in use")

    monkeypatch.setattr("aivoice.audiofile.os.replace", locked)
    monkeypatch.setattr("aivoice.audiofile.time.sleep", lambda s: None)
    with pytest.raises(PermissionError):
        write_wav(tmp_path / "a.wav", tone(0.1, 16000), 16000)
    monkeypatch.undo()
    assert calls["n"] > 1  # it was retried before giving up
    assert list(tmp_path.iterdir()) == []


def test_read_audio_returns_float32_with_non_finite_samples_zeroed(tmp_path):
    audio = tone(0.5, 48000)
    write_wav(tmp_path / "a.wav", audio, 48000)
    back = read_audio(tmp_path / "a.wav", 48000)
    assert back.dtype == np.float32 and back.flags.writeable and np.isfinite(back).all()
    assert np.allclose(back[: len(audio)], audio[: len(back)], atol=1e-3)
