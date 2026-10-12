import os
import time
import wave
from pathlib import Path

import numpy as np

MAX_SECONDS = 3600


REPLACE_ATTEMPTS = 8


class AudioFileError(ValueError):
    pass


def replace_file(src, dst) -> None:
    """os.replace, retried briefly: Windows refuses while antivirus or an indexer has the file open."""
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS:
                raise
            time.sleep(0.05 * attempt)


def read_audio(path, sample_rate: int = 48000) -> np.ndarray:
    """Decode any audio file PyAV understands to mono float32 at sample_rate."""
    import av

    path = Path(path)
    name = path.name
    # Only real files are opened, and PyAV gets a file object, never a string: FFmpeg would treat
    # "http://...", "concat:..." and similar strings as protocols and could touch the network.
    if not path.is_file():
        raise AudioFileError(f"{name} was not found or is not a file")
    chunks: list[np.ndarray] = []
    total = 0
    try:
        with open(path, "rb") as fh, av.open(fh) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioFileError(f"{name} has no audio in it")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
            # Decode lazily (never build a list) so an over-long file is refused before it is fully read.
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    block = out.to_ndarray().reshape(-1)
                    chunks.append(block)
                    total += len(block)
                if total > MAX_SECONDS * sample_rate:
                    raise AudioFileError(f"{name} is longer than {MAX_SECONDS / 60:g} minutes")
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    except AudioFileError:
        raise
    except Exception as exc:  # noqa: BLE001 - PyAV raises many types for bad input
        raise AudioFileError(f"Could not read {name}: {exc}") from exc
    if not chunks or sum(len(c) for c in chunks) == 0:
        raise AudioFileError(f"{name} contains no audio")
    # In place: an hour of audio is 690 MB, so neither step may make a second full copy.
    return np.nan_to_num(np.concatenate(chunks).astype(np.float32, copy=False), copy=False)


def write_wav(path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono 16-bit PCM. Writes to a temp file first so a crash never leaves a half file."""
    pcm = (np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32)), -1.0, 1.0) * 32767.0).astype("<i2")
    tmp = Path(str(path) + ".tmp")
    try:
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        replace_file(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_wav(path) -> tuple[np.ndarray, int]:
    name = Path(path).name
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise AudioFileError(f"{name} is not a 16-bit WAV file")
            sr = w.getframerate()
            channels = w.getnchannels()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
    except AudioFileError:
        raise
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        raise AudioFileError(f"Could not read {name}: {exc}") from exc
    if channels > 1:
        data = data[: len(data) // channels * channels].reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sr
