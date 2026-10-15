"""Small signal helpers shared by calibration and live mode."""
import numpy as np

RUMBLE_HZ = 60.0


def remove_rumble(audio: np.ndarray, sample_rate: int, cutoff: float = RUMBLE_HZ) -> np.ndarray:
    """The audio without what lies below `cutoff`: a microphone's DC offset and slow drift, hum and handling thumps.

    None of it is heard, and the voice model ignores it, but it counts in every level: measured on a real headset,
    99 percent of its "quiet" was drift around 5 Hz, as loud as the speaker's voice, so calibration took the room
    for too loud and the quiet gate saw speech in silence. A zero-phase fourth-order Butterworth response, applied
    in the frequency domain (one FFT, nothing shifted in time), with the ends mirrored so they do not ring."""
    x = np.asarray(audio, dtype=np.float32)
    if cutoff <= 0 or len(x) < 16:
        return x
    pad = min(len(x) - 1, int(0.25 * sample_rate))
    padded = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(len(padded), 1.0 / sample_rate)
    with np.errstate(divide="ignore"):
        response = 1.0 / np.sqrt(1.0 + (cutoff / np.maximum(freqs, 1e-9)) ** 8)
    response[0] = 0.0
    return np.fft.irfft(spectrum * response, n=len(padded))[pad: pad + len(x)].astype(np.float32)
