import numpy as np
import pyworld

F0_MIN = 50.0
F0_MAX = 1100.0
_MEL_MIN = 1127.0 * np.log(1.0 + F0_MIN / 700.0)
_MEL_MAX = 1127.0 * np.log(1.0 + F0_MAX / 700.0)


def _resize(f0: np.ndarray, n: int) -> np.ndarray:
    if len(f0) == n:
        return f0
    if len(f0) == 0:
        return np.zeros(n, dtype=f0.dtype)
    idx = np.round(np.linspace(0, len(f0) - 1, n)).astype(int)
    return f0[idx]


def extract_f0(audio16k: np.ndarray, method: str, n_frames: int, rmvpe=None) -> np.ndarray:
    """Pitch for n_frames frames. "rmvpe" needs the RMVPE tracker passed in, and falls back to "dio" without it."""
    x = audio16k.astype(np.float64)
    if method not in ("dio", "harvest", "rmvpe"):
        raise ValueError(f"Unknown f0 method: {method}")
    if method == "rmvpe":
        if rmvpe is not None:
            if not np.any(np.abs(x) > 1e-6):
                return np.zeros(n_frames, dtype=np.float32)
            f0 = np.nan_to_num(rmvpe.f0(audio16k.astype(np.float32)), nan=0.0, posinf=0.0, neginf=0.0)
            return _resize(f0, n_frames).astype(np.float32)
        method = "dio"
    if not np.any(np.abs(x) > 1e-6):
        return np.zeros(n_frames, dtype=np.float32)
    if method == "dio":
        f0, t = pyworld.dio(x, 16000, f0_floor=F0_MIN, f0_ceil=F0_MAX, frame_period=10)
        f0 = pyworld.stonemask(x, f0, t, 16000)
    else:
        f0, _ = pyworld.harvest(x, 16000, f0_floor=F0_MIN, f0_ceil=F0_MAX, frame_period=10)
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    return _resize(f0, n_frames).astype(np.float32)


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    return (f0 * 2.0 ** (semitones / 12.0)).astype(np.float32)


def to_coarse(f0: np.ndarray) -> np.ndarray:
    mel = 1127.0 * np.log(1.0 + f0.astype(np.float64) / 700.0)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - _MEL_MIN) * 254.0 / (_MEL_MAX - _MEL_MIN) + 1.0
    return np.rint(np.clip(mel, 1.0, 255.0)).astype(np.int64)
