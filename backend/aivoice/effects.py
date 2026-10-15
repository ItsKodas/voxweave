"""Sound effects on the converted voice, after everything else: lo-fi, static, reverb and echo.

Live mode runs them on each chunk it plays, so they keep their state from chunk to chunk. Reverb, echo and the
lo-fi bandwidth are all linear, so they are one impulse response (dry sound, plus echo taps, plus a decaying
reverb tail, all through a low-pass filter), applied by FFT convolution with the tail of each chunk carried into
the next: an echo longer than a chunk still lands where it should. The impulse response is rebuilt only when a
setting changes. Lo-fi's fewer bits and lower sample rate, and static (hiss and crackle), come after it.
"""
import numpy as np

ECHO_REPEATS = 6
ECHO_FEEDBACK = 0.55  # each repeat of the echo at this much of the one before
REVERB_PREDELAY_S = 0.012


def _lowpass_taps(cutoff: float, sample_rate: int, taps: int = 127) -> np.ndarray:
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / sample_rate * n) * np.hamming(taps)
    return h / h.sum()


class Effects:
    def __init__(self, sample_rate: int, seed: int = 7):
        self._sr = sample_rate
        self._key = None
        self._ir: np.ndarray | None = None
        self._tail = np.zeros(0, np.float64)
        self._hold_phase = 0
        self._held = 0.0
        self._rng = np.random.default_rng(seed)

    def _impulse(self, lofi: float, reverb: float, room: float, echo: float, echo_ms: float) -> np.ndarray | None:
        """The linear effects as one impulse response, or None when they are all off."""
        if lofi <= 0 and reverb <= 0 and echo <= 0:
            return None
        sr = self._sr
        length = 1
        if echo > 0:
            length = max(length, int(ECHO_REPEATS * echo_ms / 1000 * sr) + 1)
        decay = 0.3 + 2.2 * room  # seconds for the reverb to fall by 60 dB
        if reverb > 0:
            length = max(length, int((REVERB_PREDELAY_S + decay) * sr))
        ir = np.zeros(length)
        ir[0] = 1.0 - 0.4 * reverb  # the dry sound, a little quieter under a lot of reverb
        if echo > 0:
            step = int(echo_ms / 1000 * sr)
            for k in range(1, ECHO_REPEATS + 1):
                if k * step < length:
                    ir[k * step] += echo * ECHO_FEEDBACK ** (k - 1)
        if reverb > 0:
            start = int(REVERB_PREDELAY_S * sr)
            t = np.arange(length - start) / sr
            tail = np.random.default_rng(11).standard_normal(len(t)) * np.exp(-6.9 * t / decay)
            tail *= reverb / np.sqrt(np.sum(tail ** 2))  # the tail's energy set by the slider
            ir[start:] += tail
        if lofi > 0:
            cutoff = 20000.0 * (2800.0 / 20000.0) ** lofi  # 20 kHz down to 2.8 kHz, evenly in octaves
            if cutoff < 0.45 * sr:
                ir = np.convolve(ir, _lowpass_taps(cutoff, sr))[: len(ir) + 126]
        return ir

    def _convolve(self, x: np.ndarray, ir: np.ndarray) -> np.ndarray:
        n = len(x) + len(ir) - 1
        size = 1 << (n - 1).bit_length()
        y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:n]
        if len(self._tail):
            if len(self._tail) > len(y):
                y = np.concatenate([y, np.zeros(len(self._tail) - len(y))])
            y[: len(self._tail)] += self._tail
        self._tail = y[len(x):].copy()
        return y[: len(x)]

    def process(self, block: np.ndarray, params) -> np.ndarray:
        lofi = float(getattr(params, "fx_lofi", 0.0))
        static = float(getattr(params, "fx_static", 0.0))
        reverb = float(getattr(params, "fx_reverb", 0.0))
        room = float(getattr(params, "fx_room", 0.5))
        echo = float(getattr(params, "fx_echo", 0.0))
        echo_ms = float(getattr(params, "fx_echo_ms", 250.0))
        if lofi <= 0 and static <= 0 and reverb <= 0 and echo <= 0 and not len(self._tail):
            return block
        x = np.asarray(block, dtype=np.float64)
        key = (round(lofi, 3), round(reverb, 3), round(room, 3), round(echo, 3), round(echo_ms))
        if key != self._key:
            self._key, self._ir = key, self._impulse(lofi, reverb, room, echo, echo_ms)
        if self._ir is not None or len(self._tail):
            x = self._convolve(x, self._ir if self._ir is not None else np.ones(1))
        if lofi > 0:
            # Fewer bits (16 down to 5) and a lower sample rate (each sample held up to 6 times).
            hold = 1 + int(round(5 * lofi))
            if hold > 1 and len(x):
                # Sample i (counting on from the last block) takes the value at the start of its group of `hold`.
                pos = self._hold_phase + np.arange(len(x))
                starts = (pos // hold) * hold - self._hold_phase
                held = np.where(starts >= 0, x[np.maximum(starts, 0)], self._held)
                self._held = float(held[-1])
                self._hold_phase = int((self._hold_phase + len(x)) % hold)
                x = held
            levels = 2 ** (16 - 11 * lofi)
            x = np.round(x * levels) / levels
        if static > 0:
            hiss = self._rng.standard_normal(len(x)) * (10 ** ((-62 + 36 * static) / 20))  # -62 dB to -26 dB
            crackle = np.zeros(len(x))
            clicks = self._rng.random(len(x)) < (2 + 40 * static) / self._sr
            crackle[clicks] = self._rng.uniform(-1, 1, int(clicks.sum())) * (0.015 + 0.06 * static)
            x = x + hiss + crackle
        return np.clip(x, -1.0, 1.0).astype(np.float32)
