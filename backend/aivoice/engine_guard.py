"""A last check on what a voice model returns, before anyone can hear it.

A model whose training went wrong can answer every input with noise at full volume. Nothing about its file
looks different, so the only place to catch it is its output: speech never fills a whole window at this level.
Raising here uses the paths that already exist for a failed conversion: live mode plays silence and stops with
the message after a few failures, and text to speech fails the sentence.
"""
import numpy as np

# Converted speech sits around -25 dB (RMS over a window); the loudest real voice stays far below this.
NOISE_RMS = 0.5  # -6 dB relative to full scale
NOISE_PEAK = 1.5  # sample values are meant to stay within 1.0
MESSAGE = ("The voice model produced loud noise instead of a voice, so nothing was played. The model is "
           "damaged, most likely by a training run that went wrong: train this voice again.")


class DamagedModel(ValueError):
    pass


def check(audio) -> None:
    """Raises DamagedModel when audio is the full-volume noise of a broken model."""
    out = np.asarray(audio, dtype=np.float32)
    if out.size == 0:
        return
    finite = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.abs(finite).max()) > NOISE_PEAK or float(np.sqrt(np.mean(finite * finite))) > NOISE_RMS:
        raise DamagedModel(MESSAGE)


class GuardedEngine:
    """A voice engine whose output is checked before it is returned."""

    def __init__(self, inner):
        self._inner = inner

    def convert(self, audio, sample_rate, pitch_shift, params):
        out = self._inner.convert(audio, sample_rate, pitch_shift, params)
        check(out)
        return out

    def __getattr__(self, name):  # anything else the engine offers is passed through
        return getattr(self._inner, name)
