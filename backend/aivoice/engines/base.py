from typing import Protocol

import numpy as np

from aivoice.presets import LiveParams


class VoiceEngine(Protocol):
    def convert(self, audio: np.ndarray, sample_rate: int, pitch_shift: float, params: LiveParams) -> np.ndarray:
        """Convert mono float32 audio. Must return float32 of the same length, time-aligned at the end."""
        ...
