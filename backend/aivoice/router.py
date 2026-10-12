from dataclasses import dataclass

import numpy as np

from aivoice.ringbuffer import RingBuffer


@dataclass
class _Output:
    buffer: RingBuffer
    gain: float
    enabled: bool = True


class Router:
    """Fans one mono stream out to named outputs, each with its own buffer and gain."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._outputs: dict[str, _Output] = {}

    def add_output(self, name: str, gain: float = 1.0) -> None:
        self._outputs[name] = _Output(RingBuffer(self._capacity), gain)

    def has(self, name: str) -> bool:
        return name in self._outputs

    def set_gain(self, name: str, gain: float) -> None:
        self._outputs[name].gain = gain

    def set_enabled(self, name: str, enabled: bool) -> None:
        self._outputs[name].enabled = enabled

    def push(self, audio: np.ndarray) -> None:
        for out in self._outputs.values():
            out.buffer.write(audio * out.gain if out.enabled else np.zeros_like(audio))

    def read(self, name: str, n: int) -> np.ndarray:
        return self._outputs[name].buffer.read(n)

    def buffered(self, name: str) -> int:
        return self._outputs[name].buffer.available

    def underruns(self) -> int:
        return sum(o.buffer.underruns for o in self._outputs.values())
