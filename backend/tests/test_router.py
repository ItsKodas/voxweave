import threading

import numpy as np
import pytest

from aivoice.ringbuffer import RingBuffer
from aivoice.router import Router


def test_fans_out_with_gain():
    r = Router(100)
    r.add_output("mic")
    r.add_output("monitor", gain=0.5)
    r.push(np.ones(4, dtype=np.float32))
    assert r.read("mic", 4).tolist() == [1, 1, 1, 1]
    assert r.read("monitor", 4).tolist() == [0.5, 0.5, 0.5, 0.5]


def test_disabled_output_gets_silence_and_stays_in_sync():
    r = Router(100)
    r.add_output("monitor")
    r.set_enabled("monitor", False)
    r.push(np.ones(4, dtype=np.float32))
    assert r.buffered("monitor") == 4
    assert r.read("monitor", 4).tolist() == [0, 0, 0, 0]


def test_set_gain():
    r = Router(100)
    r.add_output("monitor")
    r.set_gain("monitor", 0.25)
    r.push(np.ones(2, dtype=np.float32))
    assert r.read("monitor", 2).tolist() == [0.25, 0.25]


def test_has_and_unknown_output():
    r = Router(100)
    r.add_output("mic")
    assert r.has("mic") and not r.has("monitor")
    with pytest.raises(KeyError):
        r.read("monitor", 1)


def test_underruns_summed():
    r = Router(100)
    r.add_output("mic")
    r.push(np.ones(1, dtype=np.float32))
    r.read("mic", 5)
    assert r.underruns() == 1


def test_push_is_safe_while_another_thread_adds_an_output(monkeypatch):
    # Deterministic race: while push() is writing to the first output, another thread
    # adds a new output. Iterating the live dict would raise RuntimeError.
    r = Router(100)
    r.add_output("mic")
    r.add_output("monitor")
    added = []
    real_write = RingBuffer.write

    def write_then_add(self, data):
        real_write(self, data)
        if not added:
            added.append(True)
            t = threading.Thread(target=r.add_output, args=("late",))
            t.start()
            t.join(timeout=2)
            assert not t.is_alive()

    monkeypatch.setattr(RingBuffer, "write", write_then_add)
    r.push(np.ones(4, dtype=np.float32))
    assert r.has("late")
    assert r.underruns() == 0


def test_prime_replaces_buffered_audio_with_exact_silence_on_every_output():
    router = Router(1000)
    router.add_output("mic")
    router.add_output("monitor", gain=2.0)
    router.push(np.full(300, 0.5, dtype=np.float32))
    router.prime(100)
    for name in ("mic", "monitor"):
        assert router.buffered(name) == 100
        assert np.all(router.read(name, 100) == 0.0)
    assert router.underruns() == 0
