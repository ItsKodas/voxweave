import numpy as np
import pytest

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
