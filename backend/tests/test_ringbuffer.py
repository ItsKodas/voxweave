import numpy as np

from aivoice.ringbuffer import RingBuffer


def test_fifo_order():
    rb = RingBuffer(10)
    rb.write(np.arange(4, dtype=np.float32))
    assert rb.available == 4
    assert rb.read(4).tolist() == [0, 1, 2, 3]
    assert rb.available == 0


def test_wraparound():
    rb = RingBuffer(5)
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    rb.read(2)
    rb.write(np.array([4, 5, 6], dtype=np.float32))
    assert rb.read(4).tolist() == [3, 4, 5, 6]


def test_short_read_pads_with_silence_and_counts_underrun():
    rb = RingBuffer(10)
    rb.write(np.ones(2, dtype=np.float32))
    assert rb.read(5).tolist() == [1, 1, 0, 0, 0]
    assert rb.underruns == 1


def test_no_underrun_counted_before_first_write():
    rb = RingBuffer(10)
    assert rb.read(4).tolist() == [0, 0, 0, 0]
    assert rb.underruns == 0


def test_overflow_drops_oldest():
    rb = RingBuffer(4)
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    rb.write(np.array([4, 5, 6], dtype=np.float32))
    assert rb.overflows == 1
    assert rb.read(4).tolist() == [3, 4, 5, 6]


def test_write_larger_than_capacity_keeps_newest():
    rb = RingBuffer(3)
    rb.write(np.arange(10, dtype=np.float32))
    assert rb.read(3).tolist() == [7, 8, 9]


def test_oversized_write_into_nonempty_buffer_counts_one_overflow():
    rb = RingBuffer(3)
    rb.write(np.ones(2, dtype=np.float32))
    rb.write(np.arange(10, dtype=np.float32))
    assert rb.overflows == 1
    assert rb.read(3).tolist() == [7, 8, 9]
