import numpy as np
import pytest

from aivoice.engines.f0 import extract_f0
from aivoice.engines.pitch import THRESHOLD, Rmvpe, decode, fill_gaps, fit_to_range, stft_magnitude
from aivoice.engines.rvc import protect_consonants


def test_the_stft_matches_rvcs_framing():
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    magnitude = stft_magnitude(audio)
    assert magnitude.shape == (1, 513, 16000 // 160 + 1)
    assert stft_magnitude(np.zeros(100, np.float32)).shape[1] == 513  # shorter than the reflect pad


def test_decode_turns_a_salience_peak_into_its_pitch_and_weak_frames_into_none():
    salience = np.zeros((3, 360))
    salience[0, 100] = 1.0  # bin 100: 20 * 100 + 1997.38 cents
    salience[1, 200], salience[1, 201] = 0.5, 0.5  # half way between two bins
    salience[2, 50] = THRESHOLD / 2  # too weak to be a pitch
    f0 = decode(salience)
    assert f0[0] == pytest.approx(10 * 2 ** ((2000 + 1997.3794) / 1200), rel=1e-4)
    assert f0[1] == pytest.approx(10 * 2 ** ((4010 + 1997.3794) / 1200), rel=1e-4)
    assert f0[2] == 0


class FakeSession:
    def __init__(self):
        self.shapes = []

    def get_inputs(self):
        return [type("I", (), {"name": "magnitude"})()]

    def run(self, _, feed):
        frames = feed["magnitude"].shape[2]
        self.shapes.append(feed["magnitude"].shape)
        salience = np.zeros((1, frames, 360), np.float32)
        salience[0, :, 120] = 1.0
        return [salience]


def test_rmvpe_pads_to_a_multiple_of_32_frames_and_returns_one_pitch_per_10_ms():
    session = FakeSession()
    f0 = Rmvpe(session).f0(np.ones(16000, np.float32) * 0.1)
    assert session.shapes[0][2] % 32 == 0 and len(f0) == 101 and np.all(f0 > 0)


def test_extract_f0_uses_rmvpe_when_given_and_falls_back_to_dio_without_it():
    audio = (0.3 * np.sin(2 * np.pi * 150 * np.arange(16000) / 16000)).astype(np.float32)
    with_rmvpe = extract_f0(audio, "rmvpe", 50, rmvpe=Rmvpe(FakeSession()))
    assert len(with_rmvpe) == 50 and np.allclose(with_rmvpe, with_rmvpe[0])
    without = extract_f0(audio, "rmvpe", 50)
    assert np.median(without[without > 0]) == pytest.approx(150, rel=0.05)
    assert not extract_f0(np.zeros(16000, np.float32), "rmvpe", 50, rmvpe=Rmvpe(FakeSession())).any()


def test_short_dropouts_inside_speech_are_bridged_and_real_pauses_are_not():
    f0 = np.array([100, 0, 0, 110, 0, 0, 0, 0, 0, 0, 120, 0, 125], dtype=np.float32)
    out = fill_gaps(f0, max_gap=4)
    assert np.all(out[1:3] > 100) and np.all(out[1:3] < 110)  # bridged, in between
    assert not out[4:10].any()  # six frames: a pause, left alone
    assert 120 < out[11] < 125
    assert out[0] == 100 and out[3] == 110
    assert not fill_gaps(np.zeros(5, np.float32)).any()


def test_pitches_outside_the_voices_range_are_moved_to_the_nearest_it_knows():
    low, high = 180.0, 320.0
    f0 = np.array([0, 250, 330, 400, 640, 90, 150, 170], dtype=np.float32)
    out = fit_to_range(f0, low, high)
    assert out[0] == 0 and out[1] == 250  # silence and in-range pitches are untouched
    assert high < out[2] < 330 and high < out[3] < 400  # just above: pulled to the edge, the rise kept
    assert out[3] > out[2]  # the contour keeps its order
    assert low <= out[4] <= high * 1.1  # an octave too high: folded down, not squashed
    assert low * 0.9 <= out[5] <= high  # an octave too low: folded up
    assert low * 0.95 < out[6] < low and out[7] > out[6]  # just below: pulled up, order kept
    assert fit_to_range(f0, 0, 0).tolist() == f0.tolist()  # no range known: nothing changes


def test_consonants_keep_their_features_where_there_is_no_pitch():
    phone = np.ones((1, 4, 3), np.float32)  # pulled towards the training voice by the index
    phone0 = np.zeros((1, 4, 3), np.float32)  # as the speaker said it
    pitchf = np.array([200, 0, 210, 0], np.float32)
    out = protect_consonants(phone, phone0, pitchf, 0.33)
    assert out[0, 0].tolist() == [1, 1, 1] and out[0, 2].tolist() == [1, 1, 1]  # voiced: the index's result
    assert np.allclose(out[0, 1], 0.33) and np.allclose(out[0, 3], 0.33)  # unvoiced: mostly the original
    assert protect_consonants(phone, phone0, pitchf, 0.5) is phone  # 0.5 or more protects nothing
