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


def test_pitches_outside_the_voices_range_are_moved_toward_the_nearest_it_knows():
    low, high = 180.0, 320.0
    f0 = np.array([0, 250, 330, 400, 640, 90, 150, 170], dtype=np.float32)
    out = fit_to_range(f0, low, high)
    assert out[0] == 0 and out[1] == 250  # silence and in-range pitches are untouched
    assert high < out[2] < 330 and high < out[3] < 400  # above: pulled toward the edge
    assert out[2] < out[3] < out[4]  # the contour keeps its order: nothing is folded down an octave
    assert out[4] == pytest.approx(high * 2 ** 0.5, rel=0.01)  # an octave above keeps half of it
    assert out[5] < out[6] < out[7] < low and out[5] == pytest.approx(low / 2 ** 0.5, rel=0.01)
    assert fit_to_range(f0, 0, 0).tolist() == f0.tolist()  # no range known: nothing changes


def test_a_rise_past_the_top_of_the_range_keeps_rising_instead_of_going_flat():
    high = 340.0
    rise = np.geomspace(250, 520, 40).astype(np.float32)  # a male voice shifted up, reaching high
    out = fit_to_range(rise, 120.0, high)
    assert np.all(np.diff(out) > 0)
    over = 12 * np.log2(out[-1] / high)
    assert over == pytest.approx(0.5 * 12 * np.log2(520 / high), rel=0.02)  # half the height above the edge kept


def test_consonants_keep_their_features_where_there_is_no_pitch():
    phone = np.ones((1, 4, 3), np.float32)  # pulled towards the training voice by the index
    phone0 = np.zeros((1, 4, 3), np.float32)  # as the speaker said it
    pitchf = np.array([200, 0, 210, 0], np.float32)
    out = protect_consonants(phone, phone0, pitchf, 0.33)
    assert out[0, 0].tolist() == [1, 1, 1] and out[0, 2].tolist() == [1, 1, 1]  # voiced: the index's result
    assert np.allclose(out[0, 1], 0.33) and np.allclose(out[0, 3], 0.33)  # unvoiced: mostly the original
    assert protect_consonants(phone, phone0, pitchf, 0.5) is phone  # 0.5 or more protects nothing


SR16 = 16000


def tone(seconds, hz=110.0, amp=0.3):
    t = np.arange(int(seconds * SR16)) / SR16
    return (amp * sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6))).astype(np.float32)


def hiss(seconds, amp=0.1):
    noise = np.random.default_rng(0).standard_normal(int(seconds * SR16))
    return (amp * np.diff(np.diff(noise, prepend=0), prepend=0)).astype(np.float32)  # mostly high frequencies


def test_voice_like_frames_tells_voice_from_hiss_and_silence():
    from aivoice.engines.pitch import voice_like_frames

    audio = np.concatenate([tone(0.5), hiss(0.5), np.zeros(8000, np.float32)])
    flags = voice_like_frames(audio, len(audio) // 160)
    assert flags[5:45].mean() > 0.9  # the tone
    assert flags[55:95].mean() < 0.1  # the hiss, like an "s"
    assert flags[105:145].mean() < 0.1  # silence


def test_creak_where_the_tracker_lost_the_pitch_holds_the_last_good_pitch():
    from aivoice.engines.pitch import repair_creak

    f0 = np.array([110.0] * 20 + [0.0] * 10 + [50.0] * 5 + [105.0] * 5, np.float32)
    voice_like = np.ones(len(f0), bool)
    out = repair_creak(f0, voice_like)
    assert np.all(out[20:35] == 110.0)  # lost and far-too-low frames held at the last good pitch
    assert np.all(out[35:] == 105.0)


def test_consonants_pauses_and_long_stretches_are_left_alone():
    from aivoice.engines.pitch import repair_creak

    f0 = np.array([110.0] * 20 + [0.0] * 10 + [110.0] * 10, np.float32)
    hissing = np.ones(len(f0), bool)
    hissing[20:30] = False  # an "s": not voice-like
    assert np.all(repair_creak(f0, hissing)[20:30] == 0)
    long_gap = np.array([110.0] * 20 + [0.0] * 60, np.float32)
    out = repair_creak(long_gap, np.ones(len(long_gap), bool), max_frames=40)
    assert np.all(out[20:60] == 110.0) and np.all(out[60:] == 0)  # held for 0.4 s, no longer
    assert not repair_creak(np.zeros(10, np.float32), np.ones(10, bool)).any()  # nothing good to hold


def _glide(n=60, start=200.0, end=240.0):
    return np.geomspace(start, end, n).astype(np.float32)


def test_octave_errors_up_and_down_are_moved_back_to_the_speakers_pitch():
    from aivoice.engines.pitch import fix_octave_errors

    f0 = _glide()
    f0[20:24] *= 2.0  # the tracker jumped an octave up for 40 ms
    f0[40:43] /= 2.0  # and down
    fixed = fix_octave_errors(f0)
    assert np.allclose(fixed, _glide(), rtol=1e-4)


def test_a_short_spike_that_is_not_an_octave_is_replaced_by_the_pitch_around_it():
    from aivoice.engines.pitch import fix_octave_errors

    f0 = _glide()
    f0[30:33] *= 2 ** (7 / 12)  # 7 semitones up for 30 ms
    fixed = fix_octave_errors(f0)
    assert np.all(np.abs(12 * np.log2(fixed[30:33] / _glide()[30:33])) < 0.5)


def test_real_intonation_and_pauses_are_kept():
    from aivoice.engines.pitch import fix_octave_errors

    rise = np.concatenate([np.full(30, 150.0), np.geomspace(150, 300, 15), np.full(30, 300.0)]).astype(np.float32)
    assert np.allclose(fix_octave_errors(rise), rise)  # an octave rise over 150 ms is speech, not an error
    gapped = np.concatenate([np.full(20, 200.0), np.zeros(10), np.full(20, 200.0)]).astype(np.float32)
    assert np.array_equal(fix_octave_errors(gapped), gapped)


def test_smoothing_takes_out_wobble_and_lone_blips_but_keeps_a_glide():
    from aivoice.engines.pitch import smooth_pitch

    glide = _glide()
    wobbly = glide.copy()
    wobbly[::7] *= 2 ** (1.5 / 12)  # single-frame wobbles of 1.5 semitones
    smooth = smooth_pitch(wobbly)
    assert np.all(np.abs(12 * np.log2(smooth / glide)) < 0.2)
    blip = np.zeros(40, np.float32)
    blip[20:22] = 180.0  # 20 ms of "pitch" in a pause
    assert np.all(smooth_pitch(blip) == 0)


def test_join_pitch_starts_from_what_the_last_window_used_and_ends_on_this_ones():
    from aivoice.engines.pitch import blend_join

    current = np.full(50, 220.0, np.float32)
    previous = np.full(50, 200.0, np.float32)
    # the previous window covered this window's frames 0-29 (its frames 20-49)
    out = blend_join(current, previous, offset=20, start=10, length=20)
    assert np.allclose(out[:10], 200.0)  # before the join: what was already played
    assert out[10] == pytest.approx(200.0, rel=0.01) and out[29] == pytest.approx(220.0, rel=0.01)
    assert np.all(np.diff(out[10:30]) >= 0)  # a smooth move between the two
    assert np.allclose(out[30:], 220.0)


def test_join_blend_leaves_unvoiced_frames_and_missing_overlap_alone():
    from aivoice.engines.pitch import blend_join

    current = np.full(50, 220.0, np.float32)
    current[12] = 0.0
    previous = np.full(50, 200.0, np.float32)
    out = blend_join(current, previous, offset=20, start=10, length=20)
    assert out[12] == 0.0
    assert np.array_equal(blend_join(current, previous, offset=60, start=10, length=20), current)  # no overlap
