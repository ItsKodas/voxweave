import json

import numpy as np
import pytest

from aivoice.quicktrain import (ClipScores, GOOD_EPOCHS, MIN_EPOCHS, estimate_seconds, measured_rate, plan,
                                score_clip)

SR = 48000


def speech(seconds, amp=0.2, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    envelope = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)  # bursts and pauses, like speech
    return (amp * envelope * np.sin(2 * np.pi * 150 * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


def scored(n, seconds=6.0, kind="speech", score=1.0, start=0):
    return [(f"{i:04d}.orig.wav", seconds, score - i * 0.001, kind) for i in range(start, start + n)]


def test_clean_speech_scores_above_noisy_speech_and_unusable_clips_are_left_out():
    clean = score_clip(speech(6.0), SR)
    noisy = score_clip(speech(6.0, noise=0.05), SR)
    assert clean > noisy
    assert score_clip(speech(0.5), SR) is None  # too short
    assert score_clip(speech(6.0, amp=0.0005, noise=0.0001), SR) is None  # too quiet
    assert score_clip(speech(6.0), SR, kind="noise") is None
    assert score_clip(speech(6.0), SR, kind="laugh") < clean


def test_a_longer_limit_uses_more_audio_and_never_runs_over():
    clips = scored(400)  # 40 minutes
    previous = None
    for minutes in (10, 20, 30, 60, 120):
        p = plan(clips, minutes, 0.05)
        assert p.estimated_minutes <= minutes + 0.05
        assert MIN_EPOCHS <= p.epochs
        if previous is not None:
            assert p.audio_minutes >= previous.audio_minutes
        previous = p
    assert plan(clips, 120, 0.05).audio_minutes == 40.0  # with time to spare, everything is used


def test_with_enough_time_a_good_number_of_epochs_is_kept_before_adding_audio():
    p = plan(scored(400), 30, 0.05)
    assert p.epochs >= GOOD_EPOCHS and p.audio_minutes < 40


def test_the_best_clips_are_chosen_and_sounds_stay_a_small_share():
    clips = scored(100, score=2.0) + scored(100, score=0.5, start=100) + scored(50, kind="laugh", score=3.0, start=200)
    p = plan(clips, 15, 0.05)
    chosen = set(p.clips)
    best = {c[0] for c in clips[:100]}
    worst = {c[0] for c in clips[100:200]}
    laughs = {c[0] for c in clips[200:]}
    assert len(chosen & best) >= len(chosen & worst)
    assert len(chosen & laughs) * 6.0 <= 0.1 * p.audio_minutes * 60 + 6.0


def test_too_little_time_or_audio_is_refused_with_what_would_do():
    with pytest.raises(ValueError, match="too short to train a usable voice. Allow at least"):
        plan(scored(400), 3, 0.05)
    with pytest.raises(ValueError, match="minutes of usable audio"):
        plan(scored(20), 60, 0.05)  # two minutes of audio
    for bad in (0, -5, float("nan")):
        with pytest.raises(ValueError):
            plan(scored(400), bad, 0.05)


def test_the_estimate_grows_with_audio_and_epochs():
    assert estimate_seconds(600, 100, 0.05) < estimate_seconds(1200, 100, 0.05) < estimate_seconds(1200, 200, 0.05)


def test_the_rate_is_measured_from_earlier_runs_when_there_are_any(tmp_path):
    assert measured_rate(tmp_path) == 0.05
    folder = tmp_path / "voices" / "v" / "training"
    folder.mkdir(parents=True)
    (folder.parent / "voice.json").write_text(json.dumps({"clips": [{"seconds": 370.0}]}))  # 100 slices
    lines = "\n".join(f"[train] INFO:x:====> Epoch: {i} [2026-10-14 13:42:30] | (0:00:11.500000)" for i in range(5))
    (folder / "worker.log").write_text(lines)
    assert measured_rate(tmp_path) == pytest.approx(0.1)


def test_clip_scores_are_remembered_until_a_clip_changes(tmp_path):
    from aivoice.library import Library

    library = Library(tmp_path)
    voice = library.create("V")
    library.add_clip(voice.id, speech(6.0), "words", "recorded")
    library.add_clip(voice.id, speech(0.5), "short", "recorded")
    scores = ClipScores(tmp_path, library)
    first = scores.scores(voice.id)
    assert [c[0] for c in first] == ["0001.orig.wav"]  # the short one is not usable
    cache = tmp_path / "training-plans" / f"{voice.id}.json"
    assert cache.exists()
    library.set_kind(voice.id, 1, "noise")
    assert scores.scores(voice.id) == []  # a changed label is scored again
