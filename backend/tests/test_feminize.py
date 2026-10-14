import json

import numpy as np
import pyworld
import pytest

from aivoice.feminize import (FeminizeEngine, FeminizeJob, FeminizeSettings, feminize, load_settings,
                              save_settings)
from aivoice.library import Library
from aivoice.presets import PRESETS

SR = 48000


def resonator(x, f, sr, bandwidth=80.0):
    """A two-pole resonance at f Hz (a formant), run sample by sample."""
    r = np.exp(-np.pi * bandwidth / sr)
    a1, a2 = -2 * r * np.cos(2 * np.pi * f / sr), r * r
    y = np.zeros(len(x))
    y1 = y2 = 0.0
    for i, v in enumerate(x):
        y0 = (1 - r) * v - a1 * y1 - a2 * y2
        y[i], y2, y1 = y0, y1, y0
    return y


def vowel(seconds=1.5, f0=120.0, formants=(700.0, 1200.0, 2600.0), sr=SR, amp=0.3):
    """A vowel-like sound: a pulse train at f0 through resonators at the formant frequencies."""
    n = int(seconds * sr)
    pulses = np.zeros(n)
    pulses[:: int(sr / f0)] = 1.0
    out = pulses
    for f in formants:
        out = resonator(out, f, sr)
    out = out / np.max(np.abs(out)) * amp
    return out.astype(np.float32)


def median_f0(x, sr=SR):
    f0, _ = pyworld.harvest(x.astype(np.float64), sr, f0_floor=60, f0_ceil=800)
    return float(np.median(f0[f0 > 0]))


def first_formant(x, sr=SR, low=300.0, high=1000.0):
    """The strongest peak of the mean spectral envelope between low and high Hz."""
    x = x.astype(np.float64)
    f0, t = pyworld.harvest(x, sr, f0_floor=60, f0_ceil=800)
    envelope = pyworld.cheaptrick(x, f0, t, sr)[f0 > 0].mean(axis=0)
    freqs = np.linspace(0, sr / 2, envelope.shape[0])
    band = (freqs >= low) & (freqs <= high)
    # The centre of mass of the resonance, not its highest bin: the estimated envelope leans towards the nearest
    # harmonic, and harmonics are 120 to 160 Hz apart here.
    weights = envelope[band] ** 2
    return float(np.sum(freqs[band] * weights) / np.sum(weights))


def test_pitch_and_formants_move_by_the_amounts_asked_for():
    x = vowel()
    y = feminize(x, SR, FeminizeSettings(pitch_semitones=5.0, formant_shift=1.15, breathiness=0.0))
    assert median_f0(y) == pytest.approx(120.0 * 2 ** (5 / 12), rel=0.03)
    assert first_formant(x) == pytest.approx(700.0, rel=0.06)
    assert first_formant(y) == pytest.approx(700.0 * 1.15, rel=0.06)


def test_pitch_alone_leaves_the_formants_where_they_are():
    x = vowel()
    y = feminize(x, SR, FeminizeSettings(pitch_semitones=5.0, formant_shift=1.0, breathiness=0.0))
    assert median_f0(y) == pytest.approx(160.0, rel=0.03)
    assert first_formant(y) == pytest.approx(first_formant(x), rel=0.06)


def test_breathiness_adds_noise_between_the_harmonics():
    x = vowel()
    plain = feminize(x, SR, FeminizeSettings(breathiness=0.0))
    breathy = feminize(x, SR, FeminizeSettings(breathiness=0.6))

    def noise_floor(y):
        spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        return float(np.median(spectrum)) / float(np.max(spectrum))

    assert noise_floor(breathy) > 2 * noise_floor(plain)


def test_the_result_keeps_the_length_and_loudness_and_never_clips():
    x = vowel(amp=0.95)
    y = feminize(x, SR, FeminizeSettings())
    assert len(y) == len(x) and y.dtype == np.float32
    assert np.sqrt(np.mean(y ** 2)) == pytest.approx(np.sqrt(np.mean(x ** 2)), rel=0.15)
    assert np.abs(y).max() <= 1.0 and np.all(np.isfinite(y))


def test_silence_short_and_broken_input_give_silence_not_an_error():
    s = FeminizeSettings()
    assert not feminize(np.zeros(SR, dtype=np.float32), SR, s).any()
    assert len(feminize(np.zeros(100, dtype=np.float32), SR, s)) == 100
    broken = vowel()
    broken[100:200] = np.nan
    assert np.all(np.isfinite(feminize(broken, SR, s)))


@pytest.mark.parametrize("kwargs", [dict(pitch_semitones=20), dict(formant_shift=2.0), dict(formant_shift=0.5),
                                    dict(breathiness=-0.1), dict(breathiness=1.0), dict(pitch_semitones=float("nan")),
                                    dict(pitch_semitones=True)])
def test_settings_outside_their_ranges_are_refused(kwargs):
    with pytest.raises(ValueError):
        FeminizeSettings(**kwargs).validated()


def window(params, tail):
    size = SR * params.context_ms // 1000 + SR * params.chunk_ms // 1000 + SR * params.crossfade_ms // 1000
    out = np.zeros(size, dtype=np.float32)
    out[-len(tail):] = tail[-size:]
    return out


def test_the_live_preview_returns_a_full_window_with_the_newest_part_feminized():
    params = PRESETS["balanced"]
    engine = FeminizeEngine(FeminizeSettings(pitch_semitones=5.0))
    w = window(params, vowel(2.0))
    out = engine.convert(w, SR, 0.0, params)
    assert out.shape == w.shape and out.dtype == np.float32 and np.all(np.isfinite(out))
    played = out[-(SR * (params.chunk_ms + params.crossfade_ms) // 1000):]
    assert np.abs(played).max() > 0.05
    assert median_f0(played) == pytest.approx(160.0, rel=0.06)
    # The live pitch slider adds to the preview's own shift.
    raised = engine.convert(w, SR, 3.0, params)[-len(played):]
    assert median_f0(raised) == pytest.approx(120.0 * 2 ** (8 / 12), rel=0.06)


def test_the_live_preview_is_quiet_for_silence():
    params = PRESETS["balanced"]
    out = FeminizeEngine(FeminizeSettings()).convert(np.zeros(window(params, vowel()).shape, np.float32), SR, 0.0, params)
    assert not out.any()


def test_settings_are_kept_and_a_damaged_file_gives_the_defaults(tmp_path):
    assert load_settings(tmp_path) == FeminizeSettings()
    saved = save_settings(tmp_path, FeminizeSettings(6.0, 1.2, 0.3))
    assert load_settings(tmp_path) == saved == FeminizeSettings(6.0, 1.2, 0.3)
    for text in ("nonsense", "{}", json.dumps({"pitch_semitones": 99, "formant_shift": 1.1, "breathiness": 0.1})):
        (tmp_path / "feminize.json").write_text(text)
        assert load_settings(tmp_path) == FeminizeSettings()
    with pytest.raises(ValueError):
        save_settings(tmp_path, FeminizeSettings(pitch_semitones=50))


def voice_with_clips(tmp_path, n=3):
    library = Library(tmp_path)
    voice = library.create("Kai")
    for i in range(n):
        library.add_clip(voice.id, vowel(1.2, f0=110 + 5 * i), f"Sentence number {i}.", "recorded")
    return library, voice


def test_a_feminized_voice_is_made_from_every_clip_with_the_same_transcripts(tmp_path):
    library, voice = voice_with_clips(tmp_path)
    before = sorted((p.name, p.read_bytes()) for p in (tmp_path / "voices" / voice.id / "clips").iterdir())
    job = FeminizeJob(library)
    started = job.start(voice.id, FeminizeSettings(pitch_semitones=5.0))
    assert started["active"] and started["total"] == 3 and started["voice_name"] == "Kai (feminized)"
    job.join(60)
    status = job.status()
    assert status["finished"] and not status["active"] and status["error"] is None and status["done"] == 3
    made = library.get(status["voice_id"])
    assert [c.text for c in made.clips] == ["Sentence number 0.", "Sentence number 1.", "Sentence number 2."]
    assert all(c.source == "imported" and c.origin.startswith("feminized from clip") for c in made.clips)
    audio, _ = __import__("aivoice.audiofile", fromlist=["read_wav"]).read_wav(
        library.clip_audio_path(made.id, made.clips[0].index))
    assert median_f0(audio) == pytest.approx(110 * 2 ** (5 / 12), rel=0.04)
    # The source voice is untouched, byte for byte.
    assert before == sorted((p.name, p.read_bytes()) for p in (tmp_path / "voices" / voice.id / "clips").iterdir())
    assert made.models == []  # a dataset, ready to train


def test_one_job_at_a_time_a_custom_name_and_a_voice_without_transcripts(tmp_path):
    library, voice = voice_with_clips(tmp_path, n=2)
    job = FeminizeJob(library)
    job.start(voice.id, FeminizeSettings(), name="Kai soft")
    with pytest.raises(ValueError, match="already being made"):
        job.start(voice.id, FeminizeSettings())
    job.join(60)
    assert job.status()["voice_name"] == "Kai soft"
    empty = library.create("Empty")
    with pytest.raises(ValueError, match="no clips"):
        FeminizeJob(library).start(empty.id, FeminizeSettings())
    with pytest.raises(ValueError):
        FeminizeJob(library).start(voice.id, FeminizeSettings(pitch_semitones=40))
    assert len(library.list()) == 3  # nothing was made for the refused starts


def test_a_cancelled_job_stops_and_keeps_what_it_made(tmp_path):
    library, voice = voice_with_clips(tmp_path, n=3)
    job = FeminizeJob(library)
    job.cancel()  # a cancel before a start does not carry over
    job.start(voice.id, FeminizeSettings())
    job.cancel()
    job.join(60)
    status = job.status()
    assert status["finished"] and status["cancelled"] and status["done"] < 3
