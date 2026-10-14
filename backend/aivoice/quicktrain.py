"""Training to a time limit: choosing the best clips and the number of epochs that fit.

A training run costs time in two ways: preparing the audio (slicing it, finding its pitch, extracting features),
which grows with the amount of audio, and the epochs, each of which goes through every slice once. Measured on the
development PC (RX 7900 XTX, half precision, batch size 8) an epoch took about 0.05 seconds per slice of audio
(RVC slices clips into pieces of up to 3.7 seconds): koda 6 s, trey 10 s, thomas 14 s per epoch. When a voice
has been trained before, its own logs give the measured rate instead.

Given a time limit, the plan keeps enough epochs for a usable model and as much of the best audio as then fits.
"The best" is judged per clip from its level (not too quiet, not distorted), how far its speech stands above its
background noise, how much of it is voiced, and its length; laughs and other sounds are kept, but at most a tenth
of the chosen audio.
"""
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aivoice.audiofile import read_wav
from aivoice.dataset import level_problem

SLICE_SECONDS = 3.7
SECONDS_PER_SLICE_EPOCH = 0.05  # measured, see above
EPOCH_OVERHEAD_S = 1.5  # per epoch: start-up of the epoch, and a checkpoint every few epochs
FIXED_OVERHEAD_S = 120.0  # starting the worker, building the index and exporting the model
PREPARE_S_PER_AUDIO_MINUTE = 8.0  # slicing, pitch and features
MIN_EPOCHS = 40
GOOD_EPOCHS = 150
MAX_EPOCHS = 400
MIN_AUDIO_MINUTES = 4.0
NONVERBAL_SHARE = 0.1
_EPOCH_LINE = re.compile(r"====> Epoch: \d+ \[[^]]*\] \| \((\d+):(\d+):([\d.]+)\)")


@dataclass(frozen=True)
class Plan:
    clips: list  # file names, e.g. "0012.orig.wav"
    audio_minutes: float
    available_minutes: float
    epochs: int
    estimated_minutes: float
    seconds_per_slice_epoch: float


def _frame_db(audio: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, sr // 50)
    frames = len(audio) // n
    if frames == 0:
        return np.zeros(1)
    x = audio[: frames * n].astype(np.float64).reshape(frames, n)
    return 20 * np.log10(np.sqrt(np.mean(x * x, axis=1)) + 1e-9)


def score_clip(audio: np.ndarray, sr: int, kind: str = "speech") -> float | None:
    """How good a clip is to train on (higher is better), or None when it should not be used at all."""
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    seconds = len(audio) / sr if sr else 0.0
    if seconds < 1.0 or kind == "noise" or level_problem(audio) is not None:
        return None
    levels = _frame_db(audio, sr)
    snr = float(np.percentile(levels, 90) - np.percentile(levels, 10))
    score = min(max(snr, 0.0), 45.0) / 45.0  # speech well above the background
    score += 0.3 if 2.0 <= seconds <= 12.0 else 0.1
    score += 0.3 if kind == "speech" else 0.0
    peak = float(np.max(np.abs(audio)))
    score -= 0.3 if peak > 0.98 else 0.0  # touching full scale: likely clipped somewhere
    return round(score, 4)


class ClipScores:
    """Scores of a voice's clips, remembered on disk until a clip changes (scoring reads every clip once)."""

    def __init__(self, data: Path, library):
        self._folder = data / "training-plans"
        self._library = library

    def scores(self, voice_id: str) -> list[tuple[str, float, float, str]]:
        """(file name, seconds, score, kind) for every usable clip."""
        voice = self._library.get(voice_id)
        cache_path = self._folder / f"{voice.id}.json"
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        result, changed = [], False
        for clip in voice.clips:
            name = f"{clip.index:04d}.orig.wav"
            key = f"{clip.seconds}:{clip.kind}"
            entry = cache.get(name)
            if not (isinstance(entry, dict) and entry.get("key") == key):
                try:
                    audio, sr = read_wav(self._library.clip_audio_path(voice.id, clip.index))
                except ValueError:
                    continue
                entry = {"key": key, "score": score_clip(audio, sr, clip.kind)}
                cache[name] = entry
                changed = True
            if entry.get("score") is not None:
                result.append((name, float(clip.seconds), float(entry["score"]), clip.kind))
        if changed:
            try:
                self._folder.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            except OSError:
                pass  # scored again next time
        return result


def measured_rate(data: Path) -> float:
    """Seconds per slice per epoch, from the epochs of earlier training runs on this PC when there are any."""
    rates = []
    for log in sorted((data / "voices").glob("*/training/worker.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
            voice = json.loads((log.parent.parent / "voice.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        epochs = [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in _EPOCH_LINE.findall(text)][-10:]
        seconds = sum(float(c.get("seconds", 0)) for c in voice.get("clips", []))
        slices = seconds / SLICE_SECONDS
        if len(epochs) >= 3 and slices > 20:
            rates.append((float(np.median(epochs)) - EPOCH_OVERHEAD_S) / slices)
    rates = [r for r in rates if 0.005 < r < 1.0]
    return float(np.median(rates)) if rates else SECONDS_PER_SLICE_EPOCH


def estimate_seconds(audio_seconds: float, epochs: int, rate: float) -> float:
    slices = audio_seconds / SLICE_SECONDS
    return FIXED_OVERHEAD_S + PREPARE_S_PER_AUDIO_MINUTE * audio_seconds / 60 + epochs * (EPOCH_OVERHEAD_S + rate * slices)


def _best_first(scored: list[tuple[str, float, float, str]]) -> list[tuple[str, float, float, str]]:
    """Clips in the order they are chosen: best first, with sounds (laughs, coughs) held to a tenth of the audio."""
    speech = sorted((c for c in scored if c[3] == "speech"), key=lambda c: -c[2])
    sounds = sorted((c for c in scored if c[3] != "speech"), key=lambda c: -c[2])
    order, total, sound_total = [], 0.0, 0.0
    queue = speech + sounds
    deferred = []
    for clip in queue:
        if clip[3] != "speech" and sound_total + clip[1] > NONVERBAL_SHARE * max(total + clip[1], 1.0):
            deferred.append(clip)
            continue
        order.append(clip)
        total += clip[1]
        sound_total += clip[1] if clip[3] != "speech" else 0.0
    return order + deferred


def plan(scored: list[tuple[str, float, float, str]], minutes: float, rate: float) -> Plan:
    """The clips and epochs that fit in `minutes`. Raises ValueError when the time is too short for a usable model."""
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("The time limit must be a number of minutes above 0")
    budget = minutes * 60.0
    order = _best_first(scored)
    available = sum(c[1] for c in scored)
    if available / 60 < MIN_AUDIO_MINUTES:
        raise ValueError(f"This voice has {available / 60:.1f} minutes of usable audio. Training needs at least "
                         f"{MIN_AUDIO_MINUTES:.0f}.")
    cumulative = np.cumsum([c[1] for c in order])
    minimum = int(np.searchsorted(cumulative, MIN_AUDIO_MINUTES * 60)) + 1

    def epochs_for(count: int) -> int:
        seconds = float(cumulative[count - 1])
        slices = seconds / SLICE_SECONDS
        spare = budget - FIXED_OVERHEAD_S - PREPARE_S_PER_AUDIO_MINUTE * seconds / 60
        return int(spare // (EPOCH_OVERHEAD_S + rate * slices)) if spare > 0 else 0

    # As much audio as still leaves a good number of epochs; with little time, less audio and fewer epochs.
    count = minimum
    for n in range(len(order), minimum - 1, -1):
        if epochs_for(n) >= GOOD_EPOCHS:
            count = n
            break
    epochs = min(MAX_EPOCHS, epochs_for(count))
    if epochs < MIN_EPOCHS:
        needed = estimate_seconds(float(cumulative[minimum - 1]), MIN_EPOCHS, rate) / 60
        raise ValueError(f"{minutes:g} minutes is too short to train a usable voice. Allow at least "
                         f"{math.ceil(needed)} minutes.")
    chosen = order[:count]
    audio_seconds = float(cumulative[count - 1])
    return Plan(sorted(c[0] for c in chosen), round(audio_seconds / 60, 1), round(available / 60, 1), epochs,
                round(estimate_seconds(audio_seconds, epochs, rate) / 60, 1), round(rate, 4))


def plan_for(data: Path, library, voice_id: str, minutes: float) -> dict:
    p = plan(ClipScores(data, library).scores(voice_id), float(minutes), measured_rate(data))
    out = asdict(p)
    out["clip_count"] = len(p.clips)
    return out
