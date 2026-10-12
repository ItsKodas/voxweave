# Stage 2: Recording and Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user build a voice dataset in the app, either by reading prompted sentences into the microphone or by dropping in existing audio files, with clips checked, transcribed and stored ready for training.

**Architecture:** New backend modules sit beside the live path and never touch it: `audiofile` (decode and write), `dataset` (pure numpy trimming, level checks, splitting), `library` (voices and clips on disk), `prompts`, `recorder` (microphone capture), `transcribe` (Whisper on CPU), `importer` (background job) and `studio` (coordinator). They are exposed under `/library/...` routes. The frontend gains a Voices tab next to the existing Live panel.

**Tech Stack:** Python 3.12, FastAPI, numpy, soxr, PyAV (decoding), faster-whisper on CTranslate2 (CPU, int8), stdlib `wave`. React 19, TypeScript, Vitest. No new npm or Rust dependencies.

**Spec:** `docs/superpowers/specs/2026-10-13-ai-voice-design.md` (sections "Recording", "Data layout", "Build stages" item 2)

## Global Constraints

- The live audio path is frozen. Do not modify `processor.py`, `live.py`, `router.py`, `ringbuffer.py` or `engines/`. Stage 1's fail-closed tests must keep passing unchanged.
- New Python packages allowed, and only these: `faster-whisper`, `ctranslate2`, `huggingface-hub`, `tokenizers`, `av`, `tqdm`. The plain `onnxruntime` package must NOT be installed: it shares a module name with `onnxruntime-directml` and would remove GPU support from live mode. `faster-whisper` is therefore installed with `--no-deps`.
- No new npm packages and no new Rust crates.
- Recordings and imported audio never leave the machine. The only network access is the one-time Whisper model download from Hugging Face.
- Stored audio is mono. Each clip is kept twice: `NNNN.orig.wav` at 48000 Hz and `NNNN.wav` at 16000 Hz, both 16-bit PCM, plus `NNNN.txt` with the transcript.
- Deleting a voice moves its folder to `data/trash/`; it is never erased by the app.
- Clip limits: a clip is rejected below 1.0 s; a recording is rejected above 20 s; imported audio is split into pieces of at most 15 s.
- The transcript language is English. The Whisper model defaults to `small.en` and can be changed with `whisper_model` in `data/settings.json`.
- Recording and live mode are mutually exclusive, and device refresh is refused during either.
- All user-facing failures raise `ValueError` (or a subclass), which the API turns into HTTP 400 with a readable `detail`.
- Run commands from the repo root `D:\Projects\AI Voice` in Git Bash. Backend tests: `backend/.venv/Scripts/python -m pytest backend/tests -q`. Frontend tests from `app/`: `npm test`, `npx tsc --noEmit`.

## Review Focus

Conditions the spec implies but does not spell out. Each has a test in the task named.

1. **An imported file with no pauses** (continuous speech or music). Every piece must still be at most 15 s and no audio may be lost or duplicated. Test in Task 3.
2. **A dropped file that is not audio** (a renamed text file, a folder, an unsupported type, a missing path). That file is reported as failed with its name; the other files still import; nothing crashes. Tests in Task 2 and Task 7.
3. **Stop pressed at once, or a take of pure silence.** The take is rejected with a clear reason and nothing is written to disk. Tests in Task 3 and Task 8.
4. **Voice names with awkward characters**: emoji, slashes, `..`, only punctuation, duplicates, very long names. The folder name is always safe, the display name is preserved, and duplicates get distinct ids. Test in Task 4.
5. **Damage on disk**: a half-written or corrupt `voice.json`, or leftover audio files from an interrupted save. Other voices stay listed, nothing is deleted, and the next save succeeds. Test in Task 4.

## File Structure

```
backend/aivoice/
  audiofile.py        read_audio() via PyAV, write_wav()/read_wav() via stdlib wave
  dataset.py          trim_silence(), level_problem(), clip_problem(), split_on_silence()
  library.py          Library: voices and clips on disk, voices.json summary
  prompts.py          load_prompts(), next_prompt()
  prompts_en.txt      200 sentences, one per line
  recorder.py         Recorder: microphone capture with level and duration
  transcribe.py       WhisperTranscriber
  importer.py         Importer: background import job with status
  studio.py           Studio: guided recording flow, combined status
  api_library.py      library_router(): the /library routes
  api.py              (modify) mount the router, mutual exclusion with live mode
  config.py           (modify) whisper_model()
  __main__.py         (modify) build and pass the studio
backend/tests/
  test_audiofile.py test_dataset.py test_library.py test_prompts.py test_recorder.py
  test_transcribe.py test_importer.py test_studio.py test_api_library.py
app/src/
  api.ts              (modify) export call() and origin()
  studioApi.ts        typed client for /library
  useStudioStatus.ts  polling hook for recording level and import progress
  dropEvents.ts       thin wrapper over Tauri's drag-and-drop event
  VoicesPanel.tsx     voice list, create, delete, and the selected voice's studio
  RecordSession.tsx   guided recording
  ImportDrop.tsx      drop zone and import progress
  ClipList.tsx        clips with play, edit transcript, delete
  App.tsx             (modify) Live and Voices tabs
data/
  voices.json                         summary of all voices
  voices/<voice-id>/voice.json        name, created, clip records (source of truth)
  voices/<voice-id>/clips/NNNN.orig.wav, NNNN.wav, NNNN.txt
  pretrained/whisper/                 downloaded Whisper model
  trash/                              deleted voices
```

Two notes on the spec's wording. The spec names a `voices` module; here it is `library.py`, because the stage 1 route `/voices` already means "voice models available for live mode" and stays unchanged. The spec's `data/voices.json` manifest is written as a summary after every change, while each voice's own `voice.json` is the source of truth so that one damaged file cannot hide every voice.

---

### Task 1: Install transcription packages without breaking DirectML

A gate. `faster-whisper` declares a dependency on the plain `onnxruntime` package, which would overwrite the DirectML build that live mode needs.

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `docs/superpowers/notes/2026-10-13-whisper-setup.md`

**Interfaces:**
- Produces: importable `faster_whisper` and `av` in `backend/.venv`; the Whisper model `small.en` downloaded under `data/pretrained/whisper/`; `DmlExecutionProvider` still available.

- [ ] **Step 1: Record the baseline**

```bash
backend/.venv/Scripts/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

Expected: the list contains `DmlExecutionProvider`. Save the output for the notes file.

- [ ] **Step 2: Add the extra to `backend/pyproject.toml`**

Under `[project.optional-dependencies]`, add this line after the `dev` line:

```toml
# faster-whisper itself is installed separately with --no-deps: it depends on the plain
# onnxruntime package, which would overwrite onnxruntime-directml and disable GPU live mode.
transcribe = ["ctranslate2>=4.0,<5", "huggingface-hub>=0.21", "tokenizers>=0.13,<1", "av>=11", "tqdm"]
```

- [ ] **Step 3: Install**

```bash
uv pip install --python backend/.venv -e "backend[dev,transcribe]"
uv pip install --python backend/.venv --no-deps faster-whisper
```

- [ ] **Step 4: Verify DirectML survived and nothing unexpected was installed**

```bash
backend/.venv/Scripts/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
uv pip list --python backend/.venv | grep -i -E "onnxruntime|faster|ctranslate2|^av |tokenizers|huggingface"
```

Expected: the provider list still contains `DmlExecutionProvider`; `uv pip list` shows `onnxruntime-directml` and does NOT show a separate `onnxruntime` package. If plain `onnxruntime` is present, STOP and report: do not try to repair the environment by uninstalling packages without recording exactly what was done.

- [ ] **Step 5: Download the model and run it once**

```bash
backend/.venv/Scripts/python -c "
import numpy as np, time
from faster_whisper import WhisperModel
t = time.time()
model = WhisperModel('small.en', device='cpu', compute_type='int8', download_root='data/pretrained/whisper')
print('loaded in', round(time.time() - t, 1), 's')
audio = (np.random.default_rng(0).uniform(-0.01, 0.01, 16000 * 5)).astype(np.float32)
t = time.time()
segments, info = model.transcribe(audio, language='en', beam_size=5, condition_on_previous_text=False)
print('text:', repr(' '.join(s.text.strip() for s in segments)), 'in', round(time.time() - t, 1), 's')
"
find data/pretrained/whisper -name "model.bin" | head
```

Expected: the model downloads (about 480 MB), loads, and returns a string (probably empty for noise). One `model.bin` is found. Record the folder it sits in.

- [ ] **Step 6: Run the stage 1 backend suite**

Run: `backend/.venv/Scripts/python -m pytest backend/tests -q`
Expected: all pass, with `test_rvc_real.py` passing rather than skipped. This proves live mode still has its GPU.

- [ ] **Step 7: Write the notes file and commit**

`docs/superpowers/notes/2026-10-13-whisper-setup.md` records: the two install commands, why `--no-deps` is used, the before and after provider lists, installed versions of the six packages, the model folder path, and the load and transcribe timings.

```bash
git add backend/pyproject.toml docs/superpowers/notes/2026-10-13-whisper-setup.md
git commit -m "chore: add transcription packages without replacing onnxruntime-directml"
```

---

### Task 2: Audio file reading and writing

**Files:**
- Create: `backend/aivoice/audiofile.py`
- Test: `backend/tests/test_audiofile.py`

**Interfaces:**
- Produces: `AudioFileError(ValueError)`; `read_audio(path, sample_rate: int = 48000) -> np.ndarray` (mono float32 at `sample_rate`, any format PyAV can decode); `write_wav(path, audio: np.ndarray, sample_rate: int) -> None` (16-bit PCM mono, written atomically); `read_wav(path) -> tuple[np.ndarray, int]`; `MAX_SECONDS = 3600`.

- [ ] **Step 1: Write the failing tests**

```python
import wave

import numpy as np
import pytest

from aivoice.audiofile import AudioFileError, read_audio, read_wav, write_wav


def tone(seconds, sr, freq=220.0, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_stereo_wav(path, left, right, sr):
    data = np.stack([left, right], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767).astype("<i2").tobytes())


def test_wav_round_trip(tmp_path):
    audio = tone(1.0, 48000)
    write_wav(tmp_path / "a.wav", audio, 48000)
    back, sr = read_wav(tmp_path / "a.wav")
    assert sr == 48000 and back.dtype == np.float32
    assert np.allclose(back, audio, atol=1e-4)


def test_write_wav_clips_out_of_range_values(tmp_path):
    write_wav(tmp_path / "a.wav", np.array([2.0, -2.0, 0.0], dtype=np.float32), 16000)
    back, _ = read_wav(tmp_path / "a.wav")
    assert back[0] == pytest.approx(1.0, abs=1e-3) and back[1] == pytest.approx(-1.0, abs=1e-3)


def test_write_wav_leaves_no_temp_file(tmp_path):
    write_wav(tmp_path / "a.wav", tone(0.1, 16000), 16000)
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_read_audio_matches_written_wav(tmp_path):
    audio = tone(1.0, 48000)
    write_wav(tmp_path / "a.wav", audio, 48000)
    back = read_audio(tmp_path / "a.wav")
    assert back.dtype == np.float32 and back.ndim == 1
    assert abs(len(back) - 48000) <= 2
    assert np.allclose(back[:47000], audio[:47000], atol=1e-3)


def test_read_audio_resamples_to_requested_rate(tmp_path):
    write_wav(tmp_path / "a.wav", tone(1.0, 44100), 44100)
    back = read_audio(tmp_path / "a.wav", 48000)
    assert abs(len(back) - 48000) <= 64


def test_read_audio_downmixes_stereo(tmp_path):
    left = tone(0.5, 48000, amp=0.5)
    write_stereo_wav(tmp_path / "s.wav", left, -left, 48000)
    back = read_audio(tmp_path / "s.wav")
    assert back.ndim == 1
    assert np.max(np.abs(back)) < 0.01


def test_text_file_with_audio_extension_is_a_clear_error(tmp_path):
    fake = tmp_path / "notes.mp3"
    fake.write_text("this is not audio")
    with pytest.raises(AudioFileError, match="notes.mp3"):
        read_audio(fake)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(AudioFileError, match="gone.wav"):
        read_audio(tmp_path / "gone.wav")


def test_directory_is_a_clear_error(tmp_path):
    folder = tmp_path / "folder.wav"
    folder.mkdir()
    with pytest.raises(AudioFileError, match="folder.wav"):
        read_audio(folder)


def test_empty_audio_file_is_a_clear_error(tmp_path):
    write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32), 48000)
    with pytest.raises(AudioFileError, match="empty.wav"):
        read_audio(tmp_path / "empty.wav")


def test_too_long_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("aivoice.audiofile.MAX_SECONDS", 1)
    write_wav(tmp_path / "long.wav", tone(3.0, 16000), 16000)
    with pytest.raises(AudioFileError, match="long.wav"):
        read_audio(tmp_path / "long.wav", 16000)
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_audiofile.py -q`
Expected: FAIL with `ModuleNotFoundError: aivoice.audiofile`.

- [ ] **Step 3: Implement `backend/aivoice/audiofile.py`**

```python
import os
import wave
from pathlib import Path

import numpy as np

MAX_SECONDS = 3600


class AudioFileError(ValueError):
    pass


def read_audio(path, sample_rate: int = 48000) -> np.ndarray:
    """Decode any audio file PyAV understands to mono float32 at sample_rate."""
    import av

    name = Path(path).name
    chunks: list[np.ndarray] = []
    total = 0
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioFileError(f"{name} has no audio in it")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
            # Decode lazily (never build a list) so an over-long file is refused before it is fully read.
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    block = out.to_ndarray().reshape(-1)
                    chunks.append(block)
                    total += len(block)
                if total > MAX_SECONDS * sample_rate:
                    raise AudioFileError(f"{name} is longer than {MAX_SECONDS // 60} minutes")
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    except AudioFileError:
        raise
    except Exception as exc:  # noqa: BLE001 - PyAV raises many types for bad input
        raise AudioFileError(f"Could not read {name}: {exc}") from exc
    if not chunks or sum(len(c) for c in chunks) == 0:
        raise AudioFileError(f"{name} contains no audio")
    return np.nan_to_num(np.concatenate(chunks).astype(np.float32))


def write_wav(path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono 16-bit PCM. Writes to a temp file first so a crash never leaves a half file."""
    pcm = (np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32)), -1.0, 1.0) * 32767.0).astype("<i2")
    tmp = Path(str(path) + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data.astype(np.float32), sr
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_audiofile.py -q`
Expected: 11 passed. If PyAV's API differs (for example the flush call), adapt the implementation and keep the tests.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/audiofile.py backend/tests/test_audiofile.py
git commit -m "feat: decode audio files and write clip WAVs"
```

---

### Task 3: Dataset processing

Pure numpy. No files, no devices.

**Files:**
- Create: `backend/aivoice/dataset.py`
- Test: `backend/tests/test_dataset.py`

**Interfaces:**
- Produces: `trim_silence(audio, sr, threshold_db=-40.0, pad_ms=100) -> np.ndarray`; `level_problem(audio) -> str | None` returning `"too_quiet"`, `"clipped"` or `None`; `clip_problem(audio, sr) -> str | None` returning `"too_quiet"`, `"too_short"`, `"too_long"`, `"clipped"` or `None` for an already trimmed take; `split_on_silence(audio, sr, max_seconds=15.0, min_seconds=1.0, threshold_db=-40.0, min_silence_ms=300, pad_ms=100) -> list[np.ndarray]`; `PROBLEM_MESSAGES: dict[str, str]`; constants `MIN_SECONDS = 1.0`, `MAX_RECORD_SECONDS = 20.0`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from aivoice.dataset import (
    MAX_RECORD_SECONDS, PROBLEM_MESSAGES, clip_problem, level_problem, split_on_silence, trim_silence,
)

SR = 48000
RNG = np.random.default_rng(0)


def noise(seconds, amp=0.3):
    return RNG.uniform(-amp, amp, int(seconds * SR)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_trim_removes_leading_and_trailing_silence_keeping_padding():
    audio = np.concatenate([silence(1.0), noise(2.0), silence(1.5)])
    trimmed = trim_silence(audio, SR)
    assert abs(len(trimmed) / SR - 2.2) < 0.03


def test_trim_of_silence_is_empty():
    assert len(trim_silence(silence(2.0), SR)) == 0


def test_trim_of_empty_input_is_empty():
    assert len(trim_silence(np.zeros(0, dtype=np.float32), SR)) == 0


def test_trim_keeps_internal_pauses():
    audio = np.concatenate([noise(1.0), silence(1.0), noise(1.0)])
    assert abs(len(trim_silence(audio, SR)) / SR - 3.0) < 0.03


def test_trim_survives_nan():
    audio = noise(1.0)
    audio[100] = np.nan
    assert np.all(np.isfinite(trim_silence(audio, SR)))


def test_level_ok():
    assert level_problem(noise(2.0)) is None


def test_level_too_quiet():
    assert level_problem(noise(2.0, amp=0.01)) == "too_quiet"


def test_level_clipped():
    audio = noise(2.0)
    audio[1000:1400] = 1.0
    assert level_problem(audio) == "clipped"


def test_a_few_full_scale_samples_are_not_clipping():
    audio = noise(2.0)
    audio[10:14] = 1.0
    assert level_problem(audio) is None


def test_clip_problem_order():
    assert clip_problem(np.zeros(0, dtype=np.float32), SR) == "too_quiet"
    assert clip_problem(noise(0.5), SR) == "too_short"
    assert clip_problem(noise(MAX_RECORD_SECONDS + 1), SR) == "too_long"
    assert clip_problem(noise(3.0, amp=0.01), SR) == "too_quiet"
    assert clip_problem(noise(3.0), SR) is None


def test_every_problem_has_a_message():
    assert set(PROBLEM_MESSAGES) == {"too_quiet", "too_short", "too_long", "clipped"}
    assert all(len(m) > 20 for m in PROBLEM_MESSAGES.values())


def test_split_separates_utterances_at_pauses():
    audio = np.concatenate([silence(0.5), noise(2.0), silence(0.5), noise(2.0), silence(0.5), noise(2.0), silence(0.5)])
    pieces = split_on_silence(audio, SR)
    assert len(pieces) == 3
    assert all(abs(len(p) / SR - 2.2) < 0.03 for p in pieces)


def test_split_ignores_short_gaps():
    audio = np.concatenate([noise(2.0), silence(0.2), noise(2.0)])
    assert len(split_on_silence(audio, SR)) == 1


def test_split_drops_pieces_under_one_second():
    audio = np.concatenate([noise(0.5), silence(1.0), noise(2.0)])
    pieces = split_on_silence(audio, SR)
    assert len(pieces) == 1 and len(pieces[0]) / SR > 1.9


def test_split_of_silence_is_empty():
    assert split_on_silence(silence(5.0), SR) == []


def test_split_of_empty_input_is_empty():
    assert split_on_silence(np.zeros(0, dtype=np.float32), SR) == []


def test_continuous_audio_is_cut_into_pieces_of_at_most_fifteen_seconds_without_loss():
    audio = noise(40.0)
    pieces = split_on_silence(audio, SR)
    assert len(pieces) >= 3
    assert all(len(p) <= 15 * SR for p in pieces)
    assert np.array_equal(np.concatenate(pieces), audio)


def test_long_utterance_is_cut_at_a_quiet_moment():
    quiet = noise(0.2, amp=0.02)  # above the silence threshold, so it does not end the region
    audio = np.concatenate([noise(13.0), quiet, noise(10.0)])
    pieces = split_on_silence(audio, SR)
    assert len(pieces) == 2
    first_end = len(pieces[0]) / SR
    assert 13.0 <= first_end <= 13.3
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: aivoice.dataset`.

- [ ] **Step 3: Implement `backend/aivoice/dataset.py`**

```python
import numpy as np

FRAME_MS = 10
MIN_SECONDS = 1.0
MAX_RECORD_SECONDS = 20.0
QUIET_PEAK = 0.05
CLIP_LEVEL = 0.999

PROBLEM_MESSAGES = {
    "too_quiet": "Nothing was heard, or it was too quiet. Check the microphone and speak closer to it.",
    "too_short": "That take was too short. Read the whole sentence before pressing Stop.",
    "too_long": "That take was too long. Keep each take under 20 seconds.",
    "clipped": "The recording distorted. Move back from the microphone or lower its level, then try again.",
}


def _clean(audio) -> np.ndarray:
    return np.nan_to_num(np.asarray(audio, dtype=np.float32)).reshape(-1)


def _frame_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """Loudness of each whole 10 ms frame in dBFS."""
    n = sr * FRAME_MS // 1000
    frames = len(audio) // n
    if frames == 0:
        return np.empty(0)
    x = audio[: frames * n].reshape(frames, n).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(x * x, axis=1)), 1e-10))


def trim_silence(audio, sr: int, threshold_db: float = -40.0, pad_ms: int = 100) -> np.ndarray:
    audio = _clean(audio)
    loud = np.flatnonzero(_frame_db(audio, sr) > threshold_db)
    if len(loud) == 0:
        return audio[:0]
    n = sr * FRAME_MS // 1000
    pad = sr * pad_ms // 1000
    return audio[max(0, int(loud[0]) * n - pad) : min(len(audio), (int(loud[-1]) + 1) * n + pad)]


def level_problem(audio: np.ndarray) -> str | None:
    if len(audio) == 0 or float(np.max(np.abs(audio))) < QUIET_PEAK:
        return "too_quiet"
    clipped = int(np.count_nonzero(np.abs(audio) >= CLIP_LEVEL))
    if clipped > max(8, int(len(audio) * 0.0002)):
        return "clipped"
    return None


def clip_problem(audio: np.ndarray, sr: int) -> str | None:
    """Why an already trimmed take is unusable, or None when it is fine."""
    if len(audio) == 0:
        return "too_quiet"
    if len(audio) < MIN_SECONDS * sr:
        return "too_short"
    if len(audio) > MAX_RECORD_SECONDS * sr:
        return "too_long"
    return level_problem(audio)


def split_on_silence(audio, sr: int, max_seconds: float = 15.0, min_seconds: float = 1.0,
                     threshold_db: float = -40.0, min_silence_ms: int = 300, pad_ms: int = 100) -> list[np.ndarray]:
    """Cut audio into utterances at pauses. An utterance longer than max_seconds is cut at its
    quietest moment in the last 3 seconds of each window. Pieces under min_seconds are dropped."""
    audio = _clean(audio)
    db = _frame_db(audio, sr)
    n = sr * FRAME_MS // 1000
    gap = max(1, min_silence_ms // FRAME_MS)

    regions: list[tuple[int, int]] = []
    start = last = None
    for i in np.flatnonzero(db > threshold_db):
        i = int(i)
        if start is None:
            start = i
        elif i - last - 1 >= gap:
            regions.append((start, last + 1))
            start = i
        last = i
    if start is not None:
        regions.append((start, last + 1))

    pad = sr * pad_ms // 1000
    limit_frames = (int(max_seconds * sr) - 2 * pad) // n  # leaves room for padding at both ends
    search = min(limit_frames - 1, 3000 // FRAME_MS)

    pieces: list[np.ndarray] = []
    for first, end in regions:
        cuts = [first]
        while end - cuts[-1] > limit_frames:
            window_end = cuts[-1] + limit_frames
            lo = window_end - search
            cuts.append(lo + int(np.argmin(db[lo:window_end])))
        cuts.append(end)
        for k in range(len(cuts) - 1):
            a = cuts[k] * n - (pad if k == 0 else 0)
            b = cuts[k + 1] * n + (pad if k == len(cuts) - 2 else 0)
            piece = audio[max(0, a) : min(len(audio), b)]
            if len(piece) >= min_seconds * sr:
                pieces.append(piece)
    return pieces
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_dataset.py -q`
Expected: 18 passed. If `test_long_utterance_is_cut_at_a_quiet_moment` fails because the quiet stretch falls outside the 3 second search window, reason through the arithmetic (window end is at 14.8 s, search starts at 11.8 s, the quiet stretch is at 13.0 to 13.2 s) and fix the code, not the test.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/dataset.py backend/tests/test_dataset.py
git commit -m "feat: silence trimming, level checks, and splitting for clips"
```

---

### Task 4: Voice library on disk

**Files:**
- Create: `backend/aivoice/library.py`
- Test: `backend/tests/test_library.py`

**Interfaces:**
- Consumes: `audiofile.write_wav`, `audiofile.read_wav`.
- Produces: `LibraryError(ValueError)`; `Clip(index: int, text: str, seconds: float, source: str, prompt: int | None = None, origin: str | None = None)`; `Voice(id: str, name: str, created: str, clips: list[Clip])` with property `minutes: float`; `Library(data: Path)` with `list() -> list[Voice]`, `create(name: str) -> Voice`, `get(voice_id: str) -> Voice`, `delete(voice_id: str) -> None`, `add_clip(voice_id, audio: np.ndarray, text: str, source: str, prompt: int | None = None, origin: str | None = None) -> Clip` (audio is mono float32 at 48000 Hz), `update_text(voice_id, index: int, text: str) -> Clip`, `delete_clip(voice_id, index: int) -> None`, `clip_audio_path(voice_id, index: int) -> Path` (the 48 kHz file). `source` is `"recorded"` or `"imported"`. Clip indices start at 1 and are never reused within a voice.

- [ ] **Step 1: Write the failing tests**

```python
import json
import re

import numpy as np
import pytest

from aivoice.audiofile import read_wav
from aivoice.library import Library, LibraryError

SR = 48000


def audio(seconds=2.0):
    return np.random.default_rng(1).uniform(-0.3, 0.3, int(seconds * SR)).astype(np.float32)


def test_empty_library(tmp_path):
    assert Library(tmp_path).list() == []


def test_create_and_list(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("My Voice")
    assert voice.name == "My Voice" and voice.clips == [] and voice.minutes == 0
    assert re.fullmatch(r"my-voice-[0-9a-f]{6}", voice.id)
    assert [v.id for v in lib.list()] == [voice.id]
    assert (tmp_path / "voices" / voice.id / "clips").is_dir()


@pytest.mark.parametrize("name", ["../../etc", "a/b\\c", "🎤 Mic", "...", "   spaced   ", "Ünïcödé", "CON"])
def test_awkward_names_get_safe_ids_and_keep_their_display_name(tmp_path, name):
    lib = Library(tmp_path)
    voice = lib.create(name)
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*-[0-9a-f]{6}", voice.id)
    assert voice.name == name.strip()
    assert (tmp_path / "voices" / voice.id).parent == tmp_path / "voices"
    assert lib.get(voice.id).name == name.strip()


@pytest.mark.parametrize("name", ["", "   ", "x" * 61, "line\nbreak"])
def test_invalid_names_are_rejected(tmp_path, name):
    with pytest.raises(LibraryError):
        Library(tmp_path).create(name)


def test_duplicate_names_get_distinct_ids(tmp_path):
    lib = Library(tmp_path)
    a, b = lib.create("Same"), lib.create("Same")
    assert a.id != b.id and len(lib.list()) == 2


@pytest.mark.parametrize("bad", ["", "..", "../x", "a/b", "A", "nope-000000", None])
def test_unknown_or_unsafe_ids_are_rejected(tmp_path, bad):
    lib = Library(tmp_path)
    lib.create("Real")
    with pytest.raises(LibraryError):
        lib.get(bad)


def test_add_clip_writes_three_files_and_records_it(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    clip = lib.add_clip(voice.id, audio(2.0), "Hello there.", "recorded", prompt=4)
    assert (clip.index, clip.text, clip.source, clip.prompt, clip.origin) == (1, "Hello there.", "recorded", 4, None)
    assert clip.seconds == pytest.approx(2.0, abs=0.01)
    folder = tmp_path / "voices" / voice.id / "clips"
    assert sorted(p.name for p in folder.iterdir()) == ["0001.orig.wav", "0001.txt", "0001.wav"]
    orig, sr_orig = read_wav(folder / "0001.orig.wav")
    small, sr_small = read_wav(folder / "0001.wav")
    assert sr_orig == 48000 and abs(len(orig) - 96000) <= 2
    assert sr_small == 16000 and abs(len(small) - 32000) <= 2
    assert (folder / "0001.txt").read_text(encoding="utf-8") == "Hello there."
    assert lib.get(voice.id).clips == [clip]
    assert lib.get(voice.id).minutes == pytest.approx(2.0 / 60, abs=0.01)


def test_indices_are_never_reused(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "one", "recorded")
    second = lib.add_clip(voice.id, audio(), "two", "recorded")
    lib.delete_clip(voice.id, second.index)
    third = lib.add_clip(voice.id, audio(), "three", "imported", origin="file.mp3")
    assert third.index == 3 and third.origin == "file.mp3"
    assert [c.index for c in lib.get(voice.id).clips] == [1, 3]


def test_update_text(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "wrong", "imported")
    clip = lib.update_text(voice.id, 1, "  right  ")
    assert clip.text == "right"
    assert (tmp_path / "voices" / voice.id / "clips" / "0001.txt").read_text(encoding="utf-8") == "right"
    assert lib.get(voice.id).clips[0].text == "right"


def test_update_text_rejects_overlong_text_and_unknown_clip(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "imported")
    with pytest.raises(LibraryError):
        lib.update_text(voice.id, 1, "y" * 501)
    with pytest.raises(LibraryError):
        lib.update_text(voice.id, 9, "y")


def test_delete_clip_removes_files(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    lib.delete_clip(voice.id, 1)
    assert list((tmp_path / "voices" / voice.id / "clips").iterdir()) == []
    assert lib.get(voice.id).clips == []
    with pytest.raises(LibraryError):
        lib.delete_clip(voice.id, 1)


def test_clip_audio_path(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    assert lib.clip_audio_path(voice.id, 1).name == "0001.orig.wav"
    with pytest.raises(LibraryError):
        lib.clip_audio_path(voice.id, 2)


def test_delete_voice_moves_it_to_trash(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(), "x", "recorded")
    lib.delete(voice.id)
    assert lib.list() == []
    trashed = list((tmp_path / "trash").iterdir())
    assert len(trashed) == 1 and trashed[0].name.startswith(voice.id)
    assert (trashed[0] / "clips" / "0001.orig.wav").is_file()


def test_summary_manifest_is_written(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    lib.add_clip(voice.id, audio(60.0), "x", "recorded")
    manifest = json.loads((tmp_path / "voices.json").read_text(encoding="utf-8"))
    assert manifest == [{"id": voice.id, "name": "V", "created": voice.created, "minutes": 1.0, "clips": 1, "models": []}]


def test_a_corrupt_voice_file_hides_only_that_voice_and_deletes_nothing(tmp_path):
    lib = Library(tmp_path)
    good, bad = lib.create("Good"), lib.create("Bad")
    lib.add_clip(bad.id, audio(), "x", "recorded")
    (tmp_path / "voices" / bad.id / "voice.json").write_text("{half written")
    assert [v.id for v in Library(tmp_path).list()] == [good.id]
    assert (tmp_path / "voices" / bad.id / "clips" / "0001.orig.wav").is_file()


def test_leftover_files_from_an_interrupted_save_do_not_block_the_next_save(tmp_path):
    lib = Library(tmp_path)
    voice = lib.create("V")
    folder = tmp_path / "voices" / voice.id / "clips"
    (folder / "0001.orig.wav").write_bytes(b"junk")
    (folder / "0001.wav.tmp").write_bytes(b"junk")
    clip = lib.add_clip(voice.id, audio(), "x", "recorded")
    assert clip.index == 1
    _, sr = read_wav(folder / "0001.orig.wav")
    assert sr == 48000


def test_a_stray_folder_without_voice_json_is_ignored(tmp_path):
    (tmp_path / "voices" / "stray-folder").mkdir(parents=True)
    assert Library(tmp_path).list() == []


def test_voices_are_listed_oldest_first(tmp_path):
    lib = Library(tmp_path)
    ids = [lib.create(n).id for n in ("b", "a", "c")]
    assert [v.id for v in lib.list()] == ids
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_library.py -q`
Expected: FAIL with `ModuleNotFoundError: aivoice.library`.

- [ ] **Step 3: Implement `backend/aivoice/library.py`**

```python
import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soxr

from aivoice.audiofile import write_wav

SAMPLE_RATE = 48000
TRAIN_RATE = 16000
MAX_NAME = 60
MAX_TEXT = 500
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class LibraryError(ValueError):
    pass


@dataclass
class Clip:
    index: int
    text: str
    seconds: float
    source: str
    prompt: int | None = None
    origin: str | None = None


@dataclass
class Voice:
    id: str
    name: str
    created: str
    clips: list[Clip] = field(default_factory=list)

    @property
    def minutes(self) -> float:
        return round(sum(c.seconds for c in self.clips) / 60.0, 2)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class Library:
    """Voices and their clips on disk. Each voice's voice.json is the source of truth."""

    def __init__(self, data: Path):
        self._data = data
        self._root = data / "voices"
        self._lock = threading.RLock()

    def _dir(self, voice_id) -> Path:
        if not isinstance(voice_id, str) or not ID_RE.match(voice_id):
            raise LibraryError("Unknown voice")
        return self._root / voice_id

    def _load(self, voice_id) -> Voice:
        folder = self._dir(voice_id)
        try:
            raw = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
            return Voice(voice_id, str(raw["name"]), str(raw["created"]), [Clip(**c) for c in raw["clips"]])
        except (OSError, ValueError, KeyError, TypeError):
            raise LibraryError("Unknown voice") from None

    def _save(self, voice: Voice) -> None:
        _atomic_json(self._dir(voice.id) / "voice.json",
                     {"name": voice.name, "created": voice.created, "clips": [asdict(c) for c in voice.clips]})
        self._write_summary()

    def _write_summary(self) -> None:
        self._data.mkdir(parents=True, exist_ok=True)
        _atomic_json(self._data / "voices.json", [
            {"id": v.id, "name": v.name, "created": v.created, "minutes": v.minutes, "clips": len(v.clips), "models": []}
            for v in self.list()
        ])

    def list(self) -> list[Voice]:
        with self._lock:
            if not self._root.is_dir():
                return []
            voices = []
            for folder in self._root.iterdir():
                try:
                    voices.append(self._load(folder.name))
                except LibraryError:
                    continue  # damaged or foreign folder: leave it alone
            return sorted(voices, key=lambda v: (v.created, v.id))

    def get(self, voice_id) -> Voice:
        with self._lock:
            return self._load(voice_id)

    def create(self, name: str) -> Voice:
        name = (name or "").strip()
        if not name or len(name) > MAX_NAME or any(ord(ch) < 32 for ch in name):
            raise LibraryError(f"A voice name must be 1 to {MAX_NAME} characters on one line")
        with self._lock:
            voice_id = f"{_slug(name)}-{secrets.token_hex(3)}"
            (self._root / voice_id / "clips").mkdir(parents=True)
            created = time.strftime("%Y-%m-%dT%H:%M:%S") + f".{time.time_ns() % 1_000_000_000:09d}"
            voice = Voice(voice_id, name, created)
            self._save(voice)
            return voice

    def delete(self, voice_id) -> None:
        """Moves the voice to data/trash. Nothing is erased."""
        with self._lock:
            self._load(voice_id)
            trash = self._data / "trash"
            trash.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self._dir(voice_id)), str(trash / f"{voice_id}-{int(time.time())}"))
            self._write_summary()

    def add_clip(self, voice_id, audio: np.ndarray, text: str, source: str,
                 prompt: int | None = None, origin: str | None = None) -> Clip:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        with self._lock:
            voice = self._load(voice_id)
            index = max((c.index for c in voice.clips), default=0) + 1
            clips = self._dir(voice_id) / "clips"
            clips.mkdir(exist_ok=True)
            stem = f"{index:04d}"
            # Audio first, record last: a crash leaves stray files, which the next save overwrites.
            write_wav(clips / f"{stem}.orig.wav", audio, SAMPLE_RATE)
            write_wav(clips / f"{stem}.wav", soxr.resample(audio, SAMPLE_RATE, TRAIN_RATE), TRAIN_RATE)
            text = (text or "").strip()[:MAX_TEXT]
            (clips / f"{stem}.txt").write_text(text, encoding="utf-8")
            clip = Clip(index, text, round(len(audio) / SAMPLE_RATE, 2), source, prompt, origin)
            voice.clips.append(clip)
            self._save(voice)
            return clip

    def _find(self, voice: Voice, index: int) -> Clip:
        clip = next((c for c in voice.clips if c.index == index), None)
        if clip is None:
            raise LibraryError("Unknown clip")
        return clip

    def update_text(self, voice_id, index: int, text: str) -> Clip:
        text = (text or "").strip()
        if len(text) > MAX_TEXT:
            raise LibraryError(f"A transcript can be at most {MAX_TEXT} characters")
        with self._lock:
            voice = self._load(voice_id)
            clip = self._find(voice, index)
            clip.text = text
            (self._dir(voice_id) / "clips" / f"{index:04d}.txt").write_text(text, encoding="utf-8")
            self._save(voice)
            return clip

    def delete_clip(self, voice_id, index: int) -> None:
        with self._lock:
            voice = self._load(voice_id)
            voice.clips.remove(self._find(voice, index))
            self._save(voice)
            for suffix in (".orig.wav", ".wav", ".txt"):
                (self._dir(voice_id) / "clips" / f"{index:04d}{suffix}").unlink(missing_ok=True)

    def clip_audio_path(self, voice_id, index: int) -> Path:
        with self._lock:
            self._find(self._load(voice_id), index)
            path = self._dir(voice_id) / "clips" / f"{index:04d}.orig.wav"
            if not path.is_file():
                raise LibraryError("The audio for this clip is missing")
            return path
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_library.py -q`
Expected: all passed (36 with parametrisation).

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/library.py backend/tests/test_library.py
git commit -m "feat: voice library with clips stored on disk"
```

---

### Task 5: Sentence prompts

**Files:**
- Create: `backend/aivoice/prompts_en.txt`, `backend/aivoice/prompts.py`
- Modify: `backend/pyproject.toml` (ship the text file with the package)
- Test: `backend/tests/test_prompts.py`

**Interfaces:**
- Produces: `load_prompts() -> tuple[str, ...]` (200 sentences, cached); `next_prompt(recorded: set[int], total: int, after: int | None = None) -> int | None` returning the next sentence index not yet recorded, starting after `after` and wrapping around, or `None` when every sentence is recorded.

The sentences must be original writing by the implementer, not copied from any published sentence list, and must meet the rules the tests enforce. The first ten lines of the file are fixed and given below so the style is clear.

- [ ] **Step 1: Write the failing tests**

```python
import re
from collections import Counter

from aivoice.prompts import load_prompts, next_prompt

FIRST_TEN = [
    "The quiet river bends just north of the old stone bridge.",
    "She packed a thick wool jumper for the chilly evening hike.",
    "Which of these boxes should go up to the attic today?",
    "A young fox jumped over the fence and vanished in the hedge.",
    "Please measure the garage door before you order a new one.",
    "They usually share a pot of strong tea after the morning shift.",
    "The children enjoyed the noisy parade through the town square.",
    "He fixed the buzzing light with a ladder and a spare fuse.",
    "Your television is far too loud for this time of night.",
    "We watched the purple clouds drift slowly above the harbour.",
]

SOUNDS = ["th", "sh", "ch", "ng", "wh", "oo", "ee", "ou", "ow", "oi", "oy", "ar", "er", "or", "ai", "ay",
          "ea", "igh", "qu", "ck", "j", "v", "z", "x", "sion", "sure", "ure", "au", "aw", "ph"]


def test_two_hundred_unique_sentences():
    prompts = load_prompts()
    assert len(prompts) == 200
    assert len(set(p.lower() for p in prompts)) == 200
    assert list(prompts[:10]) == FIRST_TEN


def test_every_sentence_is_readable_aloud_in_one_breath():
    for p in load_prompts():
        words = p.split()
        assert 6 <= len(words) <= 14, p
        assert len(p) <= 90, p
        assert p[0].isupper() and p[-1] in ".?", p
        assert re.fullmatch(r"[A-Za-z ,.'?-]+", p), p
        assert "  " not in p and p == p.strip(), p


def test_the_set_covers_english_sounds():
    text = " ".join(load_prompts()).lower()
    letters = Counter(ch for ch in text if ch.isalpha())
    assert all(letters[ch] >= 10 for ch in "abcdefghijklmnopqrstuvwxyz"), sorted(letters.items(), key=lambda kv: kv[1])[:5]
    for sound in SOUNDS:
        assert text.count(sound) >= 5, sound


def test_no_word_dominates():
    words = Counter(w.strip(",.?'").lower() for p in load_prompts() for w in p.split())
    common = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "with", "is", "was", "her", "his", "at", "by", "from", "that", "this", "it", "you", "we", "they", "he", "she", "i", "my", "your", "our", "their"}
    worst = max((count for word, count in words.items() if word not in common), default=0)
    assert worst <= 8


def test_about_one_in_ten_is_a_question():
    questions = sum(p.endswith("?") for p in load_prompts())
    assert 10 <= questions <= 40


def test_next_prompt_starts_at_the_first_unrecorded():
    assert next_prompt(set(), 5) == 0
    assert next_prompt({0, 1}, 5) == 2


def test_next_prompt_after_skips_forward_and_wraps():
    assert next_prompt(set(), 5, after=1) == 2
    assert next_prompt({2, 3}, 5, after=1) == 4
    assert next_prompt({0}, 5, after=4) == 1


def test_skipping_the_only_one_left_returns_it_again():
    assert next_prompt({0, 1, 2, 4}, 5, after=3) == 3


def test_next_prompt_none_when_all_recorded():
    assert next_prompt({0, 1, 2}, 3) is None
    assert next_prompt({0, 1, 2}, 3, after=1) is None


def test_out_of_range_after_is_treated_as_start():
    assert next_prompt(set(), 3, after=99) == 0
    assert next_prompt(set(), 3, after=-5) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_prompts.py -q`
Expected: FAIL with `ModuleNotFoundError: aivoice.prompts`.

- [ ] **Step 3: Implement `backend/aivoice/prompts.py`**

```python
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_prompts() -> tuple[str, ...]:
    text = Path(__file__).with_name("prompts_en.txt").read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def next_prompt(recorded: set[int], total: int, after: int | None = None) -> int | None:
    """The next sentence to read: the first one not yet recorded after `after`, wrapping round."""
    if after is None or not 0 <= after < total:
        order = range(total)
    else:
        order = [*range(after + 1, total), *range(0, after + 1)]
    return next((i for i in order if i not in recorded), None)
```

- [ ] **Step 4: Write `backend/aivoice/prompts_en.txt`**

Exactly 200 lines, one sentence per line, no blank lines, no numbering. Lines 1 to 10 are the `FIRST_TEN` sentences above, verbatim. Write the other 190 yourself: everyday, neutral sentences a person can read aloud naturally, varied in rhythm and subject, British or American spelling consistently, no names of real people, no brands, nothing offensive. Run the tests repeatedly and revise sentences until every rule passes; the sound-coverage test tells you which spelling pattern is short.

- [ ] **Step 5: Ship the text file with the package**

In `backend/pyproject.toml`, add after the `[tool.setuptools.packages.find]` block:

```toml
[tool.setuptools.package-data]
aivoice = ["*.txt"]
```

- [ ] **Step 6: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_prompts.py -q`
Expected: 10 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/aivoice/prompts.py backend/aivoice/prompts_en.txt backend/pyproject.toml backend/tests/test_prompts.py
git commit -m "feat: two hundred recording prompts and prompt selection"
```

---

### Task 6: Microphone recorder

**Files:**
- Create: `backend/aivoice/recorder.py`
- Test: `backend/tests/test_recorder.py`

**Interfaces:**
- Consumes: a stream factory with the `SoundDeviceStreams.input(device, samplerate, on_audio)` shape; `config.SAMPLE_RATE`.
- Produces: `RecorderError(ValueError)`; `Recorder(stream_factory, sample_rate: int = SAMPLE_RATE, max_seconds: float = 30.0)` with `start(device: int) -> None`, `stop() -> np.ndarray` (mono float32), `cancel() -> None`, `status() -> dict` with keys `recording: bool`, `level: float`, `seconds: float`, `full: bool`, and property `recording: bool`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from aivoice.recorder import Recorder, RecorderError
from tests.test_live import FakeStream

SR = 48000


class CapturingStreams:
    """Stream factory double that hands the test the recorder's audio callback."""

    def __init__(self, fail=False):
        self.streams = []
        self.on_audio = None
        self.fail = fail

    def input(self, device, samplerate, on_audio):
        if self.fail:
            raise RuntimeError("device busy")
        self.on_audio = on_audio
        stream = FakeStream("in", device)
        self.streams.append(stream)
        return stream


def block(seconds, value=0.25):
    return np.full((int(seconds * SR), 1), value, dtype=np.float32)


def test_records_what_the_microphone_delivers():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(7)
    assert rec.recording and streams.streams[0].device == 7 and streams.streams[0].active
    streams.on_audio(block(0.5, 0.25))
    streams.on_audio(block(0.25, -0.5))
    status = rec.status()
    assert status == {"recording": True, "level": 0.5, "seconds": 0.75, "full": False}
    audio = rec.stop()
    assert audio.dtype == np.float32 and audio.shape == (int(0.75 * SR),)
    assert audio[0] == 0.25 and audio[-1] == -0.5
    assert not rec.recording and streams.streams[0].closed


def test_stereo_blocks_are_downmixed():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(1)
    streams.on_audio(np.stack([np.ones(480), -np.ones(480)], axis=1).astype(np.float32))
    assert np.allclose(rec.stop(), 0.0)


def test_stop_without_audio_returns_empty():
    rec = Recorder(CapturingStreams())
    rec.start(1)
    assert rec.stop().shape == (0,)


def test_recording_stops_growing_at_the_limit():
    streams = CapturingStreams()
    rec = Recorder(streams, max_seconds=1.0)
    rec.start(1)
    for _ in range(5):
        streams.on_audio(block(0.5))
    assert rec.status()["full"] is True
    assert len(rec.stop()) <= int(1.5 * SR)


def test_start_twice_is_refused():
    rec = Recorder(CapturingStreams())
    rec.start(1)
    with pytest.raises(RecorderError, match="Already recording"):
        rec.start(1)
    rec.cancel()


def test_stop_when_idle_is_refused():
    with pytest.raises(RecorderError, match="Not recording"):
        Recorder(CapturingStreams()).stop()


def test_cancel_discards_and_is_safe_when_idle():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.cancel()
    rec.start(1)
    streams.on_audio(block(0.5))
    rec.cancel()
    assert not rec.recording and streams.streams[0].closed
    assert rec.status() == {"recording": False, "level": 0.0, "seconds": 0.0, "full": False}


def test_a_device_that_will_not_open_is_a_clear_error():
    rec = Recorder(CapturingStreams(fail=True))
    with pytest.raises(RecorderError, match="Could not open the microphone"):
        rec.start(1)
    assert not rec.recording


def test_late_callback_after_stop_is_ignored():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(1)
    rec.stop()
    streams.on_audio(block(0.5))
    assert rec.status()["seconds"] == 0.0


def test_bad_block_from_the_driver_does_not_raise():
    streams = CapturingStreams()
    rec = Recorder(streams)
    rec.start(1)
    streams.on_audio(None)
    streams.on_audio(block(0.1))
    assert len(rec.stop()) == int(0.1 * SR)
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_recorder.py -q`
Expected: FAIL with `ModuleNotFoundError: aivoice.recorder`.

- [ ] **Step 3: Implement `backend/aivoice/recorder.py`**

```python
import threading

import numpy as np

from aivoice.config import SAMPLE_RATE


class RecorderError(ValueError):
    pass


class Recorder:
    """Captures the microphone into memory. One take at a time."""

    def __init__(self, stream_factory, sample_rate: int = SAMPLE_RATE, max_seconds: float = 30.0):
        self._factory = stream_factory
        self._sr = sample_rate
        self._max = int(max_seconds * sample_rate)
        self._lock = threading.Lock()        # start / stop / cancel
        self._data_lock = threading.Lock()   # blocks shared with the audio callback
        self._stream = None
        self._blocks: list[np.ndarray] = []
        self._samples = 0
        self._level = 0.0

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def _on_audio(self, indata) -> None:
        try:
            mono = indata.reshape(-1) if indata.ndim == 1 or indata.shape[1] == 1 else indata.mean(axis=1)
            with self._data_lock:
                if self._stream is None:
                    return
                self._level = float(np.max(np.abs(mono))) if len(mono) else 0.0
                if self._samples < self._max:
                    self._blocks.append(mono.astype(np.float32, copy=True))
                    self._samples += len(mono)
        except Exception:  # noqa: BLE001 - never raise into the audio driver thread
            pass

    def _reset(self) -> None:
        with self._data_lock:
            self._blocks = []
            self._samples = 0
            self._level = 0.0

    def _close(self) -> None:
        stream = self._stream
        with self._data_lock:
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - an unplugged device must not block stopping
                pass

    def start(self, device: int) -> None:
        with self._lock:
            if self._stream is not None:
                raise RecorderError("Already recording")
            self._reset()
            try:
                stream = self._factory.input(device, self._sr, self._on_audio)
                with self._data_lock:
                    self._stream = stream
                stream.start()
            except Exception as exc:  # noqa: BLE001
                self._close()
                raise RecorderError(f"Could not open the microphone: {exc}") from exc

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._stream is None:
                raise RecorderError("Not recording")
            self._close()
            with self._data_lock:
                audio = np.concatenate(self._blocks) if self._blocks else np.zeros(0, dtype=np.float32)
            self._reset()
            return audio.astype(np.float32)

    def cancel(self) -> None:
        with self._lock:
            self._close()
            self._reset()

    def status(self) -> dict:
        with self._data_lock:
            return {"recording": self._stream is not None, "level": self._level,
                    "seconds": round(self._samples / self._sr, 2), "full": self._samples >= self._max}
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_recorder.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/recorder.py backend/tests/test_recorder.py
git commit -m "feat: microphone recorder with level and duration"
```

---

### Task 7: Transcription and the import job

**Files:**
- Modify: `backend/aivoice/config.py`
- Create: `backend/aivoice/transcribe.py`, `backend/aivoice/importer.py`
- Test: `backend/tests/test_transcribe.py`, `backend/tests/test_importer.py`
- Modify: `backend/tests/test_config.py` (add tests)

**Interfaces:**
- Consumes: `audiofile.read_audio`, `audiofile.AudioFileError`, `dataset.split_on_silence`, `dataset.level_problem`, `library.Library`.
- Produces: `config.whisper_model(data: Path) -> str` (default `"small.en"`, raises `ValueError` for a value that is not a plain model name). `transcribe.WhisperTranscriber(model_name: Callable[[], str], download_root: Path, model_factory=None)` with `transcribe(audio16k: np.ndarray) -> str`. `importer.AUDIO_EXTENSIONS`; `importer.Importer(library, transcriber, read_audio=read_audio)` with `start(voice_id: str, paths: list[str]) -> None`, `cancel() -> None`, `wait(timeout: float | None = None) -> None`, `status() -> dict` with keys `active: bool`, `finished: bool`, `total_files: int`, `done_files: int`, `current: str | None`, `clips_added: int`, `skipped: int`, `errors: list[str]`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_config.py`:

```python
def test_whisper_model_default(tmp_path):
    assert config.whisper_model(tmp_path) == "small.en"


def test_whisper_model_from_settings(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"whisper_model": "base.en"}))
    assert config.whisper_model(tmp_path) == "base.en"


@pytest.mark.parametrize("bad", [5, "", "../evil", "a b", None, "x" * 80])
def test_whisper_model_rejects_odd_values(tmp_path, bad):
    (tmp_path / "settings.json").write_text(json.dumps({"whisper_model": bad}))
    with pytest.raises(ValueError, match="whisper_model"):
        config.whisper_model(tmp_path)
```

Add `import pytest` at the top of `test_config.py` if it is not already imported.

`backend/tests/test_transcribe.py`:

```python
import numpy as np

from aivoice.transcribe import WhisperTranscriber


class Segment:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([Segment(" Hello there. "), Segment("How are you?")]), object()


def make(name="small.en"):
    built = []
    model = FakeModel()

    def factory(model_name, root):
        built.append((model_name, root))
        return model

    return WhisperTranscriber(lambda: name, "root", model_factory=factory), model, built


def test_joins_segments_into_one_clean_line():
    transcriber, _, _ = make()
    assert transcriber.transcribe(np.zeros(16000, dtype=np.float32)) == "Hello there. How are you?"


def test_model_is_loaded_lazily_and_once():
    transcriber, _, built = make()
    assert built == []
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    assert built == [("small.en", "root")]


def test_english_models_force_english_and_others_detect():
    transcriber, model, _ = make("small.en")
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    assert model.calls[0][1]["language"] == "en"
    transcriber, model, _ = make("small")
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    assert model.calls[0][1]["language"] is None


def test_audio_is_passed_as_float32():
    transcriber, model, _ = make()
    transcriber.transcribe(np.zeros(100, dtype=np.float64))
    assert model.calls[0][0].dtype == np.float32
```

`backend/tests/test_importer.py`:

```python
import numpy as np
import pytest

from aivoice.audiofile import AudioFileError, write_wav
from aivoice.importer import AUDIO_EXTENSIONS, Importer
from aivoice.library import Library, LibraryError

SR = 48000
RNG = np.random.default_rng(3)


def noise(seconds, amp=0.3):
    return RNG.uniform(-amp, amp, int(seconds * SR)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


class FakeTranscriber:
    def __init__(self, fail=False):
        self.count = 0
        self.fail = fail

    def transcribe(self, audio16k):
        if self.fail:
            raise RuntimeError("model download failed")
        self.count += 1
        return f"line {self.count}"


def setup(tmp_path, transcriber=None):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")
    return lib, voice, Importer(lib, transcriber or FakeTranscriber())


def wav(tmp_path, name, audio):
    path = tmp_path / name
    write_wav(path, audio, SR)
    return str(path)


def run(importer, voice, paths):
    importer.start(voice.id, paths)
    importer.wait(10)
    return importer.status()


def test_imports_each_utterance_as_a_transcribed_clip(tmp_path):
    lib, voice, importer = setup(tmp_path)
    path = wav(tmp_path, "talk.wav", np.concatenate([noise(2.0), silence(0.6), noise(3.0)]))
    status = run(importer, voice, [path])
    assert status == {"active": False, "finished": True, "total_files": 1, "done_files": 1, "current": None,
                      "clips_added": 2, "skipped": 0, "errors": []}
    clips = lib.get(voice.id).clips
    assert [(c.text, c.source, c.origin) for c in clips] == [("line 1", "imported", "talk.wav"), ("line 2", "imported", "talk.wav")]


def test_bad_files_are_reported_by_name_and_the_rest_still_import(tmp_path):
    lib, voice, importer = setup(tmp_path)
    fake = tmp_path / "notes.mp3"
    fake.write_text("not audio")
    folder = tmp_path / "folder.wav"
    folder.mkdir()
    doc = tmp_path / "essay.docx"
    doc.write_text("words")
    good = wav(tmp_path, "good.wav", noise(2.0))
    status = run(importer, voice, [str(fake), str(folder), str(doc), str(tmp_path / "gone.wav"), good])
    assert status["clips_added"] == 1 and status["done_files"] == 5 and status["finished"]
    joined = " | ".join(status["errors"])
    assert len(status["errors"]) == 4
    for name in ("notes.mp3", "folder.wav", "essay.docx", "gone.wav"):
        assert name in joined
    assert len(lib.get(voice.id).clips) == 1


def test_a_file_with_no_speech_is_reported(tmp_path):
    _, voice, importer = setup(tmp_path)
    status = run(importer, voice, [wav(tmp_path, "quiet.wav", silence(3.0))])
    assert status["clips_added"] == 0
    assert status["errors"] == ["quiet.wav: no speech was found"]


def test_too_quiet_pieces_are_skipped_and_counted(tmp_path):
    _, voice, importer = setup(tmp_path)
    audio = np.concatenate([noise(2.0, amp=0.02), silence(0.6), noise(2.0)])
    status = run(importer, voice, [wav(tmp_path, "mixed.wav", audio)])
    assert (status["clips_added"], status["skipped"]) == (1, 1)


def test_transcription_failure_stops_the_job_with_one_clear_error(tmp_path):
    lib, voice, importer = setup(tmp_path, FakeTranscriber(fail=True))
    paths = [wav(tmp_path, "a.wav", noise(2.0)), wav(tmp_path, "b.wav", noise(2.0))]
    status = run(importer, voice, paths)
    assert status["finished"] and status["clips_added"] == 0
    assert status["errors"] == ["Transcription failed: model download failed"]
    assert lib.get(voice.id).clips == []


def test_unknown_voice_and_empty_or_oversized_lists_are_refused(tmp_path):
    _, voice, importer = setup(tmp_path)
    with pytest.raises(LibraryError):
        importer.start("nope-000000", ["a.wav"])
    with pytest.raises(ValueError, match="No files"):
        importer.start(voice.id, [])
    with pytest.raises(ValueError, match="at most"):
        importer.start(voice.id, ["a.wav"] * 201)


def test_a_second_import_is_refused_while_one_runs(tmp_path):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")
    import threading

    gate = threading.Event()

    def slow_read(path, sample_rate=SR):
        gate.wait(5)
        return noise(2.0)

    importer = Importer(lib, FakeTranscriber(), read_audio=slow_read)
    real = wav(tmp_path, "a.wav", noise(2.0))
    importer.start(voice.id, [real])
    assert importer.status()["active"] is True
    with pytest.raises(ValueError, match="already running"):
        importer.start(voice.id, [real])
    gate.set()
    importer.wait(10)
    assert importer.status()["clips_added"] == 1


def test_cancel_stops_before_the_next_file(tmp_path):
    lib = Library(tmp_path / "data")
    voice = lib.create("V")
    importer_box = {}

    def read_then_cancel(path, sample_rate=SR):
        importer_box["importer"].cancel()
        return noise(2.0)

    importer = Importer(lib, FakeTranscriber(), read_audio=read_then_cancel)
    importer_box["importer"] = importer
    paths = [wav(tmp_path, f"{n}.wav", noise(2.0)) for n in "abc"]
    status = run(importer, voice, paths)
    assert status["finished"] and status["done_files"] < 3
    assert status["clips_added"] == 0


def test_status_before_any_import():
    lib = Library.__new__(Library)
    assert Importer(lib, FakeTranscriber()).status() == {
        "active": False, "finished": False, "total_files": 0, "done_files": 0, "current": None,
        "clips_added": 0, "skipped": 0, "errors": []}


def test_common_formats_are_accepted():
    assert {".wav", ".mp3", ".flac", ".m4a", ".ogg"} <= AUDIO_EXTENSIONS
    assert AudioFileError
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_config.py backend/tests/test_transcribe.py backend/tests/test_importer.py -q`
Expected: FAIL with `ModuleNotFoundError` for `aivoice.transcribe` and `aivoice.importer`, and `AttributeError` for `config.whisper_model`.

- [ ] **Step 3: Implement**

Append to `backend/aivoice/config.py` (add `import re` at the top):

```python
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def whisper_model(data: Path) -> str:
    """The Whisper model used to transcribe imported audio. Raises ValueError for an odd value."""
    raw = load_settings(data).get("whisper_model", "small.en")
    if not isinstance(raw, str) or not _MODEL_NAME.match(raw):
        raise ValueError("whisper_model in settings.json must be a model name such as small.en")
    return raw
```

`backend/aivoice/transcribe.py`:

```python
from pathlib import Path
from typing import Callable

import numpy as np


def _load_whisper(name: str, root):
    from faster_whisper import WhisperModel

    # CPU with 8-bit weights: CTranslate2 has no AMD GPU backend, and this is not time critical.
    return WhisperModel(name, device="cpu", compute_type="int8", download_root=str(root))


class WhisperTranscriber:
    """Speech to text for imported clips. The model loads on first use (and downloads once)."""

    def __init__(self, model_name: Callable[[], str], download_root: Path | str, model_factory=None):
        self._name = model_name
        self._root = download_root
        self._factory = model_factory or _load_whisper
        self._model = None
        self._language: str | None = None

    def transcribe(self, audio16k: np.ndarray) -> str:
        if self._model is None:
            name = self._name()
            self._model = self._factory(name, self._root)
            self._language = "en" if name.endswith(".en") else None
        segments, _ = self._model.transcribe(
            np.asarray(audio16k, dtype=np.float32), language=self._language, beam_size=5,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
```

`backend/aivoice/importer.py`:

```python
import threading
from pathlib import Path

import soxr

from aivoice.audiofile import AudioFileError, read_audio
from aivoice.dataset import level_problem, split_on_silence
from aivoice.library import SAMPLE_RATE, TRAIN_RATE, Library

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}
MAX_FILES = 200


class _TranscriptionFailed(Exception):
    pass


def _idle() -> dict:
    return {"active": False, "finished": False, "total_files": 0, "done_files": 0, "current": None,
            "clips_added": 0, "skipped": 0, "errors": []}


class Importer:
    """Turns dropped audio files into transcribed clips on a background thread. One job at a time."""

    def __init__(self, library: Library, transcriber, read_audio=read_audio):
        self._library = library
        self._transcriber = transcriber
        self._read = read_audio
        self._lock = threading.Lock()
        self._status = _idle()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def start(self, voice_id: str, paths: list[str]) -> None:
        self._library.get(voice_id)
        if not paths:
            raise ValueError("No files to import")
        if len(paths) > MAX_FILES:
            raise ValueError(f"Import at most {MAX_FILES} files at a time")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("An import is already running")
            self._cancel.clear()
            self._status = {**_idle(), "active": True, "total_files": len(paths)}
            self._thread = threading.Thread(target=self._run, args=(voice_id, list(paths)), daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "errors": list(self._status["errors"])}

    def _set(self, **changes) -> None:
        with self._lock:
            self._status.update(changes)

    def _bump(self, key: str) -> None:
        with self._lock:
            self._status[key] += 1

    def _error(self, message: str) -> None:
        with self._lock:
            self._status["errors"].append(message)

    def _import_file(self, voice_id: str, path: Path) -> None:
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise AudioFileError(f"{path.name} is not a supported audio file")
        if not path.is_file():
            raise AudioFileError(f"{path.name} was not found or is not a file")
        pieces = split_on_silence(self._read(path, SAMPLE_RATE), SAMPLE_RATE)
        if not pieces:
            raise AudioFileError(f"{path.name}: no speech was found")
        for piece in pieces:
            if self._cancel.is_set():
                return
            if level_problem(piece) == "too_quiet":
                self._bump("skipped")
                continue
            try:
                text = self._transcriber.transcribe(soxr.resample(piece, SAMPLE_RATE, TRAIN_RATE))
            except Exception as exc:  # noqa: BLE001
                raise _TranscriptionFailed(str(exc)) from exc
            self._library.add_clip(voice_id, piece, text, "imported", origin=path.name)
            self._bump("clips_added")

    def _run(self, voice_id: str, paths: list[str]) -> None:
        try:
            for raw in paths:
                if self._cancel.is_set():
                    break
                path = Path(raw)
                self._set(current=path.name)
                try:
                    self._import_file(voice_id, path)
                except _TranscriptionFailed as exc:
                    self._error(f"Transcription failed: {exc}")
                    break
                except AudioFileError as exc:
                    self._error(str(exc))
                except Exception as exc:  # noqa: BLE001 - one bad file must not end the job
                    self._error(f"{path.name}: {exc}")
                self._bump("done_files")
        finally:
            self._set(active=False, finished=True, current=None)
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_config.py backend/tests/test_transcribe.py backend/tests/test_importer.py -q`
Expected: all passed. In `test_bad_files_are_reported_by_name_and_the_rest_still_import`, the missing file and the folder are both caught by the `is_file` check and the `.docx` by the extension check; `notes.mp3` reaches the real decoder and fails there.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/config.py backend/aivoice/transcribe.py backend/aivoice/importer.py backend/tests/test_config.py backend/tests/test_transcribe.py backend/tests/test_importer.py
git commit -m "feat: Whisper transcription and background audio import"
```

---

### Task 8: Studio coordinator and the library API

**Files:**
- Create: `backend/aivoice/studio.py`, `backend/aivoice/api_library.py`
- Modify: `backend/aivoice/api.py`, `backend/aivoice/__main__.py`
- Test: `backend/tests/test_studio.py`, `backend/tests/test_api_library.py`

**Interfaces:**
- Consumes: `Library`, `Clip`, `Recorder`, `Importer`, `load_prompts`, `next_prompt`, `trim_silence`, `clip_problem`, `PROBLEM_MESSAGES`, `devices.get_device`, `LiveManager.stats`.
- Produces: `studio.Studio(library, recorder, importer, list_devices, prompts: tuple[str, ...] | None = None, sample_rate: int = 48000)` with property `recording: bool` and methods `prompt(voice_id, after: int | None = None) -> dict` (`{"index": int | None, "text": str | None, "total": int, "recorded": int}`), `start_recording(voice_id, device: int, prompt: int) -> None`, `stop_recording() -> dict` (`{"accepted": bool, "reason": str | None, "message": str | None, "clip": dict | None}`), `cancel_recording() -> None`, `status() -> dict` (the recorder status plus `"import": <importer status>`). `api_library.library_router(studio, library, importer, live_running: Callable[[], bool]) -> APIRouter`. `api.create_app(..., studio=None, library=None, importer=None)`: when `studio` is given the `/library` routes are mounted, `POST /live/start` is refused while recording, and `POST /devices/refresh` is refused while recording.
- Routes (all under `/library`):
  - `GET /voices` returns `[{"id", "name", "created", "minutes", "clips"}]`
  - `POST /voices` body `{"name"}` returns one such object
  - `DELETE /voices/{voice_id}` returns `{"deleted": true}`
  - `GET /voices/{voice_id}/clips` returns `[{"index", "text", "seconds", "source", "prompt", "origin"}]`
  - `PATCH /voices/{voice_id}/clips/{index}` body `{"text"}` returns the clip
  - `DELETE /voices/{voice_id}/clips/{index}` returns `{"deleted": true}`
  - `GET /voices/{voice_id}/clips/{index}/audio` returns the 48 kHz WAV
  - `GET /voices/{voice_id}/prompt?after=N` returns the prompt object
  - `POST /record/start` body `{"voice_id", "input_device", "prompt"}` returns status
  - `POST /record/stop` returns the take result
  - `POST /record/cancel` returns status
  - `POST /voices/{voice_id}/import` body `{"paths": [str]}` returns status
  - `POST /import/cancel` returns status
  - `GET /status` returns status

- [ ] **Step 1: Write the failing studio tests**

`backend/tests/test_studio.py`:

```python
import numpy as np
import pytest

from aivoice.devices import Device
from aivoice.importer import Importer
from aivoice.library import Library, LibraryError
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from tests.test_recorder import CapturingStreams

SR = 48000
PROMPTS = ("First sentence here.", "Second sentence here.", "Third sentence here.")
DEVICES = [Device(1, "Headset Mic", 1, 0), Device(2, "Headset", 0, 2)]


class NoTranscriber:
    def transcribe(self, audio16k):
        return ""


def make(tmp_path):
    lib = Library(tmp_path)
    streams = CapturingStreams()
    studio = Studio(lib, Recorder(streams), Importer(lib, NoTranscriber()), lambda: DEVICES, prompts=PROMPTS)
    return studio, lib, streams, lib.create("V")


def speak(streams, seconds=2.0, amp=0.3, lead=0.5):
    rng = np.random.default_rng(5)
    streams.on_audio(np.zeros((int(lead * SR), 1), dtype=np.float32))
    streams.on_audio(rng.uniform(-amp, amp, (int(seconds * SR), 1)).astype(np.float32))
    streams.on_audio(np.zeros((int(lead * SR), 1), dtype=np.float32))


def test_prompt_for_a_new_voice(tmp_path):
    studio, _, _, voice = make(tmp_path)
    assert studio.prompt(voice.id) == {"index": 0, "text": "First sentence here.", "total": 3, "recorded": 0}


def test_accepted_take_is_trimmed_saved_and_advances_the_prompt(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    assert studio.recording and studio.status()["recording"] is True
    speak(streams)
    result = studio.stop_recording()
    assert result["accepted"] is True and result["reason"] is None and result["message"] is None
    clip = result["clip"]
    assert (clip["index"], clip["text"], clip["source"], clip["prompt"]) == (1, "First sentence here.", "recorded", 0)
    assert clip["seconds"] == pytest.approx(2.2, abs=0.05)
    assert not studio.recording
    assert studio.prompt(voice.id) == {"index": 1, "text": "Second sentence here.", "total": 3, "recorded": 1}
    assert len(lib.get(voice.id).clips) == 1


@pytest.mark.parametrize("seconds, amp, reason", [(0.0, 0.3, "too_quiet"), (0.3, 0.3, "too_short"), (2.0, 0.01, "too_quiet")])
def test_rejected_takes_write_nothing(tmp_path, seconds, amp, reason):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    if seconds:
        speak(streams, seconds, amp)
    result = studio.stop_recording()
    assert result["accepted"] is False and result["reason"] == reason and result["clip"] is None
    assert len(result["message"]) > 20
    assert lib.get(voice.id).clips == []
    assert list((tmp_path / "voices" / voice.id / "clips").iterdir()) == []


def test_stop_pressed_at_once_is_rejected_cleanly(tmp_path):
    studio, lib, _, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    assert studio.stop_recording()["reason"] == "too_quiet"
    assert lib.get(voice.id).clips == []


def test_redoing_a_sentence_replaces_the_earlier_take(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    for _ in range(2):
        studio.start_recording(voice.id, 1, 0)
        speak(streams)
        studio.stop_recording()
    clips = lib.get(voice.id).clips
    assert [(c.index, c.prompt) for c in clips] == [(2, 0)]


def test_a_rejected_redo_keeps_the_earlier_take(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    studio.stop_recording()
    studio.start_recording(voice.id, 1, 0)
    assert studio.stop_recording()["accepted"] is False
    assert [c.index for c in lib.get(voice.id).clips] == [1]


def test_skip_moves_to_the_next_unrecorded_sentence(tmp_path):
    studio, _, _, voice = make(tmp_path)
    assert studio.prompt(voice.id, after=0)["index"] == 1
    assert studio.prompt(voice.id, after=2)["index"] == 0


def test_all_sentences_recorded(tmp_path):
    studio, _, streams, voice = make(tmp_path)
    for i in range(3):
        studio.start_recording(voice.id, 1, i)
        speak(streams)
        studio.stop_recording()
    assert studio.prompt(voice.id) == {"index": None, "text": None, "total": 3, "recorded": 3}


def test_start_validation(tmp_path):
    studio, _, _, voice = make(tmp_path)
    with pytest.raises(LibraryError):
        studio.start_recording("nope-000000", 1, 0)
    with pytest.raises(ValueError, match="Unknown sentence"):
        studio.start_recording(voice.id, 1, 3)
    with pytest.raises(ValueError, match="not an input"):
        studio.start_recording(voice.id, 2, 0)
    with pytest.raises(ValueError, match="no longer available"):
        studio.start_recording(voice.id, 9, 0)
    assert not studio.recording


def test_stop_when_not_recording_is_a_clear_error(tmp_path):
    studio, _, _, _ = make(tmp_path)
    with pytest.raises(ValueError, match="Not recording"):
        studio.stop_recording()


def test_cancel_discards_the_take(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    studio.cancel_recording()
    assert not studio.recording and lib.get(voice.id).clips == []


def test_voice_deleted_mid_take_is_a_clear_error_and_leaves_the_recorder_idle(tmp_path):
    studio, lib, streams, voice = make(tmp_path)
    studio.start_recording(voice.id, 1, 0)
    speak(streams)
    lib.delete(voice.id)
    with pytest.raises(LibraryError):
        studio.stop_recording()
    assert not studio.recording


def test_status_includes_import_progress(tmp_path):
    studio, _, _, _ = make(tmp_path)
    status = studio.status()
    assert status["recording"] is False and status["import"]["active"] is False
```

- [ ] **Step 2: Write the failing API tests**

`backend/tests/test_api_library.py`:

```python
import numpy as np
from fastapi.testclient import TestClient

from aivoice.api import ALLOWED_HOSTS, create_app
from aivoice.audiofile import write_wav
from aivoice.importer import Importer
from aivoice.library import Library
from aivoice.manager import LiveManager
from aivoice.recorder import Recorder
from aivoice.studio import Studio
from tests.test_live import FakeStreams, Identity
from tests.test_manager import DEVICES
from tests.test_pretrained import add_voice
from tests.test_recorder import CapturingStreams
from tests.test_studio import PROMPTS, speak

SR = 48000


class Echo:
    def transcribe(self, audio16k):
        return "heard"


def make(tmp_path, refreshed=None):
    add_voice(tmp_path, "demo")
    library = Library(tmp_path)
    mic = CapturingStreams()
    importer = Importer(library, Echo())
    studio = Studio(library, Recorder(mic), importer, lambda: DEVICES, prompts=PROMPTS)
    refresh = (lambda: refreshed.append(1)) if refreshed is not None else None
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams(), refresh_devices=refresh)
    app = create_app(manager, tmp_path, lambda: DEVICES, ["DmlExecutionProvider"],
                     allowed_hosts=[*ALLOWED_HOSTS, "testserver"], studio=studio, library=library, importer=importer)
    return TestClient(app), mic, importer


def new_voice(client, name="V"):
    return client.post("/library/voices", json={"name": name}).json()


def record(client, mic, voice_id, prompt=0, **kw):
    assert client.post("/library/record/start", json={"voice_id": voice_id, "input_device": 1, "prompt": prompt}).status_code == 200
    speak(mic, **kw)
    return client.post("/library/record/stop").json()


def test_create_list_delete_voice(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client, "My Voice")
    assert voice["name"] == "My Voice" and voice["minutes"] == 0 and voice["clips"] == 0
    assert [v["id"] for v in client.get("/library/voices").json()] == [voice["id"]]
    assert client.delete(f"/library/voices/{voice['id']}").json() == {"deleted": True}
    assert client.get("/library/voices").json() == []


def test_bad_voice_name_is_400(tmp_path):
    client, _, _ = make(tmp_path)
    res = client.post("/library/voices", json={"name": "   "})
    assert res.status_code == 400 and "name" in res.json()["detail"]


def test_unknown_voice_is_400_everywhere(tmp_path):
    client, _, _ = make(tmp_path)
    for method, path in [("get", "/library/voices/nope-000000/clips"), ("get", "/library/voices/nope-000000/prompt"),
                         ("delete", "/library/voices/nope-000000"), ("get", "/library/voices/..%2F..%2Fx/clips")]:
        assert getattr(client, method)(path).status_code in (400, 404)


def test_guided_recording_round_trip(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    assert client.get(f"/library/voices/{voice['id']}/prompt").json()["index"] == 0
    result = record(client, mic, voice["id"])
    assert result["accepted"] is True and result["clip"]["text"] == PROMPTS[0]
    clips = client.get(f"/library/voices/{voice['id']}/clips").json()
    assert [c["index"] for c in clips] == [1]
    assert client.get(f"/library/voices/{voice['id']}/prompt").json()["index"] == 1
    assert client.get(f"/library/voices/{voice['id']}/prompt", params={"after": 1}).json()["index"] == 2
    assert client.get("/library/voices").json()[0]["clips"] == 1


def test_rejected_take_returns_reason_and_message(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    result = record(client, mic, voice["id"], amp=0.01)
    assert result == {"accepted": False, "reason": "too_quiet", "message": result["message"], "clip": None}
    assert client.get(f"/library/voices/{voice['id']}/clips").json() == []


def test_clip_audio_edit_and_delete(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    record(client, mic, voice["id"])
    base = f"/library/voices/{voice['id']}/clips/1"
    audio = client.get(f"{base}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav" and audio.content[:4] == b"RIFF"
    assert client.patch(base, json={"text": "Fixed."}).json()["text"] == "Fixed."
    assert client.delete(base).json() == {"deleted": True}
    assert client.get(f"{base}/audio").status_code == 400


def test_status_while_recording_and_cancel(tmp_path):
    client, mic, _ = make(tmp_path)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    speak(mic)
    status = client.get("/library/status").json()
    assert status["recording"] is True and status["seconds"] > 1 and status["import"]["active"] is False
    assert client.post("/library/record/cancel").json()["recording"] is False
    assert client.post("/library/record/stop").status_code == 400


def test_recording_is_refused_while_live_and_live_while_recording(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client)
    start_live = {"input_device": 1, "mic_device": 3, "voice": "demo"}
    record_body = {"voice_id": voice["id"], "input_device": 1, "prompt": 0}

    assert client.post("/live/start", json=start_live).status_code == 200
    res = client.post("/library/record/start", json=record_body)
    assert res.status_code == 400 and res.json()["detail"] == "Stop live mode before recording"
    client.post("/live/stop")

    assert client.post("/library/record/start", json=record_body).status_code == 200
    res = client.post("/live/start", json=start_live)
    assert res.status_code == 400 and res.json()["detail"] == "Stop recording before starting live mode"
    assert client.get("/live/stats").json()["running"] is False
    client.post("/library/record/cancel")


def test_device_refresh_is_refused_while_recording(tmp_path):
    refreshed = []
    client, _, _ = make(tmp_path, refreshed)
    voice = new_voice(client)
    client.post("/library/record/start", json={"voice_id": voice["id"], "input_device": 1, "prompt": 0})
    res = client.post("/devices/refresh")
    assert res.status_code == 400 and res.json()["detail"] == "Stop recording before refreshing devices"
    assert refreshed == []
    client.post("/library/record/cancel")
    assert client.post("/devices/refresh").status_code == 200


def test_import_files(tmp_path):
    client, _, importer = make(tmp_path)
    voice = new_voice(client)
    path = tmp_path / "talk.wav"
    write_wav(path, np.random.default_rng(2).uniform(-0.3, 0.3, 2 * SR).astype(np.float32), SR)
    res = client.post(f"/library/voices/{voice['id']}/import", json={"paths": [str(path)]})
    assert res.status_code == 200
    importer.wait(10)
    status = client.get("/library/status").json()["import"]
    assert status["finished"] and status["clips_added"] == 1
    assert client.get(f"/library/voices/{voice['id']}/clips").json()[0]["text"] == "heard"
    assert client.post("/library/import/cancel").status_code == 200


def test_import_with_no_files_is_400(tmp_path):
    client, _, _ = make(tmp_path)
    voice = new_voice(client)
    assert client.post(f"/library/voices/{voice['id']}/import", json={"paths": []}).status_code == 400


def test_library_routes_obey_the_origin_rule(tmp_path):
    client, _, _ = make(tmp_path)
    res = client.post("/library/voices", json={"name": "X"}, headers={"Origin": "https://evil.example"})
    assert res.status_code == 403
    assert client.get("/library/voices").json() == []


def test_app_without_a_studio_has_no_library_routes(tmp_path):
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), FakeStreams())
    client = TestClient(create_app(manager, tmp_path, lambda: DEVICES, [], allowed_hosts=["testserver"]))
    assert client.get("/library/voices").status_code == 404
```

- [ ] **Step 3: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_studio.py backend/tests/test_api_library.py -q`
Expected: FAIL with `ModuleNotFoundError: aivoice.studio`.

- [ ] **Step 4: Implement `backend/aivoice/studio.py`**

```python
import threading
from dataclasses import asdict
from typing import Callable

from aivoice.dataset import PROBLEM_MESSAGES, clip_problem, trim_silence
from aivoice.devices import Device, get_device
from aivoice.importer import Importer
from aivoice.library import SAMPLE_RATE, Library
from aivoice.prompts import load_prompts, next_prompt
from aivoice.recorder import Recorder


class Studio:
    """The guided recording flow: which sentence is next, taking a take, and judging it."""

    def __init__(self, library: Library, recorder: Recorder, importer: Importer,
                 list_devices: Callable[[], list[Device]], prompts: tuple[str, ...] | None = None,
                 sample_rate: int = SAMPLE_RATE):
        self._library = library
        self._recorder = recorder
        self._importer = importer
        self._list_devices = list_devices
        self._prompts = prompts if prompts is not None else load_prompts()
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._take: tuple[str, int] | None = None

    @property
    def recording(self) -> bool:
        return self._recorder.recording

    def prompt(self, voice_id, after: int | None = None) -> dict:
        voice = self._library.get(voice_id)
        recorded = {c.prompt for c in voice.clips if c.prompt is not None}
        index = next_prompt(recorded, len(self._prompts), after)
        return {"index": index, "text": None if index is None else self._prompts[index],
                "total": len(self._prompts), "recorded": len(recorded)}

    def start_recording(self, voice_id, device: int, prompt: int) -> None:
        with self._lock:
            self._library.get(voice_id)
            if not isinstance(prompt, int) or not 0 <= prompt < len(self._prompts):
                raise ValueError("Unknown sentence")
            source = get_device(self._list_devices(), device)
            if source.max_input < 1:
                raise ValueError(f"{source.name} is not an input device")
            self._recorder.start(device)
            self._take = (voice_id, prompt)

    def stop_recording(self) -> dict:
        with self._lock:
            audio = self._recorder.stop()
            voice_id, prompt = self._take
            self._take = None
            take = trim_silence(audio, self._sr)
            problem = clip_problem(take, self._sr)
            if problem is not None:
                return {"accepted": False, "reason": problem, "message": PROBLEM_MESSAGES[problem], "clip": None}
            earlier = [c.index for c in self._library.get(voice_id).clips if c.prompt == prompt]
            clip = self._library.add_clip(voice_id, take, self._prompts[prompt], "recorded", prompt=prompt)
            for index in earlier:  # a redo replaces the earlier take, but only once the new one is saved
                self._library.delete_clip(voice_id, index)
            return {"accepted": True, "reason": None, "message": None, "clip": asdict(clip)}

    def cancel_recording(self) -> None:
        with self._lock:
            self._recorder.cancel()
            self._take = None

    def status(self) -> dict:
        return {**self._recorder.status(), "import": self._importer.status()}
```

- [ ] **Step 5: Implement `backend/aivoice/api_library.py`**

```python
from dataclasses import asdict
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from aivoice.importer import Importer
from aivoice.library import Library, Voice
from aivoice.studio import Studio


class VoiceBody(BaseModel):
    name: str


class TextBody(BaseModel):
    text: str


class RecordBody(BaseModel):
    voice_id: str
    input_device: int
    prompt: int


class ImportBody(BaseModel):
    paths: list[str]


def _summary(voice: Voice) -> dict:
    return {"id": voice.id, "name": voice.name, "created": voice.created, "minutes": voice.minutes,
            "clips": len(voice.clips)}


def library_router(studio: Studio, library: Library, importer: Importer,
                   live_running: Callable[[], bool]) -> APIRouter:
    router = APIRouter(prefix="/library")

    @router.get("/voices")
    def voices() -> list[dict]:
        return [_summary(v) for v in library.list()]

    @router.post("/voices")
    def create_voice(body: VoiceBody) -> dict:
        return _summary(library.create(body.name))

    @router.delete("/voices/{voice_id}")
    def delete_voice(voice_id: str) -> dict:
        if studio.recording:
            raise ValueError("Stop recording before deleting a voice")
        library.delete(voice_id)
        return {"deleted": True}

    @router.get("/voices/{voice_id}/clips")
    def clips(voice_id: str) -> list[dict]:
        return [asdict(c) for c in library.get(voice_id).clips]

    @router.patch("/voices/{voice_id}/clips/{index}")
    def edit_clip(voice_id: str, index: int, body: TextBody) -> dict:
        return asdict(library.update_text(voice_id, index, body.text))

    @router.delete("/voices/{voice_id}/clips/{index}")
    def delete_clip(voice_id: str, index: int) -> dict:
        library.delete_clip(voice_id, index)
        return {"deleted": True}

    @router.get("/voices/{voice_id}/clips/{index}/audio")
    def clip_audio(voice_id: str, index: int) -> FileResponse:
        return FileResponse(library.clip_audio_path(voice_id, index), media_type="audio/wav")

    @router.get("/voices/{voice_id}/prompt")
    def prompt(voice_id: str, after: int | None = None) -> dict:
        return studio.prompt(voice_id, after)

    @router.post("/record/start")
    def record_start(body: RecordBody) -> dict:
        if live_running():
            raise ValueError("Stop live mode before recording")
        studio.start_recording(body.voice_id, body.input_device, body.prompt)
        return studio.status()

    @router.post("/record/stop")
    def record_stop() -> dict:
        return studio.stop_recording()

    @router.post("/record/cancel")
    def record_cancel() -> dict:
        studio.cancel_recording()
        return studio.status()

    @router.post("/voices/{voice_id}/import")
    def import_files(voice_id: str, body: ImportBody) -> dict:
        importer.start(voice_id, body.paths)
        return studio.status()

    @router.post("/import/cancel")
    def import_cancel() -> dict:
        importer.cancel()
        return studio.status()

    @router.get("/status")
    def status() -> dict:
        return studio.status()

    return router
```

- [ ] **Step 6: Modify `backend/aivoice/api.py`**

Change the `create_app` signature to add three optional parameters at the end:

```python
def create_app(manager: LiveManager, data: Path, list_devices: Callable[[], list[Device]],
               providers: list[str], default_devices: Callable[[], dict] | None = None,
               allowed_hosts: list[str] = ALLOWED_HOSTS, studio=None, library=None, importer=None) -> FastAPI:
```

Replace the `refresh_devices` and `start` route bodies with:

```python
    @app.post("/devices/refresh")
    def refresh_devices() -> dict:
        if studio is not None and studio.recording:
            raise ValueError("Stop recording before refreshing devices")
        manager.refresh_devices()
        return device_payload()
```

```python
    @app.post("/live/start")
    def start(body: StartBody) -> dict:
        if studio is not None and studio.recording:
            raise ValueError("Stop recording before starting live mode")
        manager.start(StartConfig(**body.model_dump()))
        return manager.stats()
```

Just before `return app`, add:

```python
    if studio is not None:
        from aivoice.api_library import library_router

        app.include_router(library_router(studio, library, importer, lambda: bool(manager.stats()["running"])))
```

- [ ] **Step 7: Modify `backend/aivoice/__main__.py`**

Replace `build_app` with:

```python
def build_app():
    import onnxruntime as ort

    from aivoice.importer import Importer
    from aivoice.library import Library
    from aivoice.recorder import Recorder
    from aivoice.studio import Studio
    from aivoice.transcribe import WhisperTranscriber

    data = config.data_dir()
    streams = SoundDeviceStreams()
    manager = LiveManager(data, list_devices, default_engine_factory(data), streams,
                          refresh_devices=refresh_devices)
    library = Library(data)
    # The model name is read when an import first needs it, so a bad setting cannot stop startup.
    importer = Importer(library, WhisperTranscriber(lambda: config.whisper_model(data), data / "pretrained" / "whisper"))
    studio = Studio(library, Recorder(streams), importer, list_devices)
    return create_app(manager, data, list_devices, ort.get_available_providers(), default_devices=default_devices,
                      studio=studio, library=library, importer=importer)
```

- [ ] **Step 8: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests -q`
Expected: every test passes, including all stage 1 tests unchanged.

- [ ] **Step 9: Smoke test the real server without opening a microphone**

Start `backend/.venv/Scripts/python -m aivoice --port 8765` in the background, then:

```bash
curl -s -X POST http://127.0.0.1:8765/library/voices -H "Content-Type: application/json" -d '{"name":"Smoke Test"}'
curl -s http://127.0.0.1:8765/library/voices
curl -s http://127.0.0.1:8765/library/status
```

Expected: the voice is created and listed, and status shows `"recording": false`. Then delete that voice with `curl -s -X DELETE http://127.0.0.1:8765/library/voices/<id>` and stop the server. Do NOT call `/library/record/start` or `/live/start` against the real server. Remove the smoke-test folder from `data/trash/` afterwards, since it is the only thing in it you created.

- [ ] **Step 10: Commit**

```bash
git add backend
git commit -m "feat: guided recording flow and library API"
```

---

### Task 9: Frontend: tabs, library client, and the voice list

**Files:**
- Modify: `app/src/api.ts` (export `call` and `origin`), `app/src/App.tsx`, `app/src/App.css`, `app/src/App.test.tsx`
- Create: `app/src/studioApi.ts`, `app/src/VoicesPanel.tsx`, `app/src/VoiceStudio.tsx`
- Test: `app/src/VoicesPanel.test.tsx`

**Interfaces:**
- Consumes: the `/library` routes from Task 8.
- Produces: `studioApi` with `voices()`, `createVoice(name)`, `deleteVoice(id)`, `clips(id)`, `editClip(id, index, text)`, `deleteClip(id, index)`, `clipAudioUrl(id, index)`, `prompt(id, after?)`, `recordStart(voiceId, inputDevice, prompt)`, `recordStop()`, `recordCancel()`, `importFiles(id, paths)`, `importCancel()`, `status()`; types `LibraryVoice`, `Clip`, `Prompt`, `TakeResult`, `ImportStatus`, `StudioStatus`. `VoicesPanel({ active }: { active: boolean })`. `VoiceStudio({ voice, active, onChanged }: { voice: LibraryVoice; active: boolean; onChanged: () => void })`, which in this task shows only the minutes bar and is filled in by Tasks 10 and 11.

- [ ] **Step 1: Export the HTTP helpers**

In `app/src/api.ts`, add `export` in front of `async function origin()` and `async function call<T>(`. No other change.

- [ ] **Step 2: Write `app/src/studioApi.ts`**

```ts
import { call, origin } from "./api";

export type LibraryVoice = { id: string; name: string; created: string; minutes: number; clips: number };
export type Clip = {
  index: number;
  text: string;
  seconds: number;
  source: "recorded" | "imported";
  prompt: number | null;
  origin: string | null;
};
/** `index` is null once every sentence has been recorded. */
export type Prompt = { index: number | null; text: string | null; total: number; recorded: number };
export type TakeResult = { accepted: boolean; reason: string | null; message: string | null; clip: Clip | null };
export type ImportStatus = {
  active: boolean;
  finished: boolean;
  total_files: number;
  done_files: number;
  current: string | null;
  clips_added: number;
  skipped: number;
  errors: string[];
};
export type StudioStatus = { recording: boolean; level: number; seconds: number; full: boolean; import: ImportStatus };

const voicePath = (id: string) => `/library/voices/${encodeURIComponent(id)}`;

export const studioApi = {
  voices: () => call<LibraryVoice[]>("GET", "/library/voices"),
  createVoice: (name: string) => call<LibraryVoice>("POST", "/library/voices", { name }),
  deleteVoice: (id: string) => call<{ deleted: boolean }>("DELETE", voicePath(id)),
  clips: (id: string) => call<Clip[]>("GET", `${voicePath(id)}/clips`),
  editClip: (id: string, index: number, text: string) => call<Clip>("PATCH", `${voicePath(id)}/clips/${index}`, { text }),
  deleteClip: (id: string, index: number) => call<{ deleted: boolean }>("DELETE", `${voicePath(id)}/clips/${index}`),
  clipAudioUrl: async (id: string, index: number) => `http://${await origin()}${voicePath(id)}/clips/${index}/audio`,
  prompt: (id: string, after?: number) =>
    call<Prompt>("GET", `${voicePath(id)}/prompt${after === undefined ? "" : `?after=${after}`}`),
  recordStart: (voiceId: string, inputDevice: number, prompt: number) =>
    call<StudioStatus>("POST", "/library/record/start", { voice_id: voiceId, input_device: inputDevice, prompt }),
  recordStop: () => call<TakeResult>("POST", "/library/record/stop"),
  recordCancel: () => call<StudioStatus>("POST", "/library/record/cancel"),
  importFiles: (id: string, paths: string[]) => call<StudioStatus>("POST", `${voicePath(id)}/import`, { paths }),
  importCancel: () => call<StudioStatus>("POST", "/library/import/cancel"),
  status: () => call<StudioStatus>("GET", "/library/status"),
};
```

- [ ] **Step 3: Write the failing tests**

`app/src/VoicesPanel.test.tsx`:

```tsx
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VoicesPanel } from "./VoicesPanel";

vi.mock("./studioApi", () => ({
  studioApi: { voices: vi.fn(), createVoice: vi.fn(), deleteVoice: vi.fn() },
}));
vi.mock("./VoiceStudio", () => ({
  VoiceStudio: ({ voice, onChanged }: { voice: { name: string }; onChanged: () => void }) => (
    <button onClick={onChanged}>studio for {voice.name}</button>
  ),
}));
import { studioApi } from "./studioApi";

const A = { id: "a-111111", name: "Alpha", created: "1", minutes: 4.5, clips: 30 };
const B = { id: "b-222222", name: "Beta", created: "2", minutes: 0, clips: 0 };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(studioApi.voices).mockResolvedValue([A, B]);
  vi.mocked(studioApi.createVoice).mockResolvedValue({ id: "c-333333", name: "Gamma", created: "3", minutes: 0, clips: 0 });
  vi.mocked(studioApi.deleteVoice).mockResolvedValue({ deleted: true });
});

describe("VoicesPanel", () => {
  it("lists voices with their recorded minutes and opens the first one", async () => {
    render(<VoicesPanel active />);
    const list = await screen.findByRole("list", { name: "Voices" });
    expect(within(list).getByText("Alpha")).toBeInTheDocument();
    expect(within(list).getByText(/4\.5 min/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "studio for Alpha" })).toBeInTheDocument();
  });

  it("switches the studio when another voice is chosen", async () => {
    render(<VoicesPanel active />);
    await userEvent.click(await screen.findByRole("button", { name: /Beta/ }));
    expect(screen.getByRole("button", { name: "studio for Beta" })).toBeInTheDocument();
  });

  it("explains what to do when there are no voices", async () => {
    vi.mocked(studioApi.voices).mockResolvedValue([]);
    render(<VoicesPanel active />);
    expect(await screen.findByText(/create a voice to start recording/i)).toBeInTheDocument();
  });

  it("creates a voice, selects it and clears the name field", async () => {
    render(<VoicesPanel active />);
    await screen.findByRole("list", { name: "Voices" });
    vi.mocked(studioApi.voices).mockResolvedValue([A, B, { id: "c-333333", name: "Gamma", created: "3", minutes: 0, clips: 0 }]);
    const field = screen.getByLabelText("New voice name");
    await userEvent.type(field, "Gamma");
    await userEvent.click(screen.getByRole("button", { name: "Create voice" }));
    expect(studioApi.createVoice).toHaveBeenCalledWith("Gamma");
    expect(await screen.findByRole("button", { name: "studio for Gamma" })).toBeInTheDocument();
    expect(field).toHaveValue("");
  });

  it("cannot create a voice with a blank name", async () => {
    render(<VoicesPanel active />);
    await screen.findByRole("list", { name: "Voices" });
    expect(screen.getByRole("button", { name: "Create voice" })).toBeDisabled();
    await userEvent.type(screen.getByLabelText("New voice name"), "   ");
    expect(screen.getByRole("button", { name: "Create voice" })).toBeDisabled();
  });

  it("shows the backend's message when creating fails", async () => {
    vi.mocked(studioApi.createVoice).mockRejectedValue(new Error("A voice name must be 1 to 60 characters on one line"));
    render(<VoicesPanel active />);
    await screen.findByRole("list", { name: "Voices" });
    await userEvent.type(screen.getByLabelText("New voice name"), "x");
    await userEvent.click(screen.getByRole("button", { name: "Create voice" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("1 to 60 characters");
  });

  it("deletes a voice only after confirmation", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<VoicesPanel active />);
    await userEvent.click(await screen.findByRole("button", { name: "Delete voice Alpha" }));
    expect(confirm).toHaveBeenCalled();
    expect(confirm.mock.calls[0][0]).toMatch(/Alpha/);
    expect(confirm.mock.calls[0][0]).toMatch(/trash/i);
    expect(studioApi.deleteVoice).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    vi.mocked(studioApi.voices).mockResolvedValue([B]);
    await userEvent.click(screen.getByRole("button", { name: "Delete voice Alpha" }));
    expect(studioApi.deleteVoice).toHaveBeenCalledWith("a-111111");
    expect(await screen.findByRole("button", { name: "studio for Beta" })).toBeInTheDocument();
    confirm.mockRestore();
  });

  it("reloads the list when the studio reports a change", async () => {
    render(<VoicesPanel active />);
    await userEvent.click(await screen.findByRole("button", { name: "studio for Alpha" }));
    await vi.waitFor(() => expect(studioApi.voices).toHaveBeenCalledTimes(2));
  });

  it("shows a load error with a retry", async () => {
    vi.mocked(studioApi.voices).mockRejectedValueOnce(new Error("backend unreachable"));
    render(<VoicesPanel active />);
    expect(await screen.findByRole("alert")).toHaveTextContent("backend unreachable");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("list", { name: "Voices" })).toBeInTheDocument();
  });

  it("does not load until the tab is first shown", async () => {
    const { rerender } = render(<VoicesPanel active={false} />);
    expect(studioApi.voices).not.toHaveBeenCalled();
    rerender(<VoicesPanel active />);
    await vi.waitFor(() => expect(studioApi.voices).toHaveBeenCalledTimes(1));
  });
});
```

Add to `app/src/App.test.tsx`, next to the other `vi.mock` calls:

```tsx
vi.mock("./VoicesPanel", () => ({
  VoicesPanel: ({ active }: { active: boolean }) => <p>voices panel {active ? "active" : "idle"}</p>,
}));
```

and add these tests inside the existing `describe("App", ...)` block (import `userEvent` from `@testing-library/user-event` at the top; because the file uses fake timers, create the user with `userEvent.setup({ advanceTimers: vi.advanceTimersByTime })`):

```tsx
  it("starts on the Live tab with the Voices panel idle", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    expect(screen.getByRole("tab", { name: "Live" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: "live panel" })).toBeVisible();
    expect(screen.getByText("voices panel idle")).not.toBeVisible();
  });

  it("switches to the Voices tab and keeps the Live panel mounted", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await advance(0);
    await user.click(screen.getByRole("tab", { name: "Voices" }));
    expect(screen.getByText("voices panel active")).toBeVisible();
    expect(screen.getByRole("button", { name: "live panel", hidden: true })).not.toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Live" }));
    expect(screen.getByRole("button", { name: "live panel" })).toBeVisible();
  });
```

- [ ] **Step 4: Run to verify failure**

Run from `app/`: `npm test`
Expected: FAIL, cannot resolve `./VoicesPanel`, and the two new App tests fail.

- [ ] **Step 5: Implement**

`app/src/VoiceStudio.tsx`:

```tsx
import { LibraryVoice } from "./studioApi";

export const USABLE_MINUTES = 10;
export const GOOD_MINUTES = 30;

export function MinutesBar({ minutes }: { minutes: number }) {
  const state = minutes >= GOOD_MINUTES ? "Plenty for a good voice." : minutes >= USABLE_MINUTES ? "Enough for a usable voice." : "";
  return (
    <div className="minutes">
      <progress max={GOOD_MINUTES} value={Math.min(minutes, GOOD_MINUTES)} aria-label="Recorded minutes" />
      <p>
        {minutes.toFixed(1)} min of clean audio. {USABLE_MINUTES} min is usable, {GOOD_MINUTES} min is good. {state}
      </p>
    </div>
  );
}

export function VoiceStudio({ voice }: { voice: LibraryVoice; active: boolean; onChanged: () => void }) {
  return (
    <section className="studio" aria-label={`Voice ${voice.name}`}>
      <h2>{voice.name}</h2>
      <MinutesBar minutes={voice.minutes} />
    </section>
  );
}
```

`app/src/VoicesPanel.tsx`:

```tsx
import { useCallback, useEffect, useState } from "react";
import { LibraryVoice, studioApi } from "./studioApi";
import { VoiceStudio } from "./VoiceStudio";

export function VoicesPanel({ active }: { active: boolean }) {
  const [voices, setVoices] = useState<LibraryVoice[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [shown, setShown] = useState(active);

  useEffect(() => {
    if (active) setShown(true);
  }, [active]);

  const load = useCallback(async (select?: string) => {
    try {
      const list = await studioApi.voices();
      setVoices(list);
      setError(null);
      setSelected((current) => {
        const wanted = select ?? current;
        return list.some((v) => v.id === wanted) ? wanted! : (list[0]?.id ?? null);
      });
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    if (shown) load();
  }, [shown, load]);

  const create = async () => {
    setBusy(true);
    try {
      const voice = await studioApi.createVoice(name.trim());
      setName("");
      await load(voice.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (voice: LibraryVoice) => {
    const ok = window.confirm(
      `Delete the voice "${voice.name}" and its ${voice.clips} clips? It is moved to the data/trash folder, not erased.`,
    );
    if (!ok) return;
    try {
      await studioApi.deleteVoice(voice.id);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const current = voices?.find((v) => v.id === selected) ?? null;

  return (
    <section className="voices">
      {error && (
        <p role="alert" className="error">
          {error} {voices === null && <button onClick={() => load()}>Retry</button>}
        </p>
      )}
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          create();
        }}
      >
        <label>
          New voice name
          <input value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
        </label>
        <button type="submit" disabled={busy || name.trim() === ""}>Create voice</button>
      </form>

      {voices !== null && voices.length === 0 && <p className="hint">Create a voice to start recording.</p>}
      {voices !== null && voices.length > 0 && (
        <ul aria-label="Voices" className="voice-list">
          {voices.map((v) => (
            <li key={v.id} className={v.id === selected ? "selected" : ""}>
              <button className="link" aria-pressed={v.id === selected} onClick={() => setSelected(v.id)}>
                <strong>{v.name}</strong> <span>{v.minutes.toFixed(1)} min, {v.clips} clips</span>
              </button>
              <button className="danger" aria-label={`Delete voice ${v.name}`} onClick={() => remove(v)}>Delete</button>
            </li>
          ))}
        </ul>
      )}
      {current && <VoiceStudio key={current.id} voice={current} active={active} onChanged={() => load()} />}
    </section>
  );
}
```

In `app/src/App.tsx`, import `VoicesPanel` and replace the `Main` component's returned JSX with tabs. The full new `Main`:

```tsx
function Main() {
  const stats = useStats();
  const latest = useRef(stats);
  latest.current = stats;
  const [actionError, setActionError] = useState<string | null>(null);
  const [devicesVersion, setDevicesVersion] = useState(0);
  const [tab, setTab] = useState<"live" | "voices">("live");

  useLiveShortcuts(latest, setActionError);

  return (
    <>
      <div role="tablist" className="tabs">
        <button role="tab" aria-selected={tab === "live"} onClick={() => setTab("live")}>Live</button>
        <button role="tab" aria-selected={tab === "voices"} onClick={() => setTab("voices")}>Voices</button>
      </div>
      {/* Both stay mounted so switching tabs never resets a form or interrupts a session. */}
      <div role="tabpanel" hidden={tab !== "live"}>
        <SetupCheck refreshKey={devicesVersion} />
        <LivePanel stats={stats} actionError={actionError} onDevicesRefreshed={() => setDevicesVersion((v) => v + 1)} />
      </div>
      <div role="tabpanel" hidden={tab !== "voices"}>
        <VoicesPanel active={tab === "voices"} />
      </div>
    </>
  );
}
```

Append to `app/src/App.css`:

```css
.tabs { display: flex; gap: 4px; margin-bottom: 12px; border-bottom: 1px solid #343843; }
.tabs button { background: none; color: #aab0bb; border-radius: 6px 6px 0 0; padding: 8px 16px; }
.tabs button[aria-selected="true"] { color: #fff; background: #22252c; }
.row { display: flex; gap: 8px; align-items: flex-end; }
.row label { flex: 1; margin: 0; }
.row input { display: block; width: 100%; margin-top: 4px; padding: 6px; background: #22252c; color: inherit; border: 1px solid #343843; border-radius: 6px; }
.voice-list { list-style: none; padding: 0; margin: 12px 0; }
.voice-list li { display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 6px; }
.voice-list li.selected { background: #22252c; }
button.link { flex: 1; text-align: left; background: none; color: inherit; font-weight: 400; padding: 6px; }
button.link span { color: #aab0bb; font-size: 0.8rem; }
button.danger { background: #5a2a30; padding: 4px 10px; font-size: 0.8rem; }
.minutes progress { width: 100%; height: 10px; }
.minutes p { font-size: 0.8rem; color: #aab0bb; margin: 4px 0 12px; }
[hidden] { display: none !important; }
```

- [ ] **Step 6: Run to verify pass**

Run from `app/`: `npm test && npx tsc --noEmit`
Expected: all tests pass, no type errors. If the existing App tests that look for "live panel" break because of the tab markup, fix the test queries, not the behaviour.

- [ ] **Step 7: Commit**

```bash
git add app/src
git commit -m "feat: Voices tab with voice list, create and delete"
```

---

### Task 10: Frontend: guided recording

**Files:**
- Create: `app/src/useStudioStatus.ts`, `app/src/RecordSession.tsx`
- Modify: `app/src/VoiceStudio.tsx`, `app/src/App.css`
- Test: `app/src/useStudioStatus.test.tsx`, `app/src/RecordSession.test.tsx`

**Interfaces:**
- Consumes: `studioApi` (Task 9), `api.devices()` (stage 1).
- Produces: `useStudioStatus(enabled: boolean, intervalMs = 200): StudioStatus | null` (polls `studioApi.status()` while enabled; null before the first answer and when disabled). `RecordSession({ voiceId, active, onSaved }: { voiceId: string; active: boolean; onSaved: () => void })`.

Behaviour of `RecordSession`:
- Loads input devices (preselecting `default_input`) and the next prompt for the voice.
- Shows the sentence, "Sentence N of 200", and how many are recorded.
- **Record** starts a take. While recording it shows a level meter, elapsed seconds, **Stop** and **Cancel**.
- **Stop** ends the take. A rejected take shows the backend's message and stays on the same sentence. An accepted take shows "Saved", an audio player for the take, **Next sentence** and **Redo**.
- **Skip** (when not recording) moves to the next unrecorded sentence without saving.
- When every sentence is recorded it says so and offers no Record button.
- Any API error is shown in a `role="alert"`.
- The space bar is not bound; buttons only.

- [ ] **Step 1: Write the failing tests**

`app/src/useStudioStatus.test.tsx`:

```tsx
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStudioStatus } from "./useStudioStatus";

vi.mock("./studioApi", () => ({ studioApi: { status: vi.fn() } }));
import { studioApi } from "./studioApi";

const STATUS = {
  recording: true, level: 0.4, seconds: 1.2, full: false,
  import: { active: false, finished: false, total_files: 0, done_files: 0, current: null, clips_added: 0, skipped: 0, errors: [] },
};
const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(studioApi.status).mockReset().mockResolvedValue(STATUS);
});
afterEach(() => vi.useRealTimers());

describe("useStudioStatus", () => {
  it("is null and silent while disabled", async () => {
    const { result } = renderHook(() => useStudioStatus(false));
    await tick(1000);
    expect(result.current).toBeNull();
    expect(studioApi.status).not.toHaveBeenCalled();
  });

  it("polls while enabled and returns the latest status", async () => {
    const { result } = renderHook(() => useStudioStatus(true, 200));
    await tick(0);
    expect(result.current).toEqual(STATUS);
    await tick(650);
    expect(vi.mocked(studioApi.status).mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it("stops polling and clears when disabled or unmounted", async () => {
    const { result, rerender, unmount } = renderHook(({ on }) => useStudioStatus(on, 200), { initialProps: { on: true } });
    await tick(250);
    rerender({ on: false });
    expect(result.current).toBeNull();
    const calls = vi.mocked(studioApi.status).mock.calls.length;
    await tick(1000);
    expect(studioApi.status).toHaveBeenCalledTimes(calls);
    unmount();
  });

  it("keeps the last status and keeps polling through a failed poll", async () => {
    const { result } = renderHook(() => useStudioStatus(true, 200));
    await tick(0);
    vi.mocked(studioApi.status).mockRejectedValueOnce(new Error("blip"));
    await tick(200);
    expect(result.current).toEqual(STATUS);
    await tick(400);
    expect(vi.mocked(studioApi.status).mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it("does not overlap requests when the backend is slow", async () => {
    let release: (v: typeof STATUS) => void = () => {};
    vi.mocked(studioApi.status).mockReturnValue(new Promise((r) => { release = r; }));
    renderHook(() => useStudioStatus(true, 200));
    await tick(1000);
    expect(studioApi.status).toHaveBeenCalledTimes(1);
    release(STATUS);
  });
});
```

`app/src/RecordSession.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RecordSession } from "./RecordSession";

vi.mock("./api", () => ({ api: { devices: vi.fn() } }));
vi.mock("./studioApi", () => ({
  studioApi: { prompt: vi.fn(), recordStart: vi.fn(), recordStop: vi.fn(), recordCancel: vi.fn(), clipAudioUrl: vi.fn() },
}));
const status = vi.hoisted(() => ({ current: null as null | { recording: boolean; level: number; seconds: number; full: boolean } }));
vi.mock("./useStudioStatus", () => ({ useStudioStatus: () => status.current }));
import { api } from "./api";
import { studioApi } from "./studioApi";

const P0 = { index: 0, text: "The quiet river bends north.", total: 200, recorded: 0 };
const P1 = { index: 1, text: "She packed a thick wool jumper.", total: 200, recorded: 1 };
const CLIP = { index: 7, text: P0.text, seconds: 2.4, source: "recorded" as const, prompt: 0, origin: null };
const IDLE = { recording: false, level: 0, seconds: 0, full: false, import: {} as never };

beforeEach(() => {
  vi.resetAllMocks();
  status.current = null;
  vi.mocked(api.devices).mockResolvedValue({
    devices: [
      { index: 1, name: "Webcam Mic", max_input: 1, max_output: 0 },
      { index: 4, name: "Headset Mic", max_input: 1, max_output: 0 },
      { index: 5, name: "Headset", max_input: 0, max_output: 2 },
    ],
    virtual_mic: null, default_input: 4, default_output: 5,
  });
  vi.mocked(studioApi.prompt).mockResolvedValue(P0);
  vi.mocked(studioApi.recordStart).mockResolvedValue(IDLE);
  vi.mocked(studioApi.recordCancel).mockResolvedValue(IDLE);
  vi.mocked(studioApi.recordStop).mockResolvedValue({ accepted: true, reason: null, message: null, clip: CLIP });
  vi.mocked(studioApi.clipAudioUrl).mockResolvedValue("http://127.0.0.1:1/audio");
});

const ready = () => screen.findByRole("button", { name: "Record" });

describe("RecordSession", () => {
  it("shows the sentence, progress, and only input devices with the default chosen", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await ready();
    expect(screen.getByText(P0.text)).toBeInTheDocument();
    expect(screen.getByText(/Sentence 1 of 200/)).toBeInTheDocument();
    const select = screen.getByLabelText("Microphone") as HTMLSelectElement;
    expect(select.querySelectorAll("option")).toHaveLength(2);
    expect(select.value).toBe("4");
  });

  it("records, stops, and shows the saved take with playback", async () => {
    const onSaved = vi.fn();
    render(<RecordSession voiceId="v-1" active onSaved={onSaved} />);
    await userEvent.click(await ready());
    expect(studioApi.recordStart).toHaveBeenCalledWith("v-1", 4, 0);
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(await screen.findByText(/Saved/)).toBeInTheDocument();
    expect(onSaved).toHaveBeenCalledTimes(1);
    expect(studioApi.clipAudioUrl).toHaveBeenCalledWith("v-1", 7);
    expect(await screen.findByLabelText("Play back the take")).toHaveAttribute("src", "http://127.0.0.1:1/audio");
    expect(screen.getByRole("button", { name: "Next sentence" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Redo" })).toBeInTheDocument();
  });

  it("shows the level and elapsed time while recording", async () => {
    const { rerender } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await screen.findByRole("button", { name: "Stop" });
    status.current = { recording: true, level: 0.6, seconds: 3.2, full: false };
    rerender(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    expect(screen.getByLabelText("Input level")).toHaveValue(0.6);
    expect(screen.getByText(/3\.2 s/)).toBeInTheDocument();
  });

  it("keeps the same sentence and shows why when a take is rejected", async () => {
    vi.mocked(studioApi.recordStop).mockResolvedValue({ accepted: false, reason: "clipped", message: "The recording distorted. Move back.", clip: null });
    const onSaved = vi.fn();
    render(<RecordSession voiceId="v-1" active onSaved={onSaved} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The recording distorted");
    expect(screen.getByText(P0.text)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Record" })).toBeInTheDocument();
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("moves on with Next sentence", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    vi.mocked(studioApi.prompt).mockResolvedValue(P1);
    await userEvent.click(await screen.findByRole("button", { name: "Next sentence" }));
    expect(await screen.findByText(P1.text)).toBeInTheDocument();
    expect(screen.getByText(/Sentence 2 of 200/)).toBeInTheDocument();
    expect(screen.queryByText(/Saved/)).not.toBeInTheDocument();
  });

  it("Redo records the same sentence again", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await userEvent.click(await screen.findByRole("button", { name: "Redo" }));
    expect(studioApi.recordStart).toHaveBeenLastCalledWith("v-1", 4, 0);
  });

  it("Skip asks for the sentence after this one", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await ready();
    vi.mocked(studioApi.prompt).mockResolvedValue(P1);
    await userEvent.click(screen.getByRole("button", { name: "Skip" }));
    expect(studioApi.prompt).toHaveBeenLastCalledWith("v-1", 0);
    expect(await screen.findByText(P1.text)).toBeInTheDocument();
  });

  it("Cancel discards the take", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(studioApi.recordCancel).toHaveBeenCalled();
    expect(await ready()).toBeInTheDocument();
    expect(studioApi.recordStop).not.toHaveBeenCalled();
  });

  it("shows the backend's message when recording cannot start", async () => {
    vi.mocked(studioApi.recordStart).mockRejectedValue(new Error("Stop live mode before recording"));
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    expect(await screen.findByRole("alert")).toHaveTextContent("Stop live mode before recording");
    expect(screen.getByRole("button", { name: "Record" })).toBeInTheDocument();
  });

  it("says when every sentence is recorded", async () => {
    vi.mocked(studioApi.prompt).mockResolvedValue({ index: null, text: null, total: 200, recorded: 200 });
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    expect(await screen.findByText(/every sentence is recorded/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Record" })).not.toBeInTheDocument();
  });

  it("cannot record without a microphone", async () => {
    vi.mocked(api.devices).mockResolvedValue({ devices: [], virtual_mic: null, default_input: null, default_output: null });
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    expect(await ready()).toBeDisabled();
    expect(screen.getByText(/no microphone was found/i)).toBeInTheDocument();
  });

  it("cancels an in-progress take when it is unmounted", async () => {
    const { unmount } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await screen.findByRole("button", { name: "Stop" });
    unmount();
    expect(studioApi.recordCancel).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run from `app/`: `npm test`
Expected: FAIL, cannot resolve `./useStudioStatus` and `./RecordSession`.

- [ ] **Step 3: Implement `app/src/useStudioStatus.ts`**

```ts
import { useEffect, useState } from "react";
import { StudioStatus, studioApi } from "./studioApi";

/** Polls recording level and import progress while enabled. One request at a time. */
export function useStudioStatus(enabled: boolean, intervalMs = 200): StudioStatus | null {
  const [status, setStatus] = useState<StudioStatus | null>(null);
  useEffect(() => {
    if (!enabled) {
      setStatus(null);
      return;
    }
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const next = await studioApi.status();
        if (!stopped) setStatus(next);
      } catch {
        // keep the last status; the next poll may succeed
      }
      if (!stopped) timer = setTimeout(poll, intervalMs);
    };
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [enabled, intervalMs]);
  return status;
}
```

- [ ] **Step 4: Implement `app/src/RecordSession.tsx`**

```tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { api, Device } from "./api";
import { Clip, Prompt, studioApi } from "./studioApi";
import { useStudioStatus } from "./useStudioStatus";

export function RecordSession({ voiceId, active, onSaved }: { voiceId: string; active: boolean; onSaved: () => void }) {
  const [inputs, setInputs] = useState<Device[]>([]);
  const [input, setInput] = useState<number | null>(null);
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  const [recording, setRecording] = useState(false);
  const [saved, setSaved] = useState<{ clip: Clip; url: string | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const recordingRef = useRef(false);
  recordingRef.current = recording;
  const status = useStudioStatus(active && recording);

  const fail = (e: unknown) => setError((e as Error).message);

  const loadPrompt = useCallback(
    async (after?: number) => {
      setPrompt(await studioApi.prompt(voiceId, after));
      setSaved(null);
    },
    [voiceId],
  );

  useEffect(() => {
    let gone = false;
    (async () => {
      try {
        const list = await api.devices();
        if (gone) return;
        const ins = list.devices.filter((d) => d.max_input > 0);
        setInputs(ins);
        setInput(ins.some((d) => d.index === list.default_input) ? list.default_input : (ins[0]?.index ?? null));
        await loadPrompt();
      } catch (e) {
        if (!gone) fail(e);
      }
    })();
    return () => {
      gone = true;
      if (recordingRef.current) studioApi.recordCancel().catch(() => undefined);
    };
  }, [loadPrompt]);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const record = () =>
    run(async () => {
      if (input === null || prompt?.index == null) return;
      await studioApi.recordStart(voiceId, input, prompt.index);
      setSaved(null);
      setRecording(true);
    });

  const stop = () =>
    run(async () => {
      setRecording(false);
      const result = await studioApi.recordStop();
      if (!result.accepted || !result.clip) {
        setError(result.message ?? "That take could not be used.");
        return;
      }
      const clip = result.clip;
      setSaved({ clip, url: null });
      onSaved();
      const url = await studioApi.clipAudioUrl(voiceId, clip.index);
      setSaved({ clip, url });
    });

  const cancel = () =>
    run(async () => {
      setRecording(false);
      await studioApi.recordCancel();
    });

  if (prompt && prompt.index === null) {
    return <p className="hint">Every sentence is recorded. Import more audio below, or redo clips from the list.</p>;
  }

  return (
    <section className="record">
      {error && <p role="alert" className="error">{error}</p>}
      <label>
        Microphone
        <select value={input ?? ""} disabled={recording} onChange={(e) => setInput(Number(e.target.value))}>
          {inputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      {inputs.length === 0 && <p className="hint">No microphone was found. Plug one in and use Refresh devices on the Live tab.</p>}

      {prompt && prompt.index !== null && (
        <>
          <p className="progress-line">
            Sentence {prompt.index + 1} of {prompt.total}. {prompt.recorded} recorded.
          </p>
          <p className="sentence">{prompt.text}</p>
        </>
      )}

      {recording ? (
        <>
          <label className="meter">
            Input level
            <progress aria-label="Input level" max={1} value={Math.min(1, status?.level ?? 0)} />
          </label>
          <p>{(status?.seconds ?? 0).toFixed(1)} s{status?.full ? ". The take is at its maximum length." : ""}</p>
          <div className="actions">
            <button onClick={stop} disabled={busy}>Stop</button>
            <button className="secondary" onClick={cancel} disabled={busy}>Cancel</button>
          </div>
        </>
      ) : saved ? (
        <>
          <p className="ok">Saved. {saved.clip.seconds.toFixed(1)} s.</p>
          {saved.url && <audio aria-label="Play back the take" controls src={saved.url} />}
          <div className="actions">
            <button onClick={() => run(() => loadPrompt())} disabled={busy}>Next sentence</button>
            <button className="secondary" onClick={record} disabled={busy}>Redo</button>
          </div>
        </>
      ) : (
        <div className="actions">
          <button onClick={record} disabled={busy || input === null || prompt === null}>Record</button>
          <button className="secondary" disabled={busy || prompt === null}
            onClick={() => run(() => loadPrompt(prompt?.index ?? undefined))}>Skip</button>
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 5: Mount it in `app/src/VoiceStudio.tsx`**

Replace the `VoiceStudio` function (keep `MinutesBar` and the constants) with:

```tsx
export function VoiceStudio({ voice, active, onChanged }: { voice: LibraryVoice; active: boolean; onChanged: () => void }) {
  return (
    <section className="studio" aria-label={`Voice ${voice.name}`}>
      <h2>{voice.name}</h2>
      <MinutesBar minutes={voice.minutes} />
      <h3>Record</h3>
      <RecordSession voiceId={voice.id} active={active} onSaved={onChanged} />
    </section>
  );
}
```

and add `import { RecordSession } from "./RecordSession";` at the top.

Append to `app/src/App.css`:

```css
h3 { font-size: 0.9rem; margin: 16px 0 6px; color: #aab0bb; text-transform: uppercase; letter-spacing: 0.04em; }
.sentence { font-size: 1.25rem; line-height: 1.4; padding: 14px; background: #22252c; border-radius: 8px; margin: 8px 0; }
.progress-line { font-size: 0.8rem; color: #aab0bb; margin: 8px 0 0; }
.ok { color: #7bd88f; }
button.secondary { background: #343843; }
audio { width: 100%; margin: 6px 0; }
```

- [ ] **Step 6: Run to verify pass**

Run from `app/`: `npm test && npx tsc --noEmit`
Expected: all pass, no type errors.

- [ ] **Step 7: Commit**

```bash
git add app/src
git commit -m "feat: guided recording session"
```

---

### Task 11: Frontend: import and the clip list

**Files:**
- Create: `app/src/dropEvents.ts`, `app/src/ImportDrop.tsx`, `app/src/ClipList.tsx`
- Modify: `app/src/VoiceStudio.tsx`, `app/src/App.css`
- Test: `app/src/ImportDrop.test.tsx`, `app/src/ClipList.test.tsx`, `app/src/VoiceStudio.test.tsx`

**Interfaces:**
- Consumes: `studioApi`, `useStudioStatus`.
- Produces: `onFilesDropped(handler: (paths: string[]) => void): Promise<() => void>` in `dropEvents.ts` (subscribes to the Tauri window's file drop and returns an unsubscribe function). `ImportDrop({ voiceId, active, onImported }: { voiceId: string; active: boolean; onImported: () => void })`. `ClipList({ voiceId, version, onChanged }: { voiceId: string; version: number; onChanged: () => void })`, which reloads whenever `version` changes.

Behaviour:
- `ImportDrop` listens for dropped files only while `active`. On a drop it starts an import for this voice and then shows progress from the polled status: "Importing file 2 of 5: name.mp3", clips added so far, and a **Cancel import** button. When the job finishes it shows "Added N clips", the number skipped as too quiet, and each error on its own line, then calls `onImported` once. A note tells the user the first import downloads the speech model (about 500 MB) and that transcripts are in English. A drop while an import is running is ignored with a short notice.
- `ClipList` shows each clip's number, where it came from, an editable transcript (saved when the field loses focus and the text changed), its length, **Play** and **Delete**. One shared audio element plays the chosen clip. Clips are shown newest first.

- [ ] **Step 1: Write `app/src/dropEvents.ts`**

```ts
import { getCurrentWebview } from "@tauri-apps/api/webview";

/** Calls handler with the file paths whenever files are dropped on the app window. */
export async function onFilesDropped(handler: (paths: string[]) => void): Promise<() => void> {
  return getCurrentWebview().onDragDropEvent((event) => {
    if (event.payload.type === "drop" && event.payload.paths.length > 0) handler(event.payload.paths);
  });
}
```

This module has no test of its own: it is a one-call wrapper over Tauri and is mocked everywhere else. Confirm with `npx tsc --noEmit` that `getCurrentWebview` and the `drop` payload shape exist in the installed `@tauri-apps/api`; if the API differs, adapt this file only and keep its exported signature.

- [ ] **Step 2: Write the failing tests**

`app/src/ImportDrop.test.tsx`:

```tsx
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ImportDrop } from "./ImportDrop";

const drop = vi.hoisted(() => ({ handler: null as null | ((paths: string[]) => void), unsubscribed: 0 }));
vi.mock("./dropEvents", () => ({
  onFilesDropped: vi.fn(async (handler: (paths: string[]) => void) => {
    drop.handler = handler;
    return () => { drop.unsubscribed += 1; drop.handler = null; };
  }),
}));
vi.mock("./studioApi", () => ({ studioApi: { importFiles: vi.fn(), importCancel: vi.fn() } }));
const status = vi.hoisted(() => ({ current: null as unknown }));
vi.mock("./useStudioStatus", () => ({ useStudioStatus: () => status.current }));
import { studioApi } from "./studioApi";

const imp = (over: object) => ({
  recording: false, level: 0, seconds: 0, full: false,
  import: { active: false, finished: false, total_files: 0, done_files: 0, current: null, clips_added: 0, skipped: 0, errors: [], ...over },
});

beforeEach(() => {
  vi.resetAllMocks();
  drop.handler = null;
  drop.unsubscribed = 0;
  status.current = null;
  vi.mocked(studioApi.importFiles).mockResolvedValue(imp({ active: true, total_files: 2 }) as never);
  vi.mocked(studioApi.importCancel).mockResolvedValue(imp({}) as never);
});

const dropFiles = (paths: string[]) => act(async () => { drop.handler?.(paths); });

describe("ImportDrop", () => {
  it("explains what to do and what the first import downloads", async () => {
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    expect(await screen.findByText(/drop audio files/i)).toBeInTheDocument();
    expect(screen.getByText(/500 MB/)).toBeInTheDocument();
    expect(screen.getByText(/English/)).toBeInTheDocument();
  });

  it("starts an import for this voice when files are dropped", async () => {
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3", "C:\\b.wav"]);
    expect(studioApi.importFiles).toHaveBeenCalledWith("v-1", ["C:\\a.mp3", "C:\\b.wav"]);
  });

  it("shows progress and a cancel button while importing", async () => {
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3", "C:\\b.wav"]);
    status.current = imp({ active: true, total_files: 2, done_files: 1, current: "b.wav", clips_added: 4 });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    expect(screen.getByText(/Importing file 2 of 2: b\.wav/)).toBeInTheDocument();
    expect(screen.getByText(/4 clips so far/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel import" }));
    expect(studioApi.importCancel).toHaveBeenCalled();
  });

  it("reports the result and errors once, and tells the parent once", async () => {
    const onImported = vi.fn();
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3", "C:\\notes.mp3"]);
    status.current = imp({ finished: true, total_files: 2, done_files: 2, clips_added: 6, skipped: 1, errors: ["Could not read notes.mp3: invalid data"] });
    rerender(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    expect(await screen.findByText(/Added 6 clips/)).toBeInTheDocument();
    expect(screen.getByText(/1 skipped as too quiet/)).toBeInTheDocument();
    expect(screen.getByText("Could not read notes.mp3: invalid data")).toBeInTheDocument();
    rerender(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    expect(onImported).toHaveBeenCalledTimes(1);
  });

  it("ignores a drop while an import is running and says so", async () => {
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3"]);
    status.current = imp({ active: true, total_files: 1, current: "a.mp3" });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await dropFiles(["C:\\b.mp3"]);
    expect(studioApi.importFiles).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/wait for the current import/i)).toBeInTheDocument();
  });

  it("shows the backend's message when the import cannot start", async () => {
    vi.mocked(studioApi.importFiles).mockRejectedValue(new Error("An import is already running"));
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3"]);
    expect(await screen.findByRole("alert")).toHaveTextContent("An import is already running");
  });

  it("does not listen for drops while its tab is hidden, and unsubscribes on unmount", async () => {
    const { rerender, unmount } = render(<ImportDrop voiceId="v-1" active={false} onImported={() => {}} />);
    await act(async () => {});
    expect(drop.handler).toBeNull();
    rerender(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    unmount();
    expect(drop.unsubscribed).toBe(1);
  });
});
```

`app/src/ClipList.test.tsx`:

```tsx
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ClipList } from "./ClipList";

vi.mock("./studioApi", () => ({
  studioApi: { clips: vi.fn(), editClip: vi.fn(), deleteClip: vi.fn(), clipAudioUrl: vi.fn() },
}));
import { studioApi } from "./studioApi";

const C1 = { index: 1, text: "First one.", seconds: 2.5, source: "recorded" as const, prompt: 0, origin: null };
const C2 = { index: 2, text: "Second won.", seconds: 4.0, source: "imported" as const, prompt: null, origin: "talk.mp3" };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(studioApi.clips).mockResolvedValue([C1, C2]);
  vi.mocked(studioApi.editClip).mockImplementation(async (_id, index, text) => ({ ...(index === 1 ? C1 : C2), text }));
  vi.mocked(studioApi.deleteClip).mockResolvedValue({ deleted: true });
  vi.mocked(studioApi.clipAudioUrl).mockImplementation(async (_id, index) => `http://x/${index}`);
});

const rows = async () => within(await screen.findByRole("list", { name: "Clips" })).getAllByRole("listitem");

describe("ClipList", () => {
  it("lists clips newest first with source and length", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const items = await rows();
    expect(items).toHaveLength(2);
    expect(within(items[0]).getByDisplayValue("Second won.")).toBeInTheDocument();
    expect(items[0]).toHaveTextContent("talk.mp3");
    expect(items[0]).toHaveTextContent("4.0 s");
    expect(items[1]).toHaveTextContent(/recorded/i);
  });

  it("says when there are no clips yet", async () => {
    vi.mocked(studioApi.clips).mockResolvedValue([]);
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    expect(await screen.findByText(/no clips yet/i)).toBeInTheDocument();
  });

  it("saves an edited transcript when the field loses focus", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const field = within((await rows())[0]).getByLabelText("Transcript of clip 2");
    await userEvent.clear(field);
    await userEvent.type(field, "Second one.");
    await userEvent.tab();
    expect(studioApi.editClip).toHaveBeenCalledWith("v-1", 2, "Second one.");
  });

  it("does not save when the text is unchanged", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[0]).getByLabelText("Transcript of clip 2"));
    await userEvent.tab();
    expect(studioApi.editClip).not.toHaveBeenCalled();
  });

  it("restores the text and shows the error when saving fails", async () => {
    vi.mocked(studioApi.editClip).mockRejectedValue(new Error("A transcript can be at most 500 characters"));
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const field = within((await rows())[0]).getByLabelText("Transcript of clip 2");
    await userEvent.type(field, " extra");
    await userEvent.tab();
    expect(await screen.findByRole("alert")).toHaveTextContent("at most 500 characters");
    expect(field).toHaveValue("Second won.");
  });

  it("plays the chosen clip in the shared player", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Play clip 1" }));
    expect(await screen.findByLabelText("Clip player")).toHaveAttribute("src", "http://x/1");
  });

  it("deletes a clip, removes it from the list and tells the parent", async () => {
    const onChanged = vi.fn();
    render(<ClipList voiceId="v-1" version={0} onChanged={onChanged} />);
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Delete clip 2" }));
    expect(studioApi.deleteClip).toHaveBeenCalledWith("v-1", 2);
    await vi.waitFor(async () => expect(await rows()).toHaveLength(1));
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it("reloads when the version changes", async () => {
    const { rerender } = render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await rows();
    rerender(<ClipList voiceId="v-1" version={1} onChanged={() => {}} />);
    await vi.waitFor(() => expect(studioApi.clips).toHaveBeenCalledTimes(2));
  });

  it("shows a load error", async () => {
    vi.mocked(studioApi.clips).mockRejectedValue(new Error("Unknown voice"));
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Unknown voice");
  });
});
```

`app/src/VoiceStudio.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MinutesBar, VoiceStudio } from "./VoiceStudio";

vi.mock("./RecordSession", () => ({
  RecordSession: ({ onSaved }: { onSaved: () => void }) => <button onClick={onSaved}>record saved</button>,
}));
vi.mock("./ImportDrop", () => ({
  ImportDrop: ({ onImported }: { onImported: () => void }) => <button onClick={onImported}>import done</button>,
}));
vi.mock("./ClipList", () => ({
  ClipList: ({ version, onChanged }: { version: number; onChanged: () => void }) => (
    <button onClick={onChanged}>clips v{version}</button>
  ),
}));

const VOICE = { id: "v-1", name: "Alpha", created: "1", minutes: 12.3, clips: 80 };

describe("MinutesBar", () => {
  it("shows minutes against the usable and good marks", () => {
    render(<MinutesBar minutes={4.25} />);
    expect(screen.getByLabelText("Recorded minutes")).toHaveValue(4.25);
    expect(screen.getByText(/4\.3 min of clean audio/)).toBeInTheDocument();
    expect(screen.queryByText(/usable voice\./)).not.toBeInTheDocument();
  });

  it("says when there is enough for a usable voice and for a good one", () => {
    const { rerender } = render(<MinutesBar minutes={10} />);
    expect(screen.getByText(/Enough for a usable voice\./)).toBeInTheDocument();
    rerender(<MinutesBar minutes={45} />);
    expect(screen.getByText(/Plenty for a good voice\./)).toBeInTheDocument();
    expect(screen.getByLabelText("Recorded minutes")).toHaveValue(30);
  });
});

describe("VoiceStudio", () => {
  it("refreshes the clip list and the parent after a saved take, an import, or a clip change", async () => {
    const onChanged = vi.fn();
    render(<VoiceStudio voice={VOICE} active onChanged={onChanged} />);
    expect(screen.getByRole("heading", { name: "Alpha" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "record saved" }));
    expect(screen.getByRole("button", { name: "clips v1" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "import done" }));
    expect(screen.getByRole("button", { name: "clips v2" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "clips v2" }));
    expect(onChanged).toHaveBeenCalledTimes(3);
  });
});
```

- [ ] **Step 3: Run to verify failure**

Run from `app/`: `npm test`
Expected: FAIL, cannot resolve `./ImportDrop` and `./ClipList`.

- [ ] **Step 4: Implement `app/src/ImportDrop.tsx`**

```tsx
import { useEffect, useRef, useState } from "react";
import { onFilesDropped } from "./dropEvents";
import { studioApi } from "./studioApi";
import { useStudioStatus } from "./useStudioStatus";

export function ImportDrop({ voiceId, active, onImported }: { voiceId: string; active: boolean; onImported: () => void }) {
  const [started, setStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const status = useStudioStatus(active && started);
  const job = status?.import ?? null;
  const running = started && !(job?.finished ?? false);
  const runningRef = useRef(running);
  runningRef.current = running;
  const reported = useRef(false);

  useEffect(() => {
    if (!active) return;
    let unsubscribe: (() => void) | null = null;
    let gone = false;
    const handle = (paths: string[]) => {
      if (runningRef.current) {
        setNotice("Wait for the current import to finish before dropping more files.");
        return;
      }
      setError(null);
      setNotice(null);
      reported.current = false;
      studioApi
        .importFiles(voiceId, paths)
        .then(() => setStarted(true))
        .catch((e: Error) => setError(e.message));
    };
    onFilesDropped(handle).then((off) => {
      if (gone) off();
      else unsubscribe = off;
    });
    return () => {
      gone = true;
      unsubscribe?.();
    };
  }, [active, voiceId]);

  useEffect(() => {
    if (started && job?.finished && !reported.current) {
      reported.current = true;
      onImported();
    }
  }, [started, job?.finished, onImported]);

  return (
    <section className="import">
      {error && <p role="alert" className="error">{error}</p>}
      <div className="dropzone">
        <p>Drop audio files anywhere on this window to add them to this voice.</p>
        <p className="hint">
          Each file is cut into clips at pauses and transcribed in English. The first import downloads the speech
          model, about 500 MB.
        </p>
      </div>
      {notice && <p className="hint">{notice}</p>}
      {started && job && !job.finished && (
        <div>
          <p>
            Importing file {Math.min(job.done_files + 1, job.total_files)} of {job.total_files}
            {job.current ? `: ${job.current}` : ""}. {job.clips_added} clips so far.
          </p>
          <progress max={job.total_files} value={job.done_files} aria-label="Import progress" />
          <div className="actions">
            <button className="secondary" onClick={() => studioApi.importCancel().catch((e: Error) => setError(e.message))}>
              Cancel import
            </button>
          </div>
        </div>
      )}
      {started && job?.finished && (
        <div>
          <p className="ok">
            Added {job.clips_added} clips.{job.skipped > 0 ? ` ${job.skipped} skipped as too quiet.` : ""}
          </p>
          {job.errors.length > 0 && (
            <ul className="errors">
              {job.errors.map((line, i) => <li key={i}>{line}</li>)}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 5: Implement `app/src/ClipList.tsx`**

```tsx
import { useEffect, useState } from "react";
import { Clip, studioApi } from "./studioApi";

function ClipRow({ voiceId, clip, onPlay, onDeleted, onError }: {
  voiceId: string;
  clip: Clip;
  onPlay: (index: number) => void;
  onDeleted: (index: number) => void;
  onError: (message: string | null) => void;
}) {
  const [saved, setSaved] = useState(clip.text);
  const [text, setText] = useState(clip.text);

  const save = async () => {
    if (text.trim() === saved) return;
    try {
      const next = await studioApi.editClip(voiceId, clip.index, text.trim());
      setSaved(next.text);
      setText(next.text);
      onError(null);
    } catch (e) {
      setText(saved);
      onError((e as Error).message);
    }
  };

  const remove = async () => {
    try {
      await studioApi.deleteClip(voiceId, clip.index);
      onDeleted(clip.index);
    } catch (e) {
      onError((e as Error).message);
    }
  };

  return (
    <li>
      <span className="tag">
        #{clip.index} {clip.source === "imported" ? (clip.origin ?? "imported") : "recorded"} · {clip.seconds.toFixed(1)} s
      </span>
      <input aria-label={`Transcript of clip ${clip.index}`} value={text} maxLength={500}
        onChange={(e) => setText(e.target.value)} onBlur={save} />
      <button className="secondary" aria-label={`Play clip ${clip.index}`} onClick={() => onPlay(clip.index)}>Play</button>
      <button className="danger" aria-label={`Delete clip ${clip.index}`} onClick={remove}>Delete</button>
    </li>
  );
}

export function ClipList({ voiceId, version, onChanged }: { voiceId: string; version: number; onChanged: () => void }) {
  const [clips, setClips] = useState<Clip[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);

  useEffect(() => {
    let gone = false;
    studioApi
      .clips(voiceId)
      .then((list) => {
        if (!gone) {
          setClips(list);
          setError(null);
        }
      })
      .catch((e: Error) => !gone && setError(e.message));
    return () => {
      gone = true;
    };
  }, [voiceId, version]);

  const play = async (index: number) => {
    try {
      setPlaying(await studioApi.clipAudioUrl(voiceId, index));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const deleted = (index: number) => {
    setClips((list) => (list ?? []).filter((c) => c.index !== index));
    onChanged();
  };

  return (
    <section className="clips">
      {error && <p role="alert" className="error">{error}</p>}
      {playing && <audio aria-label="Clip player" controls autoPlay src={playing} />}
      {clips !== null && clips.length === 0 && <p className="hint">No clips yet. Record a sentence or drop in audio files.</p>}
      {clips !== null && clips.length > 0 && (
        <ul aria-label="Clips" className="clip-list">
          {[...clips].sort((a, b) => b.index - a.index).map((clip) => (
            <ClipRow key={clip.index} voiceId={voiceId} clip={clip} onPlay={play} onDeleted={deleted} onError={setError} />
          ))}
        </ul>
      )}
    </section>
  );
}
```

- [ ] **Step 6: Finish `app/src/VoiceStudio.tsx`**

Replace the `VoiceStudio` function with:

```tsx
export function VoiceStudio({ voice, active, onChanged }: { voice: LibraryVoice; active: boolean; onChanged: () => void }) {
  const [version, setVersion] = useState(0);
  const changed = useCallback(() => {
    setVersion((v) => v + 1);
    onChanged();
  }, [onChanged]);

  return (
    <section className="studio" aria-label={`Voice ${voice.name}`}>
      <h2>{voice.name}</h2>
      <MinutesBar minutes={voice.minutes} />
      <h3>Record</h3>
      <RecordSession voiceId={voice.id} active={active} onSaved={changed} />
      <h3>Import</h3>
      <ImportDrop voiceId={voice.id} active={active} onImported={changed} />
      <h3>Clips</h3>
      <ClipList voiceId={voice.id} version={version} onChanged={onChanged} />
    </section>
  );
}
```

with these imports at the top of the file:

```tsx
import { useCallback, useState } from "react";
import { ClipList } from "./ClipList";
import { ImportDrop } from "./ImportDrop";
import { RecordSession } from "./RecordSession";
import { LibraryVoice } from "./studioApi";
```

Note the test expects a clip deletion to call the parent's `onChanged` without bumping `version` (the list already removed the row itself), while a saved take or a finished import bumps `version` so the list reloads.

Append to `app/src/App.css`:

```css
.dropzone { border: 2px dashed #343843; border-radius: 8px; padding: 12px; text-align: center; }
.dropzone p { margin: 4px 0; }
.errors { color: #ffb4bb; font-size: 0.8rem; padding-left: 18px; }
.clip-list { list-style: none; padding: 0; }
.clip-list li { display: grid; grid-template-columns: 1fr auto auto; gap: 6px; align-items: center; margin-bottom: 8px; }
.clip-list .tag { grid-column: 1 / -1; font-size: 0.72rem; color: #7d8491; }
.clip-list input { padding: 6px; background: #22252c; color: inherit; border: 1px solid #343843; border-radius: 6px; }
.clip-list button { padding: 4px 10px; font-size: 0.8rem; }
```

- [ ] **Step 7: Run to verify pass**

Run from `app/`: `npm test && npx tsc --noEmit && npm run build`
Expected: all tests pass, no type errors, the build succeeds.

- [ ] **Step 8: Commit**

```bash
git add app/src
git commit -m "feat: audio import by drag and drop, and the clip list"
```

---

### Task 12: Documentation and the full check

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/notes/2026-10-13-stage-1-follow-ups.md` (append a "Stage 2" section)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Run every automated suite**

```bash
backend/.venv/Scripts/python -m pytest backend/tests -q
cd app && npm test && npx tsc --noEmit && npm run build && cd src-tauri && cargo test
```

Expected: all pass with no warnings. `test_rvc_real.py` passes rather than skips, which shows live mode still has DirectML after the new packages.

- [ ] **Step 2: Update `README.md`**

Read the README and the code first, then make these changes so every statement matches the code:

- In the introduction and status section: the app now does live voice change and dataset building (recording and import). Training and text to speech are still later stages.
- Setup: replace the backend install command with the two commands from Task 1 Step 3, with one sentence explaining why `faster-whisper` is installed with `--no-deps`, and a warning never to install the plain `onnxruntime` package into this environment.
- A new section "Building a voice dataset" covering: the Voices tab; creating a voice; guided recording (Record, Stop, Cancel, Skip, Redo, Next sentence, the four rejection reasons in plain words); the minutes bar and its 10 and 30 minute marks; importing by dropping files on the window, the accepted file types (`.wav .flac .mp3 .ogg .opus .m4a .aac .wma`), splitting at pauses into clips of at most 15 seconds, English transcription, the one-time model download of about 500 MB, and that transcripts should be checked and corrected in the clip list; editing and deleting clips; that deleting a voice moves it to `data/trash/`.
- Where the data lives: `data/voices/<id>/clips/` with `NNNN.orig.wav` (48 kHz), `NNNN.wav` (16 kHz) and `NNNN.txt`, the per-voice `voice.json`, and the `data/voices.json` summary.
- Settings: `whisper_model` in `data/settings.json` (default `small.en`; a model without `.en` detects the language).
- Behaviour notes: recording and live mode cannot run at the same time; Refresh devices is unavailable while recording.
- Privacy: recordings and imports stay on this computer; the only network use is the one-time model download.
- A "Stage 2 first run checklist" for the human: create a voice; record a sentence and hear it back; trigger each rejection (stay silent, speak very quietly, shout into the mic, press Stop at once); Skip and Redo; watch the minutes bar grow; drop a real speech file and check the clips and transcripts; drop a non-audio file and see it reported; edit a transcript and confirm the `.txt` file changed; delete a clip; delete a voice and find it in `data/trash/`; confirm recording is refused while live mode runs and the reverse.
- Known limitations: transcripts are English only by default; transcription runs on the CPU; the first import waits for the model download with no separate progress bar; dataset files are not used by live mode until training exists.

- [ ] **Step 3: Append to the follow-ups note**

Add a "Stage 2" section to `docs/superpowers/notes/2026-10-13-stage-1-follow-ups.md` listing what a human has not yet verified (everything in the first run checklist) and any decisions or gaps found during the build.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/notes/2026-10-13-stage-1-follow-ups.md
git commit -m "docs: recording and import in the README, stage 2 checklist"
```
