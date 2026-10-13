# Stage 4: Text to Speech Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Type text, pick one of your voices and a speed/quality preset, and hear it spoken in that voice through the headphones, the virtual mic, or both, starting before a long text has finished synthesising, with an option to keep the result as a WAV file.

**Architecture:** Speech is synthesised by F5-TTS (zero-shot from one of the voice's own recorded clips) in a long-lived worker process with its own Python environment (`tts/.venv`, AMD ROCm PyTorch), speaking JSON lines over stdin and stdout, one sentence per model call. The backend's `TtsManager` owns the worker, turns each finished sentence into 48 kHz audio, optionally passes it through the voice's trained RVC model with the existing DirectML engine (the "Best" preset), and feeds a small playback queue that writes to the chosen output devices. A new Speak tab drives it.

**Tech Stack:** Python 3.12, torch 2.9.1+rocm7.2.1 (AMD wheels), `f5-tts==1.1.22`, Vocos; the existing backend (FastAPI, onnxruntime-directml, sounddevice, soxr); React 19, TypeScript, Vitest. No new backend or npm packages.

**Spec:** `docs/superpowers/specs/2026-10-13-ai-voice-design.md` (sections "Text to speech", "Failure handling", "Build stages" item 4)

**Spike findings (required reading for every task):** `docs/superpowers/notes/2026-10-13-stage-4-tts-spike.md`. It holds the exact install recipe with hashes, the environment variables, measured speeds, the three import shims the worker needs, the streaming, reference-clip and RVC-pass findings, a proposed worker protocol, and the open risks. A working throwaway prototype of the worker is in the gitignored folder `spike/stage4/scripts/` (`tts_worker.py`, `f5lib.py`, `worker_client.py`, `rvc_pass.py`, `stream.py`): read it and port what the tasks name; never import from that folder.

## How this plan differs from the spec, and why

- **One synthesis engine, F5-TTS, for all three presets.** The spec named GPT-SoVITS for the "Best" preset. The spike found GPT-SoVITS would need a third environment, has several blockers on this GPU's PyTorch build, and its fine-tuning would take weeks with real dead-end risk. "Best" is instead F5-TTS at full steps followed by the voice's own trained RVC model, which reuses stage 3's models and stage 1's engine. GPT-SoVITS fine-tuning is not built.
- **Presets:** Fast is 8 sampling steps, Balanced is 16 (measured as accurate as 32 and twice as fast), Best is 32 steps plus the RVC pass and is offered only for voices with a trained model.
- **TTS runs natively on Windows in its own environment** (the spec's fallback placed it in WSL2).
- **This plan gives interfaces, behaviour and tests rather than complete listings**, as stage 3 did, because the worker must match a real model the spike already scripted.

## Global Constraints

- Do not modify stage 1's live audio files (`backend/aivoice/processor.py`, `live.py`, `router.py`, `ringbuffer.py`, `engines/`), stage 2's `library.py` data-safety code, or stage 3's `trainer/` and `training.py`, beyond the small additions this plan names. All existing tests keep passing unchanged.
- The backend environment gets no new packages and never gains torch. F5-TTS runs only in `tts/.venv`. The RVC pass runs in the backend with the existing `RvcEngine`.
- The TTS environment installs exactly: AMD's seven ROCm 7.2.1 wheels (hash-pinned, same as the trainer), `f5-tts==1.1.22` without its declared dependencies, and the packages pinned in `tts/requirements.lock`. Model files: the four files listed in the spike notes, each verified by SHA-256 before use.
- Model licence: the F5-TTS weights are CC-BY-NC-4.0 (non-commercial). The README and the setup script's output say so plainly.
- Synthesis makes no network requests. The worker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- One activity at a time: text to speech is refused while live mode, a recording, or training is running; live mode, recording and training are refused while speech is being synthesised or played. Starting training stops the TTS worker (both need the GPU).
- The reference clip for a voice is one of its own recorded or imported clips between 3 and 12 seconds with its transcript; clips of 5 to 10 seconds are preferred. Reference plus generated audio stays under 30 seconds per model call.
- Text limits: 1 to 5000 characters per request.
- All pipeline audio handed to playback is mono float32 at 48000 Hz.
- User recordings are never modified. Synthesised audio is written only under `data/tts/`.
- All user-facing failures raise `ValueError` (HTTP 400 with a readable `detail`).
- Paths contain a space. Every subprocess call passes arguments as a list.
- Run commands from the repo root in Git Bash. Backend tests: `backend/.venv/Scripts/python -m pytest backend/tests -q -W error`. Frontend from `app/`: `npm test`, `npx tsc --noEmit`.

## Review Focus

1. **The model fails silently.** A too-long reference, a too-long piece, or text the model cannot say produces babble, a dropped last word, or near-silence with no error. The pipeline must enforce the length limits itself and must detect and report an empty or near-silent piece instead of playing nothing. Tests in Task 2 and Task 3.
2. **Text that is not prose:** one very long sentence with no punctuation, only punctuation or whitespace, emoji, URLs, digits, several blank lines, 5000 characters. Splitting must always yield pieces within the limit and never an empty piece. Test in Task 2.
3. **Stop pressed mid-sentence, or a new request while one is speaking.** Audio stops within a fraction of a second, the worker stays usable, nothing keeps playing, and files of the stopped job are cleaned up. Tests in Task 5 and Task 6.
4. **The worker dies or hangs** (out of memory, killed, the app quits). No process is left on the GPU; the next request starts a fresh worker; the user sees a clear error for the request that was lost. Test in Task 6.
5. **The output device disappears or the voice changes underneath** (device unplugged mid-playback; the voice's reference clip or model deleted between requests). A clear error, no crash, no stuck "speaking" state. Tests in Task 5 and Task 6.

## File Structure

```
tts/
  requirements.txt        direct dependencies of the worker (from the spike)
  requirements.lock       pinned versions of everything except the AMD wheels
  pretrained.json         model files: url, local path, bytes, sha256
  setup.sh                builds tts/.venv and downloads models (idempotent)
  verify.py               checks the environment, prints a JSON report
  shims.py                the import shims this torch build needs
  tts_worker.py           the worker: JSON lines in, JSON lines and WAV files out
  .venv/  models/  .cache/  built by setup.sh (gitignored)
backend/aivoice/
  tts_text.py             split text into speakable pieces
  tts_audio.py            trim, fade, limit, resample, join
  tts_reference.py        choose a voice's reference clip
  tts_rvc.py              offline RVC pass in a fixed window
  playback.py             queue player to output devices, save to WAV
  tts.py                  TtsManager: worker lifecycle and the speak pipeline
  api_tts.py              /tts routes
  api.py, api_library.py, api_training.py, __main__.py   (modify) mount, exclusion
backend/tests/
  test_tts_text.py test_tts_audio.py test_tts_reference.py test_tts_rvc.py
  test_playback.py test_tts_worker.py test_tts.py test_api_tts.py test_tts_setup.py
  test_tts_real.py        real end-to-end run, opt-in
app/src/
  ttsApi.ts  useTtsStatus.ts  SpeakPanel.tsx
  App.tsx (modify: third tab)
data/tts/                 temporary pieces and saved WAV files
```

Tests for `tts/*.py` live in `backend/tests/` and load the modules by file path, as the trainer's tests do, because those modules must be testable without the TTS environment (the model is injected).

---

### Task 1: TTS environment

**Files:**
- Create: `tts/requirements.txt`, `tts/requirements.lock`, `tts/pretrained.json`, `tts/setup.sh`, `tts/verify.py`
- Modify: `.gitignore` (`tts/.venv/`, `tts/models/`, `tts/.cache/`), `.gitattributes` (`tts/setup.sh text eol=lf`)
- Test: `backend/tests/test_tts_setup.py`

**Interfaces:**
- Produces: `tts/.venv` (Python 3.12, `torch.__version__` ending `+rocm7.2.1`, GPU available, `f5_tts` importable after the shims), `tts/models/F5TTS_v1_Base/{model_1250000.safetensors,vocab.txt}`, `tts/models/vocos-mel-24khz/{config.yaml,pytorch_model.bin}`, caches under `tts/.cache/`. `tts/verify.py` prints `{"ok": bool, "checks": [{"id", "ok", "message"}]}` with ids `venv`, `torch_gpu`, `f5_package`, `models`; `--fast` skips `torch_gpu`; standard library only; `run_checks(root, run=subprocess.run)` is injectable.

- [ ] Copy the direct requirements from `spike/stage4/requirements_f5_infer.txt` (dropping spike-only `psutil` and `huggingface_hub` unless the worker needs them) and build `requirements.lock` from the spike's `logs/pip_freeze.txt`, excluding torch, torchaudio, torchvision, any `rocm` package and `f5-tts` itself (installed separately with `--no-deps` at the pinned version).
- [ ] Tests first (as `backend/tests/test_trainer_setup.py` does for the trainer): the manifest has exactly the four model files with the spike's exact bytes and SHA-256; the lock contains no torch or rocm package and pins `numpy` at 2 or newer; each verify check fails with a message naming the fix when its probe is unhealthy; `verify.py` is standard library only.
- [ ] Write `tts/setup.sh` modelled on `trainer/setup.sh` (read it): same hash-pinned AMD wheel URLs, `--no-python-downloads`, idempotent, downloads to `.part` then verifies SHA-256 then renames, works from any directory and with the space in the path, reuses an existing uv cache when one is present (`trainer/.cache/uv`, else `spike/stage3/cache/uv`) by pointing `UV_CACHE_DIR` at it for the run, installs the lock under a constraint that pins the three ROCm torch packages, then `f5-tts==1.1.22` with `--no-deps`, prints the weights licence notice, and ends by running `tts/verify.py`.
- [ ] Run it on this machine; record wall time, the verify output and folder sizes; run it a second time (must be a quick no-op).
- [ ] Commit: `feat: reproducible text-to-speech environment`.

---

### Task 2: Text splitting and audio post-processing

Pure backend modules (numpy, soxr; no devices, no model).

**Files:**
- Create: `backend/aivoice/tts_text.py`, `backend/aivoice/tts_audio.py`
- Test: `backend/tests/test_tts_text.py`, `backend/tests/test_tts_audio.py`

**Interfaces:**
- `tts_text.MAX_TEXT = 5000`. `tts_text.clean(text: str) -> str` (normalise newlines and runs of whitespace, strip control characters; raises `ValueError` for empty or over-long text). `tts_text.split(text: str, max_chars: int) -> list[str]`: sentences end at `. ! ?` (and their combinations, with closing quotes or brackets) followed by whitespace or end of text, and at blank lines; a sentence longer than `max_chars` UTF-8 bytes is split at `; : ,` then at spaces, then hard; pieces are stripped; no piece is empty or lacks a letter or digit (such pieces are dropped); a piece that does not end in sentence punctuation gets a full stop appended (the model reads more steadily); order preserved. `max_chars` below 20 raises `ValueError`.
- `tts_audio.SAMPLE_RATE = 48000`. `tts_audio.trim_edges(audio, sr, rel_db=-30.0, frame_ms=20, keep_ms=30) -> np.ndarray` (trim leading and trailing frames quieter than `rel_db` below the piece's loudest frame, keeping a little context; all-quiet input returns empty). `tts_audio.finish(audio, sr, fade_ms=10, peak=0.97) -> np.ndarray` (resample to 48 kHz with soxr when needed, fade both edges, scale down only if the peak exceeds the limit, clean NaN and infinity). `tts_audio.is_silent(audio, sr) -> bool` (shorter than 0.15 s, or peak below 0.005). `tts_audio.join(pieces: list[np.ndarray], pause_ms=300, sr=48000) -> np.ndarray`. `tts_audio.max_chars_for(ref_text: str, ref_seconds: float, speed: float = 1.0, limit_seconds: float = 30.0) -> int` giving the largest piece, in UTF-8 bytes, such that reference plus generated audio stays under the limit, using the package's own duration formula from the spike notes with a 15 percent safety margin.

- [ ] Tests first. For `split`: ordinary prose; abbreviations are not handled specially (document it); a 3000-character sentence with no punctuation; only punctuation; emoji and URLs; digits and decimals ("3.5 metres" must not split inside the number); blank lines; exactly 5000 characters; every result within `max_chars` bytes, none empty, concatenation (ignoring whitespace and added full stops) equals the cleaned input's speakable content. For audio: trimming a piece with 1.5 s of low room tone before speech; an all-quiet piece; NaN input; peak 1.37 scaled to 0.97 and a quiet piece left unscaled; 24 kHz input resampled to 48 kHz with the right length; fades at both edges; join lengths; `max_chars_for` against the spike's worked example (the shipped reference gives 178 before the margin) and its monotonic behaviour.
- [ ] Implement, run, commit: `feat: text splitting and speech post-processing`.

---

### Task 3: The TTS worker

**Files:**
- Create: `tts/shims.py`, `tts/tts_worker.py`
- Test: `backend/tests/test_tts_worker.py`

**Interfaces:**
- Start: `tts/.venv/Scripts/python.exe -u tts/tts_worker.py --model-dir tts/models [--parent-pid N]`. Environment variables per section 4 of the spike notes are set by the launcher (Task 6); the worker also sets the safe ones itself if missing.
- The protocol is the one in section 10.2 of the spike notes (commands `set_reference`, `synthesize`, `cancel`, `shutdown`; events `ready`, `reference`, `started`, `audio`, `done`, `cancelled`, `error`), with these additions: `synthesize` takes `pieces: list[str]` (the backend does the splitting, Task 2) instead of raw text, plus `nfe` (8, 16 or 32), `speed`, `seed`; every `audio` event carries `index`, `path`, `seconds`, `sample_rate` and the piece's `text`; the worker writes each piece as a float32 WAV through a `.part` rename, untrimmed (the backend post-processes); `error` events carry a `code` from `bad_request`, `bad_json`, `unknown_command`, `out_of_memory`, `internal`, `piece_failed`.
- The module exposes `serve(stdin, stdout, engine, parent_alive=None) -> int` where `engine` is an object with `load()`, `set_reference(wav_path, text) -> float` and `synthesize_piece(text, nfe, speed, seed, should_stop) -> (np.ndarray, int)`, so tests inject a fake engine and never import torch. The real engine class lives in the same file and imports torch and `f5_tts` only inside `load()`, after `shims.apply()`.
- `shims.apply(stub_training_imports: bool = True)` installs the `torch.distributed.ReduceOp` shim and, optionally, the import stubs from section 10.3 of the spike notes.

Behaviour (each a test with the fake engine and in-memory streams):

1. Emits `ready` once after `engine.load()`; a load failure emits `error` with code `internal` and exits non-zero.
2. `set_reference` validates the file exists, reads its length, and refuses a clip under 3 s or over 12 s with `bad_request` and a readable message; an empty transcript is refused.
3. `synthesize` before any reference is `bad_request`. With a reference it emits `started` with the piece count, one `audio` per piece in order, then `done`. Files exist when their event is emitted.
4. A piece whose audio is empty, all NaN, or shorter than 0.1 s is retried once with a different seed; if it fails again the job ends with `error` code `piece_failed` naming the piece index, and earlier pieces' events stand.
5. `cancel` during synthesis stops before the next piece (and inside a piece through `should_stop`), emits `cancelled`, and the worker accepts the next request. A cancel for an unknown or finished id is ignored.
6. Malformed JSON and unknown commands produce `bad_json` and `unknown_command` and leave the worker running.
7. An out-of-memory error from the engine emits `out_of_memory`, and the worker stays alive for the next request.
8. Nothing but JSON lines is ever written to stdout (library output is redirected to stderr); every line is flushed.
9. `shutdown`, end of input, or the parent process disappearing ends the worker with exit code 0 after finishing nothing further.
10. Only one synthesis runs at a time; a second `synthesize` while one runs is `bad_request`.

- [ ] Tests first; implement by porting `spike/stage4/scripts/tts_worker.py` and `f5lib.py` (forcing half precision, loading models from explicit local paths, reading audio with soundfile, never calling `torchaudio.load` or the package's reference preprocessing); run; commit `feat: text-to-speech worker process`.

---

### Task 4: Reference clip selection and the offline RVC pass

**Files:**
- Create: `backend/aivoice/tts_reference.py`, `backend/aivoice/tts_rvc.py`
- Test: `backend/tests/test_tts_reference.py`, `backend/tests/test_tts_rvc.py`

**Interfaces:**
- `tts_reference.choose(voice: Voice) -> Clip` picks the reference clip: only clips with a non-empty transcript and a length from 3.0 to 12.0 s; recorded clips before imported ones; then the length closest to 8 s; then the lowest index. Raises `ValueError("This voice has no clip between 3 and 12 seconds with a transcript. Record a sentence for it first.")` when none qualifies. `tts_reference.describe(voice) -> dict` returns `{"ok": bool, "clip": index | None, "seconds": float | None, "message": str}` for the UI.
- `tts_rvc.WINDOW_SECONDS = 20`. `tts_rvc.OfflineConverter(engine, params, sample_rate=48000)` with `convert(piece: np.ndarray) -> np.ndarray`: copies the piece into a zero buffer of exactly the window length, calls `engine.convert(buffer, 48000, 0.0, params)` once, and returns the result cut to the piece's length (the engine keeps the end aligned, so place the piece at the END of the buffer and cut from the end; verify with a test that uses a time-shifting fake engine). A piece longer than the window is split at its quietest 20 ms frame in the middle third and the parts converted separately, joined without overlap. `warm()` runs one silent window. The engine is built by the same factory live mode uses; `params` is the `high_quality` preset with `index_rate` kept.

- [ ] Tests first: selection order and every exclusion; a voice with no qualifying clip; the converter keeps length and alignment for pieces of 1, 7, 19.9 and 20 s; a 45 s piece is split and rejoined to the same length; every engine call receives exactly the window length (the DirectML fixed-shape rule); a non-finite or wrong-length engine result raises a clear error instead of returning it.
- [ ] Implement, run, commit: `feat: reference clip selection and offline voice conversion`.

---

### Task 5: Playback queue

**Files:**
- Create: `backend/aivoice/playback.py`
- Test: `backend/tests/test_playback.py`

**Interfaces:**
- `Player(stream_factory, sample_rate=48000)` using the existing `SoundDeviceStreams.output(device, samplerate, fill)`. `start(devices: dict[str, int], gains: dict[str, float] | None = None)`, `enqueue(audio: np.ndarray, pause_ms: int = 300)`, `finish()` (no more audio is coming), `stop()` (silence at once, discard the queue, close streams), `wait(timeout) -> bool`, `status() -> {"playing": bool, "queued_seconds": float, "played_seconds": float, "error": str | None}`. Each output has its own buffer so one device cannot starve another; the fill callback never blocks and returns silence when empty; playback ends, and streams close, when the queue is empty after `finish()`.
- `save_wav(path, pieces, pause_ms=300)` writes the joined audio with `audiofile.write_wav`.

Behaviour (tests with the fake stream factory pattern from `backend/tests/test_live.py`): pieces play in order on every output with the pause between them; an output that falls behind does not delay the other; `stop()` mid-piece yields silence on the very next callback and closes streams; a stream that goes inactive (device unplugged) sets `error` and ends playback without raising into the callback thread; enqueue after stop is ignored; start twice replaces the first session cleanly; a device that cannot be opened raises `ValueError` and leaves nothing open.

- [ ] Tests first, implement, commit: `feat: playback queue for synthesised speech`.

---

### Task 6: TTS manager

**Files:**
- Create: `backend/aivoice/tts.py`
- Test: `backend/tests/test_tts.py`

**Interfaces:**
- `TtsError(ValueError)`. `TtsManager(data, library, tts_root, player, list_devices, launch=None, verify=None, rvc_factory=None, now=time.time)`.
- `speak(voice_id, text, preset, mic_device=None, monitor_device=None, save=False) -> dict`; `stop() -> dict`; `status() -> dict`; `environment(refresh=False) -> dict`; `unload() -> None` (stop the worker; used before training and on shutdown); `shutdown()`; property `busy: bool` (a request is being synthesised or played); `is_active() -> bool` (cheap, lock-only, for route exclusion).
- `status()` keys: `engine` (`stopped`, `starting`, `ready`, `failed`), `state` (`idle`, `preparing`, `speaking`, `stopping`), `voice_id`, `preset`, `piece` and `pieces`, `error`, `saved_path`, `started`.
- Presets: `fast` (8 steps), `balanced` (16), `best` (32 plus the RVC pass; refused with a clear message when the voice has no trained model).
- Reuse, do not copy, stage 3's verified process handling in `training.py` (the worker launch without a console, the tree snapshot and verified kill, the survivor rules): import those helpers; if they need to be shared more cleanly, move them into a small module both managers import, keeping `training.py`'s tests passing unchanged.

Behaviour (each a test with a fake worker handle that speaks the protocol through in-memory pipes, a fake player and a fake converter):

1. `speak` validates in order: text cleans to 1 to 5000 characters; known preset; voice exists; voice has a reference clip (Task 4's message); Best needs a model; at least one output device, both valid outputs and different from each other; not already busy ("Already speaking. Stop it first."); environment ready (lists failing checks). Nothing starts when validation fails.
2. The worker is started on first use (state `starting` until `ready`), reused afterwards, and its reference is set only when the chosen clip (voice, clip index) differs from the one loaded.
3. Pieces come from `tts_text.split` with `tts_audio.max_chars_for` for the reference. Each `audio` event is read from its file, trimmed, checked with `is_silent` (a silent piece fails the request with "The voice engine produced no sound for sentence N"), converted through the RVC pass for Best, finished, enqueued for playback, and its temporary file deleted. `piece` advances as pieces are enqueued. Playback of the first piece starts before later pieces are synthesised.
4. With `save`, the finished pieces are also written to `data/tts/<UTC timestamp>-<voice id>.wav` when the job completes, and `saved_path` reports it. A stopped or failed job saves nothing and leaves no temporary files.
5. `stop()` cancels the worker job, stops playback at once, cleans temporary files, and returns to `idle`; the worker remains loaded. Stop when idle is harmless.
6. Worker death during a request (pipe end of file, or an `error` event with `internal`) fails the request with a clear message including the tail of the worker's stderr log, kills and verifies the process tree, and the next `speak` starts a fresh worker. A worker that does not answer within a timeout (60 s for a piece, 120 s for start-up) is treated as dead.
7. The worker is unloaded after 10 minutes without a request (checked when `status()` is polled, by the injected clock), on `unload()`, and on `shutdown()`; never while busy. Unload verifies the tree is gone the same way training's kills do.
8. A playback error (device lost) fails the request with the player's message and cancels the worker job.
9. `status()` is cheap, never raises, and never performs a kill while holding the manager's lock.
10. Start-up recovery: temporary files left under `data/tts/tmp/` by a previous run are removed on construction; saved WAV files are never touched.

- [ ] Tests first, implement, run the full backend suite, commit: `feat: text-to-speech manager with streaming playback`.

---

### Task 7: TTS API and exclusion

**Files:**
- Create: `backend/aivoice/api_tts.py`
- Modify: `backend/aivoice/api.py`, `api_library.py`, `api_training.py`, `__main__.py`
- Test: `backend/tests/test_api_tts.py`

**Interfaces:**
- Routes under `/tts`, with the same request-origin protection as `/library`: `GET /tts/status`; `GET /tts/environment?refresh=`; `GET /tts/voices` (library voices with `{"id", "name", "has_model", "reference": {"ok", "seconds", "message"}}`); `POST /tts/speak` body `{"voice_id", "text", "preset", "mic_device"?, "monitor_device"?, "save"?}`; `POST /tts/stop`.
- `create_app(..., tts=None)`; `__main__.build_app` constructs the player and the manager.
- Exclusion under the existing audio lock, using only cheap lock-only reads: speak is refused while live mode runs, a recording is in progress, or training is active; live start, recording start and training start are refused while TTS is busy; training start additionally calls `tts.unload()` before launching (outside long lock holds); device refresh is refused while TTS is busy. Deleting a voice that is being spoken is refused.
- The environment route never starts a GPU probe while live mode or training is running (same rule as `/training/environment`).
- On app shutdown, `tts.shutdown()` is called and bounded.

- [ ] Tests first (with the fakes from `test_tts.py`): happy paths; every refusal with its exact message and proof nothing started; origin protection; an app without `tts` behaves as before; `/tts/voices` contents; shutdown stops the worker.
- [ ] Implement, full backend suite, smoke test the real server (`/tts/environment`, `/tts/status`, `/tts/voices` only; never `/tts/speak`), commit: `feat: text-to-speech API`.

---

### Task 8: Speak tab

**Files:**
- Create: `app/src/ttsApi.ts`, `app/src/useTtsStatus.ts`, `app/src/SpeakPanel.tsx`
- Modify: `app/src/App.tsx`, `app/src/App.css`
- Test: `app/src/SpeakPanel.test.tsx`, `app/src/useTtsStatus.test.tsx`, additions to `App.test.tsx`

Behaviour (each a test):

1. A third tab, Speak, beside Live and Voices; all three stay mounted.
2. Voice select lists `/tts/voices`; a voice without a usable reference is shown disabled with its message; with no voices at all the panel explains that a voice needs one recorded sentence.
3. A text area with a character count against 5000; Speak is disabled when empty or over the limit.
4. Speed/quality select: Fast, Balanced, Best. Best is disabled, with the reason, for a voice without a trained model.
5. Output: "Listen on" (a device select, defaulting to the Windows default output, never a VoiceMeeter or cable device) and "Send to virtual mic" (a checkbox that appears only when a virtual mic exists); at least one must be chosen.
6. "Keep as a WAV file" checkbox; after a saved job the panel shows the file's path.
7. **Speak** starts the request; while the engine is starting it shows "Starting the voice engine. The first time takes about 20 seconds."; while speaking it shows "Speaking sentence N of M" and a **Stop** button.
8. Errors from the backend (including refusals such as live mode running) appear in a `role="alert"`; a failed request returns the panel to idle.
9. When the environment is not ready, each failing check's message is listed with "Text to speech is not set up on this computer. See Text to speech setup in the README." and **Check again**; no Speak button.
10. Polling: twice a second only while a request is active and the Speak tab is visible, plus once on becoming visible; one request at a time.
11. Ctrl+Enter in the text area speaks; Escape stops while speaking.

- [ ] Tests first, implement, `npm test`, `npx tsc --noEmit`, `npm run build`, commit: `feat: Speak tab for text to speech`.

---

### Task 9: Real end-to-end check and documentation

**Files:**
- Create: `backend/tests/test_tts_real.py`, `docs/superpowers/notes/2026-10-13-stage-4-first-real-run.md`
- Modify: `README.md`, `docs/superpowers/notes/2026-10-13-stage-1-follow-ups.md`

- [ ] Opt-in real test (`AIVOICE_RUN_TTS_TEST=1`, skipped unless `tts/verify.py --fast` passes): drive the real `TtsManager` with the real worker, a temporary data folder, a temporary library voice whose one clip is the reference WAV that ships inside the installed `f5_tts` package (copied into the voice with its transcript), a fake player that records what it is given, and a fake output device list. Speak a three-sentence text at Fast and at Balanced: assert three pieces arrive in order, each non-silent, 48 kHz, of plausible length for its text; time to first piece; the worker is reused; then `stop()` mid-request and assert it returns to idle within two seconds and the next request works; then `unload()` and assert no worker process remains. If the stage 3 spike's exported test model still exists, also run Best with it and assert the converted pieces keep their length. Transcribe one piece with the backend's Whisper model if it is installed and assert most of the words match (skip that assertion if the model is absent).
- [ ] Run it, fix what the real model reveals (each fix with a unit test), record timings and findings in the notes file.
- [ ] README: a "Text to speech setup" section (`bash tts/setup.sh`, what it downloads and how large, the non-commercial licence of the F5-TTS weights stated plainly, how to verify); a "Speaking text" section (choosing a voice, what the reference clip is and how to get a good one, the three presets and their measured speeds, outputs, saving, Stop, what happens with long text); that text to speech is unavailable while live mode, recording or training runs; that it makes no network requests; a "Stage 4 first run checklist" (set up; record or pick a clear 5 to 10 second sentence for a voice; speak a sentence at each preset and listen; speak a long paragraph and confirm it starts early; Stop mid-way; save a file and play it; send to the virtual mic and confirm another app hears it; compare Balanced against Best for likeness); known limitations (nobody has listened to any output yet; English only; numbers and abbreviations are read as the model sees them; each sentence is spoken on its own so intonation does not carry across sentences; the "Best" preset replaces the spec's GPT-SoVITS; text to speech and live mode cannot run together).
- [ ] Follow-ups note: a "Stage 4" section with decisions, open items and what a human has not verified.
- [ ] Run every suite; commit `docs: text to speech setup and use, stage 4 checklist`.
