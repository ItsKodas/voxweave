# Stage 3: Voice Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an RVC voice model from a voice's recorded clips with one button, show progress with Cancel and Resume, and make the finished model selectable in live mode.

**Architecture:** Training runs natively on Windows in its own Python environment (`trainer/.venv`, AMD's ROCm PyTorch) using a pinned, patched clone of the RVC project. A standalone orchestrator, `trainer/worker.py`, runs the pipeline stages as subprocesses and writes JSON progress lines to a file. The backend's `training` module launches the worker, tails that file, and exposes status over HTTP. The finished model is exported to ONNX with the existing exporter and stored in the voice's folder, where live mode finds it.

**Tech Stack:** Python 3.12, torch 2.9.1+rocm7.2.1 (AMD wheels), RVC WebUI commit `81eed5e8f68b6bed1789f682fe78cdd324495afc` with two patches, ffmpeg, the existing `tools/export_rvc_onnx.py`. Backend: FastAPI. Frontend: React 19, TypeScript, Vitest. No new backend or npm packages.

**Spec:** `docs/superpowers/specs/2026-10-13-ai-voice-design.md` (sections "Training", "Training worker", "Failure handling", "Build stages" item 3)

**Spike findings (required reading for every task):** `docs/superpowers/notes/2026-10-13-stage-3-training-spike.md`. It holds the exact install recipe, every pipeline command, verbatim log lines, resume and cancel behaviour, timings, both patches, and the open risks. Working throwaway scripts from the spike are in the gitignored folder `spike/stage3/scripts/` and `spike/stage3/patches/` (read them; port what the tasks name; do not import from that folder).

## How this plan differs from the spec, and why

- **Native Windows, not WSL2.** The spike proved RVC trains on the RX 7900 XTX with AMD's official PyTorch-on-Windows wheels. The WSL2 Ubuntu distro on this PC is broken, and native training needs no path translation. The spec's intent (train on the AMD GPU, progress streamed to the UI, resumable) is unchanged.
- **RVC only.** GPT-SoVITS fine-tuning moves to stage 4, where the TTS engine is chosen and proven first.
- **Checkpoints live under `trainer/rvc/logs/voice-<id>/`**, because RVC hard-codes its experiment folder. The voice's own `training/` folder holds the job record, the progress file and the worker log. Checkpoints are derived data: they are deleted after a successful run and when the voice is deleted.
- **This plan gives interfaces, behaviour and tests rather than complete listings for the worker**, because its code must match a real external pipeline. Implementers port the spike's working scripts and prove behaviour with the tests listed here plus one real end-to-end run.

## Global Constraints

- Do not modify stage 1's live audio files (`backend/aivoice/processor.py`, `live.py`, `router.py`, `ringbuffer.py`, `engines/`) or stage 2's data-safety code in `library.py` beyond the additions this plan names. All existing tests keep passing unchanged.
- The backend environment (`backend/.venv`) gets no new packages. PyTorch must never be installed into it. Training code runs only in `trainer/.venv`.
- The training environment installs exactly: AMD's seven ROCm 7.2.1 wheels, and the packages in `trainer/requirements.txt`. The RVC clone is pinned to commit `81eed5e8f68b6bed1789f682fe78cdd324495afc` with exactly the two patches in `trainer/patches/`.
- Every downloaded model file is verified against the SHA-256 recorded in the spike notes before use.
- Training and live mode never run at the same time (both need the GPU). Training and recording may not run at the same time either (one consistent rule: one audio or GPU activity at a time). Imports may run during training.
- Training makes no network requests. The worker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- A voice needs at least 5.0 minutes of clean audio to train (spec).
- Training defaults: 200 epochs, batch size 8, checkpoint every 5 epochs, 40 kHz, v2, pitch model, RMVPE pitch extraction on the GPU, mixed precision as RVC selects.
- User recordings are never modified or deleted by training. Training reads the 48 kHz `NNNN.orig.wav` files through a staging folder of hard links.
- All user-facing failures raise `ValueError` (HTTP 400 with a readable `detail`).
- Paths contain a space (`D:\Projects\AI Voice`). Every subprocess call passes arguments as a list, never a shell string.
- Run commands from the repo root in Git Bash. Backend tests: `backend/.venv/Scripts/python -m pytest backend/tests -q -W error`. Frontend from `app/`: `npm test`, `npx tsc --noEmit`.

## Review Focus

1. **The trainer is killed mid-save** (cancel, crash, power loss). The only checkpoint may be truncated. Resume must detect it and fall back to the backup copy, never silently restart from scratch. Test in Task 3.
2. **A stage exits 0 but did not do its job** (preprocess with failed files, a final model that was not written). The worker must check outputs, not exit codes alone. Test in Task 3.
3. **The app is closed, or the backend restarts, while training runs.** No orphan process may keep the GPU busy; the job shows as interrupted and can be resumed. Tests in Task 3 and Task 5.
4. **The dataset changed since the interrupted run** (clips added or deleted). Resume must not mix old features with a new clip list; it is refused with a clear message and the user starts a fresh run. Tests in Task 3 and Task 5.
5. **The training environment is missing or broken** (never set up, GPU not visible, a pretrained file missing or corrupt, ffmpeg not on PATH). Start is refused with the exact missing item; nothing crashes; live mode and recording still work. Tests in Task 5 and Task 7.

## File Structure

```
trainer/
  requirements.txt            training-only Python packages (from the spike)
  patches/rvc_rocm_dilated_conv.patch
  patches/rvc_preprocess_keep_all_tails.patch
  pretrained.json             every download: url, local path, bytes, sha256
  setup.sh                    builds the environment (idempotent)
  verify.py                   checks the environment and prints a JSON report
  rvclog.py                   pure parsers for RVC's log lines
  filelist.py                 writes filelist.txt and config.json (port of webui.py logic)
  worker.py                   the orchestrator: stages, progress file, resume, cancel
  .venv/  rvc/  .cache/       built by setup.sh (gitignored)
backend/aivoice/
  training.py                 TrainingManager: launch worker, tail progress, status, cancel
  api_training.py             /training routes
  pretrained.py               (modify) also list trained library voices
  library.py                  (modify) report which models a voice has
  api.py, __main__.py         (modify) mount routes, mutual exclusion
backend/tests/
  test_rvclog.py test_filelist.py test_worker.py test_training.py test_api_training.py
  test_worker_real.py         real end-to-end run, opt-in
app/src/
  trainingApi.ts  useTrainingStatus.ts  TrainPanel.tsx
  VoiceStudio.tsx, LivePanel.tsx, App.tsx (modify)
data/voices/<id>/
  training/job.json           the job record
  training/progress.jsonl     progress lines written by the worker
  training/worker.log         full output of every stage
  models/rvc/model.onnx, model.index, voice.json   the trained model
```

Tests for `trainer/*.py` live in `backend/tests/` and load the modules by file path (as `backend/tests/test_export_tool.py` already does for the exporter), because the trainer modules use only the standard library and must be testable without the training environment.

---

### Task 1: Training environment

Builds the environment on this machine and makes it reproducible.

**Files:**
- Create: `trainer/requirements.txt`, `trainer/patches/rvc_rocm_dilated_conv.patch`, `trainer/patches/rvc_preprocess_keep_all_tails.patch`, `trainer/pretrained.json`, `trainer/setup.sh`, `trainer/verify.py`
- Modify: `.gitignore` (add `trainer/.venv/`, `trainer/rvc/`, `trainer/.cache/`), `.gitattributes` (patches already forced to LF by `*.patch`; add `trainer/setup.sh text eol=lf`)
- Test: `backend/tests/test_trainer_setup.py`

**Interfaces:**
- Produces: a working `trainer/.venv` (Python 3.12, `torch.__version__` ending `+rocm7.2.1`, `torch.cuda.is_available()` True); `trainer/rvc/` at the pinned commit with both patches applied, the pretrained files in place, `logs/mute/`, `assets/weights/`, `assets/indices/`; `trainer/.cache/` for uv, MIOpen, numba and matplotlib caches. `trainer/verify.py` prints one JSON object `{"ok": bool, "checks": [{"id": str, "ok": bool, "message": str}]}` with check ids `venv`, `torch_gpu`, `rvc_commit`, `rvc_patches`, `pretrained`, `ffmpeg`, `exporter`, and exits 0 when `ok` else 1. It runs under any Python 3.10+ with only the standard library; the `torch_gpu` check runs the trainer's own Python as a subprocess. `--fast` skips `torch_gpu` (which takes several seconds).
- `trainer/pretrained.json` is a list of `{"url", "path", "bytes", "sha256"}` with `path` relative to `trainer/rvc/`, copied from section 2.7 of the spike notes (the seven `lj1995/VoiceConversionWebUI` entries; `mute.zip` has `"path": "../.cache/mute.zip"` and is extracted into `logs/`).

- [ ] **Step 1: Copy the proven inputs from the spike.** Copy both patch files from `spike/stage3/patches/` unchanged. Write `trainer/requirements.txt` from `spike/stage3/requirements_train_rocm.txt`, keeping its header comment and removing the two spike-only lines (`huggingface_hub`, `psutil`).

- [ ] **Step 2: Write the failing tests** in `backend/tests/test_trainer_setup.py` (load `trainer/verify.py` by path):
  - `pretrained.json` parses; has exactly seven entries; every `sha256` is 64 hex characters; every `url` starts with `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/`; the six model entries and their byte counts equal the table in the spike notes (assert the exact `(path, bytes, sha256)` set).
  - Each verify check, driven through injectable probes (the module exposes `run_checks(root: Path, run=subprocess.run, which=shutil.which) -> dict`): a missing `.venv` fails `venv` with a message naming `trainer/setup.sh`; a pretrained file with the wrong size fails `pretrained` naming that file; `which("ffmpeg")` or `which("ffprobe")` returning None fails `ffmpeg`; a `git rev-parse HEAD` result other than the pinned commit fails `rvc_commit`; a patch that `git apply --reverse --check` rejects fails `rvc_patches` naming the patch; the trainer Python printing `2.9.1+cpu False` fails `torch_gpu` with a message saying the GPU build was replaced; a missing `tools/.rvc/venv` fails `exporter`. With every probe healthy, `ok` is true and `--fast` omits `torch_gpu`.
  - `verify.py` imports nothing outside the standard library (assert by parsing its AST).

- [ ] **Step 3: Run to verify failure.** `backend/.venv/Scripts/python -m pytest backend/tests/test_trainer_setup.py -q` fails because the files do not exist.

- [ ] **Step 4: Write `trainer/pretrained.json`, `trainer/verify.py` and `trainer/setup.sh`.** `setup.sh` follows section 2 of the spike notes exactly, with these differences: everything lives under `trainer/` (`.venv`, `rvc`, `.cache`); `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME` point into `trainer/.cache/`; it is idempotent (skips a step whose result is already correct, re-applies nothing twice, re-downloads only files whose size or SHA-256 is wrong); downloads go to a `.part` file, are verified with `sha256sum`, then renamed; it ends by running `trainer/verify.py` with the trainer Python and failing if that fails. It must work from any current directory and with the space in the project path. If `spike/stage3/cache/uv` exists and `trainer/.cache/uv` does not, the script says so and uses it as the uv cache for this run by setting `UV_CACHE_DIR` to it (this avoids downloading 2 GB again; nothing is moved or deleted).

- [ ] **Step 5: Run the setup on this machine.** `bash trainer/setup.sh`. Expected: finishes with verify reporting every check ok, including `torch_gpu`. Then run it a second time: it must finish in seconds and change nothing. Record wall time and the verify output in the report. Do not change drivers or install system software; if `torch_gpu` fails, stop and report the exact diagnostics.

- [ ] **Step 6: Run the tests and commit.** All tests pass. Commit `trainer/requirements.txt`, `trainer/patches/`, `trainer/pretrained.json`, `trainer/setup.sh`, `trainer/verify.py`, `.gitignore`, `.gitattributes`, the test file: `feat: reproducible native training environment`.

---

### Task 2: Log parsing and filelist writing

Pure standard-library modules.

**Files:**
- Create: `trainer/rvclog.py`, `trainer/filelist.py`
- Test: `backend/tests/test_rvclog.py`, `backend/tests/test_filelist.py`

**Interfaces:**
- `rvclog.parse_train_line(line: str) -> dict | None` returns one of: `{"kind": "epoch_done", "epoch": int}` for a line containing `====> Epoch: N`; `{"kind": "epoch_progress", "epoch": int, "percent": int}` for `Training epoch: N [P%]` (English only; other locales return None for this kind and that is acceptable); `{"kind": "saved", "epoch": int, "path": str}` for `Saving model and optimizer state at epoch N to <path>`; `{"kind": "loss", "values": {"loss_disc": float, ...}}` for the loss line; `{"kind": "resumed", "epoch": int}` for `Loaded checkpoint '<path>' (epoch N)`; `{"kind": "final_saved", "ok": bool}` for a line containing `Saving final checkpoint:` (ok when it ends with `Success`); else None. It accepts both the stdout form (`INFO:<exp>:<message>`) and the `train.log` form (tab separated).
- `rvclog.parse_stage_line(stage: str, line: str) -> dict | None` for `preprocess`, `pitch`, `features`, `index`: returns `{"kind": "progress", "done": int, "total": int}` for the `Progress: a/b` lines, `{"kind": "completed", "success": int, "failed": int}` for `Completed | Success: a | ... Failed: b` and for preprocess's per-worker and overall lines, and `{"kind": "failed_file", "path": str}` for `[Data slicing][Failed] <path>`.
- `rvclog.is_fatal_noise(line: str) -> bool` is False for known harmless lines that contain the word Error (the MIOpen `EvaluateInvokers` line quoted in the spike notes), so a caller never treats them as failures.
- `filelist.write_filelist(rvc_root: Path, experiment: str, sample_rate: str = "40k", speaker_id: int = 0, log_interval: int | None = None, shuffle=random.shuffle) -> int` writes `logs/<experiment>/filelist.txt` and `config.json` exactly as `spike/stage3/scripts/make_filelist.py` does (a port of the logic in RVC's `webui.py`): only segments present in all four folders, two mute lines, backslashes doubled in the folder part, config copied from `configs/v1/40k.json` for 40k unless one already exists, `speaker_info` removed, `log_interval` overridden when given. Returns the number of lines written. Raises `ValueError` when no segment is present in all four folders.

- [ ] **Step 1: Write the failing tests.** For `rvclog`: one test per kind using the verbatim sample lines from section 5 of the spike notes (both the stdout and the `train.log` forms), a test that every other sample line returns None, the MIOpen noise line, and garbage input (empty string, binary-looking text, a 10,000-character line) never raising. For `filelist`: build a fake experiment tree in `tmp_path` with segments `0_0`, `0_1`, `1_0` where `1_0` lacks its feature file; assert two real lines plus two mute lines, the exact text of one line (given a known root containing a space and backslashes), deterministic order with an identity shuffle, config copied from a fake `configs/v1/40k.json`, an existing `config.json` kept (only `log_interval` changed), `speaker_info` removed, and `ValueError` when nothing matches.
- [ ] **Step 2: Run to verify failure**, implement, run to verify pass.
- [ ] **Step 3: Commit** `feat: RVC log parsers and filelist writer`.

---

### Task 3: The training worker

**Files:**
- Create: `trainer/worker.py`
- Test: `backend/tests/test_worker.py`

**Interfaces:**
- Command line: `python -P trainer/worker.py --voice-dir <data/voices/id> --job-dir <data/voices/id/training> --experiment voice-<id> --epochs 200 --batch-size 8 --save-every 5 [--resume] [--parent-pid N] --name "<display name>"`. It runs under the trainer Python but imports only the standard library plus `rvclog` and `filelist` from its own folder.
- The module exposes `run(args, runner=None, clock=time.time, kill_tree=None) -> int` so tests can inject a fake stage runner. A runner is `runner(stage: str, argv: list[str], cwd: Path, env: dict, on_line: Callable[[str], None], should_stop: Callable[[], bool]) -> int`.
- Progress file `<job-dir>/progress.jsonl`: one JSON object per line, appended and flushed. Every line has `time` (seconds since epoch), `stage` (one of `prepare`, `preprocess`, `pitch`, `features`, `train`, `index`, `export`), `stage_index` (1 to 7), `stages` (7), `percent` (0 to 100 overall, never decreasing within a run), `epoch` and `epochs` (training stage, else null), `eta_seconds` (null until two training epochs have completed), `message` (short, plain English). The last line of a run has `"done": true`, `"ok": bool`, `"error": str | null`, `"cancelled": bool`.
- Overall percent weights: prepare 1, preprocess 2, pitch 4, features 6, train 84, index 1, export 2.
- Dataset fingerprint: SHA-256 over the sorted list of `(clip file name, size, mtime_ns)` of the staged `*.orig.wav` files, stored in `trainer/rvc/logs/<experiment>/dataset.json` with the settings (`epochs` is not part of it; `batch_size` and sample rate are).
- Exit code 0 on success, 1 on failure, 2 when cancelled.

Behaviour (each item is a test):

1. **Prepare.** Creates a staging folder `trainer/rvc/logs/<experiment>/staging/` containing hard links (falling back to copies if linking fails) named exactly as the source `NNNN.orig.wav` files, and nothing else. A voice with no `*.orig.wav` files fails with a clear message. On a fresh (non-resume) run the whole experiment folder is removed first.
2. **Stage commands.** Each stage is launched with exactly the argument list, working directory and environment from sections 3 and 4 of the spike notes (assert the full argv and the env keys `PYTHONPATH`, `PYTHONSAFEPATH`, `RVC_CUDA_GRAPH`, `ROCM_SDK_TARGET_FAMILY`, `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`, `MIOPEN_LOG_LEVEL`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `PYTHONIOENCODING`, `PYTHONUNBUFFERED`, with caches under `trainer/.cache/`). Training uses `-l 1 -c 0 -sw 0 -v v2 -sr 40k -f0 1 -g 0`, `-bs`, `-te` and `-se` from the arguments.
3. **Output checks.** Preprocess with any `[Data slicing][Failed]` line, or producing no segments, fails the job with the count and first file name. Pitch or features reporting `Failed: n` with n > 0 fails. Training that exits 0 without `trainer/rvc/assets/weights/<experiment>.pth` existing and larger than 1 MB fails. Index that leaves no `added_*.index` fails. A non-zero exit code fails with the last 20 output lines included in the error.
4. **Progress and ETA.** Stage progress lines move the overall percent within the stage's band. Training percent is `epochs completed / total`. ETA is the mean duration of the last ten completed epochs times epochs remaining, plus nothing else; it is null before two epochs complete.
5. **Checkpoint backup.** After each `epoch_done` for an epoch that was saved (epoch divisible by save-every), the worker copies `G_2333333.pth` and `D_2333333.pth` to `G_2333333.pth.bak` and `D_2333333.pth.bak` (write to a temp name, then replace). It never copies while a save is in progress (between a `saved` line and the following `epoch_done`).
6. **Resume.** With `--resume`: the experiment folder is kept; if `dataset.json` is missing or its fingerprint or settings differ from the current dataset, the job fails with "The clips changed since this training run started. Start a new training run." Completed stages are skipped using their outputs (pitch and feature stages are always re-run because they skip finished segments themselves). After launching training, if no `resumed` line appears before the first `epoch_progress` or `epoch_done` line, the worker stops the trainer, restores the `.bak` pair over the primary pair, and tries once more; if that also does not resume, the job fails with "The saved training state could not be read" and nothing is deleted. A resume with no checkpoint at all fails the same way.
7. **Cancel.** The worker polls for a file `<job-dir>/cancel`. When it appears during a non-training stage the stage's process tree is killed at once. During training the worker waits for the next safe moment (an `epoch_done` line, when no save is in progress) for at most 60 seconds, then kills the tree regardless. The final progress line has `cancelled: true`, exit code 2. The cancel file is removed at start and at exit.
8. **Parent watchdog.** With `--parent-pid`, if that process disappears the worker kills the running stage's tree and exits as cancelled with the message "The app was closed". Use the standard library only (on Windows, `OpenProcess` through `ctypes`, or `tasklist`).
9. **Killing a tree** uses `taskkill /PID <pid> /T /F` (list arguments, no window), because the PID a launcher returns is a shim (spike notes, section 6).
10. **Export.** After the index stage the worker runs the existing exporter with the exporter's own Python: `tools/.rvc/venv/Scripts/python tools/export_rvc_onnx.py <small .pth> <experiment> --index <added index> --name <display name>`, with an environment variable `AIVOICE_EXPORT_DIR` naming the output folder. This requires one small change to `tools/export_rvc_onnx.py`: when `AIVOICE_EXPORT_DIR` is set, write `model.onnx`, `model.index` and `voice.json` there instead of `data/pretrained/rvc/<id>/` (add a test to `backend/tests/test_export_tool.py`). The worker exports into `<voice-dir>/models/rvc.new/`, checks that `model.onnx` is larger than 1 MB and `voice.json` parses with a `sample_rate`, then swaps it into `<voice-dir>/models/rvc/` (rename the old folder aside, rename the new one in, delete the old one). A failed export leaves any previous model untouched.
11. **Clean-up.** After a successful export the experiment folder under `trainer/rvc/logs/` and `trainer/rvc/assets/weights/<experiment>.pth` are deleted. After a failure or cancel they are kept.
12. **Logging.** Every line of every stage's output is appended to `<job-dir>/worker.log` with the stage name prefixed. The progress file is truncated at the start of a fresh run and appended to on resume. No transcript text is logged.
13. **Robustness.** An exception anywhere ends the run with a final progress line carrying the error message; the worker never exits without writing a final line (test by making the runner raise).

- [ ] **Step 1: Write the failing tests** in `backend/tests/test_worker.py`, loading `trainer/worker.py` by path. Use a fake runner that, per stage, creates the output files that stage would create (in a `tmp_path` fake of `trainer/rvc`), feeds scripted log lines to `on_line`, and returns an exit code. Use a fake clock for ETA. One test per behaviour item above (items 6 and 7 need several each: fingerprint mismatch, missing checkpoint, primary unreadable then backup works, backup also unreadable, cancel in a short stage, cancel during training at a safe moment, cancel during a save window, cancel timeout).
- [ ] **Step 2: Run to verify failure**, implement `trainer/worker.py` with a real default runner (`subprocess.Popen`, list arguments, `CREATE_NO_WINDOW`, line-buffered text output merged from stdout and stderr, UTF-8 with replacement), run to verify pass. The paths of `trainer/rvc`, `trainer/.cache`, `trainer/.venv` and `tools/` are derived from the worker's own location, overridable for tests.
- [ ] **Step 3: Commit** `feat: training worker with progress, resume, and safe cancel`.

---

### Task 4: Real end-to-end run

Proves the worker against the real pipeline. Uses the GPU for a few minutes.

**Files:**
- Create: `backend/tests/test_worker_real.py`
- Create: `docs/superpowers/notes/2026-10-13-stage-3-first-real-run.md`

- [ ] **Step 1: Write the opt-in test.** It is skipped unless the environment variable `AIVOICE_RUN_TRAIN_TEST=1` is set and `trainer/verify.py --fast` passes. It builds a temporary voice folder with about 60 seconds of synthetic speech-like clips named `0001.orig.wav` onward (generate them in the test with numpy at 48 kHz, in the manner of `spike/stage3/scripts/make_dataset.py`, written with `aivoice.audiofile.write_wav`), runs `trainer/worker.py` with the trainer Python as a subprocess for 2 epochs, batch size 4, save-every 1, and asserts: exit code 0; the last progress line is `done` and `ok`; percent never decreased; all seven stages appeared; `models/rvc/model.onnx`, `model.index` and `voice.json` exist; the experiment folder was cleaned up; `aivoice.engines.rvc.RvcEngine` loads the model on DirectML and converts one second of audio to finite output of the same length.
- [ ] **Step 2: A second opt-in test for cancel and resume**: start a 6-epoch run, create the cancel file after the first `epoch_done` line appears in the progress file, assert exit code 2 and that no trainer process remains (no `python.exe` whose command line contains the experiment name); then run again with `--resume` and assert it finishes ok and its log shows a `resumed` event.
- [ ] **Step 3: Run both** with `AIVOICE_RUN_TRAIN_TEST=1`. Fix whatever the real pipeline reveals in `trainer/worker.py`, `rvclog.py` or `filelist.py`, adding a unit test in `test_worker.py`, `test_rvclog.py` or `test_filelist.py` for each fix. Do not weaken unit tests to match; if the real behaviour contradicts a unit test, correct the test's premise and say so.
- [ ] **Step 4: Write the notes file**: commands run, wall time per stage, the progress file from the successful run, anything that differed from the spike notes, and every fix made.
- [ ] **Step 5: Run the full backend suite without the variable** (the real tests skip) and commit: `test: real end-to-end training run through the worker`.

---

### Task 5: Backend training manager

**Files:**
- Create: `backend/aivoice/training.py`
- Test: `backend/tests/test_training.py`

**Interfaces:**
- `TrainingError(ValueError)`.
- `TrainingManager(data: Path, library: Library, trainer_root: Path, launch=None, verify=None, pid_alive=None, now=time.time)`. `launch(argv: list[str], log_path: Path) -> handle` starts the worker detached from the backend's console and returns an object with `pid`, `poll() -> int | None`, and `kill_tree()`. `verify(fast: bool) -> dict` returns the `trainer/verify.py` report (the default runs it as a subprocess with the backend's Python and caches a passing full report for the life of the process).
- `start(voice_id: str, epochs: int = 200, batch_size: int = 8, resume: bool = False) -> dict`; `cancel() -> dict`; `status() -> dict`; `environment(refresh: bool = False) -> dict`; property `running: bool`; `forget(voice_id: str) -> None` (removes that voice's experiment folder and small model; used when a voice is deleted).
- `status()` keys: `state` (`idle`, `running`, `cancelling`, `done`, `failed`, `cancelled`, `interrupted`), `voice_id`, `stage`, `stage_index`, `stages`, `epoch`, `epochs`, `percent`, `eta_seconds`, `message`, `error`, `started`, `finished`, `can_resume`.
- Job record `data/voices/<id>/training/job.json` (atomic writes): `voice_id`, `experiment`, `epochs`, `batch_size`, `state`, `started`, `finished`, `worker_pid`, `error`.

Behaviour (each item is a test, using a fake `launch` whose handle is driven by the test and a progress file the test writes):

1. `start` validates in this order and raises `TrainingError` with these meanings: unknown voice; epochs outside 20 to 1000; batch size not 4, 8 or 16; a job already running ("Training is already running"); the voice has under 5.0 minutes ("This voice has X.X minutes of clean audio. Training needs at least 5."); the environment check fails (the message lists each failing check's message). Nothing is launched and no file is written when validation fails.
2. `start` launches the worker with the trainer's Python, `-P`, the worker path, and every argument of Task 3's command line, including `--parent-pid` with the backend's own pid and the voice's display name; writes `job.json`; state becomes `running`.
3. `status()` reads new lines appended to `progress.jsonl` since the last call (it keeps a file offset; a partial last line is left for the next call; malformed lines are skipped) and reflects the latest stage, epoch, percent, ETA and message.
4. When the handle reports exit: a final progress line with `ok` true gives state `done` with `finished` set; `cancelled` true gives `cancelled`; otherwise `failed` with the line's error; an exit with no final line gives `failed` with "The training process stopped unexpectedly. See training/worker.log."
5. `cancel()` creates the cancel file and sets state `cancelling`; if the worker has not exited 90 seconds later (checked on the next `status()` calls with the injected clock) the manager kills the tree and marks the job `cancelled`. `cancel()` with nothing running raises "Training is not running".
6. `resume=True` requires the voice's last job to be `failed`, `cancelled` or `interrupted` and a checkpoint to exist in the experiment folder; otherwise "There is no interrupted training run to resume for this voice". `can_resume` in status reflects the same rule. Resume reuses the job's `epochs` and `batch_size` unless given.
7. On construction the manager scans every voice's `job.json`: a job recorded as `running` or `cancelling` whose `worker_pid` is not alive becomes `interrupted` (the app was closed); one whose pid is alive and is a worker for that experiment is not adopted but is killed and marked `interrupted` (a previous backend died; never leave an unowned trainer on the GPU). Test both with the injected `pid_alive` and a recorded kill.
8. Only one job at a time across all voices. `status()` reports the most recent job of the voice it is asked about when given `voice_id`, else the running or most recently finished job.
9. `environment()` returns the verify report plus `ready: bool`; a failing report is never cached, so fixing the setup and asking again works without restarting.
10. `forget` refuses while that voice is training.

- [ ] **Step 1: Write the failing tests**, **Step 2: implement**, **Step 3: run the full backend suite**, **Step 4: commit** `feat: training manager with status, cancel, resume, and recovery`.

---

### Task 6: Trained models in the library and in live mode

**Files:**
- Modify: `backend/aivoice/library.py`, `backend/aivoice/pretrained.py`
- Test: `backend/tests/test_library.py` (add), `backend/tests/test_pretrained.py` (add)

**Interfaces:**
- `Voice.models: list[str]` (a property computed by the library when loading: `["rvc"]` when `models/rvc/model.onnx` and a parsable `models/rvc/voice.json` exist, else `[]`). The `data/voices.json` summary's `models` field carries it.
- `pretrained.list_voices(data)` returns the exported voices under `data/pretrained/rvc/` as before, followed by each library voice that has an RVC model, as `PretrainedVoice(id=<voice id>, name=<the library voice's display name>, model_path, index_path, sample_rate)`. A library voice with a damaged model folder is skipped, never raising.
- `Library.delete(voice_id)` is unchanged; the API layer calls `TrainingManager.forget` (Task 7).

- [ ] Tests first: a voice with and without a model; a model folder with a missing or corrupt `voice.json`; the summary file; ordering (pretrained first, then library voices oldest first); an id collision between a pretrained folder and a library voice is impossible by construction but the list must still not contain duplicates (pretrained wins); live mode's `LiveManager.start` accepts a library voice id (use the existing fakes).
- [ ] Implement, run the full backend suite, commit `feat: trained voice models appear in live mode`.

---

### Task 7: Training API and mutual exclusion

**Files:**
- Create: `backend/aivoice/api_training.py`
- Modify: `backend/aivoice/api.py`, `backend/aivoice/api_library.py`, `backend/aivoice/__main__.py`
- Test: `backend/tests/test_api_training.py`

**Interfaces:**
- Routes under `/training`, protected by the same request-origin rule as `/library` (reuse it): `GET /training/status?voice_id=`; `GET /training/environment?refresh=`; `POST /training/start` body `{"voice_id", "epochs"?, "batch_size"?, "resume"?}`; `POST /training/cancel`. All return the status object (environment returns its report).
- `create_app(..., training=None)`.
- Exclusion, all under the existing audio lock: training start is refused while live mode runs ("Stop live mode before training") or a recording is in progress ("Stop recording before training"); live start is refused while training runs ("Training is running. Wait for it to finish or cancel it before starting live mode"); recording start is refused while training runs; device refresh stays allowed during training. Deleting a voice is refused while it is training; deleting any other voice is allowed and calls `training.forget` for the deleted voice.
- `__main__.build_app` constructs `TrainingManager(data, library, <repo>/trainer)`.

- [ ] Tests first (with a `TrainingManager` built on the fakes from `test_training.py`): each route's happy path; every refusal above with its exact message and proof that nothing started; `/training/start` validation errors are 400 with the manager's message; a cross-site request is 403; an app built without `training` has no `/training` routes and behaves as before; `/voices` lists a trained library voice after a job completes (drive the fake to completion and create the model files).
- [ ] Implement, run the full backend suite, smoke test the real server (`/training/environment` and `/training/status` only; never start training or live mode on the real server in this task), commit `feat: training API with GPU exclusion`.

---

### Task 8: Training in the Voices tab

**Files:**
- Create: `app/src/trainingApi.ts`, `app/src/useTrainingStatus.ts`, `app/src/TrainPanel.tsx`
- Modify: `app/src/VoiceStudio.tsx`, `app/src/VoicesPanel.tsx`, `app/src/App.tsx`, `app/src/LivePanel.tsx`, `app/src/studioApi.ts` (the voice summary gains `models: string[]`; the backend summary route must return it, add that to `api_library.py` with a test), `app/src/App.css`
- Test: `app/src/TrainPanel.test.tsx`, `app/src/useTrainingStatus.test.tsx`, additions to `LivePanel.test.tsx` and `VoiceStudio.test.tsx`

**Interfaces:**
- `trainingApi`: `status(voiceId?)`, `environment(refresh?)`, `start({voice_id, epochs, batch_size, resume})`, `cancel()`; types `TrainingStatus`, `TrainingEnvironment`.
- `useTrainingStatus(voiceId, enabled)`: polls once a second while enabled and the state is `running` or `cancelling`, once on mount and after each action otherwise; never overlaps requests.
- `TrainPanel({ voice, active, onModelReady })`.

Behaviour (each a test):

1. Below 5 minutes it shows "Record or import at least 5 minutes to train. You have X.X." and no Start button.
2. With enough audio and a ready environment it shows epochs (a select: 100, 200, 300; default 200), an Advanced disclosure with batch size (4, 8, 16; default 8), a rough time estimate ("About N minutes" computed as epochs times minutes of audio times 0.95 seconds, rounded up to the next 5 minutes), and **Start training**.
3. When the environment is not ready it lists each failing check's message, shows "Training is not set up on this computer. See Training setup in the README.", offers **Check again** (calls `environment(true)`), and no Start button.
4. While running it shows the stage name in plain words (Preparing, Slicing audio, Finding pitch, Extracting features, Training, Building index, Exporting), "Epoch N of M" during training, a progress bar with the percent, the time remaining ("About 12 minutes left", or "Estimating time…" while null), and **Cancel**. It says that live mode and recording are unavailable until training finishes.
5. Cancel asks for confirmation ("Stop training? You can resume it later."), then shows "Stopping…" until the state changes.
6. `done`: "Training finished. This voice is now available on the Live tab." and calls `onModelReady` once; the Live tab's voice list reloads (App passes a `voicesVersion` down to `LivePanel`, which reloads voices when it changes and keeps the current selection if it still exists).
7. `failed`: shows the error in a `role="alert"`, where the log is (`data/voices/<id>/training/worker.log`), and **Resume** if `can_resume`, plus **Start over**.
8. `cancelled` or `interrupted`: shows what happened ("Training was stopped" or "Training was interrupted when the app closed"), **Resume** and **Start over**. Start over confirms that the progress so far will be discarded.
9. A voice that already has a model shows "Trained model: ready" and the Start button reads **Train again**; training again confirms that the current model is replaced only when the new training finishes.
10. Backend refusals (for example live mode running) appear in the alert with the backend's message.
11. Another voice's training in progress: this voice's panel shows "Another voice is training" with that job's percent and no Start button.

- [ ] Tests first, implement, `npm test`, `npx tsc --noEmit`, `npm run build`, commit `feat: train a voice model from the Voices tab`.

---

### Task 9: Documentation and the full check

**Files:**
- Modify: `README.md`, `docs/superpowers/notes/2026-10-13-stage-1-follow-ups.md`

- [ ] Run every suite (backend with `-W error`, frontend with `tsc` and build, `cargo test`), then the two real training tests once more with `AIVOICE_RUN_TRAIN_TEST=1`.
- [ ] README, matched to the code: a "Training setup" section (`bash trainer/setup.sh`, what it downloads and how big, the driver requirement stated as "AMD Adrenalin 26.2.2 or newer", ffmpeg on PATH, how to re-run verify); a "Training a voice" section (the 5 minute minimum, epochs and what they mean in practice, expected times from the spike table, what each stage is, Cancel and Resume, what happens when the app is closed, that live mode and recording are unavailable during training, where the model, log and job record live, how to use the model on the Live tab, retraining); that training runs natively on Windows and makes no network requests; disk use (environment size, 1.3 GB of temporary checkpoints during a run); a "Stage 3 first run checklist" for the human (run setup, train a real voice of at least 5 minutes for a small number of epochs first, cancel and resume once, close the app mid-training and reopen, then a full run and listen to it on the Live tab); known limitations (only synthetic audio has been trained so far; quality and long-run stability are unproven; GPT-SoVITS is not trained yet; two carried patches against a pinned upstream commit).
- [ ] Append a "Stage 3" section to the follow-ups note: decisions, deferred findings still open, what a human has not verified.
- [ ] Commit `docs: training setup and use, stage 3 checklist`.
