# Stage 1 follow-ups

Recorded at the end of the stage 1 build. Source: the build ledger.

## Decisions made during the build

- Ruling: work on branch stage-1-live-voice in the main checkout instead of a git worktree — repo is new and docs-only, and gitignored data/ models would not follow into a worktree — if wrong: branch can be moved, nothing lost.
- Ruling: no voice .pth was supplied, so Task 1 benchmark and Task 7 real-model test use a randomly initialised standard RVC v2 40k network exported as voice id "bench" via a throwaway spike script (inference speed is identical to a trained voice; output is noise) — avoids blocking and avoids downloading unapproved files — if wrong: benchmark still valid; user must export a real voice before the listening check.
- Ruling: Task 13 steps 2-3 (manual listening, preset tuning by ear) are left for the human partner; Task 13 does steps 1, 4, 5 only — if wrong: presets ship with untuned starting values.
- Ruling: accept the gate measured with a fresh ONNX session pair per preset (DirectML is fast only for the first input shape a session sees) — the engine is built per start with one fixed window, so this matches real use — if wrong: live mode would stutter and Task 13 listening shows it.
- Ruling: high_quality preset uses f0_method "dio" not "harvest" (harvest is slower than real time on this CPU); harvest stays selectable in Advanced — spec wants an accurate pitch extractor for high quality, so this is a known gap; a GPU pitch model (RMVPE) is the follow-up — if wrong: high-quality pitch tracking is no better than balanced.
- Ruling: Tasks 3, 4, 5 (presets, ringbuffer+router, devices) are dispatched to ONE implementer in sequence with one commit each and reviewed as one package — each is a small isolated module with complete code in the plan — if wrong: a defect in one delays all three, review loop still catches it.
- Ruling: Task 10 restricts CORS allow_origins to the Tauri webview origins (http://tauri.localhost, https://tauri.localhost, tauri://localhost, http://localhost:1420) instead of "*" — reviewer noted any web page in the browser could otherwise drive the local API — if wrong: the app window gets CORS errors and the origin list needs one more entry.
- Ruling: Task 6 finding accepted against plan text — add non-identity engine tests; the spec's fail-closed rule is the authority and the plan's tests under-protect it — if wrong: a few redundant tests.
- Ruling: while muted, the processor rolls ZEROS into its input window instead of mic audio, so nothing said while muted can be heard after unmute (reviewer showed up to crossfade ms of muted speech was emitted, converted and faded) — stricter than the plan — if wrong: first chunk after unmute has silence context and may sound slightly rough.
- Ruling: do not commit on the subagent's behalf and do not keep httpx2; revert to httpx with a narrow pytest warning filter, then let the implementer commit — a denied action is not routed around, and a new dependency is the owner's call — if wrong: one deprecation warning is filtered rather than fixed; migrating to httpx2 later is a one-line change. SURFACE TO USER.
- Ruling: closing the window hides it to the tray instead of exiting (tray Quit exits) — spec describes a tray app whose live mode keeps running; plan code did not implement it and review flagged that live mode would stop on window close — if wrong: user must use tray Quit to exit, which is one extra click.
- Ruling: Task 12 findings accepted against plan text — a failed mute must be visible to the user; the spec's fail-closed intent outranks the plan's code — if wrong: extra error banners.
- Ruling: final fix wave covers all 7 Important plus minors F8-F11 (backend log file, drop harvest dither, exporter hardening, README accuracy); other deferred minors stay deferred per the final reviewer's "can wait" triage — if wrong: small polish items remain for the owner.
- Ruling: mic_device becomes optional so listen-only works without VoiceMeeter (spec: "Monitor-only use still works") and the UI never auto-selects a real output as the virtual mic — if wrong: one extra select option.

## Deferred minor findings

- Task 1: minor (deferred): export_rvc_onnx.py guard runs at import, raw FileNotFoundError if git missing, --help fails on unpatched clone
- Task 1: minor (deferred): export_rvc_onnx.py does not assert torch >= 2.6 (safety basis for RVC internal torch.load)
- Task 1: minor (deferred): notes file sentence about exporter verification slightly stale
- Task 2: minor (deferred): StarletteDeprecationWarning from TestClient in test output (plan-mandated test + unpinned httpx)
- Task 2: minor (deferred): load_settings does not validate dml_device_id type
- Tasks 3-5: minor (deferred): RingBuffer capacity=0 / negative n unguarded
- Tasks 3-5: minor (deferred): input callback ignores status; test helper imports Device inside function; no multithreaded RingBuffer smoke test
- Task 6: minor (deferred): mute/failure is a hard cut (click) rather than a fade-out
- Tasks 3-5: minor (deferred): input callback on_audio has no try/except (exception would propagate into audio thread)
- Task 7: minor (deferred): f0.py harvest path adds fixed -60 dBFS dither only to satisfy a synthetic sine test; replace with a realistic test signal and drop the dither
- Task 7: minor (deferred): make_session does not verify DmlExecutionProvider is active; silent CPU fallback would be slow with no error
- Task 7: minor (deferred): blend_index with index smaller than k=8; real-model test has no timing log; faiss index path only unit-tested with fakes
- Task 8: minor (deferred): join-timeout branch (worker stuck > 2 s in process) untested; weak silence assertions; in_level freezes after failure; feed-failure-through-worker path untested
- Task 9: minor (deferred): warm-up test does not assert the window sample count; stats() writes _last_error without the lock; set_mode/update read session without the lock
- Task 10: minor (deferred): no Host header validation (DNS rebinding) and no Origin check on mutating HTTP routes (cross-site simple POST can hit /live/stop); add TrustedHostMiddleware for 127.0.0.1/localhost and reject disallowed Origin on non-GET routes
- Task 11: minor (deferred): kill_tree runs while holding the child mutex; if only the launcher dies the interpreter survives (its parent pid is the app) and holds the port, so respawns fail until it exits
- Task 11: minor (deferred): tray menu, hotkey, hide-on-close, tray Quit, graceful Exit path never exercised by a human
- Task 12: minor (deferred): stale actionError until next hotkey success; pitch slider keeps unaccepted value after failure; useStats accepts valid-JSON non-Stats frames
- Task 13: minor (deferred): README understates harvest slowness (too slow at every preset); bench-voice wording wrong for a fresh clone; commands are Git Bash flavoured but not labelled; "restart to see new voice" advice

Items above about the harvest dither, the DirectML CPU-fallback check, settings type validation, Host/Origin enforcement, the exporter guard and torch check, and README wording were fixed in the final fix wave.

## Still open after the final review

- A data/settings.json that is not valid JSON is silently ignored (falls back to GPU 0) and the setup check reports settings as valid.
- The Listen on select still lists the device chosen as Virtual mic; choosing the same device is rejected by the backend rather than prevented in the UI.
- Not yet verified by a human: listening quality with a real voice, preset tuning, tray menu, Ctrl+Alt+V hotkey, hide to tray, tray Quit, device unplug detection, backend log redirection, device refresh inside a running backend.
- No GPU pitch detector yet (RMVPE); all presets use the fast CPU detector.
- Starlette recommends migrating the test client from httpx to httpx2; deferred pending owner approval of the new dependency.

## Stage 2 (recording and import)

Recorded at the end of the stage 2 build. Source: the build ledger. Every ruling states what it costs if wrong.

### Decisions made during the build

- Ruling: proceed without human plan review — the user instructed hands-free execution through all stages — if wrong: the plan can be amended and work redone on the branch before merge.
- Ruling: transcripts default to English (small.en) — the user writes in English and the bundled prompts are English — if wrong: set whisper_model to a multilingual model; prompts would need a second language file.
- Ruling: tasks are batched as 2+3, 4+5, 6+7 (small pure modules with complete code), each batch one implementer and one review — if wrong: a defect in one delays its pair.
- Ruling: new Python packages (faster-whisper, ctranslate2, huggingface-hub, tokenizers, av, tqdm) and the ~480 MB Whisper model download are covered by the approved spec ("Whisper runs locally") and the hands-free instruction — if wrong: packages can be uninstalled and data/pretrained/whisper deleted.
- Ruling: the recording sentence "It rained all week, yet the garden has never looked greener." stays unchanged — past simple followed by present perfect is correct, natural English, and the original finding was a Minor — if wrong: one prompt line reads slightly oddly.
- Ruling: Task 8 fix (backend only) and Task 10 (app/src only) run at the same time with separate implementers, each staging only its own directory — the two touch disjoint files and waiting costs wall-clock for no safety gain — if wrong: a git index.lock collision, which each is told to retry.
- Ruling: clip audio is fetched by the frontend with fetch() and played from a blob URL instead of pointing <audio src> at the backend — fetch always sends Origin, so the new /library origin check works regardless of what Referer the webview sends for media requests — carried into Task 11 — if wrong: slightly more memory per played clip.
- Ruling: the import status gains a `voice_id` field in the backend so the UI can tell whose job it is — simplest correct attribution; the router already tracks it — if wrong: one extra field in a status payload.

### Fixed during the build

These were deferred or found open earlier in the build and are fixed in the code. Each names the commit.

- Same-site requests to `/library`: a page on another localhost port could play clips by sending neither Origin nor Referer. Any request that carries Sec-Fetch-Site (other than "none") must now name the app in Origin or Referer. Fixed in bdbe5c6.
- A voice delete could race the save inside `record_stop`. `record_stop` now takes the voices lock after the audio lock. Fixed in bdbe5c6.
- The import status had no `voice_id`, so the import screen could adopt another voice's running import, and a second import in one mount could report the previous job's result (the two Important findings of the Task 11 review). The status now carries `voice_id` and a job number (bdbe5c6) and the screen follows only its own voice's job (97ed907).
- A failed Cancel (or the cancel sent when the tab goes inactive) left the screen idle while the backend might still be recording. The recording controls now stay until a cancel is known to have succeeded, and the cancel is retried when the tab is active again. Fixed in 97ed907.
- No test for a take over 20 seconds, and no test driving the locks with racing threads: both exist since 98807fa (`test_a_take_over_20_seconds_is_rejected_as_too_long`, `test_device_changes_wait_for_a_closing_microphone_stream`) and bdbe5c6 (`test_a_voice_delete_waits_for_a_take_that_is_being_saved`).

Fixed in the final fix wave (the commits after 97ed907 on this branch; see `git log`):

- Deleting a clip was one click and erased its files. It now asks first, and the files are moved to `data/trash/<voice-id>/clips/`; a redo moves the replaced take there too.
- A disk failure while saving showed "Request failed (500)". Creating a voice, saving a clip, saving a transcript and deleting a clip now report a readable reason as a 400.
- Creating a voice looped forever (holding the library lock) when `data/voices` could not be created because a file was in its way. Found while writing the disk-failure tests; it is now a readable error.
- Whisper contacted Hugging Face on the first import of every backend process. A model that is on disk is now loaded with `local_files_only=True`, and hub telemetry is turned off.
- The import screen polled the status five times a second for as long as the voice stayed selected. It now stops when nothing is running.
- A damaged voice folder was skipped silently; it is now named in the log with the reason.
- The WAV rename was not retried on Windows file-in-use errors; it shares the retry helper with the JSON writes now.
- Reading a long file made a second full copy of the audio in memory.
- No warning before a take got too long; the microphone list on the Voices tab did not follow Refresh devices; an emptied transcript was saved; a quiet import gave no hint; refused `/library` requests left no trace in the log.

### Deferred minor findings

Copied from the ledger, with the items fixed since removed (see above). These are still open.

- Task 1: minor (deferred): huggingface-hub has no upper bound (resolved 2.2.0); notes do not list transitive packages; Whisper returns "You" for near-silence (do not rely on empty transcripts)
- Tasks 2+3: minor (deferred): duration cap not checked on resampler flush; peak memory for hour-long files (one avoidable copy was removed in the final fix wave; the decoded blocks and the joined array still exist side by side while they are joined); split_on_silence with tiny max_seconds or large pad_ms
- Tasks 4+5: minor (deferred): `created` uses local time without an offset, so listing order can be wrong in the repeated daylight-saving hour
- Task 9: minor (deferred): VoicesPanel load has no stale-response guard; error banner persists until next successful load; tabs lack aria-controls; confirm spy not restored in finally
- Task 8: minor (deferred): real webview headers unverified by a human (a refused request is now logged with its headers, which makes this checkable on the first run); stop holds the audio lock through the disk save
- Task 10: minor (deferred): tab going inactive during an in-flight Stop fires a concurrent cancel; in-flight-start test does not assert recordStop was not called

### Open at the end of the build

- A take that cannot be written to disk is lost. `Studio.stop_recording` holds the take only in memory; when the save fails the user now gets a readable reason and the earlier take of that sentence is kept, but the new take cannot be retried or saved elsewhere. Keeping the audio for a retry is a design change, not done here.
- A deleted clip can only be restored by hand (drop its `.orig.wav` from `data/trash/<voice-id>/clips/` onto the window to import it again). There is no undo in the app, and the app never empties the trash, so it grows until the user clears it.
- A clip file that cannot be moved to the trash (in use, or the trash folder unusable) is left in the voice's `clips` folder as a stray: logged, never listed or served, and not cleaned up later.
- The backend still accepts an empty transcript over the API; only the Clips list refuses to save one. An import can still produce a clip whose transcript is empty if Whisper returns nothing.
- The quiet-import hint is a fixed sentence in the import screen; `backend/aivoice/importer.py` only counts skipped pieces, as before, and was not changed.
- The local-only Whisper load falls back to the downloading load on any error, so a damaged copy of the model on disk leads to a network request (to repair it) rather than an error message. If the machine is offline at that point the import fails with the library's network error.
- The log line for a skipped voice folder is written once per folder and reason per backend run, not on every listing.
- The stage 1 items under "Still open after the final review" still apply.

### Not verified by a human

Everything below was exercised only with fake devices, generated audio and a fake transcriber. The suites passing says nothing about these.

- The whole "Stage 2 first run checklist" in `README.md` (13 steps).
- Real microphone recording: device selection, the level meter, the four rejection messages with a real voice and a real room, and that leaving the tab stops the microphone.
- Drag and drop in the real Tauri window: that dropped file paths reach the Voices tab, and that nothing happens on the Live tab.
- Clip playback in the real webview: clips are fetched with `fetch()` and played from a blob URL, and the `/library` origin check was written against the headers the webview is expected to send, not ones observed. If the real webview sends different Origin or Referer values, playback and every Voices call would fail with a 403.
- Real Whisper transcription quality: how accurate `small.en` is on real recordings and on music or noisy files, the first-use download of about 500 MB, and import speed on this CPU. Whisper returns text such as "You" for near-silence, so quiet pieces are filtered by level, not by empty text.
- Recording and live mode refusing each other on real hardware, and Refresh devices being refused during a take.
- Added by the final fix wave, also tested only with fakes: the delete confirmation in the real webview (`window.confirm` inside Tauri); clip files landing in `data/trash/<voice-id>/clips/` on a real disk; the real Whisper package loading an already downloaded model with `local_files_only=True` and making no network request (the tests use a stand-in for the package, never the real one); the "Refused" lines reaching `data/backend.log`; the microphone list following Refresh devices with real devices.

## Stage 2: complete list of decisions made during the build

Copied from the build ledger when the stage was merged. Each line ends with what it costs if wrong.

- Ruling: proceed without human plan review — the user instructed hands-free execution through all stages — if wrong: the plan can be amended and work redone on the branch before merge.
- Ruling: transcripts default to English (small.en) — the user writes in English and the bundled prompts are English — if wrong: set whisper_model to a multilingual model; prompts would need a second language file.
- Ruling: tasks are batched as 2+3, 4+5, 6+7 (small pure modules with complete code), each batch one implementer and one review — if wrong: a defect in one delays its pair.
- Ruling: new Python packages (faster-whisper, ctranslate2, huggingface-hub, tokenizers, av, tqdm) and the ~480 MB Whisper model download are covered by the approved spec ("Whisper runs locally") and the hands-free instruction — if wrong: packages can be uninstalled and data/pretrained/whisper deleted.
- Ruling: that sentence stays — past simple followed by present perfect is correct, natural English, and the original finding was a Minor — if wrong: one prompt line reads slightly oddly.
- Ruling: Task 8 fix (backend only) and Task 10 (app/src only) run at the same time with separate implementers, each staging only its own directory — the two touch disjoint files and waiting costs wall-clock for no safety gain — if wrong: a git index.lock collision, which each is told to retry.
- Ruling: clip audio is fetched by the frontend with fetch() and played from a blob URL instead of pointing <audio src> at the backend — fetch always sends Origin, so the new /library origin check works regardless of what Referer the webview sends for media requests — carried into Task 11 — if wrong: slightly more memory per played clip.
- Ruling: the import status gains a `voice_id` field in the backend so the UI can tell whose job it is — simplest correct attribution; the router already tracks it — if wrong: one extra field in a status payload.
- Ruling: Task 12's README is reviewed as part of the final whole-branch review instead of a separate task review — it is docs-only and will be revised again after the fix rounds — if wrong: a README inaccuracy survives one extra review hop.
- Ruling: Task 11 fix round 1 also carries three earlier FIX AT FINAL items (failed-cancel UI state, voices lock in record_stop, same-site /library tightening) so the final review sees them fixed — if wrong: that fix round is larger and its re-review must cover more.
- Ruling: deleted clips and takes replaced by a redo are moved to data/trash/<voice-id>/clips/ rather than erased, with a confirmation in the UI — recordings are irreplaceable and the README promises nothing is erased — if wrong: the trash folder grows until the user empties it.
- Ruling: quiet imported audio is not auto-normalised; the result message tells the user how to fix it — changing a user's audio silently is riskier than a hint — if wrong: quiet recordings need an extra manual step.

## Stage 2: left open after the final re-review

- If every one of 999 candidate trash names for a clip is taken in the same second, the clip's files stay in the voice folder with a warning (nothing is overwritten).
- Deleting a voice when the data/trash folder cannot be created returns an unreadable error.
- A trashed recorded take restored by re-importing its .orig.wav comes back as an imported clip and no longer counts as that sentence being recorded.
- A take that cannot be saved (for example a full disk) is reported clearly but its audio, held only in memory, is lost. Remedy if wanted: keep the last unsaved take in memory and offer Try saving again.
- Nothing in stage 2 has been exercised by a human. See the stage 2 first-run checklist in the README; the first item (the Voices tab loads at all) checks the request-origin rule against the real webview.

## Stage 3 planning note

- The WSL2 Ubuntu distro on this PC is broken (its ext4.vhdx is missing). AMD now ships official PyTorch on Windows wheels with ROCm that list the RX 7900 XTX, so stage 3 evaluates native Windows training first.

## Stage 3: complete list of decisions made during the build

Copied from the build ledger (`.superpowers/sdd/2026-10-13-stage-3-training/progress.md`). Each line ends with what it costs if wrong.

- Ruling: train natively on Windows with AMD's ROCm PyTorch instead of WSL2 — the spike proved it works and the WSL2 distro on this PC is broken — if wrong: the worker's stage commands would need a WSL wrapper; the backend and UI are unaffected.
- Ruling: stage 3 trains RVC only; GPT-SoVITS fine-tuning moves to stage 4 — its engine is unproven on this GPU and belongs with TTS — if wrong: stage 4 is larger.
- Ruling: carry two patches against RVC commit 81eed5e (dilated-convolution speed fix, preprocess tail fix) — 3.4x training speed and no silently dropped audio — if wrong: revert the patches; training still works but slower.
- Ruling: the plan gives interfaces, behaviour and tests for the worker instead of complete listings — its code must match a real external pipeline the spike already scripted — if wrong: implementers need more review rounds.
- Ruling: no real recordings exist, so the end-to-end proof uses synthetic speech-like audio — the pipeline is proven, voice quality is not — if wrong: the first real training run by the owner reveals problems the synthetic run could not.
- Ruling: training is tied to the app's lifetime (the worker exits when the backend exits; the job becomes interrupted and resumable) — an unseen process holding the GPU after Quit would be worse — if wrong: users must keep the app open during long training.
- Ruling: accept the worker's deviation that a resume with no checkpoint ever saved trains from epoch 1 — nothing that existed is lost, and a saved-then-missing checkpoint still fails — provided the checkpoint mark is set at the first saved epoch, not the first backup — if wrong: an early-cancelled run resumes from scratch without saying so.
- Ruling: Resume is offered for any stopped run whose prepared stages exist, not only after the first checkpoint — the worker already supports it and it saves the user minutes of preparation — if wrong: a resume before any checkpoint retrains from epoch 1, which the UI now says.
- Ruling: a replaced trained model goes to data/trash instead of being erased — keeps the app's "nothing is erased" promise uniform — if wrong: about 110 MB per retrain accumulates in the trash.

## Stage 3: minor findings still open

Updated 2026-10-13 after the final whole-branch review and its fix wave. The items that were marked "FIX AT FINAL" are fixed and are listed under "closed by the final fix wave" below; what follows first is what is still open. None loses user data.

- Environment setup: a half-written clone gives a raw git error. `verify.py` checks pretrained file sizes only (`setup.sh` checks SHA-256). Behaviour when `trainer/.venv` exists with a different Python is not defined. `trainer/requirements.lock` pins what `requirements.txt` pulls in; the three packages only PyTorch itself needs (jinja2, mpmath, sympy) are not pinned, and the lock has to be renewed by hand when `requirements.txt` changes.
- Library: `Library.list()` does not catch `RecursionError` from a deeply nested `voice.json` (pre-existing, the library's own file).
- Manager and routes: `TrainingManager.start` still brings the job up to date itself (which can mean a kill) while the start route holds the audio and voices locks. The route does the same thing just before it takes the locks, so inside them it is normally a no-op; only a kill that becomes due in between would run under the locks. A voice whose worker has just ended is still "training" for the delete route, live start and recording start until the next status poll (at most a second later while the app is open) has settled it.
- Manager: shutdown during a start's environment check can let a worker launch afterwards (the worker's parent watchdog ends it). Start reports a not-ready environment before an unknown voice or the recording refusal, and an idle refresh racing a start can put the full check inside the locks. For a worker whose process tree was never captured, "gone" means only the launcher shim exited, and `_pid_alive` treats "could not look" as gone.
- Resume: what the screen offers and what the backend accepts are now one rule. A run refused because the clips changed stays resumable (the worker reports it as such; the user can put the clips back, and the worker checks them again on every resume, deleting nothing). Only a failure the worker reports as one a resume can never get past (the saved state cannot be read, the prepared audio is incomplete) takes Resume away, in the screen and in the backend alike. One odd case remains: a crash while an old experiment is being removed leaves a run that says "The clips changed" on every resume although the clips did not change; Start over is the way out.
- Worker: the longer wait for the end of an epoch after Cancel (up to 15 minutes with long epochs) applies only to a cancel from the app. A worker whose app has gone still waits one minute at most, so that nothing unowned stays on the GPU; with epochs longer than a minute it can therefore be ended in the middle of a save, and the resume then falls back to the checkpoint backup. When the trash cannot be used, a replaced model stays beside the new one as `models/rvc.old` and is moved (or, failing that again, deleted) by the next successful run. A `.bak` checkpoint pair left mixed by a crash between its two renames is used as it is if the primary pair is later damaged. A crash during the clean-up after a successful run makes a later resume say "The clips changed".
- Trash: models replaced by retraining accumulate under `data/trash/<id>/` (about 150 MB each for 10 minutes of audio); nothing removes them.
- Display language: progress, time remaining and checkpoint backups depend on English log lines from RVC (see the README); on a display language RVC translates without an English fallback a run completes without them.
- Frontend: both the Train box and the app-wide follower report a finished run, so the Live voice list can be reloaded twice in a row (harmless). Two frontend tests do not guard what their names say (overlap test, mount-again test), and the stopping flag is not independently covered. Several Train box tests asserted a button in the same tick as the status text although the button arrives one request later; four were made to wait for the button after they were seen failing under load, and others of the same shape may exist.

### Closed by the final fix wave

- The voice-delete guard no longer depends on `training.status()` (which swallows errors); the routes that hold the audio or voices lock only call reads that never end a process (`is_active`, `active_voice`, `require_not_training`), and `forget` no longer services a kill.
- No environment probe runs beside live mode or a training job: the last report is returned, or one marked unknown.
- The app follows a running job on every tab and reloads the Live voice list when it finishes and whenever the Live tab is shown; a failed first status poll shows a message, retries every 3 seconds and has a Retry button; the environment report is remembered across voice switches.
- Resume is offered from the moment the clips are sliced, not only after the first checkpoint.
- A model parked as `models/rvc.old` by a kill between the two swap renames is put back when the backend starts.
- A replaced model is moved to the trash instead of being erased.
- A voice name that looks like a command-line option no longer fails the export at the end of a run.
- The backend's own shutdown asks the worker to stop and waits up to 5 seconds before killing it.
- The wait for the end of an epoch after Cancel grows with the length of the epochs (up to 15 minutes).
- `mute.zip` is extracted to a temporary folder and renamed, and `verify.py` checks all four silence files a filelist names.
- `is_fatal_noise`, which had no caller, was removed from the log parser.
- The worker reports the epoch it really resumed from after falling back to a backup, words the old-format stage record refusal properly, and logs a warning after three skipped checkpoint backups in a row.
- The re-review of the worker's last fix round (commit 2e86d54, with the real-run commit d5e0e15) was completed and came back clean.

## Stage 3: not verified by a human

Everything below was exercised only with synthetic audio (about 60 seconds of speech-like test sound, 2 to 8 epochs) and fakes. The suites passing says nothing about these.

- The whole "Stage 3 first run checklist" in `README.md` (10 steps), and above all: **a real voice trained and then heard on the Live tab.** Voice quality, how many epochs are enough, and whether the index blend helps are all unknown.
- A multi-hour run (200 to 300 epochs on 30 minutes of audio, about 1 h 35 min predicted from the 10 minute measurement). Stability, memory growth and heat over that time are unproven; the time estimates on screen and in the README are extrapolations.
- Quitting the real app (tray Quit, Ctrl+C in `tauri dev`) mid-training leaves no process on the GPU, and Resume after it. The tests kill the worker directly and use fakes for the app side.
- The Train box in the real window: stage names, bar, Cancel and Start over confirmations (`window.confirm` inside Tauri), the messages for each refusal, and the Live voice list picking up a finished model.
- `bash trainer/setup.sh` on a machine that is not this one (clean caches, other drivers, an older Adrenalin than 26.8.1; 26.2.2 is AMD's stated minimum and has not been tried), and the real download sizes there.
- Other GPUs, and training with `ffmpeg` and `ffprobe` installed differently from this PC's.
- The real GPU tests last passed 4 in 17 min 2 s on this machine, while another experiment shared the GPU (the first three took 5 min 32 s on a free GPU); they are skipped in the normal suite. The fourth drives the real training manager, but still without the app's window or the Tauri shell.

## Stage 3: complete list of decisions made during the build

Copied from the build ledger when the stage was merged. Each line ends with what it costs if wrong.

- Ruling: train natively on Windows with AMD's ROCm PyTorch instead of WSL2 — the spike proved it works and the WSL2 distro on this PC is broken — if wrong: the worker's stage commands would need a WSL wrapper; the backend and UI are unaffected.
- Ruling: stage 3 trains RVC only; GPT-SoVITS fine-tuning moves to stage 4 — its engine is unproven on this GPU and belongs with TTS — if wrong: stage 4 is larger.
- Ruling: carry two patches against RVC commit 81eed5e (dilated-convolution speed fix, preprocess tail fix) — 3.4x training speed and no silently dropped audio — if wrong: revert the patches; training still works but slower.
- Ruling: the plan gives interfaces, behaviour and tests for the worker instead of complete listings — its code must match a real external pipeline the spike already scripted — if wrong: implementers need more review rounds.
- Ruling: no real recordings exist, so the end-to-end proof uses synthetic speech-like audio — the pipeline is proven, voice quality is not — if wrong: the first real training run by the owner reveals problems the synthetic run could not.
- Ruling: training is tied to the app's lifetime (the worker exits when the backend exits; the job becomes interrupted and resumable) — an unseen process holding the GPU after Quit would be worse — if wrong: users must keep the app open during long training.
- Ruling: accept the worker's deviation that a resume with no checkpoint ever saved trains from epoch 1 — nothing that existed is lost, and a saved-then-missing checkpoint still fails — provided the checkpoint mark is set at the first saved epoch, not the first backup — if wrong: an early-cancelled run resumes from scratch without saying so.
- Ruling: Resume is offered for any stopped run whose prepared stages exist, not only after the first checkpoint — the worker already supports it and it saves the user minutes of preparation — if wrong: a resume before any checkpoint retrains from epoch 1, which the UI now says.
- Ruling: a replaced trained model goes to data/trash instead of being erased — keeps the app's "nothing is erased" promise uniform — if wrong: about 110 MB per retrain accumulates in the trash.

## Stage 3: left open after the final re-review

- If the backend disappears after Cancel was pressed, the worker can hold the GPU for its long cancel wait (up to 15 minutes) instead of 60 seconds; the next backend start still stops it.
- A training panel that checked the environment while live mode was running keeps saying it can be checked once live mode stops, until Check again is pressed or another voice is selected.
- The training start route still runs the fast environment check, and one servicing call, while holding the audio and voices locks.
- Nothing in stage 3 has been exercised by a human, and only synthetic speech-like audio has been trained. See the stage 3 first-run checklist in the README.

## Stage 4 (text to speech)

Written 2026-10-13 at the end of the build, before the final whole-branch review, and brought up to date after the final fix wave that followed it (the sections "Rulings of the final review", "Closed by the final fix wave" and "What the final fix wave changed for the user" are from that wave). Sources: the build ledger (`.superpowers/sdd/2026-10-13-stage-4-text-to-speech/progress.md`) and the two real runs (`docs/superpowers/notes/2026-10-13-stage-4-first-real-run.md`).

### Decisions made during the build

Each line ends with what it costs if wrong.

- One speech model, F5-TTS, serves all three presets, and "Best" is F5-TTS followed by the voice's own trained RVC model. GPT-SoVITS, which the design named for Best, was not built and nothing fine-tunes a text-to-speech model on a voice. Why: the spike found several blockers for GPT-SoVITS on this GPU's PyTorch build and weeks of uncertain work, while F5-TTS plus RVC needed no new training code. If wrong: Best is less faithful to the voice than a fine-tuned model would be; one can be added later without disturbing this design.
- Balanced uses 16 sampling steps, not 32. Why: measured as accurate as 32 and twice as fast. If wrong: one constant.
- Fast uses 10 sampling steps, not the 8 it was planned and first built with. Why: measured with random seeds and a speech recogniser, 25 percent of sentences had a wrong word at 8 steps and 6.7 percent lost three words or more; at 10 steps 2.8 percent and none (Balanced: 1.7 percent and none), for 0.1 second more per sentence. The rule applied: the smallest step count with at most about twice Balanced's rate of sentences with a wrong word and at most 2 percent losing three or more. If wrong: one constant in `tts.py` (the worker accepts 8, 10, 16 and 32); the margin is small (10 of 360 against a bar of about 12) and rests on one reference clip.
- Text to speech cannot run alongside live mode, recording or training: one activity at a time. Why: sharing the GPU would make live audio stutter, and one rule for everything is simple. If wrong: a user cannot have typed text spoken while live mode is on without stopping it first.
- The F5-TTS weights' CC-BY-NC-4.0 licence (non-commercial) is accepted for this personal app and stated in the README and in the setup script's output. If wrong: the owner must not use the output commercially, or must swap the model.
- The first five tasks were built in parallel by separate implementers on separate files. Why: time. If wrong: a git lock collision, retried.
- The plan gave interfaces, behaviour and tests rather than complete code, as in stage 3. If wrong: more review rounds.
- The process-ending and environment-check logic that `tts.py` has in common with `training.py` was first planned to move into a shared module in the final fix wave. It was not moved: see "Rulings of the final review".
- A speak request refused because something else is active answers with HTTP 409, while the matching refusals on the other routes answer 400. Why: the screen shows the message for any failure, and changing the stage 3 routes would break their tests. If wrong: a one-line change.
- Deleting any voice is refused while something is being spoken, not only the voice being spoken. Why: the simplest safe rule; requests last seconds. If wrong: add a way to ask which voice is active.
- (From the real run.) The start of each piece is cut by rules measured on one reference clip: sound in the first 60 ms that ends within 0.4 s and is followed by 0.4 s of quiet is taken to be what is left of the reference clip, and removed with the pause after it. Why: without it, 74 percent of Fast pieces and a few Balanced ones kept a blip and a second of room tone before the sentence. If wrong: with another voice's clip a short first word followed by a long pause could be removed, or a blip could remain; the thresholds are parameters of `tts_audio.drop_reference_tail`.

### Rulings of the final review

Recorded with what each costs if wrong.

- **The shared process and environment module was NOT extracted.** The verified-kill and environment-check logic of `training.py` stays duplicated in `tts.py` (about 110 lines: the list of a worker's processes, the kill that is checked, the retry, the environment report; `tts.py` also imports training's private helpers `_Worker`, `_kill_tree`, `_end_survivors` and others). Why: extracting it means editing the safety-critical code of stage 3 at the very end of the stage, and `tts.py` works as it is. Cost: a fix to one copy must be made in the other by hand.
- **Refusal status codes stay mixed**: a speak request refused because something else is active answers 409, the matching refusals on the other routes answer 400. The screen shows the message for any failure. Cost if wrong: a one-line change.
- **Deleting a voice is refused for any voice while something is being spoken**, not only for the voice being spoken. Cost if wrong: add a way to ask which voice is active.
- **Fast is 10 sampling steps**, by measurement (see the decisions above). The plan's "8" is superseded.
- **Model files are verified by size only when the app runs**; SHA-256 is checked once, when `tts/setup.sh` downloads them. The plan said "verified by SHA-256 before use". Why: the files are loaded as safetensors and with `torch.load(weights_only=True)`, which do not run code from the file, and hashing 1.4 GB on every start buys little. Cost if wrong: a file changed after download to another file of exactly the same size is not noticed; add a hash at backend start.
- **The swappable `engines.tts` interface named in the design was not built.** The manager speaks the worker's protocol directly, and there is one engine. Cost if wrong: a second text-to-speech engine means a second manager or an extraction at that point.
- **Number rewriting covers only the measured failures**: whole numbers with thousands separators and phone numbers. Plain integers, decimals, years, times and dates are left alone because the model read them correctly. Cost if wrong: extend `tts_text.write_out_numbers`.
- **Text to speech needs the AMD ROCm PyTorch environment and has no DirectML or CPU fallback**, so the design's line "zero-shot TTS still works without ROCm" does not hold. Stated in the README.
- **The reference clip is chosen automatically, with no manual override yet** (the design planned one). Stated in the README.

### Closed by the final fix wave

Each with the commit's test as its evidence; details in `.superpowers/sdd/2026-10-13-stage-4-text-to-speech/final-fix-backend-report.md`.

- **The idle unload now happens with nobody asking.** It ran only inside the status poll, and the screen polls only while a request is active, so in the real app the engine was never unloaded. The manager now has its own housekeeping thread (started with the first worker, ended at shutdown) that looks every 30 seconds: idle timeout, a worker that died while idle, and kills due another try. Shown in the real run too: an idle worker was gone 12 seconds after the idle time was shortened, with no status call.
- **A reference clip that is too quiet no longer blocks a voice**, and refusals are said in terms the user can act on: the clip's level is looked at when the clip is chosen (the worker's own floor, RMS 0.003), a louder clip is preferred, and a voice with none is listed as not usable with "Clip N is too quiet to copy. Record a louder clip of 5 to 10 seconds." Refusals from the worker name the clip number of the Voices tab instead of a file, and "choose another clip" (which the app cannot do) is gone.
- The quick environment check is capped at 20 seconds (it ran with 300 while route locks were held), and a check that runs out of time has its whole process tree ended, in `tts.py` and for the probes inside `tts/verify.py`.
- Speech on the listen device uses the Live tab's listen volume, as the backend last learned it.
- Every piece is brought to one peak level (0.89) before it is played or saved; neighbours could differ by about 3 dB, and by far more with a quiet reference clip.
- Only one full environment check runs at a time; a second caller waits for the first and takes its result.
- **Numbers the model misreads are written out** (`tts_text.write_out_numbers`, applied when the text is split): whole numbers with thousands separators up to 999,999,999,999 with optional decimals, and phone numbers digit by digit. Measured afterwards with the real model at Balanced and the backend's speech recogniser: "1,200", "2,500,000", "555 0142" and "0412 345 678" were each heard correctly in 5 of 5 tries (before: "1,200" never, the phone number about half the time). What remains unhandled: negative numbers and currency signs (the sign stays where it is), ordinals such as "1,200th", a trillion and more, thousands grouped with spaces or dots, phone numbers written as one unbroken run without a leading 0 or plus sign, French-style dotted phone numbers, and every digit string that could be a date, a range, a decimal or a list, which is left alone on purpose.
- Setup script: `-r requirements.lock` and `-r requirements.txt` are relative paths like the constraint file, pinned by a test; `curl` retries three times.
- Playback: the "stopped responding" timer starts when the streams have been started; `stop()` returns only when the streams are closed (at most 2 seconds for a close that hangs).
- Manager: a log line when unload or shutdown returns while a worker is still being launched; a test for the failure to start the reader thread (which now also lists the worker's processes before ending it); an error after the worker process was created but before a handle to it exists ends the process tree before the error goes on; the real 50 ms waits in the tests are gone.
- The limits of the start-of-piece trimming are pinned by tests (below) and the real test's loose ends are tidied (worker_pids, no private helper, the timing in its header).
- The worker's exit-code test now fails if `run()` falls back to the ordinary interpreter exit (it raises once the reader thread is running).
- Checked and found already true: saving a result writes under another name (`<name>.wav.tmp`) and moves it into place, removing the temporary file on failure (the list asked for `.part`; the existing `.tmp` does the same job); `api_training.py` has no dead `if False` any more; the backend's 10 second shutdown wait being shorter than the manager's own waits is by design (the stop keeps running on its thread, see the comment at `SHUTDOWN_WAIT` in `api.py`).
- Fixed by the frontend half of the wave: an engine that failed while idle is now shown on the Speak tab ("The voice engine stopped. It starts again the next time you speak."), and the Speak tab reloads its device list.

### What the final fix wave changed for the user

- The voice engine is unloaded after 10 to 10.5 idle minutes whichever tab is open; before, it stayed on the graphics card until training started or the app quit.
- Pressing Speak in the second in which that unload is in progress is refused with "The voice engine of an earlier request is still shutting down ... Try again in a moment." (Before, the status poll or the request itself did the unload, so this could not be seen.)
- While an idle engine is being stopped as planned the status says "stopped", not "failed": the screen's new "The voice engine stopped" message is only for an engine that died, hung, could not be started or could not be ended.
- A voice whose only usable clips are too quiet is greyed out with the reason instead of failing when spoken; a voice with a quiet preferred clip and a louder other clip now speaks, from the louder clip.
- Messages about a refused reference clip name "Clip N" and end with what can be done.
- Speech is louder than before for most voices and equally loud from sentence to sentence (peak 0.89 for every piece; before, whatever the model produced, at most 0.97), and much louder for a voice with a quiet reference clip.
- Speech on the listen device follows the Live tab's listen volume once live mode has been started (or its slider moved while live) since the app opened; the virtual mic is unaffected.
- "1,200" is spoken as "one thousand two hundred" and phone numbers digit by digit; other numbers as before.
- An environment check that hangs is given up after 20 seconds (quick) or 5 minutes (full) with a message, instead of holding Speak, live start, recording and training for up to 5 minutes.
- Stopping speech can take up to 2 seconds longer to return in the rare case that an audio device hangs while closing (the sound itself still stops at once).

### Open items

- The verified-kill and environment-check logic is duplicated between `training.py` and `tts.py` (ruling above).
- Nothing detects a wrongly spoken sentence. After Fast was raised from 8 to 10 steps the measured rate of sentences with a wrong word is 2.8 percent at Fast and 1.7 percent at Balanced, with the model's own sample clip and a speech recogniser as the judge; a real voice's clip may give other figures.
- At 8 steps errors grew with the length of the piece (one long piece in ten lost three words or more); at 10 steps and above none did, so the piece limit was left as it is.
- The model's pieces run to the very end of their allotted time; a last word was never lost in the checked pieces, but there is no margin.
- At 8 steps about 4 percent of Fast pieces kept their full length because no pause could be found at their start (mostly the garbled ones); this was not counted again at 10 steps.
- **The start-of-piece trimming has two known limits, pinned by tests and not hardened**: a real first word of under 0.4 seconds that begins the piece and is followed by a pause of 0.4 seconds or more is cut together with the pause (however long the pause is), and a first sound shorter than 60 ms followed by a gap is dropped. Hardening was considered (drop a lead-in only when the gap is much quieter than the sound, and only up to about 1.5 seconds) and rejected: the measured lead-ins are a sound 12 to 30 dB below the loudest frame followed by 0.4 to 1.2 seconds of room tone that is itself only 30 to 45 dB down (the spike saw up to 2 seconds of it), so either condition would leave the reference clip's tail in some real pieces. The thresholds rest on one reference clip.
- Best loads the voice's RVC model again for every request (about 2 seconds before the first sound) and drops it afterwards.
- `is_silent` looks at the peak only, so a single click counts as speech (and such a piece is now also raised to the common level); a sentence cut hard at an over-long word can start a piece with punctuation.
- The level of a piece is set by its peak, not by its loudness, so a piece with one sharp peak is played quieter than its neighbours.
- There is no speed control for speech (fixed at 1.0; the design does not ask for one).
- The listen volume reaches the backend only when live mode is started or its slider is moved while live mode runs; a slider moved while live mode is off is not seen by text to speech, and the Speak tab has no volume control of its own. (The frontend was told not to add request fields in this wave; `speak` on the manager already takes a `monitor_gain`.)
- Pressing Speak while the housekeeping thread is ending an idle engine is refused instead of waiting the fraction of a second the kill takes.
- The level of a clip is measured once and remembered by the file's size and date; a voice with very many quiet clips makes the first listing of voices read each of them once.
- A result that was being saved when the backend was killed can leave a `<name>.wav.tmp` under `data/tts/`; nothing removes it.
- The uv package cache that `tts/.venv` and `trainer/.venv` hard-link into lives under `spike/stage3/cache/uv` on the development machine. Deleting that folder does not break the environments (hard links keep the files), but a later setup would download everything again.
- The Speak tab polls the status on every tab while a request is active (the plan said only while visible), so that the other tabs know when speech ends.
- Recording and training are not known to the Speak tab in advance: it shows the backend's refusal after Speak is pressed. Starting a recording is not disabled on the Voices tab while speech plays; the refusal message appears instead.
- A voice whose reference clip's audio file is missing is refused when spoken ("The audio of clip N ... is missing") unless another clip is known to be loud enough, which is then used.
- A request stopped near its end saves nothing, even with "Keep as a WAV file" ticked.
- The first speak request of a backend can take about 3 seconds to answer if the Speak tab's own environment check has not run yet.
- **A live session that fails by itself closes its streams outside the audio lock** (stage 1's `live.py`, a frozen file; found in the Task 7 review). The route-level stop is under the lock, but a session that ends on its own error is not, so for a moment another activity could open a device beside streams that are being closed. The devices are already dead at that point.
- **The offline converter returns a view of the engine's array** (`tts_rvc.OfflineConverter.convert`, Task 4 review): a caller that changed the result in place would change the engine's own buffer. The manager copies before it changes anything (`trim_edges` and `finish` copy), so nothing goes wrong today.
- **From the Task 2 review**: one assertion in the text tests is weaker than its name, and the worked example in `test_tts_audio.py` ("4.89 s ... gives 178") is a figure fitted to make the package's formula come out, not a measured clip length (the measured sample is 5.33 s).
- Several other tests are looser than their names (three in the reference and RVC tests); the worker real test builds its worker outside the fixture's clean-up block.

### Not verified by a human

Everything in stage 4 was exercised by automated tests only: fakes for the app side, and for the real model a stand-in player with a speech recogniser as the only listener.

- **Nobody has listened to any synthesised speech.** Whether it sounds like the voice, sounds natural, starts and ends cleanly, and whether the 0.3 second pauses are right, are all unknown. The reference used in every real run is the sample clip that ships with the model, not a voice from this app.
- **No real voice has been recorded or trained.** The Best preset was run only with stage 3's test model, trained on about a minute of synthetic sound: it proves the two models run together and the timing, not that Best sounds better than Balanced.
- **The app has never been driven by hand.** The Speak tab in the real window, real playback on a real device, the virtual microphone being heard by another app, saving and playing a file, Stop and Escape, the tab's messages, and Quit leaving no worker behind are covered only by tests with fake devices and a fake backend. The real `/tts` routes have never served the real manager.
- The whole "Stage 4 first run checklist" in `README.md` (10 steps).
- `bash tts/setup.sh` on a machine that is not this one (clean caches, another driver), and the real download sizes there. Here the AMD wheels were already cached.
- Reference clips other than the sample: a real room, a fast or slow speaker, a clip at the 3 or 12 second limits, an automatic transcript with mistakes.
- Text other than plain English prose in the real model: abbreviations, names, very short pieces ("Hi."), a 5000 character text from start to end, other languages. Numbers: the rewritten forms of four sentences were checked by a speech recogniser at Balanced (20 of 20 tries heard correctly); nobody has listened to them, and Fast and Best were not measured.
- Long sessions: many requests in a row, memory over time. The idle unload was run for real only with the idle time shortened to 2 seconds (the engine was gone within one housekeeping round); the full 10 minutes in the running app, and Task Manager showing the engine gone while another tab is open, have not been watched by anyone.
- **Everything the final fix wave changed in what is heard**: that sentences now sound equally loud and that the common level (peak 0.89) is comfortable, also at Best and with a quiet reference clip, where the gain is large and raises the clip's noise with the speech; that the Live tab's listen volume is the right volume for speech; how the model says the written-out numbers.
- A real voice whose clip is too quiet: the greyed-out entry and its reason on the Speak tab, and the message for a clip the engine refuses, have been seen only in tests.
- An environment check that really hangs (the 20 second limit and the ending of its processes were tested with fakes only).
- Refreshing devices and then speaking, and the new first-start wording, on the real Speak tab (frontend half of the wave).
- Other GPUs. Graphics memory was read from Windows on a 24 GB card (about 2.8 GB for the loaded engine, 5.3 GB with the RVC model beside it); a card with much less memory, or other programs using it, was not tried.
