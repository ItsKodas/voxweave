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
