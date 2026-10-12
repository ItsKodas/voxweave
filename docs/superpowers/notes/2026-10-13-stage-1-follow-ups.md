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

### Deferred minor findings

Copied from the ledger. Where the code at the end of the build shows an item was fixed since, it is marked.

- Task 1: minor (deferred): huggingface-hub has no upper bound (resolved 2.2.0); notes do not list transitive packages; Whisper returns "You" for near-silence (do not rely on empty transcripts)
- Tasks 2+3: minor (deferred): duration cap not checked on resampler flush; peak memory for hour-long files; split_on_silence with tiny max_seconds or large pad_ms
- Tasks 4+5: minor (deferred): `created` uses local time without an offset, so listing order can be wrong in the repeated daylight-saving hour
- Task 8: minor (deferred): no test drives the lock with racing threads; no Task 8 test for a >20 s take
- Task 9: minor (deferred): VoicesPanel load has no stale-response guard; error banner persists until next successful load; tabs lack aria-controls; confirm spy not restored in finally
- Task 8: minor (deferred, FIX AT FINAL): a page on another localhost port is same-site and can still play clips; tighten /library so any browser request (Sec-Fetch-Site present) must carry an allowed Origin or Referer, now that the frontend fetches audio with fetch() [partly addressed: the /library check refuses a cross-site request that names no allowed Origin or Referer, but a same-site request (another port on the same host) with neither header is still allowed]
- Task 8: minor (deferred, FIX AT FINAL): delete of a voice can race the save inside record_stop; take the voices lock in record_stop after the audio lock [still open: record_stop in api_library.py takes the audio lock only]
- Task 8: minor (deferred): real webview headers unverified by a human; stop holds the audio lock through the disk save
- Task 10: minor (deferred, FIX AT FINAL): a failed Cancel (or tab-inactive cancel) leaves the UI idle while the backend may still be recording, with no Stop/Cancel button to retry [still open: RecordSession.tsx clears its recording state before the cancel request is known to succeed]
- Task 10: minor (deferred): tab going inactive during an in-flight Stop fires a concurrent cancel; in-flight-start test does not assert recordStop was not called

### Open at the end of the build

- The Task 11 review found two Important issues (the mount probe in ImportDrop.tsx adopts a running import that belongs to another voice, and a second import in the same mount can report the previous job's result). The ruling above adds a `voice_id` field to the import status as the fix. At the time of writing the import status in `backend/aivoice/importer.py` has no `voice_id`, so check the ledger before relying on the Voices tab with two voices and an import running.
- The stage 1 items under "Still open after the final review" still apply.

### Not verified by a human

Everything below was exercised only with fake devices, generated audio and a fake transcriber. The suites passing says nothing about these.

- The whole "Stage 2 first run checklist" in `README.md` (11 steps).
- Real microphone recording: device selection, the level meter, the four rejection messages with a real voice and a real room, and that leaving the tab stops the microphone.
- Drag and drop in the real Tauri window: that dropped file paths reach the Voices tab, and that nothing happens on the Live tab.
- Clip playback in the real webview: clips are fetched with `fetch()` and played from a blob URL, and the `/library` origin check was written against the headers the webview is expected to send, not ones observed. If the real webview sends different Origin or Referer values, playback and every Voices call would fail with a 403.
- Real Whisper transcription quality: how accurate `small.en` is on real recordings and on music or noisy files, the first-use download of about 500 MB, and import speed on this CPU. Whisper returns text such as "You" for near-silence, so quiet pieces are filtered by level, not by empty text.
- Recording and live mode refusing each other on real hardware, and Refresh devices being refused during a take.
