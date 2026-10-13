# Stage 4: first real run of the TTS worker

Date: 2026-10-13. `tts/tts_worker.py` (the real `F5Engine`) run for the first time on the RX 7900 XTX, in the
environment built by `tts/setup.sh` (`tts/.venv`: torch 2.9.1+rocm7.2.1, f5-tts 1.1.22; `tts/models`). Nothing in
the environment was changed.

**Result: the real engine worked on its first run. No fix to the engine was needed.** All twelve checks of the
real test pass, seven runs in a row.

## Commands

```bash
# the opt-in real test (about 25 s once the kernel cache is warm); -s prints the timings
AIVOICE_RUN_TTS_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_tts_worker_real.py -q -W error -s
# optional: also write the timings as JSON
AIVOICE_TTS_RESULTS=/path/results.json AIVOICE_RUN_TTS_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_tts_worker_real.py -q -W error
```

The test starts the worker as a launcher would:
`tts/.venv/Scripts/python.exe -u tts/tts_worker.py --model-dir tts/models --parent-pid <pytest's pid>`, with the
variables of section 4 of the spike notes, caches in pytest's temporary folder except `MIOPEN_USER_DB_PATH` and
`MIOPEN_CUSTOM_CACHE_DIR`, which point at `tts/.cache/miopen` (the real kernel cache). stderr goes to a log file.
It is skipped unless `AIVOICE_RUN_TTS_TEST=1` is set and `tts/verify.py --fast` passes.

Reference: the package's `infer/examples/basic/basic_ref_en.wav` (5.33 s, RMS 0.129) with the transcript from its
`basic.toml`, "Some call me nature, others call me mother nature." Text: the spike's ten lighthouse sentences
(the first three for the three-piece requests), seed 7.

## What the test covers (all pass)

- `ready` arrives; the model is `float16`; no "warm-up failed" line in the log.
- `set_reference` reports 5.333 s.
- Three pieces at 8 steps: `started`, three `audio` in order, `done`; each file read with the backend's `read_wav`,
  finite, audible, 24 kHz, and of a believable length.
- A second request at 16 steps on the same worker.
- The backend's Whisper (small.en, CPU, 8-bit) hears the first 16-step piece word for word (15 of 15 words).
- `cancel` straight after `started` on a ten-piece request at 32 steps: `cancelled` with no piece announced; a
  following request works.
- `cancel` after the first `audio` of a ten-piece request at 32 steps (the second piece is being sampled):
  `cancelled` after 4 to 5 ms, only the announced file on disk; a following request works.
- A 1 s and a 20 s reference are refused with `bad_request`, and the earlier reference is no longer in force
  (the next `synthesize` is refused until a good reference is set).
- A reference lowered to RMS 0.0005 is refused with "too quiet".
- A transcript ending in `?` or `!` works (same audio length as with `.`: 9.067 s).
- `shutdown`: exit code 0, no process of the worker's tree left, stdout held nothing but JSON lines, no stray file
  in the output folder, no traceback in the log.

## Timings

| | First run ever (empty kernel cache) | Second run | Runs 3 to 7 (warm) |
| --- | ---: | ---: | ---: |
| imports (`import_s`) | 13.2 s | 15.3 s | 4.1 to 4.6 s |
| model load (`load_s`) | 3.2 s | 3.3 s | 2.9 to 3.3 s |
| warm-up synthesis (`warmup_s`) | 15.6 s | 12.0 s | 5.7 to 6.9 s |
| **process start to `ready`** | **32.6 s** | 31.5 s | **13.1 to 14.8 s** |
| request to first `audio`, 8 steps | 0.83 s | 0.52 s | **0.47 to 0.49 s** |
| per piece, 8 steps (three pieces) | 0.83 / 0.87 / 0.80 s | 0.52 / 0.47 / 0.41 s | 0.48 / 0.46 / 0.42 s |
| request to first `audio`, 16 steps | 0.83 s | 0.81 s | **0.81 to 0.83 s** |
| per piece, 16 steps | 0.83 / 0.89 / 0.81 s | 0.81 / 0.86 / 0.77 s | 0.80 / 0.87 / 0.77 s |
| first `audio`, 32 steps (from the cancel test) | | | 1.59 to 1.62 s |
| cancel to `cancelled`, before the first piece | 5 ms | 5 ms | 4 to 7 ms |
| cancel to `cancelled`, in the middle of a piece | | | 4 to 5 ms |

The three pieces are 87, 95 and 86 characters and come back as 9.07, 9.90 and 8.96 s of untrimmed audio
(the worker no longer trims; the backend does).

Memory: `vram_reserved_mb` 838 at `ready` (the spike measured 844); peak host memory of the worker's process tree
1.6 to 1.7 GB (the spike: 1.2 to 1.6 GB). Peak VRAM while synthesising was not read (it would need a query inside
the worker).

Warm-up on the main thread, pieces on the engine thread (review point): the very first real piece of a worker took
0.476 s at 8 steps; the same text, seed and steps later in the same session took 0.405 s. So the first piece pays
about 0.07 s, not the 3 to 4 s of an unwarmed process: the warm-up does carry over to the engine thread, and it
was left where it is.

## What differed from the spike notes

1. **Start to ready is 13 to 15 s warm, as the spike said (14.8 to 19.3 s), but the first two starts took 32 s.**
   Imports took 13 to 15 s on the first two runs and 4 s afterwards (the environment is new: Python writes its
   bytecode files on first import, which the spike's notes also expect), and the warm-up took 15.6 s and 12.0 s
   with an empty kernel cache. The backend's 120 s start-up timeout covers this. The kernel cache must persist
   (`tts/.cache/miopen`); a launcher that pointed it at a fresh folder would pay this on every start.
2. **The warm-up costs 5.7 to 6.9 s when warm, against the spike's 3.3 to 4.0 s.** This worker warms up on three
   seconds of noise and a fixed sentence instead of the user's clip, so the output length is one the kernel cache
   sees only here; either way it is paid before `ready`, not by a request.
3. **First audio at 8 steps is 0.47 to 0.49 s, against 0.41 to 0.42 s in the spike**; 16 steps 0.81 s against
   0.72 to 0.78 s; 32 steps 1.6 s against 1.55 to 1.68 s. The difference at 8 steps is the fixed per-piece work
   around sampling (writing a 9 s file, the event) plus the 0.07 s the first piece pays. Another test run
   (`backend/tests/test_tts.py`, another session) was using the processor during some of these runs.
4. **Lengths that the kernel cache has not seen cost more than the spike's 0.3 to 0.45 s**: on the first run every
   8-step piece took about 0.4 s longer than warm (0.83 against 0.42 to 0.48 s); once per length, then cached on
   disk.
5. Nothing else differed: half precision is in force (`dtype: float16`), no network use, no library output on
   stdout (the log holds only the package's own "Load vocos from local path" and "vocab / token / model" lines,
   which it prints to stdout and the worker redirects), the cancel hook stops a piece within 5 ms as measured in
   the spike (5 and 13 ms), and the worker ended with exit code 0 with nothing left running.
6. The noise warm-up produced finite audio in half precision (the worker now logs a line if it ever does not).

## Fixes

None came from the real run. The fixes made in the same change came from the review of the worker and are
unit-tested in `backend/tests/test_tts_worker.py`:

- No line of input can end the worker: a `speed` too large to be a float and text that is not valid Unicode are
  `bad_request`; any unexpected error while handling a command is an `internal` error and the worker goes on; an
  unexpected error while running a job no longer leaves the worker refusing everything as "already synthesising".
- The process always ends through its own hard exit (exit code 1 if anything escaped), never the ordinary
  interpreter exit.
- The parent is checked before the model loads and while it loads (the worker ends on the spot), and `ready` is
  not sent to a parent that is gone.
- A reference quieter than RMS 0.003 is refused as "too quiet" instead of every piece failing as near-silent.
- Bad command-line arguments give a JSON `error` line and exit code 2.
- The unknown-command message is cut to 100 characters; a reference length that is not a number has its own
  wording.

## State at the end

No worker process is running. Written outside pytest's temporary folders: the kernel cache
`tts/.cache/miopen/gfx1100_48.ukdb` and `tts/.cache/miopen/db/` (as intended), and Python's bytecode files inside
`tts/.venv`.

---

# Stage 4: first real run of the whole pipeline (manager, worker, model)

Date: 2026-10-13. The real `TtsManager` (`backend/aivoice/tts.py`) driving the real worker for the first time, on
the RX 7900 XTX, nothing else using the GPU. Until now the manager had only been run against a stand-in worker.
No audio device was opened, no backend server or app window was started, and **nobody has listened to any of
this audio**: everything below is measured by program (levels, lengths, timings, and the backend's Whisper model
as the listener).

**Result: the manager, the worker protocol, stop, reuse, unload and the Best preset all worked on the first run
with no change. One real fault was found and fixed, in how the start of each piece is trimmed. One thing was
found that is not fixed: the Fast preset gets words wrong in about three sentences out of ten.**

## Commands

```bash
# the opt-in end-to-end test (about 45 s once the kernel cache is warm); -s prints the timings
AIVOICE_RUN_TTS_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_tts_real.py -q -W error -s
# optional: also write the timings as JSON
AIVOICE_TTS_MANAGER_RESULTS=/path/results.json AIVOICE_RUN_TTS_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_tts_real.py -q -W error
```

Skipped unless `AIVOICE_RUN_TTS_TEST=1` is set and `tts/verify.py --fast` passes (the same gate as the worker
test; both files can run in one session).

## How the test is set up

- A `TtsManager` with its own launcher, its own environment check (`tts/verify.py`, really run) and its own way
  of loading a voice's RVC model. The only stand-ins are the player (it opens nothing, keeps what it is given,
  and counts everything as played the moment `finish()` is called) and the device list (three made-up devices).
- The data folder is pytest's temporary folder: the library, the voice, `tts/worker.log`, `tts/tmp/` and the
  saved WAV are all there. The real `data/` is only read, for the Best preset: `pretrained/vec-768-layer-12.onnx`
  is copied into the temporary folder, and `dml_device_id` is read from `settings.json`.
- The worker is started by the manager's own launcher, so its caches and temporary files go where the app puts
  them: `tts/.cache` (kernel cache `tts/.cache/miopen`, plus `tmp`, `hf`, `torch`, `xdg`, `mpl`, `numba`). After
  all runs `tts/.cache` holds 742 KB.
- The voice has one clip: the package's `basic_ref_en.wav` (5.33 s), resampled to 48 kHz and added through
  `Library.add_clip` as a recorded clip with the transcript from the package's `basic.toml`. The limit for a
  piece with this reference is 132 bytes, so the three lighthouse sentences (87, 95 and 86 characters) are three
  pieces and the ten are ten.
- For Best, the stage 3 spike's exported test model (`spike/stage3/handoff/data/pretrained/rvc/final3`:
  `model.onnx` 111 MB, `model.index` 29 MB, 40 kHz) is copied into the voice's `models/rvc/`, where a trained
  voice keeps its model. It was trained on about a minute of synthetic sound, so it says nothing about likeness.
  That part is skipped if the folder or the feature model is gone.
- The status is polled the way the screen polls it while each request runs.

## What the test covers (10 tests, all pass, six runs in a row with the final code)

- The environment check run by the manager passes (2.6 to 3.0 s for the full check).
- Fast, from nothing: the engine reads `starting` and then `ready`, the request goes through `preparing` and
  `speaking`, three pieces reach the player in order, each mono float32 at 48 kHz, audible, faded at both ends,
  peak at most 0.97, of a believable length, and starting with the sentence (no quiet stretch of 0.4 s in its
  first 0.7 s). The saved WAV is in the temporary `data/tts/`, at 48 kHz, exactly the three pieces plus two
  pauses of 0.3 s. Nothing is left in `data/tts/tmp/`.
- A second Fast request (to two outputs) and a Balanced one use the same worker processes; nothing is saved
  unless asked.
- Whisper (small.en, CPU) hears each of the three Balanced pieces as its own sentence, word for word, in every
  run. That is also the proof of the order.
- `stop()` while the worker is busy with the second of ten pieces: idle again after 0.04 s (the limit in the
  test is 2 s), no error, nothing saved, no temporary folder, the worker still loaded and the same processes; the
  next request plays exactly its own three pieces.
- Best is refused for the voice while it has no model. With the model in place, Best plays three pieces, twice;
  each conversion returns exactly as many samples as it was given, and the piece that is played is at most 20
  percent shorter than that (the edges are trimmed once more after conversion).
- `unload()`: both worker processes are gone within 0.3 s, the engine reads `stopped`, and `status()` never
  failed.

## Timings

Six runs with the final code, GPU otherwise idle. "First piece" is from the call to `speak()` to the moment the
piece is handed to the player, so it includes the manager's own work (reading the file, trimming, resampling to
48 kHz) on top of the worker's.

| | Measured |
| --- | ---: |
| full environment check (`tts/verify.py`, once per backend) | 2.6 to 3.0 s |
| first request after start: `speak()` to first piece (worker start included), Fast | 10.8 to 11.1 s |
| the same in the four earlier runs of this session | 12.0 to 16.6 s |
| first piece, worker loaded, Fast (8 steps) | **0.42 to 0.43 s** |
| each following piece, Fast | 0.38 to 0.42 s |
| three sentences (about 24 s of speech), Fast, whole request | 1.21 to 1.23 s |
| first piece, worker loaded, Balanced (16 steps) | **0.80 to 0.84 s** |
| each following piece, Balanced | 0.75 to 0.83 s |
| three sentences, Balanced, whole request | 2.36 to 2.41 s |
| first piece, Best (32 steps plus RVC), first Best request of a session | **4.2 to 4.6 s** |
| first piece, Best, second Best request | 3.65 to 4.15 s |
| three sentences, Best, whole request | 6.0 to 6.5 s (one run 7.8 s) |
| loading the voice's RVC model and warming it up (every Best request) | 2.3 to 2.6 s first, 1.7 to 2.2 s after |
| one RVC pass over a piece of about 8 s | 0.50 to 0.63 s |
| `stop()` to idle, in the middle of a piece | 0.037 to 0.045 s |
| `stop()` to the player being silenced | 0.1 ms |
| first piece of the request after a stop, Fast | 0.41 to 0.43 s |
| `unload()` (two processes ended and checked) | 0.25 to 0.31 s |

The played pieces are 7.5 to 9.1 s long for these sentences (the model's untrimmed 9.0 to 9.9 s less the
lead-in).

How Best spends its time: the manager sends the request to the worker and then loads the voice's model, so the
two overlap. The first piece needs the model to be loaded (about 2 s), the worker's first 32-step piece (1.6 s
alone, slower while the model is loading beside it) and one RVC pass (0.6 s). By then the worker has usually
finished the second piece, which follows 0.6 s later; the third follows 0.9 to 1.6 s after that. The RVC model
is loaded again for every Best request and dropped at its end.

**Both models fit on the card together.** Dedicated graphics memory in use, all programs together, read from
Windows (`typeperf "\GPU Adapter Memory(*)\Dedicated Usage"`) on the 24 GB card:

| | Dedicated memory in use |
| --- | ---: |
| before the worker starts (the desktop and other programs) | 3.6 GB |
| worker loaded, after its first request | 6.4 GB (2.8 GB more) |
| during a Best request (worker plus the RVC model and feature model under DirectML) | 8.9 GB (2.5 GB more; one reading of 11.1 GB) |
| after `unload()` | 3.6 GB |

The worker reports 838 MB reserved by PyTorch at `ready`; the 2.8 GB is what Windows counts for the process,
which includes what the GPU libraries hold themselves. Fast and Balanced timings were the same before and after
the Best requests.

## The fault that was found: the start of a piece

The spike notes say the model puts 1 to 2 s of quiet before each sentence and that trimming on frame levels 30 dB
below the loudest frame removes it. The stand-in worker in `test_tts.py` writes exactly that: half a second of
digital silence, then sound. The real pieces are different (282 raw pieces recorded and measured, 186 at 8 steps
and 96 at 16):

1. **The piece continues the reference clip, and its first frames are often not quiet.** The first 60 to 300 ms
   hold sound: usually the dying end of the reference's last word, 12 to 30 dB below the piece's loudest frame;
   at 8 steps sometimes a full-level stray sound of about 0.3 s that is louder than the sentence that follows.
   Then comes the pause (0.4 to 1.2 s), then the sentence.
2. **The pause is the reference's room tone, not silence**, and it lies only 30 to 45 dB below the speech, so
   single 20 ms frames of it reach the threshold.
3. In both cases the trimming found "sound" at or near the first frame and cut nothing. **The old trimming left
   the whole lead-in in 134 of 180 Fast pieces (74 percent) and 4 of 90 Balanced pieces** (ten-sentence and
   three-sentence requests; in the first small sample it was 3 of 6 at Balanced). A listener would have heard a
   blip and then a second of room tone before those sentences, on top of the 0.3 s pause between pieces, and
   the first sound of a Fast request would have come a second late.

Fix (in `backend/aivoice/tts_audio.py`, used by `tts.py` on every piece as it comes from the worker):

- `trim_edges` now starts a piece where sound first lasts 60 ms (three frames in a row), not at the first single
  frame over the threshold. The end is unchanged: the last loud frame, however short.
- New `drop_reference_tail`, applied before `trim_edges`: if lasting sound begins within the first 60 ms of the
  raw piece, has ended within 0.4 s, is followed by at least 0.4 s without lasting sound, and lasting sound comes
  after that, everything before that later sound is cut (bar 30 ms). Otherwise the piece is returned whole.

With the fix, none of the 282 recorded pieces has a quiet stretch of 0.4 s in its first 0.7 s. Whisper on all of
them, as the old trimming left them and as the new one does: Fast 129 words missing before and 138 after out of
2896 (the difference is Whisper hearing garbled pieces differently, see below), pieces with any wrong word 54
before and 55 after, pieces not beginning with the sentence's first word 7 before and 4 after; Balanced
identical before and after (1 word missing in 1448). So the fix removes the lead-in without costing words.

What the fix can get wrong: a sentence that really begins, within 60 ms of the start of the model's output, with
a word shorter than 0.4 s followed by a pause longer than 0.4 s would lose that word. It did not happen in the
282 pieces. The thresholds come from one reference clip (the package's sample, which has audible room tone and a
slow pace); a real voice's clip may behave differently, and that needs ears.

The real test failed on this before the fix (7 of 10 tests, starting with the first Fast request) and passes
with it.

## Found and not fixed

1. **Fast (8 steps) gets words wrong often with this reference.** Whisper on 180 Fast pieces: 4.8 percent of
   the words missing, **55 pieces (31 percent) with at least one wrong word, 18 pieces (10 percent) missing three
   words or more.** It depends on the sentence: "On clear nights he could count the lights of seven fishing boats
   far out on the water." went wrong in 25 of 46 tries, "He trimmed the wick..." in 5 of 46. Balanced on 90
   pieces: one word wrong in 1448. The spike measured 0 to 4 percent word errors at 8 steps on one seed; the
   manager uses a new random seed for every request, and the worker's own real test uses a fixed one, so neither
   showed how often a sentence is affected. Nothing detects this: a garbled sentence is as loud and as long as a
   good one. This is the cost of the presets as designed, not a fault in the code; whether Fast is worth offering
   as it is needs a listener and a decision.
2. **The model's pieces end without a pause.** In most raw pieces the last loud frame is the last frame of the
   piece: the speech runs to the very end of the time the model was given. Whisper heard the last word in the
   Balanced pieces, so nothing was cut off here, but there is no margin (the spike's "end-of-piece truncation"
   risk).
3. **About 4 percent of Fast pieces (8 of 186) still come out at their raw length**, with no pause the trimming
   could find. These are mostly the garbled ones.
4. Best loads the voice's model anew for every request (about 2 s before the first sound). Keeping it loaded
   between requests would make the first piece of a Best request about as fast as its later ones.

## What was the same as the fakes assumed, and what was not

Same: the protocol and its order of events; the worker being reused; the reference being set once; `stop()`
getting its `cancelled` answer at once; the worker process tree being the launcher plus the interpreter (two
processes) and both being found and ended by `unload()`; the engine reading `starting`, `ready` and `stopped`;
24 kHz pieces being brought to 48 kHz; the RVC pass keeping the length; the saved file. No timeout came anywhere
near its limit (start 11 to 17 s of 120 s, a piece under 2 s of 60 s, a cancel 0.04 s of 10 s).

Different:

- **The start of a piece** (above): the stand-in's lead-in is clean silence; the real one is a stray sound and
  room tone. This is the only difference that needed a change to the product.
- **The piece's length for a given text is fixed** (9.07, 9.90 and 8.96 s for the three sentences in every run):
  the model is given a duration, it does not choose one. The stand-in writes 1.5 s for any text.
- The first request of a backend pays for the full environment check (2.6 to 3.0 s) before the worker is even
  started, inside the `speak()` call itself: `POST /tts/speak` will take about 3 s to answer the first time
  unless the Speak tab's own check has run first (it does when the tab is first shown). In this test it had.
- `status()` showed `preparing` for the whole start of the worker (11 s or more); the screen's "Getting ready"
  text covers that.

## State at the end

No worker process is running (checked by command line: no `python.exe` with `tts_worker.py`). Written outside
pytest's temporary folders: the kernel cache and the other caches under `tts/.cache` (742 KB in all), as the app
itself would. The real `data/` was not written.
