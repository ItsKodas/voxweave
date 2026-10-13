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
