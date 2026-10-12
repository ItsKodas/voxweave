# Stage 3: first real runs of the training worker

Date: 2026-10-13. Machine: Windows 11 Pro, RX 7900 XTX, the environment built by `trainer/setup.sh`
(`trainer/.venv` with torch 2.9.1+rocm7.2.1, `trainer/rvc` at 81eed5e with both patches, exporter in `tools/.rvc`).
`trainer/verify.py --fast` reported every check ok before the runs. Nothing else was using the GPU.

## 1. Verdict

`trainer/worker.py` runs the whole real pipeline, start to finish, with no change to its stage commands,
environment, parsing or safety logic. Three real tests pass (`backend/tests/test_worker_real.py`):

1. A 2-epoch run trains, builds the index, exports, swaps the model into `models/rvc/`, cleans up, and the app's
   `RvcEngine` loads the result on DirectML and converts one second of audio to finite output of the same length.
2. A 6-epoch run is cancelled after its first finished epoch (exit code 2, no trainer process left), then resumed
   with `--resume`: it continues from the saved epoch and finishes.
3. A 6-epoch run whose worker process is killed outright (`TerminateProcess` on the worker's own Python, not the
   tree): every stage process ends with it, and a resume afterwards continues from the checkpoint and finishes.

The real runs found one mistake, in a test of mine and not in the worker (section 6). One log line was added to
the worker so the cost of the checkpoint backup can be read from `worker.log`.

Only synthetic speech-like audio was used (about 60 s, 12 clips), for 2 to 6 epochs. Voice quality and a
multi-hour run remain unproven, as in the spike.

## 2. Commands

```bash
# all three real tests (5 min 39 s with warm caches)
AIVOICE_RUN_TRAIN_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_worker_real.py -q -W error -s

# without the variable they skip
backend/.venv/Scripts/python -m pytest backend/tests -q -W error
```

`AIVOICE_TRAIN_TEST_KEEP=<folder>` copies each run's `progress.jsonl`, `worker.log` and `worker.out.log` there.

Each test starts the worker exactly as the app's training manager does:

```
trainer/.venv/Scripts/python.exe -P trainer/worker.py --voice-dir <tmp>/voices/real --job-dir <tmp>/voices/real/training
    --experiment test-real-<8 hex> --epochs 2 --batch-size 4 --save-every 1 --parent-pid <pytest pid> "--name=Real test voice"
```

The voice folder is in pytest's `tmp_path`. The experiment folder is `trainer/rvc/logs/test-real-<8 hex>/`. The
test fixture ends every process whose command line mentions the experiment and removes the experiment folder, the
small model and the index link, whether the test passed or not. After the last run `trainer/rvc/logs/` held only
`mute/`, and `assets/weights/` and `assets/indices/` were empty.

## 3. Wall time per stage

Dataset: 12 clips, 60 s, sliced by RVC into 22 segments (24 filelist lines, 6 steps per epoch at batch 4).
Times are differences between the first progress lines of consecutive stages.

| Stage | First run on this environment | Later run (warm caches) |
| --- | ---: | ---: |
| Prepare (hard links, fingerprint) | 0.02 s | 0.02 s |
| Preprocess (8 workers) | 21.8 s | 7.8 s |
| Pitch (RMVPE, GPU) | 39.9 s | 12.3 s |
| Features (HuBERT, GPU) | 25.3 s | 10.8 s |
| Train, 2 epochs | 92.3 s | 30.0 s |
| Index | 2.9 s | 2.8 s |
| Export (CPU) and swap | 9.0 s | 7.0 s |
| Whole run | 191 s | 70.7 s |

Training in detail (batch 4, 22 segments):

| | First run | Warm |
| --- | ---: | ---: |
| Trainer start to the first epoch line (includes the first epoch) | 86.7 s | about 25 s |
| First epoch, as the trainer reports it | 77.2 s | 14.0 to 15.4 s |
| Later epochs | 3.0 to 3.4 s | 3.0 to 3.4 s |
| First epoch after a resume | | 13.9 to 14.0 s |
| Last epoch line to trainer exit (small model, 1 s sleep, exit) | 2.4 s | 2.1 to 2.4 s |

Other measured costs:

- Checkpoint backup (copy of 1.3 GB, flushed to disk, checksum of the copy): 1.9 to 2.3 s per save, eleven times.
  None was rejected, although with `--save-every 1` and 3 s epochs the next save starts about a second later.
- Cancel: the cancel file appeared mid-epoch; the worker waited for the end of that epoch and ended the tree.
  From the file to the worker's exit: 2.9 s and 3.0 s. Exit code 2. No process left.
- Resume before training restarts: 13 s (pitch and features run again and skip all 22 segments), then 9 s until
  the trainer prints its two "Loaded checkpoint" lines.
- Worker killed outright: 2 worker processes (launcher shim and Python) and 2 stage processes (shim and trainer)
  existed before the kill; none was left afterwards (the test allows 60 s; the time was not measured more finely).

## 4. What differed from the spike notes

Nothing that changes the worker. Details worth knowing:

1. **A resumed trainer prints more than the spike notes list.** Besides the two untranslated
   `Loaded checkpoint '...' (epoch N)` lines there are `Loaded model weights` (twice),
   `Restored discriminator checkpoint` (translated), and two `DEBUG:<exp>:./logs\<exp>\D_2333333.pth` lines.
   The parser ignores them. The order is D then G, as the worker assumes.
2. **Start-up lines not in the notes**: a `DEBUG:infer.module.models:gin_channels: 256, ...` line and a long
   `INFO:<exp>:{'data': {...}}` line with the whole configuration. Ignored.
3. **A `UserWarning` from `lr_scheduler.py`** ("Detected call of `lr_scheduler.step()` before `optimizer.step()`")
   appears once after the first epoch of a resumed run. Harmless; it contains no word the worker treats specially.
4. **Preprocess progress lines arrive out of order** (`2/12`, `5/12`, `4/12`, ...) because eight processes report
   independently, with one `Worker completed | Success: n | Failed: 0` line per process (eight of them). The
   progress percent only ever moves up because the worker keeps the highest value; failures are summed over the
   eight lines.
5. **The index stage printed `Write progress: 1/1`** here (the spike showed `1/2` on a larger set).
6. **The first epoch ever took 77 s** at batch 4 on 22 segments (MIOpen compiling kernels); with the cache warm
   the first epoch of a run takes about 14 s and later ones about 3.1 s. The spike measured 5.1 s per epoch for
   181 s of audio at batch 8, so these are consistent.
7. **The trainer's output pipe closes promptly after `os._exit(0)`**: the worker saw the end of output and exit
   code 0 within about 2.4 s of the last epoch line, so data-loader workers do not hold the pipe open.
8. **The exporter prints about twenty lines of torch warnings** (`FutureWarning`, `TracerWarning`, constant
   folding). They are logged under `[export]` and do not affect the result.

## 5. Things to know that are not defects

- On a resume, the first training progress line reads epoch 0 at 13 % ("Loading the saved training state") for
  about 9 s, until the trainer confirms the checkpoint and the line "Continuing from epoch N" restores the real
  position. Percent never goes down within one run; across a cancel and its resume the file shows the drop.
- RVC repeats the number of the saved epoch after a resume (cancelled at epoch 2, the resumed run trains "epoch 2"
  again), as the spike notes say. The worker's epoch count does not go back.
- With `--save-every 1` on this tiny set the worker spends about 2 s of every 3 s epoch copying checkpoints. The
  app's default is every 5 epochs.

## 6. Fixes made

1. **Test only** (`test_worker_real.py`): my first version of the resume assertion required every training line of
   the resumed run to carry an epoch at or after the cancelled one. The real run showed the opening line of the
   training stage carries epoch 0 (see section 5). The assertion now checks what matters: the message
   "Continuing from epoch N" with N equal to the epoch the cancelled run ended on, and that the resumed run never
   printed "Loaded generator pretrained model". No unit test's premise was contradicted by the real pipeline, and
   no unit test was changed to fit.
2. **Worker, additive** (`trainer/worker.py`): each successful checkpoint backup is now logged
   (`[worker] checkpoint of epoch N backed up in X s`). Unit test: the backup test in `test_worker.py` asserts the
   line. This is how the 1.9 to 2.3 s figures above were measured.

No change was needed in `trainer/rvclog.py` or `trainer/filelist.py`.

## 7. Progress file of the successful 2-epoch run (warm caches)

```json
{"time": 1791844969.4501414, "stage": "prepare", "stage_index": 1, "stages": 7, "percent": 0.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Getting the clips ready"}
{"time": 1791844969.4649918, "stage": "prepare", "stage_index": 1, "stages": 7, "percent": 1.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "The clips are ready"}
{"time": 1791844969.4655118, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 1.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.76035, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 1.83, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.7947915, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 1.83, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.8354821, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 1.83, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.83599, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 1.83, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.8390722, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 2.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.843258, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 2.17, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.845826, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 2.33, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.8567007, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 2.33, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.8648949, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 3.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.9084764, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 3.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.916128, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 3.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844976.9299383, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 3.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Slicing the clips"}
{"time": 1791844977.2294235, "stage": "preprocess", "stage_index": 2, "stages": 7, "percent": 3.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "The clips are sliced"}
{"time": 1791844977.2294235, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 3.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844985.9173336, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 3.18, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844986.1208751, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 3.36, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844986.4175546, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 3.73, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844986.7054617, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 4.09, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844986.9927146, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 4.45, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844987.287101, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 4.82, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844987.602208, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 5.18, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844987.9314778, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 5.55, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844988.234705, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 5.91, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844988.5294726, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 6.27, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844988.7588496, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 6.64, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844989.063894, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 7.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844989.4846518, "stage": "pitch", "stage_index": 3, "stages": 7, "percent": 7.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Finding the pitch of each slice"}
{"time": 1791844989.4846518, "stage": "features", "stage_index": 4, "stages": 7, "percent": 7.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.355708, "stage": "features", "stage_index": 4, "stages": 7, "percent": 7.27, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.3879106, "stage": "features", "stage_index": 4, "stages": 7, "percent": 7.55, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.4372895, "stage": "features", "stage_index": 4, "stages": 7, "percent": 8.09, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.4765344, "stage": "features", "stage_index": 4, "stages": 7, "percent": 8.64, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.5176868, "stage": "features", "stage_index": 4, "stages": 7, "percent": 9.18, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.5594926, "stage": "features", "stage_index": 4, "stages": 7, "percent": 9.73, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.6013513, "stage": "features", "stage_index": 4, "stages": 7, "percent": 10.27, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.6636662, "stage": "features", "stage_index": 4, "stages": 7, "percent": 10.82, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.7217546, "stage": "features", "stage_index": 4, "stages": 7, "percent": 11.36, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.7809327, "stage": "features", "stage_index": 4, "stages": 7, "percent": 11.91, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.8230717, "stage": "features", "stage_index": 4, "stages": 7, "percent": 12.45, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791844999.8841822, "stage": "features", "stage_index": 4, "stages": 7, "percent": 13.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791845000.3262372, "stage": "features", "stage_index": 4, "stages": 7, "percent": 13.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Extracting voice features"}
{"time": 1791845000.3277738, "stage": "train", "stage_index": 5, "stages": 7, "percent": 13.0, "epoch": 0, "epochs": 2, "eta_seconds": null, "message": "Starting training"}
{"time": 1791845024.754088, "stage": "train", "stage_index": 5, "stages": 7, "percent": 55.0, "epoch": 1, "epochs": 2, "eta_seconds": null, "message": "Epoch 1 of 2"}
{"time": 1791845027.9113526, "stage": "train", "stage_index": 5, "stages": 7, "percent": 97.0, "epoch": 2, "epochs": 2, "eta_seconds": 0.0, "message": "Epoch 2 of 2"}
{"time": 1791845030.3642435, "stage": "train", "stage_index": 5, "stages": 7, "percent": 97.0, "epoch": 2, "epochs": 2, "eta_seconds": 0.0, "message": "Training finished"}
{"time": 1791845030.3647497, "stage": "index", "stage_index": 6, "stages": 7, "percent": 97.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Building the search index"}
{"time": 1791845033.079425, "stage": "index", "stage_index": 6, "stages": 7, "percent": 98.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Building the search index"}
{"time": 1791845033.1739917, "stage": "index", "stage_index": 6, "stages": 7, "percent": 98.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "The search index is built"}
{"time": 1791845033.1750112, "stage": "export", "stage_index": 7, "stages": 7, "percent": 98.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Exporting the model"}
{"time": 1791845039.88301, "stage": "export", "stage_index": 7, "stages": 7, "percent": 100.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "The model is exported"}
{"time": 1791845040.1419322, "stage": "export", "stage_index": 7, "stages": 7, "percent": 100.0, "epoch": null, "epochs": null, "eta_seconds": null, "message": "Training finished", "done": true, "ok": true, "error": null, "cancelled": false, "resumable": false}
```
