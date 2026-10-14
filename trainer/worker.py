"""The training worker: trains one RVC v2 voice model and hands it to the app.

  python -P trainer/worker.py --voice-dir <data/voices/id> --job-dir <data/voices/id/training>
      --experiment voice-<id> --epochs 200 --batch-size 8 --save-every 5 [--resume] [--parent-pid N] --name=<name>

Runs under the training environment's Python but uses only the standard library plus rvclog.py, filelist.py and
procutil.py from this folder.

It runs RVC's command-line stages one after another as subprocesses (commands and environment
are those proven in docs/superpowers/notes/2026-10-13-stage-3-training-spike.md, sections 3 and 4), appends one
JSON line per progress step to <job-dir>/progress.jsonl, copies every line of stage output to <job-dir>/worker.log,
and always ends with a final progress line. Exit code: 0 success, 1 failure, 2 cancelled.

What it protects:
- The recordings. Only <voice-dir>/clips/*.orig.wav are read, through hard links (or copies) in the experiment's
  own staging folder. Nothing under clips/ is ever written, renamed or deleted, and no file attribute is changed.
- The training state. RVC overwrites its only checkpoint pair in place and silently restarts from the pretrained
  base when it cannot read it. The worker keeps a verified second copy (.bak), stops a trainer that did not print
  its two "Loaded checkpoint" lines before it can save anything, and falls back to the copy.
- The model in use. The export goes to models/rvc.new/, is checked, and only then swapped into models/rvc/. The
  model it replaces is moved to <data>/trash/<voice-id>/models-<UTC time>/, not erased.
- The GPU. Stages are killed as a whole process tree, and on Windows every child is tied to the worker's own life.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback
import zipfile
from collections import deque
from pathlib import Path

TRAINER_DIR = Path(__file__).resolve().parent
TOOLS_DIR = TRAINER_DIR.parent / "tools"


def _sibling(name):
    """Import a helper from this file's own folder (the worker is started with -P, so the folder is not on sys.path)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rvclog = _sibling("rvclog")
filelist = _sibling("filelist")
procutil = _sibling("procutil")
process_alive, wait_for_exit = procutil.process_alive, procutil.wait_for_exit
kill_process_tree, make_runner = procutil.kill_process_tree, procutil.make_runner
bind_children_to_this_process = procutil.bind_children_to_this_process

STAGES = ("prepare", "preprocess", "pitch", "features", "train", "index", "export")
STAGE_WEIGHTS = {"prepare": 1, "preprocess": 2, "pitch": 4, "features": 6, "train": 84, "index": 1, "export": 2}
CHECKPOINTS = ("G_2333333.pth", "D_2333333.pth")  # the fixed names train.py uses with "-l 1"
SAMPLE_RATE = "40k"
# Lines that show train.py is past loading and is training. The loss line is among them because it is one RVC
# never translates, so a trainer that did not resume is caught on any Windows display language.
TRAINING_UNDER_WAY = ("epoch_progress", "loss", "saved", "epoch_done")
# After a stop request the trainer is left alone until the current epoch ends, so that it is never killed during
# a save. How long that wait may last: a minute, or twice the mean length of the last epochs when they are
# longer than that (once two are known), and never more than a quarter of an hour.
CANCEL_GRACE_SECONDS = 60
MAX_CANCEL_GRACE_SECONDS = 15 * 60
BACKUP_SKIPS_WORTH_A_WARNING = 3
ETA_WINDOW = 10
TAIL_LINES = 20
MIN_MODEL_BYTES = 1024 * 1024
SELECTED_CLIPS = "selected-clips.json"  # in the job folder: the clips a run with a time limit uses
SAFE_EXPERIMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

CLIPS_CHANGED = ("The clips changed since this training run started. Start a new training run. "
                 "Or put the clips back as they were and resume.")
STATE_UNREADABLE = "The saved training state could not be read"
STAGE_TITLES = {
    "preprocess": "Slicing the clips",
    "pitch": "Finding the pitch of each slice",
    "features": "Extracting voice features",
    "train": "Training",
    "index": "Building the search index",
    "export": "Exporting the model",
}


# Segments without pitch or features that a run tolerates: up to this share of all segments.
MAX_UNUSABLE_SHARE = 0.02


# Half precision is fast but can overflow on some recordings: every loss becomes NaN and nothing learned after
# that is a voice. The run is then started once more from the beginning in full precision.
UNSTABLE_RETRY = "Training became unstable. Starting it again in full precision, which is slower but steady"
UNSTABLE_TWICE = ("Training became unstable (its numbers overflowed) even in full precision, so no usable voice "
                  "can come from these clips as they are. Listen through the clips for ones that are not clean "
                  "speech (music, shouting, clipping, very long silences), delete them, and train again.")


class _Unstable(Exception):
    """The trainer's losses turned into NaN or infinity."""


class JobError(Exception):
    """A failure with a message meant for the user. `resumable` says whether running again with --resume could
    get further than this run did (False when only a new training run can help)."""

    def __init__(self, message, resumable=True):
        super().__init__(message)
        self.resumable = resumable


class Cancelled(Exception):
    """The run was stopped on request (cancel file) or because the app went away."""


# ---------------------------------------------------------------- files

def copy_durable(source, target):
    """Copy a file and make sure the bytes are on the disk before returning."""
    with open(source, "rb") as reader, open(target, "wb") as writer:
        shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def write_json_atomic(path, value):
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def torch_file_intact(path):
    """False when a file saved by torch.save is missing, empty, cut short or corrupt.

    torch.save writes a zip archive whose directory comes last and whose entries carry checksums, so a file that
    was being written when the trainer died (or that was copied while it was being rewritten) fails this check."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if not magic:
            return False
        if not magic.startswith(b"PK"):
            return True  # not a zip (an old torch format): nothing here can judge it
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile, ValueError, EOFError):
        return False
    except Exception:  # a damaged archive can fail in many ways inside zipfile/zlib
        return False


def _file_state(path):
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def _remove_tree(path):
    """Remove a folder of the worker's own. Never follows links and never changes file attributes, so a hard link
    to a recording is only ever unlinked."""
    if os.path.lexists(path):
        shutil.rmtree(path, onexc=_remove_read_only_name)


def _remove_read_only_name(function, path, error):
    """rmtree could not remove an entry. A read-only file (a staged link to a recording that is read-only) is
    removed by name, leaving the file and its attributes alone; anything else is still an error."""
    if os.name == "nt" and isinstance(error, PermissionError) and os.path.isfile(path):
        procutil.unlink_ignoring_read_only(path)
    else:
        raise error


def _unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------- the job

class _Training:
    """What the worker knows about one launch of train.py."""

    def __init__(self, expect_resume):
        self.expect_resume = expect_resume
        self.resumed_lines = 0     # train.py prints one "Loaded checkpoint" line for D, then one for G
        self.not_resumed = False   # it started training without loading the saved state
        self.saving = False        # between a "Saving model" line and the end of that epoch
        self.kill_now = False      # an epoch just ended with a stop pending: the one safe moment to kill
        self.unstable = False      # a loss line showed NaN or infinity: nothing after it is worth keeping
        self.stop_seen_at = None
        self.grace = float(CANCEL_GRACE_SECONDS)  # set when the stop is first seen

    @property
    def resume_confirmed(self):
        return self.resumed_lines >= 2


class Job:
    def __init__(self, args, runner, clock, parent_alive, notes=()):
        self.args, self.runner, self.clock, self.parent_alive = args, runner, clock, parent_alive
        self.notes = list(notes)
        self.voice_dir, self.job_dir = Path(args.voice_dir), Path(args.job_dir)
        self.experiment = args.experiment
        self.rvc = TRAINER_DIR / "rvc"
        self.cache = TRAINER_DIR / ".cache"
        self.python = TRAINER_DIR / ".venv" / "Scripts" / "python.exe"
        self.exp = self.rvc / "logs" / self.experiment
        self.weights = self.rvc / "assets" / "weights" / (self.experiment + ".pth")
        self.cancel_file = self.job_dir / "cancel"
        self.stage, self.percent = "prepare", 0.0
        self.epochs_done, self.epoch_times, self.eta = 0, deque(maxlen=ETA_WINDOW + 1), None
        self.backups_skipped = 0  # in a row
        self.stop_reason = None
        self.done_stages = {}
        self.fp32 = False  # full precision, after half precision turned the losses into NaN
        self.tail = deque(maxlen=TAIL_LINES)
        self._progress = self._log_file = None

    # ---- lifecycle

    def execute(self):
        ok, cancelled, error, message, resumable = False, False, None, "", True
        try:
            self._open_files()
            self._pipeline()
            ok, message, resumable = True, "Training finished", False
        except Cancelled:
            cancelled, message = True, self._stop_message()
        except KeyboardInterrupt:
            cancelled, message = True, "Training was cancelled"
        except JobError as exc:
            error, resumable = str(exc), exc.resumable
        except BaseException as exc:  # whatever happens, the run must end with a final line
            error = "%s: %s" % (type(exc).__name__, exc)
            self._log("worker", traceback.format_exc().rstrip())
        try:
            self._final_line(ok, cancelled, error, resumable,
                             message or (error or "Training failed").splitlines()[0])
        finally:
            _unlink_quietly(self.cancel_file)
            for handle in (self._progress, self._log_file):
                try:
                    if handle:
                        handle.close()
                except OSError:
                    pass
        return 0 if ok else 2 if cancelled else 1

    def _open_files(self):
        self.job_dir.mkdir(parents=True, exist_ok=True)
        _unlink(self.cancel_file)  # a request left over from an earlier run is not for this one
        mode = "a" if self.args.resume else "w"
        # The progress file first: if the log cannot be opened, the run still ends with a final progress line.
        self._progress = open(self.job_dir / "progress.jsonl", mode, encoding="utf-8", newline="\n")
        self._log_file = open(self.job_dir / "worker.log", mode, encoding="utf-8", errors="replace", newline="\n")
        for note in self.notes:
            self._log("worker", note)

    def _pipeline(self):
        self._prepare()
        for stage in STAGES[1:]:
            self._check_stop()
            self.stage = stage
            getattr(self, "_" + stage)()
        self._clean_up()

    # ---- progress, log, stop

    def _emit(self, fraction, message, final=None):
        before = sum(STAGE_WEIGHTS[s] for s in STAGES[:STAGES.index(self.stage)])
        percent = before + STAGE_WEIGHTS[self.stage] * min(1.0, max(0.0, fraction))
        self.percent = max(self.percent, round(percent, 2))  # never goes down within a run
        training = self.stage == "train"
        line = {
            "time": self.clock(), "stage": self.stage, "stage_index": STAGES.index(self.stage) + 1,
            "stages": len(STAGES), "percent": self.percent,
            "epoch": self.epochs_done if training else None, "epochs": self.args.epochs if training else None,
            "eta_seconds": self.eta if training else None, "message": message,
        }
        if final:
            line.update(final)
        if self._progress:
            self._progress.write(json.dumps(line) + "\n")
            self._progress.flush()

    def _final_line(self, ok, cancelled, error, resumable, message):
        final = {"done": True, "ok": ok, "error": error, "cancelled": cancelled, "resumable": resumable}
        try:
            if ok:
                self.percent = 100.0
            self._emit(1.0 if ok else 0.0, message, final)
        except Exception:
            self._log("worker", "could not write the final progress line:\n" + traceback.format_exc().rstrip())

    def _log(self, source, text):
        try:
            if self._log_file:
                for line in str(text).splitlines() or [""]:
                    self._log_file.write("[%s] %s\n" % (source, line))
                self._log_file.flush()
        except (OSError, ValueError):
            pass  # a log that cannot be written must not stop a training run

    def _poll_stop(self):
        if self.stop_reason is None:
            if self.cancel_file.exists():
                self.stop_reason = "cancel"
            elif self.args.parent_pid and not self.parent_alive(self.args.parent_pid):
                self.stop_reason = "parent"
            if self.stop_reason:
                self._log("worker", "stop requested: " + self._stop_message())
        return self.stop_reason

    def _stop_message(self):
        return "The app was closed" if self.stop_reason == "parent" else "Training was cancelled"

    def _check_stop(self):
        if self._poll_stop():
            raise Cancelled()

    # ---- running a stage

    def _stage_env(self):
        cache = self.cache
        for folder in ("miopen/db", "miopen/cache", "hf", "torch", "mpl", "numba"):
            (cache / folder).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": str(self.rvc), "PYTHONSAFEPATH": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
            "ROCM_SDK_TARGET_FAMILY": "custom", "RVC_CUDA_GRAPH": "0",
            "HF_HOME": str(cache / "hf"), "TORCH_HOME": str(cache / "torch"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "MIOPEN_USER_DB_PATH": str(cache / "miopen" / "db"),
            "MIOPEN_CUSTOM_CACHE_DIR": str(cache / "miopen" / "cache"), "MIOPEN_LOG_LEVEL": "3",
            "MPLCONFIGDIR": str(cache / "mpl"), "NUMBA_CACHE_DIR": str(cache / "numba"),
        })
        if self.fp32:
            env["AIVOICE_TRAIN_FP32"] = "1"  # read by the trainer (trainer/patches/rvc_train_fp32_env.patch)
        return env

    def _run(self, argv, on_line=None, should_stop=None, cwd=None, env=None):
        """Run the current stage's command. Every output line is logged; returns the exit code."""
        stage = self.stage
        self.tail.clear()

        def handle(line):
            line = line.rstrip("\r\n")
            self._log(stage, line)
            self.tail.append(line)
            if on_line:
                on_line(line)

        argv = [str(part) for part in argv]
        self._log("worker", "starting %s: %s" % (stage, json.dumps(argv)))
        code = self.runner(stage, argv, cwd or self.rvc, env or self._stage_env(), handle,
                           should_stop or (lambda: bool(self._poll_stop())))
        self._log("worker", "%s ended with exit code %s" % (stage, code))
        return code

    def _require_success(self, code):
        if code != 0:
            raise JobError("The step \"%s\" failed (exit code %s). Its last output:\n%s" % (
                STAGE_TITLES[self.stage], code, "\n".join(self.tail)))

    def _run_simple(self, argv, on_event=None, **kwargs):
        """Run a stage that may be killed the moment a stop is requested. Returns the exit code."""
        stage = self.stage

        def on_line(line):
            event = rvclog.parse_stage_line(stage, line)
            if not event:
                return
            if event["kind"] == "progress" and event["total"] > 0:
                self._emit(event["done"] / event["total"], STAGE_TITLES[stage])
            if on_event:
                on_event(event)

        self._emit(0.0, STAGE_TITLES[stage])
        code = self._run(argv, on_line, **kwargs)
        self._check_stop()
        return code

    # ---- which stages are finished (decided by the worker, never by a stage's exit code alone)

    def _is_done(self, stage):
        return bool(self.done_stages.get(stage))

    def _saved_epoch(self):
        """The last epoch whose checkpoint this worker saw saved (0 when none was)."""
        record = self.done_stages.get("checkpoint")
        epoch = record.get("epoch") if isinstance(record, dict) else None
        return epoch if isinstance(epoch, int) and not isinstance(epoch, bool) and epoch > 0 else 0

    def _mark(self, stage, done=True):
        self.done_stages[stage] = done
        write_json_atomic(self.exp / "stages.json", self.done_stages)

    # ---- stage 1: prepare

    def _prepare(self):
        self.stage = "prepare"
        args = self.args
        if not SAFE_EXPERIMENT.fullmatch(self.experiment) or self.experiment.lower() == "mute":
            raise JobError("The experiment name %r is not usable: letters, digits, '-' and '_' only."
                           % self.experiment, resumable=False)
        if not str(args.voice_dir).strip() or not self.voice_dir.is_dir():
            raise JobError("The voice folder %r does not exist." % str(args.voice_dir), resumable=False)
        if args.resume:
            # Overall progress carries on from the last saved epoch: it never reads lower than that in this run.
            saved_stages = read_json(self.exp / "stages.json")
            self.done_stages = saved_stages if isinstance(saved_stages, dict) else {}
            if self._saved_epoch() and args.epochs >= 1:
                before_training = sum(STAGE_WEIGHTS[s] for s in STAGES[:STAGES.index("train")])
                self.percent = round(before_training + STAGE_WEIGHTS["train"]
                                     * min(self._saved_epoch(), args.epochs) / args.epochs, 2)
        self._emit(0.0, "Getting the clips ready")
        if min(args.epochs, args.batch_size, args.save_every) < 1:
            raise JobError("Epochs, batch size and the save interval must each be at least 1.", resumable=False)
        if not (self.rvc / "train").is_dir():
            raise JobError("The training environment is not set up (%s is missing). Run: bash trainer/setup.sh"
                           % (self.rvc / "train"))
        self._repair_model_swap()
        clips_dir = self.voice_dir / "clips"
        clips = sorted(p for p in clips_dir.glob("*.orig.wav") if p.is_file()) if clips_dir.is_dir() else []
        if not clips:
            raise JobError("This voice has no recorded clips to train on.", resumable=False)
        # A run with a time limit trains on the clips the app chose for it (written by the app before the start).
        selected = read_json(self.job_dir / SELECTED_CLIPS)
        if isinstance(selected, list):
            names = {str(n) for n in selected}
            clips = [c for c in clips if c.name in names]
            if not clips:
                raise JobError("None of the clips chosen for this run are there any more. Start a new training run.",
                               resumable=False)
        record = {"fingerprint": self._fingerprint(clips), "clips": len(clips), "settings": {
            "batch_size": args.batch_size, "sample_rate": SAMPLE_RATE, "version": "v2", "pitch": True}}
        if args.resume:
            saved = read_json(self.exp / "dataset.json")
            if (not isinstance(saved, dict) or saved.get("fingerprint") != record["fingerprint"]
                    or saved.get("settings") != record["settings"]):
                # Still resumable: nothing was deleted and the saved state is intact, so once the clips are as
                # they were this same run can go on. Until then every resume ends here, changing nothing.
                raise JobError(CLIPS_CHANGED, resumable=True)
        else:
            self.done_stages = {}
            self._remove_training_output()
        for folder in (self.exp, self.rvc / "assets" / "weights", self.rvc / "assets" / "indices"):
            folder.mkdir(parents=True, exist_ok=True)
        self._stage_clips(clips)
        if not args.resume:
            write_json_atomic(self.exp / "dataset.json", record)
        self._emit(1.0, "The clips are ready")

    @staticmethod
    def _fingerprint(clips):
        """SHA-256 over the sorted (name, size, mtime_ns) of the *.orig.wav files. Read from the recordings
        themselves; a staged hard link is the same file, so it has the same size and time."""
        entries = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in clips)
        return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()

    def _stage_clips(self, clips):
        staging = self.exp / "staging"
        _remove_tree(staging)
        staging.mkdir(parents=True)
        for clip in clips:
            target = staging / clip.name
            # A link to a read-only recording would itself be read-only (they are one file), so that one is
            # copied. copyfile copies the bytes only, never the attribute.
            try:
                if not os.access(clip, os.W_OK):
                    raise PermissionError("the recording is read-only")
                os.link(clip, target)
            except OSError:
                shutil.copyfile(clip, target)

    def _index_links(self):
        return (self.rvc / "assets" / "indices").glob("%s_*_%s_v2.index" % (self.experiment, self.experiment))

    def _remove_training_output(self):
        """Everything an earlier run of this experiment left under trainer/rvc (never anything else).

        The records that make a run resumable go first, so an experiment that was only partly removed (a crash, a
        power cut) can never pass for one that can be resumed."""
        for record in ("dataset.json", "stages.json"):
            _unlink(self.exp / record)
        _remove_tree(self.exp)
        _unlink(self.weights)
        for path in self._snapshot_weights().values():
            _unlink(path)
        for link in self._index_links():
            _unlink(link)

    # ---- stages 2 to 4: preprocess, pitch, features

    def _segments(self):
        folder = self.exp / "0_gt_wavs"
        return len([n for n in os.listdir(folder) if n.endswith(".wav")]) if folder.is_dir() else 0

    def _preprocess(self):
        done = self.done_stages.get("preprocess")
        if done:
            expected, found = (done.get("segments") if isinstance(done, dict) else None), self._segments()
            if not isinstance(expected, int) or isinstance(expected, bool):
                raise JobError("The prepared training data cannot be checked: it was made by an older version of "
                               "the app, which did not record how many slices there were. Start a new training "
                               "run.", resumable=False)
            if expected != found:  # never train on a part of the voice without saying so
                raise JobError("The prepared training data is incomplete (%s of %s slices are there). "
                               "Start a new training run." % (found, expected), resumable=False)
            self._emit(1.0, "The clips are already sliced")
            return
        for folder in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
            _remove_tree(self.exp / folder)  # an interrupted run may have left half-written files
        self.done_stages = {}
        failed_files, failed_count = [], [0]

        def on_event(event):
            if event["kind"] == "failed_file":
                failed_files.append(event["path"])
            elif event["kind"] == "completed":
                failed_count[0] += event["failed"]

        exp_dir = self.exp.as_posix()
        code = self._run_simple([self.python, "train/preprocess.py", exp_dir + "/staging", "40000", "8", exp_dir,
                                 "False", "3.7"], on_event)
        failures = max(len(failed_files), failed_count[0])
        if failures:  # preprocess exits 0 even when it could not read a file
            first = (", starting with %s" % re.split(r"[\\/]", failed_files[0])[-1]) if failed_files else ""
            raise JobError("%d clip%s could not be prepared for training%s."
                           % (failures, "" if failures == 1 else "s", first), resumable=False)
        self._require_success(code)
        if not self._segments():
            raise JobError("Slicing the clips produced no audio to train on.", resumable=False)
        self._mark("preprocess", {"segments": self._segments()})
        self._emit(1.0, "The clips are sliced")

    def _extract(self, folders, argv, what):
        stage = self.stage
        if not self._is_done(stage):
            for folder in folders:
                _remove_tree(self.exp / folder)  # the stage skips files that exist, even half-written ones
        failed = [0]

        def on_event(event):
            if event["kind"] == "completed":
                failed[0] += event["failed"]

        code = self._run_simple(argv, on_event)
        if failed[0]:
            raise JobError("%s failed for %d segment%s." % (what, failed[0], "" if failed[0] == 1 else "s"))
        self._require_success(code)
        self._mark(stage)
        self._emit(1.0, STAGE_TITLES[stage])

    def _pitch(self):
        self._extract(("2a_f0", "2b-f0nsf"),
                      [self.python, "train/dataset/extract_f0.py", "cuda", "1", "0", "0", self.exp.as_posix(), "true"],
                      "Finding the pitch")

    def _features(self):
        self._extract(("3_feature768",),
                      [self.python, "train/dataset/extract_hubert_feature.py", "cuda:0", "1", "0", "0",
                       self.exp.as_posix(), "v2", "true"],
                      "Extracting voice features")

    # ---- stage 5: train

    def _model_ready(self):
        try:
            return self.weights.stat().st_size > MIN_MODEL_BYTES and torch_file_intact(self.weights)
        except OSError:
            return False

    def _train(self):
        self.epochs_done = min(self._saved_epoch(), self.args.epochs) if self.args.resume else 0
        if self._model_ready():  # only a resume gets here: a fresh run removed any earlier model
            self.epochs_done = self.args.epochs
            self._emit(1.0, "Training had already finished")
            return
        try:
            listed = filelist.write_filelist(self.rvc, self.experiment, SAMPLE_RATE, 0) - 2  # two silence lines
        except ValueError as exc:
            raise JobError(str(exc)) from None
        segments = self._segments()
        if listed != segments:  # the filelist leaves out a segment that lacks pitch or features, without a word
            missing = segments - listed
            # A few such segments are normal in real recordings: a stretch of breath, laughter or noise has no
            # pitch to find, and the trainer is right to skip it. More than that means something went wrong.
            if missing > segments * MAX_UNUSABLE_SHARE:
                raise JobError("%d of %d segments have no pitch or feature data." % (missing, segments))
            self._log("worker", "%d of %d segments have no pitch or feature data (silence, breath or noise) and "
                                "are left out of training." % (missing, segments))
        saved_state = [p for name in CHECKPOINTS for p in (self.exp / name, self.exp / (name + ".bak")) if p.exists()]
        expect_resume = self.args.resume and (self._is_done("checkpoint") or bool(saved_state))
        self.fp32 = (self.exp / "fp32").is_file()  # an earlier part of this run already had to switch
        self._emit(0.0, "Loading the saved training state" if expect_resume else "Starting training")
        restored = False
        while True:
            try:
                if self._train_once(expect_resume):
                    break
            except _Unstable:
                # Whatever was saved since the numbers went wrong is worthless, and nothing says when that was.
                self._drop_training_state()
                if self.fp32:
                    raise JobError(UNSTABLE_TWICE, resumable=False) from None
                self._check_stop()
                self._log("worker", "a loss turned into NaN: starting training again in full precision")
                (self.exp / "fp32").write_text("")
                self.fp32, expect_resume, restored = True, False, False
                self.epochs_done, self.eta = 0, None
                self.epoch_times.clear()
                self._emit(0.0, UNSTABLE_RETRY)
                continue
            # The trainer could not load the checkpoint pair and was stopped before it trained or saved anything.
            self._check_stop()
            if restored or not self._restore_backup():
                if restored:
                    self._undo_restore()
                raise JobError(STATE_UNREADABLE, resumable=False)
            restored = True
            self._emit(0.0, "The last save was damaged; going back to the one before it")
        if not self._model_ready():
            raise JobError("Training ended but did not write the voice model.")
        self._emit(1.0, "Training finished")

    def _train_once(self, expect_resume):
        """One launch of train.py. True when it ran to the end; False when it failed to load the saved state."""
        args, state = self.args, _Training(expect_resume)

        def on_line(line):
            event = rvclog.parse_train_line(line)
            kind = event["kind"] if event else None
            if kind == "resumed":
                state.resumed_lines += 1
                if state.resume_confirmed:
                    # The epoch the trainer really loaded: lower than the last saved one when that save could
                    # not be read and the backup was used. (The percent never goes down within a run.)
                    self.epochs_done = min(event["epoch"], args.epochs)
                    self._emit(self.epochs_done / args.epochs, "Continuing from epoch %d" % event["epoch"])
                return
            if kind in TRAINING_UNDER_WAY and expect_resume and not state.resume_confirmed:
                state.not_resumed = True  # it fell back to the pretrained base; stop it before it saves anything
                return
            if state.not_resumed:
                return
            if kind == "loss" and any(v != v or v in (float("inf"), float("-inf")) for v in event["values"].values()):
                state.unstable = True  # stopped at once: every epoch after this would be wasted
                return
            if kind == "saved":
                state.saving = True
            elif kind == "epoch_done":
                self._epoch_done(event["epoch"], state)

        def should_stop():
            if state.not_resumed or state.unstable:
                return True
            if not self._poll_stop():
                return False
            if state.stop_seen_at is None:
                state.stop_seen_at, state.grace = self.clock(), self._cancel_grace()
                # The app waits this long (and a little more) before it gives up on the worker and kills it.
                self._emit(self.epochs_done / args.epochs, "Stopping after the current epoch",
                           {"grace_seconds": state.grace})
            # A kill between a "Saving" line and the end of that epoch can cut the only checkpoint short, so wait
            # for the end of an epoch; but not for ever.
            return state.kill_now or self.clock() - state.stop_seen_at >= state.grace

        code = self._run([
            self.python, "train/train.py", "-e", self.experiment, "-sr", SAMPLE_RATE, "-f0", "1",
            "-bs", str(args.batch_size), "-g", "0", "-te", str(args.epochs), "-se", str(args.save_every),
            "-pg", "assets/pretrained_v2/f0G40k.pth", "-pd", "assets/pretrained_v2/f0D40k.pth",
            "-l", "1", "-c", "0", "-sw", "1" if self._snapshots() else "0", "-v", "v2"], on_line, should_stop)
        if state.not_resumed:
            self._log("worker", "the trainer did not load the saved state (%d of 2 checkpoint files loaded)"
                      % state.resumed_lines)
            return False
        if state.unstable:
            raise _Unstable()
        if self._poll_stop() and not (code == 0 and self._model_ready()):
            raise Cancelled()
        self._require_success(code)
        return state.resume_confirmed or not expect_resume

    def _mean_epoch_seconds(self):
        """The mean length of the last ETA_WINDOW epochs, or None until two epochs have ended in this run (the
        first one includes start-up, so it is never averaged)."""
        if len(self.epoch_times) < 2:
            return None
        return (self.epoch_times[-1] - self.epoch_times[0]) / (len(self.epoch_times) - 1)

    def _cancel_grace(self):
        """How long a stop request waits for the current epoch to end before the trainer is killed anyway."""
        mean = self._mean_epoch_seconds()
        if mean is None or self.stop_reason != "cancel":
            # Without the app nobody owns this trainer, so it leaves the GPU within the minute whatever the epochs.
            return float(CANCEL_GRACE_SECONDS)
        return float(min(MAX_CANCEL_GRACE_SECONDS, max(CANCEL_GRACE_SECONDS, 2 * mean)))

    def _snapshots(self):
        """The epochs at which a version of the voice is kept to compare (before the last epoch, a multiple of the
        save interval, since the trainer writes its small weight files only when it saves)."""
        wanted = set()
        for part in str(getattr(self.args, "snapshots", "") or "").split(","):
            part = part.strip()
            if part.isdigit():
                epoch = int(part)
                if 0 < epoch < self.args.epochs and epoch % self.args.save_every == 0:
                    wanted.add(epoch)
        return sorted(wanted)

    def _snapshot_weights(self):
        """{epoch: path} of the small weight files the trainer wrote for this experiment ("<exp>_e100_s5000.pth")."""
        found = {}
        pattern = re.compile(re.escape(self.experiment) + r"_e(\d+)_s\d+\.pth$")
        for path in self.weights.parent.glob(self.experiment + "_e*_s*.pth"):
            match = pattern.match(path.name)
            if match:
                found[int(match.group(1))] = path
        return found

    def _prune_snapshot_weights(self):
        """The trainer writes a weight file at every save; only the ones for the chosen epochs are kept."""
        keep = set(self._snapshots())
        for epoch, path in self._snapshot_weights().items():
            if epoch not in keep:
                _unlink(path)

    def _epoch_done(self, epoch, state):
        saved, state.saving = state.saving or epoch % self.args.save_every == 0, False
        if saved:
            self._prune_snapshot_weights()
        self.epoch_times.append(self.clock())
        self.epochs_done = max(self.epochs_done, min(epoch, self.args.epochs))
        mean = self._mean_epoch_seconds()
        if mean is not None:
            self.eta = round(mean * (self.args.epochs - self.epochs_done), 1)
        self._emit(self.epochs_done / self.args.epochs, "Epoch %d of %d" % (self.epochs_done, self.args.epochs))
        if saved:
            # Recorded as soon as a checkpoint is known to be on disk, whether or not a backup follows: a later
            # resume that finds no checkpoint must fail rather than train from the start in silence.
            try:
                self._mark("checkpoint", {"epoch": epoch})
            except OSError as exc:
                self._log("worker", "could not record the saved epoch: %s" % exc)
        if epoch >= self.args.epochs:
            return  # the trainer now writes the final model and ends by itself: neither kill it nor copy
        if self._poll_stop():
            state.kill_now = True
        elif saved:
            self._back_up_checkpoints(epoch)

    def _back_up_checkpoints(self, epoch):
        """Copy the checkpoint pair to .bak. Called only right after an epoch ended, when no save is running.

        The copy still proves itself before it replaces the previous backup: the source must not have changed
        while it was read, and the copy must be a complete torch file. A bad copy is dropped and the previous
        backup stays; training carries on either way."""
        temps, started, copied = [], time.monotonic(), {}
        try:
            for name in CHECKPOINTS:
                source, temp = self.exp / name, self.exp / (name + ".bak.tmp")
                temps.append(temp)
                before = copied[name] = _file_state(source)
                copy_durable(source, temp)
                if _file_state(source) != before or temp.stat().st_size != before[0] or not torch_file_intact(temp):
                    raise OSError("%s changed while it was copied, or is incomplete" % name)
            for name in CHECKPOINTS:
                # The trainer may have begun its next save while the second file was copied. Both copies must
                # still match what is on disk now, or the pair could be from two different epochs.
                if _file_state(self.exp / name) != copied[name]:
                    raise OSError("%s was saved again before the pair could be kept" % name)
            for name in CHECKPOINTS:  # both copies are good: swap the pair in together
                os.replace(self.exp / (name + ".bak.tmp"), self.exp / (name + ".bak"))
            for name in CHECKPOINTS:
                _unlink(self.exp / (name + ".failed"))
            self.backups_skipped = 0
            self._log("worker", "checkpoint of epoch %d backed up in %.1f s" % (epoch, time.monotonic() - started))
        except OSError as exc:
            self._log("worker", "checkpoint backup skipped, the previous backup is kept: %s" % exc)
            for temp in temps:
                _unlink_quietly(temp)
            self.backups_skipped += 1
            if self.backups_skipped % BACKUP_SKIPS_WORTH_A_WARNING == 0:
                self._log("worker", "WARNING: %d checkpoint backups in a row were skipped. If the trainer is stopped "
                                    "during a save now, the run can only go back to the last backup that was made, "
                                    "or cannot be resumed if there is none." % self.backups_skipped)

    def _drop_training_state(self):
        """Remove every saved checkpoint, its backups and any model written from them, and forget that a
        checkpoint was saved, so that the next launch of the trainer starts from the pretrained base."""
        for name in CHECKPOINTS:
            for suffix in ("", ".bak", ".failed"):
                _unlink(self.exp / (name + suffix))
        _unlink(self.weights)
        if "checkpoint" in self.done_stages:
            del self.done_stages["checkpoint"]
            write_json_atomic(self.exp / "stages.json", self.done_stages)

    def _restore_backup(self):
        """Put the .bak pair in place of the checkpoint pair, keeping the unreadable files as .failed.
        Returns False, having changed nothing, when there is no complete backup."""
        if not all((self.exp / (name + ".bak")).is_file() for name in CHECKPOINTS):
            self._log("worker", "there is no checkpoint backup to fall back to")
            return False
        temps = [self.exp / (name + ".restore") for name in CHECKPOINTS]
        try:
            for name, temp in zip(CHECKPOINTS, temps):
                copy_durable(self.exp / (name + ".bak"), temp)
        except OSError as exc:
            self._log("worker", "could not copy the checkpoint backup: %s" % exc)
            for temp in temps:
                _unlink_quietly(temp)
            return False
        for name, temp in zip(CHECKPOINTS, temps):
            if (self.exp / name).exists():
                os.replace(self.exp / name, self.exp / (name + ".failed"))
            os.replace(temp, self.exp / name)
        self._log("worker", "restored the checkpoint backup; the unreadable files are kept as *.failed")
        return True

    def _undo_restore(self):
        """The backup did not load either: put the original checkpoint files back, so nothing has changed."""
        for name in CHECKPOINTS:
            if (self.exp / (name + ".failed")).exists():
                os.replace(self.exp / (name + ".failed"), self.exp / name)

    # ---- stage 6: index

    def _added_index(self):
        found = sorted(self.exp.glob("added_*.index"))
        return found[0] if found else None

    def _index(self):
        if self._is_done("index") and self._added_index():
            self._emit(1.0, "The search index is already built")
            return
        for stale in [*self.exp.glob("added_*.index"), *self.exp.glob("trained_*.index"), self.exp / "total_fea.npy",
                      *self._index_links()]:
            _unlink(stale)  # train_index.py skips its work when an index exists, even a half-written one
        code = self._run_simple([self.python, "train/train_index.py", self.experiment, "v2", "assets/indices", "16",
                                 "single"])
        self._require_success(code)
        if not self._added_index():
            raise JobError("Building the search index left no index file.")
        self._mark("index")
        self._emit(1.0, "The search index is built")

    # ---- stage 7: export

    def _export(self):
        models = self.voice_dir / "models"
        new = models / "rvc.new"
        _remove_tree(new)
        new.mkdir(parents=True)
        env = dict(os.environ)
        env.pop("__PYVENV_LAUNCHER__", None)  # the exporter has its own environment
        env.update({"AIVOICE_EXPORT_DIR": str(new), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1"})
        try:
            code = self._run_simple([
                TOOLS_DIR / ".rvc" / "venv" / "Scripts" / "python.exe", TOOLS_DIR / "export_rvc_onnx.py",
                self.weights, self.experiment, "--index", self._added_index(),
                # One argument: as two, a name such as "-x" or "--cool" would be read as an option.
                "--name=" + (self.args.name or self.experiment)], cwd=TOOLS_DIR.parent, env=env)
            self._require_success(code)
            self._check_export(new)
            try:
                self._swap_model_in(models, new)
            except OSError as exc:
                raise JobError("The new model could not be put in place of the current one (%s). The current model "
                               "is unchanged; resume this training run to try again." % exc) from None
        except BaseException:
            shutil.rmtree(new, ignore_errors=True)  # whatever went wrong, the model in use was not touched
            raise
        self._export_snapshots(models)
        self._emit(1.0, "The model is exported")

    def _export_snapshots(self, models):
        """The versions of the voice kept at earlier epochs, each in models/rvc-e<epoch>/ beside the final one, for the
        person to compare by ear. They share the final model's search index. A version that cannot be exported is
        left out with a line in the log: the final model is what the run is for."""
        for stale in models.glob("rvc-e*"):  # versions from an earlier run would be mistaken for this run's
            if stale.is_dir() and not self._move_to_trash(stale):
                _remove_tree(stale)
        weights = self._snapshot_weights()
        for epoch in self._snapshots():
            path = weights.get(epoch)
            if path is None:
                self._log("worker", "no weights were saved at epoch %d, so that version is left out" % epoch)
                continue
            target, new = models / ("rvc-e%d" % epoch), models / ("rvc-e%d.new" % epoch)
            _remove_tree(new)
            new.mkdir(parents=True)
            env = dict(os.environ)
            env.pop("__PYVENV_LAUNCHER__", None)
            env.update({"AIVOICE_EXPORT_DIR": str(new), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                        "PYTHONUNBUFFERED": "1"})
            self._emit(1.0, "Exporting the version at epoch %d" % epoch)
            try:
                code = self._run_simple([
                    TOOLS_DIR / ".rvc" / "venv" / "Scripts" / "python.exe", TOOLS_DIR / "export_rvc_onnx.py",
                    path, self.experiment, "--name=%s (epoch %d)" % (self.args.name or self.experiment, epoch)],
                    cwd=TOOLS_DIR.parent, env=env)
                size = (new / "model.onnx").stat().st_size if (new / "model.onnx").exists() else 0
                info = read_json(new / "voice.json")
                if code != 0 or size <= MIN_MODEL_BYTES or not isinstance(info, dict):
                    raise OSError("the export did not produce a usable model (exit code %s)" % code)
                os.replace(new, target)
                self._log("worker", "kept the version at epoch %d in %s" % (epoch, target))
            except OSError as exc:
                shutil.rmtree(new, ignore_errors=True)
                self._log("worker", "the version at epoch %d could not be exported: %s" % (epoch, exc))
            finally:
                _unlink(path)

    @staticmethod
    def _check_export(folder):
        try:
            size = (folder / "model.onnx").stat().st_size
        except OSError:
            size = 0
        if size <= MIN_MODEL_BYTES:
            raise JobError("The export did not produce a usable model file.")
        if not (folder / "model.index").is_file():
            raise JobError("The export did not include the search index.")
        info = read_json(folder / "voice.json")
        if not isinstance(info, dict) or not isinstance(info.get("sample_rate"), int):
            raise JobError("The export did not write a readable voice.json.")

    def _swap_model_in(self, models, new):
        live, old = models / "rvc", models / "rvc.old"
        if os.path.lexists(old):  # left by an earlier swap: out of the way, to the trash if there is one
            if not self._move_to_trash(old):
                _remove_tree(old)  # it has to make room: the model in use is about to be parked under this name
                self._log("worker", "the older model left in %s was deleted" % old)
        if live.exists():
            os.replace(live, old)
        try:
            os.replace(new, live)
        except OSError:
            if old.exists():
                os.replace(old, live)
            raise
        if old.exists():
            self._retire_previous_model(old)

    def _trash_dir(self):
        """<data>/trash/<voice-id>, where the app keeps what it takes out of a voice; None when the voice folder
        is not <data>/voices/<id> (then there is no data folder to hold a trash)."""
        voice = self.voice_dir.resolve()
        if voice.parent.name != "voices" or not voice.name:
            return None
        return voice.parent.parent / "trash" / voice.name

    def _move_to_trash(self, folder):
        """Move a model folder to the trash under a name of its own (a rename on the same drive, never a copy).
        False, with the folder untouched, when there is no trash or the move failed."""
        trash = self._trash_dir()
        if trash is None:
            return False
        try:
            trash.mkdir(parents=True, exist_ok=True)
            stamp = "models-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.clock()))
            target, n = trash / stamp, 1
            while os.path.lexists(target):
                n += 1
                target = trash / ("%s-%d" % (stamp, n))
            os.rename(folder, target)
        except (OSError, OverflowError, ValueError) as exc:
            self._log("worker", "could not move %s to the trash: %s" % (folder, exc))
            return False
        self._log("worker", "the previous model was moved to %s" % target)
        return True

    def _retire_previous_model(self, old):
        """The new model is in place. The one it replaced goes to the trash; it is erased only where there is no
        data folder around the voice, and stays beside the new one when the trash cannot be used."""
        if self._move_to_trash(old):
            return
        if self._trash_dir() is None:
            shutil.rmtree(old, ignore_errors=True)
            self._log("worker", "the previous model was deleted: the voice folder is not inside a data folder, "
                                "so there is no trash to move it to")
        else:
            self._log("worker", "the previous model stays in %s" % old)

    def _repair_model_swap(self):
        """A swap cut off between its two renames leaves the model in use parked as rvc.old: put it back."""
        models = self.voice_dir / "models"
        live, old = models / "rvc", models / "rvc.old"
        if old.is_dir() and not live.exists():
            try:
                self._check_export(old)
            except JobError as exc:
                self._log("worker", "models/rvc.old was left by an interrupted export but is not a complete model "
                                    "(%s); it stays where it is and the voice has no model until training ends"
                          % exc)
                return
            os.replace(old, live)
            self._log("worker", "put back the previous model after an interrupted export")

    # ---- after a success

    def _clean_up(self):
        try:
            self._remove_training_output()
        except OSError as exc:  # the model is exported; left-over training files are not a failure
            self._log("worker", "could not remove the training files: %s" % exc)


def _unlink_quietly(path):
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------- entry points

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train one RVC voice model and export it for the app.")
    parser.add_argument("--voice-dir", required=True, help="the voice's folder (data/voices/<id>)")
    parser.add_argument("--job-dir", required=True, help="where progress.jsonl, worker.log and cancel live")
    parser.add_argument("--experiment", required=True, help="the RVC experiment name, voice-<id>")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshots", default="",
                        help="epochs at which to keep an extra version of the voice to compare, e.g. 100,150")
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--name", default=None, help="the display name written to voice.json")
    return parser.parse_args(argv)


def run(args, runner=None, clock=time.time, kill_tree=None, parent_alive=None, notes=()) -> int:
    """Run the whole job. `runner(stage, argv, cwd, env, on_line, should_stop) -> exit code` runs one stage and must
    end the stage's process tree when should_stop() returns True; `kill_tree(pid)` is how the default runner does
    that; `parent_alive(pid)` answers whether the app is still there; `notes` are lines for the top of worker.log."""
    if runner is None:
        runner = make_runner(kill_tree or kill_process_tree)
    return Job(args, runner, clock, parent_alive or process_alive, notes).execute()


def main(argv=None) -> int:
    args = parse_args(argv)
    notes = []
    if os.name == "nt" and not bind_children_to_this_process():
        notes.append("the job object could not be set up: if this worker is killed outright, the stage it is "
                     "running is not ended with it")
    return run(args, notes=notes)


if __name__ == "__main__":
    sys.exit(main())
