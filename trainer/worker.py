"""The training worker: trains one RVC v2 voice model and hands it to the app.

  python -P trainer/worker.py --voice-dir <data/voices/id> --job-dir <data/voices/id/training>
      --experiment voice-<id> --epochs 200 --batch-size 8 --save-every 5 [--resume] [--parent-pid N] --name "<name>"

Runs under the training environment's Python but uses only the standard library plus rvclog.py and filelist.py
from this folder. It runs RVC's command-line stages one after another as subprocesses (commands and environment
are those proven in docs/superpowers/notes/2026-10-13-stage-3-training-spike.md, sections 3 and 4), appends one
JSON line per progress step to <job-dir>/progress.jsonl, copies every line of stage output to <job-dir>/worker.log,
and always ends with a final progress line. Exit code: 0 success, 1 failure, 2 cancelled.

What it protects:
- The recordings. Only <voice-dir>/clips/*.orig.wav are read, through hard links (or copies) in the experiment's
  own staging folder. Nothing under clips/ is ever written, renamed or deleted, and no file attribute is changed.
- The training state. RVC overwrites its only checkpoint pair in place and silently restarts from the pretrained
  base when it cannot read it. The worker keeps a verified second copy (.bak), stops a trainer that did not print
  its two "Loaded checkpoint" lines before it can save anything, and falls back to the copy.
- The model in use. The export goes to models/rvc.new/, is checked, and only then swapped into models/rvc/.
- The GPU. Stages are killed as a whole process tree, and on Windows every child is tied to the worker's own life.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
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

STAGES = ("prepare", "preprocess", "pitch", "features", "train", "index", "export")
STAGE_WEIGHTS = {"prepare": 1, "preprocess": 2, "pitch": 4, "features": 6, "train": 84, "index": 1, "export": 2}
CHECKPOINTS = ("G_2333333.pth", "D_2333333.pth")  # the fixed names train.py uses with "-l 1"
SAMPLE_RATE = "40k"
# Lines that show train.py is past loading and is training. The loss line is among them because it is one RVC
# never translates, so a trainer that did not resume is caught on any Windows display language.
TRAINING_UNDER_WAY = ("epoch_progress", "loss", "saved", "epoch_done")
CANCEL_GRACE_SECONDS = 60
ETA_WINDOW = 10
TAIL_LINES = 20
MIN_MODEL_BYTES = 1024 * 1024
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SAFE_EXPERIMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

CLIPS_CHANGED = "The clips changed since this training run started. Start a new training run."
STATE_UNREADABLE = "The saved training state could not be read"
STAGE_TITLES = {
    "preprocess": "Slicing the clips",
    "pitch": "Finding the pitch of each slice",
    "features": "Extracting voice features",
    "train": "Training",
    "index": "Building the search index",
    "export": "Exporting the model",
}


class JobError(Exception):
    """A failure with a message meant for the user. `resumable` says whether running again with --resume could
    get further than this run did (False when only a new training run can help)."""

    def __init__(self, message, resumable=True):
        super().__init__(message)
        self.resumable = resumable


class Cancelled(Exception):
    """The run was stopped on request (cancel file) or because the app went away."""


# ---------------------------------------------------------------- processes (standard library only)

_kernel32_dll = None


def _kernel32():
    global _kernel32_dll
    if _kernel32_dll is not None:
        return _kernel32_dll
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, dword, void_p = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p
    kernel32.OpenProcess.restype, kernel32.OpenProcess.argtypes = handle, [dword, ctypes.c_int, dword]
    kernel32.WaitForSingleObject.restype, kernel32.WaitForSingleObject.argtypes = dword, [handle, dword]
    kernel32.GetCurrentProcess.restype, kernel32.GetCurrentProcess.argtypes = handle, []
    kernel32.CreateJobObjectW.restype, kernel32.CreateJobObjectW.argtypes = handle, [void_p, ctypes.c_wchar_p]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.SetInformationJobObject.argtypes = [handle, ctypes.c_int, void_p, dword]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [handle, handle]
    _kernel32_dll = kernel32
    return kernel32


_process_handles = {}  # pid -> open handle, kept so that a recycled PID is never mistaken for the original process


def process_alive(pid):
    """True while the process with this PID is running."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    synchronize, wait_timeout, access_denied = 0x00100000, 0x00000102, 5
    kernel32 = _kernel32()
    handle = _process_handles.get(pid)
    if handle is None:
        handle = kernel32.OpenProcess(synchronize, 0, int(pid))
        if not handle:
            return ctypes.get_last_error() == access_denied  # it exists but belongs to someone else
        _process_handles[pid] = handle
    return kernel32.WaitForSingleObject(handle, 0) == wait_timeout


def kill_process_tree(pid):
    """End a process and everything it started. The PID a venv launcher gives back is only a shim around the real
    Python, so ending that one process would leave the trainer and its data-loader workers running."""
    if os.name != "nt":
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        return
    try:
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass  # the caller still ends the process it started itself


_lifetime_job = None


def bind_children_to_this_process():
    """Windows: put this process in a job object that ends every process in it when its last handle closes.

    The handle is held only by this process, so if the worker dies in any way at all (even killed outright, with
    no chance to clean up) Windows ends every stage it started. Returns True when the job is in place."""
    global _lifetime_job
    if os.name != "nt":
        return False
    if _lifetime_job is not None:
        return True

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", ctypes.c_ulong), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_ulong),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_ulong),
                    ("SchedulingClass", ctypes.c_ulong)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kill_on_job_close, extended_limit_information = 0x2000, 9
    try:
        kernel32 = _kernel32()
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = kill_on_job_close
        if not kernel32.SetInformationJobObject(job, extended_limit_information, ctypes.byref(limits),
                                                ctypes.sizeof(limits)):
            return False
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            return False
    except (OSError, AttributeError):
        return False
    _lifetime_job = job  # never closed on purpose: it closes when this process ends
    return True


def make_runner(kill_tree=kill_process_tree, popen=subprocess.Popen, poll_seconds=0.25, drain_seconds=30.0):
    """The real stage runner: start the command, pass each output line to on_line, and end the whole process tree
    as soon as should_stop() says so (it is asked after every line and every poll_seconds while the stage is quiet)."""

    def runner(stage, argv, cwd, env, on_line, should_stop):
        process = popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                        creationflags=NO_WINDOW)
        lines = queue.Queue()

        def pump():
            try:
                for line in process.stdout:
                    lines.put(line)
            except (OSError, ValueError):
                pass
            finally:
                lines.put(None)

        reader = threading.Thread(target=pump, name="stage-output", daemon=True)
        reader.start()
        killed, give_up_at = False, None
        try:
            while True:
                try:
                    line = lines.get(timeout=poll_seconds)
                except queue.Empty:
                    line = ""
                if line is None:
                    break  # the output pipe closed: every process that held it has ended
                if line:
                    on_line(line.rstrip("\r\n"))
                if not killed and should_stop():
                    kill_tree(process.pid)
                    killed = True
                if give_up_at is None and (killed or (not line and process.poll() is not None)):
                    give_up_at = time.monotonic() + drain_seconds  # ended or killed, yet the pipe is still open
                elif give_up_at is not None and time.monotonic() > give_up_at:
                    break  # something still holds the pipe; do not wait for it for ever
            if not killed:
                return process.wait()
            if process.poll() is None:
                process.kill()
            try:
                return process.wait(timeout=drain_seconds)
            except subprocess.TimeoutExpired:
                return 1
        finally:
            if process.poll() is None:  # only reached when something above raised
                kill_tree(process.pid)
                try:
                    process.kill()
                    process.wait(timeout=drain_seconds)
                except (OSError, subprocess.SubprocessError):
                    pass
            reader.join(timeout=1.0)
            if not reader.is_alive():  # closing under a reader that is still blocked could hang
                try:
                    process.stdout.close()
                except (OSError, AttributeError):
                    pass

    return runner


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
        shutil.rmtree(path)


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
        self.stop_seen_at = None

    @property
    def resume_confirmed(self):
        return self.resumed_lines >= 2


class Job:
    def __init__(self, args, runner, clock, parent_alive):
        self.args, self.runner, self.clock, self.parent_alive = args, runner, clock, parent_alive
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
        self.stop_reason = None
        self.done_stages = {}
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
        self._log_file = open(self.job_dir / "worker.log", mode, encoding="utf-8", errors="replace", newline="\n")
        self._progress = open(self.job_dir / "progress.jsonl", mode, encoding="utf-8", newline="\n")

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

    def _mark(self, stage, done=True):
        self.done_stages[stage] = done
        write_json_atomic(self.exp / "stages.json", self.done_stages)

    # ---- stage 1: prepare

    def _prepare(self):
        self.stage = "prepare"
        self._emit(0.0, "Getting the clips ready")
        args = self.args
        if not SAFE_EXPERIMENT.fullmatch(self.experiment) or self.experiment.lower() == "mute":
            raise JobError("The experiment name %r is not usable: letters, digits, '-' and '_' only."
                           % self.experiment, resumable=False)
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
        record = {"fingerprint": self._fingerprint(clips), "clips": len(clips), "settings": {
            "batch_size": args.batch_size, "sample_rate": SAMPLE_RATE, "version": "v2", "pitch": True}}
        if args.resume:
            saved = read_json(self.exp / "dataset.json")
            if (not isinstance(saved, dict) or saved.get("fingerprint") != record["fingerprint"]
                    or saved.get("settings") != record["settings"]):
                raise JobError(CLIPS_CHANGED, resumable=False)
            saved_stages = read_json(self.exp / "stages.json")
            self.done_stages = saved_stages if isinstance(saved_stages, dict) else {}
        else:
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
            try:
                os.link(clip, target)
            except OSError:
                shutil.copy2(clip, target)

    def _index_links(self):
        return (self.rvc / "assets" / "indices").glob("%s_*_%s_v2.index" % (self.experiment, self.experiment))

    def _remove_training_output(self):
        """Everything an earlier run of this experiment left under trainer/rvc (never anything else)."""
        _remove_tree(self.exp)
        _unlink(self.weights)
        for link in self._index_links():
            _unlink(link)

    # ---- stages 2 to 4: preprocess, pitch, features

    def _segments(self):
        folder = self.exp / "0_gt_wavs"
        return len([n for n in os.listdir(folder) if n.endswith(".wav")]) if folder.is_dir() else 0

    def _preprocess(self):
        if self._is_done("preprocess") and self._segments():
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
        self._mark("preprocess")
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
        self.epochs_done = 0
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
            raise JobError("%d of %d segments have no pitch or feature data." % (segments - listed, segments))
        saved_state = [p for name in CHECKPOINTS for p in (self.exp / name, self.exp / (name + ".bak")) if p.exists()]
        expect_resume = self.args.resume and (self._is_done("checkpoint") or bool(saved_state))
        self._emit(0.0, "Loading the saved training state" if expect_resume else "Starting training")
        restored = False
        while True:
            if self._train_once(expect_resume):
                break
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
                    self.epochs_done = max(self.epochs_done, min(event["epoch"], args.epochs))
                    self._emit(self.epochs_done / args.epochs, "Continuing from epoch %d" % event["epoch"])
                return
            if kind in TRAINING_UNDER_WAY and expect_resume and not state.resume_confirmed:
                state.not_resumed = True  # it fell back to the pretrained base; stop it before it saves anything
                return
            if state.not_resumed:
                return
            if kind == "saved":
                state.saving = True
            elif kind == "epoch_done":
                self._epoch_done(event["epoch"], state)

        def should_stop():
            if state.not_resumed:
                return True
            if not self._poll_stop():
                return False
            if state.stop_seen_at is None:
                state.stop_seen_at = self.clock()
                self._emit(self.epochs_done / args.epochs, "Stopping after the current epoch")
            # A kill between a "Saving" line and the end of that epoch can cut the only checkpoint short, so wait
            # for the end of an epoch; but not for ever.
            return state.kill_now or self.clock() - state.stop_seen_at >= CANCEL_GRACE_SECONDS

        code = self._run([
            self.python, "train/train.py", "-e", self.experiment, "-sr", SAMPLE_RATE, "-f0", "1",
            "-bs", str(args.batch_size), "-g", "0", "-te", str(args.epochs), "-se", str(args.save_every),
            "-pg", "assets/pretrained_v2/f0G40k.pth", "-pd", "assets/pretrained_v2/f0D40k.pth",
            "-l", "1", "-c", "0", "-sw", "0", "-v", "v2"], on_line, should_stop)
        if state.not_resumed:
            self._log("worker", "the trainer did not load the saved state (%d of 2 checkpoint files loaded)"
                      % state.resumed_lines)
            return False
        if self._poll_stop() and not (code == 0 and self._model_ready()):
            raise Cancelled()
        self._require_success(code)
        return state.resume_confirmed or not expect_resume

    def _epoch_done(self, epoch, state):
        saved, state.saving = state.saving or epoch % self.args.save_every == 0, False
        self.epoch_times.append(self.clock())
        self.epochs_done = max(self.epochs_done, min(epoch, self.args.epochs))
        if len(self.epoch_times) >= 2:  # the first epoch of a run includes start-up, so it is never averaged
            mean = (self.epoch_times[-1] - self.epoch_times[0]) / (len(self.epoch_times) - 1)
            self.eta = round(mean * (self.args.epochs - self.epochs_done), 1)
        self._emit(self.epochs_done / self.args.epochs, "Epoch %d of %d" % (self.epochs_done, self.args.epochs))
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
        temps, started = [], time.monotonic()
        try:
            for name in CHECKPOINTS:
                source, temp = self.exp / name, self.exp / (name + ".bak.tmp")
                temps.append(temp)
                before = _file_state(source)
                copy_durable(source, temp)
                if _file_state(source) != before or temp.stat().st_size != before[0] or not torch_file_intact(temp):
                    raise OSError("%s changed while it was copied, or is incomplete" % name)
            for name in CHECKPOINTS:  # both copies are good: swap the pair in together
                os.replace(self.exp / (name + ".bak.tmp"), self.exp / (name + ".bak"))
            if not self._is_done("checkpoint"):
                self._mark("checkpoint")
            for name in CHECKPOINTS:
                _unlink(self.exp / (name + ".failed"))
            self._log("worker", "checkpoint of epoch %d backed up in %.1f s" % (epoch, time.monotonic() - started))
        except OSError as exc:
            self._log("worker", "checkpoint backup skipped, the previous backup is kept: %s" % exc)
            for temp in temps:
                _unlink_quietly(temp)

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
                "--name", self.args.name or self.experiment], cwd=TOOLS_DIR.parent, env=env)
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
        self._emit(1.0, "The model is exported")

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

    @staticmethod
    def _swap_model_in(models, new):
        live, old = models / "rvc", models / "rvc.old"
        _remove_tree(old)
        if live.exists():
            os.replace(live, old)
        try:
            os.replace(new, live)
        except OSError:
            if old.exists():
                os.replace(old, live)
            raise
        shutil.rmtree(old, ignore_errors=True)

    def _repair_model_swap(self):
        """A swap cut off between its two renames leaves the model in use parked as rvc.old: put it back."""
        models = self.voice_dir / "models"
        live, old = models / "rvc", models / "rvc.old"
        if old.is_dir() and not live.exists():
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
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--name", default=None, help="the display name written to voice.json")
    return parser.parse_args(argv)


def run(args, runner=None, clock=time.time, kill_tree=None, parent_alive=None) -> int:
    """Run the whole job. `runner(stage, argv, cwd, env, on_line, should_stop) -> exit code` runs one stage and must
    end the stage's process tree when should_stop() returns True; `kill_tree(pid)` is how the default runner does
    that; `parent_alive(pid)` answers whether the app is still there."""
    if runner is None:
        runner = make_runner(kill_tree or kill_process_tree)
    return Job(args, runner, clock, parent_alive or process_alive).execute()


def main(argv=None) -> int:
    args = parse_args(argv)
    bind_children_to_this_process()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
