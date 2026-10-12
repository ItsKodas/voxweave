"""Runs voice training: starts trainer/worker.py as a separate process, follows its progress file, and offers
start, cancel, resume and status. One job at a time.

Nothing from trainer/ is ever imported here (the backend has no torch): the worker and the environment check
are only ever run as subprocesses.

A job is only ever called stopped once its processes are known to be gone. Every kill is checked against a
list of the worker's processes taken beforehand, runs outside the manager's lock, and is tried again later if
anything survived.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from aivoice.library import ID_RE, Library, LibraryError, atomic_json

log = logging.getLogger(__name__)

MIN_MINUTES = 5.0
MIN_EPOCHS, MAX_EPOCHS = 20, 1000
BATCH_SIZES = (4, 8, 16)
DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE = 200, 8
SAVE_EVERY = 5
STAGES = 7
CANCEL_TIMEOUT = 90.0  # the worker gives the trainer 60 seconds to reach a safe moment; after this we stop waiting
EXIT_GRACE = 30.0  # how long a worker may linger after writing its final progress line
KILL_RETRY = 5.0  # a kill that left something running is tried again no sooner than this
TREE_REFRESH = 10.0  # how often the list of the worker's processes is renewed while it runs
TASKKILL_GRACE = 3.0  # seconds given to the processes to disappear after a successful taskkill
KILL_WAIT = 5.0  # and after killing the stragglers one by one
TAIL_BYTES = 65536
CHECKPOINTS = ("G_2333333.pth", "D_2333333.pth")  # RVC's fixed names with -l 1; the worker keeps a .bak pair
# The worker's stdout and stderr go here. worker.log is written by the worker alone: two writers corrupt it.
OUTPUT_LOG = "worker.out.log"

ACTIVE = ("running", "cancelling")
RESUMABLE = ("failed", "cancelled", "interrupted")
FINISHED = ("done",) + RESUMABLE
STOPPED_UNEXPECTEDLY = ("The training process stopped unexpectedly. "
                        "See training/worker.log and training/worker.out.log.")
NO_RESUME = "There is no interrupted training run to resume for this voice"
CANCELLED_BEFORE_START = "Training was cancelled before it started"
# Whether a failed run is worth resuming is the worker's call ("resumable" in its final line). These are its
# messages for a resume that cannot work, used only when a final line carries no such field.
NEVER_RESUMABLE = ("The clips changed since this training run started", "The saved training state could not be read")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


class TrainingError(ValueError):
    pass


class KillFailed(RuntimeError):
    """Processes that should have been stopped are still running."""


def _reason(exc: BaseException) -> str:
    return getattr(exc, "strerror", None) or str(exc) or type(exc).__name__


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pids(procs) -> str:
    return ", ".join(str(p.pid) for p in procs)


class ProcessTools:
    """Looking at and ending real processes. Injectable, so tests never touch one."""

    def tree(self, pid: int) -> list:
        """The process and everything it started, as psutil processes (each remembers its start time, so a
        pid that is reused later is never mistaken for it). Empty if the process is gone."""
        try:
            root = psutil.Process(pid)
        except Exception:  # noqa: BLE001 - gone, access denied, not a pid
            return []
        try:
            return [root] + root.children(recursive=True)
        except Exception:  # noqa: BLE001
            return [root]

    def taskkill(self, pid: int) -> int:
        """Ends a process tree; returns taskkill's exit code. The pid of a venv Python is only a launcher shim:
        the real interpreter and the trainer's own child processes are its descendants."""
        return subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=30, creationflags=_NO_WINDOW).returncode

    def kill(self, proc) -> None:
        proc.kill()

    def survivors(self, procs: list, timeout: float) -> list:
        """Those still running after waiting at most timeout seconds for them to end."""
        if not procs:
            return []
        alive = psutil.wait_procs(procs, timeout=timeout)[1]
        return [p for p in alive if p.is_running()]


def _end_survivors(procs: list, tools, grace: float = 0.0) -> list:
    """Kills, one by one, those of procs still running after grace seconds. Returns the ones it had to kill;
    raises KillFailed if any is still running afterwards."""
    alive = tools.survivors(procs, grace) if procs else []
    if not alive:
        return []
    for proc in alive:
        try:
            tools.kill(proc)
        except Exception:  # noqa: BLE001 - gone in the meantime, or refused: the check below decides
            pass
    left = tools.survivors(alive, KILL_WAIT)
    if left:
        raise KillFailed(f"processes still running: {_pids(left)}")
    return alive


def _kill_tree(pid: int, tools) -> None:
    """Ends a process and everything it started, and makes sure of it. Raises KillFailed otherwise."""
    # First the list: once a parent is gone, its children can no longer be found through it.
    snapshot = tools.tree(pid)
    try:
        code = tools.taskkill(pid)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("taskkill could not be run for process %s: %s", pid, _reason(exc))
        code = None
    killed = _end_survivors(snapshot, tools, TASKKILL_GRACE if code == 0 else 0.0)
    if killed:
        log.warning("taskkill (exit code %s) left processes of %s running; ended them directly: %s",
                    code, pid, _pids(killed))


class _Worker:
    """The running worker process, as the manager sees it."""

    def __init__(self, proc, tools):
        self._proc = proc
        self._tools = tools
        self.pid = proc.pid

    def poll(self):
        return self._proc.poll()

    def kill_tree(self) -> None:
        """Raises KillFailed if any process of the worker is still running afterwards."""
        _kill_tree(self.pid, self._tools)
        self._proc.poll()  # collect the exit code


def _launch_worker(argv: list[str], log_path: Path, tools=None) -> _Worker:
    # Its own hidden console and process group: no window appears, and closing the backend's console (or a
    # Ctrl+C in it) does not reach the worker. It ends itself when the backend's pid disappears.
    with open(log_path, "ab") as out:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                creationflags=_NO_WINDOW | _NEW_GROUP)
    return _Worker(proc, tools or ProcessTools())


def _pid_alive(pid) -> bool:
    try:
        return _whole(pid) and pid > 0 and psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001 - an answer is needed either way; "not alive" never leads to a kill
        return False


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def _is_worker_command(argv: list[str], worker: Path, experiment: str) -> bool:
    """Whether a process's argument list is our worker script running that experiment."""
    if not any(_same_path(arg, str(worker)) for arg in argv):
        return False
    return any(flag == "--experiment" and value == experiment for flag, value in zip(argv, argv[1:]))


def _is_worker_process(pid: int, experiment: str, worker: Path) -> bool:
    """False whenever it cannot be shown to be one: a pid may have been reused by an unrelated program."""
    try:
        return _is_worker_command(psutil.Process(pid).cmdline(), worker, experiment)
    except Exception:  # noqa: BLE001 - gone, access denied, anything: not confirmed
        return False


def _failed_report(message: str) -> dict:
    return {"ok": False, "checks": [{"id": "verify", "ok": False, "message": message}]}


def _clean_report(raw) -> dict:
    """The verify report as {"ok", "checks"}, or a failing one if it does not have that shape."""
    if (isinstance(raw, dict) and isinstance(raw.get("ok"), bool) and isinstance(raw.get("checks"), list)
            and all(isinstance(c, dict) and isinstance(c.get("ok"), bool) and isinstance(c.get("message"), str)
                    for c in raw["checks"])):
        return {"ok": raw["ok"] and all(c["ok"] for c in raw["checks"]), "checks": raw["checks"]}
    return _failed_report("The training environment check gave no usable result")


def _idle(voice_id=None) -> dict:
    return {"state": "idle", "voice_id": voice_id, "stage": None, "stage_index": None, "stages": STAGES,
            "epoch": None, "epochs": None, "percent": 0.0, "eta_seconds": None, "message": None, "error": None,
            "started": None, "finished": None, "can_resume": False}


def _no_progress() -> dict:
    return {"stage": None, "stage_index": None, "stages": STAGES, "epoch": None, "percent": 0.0,
            "eta_seconds": None, "message": None}


class _Job:
    """The latest training run of one voice: its record (what job.json holds) and where its progress stands."""

    def __init__(self, folder: Path, record: dict):
        self.dir = folder  # data/voices/<id>/training
        self.record = record
        self.progress = _no_progress()
        self.offset = 0  # how much of progress.jsonl has been read
        self.handle = None
        self.exited = False  # the worker process itself has ended
        self.final: dict | None = None  # the worker's last line of this run, once seen
        self.final_at: float | None = None
        self.cancel_at: float | None = None
        self.forced: tuple[str, str | None] | None = None  # must be stopped now, and how the job then ends
        self.tree: list = []  # the worker's processes as last seen, so none is lost if the worker dies first
        self.tree_at: float | None = None
        self.killing = False  # a kill is in progress outside the lock; nobody else touches the job meanwhile
        self.retry_at: float | None = None  # the last kill left something running; the next try is due then

    @property
    def voice_id(self) -> str:
        return self.record["voice_id"]

    def apply(self, raw: bytes) -> None:
        """Takes one progress line. Anything that is not a JSON object is skipped; wrongly typed values are ignored."""
        try:
            line = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001 - bad bytes, bad JSON, absurd nesting
            return
        if not isinstance(line, dict):
            return
        p = self.progress
        if isinstance(line.get("stage"), str):
            p["stage"] = line["stage"]
        for key in ("stage_index", "stages"):
            if _whole(line.get(key)):
                p[key] = line[key]
        if "epoch" in line and (line["epoch"] is None or _whole(line["epoch"])):
            p["epoch"] = line["epoch"]
        if _number(line.get("percent")):
            p["percent"] = min(100.0, max(0.0, float(line["percent"])))
        if "eta_seconds" in line and (line["eta_seconds"] is None or _number(line["eta_seconds"])):
            p["eta_seconds"] = line["eta_seconds"]
        if isinstance(line.get("message"), str):
            p["message"] = line["message"]
        if line.get("done") is True:
            self.final = line

    def read_progress(self) -> None:
        """Reads the complete lines appended since the last call. A half-written last line waits for the next call."""
        try:
            with open(self.dir / "progress.jsonl", "rb") as f:
                size = os.fstat(f.fileno()).st_size
                if size < self.offset:
                    self.offset = 0  # the worker starts a fresh run by emptying the file
                if size == self.offset:
                    return
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return
        self.offset += end + 1
        for raw in data[:end].split(b"\n"):
            if raw.strip():
                self.apply(raw)

    def read_tail(self) -> None:
        """For a run that ended before this process started: where it had got to, from the end of the file."""
        path = self.dir / "progress.jsonl"
        try:
            self.offset = max(0, path.stat().st_size - TAIL_BYTES)
            if self.offset:
                with open(path, "rb") as f:
                    f.seek(self.offset)
                    self.offset += len(f.readline())  # the rest of a line that began before the tail
        except OSError:
            return
        self.read_progress()
        self.final = None


class TrainingManager:
    def __init__(self, data: Path, library: Library, trainer_root: Path, launch=None, verify=None, pid_alive=None,
                 now=time.time, is_worker=None, kill_pid=None, processes=None):
        """launch(argv, log_path) -> handle with pid, poll() and kill_tree(); verify(fast) -> the verify.py report;
        pid_alive(pid) -> bool; is_worker(pid, experiment) -> whether that pid really is a worker for that
        experiment; kill_pid(pid) ends a process tree and raises if it could not; processes is a ProcessTools."""
        self._voices = Path(data) / "voices"
        self._library = library
        self._trainer = Path(trainer_root)
        self._worker = self._trainer / "worker.py"
        self._tools = processes or ProcessTools()
        self._launch = launch or (lambda argv, log_path: _launch_worker(argv, log_path, self._tools))
        self._verify = verify or self._run_verify
        self._pid_alive = pid_alive or _pid_alive
        self._is_worker = is_worker or (lambda pid, experiment: _is_worker_process(pid, experiment, self._worker))
        self._kill_pid = kill_pid or (lambda pid: _kill_tree(pid, self._tools))
        self._now = now
        self._lock = threading.RLock()  # guards everything below; never held while a process is being killed
        self._jobs: dict[str, _Job] = {}  # the latest job of each voice
        self._active: str | None = None  # the voice whose worker is running
        self._starting: str | None = None  # the voice whose start is waiting for the environment check
        self._start_cancelled = False  # cancel or shutdown arrived during that wait
        self._forgetting: set[str] = set()  # voices whose training files are being deleted right now
        self._passed: dict | None = None  # the last full environment report, kept only if it passed
        self._status_failed = False
        self._recover()

    # Paths

    def _experiment_dir(self, voice_id: str) -> Path:
        return self._trainer / "rvc" / "logs" / f"voice-{voice_id}"

    def _small_model(self, voice_id: str) -> Path:
        return self._trainer / "rvc" / "assets" / "weights" / f"voice-{voice_id}.pth"

    # Environment

    def _run_verify(self, fast: bool) -> dict:
        argv = [sys.executable, str(self._trainer / "verify.py")] + (["--fast"] if fast else [])
        try:
            done = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=300, creationflags=_NO_WINDOW)
        except (OSError, subprocess.SubprocessError) as exc:
            return _failed_report(f"The training environment check could not be run: {_reason(exc)}")
        lines = (done.stdout or "").strip().splitlines()
        try:
            return json.loads(lines[-1])
        except (IndexError, ValueError):
            detail = ((done.stderr or "").strip() or (done.stdout or "").strip())[-400:] or "no output"
            return _failed_report(f"The training environment check failed to run: {detail}")

    def _check(self, fast: bool) -> dict:
        try:
            return _clean_report(self._verify(fast))
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed environment, never a crash
            log.exception("The training environment check raised")
            return _failed_report(f"The training environment check could not be run: {_reason(exc)}")

    def environment(self, refresh: bool = False) -> dict:
        """The verify report plus "ready". Only a passing report is remembered, so a fixed setup shows at once."""
        with self._lock:
            if refresh:
                self._passed = None
            report = self._passed
        if report is None:
            report = self._check(fast=False)  # several seconds; never under the lock
            if report["ok"]:
                with self._lock:
                    self._passed = report
        return {**report, "ready": report["ok"]}

    def _require_environment(self) -> None:
        with self._lock:
            passed = self._passed is not None
        # Once the full check has passed, each later start only repeats the quick file checks.
        report = self._check(fast=passed)
        with self._lock:
            if not report["ok"]:
                self._passed = None
            elif not passed:
                self._passed = report
        if not report["ok"]:
            failing = "; ".join(c["message"] for c in report["checks"] if not c["ok"])
            raise TrainingError(f"Training is not set up: {failing}")

    # Records

    def _save(self, job: _Job) -> None:
        """Memory stays the truth for this process; a record that cannot be written is logged, never raised."""
        try:
            atomic_json(job.dir / "job.json", job.record)
        except OSError as exc:
            log.warning("Could not save the training record of %s: %s", job.voice_id, _reason(exc))

    @staticmethod
    def _remove_cancel_file(job_dir: Path) -> None:
        try:
            (job_dir / "cancel").unlink(missing_ok=True)
        except OSError:
            pass

    def _recover(self) -> None:
        """Loads every voice's last job. One still marked as running belongs to a backend that is gone."""
        try:
            folders = sorted(self._voices.iterdir())
        except OSError:
            return
        for folder in folders:
            try:
                self._recover_voice(folder)
            except Exception:  # noqa: BLE001 - one voice's damaged files must not stop the backend starting
                log.exception("Could not read the training record in %s", folder.name)

    def _recover_voice(self, folder: Path) -> None:
        voice_id = folder.name
        path = folder / "training" / "job.json"
        if not ID_RE.match(voice_id) or not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Ignoring the unreadable training record of %s (%s). It was not changed.",
                        voice_id, type(exc).__name__)
            return
        if not isinstance(raw, dict) or raw.get("state") not in ACTIVE + FINISHED:
            log.warning("Ignoring the training record of %s: it does not have the expected contents.", voice_id)
            return
        # The id and experiment come from the folder name, never from the file: they become paths.
        job = _Job(folder / "training", {
            "voice_id": voice_id, "experiment": f"voice-{voice_id}",
            "epochs": raw.get("epochs") if _whole(raw.get("epochs")) else None,
            "batch_size": raw.get("batch_size") if _whole(raw.get("batch_size")) else None,
            "state": raw["state"],
            "started": raw.get("started") if _number(raw.get("started")) else None,
            "finished": raw.get("finished") if _number(raw.get("finished")) else None,
            "worker_pid": raw.get("worker_pid") if _whole(raw.get("worker_pid")) else None,
            "error": raw.get("error") if isinstance(raw.get("error"), str) else None,
            "resumable": raw.get("resumable") if isinstance(raw.get("resumable"), bool) else None,
        })
        self._jobs[voice_id] = job
        job.read_tail()
        if job.record["state"] in ACTIVE:
            self._stop_leftover_worker(job)
            job.record.update(state="interrupted", finished=self._now(), worker_pid=None)
            self._remove_cancel_file(job.dir)
            self._save(job)

    def _stop_leftover_worker(self, job: _Job) -> None:
        """A worker that outlived its backend is never adopted: it is stopped so nothing unowned holds the GPU.
        Only a pid that is shown to be this voice's worker is killed; a reused pid is someone else's program."""
        pid = job.record["worker_pid"]
        try:
            if pid is None or not self._pid_alive(pid):
                return
            if not self._is_worker(pid, job.record["experiment"]):
                log.warning("Process %s recorded for the training of %s is running but could not be confirmed as "
                            "its worker, so it was left alone.", pid, job.voice_id)
                return
        except Exception:  # noqa: BLE001 - not confirmed, so not killed
            log.exception("Could not check process %s recorded for the training of %s", pid, job.voice_id)
            return
        log.warning("Stopping the training worker %s of %s left behind by an earlier backend.", pid, job.voice_id)
        try:
            self._kill_pid(pid)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not stop the training worker %s of %s left behind by an earlier backend: %s",
                        pid, job.voice_id, _reason(exc))

    # Following the running job

    def _has_checkpoint(self, voice_id: str) -> bool:
        folder = self._experiment_dir(voice_id)
        try:
            return any(all((folder / (name + suffix)).is_file() for name in CHECKPOINTS) for suffix in ("", ".bak"))
        except OSError:
            return False

    def _resume_possible(self, job: _Job | None) -> bool:
        """The hard rule, which gates start(resume=True): a stopped run with a checkpoint."""
        return job is not None and job.record["state"] in RESUMABLE and self._has_checkpoint(job.voice_id)

    def _can_resume(self, job: _Job | None) -> bool:
        """What the screen offers: the hard rule, unless the worker said this run is not worth resuming as it is."""
        if not self._resume_possible(job):
            return False
        hint = job.record.get("resumable")
        if hint is not None:
            return hint
        return not (job.record["error"] or "").startswith(NEVER_RESUMABLE)

    def _settle(self, job: _Job, state: str, error: str | None = None) -> None:
        hint = job.final.get("resumable") if job.final is not None else None
        job.record.update(state=state, finished=self._now(), error=error, worker_pid=None,
                          resumable=hint if isinstance(hint, bool) else None)
        job.progress["eta_seconds"] = None
        job.handle = job.final = job.final_at = job.cancel_at = job.forced = job.retry_at = job.tree_at = None
        job.tree = []
        self._active = None
        self._remove_cancel_file(job.dir)
        self._save(job)

    @staticmethod
    def _result(job: _Job) -> tuple[str, str | None]:
        """How the job ended. The worker's own final line counts first."""
        final = job.final
        if final is not None:
            if final.get("ok") is True:
                return "done", None
            if final.get("cancelled") is True:
                return "cancelled", None
            error = final.get("error")
            if not isinstance(error, str) or not error.strip():
                error = "Training failed. See training/worker.log."
            return "failed", error
        if job.forced is not None:
            return job.forced
        if job.cancel_at is not None:
            return "cancelled", None  # it was asked to stop and it is gone
        return "failed", STOPPED_UNEXPECTEDLY

    def _refresh_tree(self, job: _Job, now: float) -> None:
        if job.tree_at is not None and now - job.tree_at < TREE_REFRESH:
            return
        job.tree_at = now
        try:
            tree = self._tools.tree(job.handle.pid)
        except Exception:  # noqa: BLE001
            return
        if tree:  # an empty answer means the worker has just gone: the last list is the one that matters
            job.tree = tree

    def _poll(self) -> _Job | None:
        """Brings the running job up to date under the lock: new progress, or an exit. Returns the job when
        its processes now have to be stopped, which the caller does after releasing the lock (_stop)."""
        job = self._jobs.get(self._active) if self._active is not None else None
        if job is None or job.killing:
            return None
        now = self._now()
        try:
            code = job.handle.poll()
        except Exception:  # noqa: BLE001
            log.exception("Could not ask whether the training worker is still running")
            code = None
        job.read_progress()  # after the exit check, so a final line written just before the exit is never missed
        if code is None:
            self._refresh_tree(job, now)
            if job.final is not None and job.final_at is None:
                job.final_at = now
            overdue = (job.forced is not None
                       or (job.final_at is not None and now - job.final_at >= EXIT_GRACE)
                       or (job.cancel_at is not None and now - job.cancel_at >= CANCEL_TIMEOUT))
            if not overdue:
                return None
        else:
            job.exited = True
            if job.final is not None or not job.tree:
                self._settle(job, *self._result(job))
                return None
            # It went without a final line, so it did not clean up: a stage process may have outlived it.
        if job.retry_at is not None and now < job.retry_at:
            return None
        job.killing = True
        return job

    def _stop(self, job: _Job) -> None:
        """Ends the job's processes, without the lock. The job is settled only if nothing is left running;
        otherwise it stays as it is and _poll asks again after KILL_RETRY seconds."""
        failure = None
        try:
            if not job.exited:
                job.handle.kill_tree()
            leftover = _end_survivors(job.tree, self._tools)
            if leftover:
                log.warning("Stopped processes the training worker of %s left running: %s",
                            job.voice_id, _pids(leftover))
        except Exception as exc:  # noqa: BLE001
            failure = exc
        with self._lock:
            job.killing = False
            if failure is not None:
                log.warning("Could not stop the training processes of %s, trying again in %g seconds: %s",
                            job.voice_id, KILL_RETRY, _reason(failure))
                job.retry_at = self._now() + KILL_RETRY
            elif self._active == job.voice_id and self._jobs.get(job.voice_id) is job:
                self._settle(job, *self._result(job))

    def _service(self) -> None:
        with self._lock:
            job = self._poll()
        if job is not None:
            self._stop(job)

    def _present(self, job: _Job | None) -> _Job | None:
        """Forgets a finished job whose voice folder no longer exists."""
        if job is None or job.voice_id == self._active or job.dir.parent.is_dir():
            return job
        self._jobs.pop(job.voice_id, None)
        return None

    def _latest(self) -> _Job | None:
        if self._active is not None:
            return self._jobs.get(self._active)
        while self._jobs:
            job = max(self._jobs.values(), key=lambda j: j.record["finished"] or j.record["started"] or 0.0)
            if self._present(job) is not None:
                return job
        return None

    def _snapshot(self, job: _Job | None, voice_id=None) -> dict:
        if job is None:
            return _idle(voice_id if isinstance(voice_id, str) else None)
        return {"state": job.record["state"], "voice_id": job.voice_id, **job.progress,
                "epochs": job.record["epochs"], "error": job.record["error"], "started": job.record["started"],
                "finished": job.record["finished"], "can_resume": self._can_resume(job)}

    # The public surface

    @property
    def running(self) -> bool:
        self._service()
        with self._lock:
            return self._active is not None or self._starting is not None

    def status(self, voice_id: str | None = None) -> dict:
        """The latest job of that voice, or without one the running job, else the most recently finished one.
        Polled every second: reads only what the worker appended since the last call, and never raises."""
        try:
            self._service()
            with self._lock:
                if voice_id is None:
                    job = self._latest()
                else:
                    job = self._present(self._jobs.get(voice_id)) if isinstance(voice_id, str) else None
                return self._snapshot(job, voice_id)
        except Exception:  # noqa: BLE001
            if not self._status_failed:  # once: this is called every second
                self._status_failed = True
                log.exception("Could not work out the training status")
            return _idle(voice_id if isinstance(voice_id, str) else None)

    def start(self, voice_id: str, epochs: int | None = None, batch_size: int | None = None,
              resume: bool = False) -> dict:
        """Starts training (200 epochs, batch size 8 unless given). A resume keeps the stopped run's settings
        unless epochs is given; its batch size cannot change, because the prepared data depends on it."""
        self._service()
        with self._lock:
            try:
                voice = self._library.get(voice_id)
            except LibraryError:
                raise TrainingError("Unknown voice") from None
            previous = self._jobs.get(voice_id)
            kept = previous.record if resume and previous is not None else {}
            if epochs is None:
                epochs = kept.get("epochs") or DEFAULT_EPOCHS
            if batch_size is None:
                batch_size = kept.get("batch_size") or DEFAULT_BATCH_SIZE
            if not _whole(epochs) or not MIN_EPOCHS <= epochs <= MAX_EPOCHS:
                raise TrainingError(f"Epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}")
            if not _whole(batch_size) or batch_size not in BATCH_SIZES:
                raise TrainingError("Batch size must be 4, 8 or 16")
            if self._active is not None or self._starting is not None:
                raise TrainingError("Training is already running")
            if voice_id in self._forgetting:
                raise TrainingError("This voice's training files are being removed. Try again in a moment.")
            if resume:
                if not self._resume_possible(previous):
                    raise TrainingError(NO_RESUME)
                if kept.get("batch_size") not in (None, batch_size):
                    raise TrainingError(f"A resumed run keeps its batch size ({kept['batch_size']}). "
                                        "Start a new training run to change it.")
            if voice.minutes < MIN_MINUTES:
                shown = int(voice.minutes * 10 + 1e-9) / 10  # rounded down, so it never reads "5.0 ... at least 5"
                raise TrainingError(f"This voice has {shown:.1f} minutes of clean audio. Training needs at least 5.")
            self._starting = voice_id  # holds the single slot while the environment check runs outside the lock
            self._start_cancelled = False
        try:
            self._require_environment()
            with self._lock:
                if self._start_cancelled:
                    raise TrainingError(CANCELLED_BEFORE_START)
                return self._start_worker(voice, epochs, batch_size, bool(resume))
        except TrainingError:
            self._service()  # outside the lock: stops a worker whose start could not be completed
            raise
        finally:
            with self._lock:
                self._starting = None
                self._start_cancelled = False

    def _start_worker(self, voice, epochs: int, batch_size: int, resume: bool) -> dict:
        voice_dir = self._voices / voice.id
        job_dir = voice_dir / "training"
        experiment = f"voice-{voice.id}"
        progress_path = job_dir / "progress.jsonl"
        previous = self._jobs.get(voice.id)
        try:
            job = _Job(job_dir, {"voice_id": voice.id, "experiment": experiment, "epochs": epochs,
                                 "batch_size": batch_size, "state": "running", "started": self._now(),
                                 "finished": None, "worker_pid": None, "error": None, "resumable": None})
            job_dir.mkdir(exist_ok=True)  # not parents: a voice deleted a moment ago is never recreated
            if resume:
                job.offset = progress_path.stat().st_size if progress_path.exists() else 0  # the worker appends
                if previous is not None:  # the bar continues from where the stopped run was
                    job.progress.update(previous.progress, eta_seconds=None, message=None)
            else:
                open(progress_path, "wb").close()  # so the last run's final line is never read as this run's
            self._remove_cancel_file(job_dir)
            argv = [str(self._trainer / ".venv" / "Scripts" / "python.exe"), "-P", str(self._worker),
                    "--voice-dir", str(voice_dir), "--job-dir", str(job_dir), "--experiment", experiment,
                    "--epochs", str(epochs), "--batch-size", str(batch_size), "--save-every", str(SAVE_EVERY),
                    *(["--resume"] if resume else []), "--parent-pid", str(os.getpid()),
                    f"--name={voice.name}"]  # one argument, so a name starting with "-" is still a name
            handle = self._launch(argv, job_dir / OUTPUT_LOG)
        except Exception as exc:  # noqa: BLE001
            log.exception("Could not start the training worker")
            raise TrainingError(f"Could not start training: {_reason(exc)}") from exc
        # A worker now exists, and from this line on it is this manager's: whatever goes wrong below, it is
        # the active job until its processes are known to be gone.
        job.handle = handle
        self._jobs[voice.id] = job
        self._active = voice.id
        try:
            job.record["worker_pid"] = handle.pid
            atomic_json(job_dir / "job.json", job.record)
            return self._snapshot(job)
        except Exception as exc:  # noqa: BLE001
            # Without a record a later backend could not find this worker, so it does not get to run.
            what = "Could not save the training record" if isinstance(exc, OSError) else "Could not start training"
            error = f"{what}: {_reason(exc)}"
            log.error("Stopping the training worker of %s that was just started: %s", voice.id, error)
            job.record["state"] = "cancelling"
            job.forced = ("failed", error)  # start() stops it once the lock is released
            raise TrainingError(error) from exc

    def cancel(self) -> dict:
        """Asks the worker to stop at the next safe moment. If it has not exited 90 seconds later it is killed
        (on a later status call). A start that is still checking the environment is called off instead."""
        self._service()
        with self._lock:
            job = self._jobs.get(self._active) if self._active is not None else None
            if job is None:
                if self._starting is None:
                    raise TrainingError("Training is not running")
                self._start_cancelled = True
                return self._snapshot(self._jobs.get(self._starting), self._starting)
            if job.cancel_at is None:
                job.cancel_at = self._now()
                try:
                    (job.dir / "cancel").touch()
                except OSError as exc:  # the worker cannot be told; the 90 second kill still applies
                    log.warning("Could not create the cancel file: %s", _reason(exc))
                job.record["state"] = "cancelling"
                self._save(job)
            return self._snapshot(job)

    def shutdown(self) -> None:
        """For the backend's own exit: stops the worker now and leaves the job resumable. If the worker cannot
        be stopped the job stays recorded as running, and the next backend deals with it when it starts."""
        with self._lock:
            if self._starting is not None:
                self._start_cancelled = True
            job = self._jobs.get(self._active) if self._active is not None else None
            if job is not None:
                job.forced = ("interrupted", None)
                job.retry_at = None
        self._service()

    def forget(self, voice_id: str) -> None:
        """Removes what training left under trainer/ for a voice (checkpoints and the small model). These are
        derived files; nothing in the voice's own folder is touched."""
        if not isinstance(voice_id, str) or not ID_RE.match(voice_id):
            raise TrainingError("Unknown voice")
        self._service()
        with self._lock:
            if voice_id in (self._active, self._starting):
                raise TrainingError("This voice is training. Stop the training first.")
            self._jobs.pop(voice_id, None)
            self._forgetting.add(voice_id)  # no training of this voice may start while its files are going
        try:
            folder, model = self._experiment_dir(voice_id), self._small_model(voice_id)
            try:
                if folder.is_dir():
                    shutil.rmtree(folder)  # can be over a gigabyte: never under the lock
                model.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not remove all training files of %s under %s: %s",
                            voice_id, self._trainer, _reason(exc))
        finally:
            with self._lock:
                self._forgetting.discard(voice_id)
