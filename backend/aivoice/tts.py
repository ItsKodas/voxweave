"""Text to speech: owns the worker process (tts/tts_worker.py, in its own Python environment) and runs a request
from text to sound: pick the voice's reference clip, split the text, have the worker synthesise one piece at a
time, check and finish each piece as it arrives (through the voice's trained RVC model for "best"), hand it to
the player so that sound starts early, and optionally save the result. One request at a time.

Nothing from tts/ is ever imported here (the backend has no torch): the worker and the environment check are
only ever run as subprocesses.

The worker. It is started on the first request, kept for later ones, and stopped after IDLE_UNLOAD seconds without
a request (by the manager's own housekeeping thread: nobody has to ask for the status), on unload() and on
shutdown(). Its stdout carries the protocol and is read by one thread per worker,
which turns lines into events on a queue; its stderr goes to data/tts/worker.log, never into the protocol and
never into a pipe nobody reads. A worker is only ever called gone once its processes are known to be gone: the
kill is stage 3's (aivoice.training: a list of the worker's processes taken beforehand, taskkill on the tree,
whatever is left ended one by one, and all of it checked), it runs outside the manager's lock, and if anything
survived it is tried again KILL_RETRY seconds later while no new worker may start.

A request runs on its own thread, which is the only one that writes to the worker or reads its events. It never
leaves the worker owing an answer: a stopped or failed request cancels its job and waits for the worker to say
the job has ended (or for the answer to a reference still loading) before the manager is idle again, so the next
request can never be refused by the worker, nor take a late event for its own. A worker that does not answer in
time, ends, or reports an internal error is killed, and the next request starts a fresh one.

Nothing unchecked is played: a piece that is missing, unreadable, silent after trimming, or that fails the RVC
pass fails the request. A request's files live under data/tts/tmp/<job id>/ and that folder is removed however
the request ends; whatever an earlier run left under data/tts/tmp is removed at start-up. Saved results
(data/tts/*.wav) and the voices' own clips are only ever read.

Kills are performed, outside the lock, by the housekeeping thread, status (the screen's poll), speak, unload,
shutdown and the request's own thread. Whoever is about to kill a worker marks it as being killed under the lock
first, so two of them never kill the same worker at once; unload and shutdown wait for a kill another thread
has in progress. is_active and busy only take the lock: they are what a route may call while it holds another
lock.

The housekeeping thread is started with the first worker and runs until shutdown(). About every HOUSEKEEPING
seconds it does what a status poll does (a worker that died while idle, the idle timeout, kills due another
try) and nothing else: it never touches a request.
"""
import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import soxr

from aivoice import tts_audio, tts_reference, tts_text
from aivoice.audiofile import read_wav
from aivoice.devices import get_device
from aivoice.library import Library, LibraryError
from aivoice.playback import save_wav
# Stage 3's process handling, shared rather than copied: the windowless detached launch flags, the handle whose
# kill_tree() lists the tree before killing it and checks the result, and the sweep that ends what is left.
from aivoice.training import (KILL_RETRY, TREE_REFRESH, KillFailed, ProcessTools, _NEW_GROUP, _NO_WINDOW, _Worker,
                              _end_survivors, _failed_report, _kill_tree, _pids, _reason, unavailable_report)

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
# Sampling steps; "best" adds the voice's RVC model. Fast is 10, not 8: measured with random seeds, a quarter of
# the sentences had a wrong word at 8 steps, and at 10 no more than at 16 (stage 4 first-real-run notes). The
# worker has its own list of the counts it accepts (tts/tts_worker.py, STEP_COUNTS).
STEPS = {"fast": 10, "balanced": 16, "best": 32}
SPEED = 1.0
START_TIMEOUT = 120.0  # seconds a new worker has to load its model
PIECE_TIMEOUT = 60.0  # seconds the worker has for the next piece, or for loading a reference clip
CANCEL_TIMEOUT = 10.0  # seconds it has to end a cancelled job (it takes milliseconds)
STOP_WAIT = 15.0  # how long stop() waits for the request's thread to finish cleaning up
IDLE_UNLOAD = 600.0  # the worker is stopped after this long without a request
HOUSEKEEPING = 30.0  # how often the manager looks after its worker by itself, with nobody asking for the status
VERIFY_TIMEOUT = 300.0  # seconds the full environment check may take (it starts the TTS Python twice)
FAST_VERIFY_TIMEOUT = 20.0  # and the quick one, which a request runs while the audio and voices locks are held
PIPE_WAIT = 2.0  # how long the output of a check that was ended is waited for
MAX_MONITOR_GAIN = 2.0  # live mode's own limit for the listen volume
TICK = 0.25  # how often a waiting request looks at the clock, the stop flag and the player
TAIL_BYTES, TAIL_LINES, TAIL_CHARS = 4096, 6, 600
KILL_PATIENCE = 30.0  # how long unload and shutdown wait for a kill another thread has in progress (at worst 23 s)
SAVE_NAMES = 1000  # how many file names are tried for results saved in the same second
LOG_NAME, PREVIOUS_LOG_NAME = "worker.log", "worker.prev.log"
JOB_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # the worker's rule: the id becomes part of a file name

ALREADY = "Already speaking. Stop it first."
# What the installed app answers instead of checking for a text-to-speech environment it cannot have.
NOT_IN_INSTALLED_APP = ("Text to speech is not available in the installed app yet. It needs the source checkout "
                        "of Voxweave on a PC with an AMD GPU. Live mode works here, and a voice trained there "
                        "can be brought to this PC as a voice pack: drop its .voxweave file on the Voices tab.")
NOTHING_TO_SAY = "There is nothing to say in that text"
NO_MODEL = ("Best needs this voice's trained model, and it has none. Train the voice first, or choose Fast or "
            "Balanced.")
NO_SOUND = "The voice engine produced no sound for sentence {}"
OUT_OF_MEMORY = ("The graphics card ran out of memory while speaking. Close other programs that use it and try "
                 "again.")
CLOSING = "The app is closing"

_EOF = object()  # the worker's stdout ended: it is gone
_WAKE = object()  # look at the stop flag now


class TtsError(ValueError):
    pass


class _Stopped(Exception):
    """The request was stopped."""


class _Failed(Exception):
    """The request failed; the message is for the user."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def worker_environment(cache: Path | None) -> dict:
    """The worker's environment variables (section 4 of the spike notes), as the real-model test sets them. With
    a cache folder (tts/.cache), everything the libraries cache or write as temporary files lands there: the
    MIOpen cache must persist (the first call takes three times as long without it) and nothing should land in
    the user's profile. Synthesis is offline."""
    env = dict(os.environ)
    env.update({"ROCM_SDK_TARGET_FAMILY": "custom", "MIOPEN_LOG_LEVEL": "3", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled", "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore::SyntaxWarning"})
    env.pop("PYTHONDONTWRITEBYTECODE", None)  # imports are the slowest part of start-up
    if cache is not None:
        folders = {"MIOPEN_USER_DB_PATH": cache / "miopen" / "db", "MIOPEN_CUSTOM_CACHE_DIR": cache / "miopen",
                   "TEMP": cache / "tmp", "TMP": cache / "tmp", "HF_HOME": cache / "hf",
                   "TORCH_HOME": cache / "torch", "XDG_CACHE_HOME": cache / "xdg", "MPLCONFIGDIR": cache / "mpl",
                   "NUMBA_CACHE_DIR": cache / "numba"}
        for name, folder in folders.items():
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # the variable is left as it was; the worker fills in what it must have
                log.warning("Could not create %s: %s", folder, _reason(exc))
            else:
                env[name] = str(folder)
    return env


def _launch_worker(argv: list[str], log_path: Path, tools=None, cache: Path | None = None) -> _Worker:
    """Starts the worker with its own hidden console and process group, like the training worker. stdin and
    stdout are the protocol's pipes. stderr is a file, started afresh for each worker: it must not be mixed into
    stdout (the worker sends everything its libraries print to stderr), and it must never fill up (the worker
    logs while holding its lock, so a full pipe would stall a cancel)."""
    env = worker_environment(cache)
    tools = tools or ProcessTools()
    proc = None
    try:
        with open(log_path, "wb") as err:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env,
                                    creationflags=_NO_WINDOW | _NEW_GROUP)
        handle = _Worker(proc, tools)
        handle.stdin, handle.stdout = proc.stdin, proc.stdout
        return handle
    except BaseException:
        # The process exists but nobody will ever hold a handle to it: it is ended here, before the error goes on.
        if proc is not None:
            _end_process(proc, tools, "the voice engine that could not be taken into use")
            _close_pipes(proc)  # nobody reads them; a worker that somehow lived would end at the end of its input
        raise


def _end_process(proc, tools, what: str) -> None:
    """Ends a process this module started and everything it started, and checks it. Never raises: what could
    not be ended is logged with its process number, which is all that is left to do."""
    try:
        _kill_tree(proc.pid, tools, gone=lambda: proc.poll() is not None)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not stop %s (process %s); end it in Task Manager: %s", what, proc.pid, _reason(exc))
    try:
        proc.poll()  # collect the exit code
    except Exception:  # noqa: BLE001
        pass


def _close_pipes(proc) -> None:
    for pipe in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
        try:
            if pipe is not None:
                pipe.close()
        except Exception:  # noqa: BLE001 - a pipe of a process that is gone
            pass


def _run_bounded(argv: list[str], timeout: float, tools) -> subprocess.CompletedProcess:
    """subprocess.run for a check with a time limit that holds: when the time is up the whole process tree is
    ended, not only the process that was started. (The Python of a virtual environment is a launcher with the
    real interpreter as its child; subprocess.run would kill the launcher and then wait for the child's output
    for as long as the child lives.) Raises subprocess.TimeoutExpired after that."""
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException:
        _end_process(proc, tools, "the text-to-speech environment check")
        try:
            proc.communicate(timeout=PIPE_WAIT)  # lets go of its pipes; with the tree gone they have ended
        except Exception:  # noqa: BLE001 - something still holds them: they are left to the threads reading them
            pass
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _clean_report(raw) -> dict:
    """The verify report as {"ok", "checks"}, or a failing one if it does not have that shape."""
    if (isinstance(raw, dict) and isinstance(raw.get("ok"), bool) and isinstance(raw.get("checks"), list)
            and all(isinstance(c, dict) and isinstance(c.get("ok"), bool) and isinstance(c.get("message"), str)
                    for c in raw["checks"])):
        return {"ok": raw["ok"] and all(c["ok"] for c in raw["checks"]), "checks": raw["checks"]}
    return _failed_report("The text-to-speech environment check gave no usable result")


def _reference_refusal(message: str, file_name: str, clip_index: int) -> str:
    """The worker's refusal of a reference clip in words the user can act on. The worker knows the clip as a
    file; the Voices tab shows it as a numbered clip. The app picks the clip itself, so "choose another clip"
    is advice nobody can follow: what can be done is to record a better one or delete this one. A refusal that
    does not name the clip's file is about something else and is passed on as it is."""
    if file_name not in message:
        return message
    if "too quiet" in message:
        return tts_reference.too_quiet([clip_index])
    clip = f"Clip {clip_index}, this voice's reference clip,"
    text = message.strip().replace(f"The reference clip {file_name}", clip).replace(file_name, f"clip {clip_index}")
    text = re.sub(r",?\s*or choose another clip", "", text).rstrip(". ")
    return f"{text}. Delete that clip or record it again."


def _sendable(text: str) -> str:
    """Without lone surrogates: they pass through JSON but are not text the worker could encode."""
    return "".join(c for c in text if unicodedata.category(c) != "Cs")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


class _Engine:
    """One worker process: its handle, the events its stdout has produced, and what it has loaded."""

    def __init__(self, handle, log_path: Path, started: float):
        self.handle = handle
        self.pid = handle.pid
        self.log_path = log_path
        self.started = started
        self.events: queue.Queue = queue.Queue()  # parsed protocol lines, _EOF, and _WAKE
        self.ready = False  # the model is loaded; set by the reader
        self.reference: tuple | None = None  # ((voice id, clip index, transcript), seconds) of the loaded clip
        self.broken = False  # it died, hung or lost its place in the protocol: it is to be killed
        self.tree: list = []  # its processes as last seen, so none is lost if the worker itself dies first
        self.tree_at: float | None = None
        self.killing = False  # a kill is in progress outside the lock
        self.resting = threading.Event()  # no kill is in progress
        self.resting.set()
        self.retry_at: float | None = None  # the last kill left something running; the next try is due then
        self.left: list[int] = []  # the pids that kill left running, for the message
        self._writing = threading.Lock()

    def send(self, command: dict) -> None:
        data = (json.dumps(command) + "\n").encode("utf-8")  # ASCII: json escapes everything else
        with self._writing:
            self.handle.stdin.write(data)
            self.handle.stdin.flush()

    def read(self) -> None:
        """The reader thread: one JSON object per line becomes an event; the end of the pipe is the last one."""
        stdout = self.handle.stdout
        try:
            for raw in stdout:
                try:
                    event = json.loads(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)
                except ValueError:
                    event = None
                if not isinstance(event, dict):
                    log.warning("The voice engine wrote something that is not a protocol line; ignored: %.200r", raw)
                    continue
                if event.get("event") == "ready":
                    self.ready = True
                self.events.put(event)
        except Exception as exc:  # noqa: BLE001 - a pipe that cannot be read is a worker that is gone
            log.warning("Could not read from the voice engine: %s", _reason(exc))
        finally:
            self.events.put(_EOF)
            try:
                stdout.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        """After the process is known to be gone. stdout is closed by the thread that reads it."""
        with self._writing:
            try:
                self.handle.stdin.close()
            except Exception:  # noqa: BLE001 - a broken pipe cannot be flushed; it is closed all the same
                pass


class _Job:
    """One request, from speak() until its files are gone."""

    def __init__(self, id: str, voice_id: str, preset: str, text: str, clip_index: int, wav: Path, ref_text: str,
                 max_chars: int, pieces: list[str], save: bool, folder: Path, started: float):
        self.id, self.voice_id, self.preset, self.text = id, voice_id, preset, text
        self.wav, self.ref_text, self.max_chars, self.pieces = wav, ref_text, max_chars, pieces
        self.clip_index = clip_index  # the number the Voices tab shows for the reference clip
        self.key = (voice_id, clip_index, ref_text)  # which reference clip this request needs loaded
        self.save, self.folder, self.started = save, folder, started
        self.state = "preparing"
        self.piece = 0
        self.error: str | None = None
        self.saved_path: str | None = None
        self.stop = threading.Event()
        self.done = threading.Event()  # set when nothing of the request is left
        self.session = None  # the player's token: audio queued with it can never join another request
        self.engine: _Engine | None = None
        self.awaiting: str | None = None  # an answer the worker still owes this request: "reference" or "job"
        self.finished: list[np.ndarray] = []  # the pieces as played, kept only when they are to be saved

    def describe(self) -> dict:
        return {"state": self.state, "voice_id": self.voice_id, "preset": self.preset, "piece": self.piece,
                "pieces": len(self.pieces), "error": self.error, "saved_path": self.saved_path,
                "started": self.started}


class _Probe:
    """A full environment check in progress: whoever asks meanwhile waits for it and takes its report."""

    def __init__(self):
        self.done = threading.Event()
        self.report: dict | None = None


def _nothing() -> dict:
    return {"state": "idle", "voice_id": None, "preset": None, "piece": 0, "pieces": 0, "error": None,
            "saved_path": None, "started": None}


class TtsManager:
    def __init__(self, data: Path, library: Library, tts_root: Path, player, list_devices, launch=None, verify=None,
                 rvc_factory=None, now=time.time, processes=None, tick: float = TICK, pacer=None,
                 unavailable: str | None = None):
        """player is an aivoice.playback.Player; list_devices() -> the audio devices; launch(argv, log_path) -> a
        handle with pid, poll(), kill_tree(), stdin (bytes) and stdout (lines of bytes); verify(fast) -> the
        tts/verify.py report; rvc_factory(voice_id) -> an object whose convert(piece) takes and returns mono
        float32 at 48 kHz; processes is training's ProcessTools; tick is how often a waiting request looks up;
        pacer is what the housekeeping thread waits on between its rounds and what shutdown sets to end it (a
        threading.Event: wait(seconds), set(), is_set()); unavailable is the message of an app that has no text
        to speech at all (the installed app): nothing is checked or started then."""
        self._unavailable = unavailable
        self._data = Path(data)
        self._root = self._data / "tts"
        self._tmp = self._root / "tmp"
        self._library = library
        self._tts = Path(tts_root)
        self._player = player
        self._list_devices = list_devices
        self._tools = processes or ProcessTools()
        self._launch = launch or (
            lambda argv, log_path: _launch_worker(argv, log_path, self._tools, self._tts / ".cache"))
        self._verify = verify or self._run_verify
        self._rvc_factory = rvc_factory or self._default_rvc
        self._now = now
        self._tick = tick
        self._lock = threading.RLock()  # guards everything below; never held while a process is being killed
        self._job: _Job | None = None  # the request in progress
        self._last = _nothing()  # how the last request ended, on show until the next one
        self._engine: _Engine | None = None  # the worker requests are sent to
        self._doomed: list[_Engine] = []  # workers to be killed, until their processes are known to be gone
        self._engine_failed = False  # the last worker died, hung or could not be started
        self._launching = False  # the request's thread is starting a worker, outside the lock
        self._last_used = now()
        self._closed = False
        self._passed: dict | None = None  # the last full environment report, kept only if it passed
        self._last_report: dict | None = None
        self._probe: _Probe | None = None  # the full environment check in progress, if any
        self._pacer = pacer or threading.Event()  # set by shutdown: the housekeeping thread ends
        self._keeper: threading.Thread | None = None  # the housekeeping thread, from the first worker on
        self.housekeeping_failures = 0  # errors a housekeeping round swallowed; like status_failures
        self.status_failures = 0  # errors status() swallowed; anything but 0 is a bug worth a look at the log
        self._sweep_tmp()

    # ---- files

    def _sweep_tmp(self) -> None:
        """Removes what an earlier run left under data/tts/tmp. Nothing outside that folder is looked at."""
        try:
            leftovers = list(self._tmp.iterdir())
        except OSError:
            return
        for path in leftovers:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as exc:
                log.warning("Could not remove the leftover temporary file %s: %s", path, _reason(exc))

    @staticmethod
    def _remove(folder: Path) -> None:
        shutil.rmtree(folder, ignore_errors=True)
        if os.path.lexists(folder):
            log.warning("Could not remove the temporary folder %s; it is removed when the app next starts.", folder)

    def _log_tail(self, engine: _Engine) -> str:
        try:
            with open(engine.log_path, "rb") as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - TAIL_BYTES))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()][-TAIL_LINES:]
        tail = " | ".join(lines)[-TAIL_CHARS:]
        return f" Last output of the voice engine: {tail}" if tail else ""

    # ---- the environment

    def _run_verify(self, fast: bool) -> dict:
        argv = [sys.executable, str(self._tts / "verify.py")] + (["--fast"] if fast else [])
        # The quick check is run by a request while the audio and voices locks are held: it gets seconds, not
        # minutes. Either way the time limit ends the check's whole process tree.
        limit = FAST_VERIFY_TIMEOUT if fast else VERIFY_TIMEOUT
        try:
            done = _run_bounded(argv, limit, self._tools)
        except subprocess.TimeoutExpired:
            return _failed_report(f"The text-to-speech environment check did not finish within {limit:g} seconds "
                                  "and was stopped. Try again; if it keeps happening, restart the app.")
        except (OSError, subprocess.SubprocessError) as exc:
            return _failed_report(f"The text-to-speech environment check could not be run: {_reason(exc)}")
        lines = (done.stdout or "").strip().splitlines()
        try:
            return json.loads(lines[-1])
        except (IndexError, ValueError):
            detail = ((done.stderr or "").strip() or (done.stdout or "").strip())[-400:] or "no output"
            return _failed_report(f"The text-to-speech environment check failed to run: {detail}")

    def _check_environment(self, fast: bool) -> dict:
        try:
            return _clean_report(self._verify(fast))
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed environment, never a crash
            log.exception("The text-to-speech environment check raised")
            return _failed_report(f"The text-to-speech environment check could not be run: {_reason(exc)}")

    def environment(self, refresh: bool = False, probe: bool = True) -> dict:
        """The verify report plus "ready". Only a passing report is remembered, so a fixed setup shows at once.

        The full check starts a GPU process, so it never runs while a request is in progress, nor when the caller
        says not to probe (live mode or training is running). The answer is then the last report there is,
        unchanged by refresh, or one marked "unknown" when this backend has not checked yet."""
        if self._unavailable is not None:
            return unavailable_report(self._unavailable)
        with self._lock:
            quiet = not probe or self._job is not None
            if refresh and not quiet:
                self._passed = None
            report = self._passed
            if report is None and quiet:
                report = self._last_report
        if report is None:
            if quiet:
                return {"ok": False, "ready": False, "unknown": True, "checks": []}
            report = self._full_check()
        return {**report, "ready": report["ok"]}

    def _full_check(self) -> dict:
        """The full environment check, several seconds long and never under the lock. Only one runs at a time:
        it starts a GPU process, so whoever asks while one is in progress waits for it and takes its result."""
        with self._lock:
            probe, mine = self._probe, False
            if probe is None:
                probe, mine = _Probe(), True
                self._probe = probe
        if not mine:
            probe.done.wait()  # set whatever happens to the check: see the finally below
            if probe.report is not None:
                return probe.report
            return _failed_report("The text-to-speech environment check could not be run")
        try:
            report = self._check_environment(fast=False)
            with self._lock:
                self._last_report = report
                self._passed = report if report["ok"] else None
            probe.report = report
            return report
        finally:
            with self._lock:
                self._probe = None
            probe.done.set()

    def _require_environment(self) -> None:
        with self._lock:
            passed = self._passed is not None
        if not passed:
            report = self._full_check()
        else:  # once the full check has passed, each later request only repeats the quick file checks
            report = self._check_environment(fast=True)
            if not report["ok"]:
                with self._lock:
                    self._passed = None
                    self._last_report = report
        if not report["ok"]:
            failing = "; ".join(c["message"] for c in report["checks"] if not c["ok"])
            raise TtsError(f"Text to speech is not set up: {failing}")

    # ---- ending workers (always without the lock)

    def _remember_tree(self, engine: _Engine) -> None:
        """Renews the list of the worker's processes. What was seen before stays on it."""
        try:
            tree = self._tools.tree(engine.pid)
        except Exception:  # noqa: BLE001 - the last list still stands
            tree = []
        with self._lock:
            engine.tree_at = self._now()
            if tree:
                engine.tree = list(dict.fromkeys([*engine.tree, *tree]))

    def _known_processes(self, engine: _Engine) -> list:
        """The worker's remembered processes plus whatever those still alive have started since the last look."""
        known = dict.fromkeys(engine.tree)
        try:
            for proc in self._tools.survivors(list(known), 0.0):
                known.update(dict.fromkeys(self._tools.tree(proc.pid)))
        except Exception as exc:  # noqa: BLE001 - the remembered list still stands
            log.warning("Could not list the processes of the voice engine again: %s", _reason(exc))
        return list(known)

    def _reap(self, engine: _Engine) -> None:
        """Ends a worker's processes and makes sure of it; the caller has marked it as being killed. Only when
        nothing is left running is it taken off the list of workers to end; otherwise the survivors are
        remembered and the next try is due KILL_RETRY seconds later."""
        procs, failures, verified, left = list(engine.tree), [], False, []
        try:
            procs = self._known_processes(engine)
            try:
                exited = engine.handle.poll() is not None
            except Exception:  # noqa: BLE001 - not known to have ended, so it is killed
                exited = False
            if not exited:
                try:
                    engine.handle.kill_tree()
                except KillFailed as exc:
                    failures.append(exc)
                    procs = list(dict.fromkeys([*procs, *exc.survivors]))
                except Exception as exc:  # noqa: BLE001
                    failures.append(exc)
            try:
                leftover = _end_survivors(procs, self._tools)
                if leftover:
                    log.warning("Stopped processes the voice engine left running: %s", _pids(leftover))
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)
            verified = not failures
            if not verified:
                try:
                    left = [p.pid for p in self._tools.survivors(procs, 0.0)]
                except Exception:  # noqa: BLE001 - only for the message; the retry does not depend on it
                    left = []
        finally:  # also on KeyboardInterrupt or SystemExit: it must never be left marked as being killed
            with self._lock:
                engine.killing = False
                if verified:
                    if engine in self._doomed:
                        self._doomed.remove(engine)
                else:
                    engine.tree, engine.left = procs, left
                    engine.retry_at = self._now() + KILL_RETRY
                    log.warning("Could not stop the voice engine (process %s), trying again in %g seconds: %s",
                                engine.pid, KILL_RETRY, "; ".join(_reason(f) for f in failures) or "interrupted")
            engine.resting.set()
        if verified:
            engine.close()

    def _reap_doomed(self, force: bool = False) -> None:
        with self._lock:
            now = self._now()
            due = [e for e in self._doomed
                   if not e.killing and (force or e.retry_at is None or now >= e.retry_at)]
            for engine in due:
                engine.killing = True
                engine.resting.clear()
        for engine in due:
            self._reap(engine)

    def _wait_for_kills(self) -> None:
        """Waits, for at most KILL_PATIENCE seconds in all, until no other thread is in the middle of a kill."""
        deadline = time.monotonic() + KILL_PATIENCE
        with self._lock:
            in_progress = [e for e in self._doomed if e.killing]
        for engine in in_progress:
            engine.resting.wait(max(0.0, deadline - time.monotonic()))

    def _doom(self, engine: _Engine, failed: bool) -> None:
        """Lock held. From here on the worker takes no more requests and is only waiting to be killed."""
        if self._engine is engine:
            self._engine = None
        if engine not in self._doomed:
            self._doomed.append(engine)
        self._engine_failed = failed

    def _service(self) -> None:
        """What the housekeeping thread and the screen's poll keep in order: a worker that died while idle, the
        idle timeout, and kills that are due another try. The decisions are made under the lock, the kills after
        it is released."""
        with self._lock:
            engine = self._engine
            if engine is not None and self._job is None:
                try:
                    died = engine.handle.poll() is not None
                except Exception:  # noqa: BLE001
                    died = False
                if died:
                    log.warning("The voice engine (process %s) ended while idle.", engine.pid)
                    self._doom(engine, failed=True)
                elif self._now() - self._last_used >= IDLE_UNLOAD:
                    self._doom(engine, failed=False)
        self._reap_doomed()

    def _start_keeper(self) -> None:
        """Lock held, a worker is about to exist. Starts the housekeeping thread unless it is running or the
        app is closing: shutdown reads the thread under this same lock, so none can be started behind it."""
        if self._closed or (self._keeper is not None and self._keeper.is_alive()):
            return
        keeper = threading.Thread(target=self._keep, name="tts-housekeeping", daemon=True)
        try:
            keeper.start()
        except Exception as exc:  # noqa: BLE001 - the screen's poll and the next request still look after it
            log.warning("Could not start the text-to-speech housekeeping thread: %s", _reason(exc))
            return
        self._keeper = keeper

    def _keep(self) -> None:
        """The housekeeping thread: until shutdown, a round of _service about every HOUSEKEEPING seconds, and
        sooner while a kill that left something running is due another try. Kills happen inside _service,
        outside the lock, and only of workers this thread has marked as being killed under it."""
        while not self._pacer.is_set():
            try:
                with self._lock:
                    pause = KILL_RETRY if self._doomed else HOUSEKEEPING
            except Exception:  # noqa: BLE001
                pause = HOUSEKEEPING
            if self._pacer.wait(pause) or self._pacer.is_set():
                return
            try:
                self._service()
            except Exception:  # noqa: BLE001 - the thread must outlive a bad round
                self.housekeeping_failures += 1
                if self.housekeeping_failures == 1:
                    log.exception("Text-to-speech housekeeping failed")

    def _stop_keeper(self) -> None:
        """Ends the housekeeping thread and waits for it: a kill it has in progress is finished, not abandoned."""
        self._pacer.set()
        with self._lock:
            keeper = self._keeper
        if keeper is None or keeper is threading.current_thread():
            return
        keeper.join(KILL_PATIENCE)
        if keeper.is_alive():
            log.warning("The text-to-speech housekeeping thread is still ending a voice engine.")

    def _still_running(self) -> str:
        pids = sorted({pid for e in self._doomed for pid in (e.left or [e.pid])})
        return ("process " if len(pids) == 1 else "processes ") + ", ".join(str(p) for p in pids)

    # ---- the public surface

    def is_active(self) -> bool:
        """Whether a request is being prepared, synthesised, played or stopped. Only takes the manager's lock."""
        with self._lock:
            return self._job is not None

    @property
    def busy(self) -> bool:
        return self.is_active()

    def join(self, timeout: float | None = None) -> bool:
        """Waits until no request is in progress; False if one still is after timeout seconds."""
        with self._lock:
            job = self._job
        return job is None or job.done.wait(timeout)

    def _snapshot(self) -> dict:
        result = dict(self._last) if self._job is None else self._job.describe()
        if self._engine is not None:
            engine = "ready" if self._engine.ready else "starting"
        elif self._launching:
            engine = "starting"
        else:
            # "failed" is for the screen to report: a worker that died, hung or could not be started, or one
            # that could not be ended. A worker being ended as planned (the idle timeout, unload) is stopped.
            stuck = any(e.retry_at is not None for e in self._doomed)
            engine = "failed" if stuck or self._engine_failed else "stopped"
        return {"engine": engine, **result}

    def status(self) -> dict:
        """Polled twice a second. Never raises, never reads from the worker, and never kills under the lock."""
        try:
            self._service()
        except Exception:  # noqa: BLE001
            self._status_failed()
        try:
            with self._lock:
                return self._snapshot()
        except Exception:  # noqa: BLE001
            self._status_failed()
            return {"engine": "stopped", **_nothing()}

    def _status_failed(self) -> None:
        self.status_failures += 1
        if self.status_failures == 1:  # logged once: this is called twice a second
            log.exception("Could not work out the text-to-speech status")

    def _outputs(self, mic_device, monitor_device) -> dict[str, int]:
        outputs: dict[str, int] = {}
        if mic_device is not None:
            outputs["mic"] = mic_device
        if monitor_device is not None:
            if monitor_device == mic_device:
                raise TtsError("Monitor and virtual mic must be different devices")
            outputs["monitor"] = monitor_device
        if not outputs:
            raise TtsError("Choose a virtual mic or a listen device")
        devices = self._list_devices()
        for index in outputs.values():
            device = get_device(devices, index)
            if device.max_output < 1:
                raise TtsError(f"{device.name} is not an output device")
        return outputs

    def speak(self, voice_id: str, text: str, preset: str, mic_device: int | None = None,
              monitor_device: int | None = None, save: bool = False, monitor_gain: float = 1.0) -> dict:
        """Starts speaking the text in that voice and returns the status at once; status() follows it from there.
        Everything that can be refused is refused here, before anything is started. monitor_gain is the volume
        of the listen device only, as in live mode; the virtual mic always gets the speech as it is."""
        if self._unavailable is not None:
            raise TtsError(self._unavailable)
        try:
            text = tts_text.clean(text) if isinstance(text, str) else tts_text.clean("")
            # Whether anything speakable is in it does not depend on the piece size, so this needs no clip yet.
            if not tts_text.split(text, 4 * tts_text.MAX_TEXT):
                raise TtsError(NOTHING_TO_SAY)
            if preset not in STEPS:
                raise TtsError(f"Unknown preset: {preset}. Choose fast, balanced or best.")
            try:
                voice = self._library.get(voice_id)
            except LibraryError:
                raise TtsError("Unknown voice") from None
            # The first clip in order of preference that is loud enough to copy: one file is read, as a rule.
            clip = tts_reference.choose(voice, tts_reference.library_levels(self._library, voice.id))
            try:
                wav = self._library.clip_audio_path(voice.id, clip.index)
            except LibraryError:
                raise TtsError(f"The audio of clip {clip.index}, this voice's reference clip, is missing. Delete "
                               "that clip or record it again.") from None
            ref_text = _sendable(clip.text).strip()
            # The transcript as stored: the worker adds a full stop and a space, which only raises the limit.
            max_chars = tts_audio.max_chars_for(ref_text, clip.seconds, SPEED)
            # The numbers the model misreads are written out here, after the length check on the text as typed.
            pieces = tts_text.split(text, max_chars, numbers=True)
            if preset == "best" and voice.rvc_sample_rate is None:
                raise TtsError(NO_MODEL)
            outputs = self._outputs(mic_device, monitor_device)
            if not _is_number(monitor_gain) or not 0 <= monitor_gain <= MAX_MONITOR_GAIN:
                raise TtsError(f"monitor_gain must be between 0 and {MAX_MONITOR_GAIN:g}")
            # The volume is applied after every piece has been brought to PIECE_PEAK: more than this would clip.
            listen = min(float(monitor_gain), 1.0 / tts_audio.PIECE_PEAK)
            gains = {"monitor": listen} if "monitor" in outputs else {}
        except TtsError:
            raise
        except ValueError as exc:
            raise TtsError(str(exc)) from exc
        self._service()  # a worker that died while idle, or one still being ended, is dealt with first
        with self._lock:
            if self._closed:
                raise TtsError(CLOSING)
            if self._job is not None:
                raise TtsError(ALREADY)
            if self._doomed:
                raise TtsError(f"The voice engine of an earlier request is still shutting down "
                               f"({self._still_running()}). Try again in a moment.")
            job_id = "tts-" + secrets.token_hex(8)
            job = _Job(job_id, voice.id, preset, text, clip.index, wav, ref_text, max_chars, pieces, bool(save),
                       self._tmp / job_id, self._now())
            self._job = job  # holds the single slot while the checks below run outside the lock
            loaded = self._engine is not None
        try:
            if not loaded:  # a running worker is the proof; the check would cost every request half a second
                self._require_environment()
            try:
                job.session = self._player.start(outputs, gains)
            except ValueError as exc:
                raise TtsError(str(exc)) from exc
            try:
                threading.Thread(target=self._run, args=(job,), name="tts-request", daemon=True).start()
            except BaseException:
                self._quietly(self._player.stop)
                raise
        except BaseException:
            with self._lock:
                self._job = None
            job.done.set()
            raise
        return self.status()

    def stop(self) -> dict:
        """Silences the player at once, has the worker cancel the job, removes the request's files, and returns
        once the manager is idle again. The worker stays loaded. Harmless when nothing is being spoken."""
        self._stop_job()
        return self.status()

    def _stop_job(self) -> None:
        with self._lock:
            job = self._job
            if job is None:
                return
            job.stop.set()
            job.state = "stopping"
            engine = job.engine
        self._quietly(self._player.stop)  # first: what is heard ends now, whatever the worker is doing
        if engine is not None:
            engine.events.put(_WAKE)
        if not job.done.wait(STOP_WAIT):
            log.warning("The stopped text-to-speech request is still cleaning up.")

    def unload(self) -> None:
        """Stops the request in progress, if any, and the worker, so that nothing of text to speech is on the
        GPU (before training, and when the app closes). Raises if a process could not be stopped. A kill that
        another thread has in progress (the idle timeout, say) is waited for, not judged half-way."""
        self._stop_job()
        with self._lock:
            if self._engine is not None:
                self._doom(self._engine, failed=False)
        self._reap_doomed(force=True)
        self._wait_for_kills()
        self._reap_doomed(force=True)  # what the other thread's kill left running is tried once more, now
        with self._lock:
            if self._doomed:
                raise TtsError(f"The voice engine could not be stopped ({self._still_running()} is still running). "
                               "Close the app, or end it in Task Manager, before using the graphics card for "
                               "something else.")
            self._engine_failed = False
            launching = self._launching
        if launching:
            # The request was told to stop but its thread is still inside the launch. Nothing can be ended yet:
            # that thread ends the worker itself, and checks it, the moment the process exists.
            log.warning("Text to speech was unloaded while a voice engine was still being started; the engine "
                        "is stopped as soon as it exists.")

    def shutdown(self) -> None:
        """For the backend's own exit: no new request is accepted, and the request and the worker are stopped."""
        with self._lock:
            self._closed = True  # from here on no request is accepted and no housekeeping thread is started
        self._pacer.set()  # no new housekeeping round; one in the middle of a kill is waited for below
        try:
            self.unload()
        except Exception as exc:  # noqa: BLE001 - the app is closing; the worker also ends when its parent is gone
            log.warning("Text to speech did not shut down cleanly: %s", _reason(exc))
        self._stop_keeper()

    # ---- a request (its own thread)

    @staticmethod
    def _quietly(action) -> None:
        try:
            action()
        except Exception:  # noqa: BLE001
            log.exception("Text to speech could not %s", getattr(action, "__name__", "clean up"))

    def _run(self, job: _Job) -> None:
        error = None
        try:
            self._speak(job)
        except _Stopped:
            pass
        except _Failed as exc:
            error = exc.message
        except Exception as exc:  # noqa: BLE001 - whatever went wrong, the request ends and says so
            log.exception("Text to speech failed")
            error = f"Text to speech failed: {_reason(exc)}"
        finally:
            self._end(job, error)

    def _end(self, job: _Job, error: str | None) -> None:
        """Leaves nothing of the request behind: no sound, no job in the worker, no worker that is not usable,
        no files. Only then is the manager idle again."""
        try:
            interrupted = error is not None or job.stop.is_set()
            with self._lock:
                job.error = error
                if interrupted:
                    job.state = "stopping"
            if interrupted:
                self._quietly(self._player.stop)
            engine = job.engine
            if engine is not None:
                try:
                    self._settle(job, engine)
                except Exception:  # noqa: BLE001 - it may still owe an answer: it is not used again
                    log.exception("Could not bring the voice engine back to rest")
                    engine.broken = True
                if engine.broken:
                    with self._lock:
                        self._doom(engine, failed=True)
                    self._reap_doomed(force=True)
        except Exception:  # noqa: BLE001
            log.exception("Could not clean up after a text-to-speech request")
        finally:
            job.finished = []
            self._remove(job.folder)  # after the worker has let go of it, or is gone
            with self._lock:
                job.state = "idle"
                self._last = job.describe()
                self._job = None
                self._last_used = self._now()
            job.done.set()

    def _settle(self, job: _Job, engine: _Engine) -> None:
        """Has the worker end what it still owes this request: a running job is cancelled and its ending waited
        for, the answer to a reference still loading is waited for. A worker that does not get there is broken."""
        if engine.broken or job.awaiting is None:
            return
        try:
            if job.awaiting == "job":
                self._send(engine, {"cmd": "cancel", "id": job.id})
            began = self._now()
            while job.awaiting is not None:
                self._next(job, engine, began, CANCEL_TIMEOUT, "stopping", watch=False)
        except _Failed as exc:
            with self._lock:
                if job.error is None:
                    job.error = exc.message

    def _broken(self, engine: _Engine, what: str) -> _Failed:
        engine.broken = True
        return _Failed(what + self._log_tail(engine))

    def _send(self, engine: _Engine, command: dict) -> None:
        try:
            engine.send(command)
        except Exception as exc:  # noqa: BLE001 - a pipe that cannot be written is a worker that is gone
            log.warning("Could not write to the voice engine: %s", _reason(exc))
            raise self._broken(engine, "The voice engine stopped unexpectedly.") from exc

    def _watch(self, job: _Job) -> None:
        """Raises if the request was stopped or the player has failed."""
        if job.stop.is_set():
            raise _Stopped()
        try:
            error = self._player.status().get("error")
        except Exception as exc:  # noqa: BLE001
            error = f"Playback failed: {_reason(exc)}"
        if error:
            raise _Failed(str(error))

    def _next(self, job: _Job, engine: _Engine, began: float, timeout: float, doing: str, watch: bool = True) -> dict:
        """The worker's next event, having noted what it settles. Raises when the request was stopped or the
        player failed (if watch), and when the worker ended or let timeout seconds pass since began."""
        while True:
            if watch:
                self._watch(job)
            now = self._now()
            if now - began >= timeout:
                raise self._broken(engine, f"The voice engine did not answer within {timeout:g} seconds while "
                                           f"{doing}, and was stopped.")
            if engine.tree_at is None or now - engine.tree_at >= TREE_REFRESH:
                self._remember_tree(engine)
            try:
                event = engine.events.get(timeout=self._tick)
            except queue.Empty:
                continue
            if event is _WAKE:
                continue
            if event is _EOF:
                raise self._broken(engine, f"The voice engine stopped unexpectedly while {doing}.")
            kind, id = event.get("event"), event.get("id")
            if job.awaiting == "reference":
                if kind == "reference":
                    job.awaiting = None
                    seconds = event.get("seconds")
                    if _is_number(seconds) and seconds > 0:
                        engine.reference = (job.key, float(seconds))
                elif kind == "error" and id in (job.id, None):
                    job.awaiting = None  # refused: the worker has no reference now
            elif job.awaiting == "job" and id == job.id and kind in ("done", "cancelled", "error"):
                job.awaiting = None
            return event

    def _worker_error(self, engine: _Engine, event: dict) -> _Failed:
        """What an error event means for the request. Errors about the request leave the worker as it is."""
        code, index = event.get("code"), event.get("index")
        message = str(event.get("message") or "it gave no details").strip()
        if code == "piece_failed" and isinstance(index, int) and not isinstance(index, bool):
            return _Failed(NO_SOUND.format(index + 1))
        if code == "bad_request":
            return _Failed(message)
        if code == "out_of_memory":
            return _Failed(OUT_OF_MEMORY)
        # internal, or an error about the protocol itself: the worker stays up by its own rules, but it is not
        # trusted with another request.
        if not engine.ready:
            return self._broken(engine, message)  # its own account of why it could not start
        return self._broken(engine, f"The voice engine failed: {message.rstrip('.')}.")

    def _speak(self, job: _Job) -> None:
        engine = self._acquire(job)
        seconds = self._load_reference(job, engine)
        pieces = self._final_pieces(job, seconds)
        try:
            job.folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _Failed(f"Could not create the temporary folder for the speech: {_reason(exc)}") from exc
        self._watch(job)
        self._send(engine, {"cmd": "synthesize", "id": job.id, "pieces": pieces, "out_dir": str(job.folder),
                            "nfe": STEPS[job.preset], "speed": SPEED, "max_chars": job.max_chars})
        job.awaiting = "job"
        with self._lock:
            if not job.stop.is_set():
                job.state = "speaking"
        converter = self._converter(job)  # loaded while the worker is busy with the first piece
        began, received = self._now(), 0
        while True:
            event = self._next(job, engine, began, PIECE_TIMEOUT, "synthesising")
            kind, id = event.get("event"), event.get("id")
            if kind == "error" and id in (job.id, None):
                raise self._worker_error(engine, event)
            if id != job.id:
                log.warning("Ignoring an event of the voice engine that is not part of this request: %.200r", event)
                continue
            if kind == "audio":
                if event.get("index") != received:
                    raise self._broken(engine, f"The voice engine delivered its sentences out of order (sentence "
                                               f"{event.get('index')} where {received} was due).")
                # The next piece's time starts when this one is queued: a slow RVC pass does not eat into it.
                began = self._play_piece(job, received, converter, event)
                received += 1
            elif kind == "done":
                if received != len(pieces):
                    raise _Failed("The voice engine finished before every sentence was spoken")
                break
            elif kind == "cancelled":
                raise _Failed("The voice engine stopped the request by itself")
        self._player.finish()
        while not self._player.wait(self._tick):
            self._watch(job)
        self._watch(job)  # a device lost while the last piece played, or a stop at the very end
        if job.save:
            self._save(job)

    def _acquire(self, job: _Job) -> _Engine:
        """The worker for this request, started if there is none, once its model is loaded."""
        with self._lock:
            if job.stop.is_set() or self._closed:
                raise _Stopped()
            engine = self._engine
            if engine is None:
                if self._doomed:
                    raise _Failed(f"The voice engine of an earlier request is still shutting down "
                                  f"({self._still_running()}). Try again in a moment.")
                self._launching = True  # this request holds the only slot, so nobody else can be launching
                self._start_keeper()  # a worker is about to exist: from now on it is looked after unasked
            else:
                job.engine = engine
        if engine is None:
            engine = self._start_engine(job)
        self._remember_tree(engine)
        waited = False
        while not engine.ready:
            waited = True
            event = self._next(job, engine, engine.started, START_TIMEOUT, "starting")
            if event.get("event") == "error":
                raise self._worker_error(engine, event)
        while True:  # nothing is owed between requests, so whatever lies here belongs to nobody
            try:
                event = engine.events.get_nowait()
            except queue.Empty:
                break
            if event is _EOF:
                raise self._broken(engine, "The voice engine stopped unexpectedly while idle.")
        if waited:
            self._remember_tree(engine)  # the launcher's child, the real interpreter, is there by now
        return engine

    def _start_engine(self, job: _Job) -> _Engine:
        """Starts a worker for this request, without the lock: creating the folders and the process can take a
        while, and status() must not wait for it. The worker exists from the moment launch returns and is this
        manager's from that line on: if the request was stopped, or the app began closing, in the meantime, the
        worker is ended and checked like any other instead of being put to use."""
        log_path = self._root / LOG_NAME
        argv = [str(self._tts / ".venv" / "Scripts" / "python.exe"), "-u", str(self._tts / "tts_worker.py"),
                "--model-dir", str(self._tts / "models"), "--parent-pid", str(os.getpid())]
        handle = None
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            self._keep_previous_log(log_path)
            handle = self._launch(argv, log_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("Could not start the text-to-speech worker")
            raise _Failed(f"Could not start the voice engine: {_reason(exc)}") from exc
        finally:
            with self._lock:
                self._launching = False
                if handle is None:
                    self._engine_failed = True
                else:
                    engine = _Engine(handle, log_path, self._now())
                    wanted = not (job.stop.is_set() or self._closed)
                    if wanted:
                        self._engine, self._engine_failed = engine, False
                        job.engine = engine
                    else:
                        self._doom(engine, failed=False)
        try:
            threading.Thread(target=engine.read, name="tts-events", daemon=True).start()
        except Exception as exc:  # noqa: BLE001 - nobody would hear it: it does not get to run
            with self._lock:
                self._doom(engine, failed=True)
            self._remember_tree(engine)
            self._reap_doomed(force=True)
            raise _Failed(f"Could not start the voice engine: {_reason(exc)}") from exc
        if not wanted:
            self._remember_tree(engine)
            self._reap_doomed(force=True)
            raise _Stopped()
        return engine

    @staticmethod
    def _keep_previous_log(log_path: Path) -> None:
        """The last worker's output stays readable as worker.prev.log after the new worker starts its own."""
        try:
            if log_path.is_file():
                os.replace(log_path, log_path.with_name(PREVIOUS_LOG_NAME))
        except OSError as exc:
            log.warning("Could not keep the previous voice engine log: %s", _reason(exc))

    def _load_reference(self, job: _Job, engine: _Engine) -> float:
        """Has the worker load this request's reference clip unless it is the one loaded. Returns its length as
        the worker measured it."""
        loaded = engine.reference
        if loaded is not None and loaded[0] == job.key:
            return loaded[1]
        engine.reference = None
        self._watch(job)
        # The clip itself, the 48 kHz original: the worker reads it and resamples in memory. The id is not part
        # of the command, but the worker puts it on a refusal.
        self._send(engine, {"cmd": "set_reference", "id": job.id, "wav": str(job.wav), "text": job.ref_text})
        job.awaiting = "reference"
        began = self._now()
        while job.awaiting is not None:
            event = self._next(job, engine, began, PIECE_TIMEOUT, "loading the reference clip")
            if event.get("event") == "error" and event.get("id") in (job.id, None):
                if event.get("code") == "bad_request":  # the clip was refused: said in the Voices tab's terms
                    message = str(event.get("message") or "it gave no details").strip()
                    raise _Failed(_reference_refusal(message, job.wav.name, job.clip_index))
                raise self._worker_error(engine, event)
        if engine.reference is None:
            raise self._broken(engine, "The voice engine gave no usable answer about the reference clip.")
        return engine.reference[1]

    def _final_pieces(self, job: _Job, seconds: float) -> list[str]:
        """The pieces for the reference as the worker measured it; speak() split for the length on record."""
        try:
            max_chars = tts_audio.max_chars_for(job.ref_text, seconds, SPEED)
            if max_chars != job.max_chars:
                pieces = tts_text.split(job.text, max_chars, numbers=True)
                with self._lock:
                    job.max_chars, job.pieces = max_chars, pieces
        except ValueError as exc:
            raise _Failed(str(exc)) from exc
        if not job.pieces:
            raise _Failed(NOTHING_TO_SAY)
        return job.pieces

    def _default_rvc(self, voice_id: str):
        """The voice's trained model behind the engine live mode uses, with the high_quality preset, warmed up."""
        from aivoice import manager as live
        from aivoice.pretrained import PretrainedVoice
        from aivoice.tts_rvc import OfflineConverter, offline_params

        voice = self._library.get(voice_id)
        if voice.rvc_sample_rate is None:
            raise TtsError("the voice has no trained model any more")
        folder = self._data / "voices" / voice.id / "models" / "rvc"  # where pretrained.list_voices finds it
        index = folder / "model.index"
        engine = live.default_engine_factory(self._data)(PretrainedVoice(
            voice.id, voice.name, folder / "model.onnx", index if index.is_file() else None, voice.rvc_sample_rate))
        converter = OfflineConverter(engine, offline_params(), SAMPLE_RATE)
        converter.warm()
        return converter

    def _converter(self, job: _Job):
        if job.preset != "best":
            return None
        try:
            return self._rvc_factory(job.voice_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Could not load the trained model of %s", job.voice_id)
            raise _Failed(f"Could not load this voice's trained model: {_reason(exc)}") from exc

    def _play_piece(self, job: _Job, index: int, converter, event: dict) -> float:
        """Reads, checks and finishes one piece and queues it for playback. Nothing unchecked is ever queued.
        Returns the time at which it was queued."""
        number = index + 1
        # Only ever a file in the request's own folder, whatever path the event names.
        path = job.folder / Path(str(event.get("path") or "")).name
        try:
            audio, rate = read_wav(path)
            if rate <= 0:
                raise ValueError("it has no sample rate")
        except ValueError as exc:
            raise _Failed(f"Sentence {number} from the voice engine could not be read: {exc}") from exc
        # The piece continues the reference clip: what is left of the reference's last word goes first, then
        # the pause before the sentence.
        audio = tts_audio.trim_edges(tts_audio.drop_reference_tail(audio, rate), rate)
        if tts_audio.is_silent(audio, rate):
            raise _Failed(NO_SOUND.format(number))
        if converter is not None:
            try:
                if rate != SAMPLE_RATE:  # at 48 kHz, or everything the model makes above 12 kHz would be lost
                    audio = np.asarray(soxr.resample(audio, rate, SAMPLE_RATE, quality="HQ"), dtype=np.float32)
                    rate = SAMPLE_RATE
                audio = tts_audio.trim_edges(converter.convert(audio), rate)
            except Exception as exc:  # noqa: BLE001
                log.exception("Voice conversion failed")
                raise _Failed(f"Voice conversion failed for sentence {number}: {_reason(exc)}") from exc
            if tts_audio.is_silent(audio, rate):
                raise _Failed(f"Voice conversion produced no sound for sentence {number}")
        # Every piece at the same peak: the model's pieces differ by a few decibels. The gain is limited
        # (tts_audio.MAX_GAIN), because "not silent" above does not mean "speech": a piece of room tone or a
        # click, and the quiet output for a quiet reference clip, are raised by 12 dB at most, never to full scale.
        audio = tts_audio.finish(audio, rate, level=tts_audio.PIECE_PEAK)
        self._watch(job)  # stopped meanwhile: nothing more is queued
        queued = self._now()
        self._player.enqueue(audio, session=job.session)
        if job.save:
            job.finished.append(audio)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass  # the folder goes when the request ends
        with self._lock:
            job.piece = number
        return queued

    def _save(self, job: _Job) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self._now()))
        path = self._root / f"{stamp}-{job.voice_id}.wav"
        for n in range(2, SAVE_NAMES + 2):  # never over an earlier result
            if not os.path.lexists(path):
                break
            path = self._root / f"{stamp}-{job.voice_id}-{n}.wav"
        else:
            raise _Failed("The speech was played but could not be saved: too many results were saved in the same "
                          "second")
        try:
            save_wav(path, job.finished)
        except (OSError, ValueError) as exc:
            raise _Failed(f"The speech was played but could not be saved: {_reason(exc)}") from exc
        with self._lock:
            job.saved_path = str(path)
