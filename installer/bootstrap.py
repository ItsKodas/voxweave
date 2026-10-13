"""First-run setup of the installed Voxweave app. Standard library only.

The app runs this with the uv.exe that ships in the installer:

    uv run --no-project --python 3.12 <resources>/setup/bootstrap.py --root <root> --resources <resources>
        --app-version <version>

It builds the Python environment of the voice engine under <root>/runtime/backend/.venv from the pinned
requirements.lock, downloads the base speech model into <root>/data/pretrained, and writes
<root>/runtime/setup.json last. Running it again completes what is missing and repairs what is damaged; with
everything in place it finishes in seconds. It needs no administrator rights and writes nothing outside <root>.

Progress goes to standard output, one JSON object per line:

    {"step": "packages", "percent": 37, "message": "Installing the voice engine's packages"}
    {"done": true, "ok": false, "error": "..."}        (always the last line)

Everything uv and the checks print is appended to <root>/runtime/setup.log.
"""
import argparse
import contextlib
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PYTHON_VERSION = "3.12"


@dataclass(frozen=True)
class ModelSpec:
    url: str
    sha256: str
    size: int


# The ContentVec base speech model every voice needs (README, "Building from source"). The checksum is the one
# Hugging Face publishes for the file and the one of the copy the app was developed with.
MODEL_NAME = "vec-768-layer-12.onnx"
MODEL = ModelSpec(
    "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx",
    "b3886e7dff1495cda514f94f4680a7b1261e05d6929f5c764cdb17934b413c2a",
    377_655_729,
)

# What the package step leaves on disk (uv's cache plus the environment), measured on the first real run. Only
# used to turn "bytes on disk so far" into a rough percentage while uv works.
PACKAGES_DISK_BYTES = 864_000_000
# Refuse to start a first setup with less free space than this: Python, the packages, the model and headroom.
MIN_FREE_BYTES = 2_000_000_000

DOWNLOAD_ATTEMPTS = 5
CHUNK = 1 << 20
TIMEOUT = 60.0
LOG_LIMIT = 2_000_000

# Share of the progress bar each step ends at. The app shows 0 to PYTHON_DONE while uv fetches Python.
PYTHON_DONE, ENVIRONMENT_DONE, PACKAGES_DONE, VERIFY_DONE, MODEL_DONE = 3, 6, 60, 62, 99

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_VERSION_CHECK = "import sys; print('%d.%d' % sys.version_info[:2])"
# Run by the new environment's own Python. Importing the packages shows that each one loads on this computer
# (they bring DLLs), not only that it was unpacked.
_RUNTIME_CHECK = """
import importlib.metadata as metadata, json
def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
import onnxruntime, numpy, fastapi, uvicorn, sounddevice, soxr, pyworld, faiss, psutil
import ctranslate2, tokenizers, av, huggingface_hub, faster_whisper
print(json.dumps({"directml": version("onnxruntime-directml"), "plain": version("onnxruntime"),
                  "providers": onnxruntime.get_available_providers()}))
"""


class SetupError(Exception):
    """Setup cannot continue; the message is shown to the user."""


class Paths:
    def __init__(self, root, resources):
        self.root = Path(root).resolve()
        if self.root.parent == self.root:
            raise SetupError(f"{self.root} is a drive, not a folder for Voxweave")
        resources = Path(resources).resolve()
        self.runtime = self.root / "runtime"
        self.venv = self.runtime / "backend" / ".venv"
        self.python = self.venv / "Scripts" / "python.exe"
        self.cache = self.runtime / "uv-cache"
        self.pythons = self.runtime / "python"
        self.state = self.runtime / "setup.json"
        self.log = self.runtime / "setup.log"
        self.data = self.root / "data"
        self.model = self.data / "pretrained" / MODEL_NAME
        self.uv = resources / "bin" / "uv.exe"
        self.lock = resources / "setup" / "requirements.lock"


def uv_environment(paths: Paths, base=None) -> dict:
    """The environment for uv and for the environment's Python: everything uv keeps stays under the root, the
    user's own uv settings and Python installations play no part, and nothing is registered with Windows."""
    env = {key: value for key, value in (os.environ if base is None else base).items()
           if not key.upper().startswith("UV_") and key.upper() not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env.update({
        "UV_CACHE_DIR": str(paths.cache),
        "UV_PYTHON_INSTALL_DIR": str(paths.pythons),
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_PYTHON_INSTALL_REGISTRY": "0",
        "UV_PYTHON_INSTALL_BIN": "0",
        "UV_NO_CONFIG": "1",
        "UV_NO_PROGRESS": "1",
    })
    return env


def run_process(argv, env, tick=None, interval=1.0):
    """Runs a program to its end without a window and returns (exit code, everything it printed). tick() is
    called about every `interval` seconds while it runs."""
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
                                creationflags=_NO_WINDOW)
        try:
            while True:
                try:
                    code = proc.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    if tick is not None:
                        tick()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        out.seek(0)
        return code, out.read().decode("utf-8", errors="replace")


def open_url(url, offset):
    """Opens the download at byte `offset`. Returns (stream, start, total): start is where the server really
    begins (0 if it ignored the range) and total the size of the whole file, or None if it did not say."""
    request = urllib.request.Request(url, headers={"User-Agent": "Voxweave-setup"})
    if offset > 0:
        request.add_header("Range", f"bytes={offset}-")
    response = urllib.request.urlopen(request, timeout=TIMEOUT)  # noqa: S310 - the URL is a constant above
    length = response.headers.get("Content-Length")
    if response.status == 206:
        content_range = response.headers.get("Content-Range", "")  # bytes 1000-10239/10240
        try:
            span, total = content_range.split(" ", 1)[1].split("/")
            return response, int(span.split("-")[0]), int(total)
        except (IndexError, ValueError):
            response.close()
            raise OSError(f"the server sent an unreadable Content-Range: {content_range!r}") from None
    return response, 0, int(length) if length and length.isdigit() else None


@contextlib.contextmanager
def single_run(lock_path: Path):
    """Holds a lock file for the length of a setup, so two app windows never build one environment at once."""
    handle = open(lock_path, "a+b")
    try:
        try:
            import msvcrt
        except ImportError:  # not Windows: the app is Windows only, the tests may run anywhere
            yield
            return
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SetupError("Setup is already running in another Voxweave window. Wait for it to finish.") from None
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"the server answered {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return getattr(exc, "strerror", None) or str(exc) or type(exc).__name__


def _megabytes(count: int) -> str:
    return f"{count / 1_000_000:.0f} MB"


def _tree_bytes(folder: Path) -> int:
    total = 0
    stack = [str(folder)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class Bootstrap:
    def __init__(self, paths: Paths, version: str, model: ModelSpec = MODEL, run=run_process, fetch=open_url,
                 out=None, sleep=time.sleep, free_bytes=None, clock=time.monotonic):
        """run(argv, env, tick=None) -> (exit code, output); fetch(url, offset) -> (stream, start, total);
        free_bytes(path) -> free bytes on that path's disk. All three are replaced in the tests."""
        self._paths = paths
        self._version = version
        self._model = model
        self._run_process = run
        self._fetch = fetch
        self._out = sys.stdout if out is None else out
        self._sleep = sleep
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self._clock = clock
        self._env = uv_environment(paths)
        self._percent = 0

    # ---- output

    def _emit(self, payload: dict) -> None:
        self._out.write(json.dumps(payload) + "\n")
        self._out.flush()

    def _report(self, step: str, percent: float, message: str) -> None:
        self._percent = max(self._percent, min(100, int(percent)))  # the bar never goes backwards
        self._emit({"step": step, "percent": self._percent, "message": message})

    def _log(self, text: str) -> None:
        try:
            self._paths.runtime.mkdir(parents=True, exist_ok=True)
            if self._paths.log.is_file() and self._paths.log.stat().st_size > LOG_LIMIT:
                self._paths.log.unlink()
            with open(self._paths.log, "a", encoding="utf-8", errors="replace") as f:
                f.write(text if text.endswith("\n") else text + "\n")
        except OSError:
            pass  # a log that cannot be written never stops the setup

    def _run(self, argv: list, tick=None):
        self._log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {argv}")
        code, text = self._run_process(argv, self._env, tick)
        self._log(f"{text.rstrip()}\n[exit code {code}]")
        return code, text

    @staticmethod
    def _tail(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " | ".join(lines[-6:])[-800:] or "no output"

    # ---- the run

    def run(self) -> int:
        """Does the whole setup. Returns the exit code: 0 when everything is in place."""
        error = None
        try:
            self._paths.runtime.mkdir(parents=True, exist_ok=True)
            with single_run(self._paths.runtime / "setup.lock"):
                self._setup()
        except SetupError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - the app must get a last line whatever happens
            self._log(traceback.format_exc())
            error = f"Setup stopped with an unexpected error: {_reason(exc)}. The details are in {self._paths.log}"
        self._emit({"done": True, "ok": error is None, "error": error})
        return 0 if error is None else 1

    def _setup(self) -> None:
        paths = self._paths
        for needed in (paths.uv, paths.lock):
            if not needed.is_file():
                raise SetupError(f"A file of the app is missing: {needed}. Install Voxweave again.")
        paths.data.mkdir(parents=True, exist_ok=True)
        self._check_disk_space()
        self._ensure_environment()
        self._install_packages()
        self._verify_runtime()
        self._ensure_model()
        self._write_state()
        self._report("finish", 100, "Setup is complete")

    def _check_disk_space(self) -> None:
        if self._paths.python.is_file() and self._paths.model.is_file():
            return  # a repair or an update: the large downloads are already here
        try:
            free = self._free_bytes(self._paths.root)
        except OSError:
            return
        if free < MIN_FREE_BYTES:
            raise SetupError(f"Setup needs about {MIN_FREE_BYTES / 1e9:.0f} GB of free space on the disk that "
                             f"holds {self._paths.root}, and only {free / 1e9:.1f} GB is free. Free some space "
                             "and press Retry.")

    # ---- the Python environment

    def _environment_works(self) -> bool:
        if not self._paths.python.is_file():
            return False
        code, text = self._run([str(self._paths.python), "-c", _VERSION_CHECK])
        return code == 0 and text.strip() == PYTHON_VERSION

    def _ensure_environment(self) -> None:
        paths = self._paths
        self._report("environment", PYTHON_DONE, "Preparing the Python environment")
        if not self._environment_works():
            if paths.venv.exists():
                self._remove_tree(paths.venv)
            paths.venv.parent.mkdir(parents=True, exist_ok=True)
            code, text = self._run([str(paths.uv), "venv", "--python", PYTHON_VERSION, str(paths.venv)])
            if code != 0 or not paths.python.is_file():
                raise SetupError(f"The Python environment could not be created: {self._tail(text)}")
        self._report("environment", ENVIRONMENT_DONE, "The Python environment is ready")

    def _remove_tree(self, folder: Path) -> None:
        if self._paths.runtime not in folder.resolve().parents:
            raise SetupError(f"Refusing to remove {folder}: it is not inside {self._paths.runtime}")
        shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():
            raise SetupError(f"The damaged Python environment at {folder} could not be removed. Close Voxweave "
                             "completely (Quit in the tray menu), start it again and press Retry.")

    def _install_packages(self) -> None:
        paths = self._paths
        message = "Downloading and installing the voice engine's packages"
        self._report("packages", ENVIRONMENT_DONE, message)
        before = _tree_bytes(paths.cache) + _tree_bytes(paths.venv)

        def tick() -> None:
            grown = _tree_bytes(paths.cache) + _tree_bytes(paths.venv) - before
            share = min(0.97, max(0.0, grown / PACKAGES_DISK_BYTES))
            self._report("packages", ENVIRONMENT_DONE + share * (PACKAGES_DONE - ENVIRONMENT_DONE), message)

        # "sync" makes the environment hold exactly what the lock lists: missing packages are installed,
        # damaged ones replaced, anything else (a stray plain onnxruntime) removed. Every download is checked
        # against the hashes in the lock. Nothing is resolved, so nothing but the lock decides what is installed.
        # Compiling now keeps the first start of the voice engine short.
        code, text = self._run([str(paths.uv), "pip", "sync", "--python", str(paths.python), "--require-hashes",
                                "--compile-bytecode", str(paths.lock)], tick)
        if code != 0:
            raise SetupError("The voice engine's packages could not be installed. Check the internet connection "
                             f"and press Retry. Last output: {self._tail(text)}")
        self._report("packages", PACKAGES_DONE, "The packages are installed")

    def _verify_runtime(self) -> None:
        self._report("verify", PACKAGES_DONE, "Checking the installed packages")
        code, text = self._run([str(self._paths.python), "-c", _RUNTIME_CHECK])
        found = None
        if code == 0:
            try:
                found = json.loads(text.strip().splitlines()[-1])
            except (IndexError, ValueError):
                found = None
        if not isinstance(found, dict):
            raise SetupError("The installed packages do not load on this computer. If the message below names a "
                             "missing DLL, install the Microsoft Visual C++ Redistributable (x64) from "
                             "https://aka.ms/vs/17/release/vc_redist.x64.exe and press Retry. "
                             f"Last output: {self._tail(text)}")
        if found.get("plain") is not None:
            raise SetupError("The plain onnxruntime package is in the environment; it replaces the GPU runtime "
                             "(onnxruntime-directml) and must not be installed. Press Retry to rebuild it.")
        if found.get("directml") is None:
            raise SetupError("The GPU runtime onnxruntime-directml is missing from the environment. Press Retry.")
        if "DmlExecutionProvider" not in (found.get("providers") or []):
            raise SetupError("The installed GPU runtime does not offer DirectML, which live mode needs. "
                             "Press Retry.")
        self._report("verify", VERIFY_DONE, "The packages work")

    # ---- the base model

    def _ensure_model(self) -> None:
        spec, target = self._model, self._paths.model
        self._report("model", VERIFY_DONE, "Checking the base speech model")
        if target.is_file():
            if target.stat().st_size == spec.size and _file_sha256(target) == spec.sha256:
                self._report("model", MODEL_DONE, "The base speech model is installed")
                return
            target.unlink()  # damaged or not the pinned file
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        fresh_starts = 0
        failures = 0
        last = "no attempt was made"
        while failures < DOWNLOAD_ATTEMPTS:
            if part.is_file() and part.stat().st_size > spec.size:
                part.unlink()
            have = part.stat().st_size if part.is_file() else 0
            if have < spec.size:
                try:
                    self._download(part, have)
                except (OSError, http.client.HTTPException) as exc:
                    failures += 1
                    last = _reason(exc)
                    self._log(f"download attempt {failures} failed: {last}")
                    if failures < DOWNLOAD_ATTEMPTS:
                        self._sleep(min(30.0, 2.0 ** failures))
                    continue
                if part.stat().st_size < spec.size:
                    failures += 1
                    last = "the download ended early"
                    continue
            if part.stat().st_size == spec.size and _file_sha256(part) == spec.sha256:
                os.replace(part, target)
                self._report("model", MODEL_DONE, "The base speech model is installed")
                return
            part.unlink()
            fresh_starts += 1
            if fresh_starts > 1:
                raise SetupError("The base speech model was downloaded twice and its checksum was wrong both "
                                 "times, so it was not installed. Try again later.")
        raise SetupError(f"Could not download the base speech model: {last}. Check the internet connection and "
                         "press Retry; the download continues where it stopped.")

    def _download(self, part: Path, have: int) -> None:
        spec = self._model
        stream, start, _total = self._fetch(spec.url, have)
        with stream:
            if start not in (0, have):
                raise OSError(f"the server answered from byte {start}, not {have}")
            written = start
            reported = self._clock() - 1.0
            with open(part, "r+b" if start and part.is_file() else "wb") as f:
                f.seek(start)
                f.truncate()
                while True:
                    block = stream.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    written += len(block)
                    if written > spec.size:
                        raise OSError("the server sent more data than the model has")
                    now = self._clock()
                    if now - reported >= 0.5:
                        reported = now
                        share = written / spec.size
                        self._report("model", VERIFY_DONE + share * (MODEL_DONE - 1 - VERIFY_DONE),
                                     f"Downloading the base speech model ({_megabytes(written)} of "
                                     f"{_megabytes(spec.size)})")

    # ---- the record

    def _write_state(self) -> None:
        state = {"version": self._version, "python": PYTHON_VERSION,
                 "lock_sha256": hashlib.sha256(self._paths.lock.read_bytes()).hexdigest(),
                 "model_sha256": self._model.sha256, "completed": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = self._paths.state.with_name(self._paths.state.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self._paths.state)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="First-run setup of the installed Voxweave app")
    parser.add_argument("--root", required=True, help="the writable folder: runtime and data live under it")
    parser.add_argument("--resources", required=True, help="the folder the installer put the app's files in")
    parser.add_argument("--app-version", required=True, help="recorded in runtime/setup.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        paths = Paths(args.root, args.resources)
    except SetupError as exc:
        print(json.dumps({"done": True, "ok": False, "error": str(exc)}), flush=True)
        return 1
    return Bootstrap(paths, args.app_version).run()


if __name__ == "__main__":
    sys.exit(main())
