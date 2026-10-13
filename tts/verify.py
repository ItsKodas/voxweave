"""Checks the text-to-speech environment and prints one JSON object: {"ok": bool, "checks": [{id, ok, message}]}.

Standard library only, so it runs under any Python 3.10+ (the backend's or the TTS environment's). Exit code 0 when ok.
  python tts/verify.py [--fast] [--root <repo root>]
--fast skips torch_gpu (it starts the TTS Python and takes several seconds).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

F5_VERSION = "1.1.22"
TORCH_SUFFIX = "+rocm7.2.1"
SETUP_HINT = "Run: bash tts/setup.sh"
TORCH_PROBE = "import torch; print(torch.__version__, torch.cuda.is_available())"
# Asks the installed metadata instead of importing f5_tts, which only works after the worker's shims.
F5_PROBE = 'import importlib.metadata as m; print(m.version("f5-tts"))'
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
PROBE_TIMEOUT = 180  # seconds one probe in the TTS Python may take
TASKKILL_TIMEOUT = 10
PIPE_WAIT = 5  # seconds the output of a probe that was ended is waited for
_WINDOWS = os.name == "nt"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_bounded(args, capture_output=True, text=True, timeout=None, env=None, cwd=None, popen=subprocess.Popen,
                run=subprocess.run):
    """subprocess.run for a probe, with a time limit that ends the probe's whole process tree.

    tts/.venv's python.exe is a launcher and the real interpreter is its child: subprocess.run would kill the
    launcher only, and then wait for the child's output for as long as the child lives. Here the tree is ended
    first (taskkill /T /F), then the process itself, and its output is waited for a few seconds at most.
    Output is always captured as text. Raises subprocess.TimeoutExpired like subprocess.run."""
    proc = popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                 env=env, cwd=cwd, creationflags=_NO_WINDOW)
    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException:
        if _WINDOWS:
            try:
                run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdin=subprocess.DEVNULL, capture_output=True,
                    timeout=TASKKILL_TIMEOUT, creationflags=_NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                pass  # the process itself is still ended below
        try:
            proc.kill()
        except OSError:
            pass  # already gone
        try:
            proc.communicate(timeout=PIPE_WAIT)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        raise
    return subprocess.CompletedProcess(args, proc.returncode, out, err)


def _python_in(venv: Path) -> Path:
    for rel in ("Scripts/python.exe", "bin/python"):
        if (venv / rel).is_file():
            return venv / rel
    return venv / "Scripts" / "python.exe"


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:] if text else "(no output)"


def _check_venv(tts: Path):
    py = _python_in(tts / ".venv")
    if py.is_file():
        return True, f"TTS Python found at {py}"
    return False, f"tts/.venv is missing. {SETUP_HINT}"


def _probe(tts: Path, run, code: str):
    """Runs `code` in the TTS Python; returns (completed, error message)."""
    env = dict(os.environ, ROCM_SDK_TARGET_FAMILY="custom", PYTHONSAFEPATH="1")
    try:
        done = run([str(_python_in(tts / ".venv")), "-c", code], capture_output=True, text=True,
                   timeout=PROBE_TIMEOUT, env=env, cwd=str(tts))
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"could not run the TTS Python: {exc}"
    return done, None


def _check_torch_gpu(tts: Path, run):
    if not _python_in(tts / ".venv").is_file():
        return False, f"cannot test the GPU without tts/.venv. {SETUP_HINT}"
    done, error = _probe(tts, run, TORCH_PROBE)
    if error:
        return False, error
    if done.returncode != 0:
        return False, f"importing torch in tts/.venv failed: {_tail(done.stderr)}"
    parts = (done.stdout or "").split()
    if len(parts) != 2:
        return False, f"unexpected output from the TTS Python: {_tail(done.stdout)}"
    version, available = parts
    if not version.endswith(TORCH_SUFFIX):
        return False, (f"torch {version} is installed, not the ROCm build ending {TORCH_SUFFIX}: the GPU build was "
                       f"replaced. Delete tts/.venv and re-run bash tts/setup.sh")
    if available != "True":
        return False, (f"torch {version} sees no GPU (torch.cuda.is_available() is False). Check that the AMD "
                       f"graphics driver is installed and the GPU is not in use by a driver reset")
    return True, f"torch {version}, GPU available"


def _check_f5_package(tts: Path, run):
    if not _python_in(tts / ".venv").is_file():
        return False, f"cannot look for f5-tts without tts/.venv. {SETUP_HINT}"
    done, error = _probe(tts, run, F5_PROBE)
    if error:
        return False, error
    if done.returncode != 0:
        return False, f"f5-tts is not installed in tts/.venv ({_tail(done.stderr)}). {SETUP_HINT}"
    version = (done.stdout or "").strip()
    if version != F5_VERSION:
        return False, f"f5-tts {version or 'unknown'} is installed, expected {F5_VERSION}. {SETUP_HINT}"
    return True, f"f5-tts {version} installed"


def _check_models(tts: Path):
    try:
        entries = json.loads((tts / "pretrained.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"tts/pretrained.json is unreadable: {exc}"
    bad = []
    for e in entries:
        rel = e["path"]
        f = tts / "models" / rel
        if not f.is_file():
            bad.append(f"{rel} (missing)")
        elif f.stat().st_size != e["bytes"]:
            bad.append(f"{rel} (size {f.stat().st_size}, expected {e['bytes']})")
    if bad:
        return False, f"model files wrong: {'; '.join(bad)}. {SETUP_HINT}"
    return True, f"{len(entries)} model files present"


def run_checks(root: Path, run=run_bounded, fast: bool = False) -> dict:
    tts = Path(root) / "tts"
    plan = [("venv", lambda: _check_venv(tts))]
    if not fast:
        plan.append(("torch_gpu", lambda: _check_torch_gpu(tts, run)))
    plan += [
        ("f5_package", lambda: _check_f5_package(tts, run)),
        ("models", lambda: _check_models(tts)),
    ]
    checks = []
    for check_id, fn in plan:
        try:
            ok, message = fn()
        except Exception as exc:  # a broken install must be a failed check, never a crash
            ok, message = False, f"check crashed: {exc!r}"
        checks.append({"id": check_id, "ok": bool(ok), "message": message})
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify the text-to-speech environment.")
    parser.add_argument("--fast", action="store_true", help="skip the torch_gpu check")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root")
    args = parser.parse_args(argv)
    result = run_checks(args.root, run=run_bounded, fast=args.fast)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
