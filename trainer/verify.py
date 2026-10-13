"""Checks the native training environment and prints one JSON object: {"ok": bool, "checks": [{id, ok, message}]}.

Standard library only, so it runs under any Python 3.10+ (the backend's or the trainer's). Exit code 0 when ok.
  python trainer/verify.py [--fast] [--root <repo root>]
--fast skips torch_gpu (it starts the trainer's Python and takes several seconds).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
TORCH_SUFFIX = "+rocm7.2.1"
PATCHES = ["rvc_rocm_dilated_conv.patch", "rvc_preprocess_keep_all_tails.patch", "rvc_train_fp32_env.patch"]
SETUP_HINT = "Run: bash trainer/setup.sh"
TORCH_PROBE = "import torch; print(torch.__version__, torch.cuda.is_available())"
# The silence samples every training filelist names (trainer/filelist.py), relative to rvc/logs/mute.
MUTE_FILES = ("0_gt_wavs/mute40k.wav", "3_feature768/mute.npy", "2a_f0/mute.wav.npy", "2b-f0nsf/mute.wav.npy")
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _python_in(venv: Path) -> Path:
    for rel in ("Scripts/python.exe", "bin/python"):
        if (venv / rel).is_file():
            return venv / rel
    return venv / "Scripts" / "python.exe"


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:] if text else "(no output)"


def _check_venv(trainer: Path):
    py = _python_in(trainer / ".venv")
    if py.is_file():
        return True, f"trainer Python found at {py}"
    return False, f"trainer/.venv is missing. {SETUP_HINT}"


def _check_torch_gpu(trainer: Path, run):
    py = _python_in(trainer / ".venv")
    if not py.is_file():
        return False, f"cannot test the GPU without trainer/.venv. {SETUP_HINT}"
    env = dict(os.environ, ROCM_SDK_TARGET_FAMILY="custom", PYTHONSAFEPATH="1")
    try:
        done = run([str(py), "-c", TORCH_PROBE], capture_output=True, text=True, timeout=180, env=env, cwd=str(trainer))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run the trainer Python: {exc}"
    if done.returncode != 0:
        return False, f"importing torch in trainer/.venv failed: {_tail(done.stderr)}"
    parts = (done.stdout or "").split()
    if len(parts) != 2:
        return False, f"unexpected output from the trainer Python: {_tail(done.stdout)}"
    version, available = parts
    if not version.endswith(TORCH_SUFFIX):
        return False, (f"torch {version} is installed, not the ROCm build ending {TORCH_SUFFIX}: the GPU build was "
                       f"replaced. Delete trainer/.venv and re-run bash trainer/setup.sh")
    if available != "True":
        return False, (f"torch {version} sees no GPU (torch.cuda.is_available() is False). Check that the AMD "
                       f"graphics driver is installed and the GPU is not in use by a driver reset")
    return True, f"torch {version}, GPU available"


def _check_rvc_commit(trainer: Path, run):
    try:
        done = run(["git", "-C", str(trainer / "rvc"), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run git: {exc}"
    head = (done.stdout or "").strip()
    if done.returncode != 0:
        return False, f"trainer/rvc is not a git clone. {SETUP_HINT}"
    if head != PINNED_COMMIT:
        return False, f"trainer/rvc is at {head or 'unknown'}, expected {PINNED_COMMIT}. {SETUP_HINT}"
    return True, f"trainer/rvc is at {PINNED_COMMIT}"


def _check_rvc_patches(trainer: Path, run):
    bad = []
    for name in PATCHES:
        patch = trainer / "patches" / name
        try:
            done = run(["git", "-C", str(trainer / "rvc"), "apply", "--reverse", "--check", str(patch)],
                       capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"could not run git: {exc}"
        if done.returncode != 0:
            bad.append(name)
    if bad:
        return False, f"patch not applied to trainer/rvc: {', '.join(bad)}. {SETUP_HINT}"
    return True, "all patches are applied"


def _check_pretrained(trainer: Path):
    try:
        entries = json.loads((trainer / "pretrained.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"trainer/pretrained.json is unreadable: {exc}"
    rvc = trainer / "rvc"
    bad = []
    for e in entries:
        rel = e["path"]
        if rel.startswith("../"):
            continue  # the mute.zip download; what matters is that it was extracted (checked below)
        f = rvc / rel
        if not f.is_file():
            bad.append(f"{rel} (missing)")
        elif f.stat().st_size != e["bytes"]:
            bad.append(f"{rel} (size {f.stat().st_size}, expected {e['bytes']})")
    for rel in MUTE_FILES:
        if not (rvc / "logs" / "mute" / rel).is_file():
            bad.append(f"logs/mute/{rel} (silence sample missing)")
    if bad:
        return False, f"pretrained files wrong: {'; '.join(bad)}. {SETUP_HINT}"
    return True, f"{len(entries)} pretrained downloads present"


def _check_ffmpeg(which):
    missing = [n for n in ("ffmpeg", "ffprobe") if not which(n)]
    if missing:
        return False, f"{' and '.join(missing)} not found on PATH (RVC preprocessing needs both)"
    return True, "ffmpeg and ffprobe found"


def _check_exporter(root: Path):
    venv = _python_in(root / "tools" / ".rvc" / "venv")
    repo = root / "tools" / ".rvc" / "repo"
    if not venv.is_file() or not repo.is_dir():
        return False, "the ONNX exporter environment tools/.rvc (venv and repo) is missing; see the header of tools/export_rvc_onnx.py"
    return True, "tools/.rvc exporter environment found"


def run_checks(root: Path, run=subprocess.run, which=shutil.which, fast: bool = False) -> dict:
    root = Path(root)
    trainer = root / "trainer"
    plan = [("venv", lambda: _check_venv(trainer))]
    if not fast:
        plan.append(("torch_gpu", lambda: _check_torch_gpu(trainer, run)))
    plan += [
        ("rvc_commit", lambda: _check_rvc_commit(trainer, run)),
        ("rvc_patches", lambda: _check_rvc_patches(trainer, run)),
        ("pretrained", lambda: _check_pretrained(trainer)),
        ("ffmpeg", lambda: _check_ffmpeg(which)),
        ("exporter", lambda: _check_exporter(root)),
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
    parser = argparse.ArgumentParser(description="Verify the native training environment.")
    parser.add_argument("--fast", action="store_true", help="skip the torch_gpu check")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root")
    args = parser.parse_args(argv)
    result = run_checks(args.root, run=subprocess.run, which=shutil.which, fast=args.fast)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
