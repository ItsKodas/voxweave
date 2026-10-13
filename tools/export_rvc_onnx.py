"""Export an RVC v2 .pth voice to ONNX for the AI Voice backend.

One-time setup (run from the repo root; needs the patch because the stock exporter at
this tag fails and bakes a fixed 200-frame length into the graph):
    git -c core.autocrlf=false clone --depth 1 --branch 2.2.231006 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI tools/.rvc/repo
    git -C tools/.rvc/repo apply --ignore-whitespace ../../rvc_onnx_export.patch
    uv venv --python 3.10 tools/.rvc/venv
    uv pip install --python tools/.rvc/venv torch==2.6.0 numpy==1.26.4 onnx scipy --extra-index-url https://download.pytorch.org/whl/cpu

Usage:
    tools/.rvc/venv/Scripts/python tools/export_rvc_onnx.py <voice.pth> <voice-id> [--index file.index] [--name="Display Name"]

The files are written to data/pretrained/rvc/<voice-id>/, or to the folder named by the environment variable
AIVOICE_EXPORT_DIR when it is set (the training worker uses this to export into a voice's own folder).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RVC_REPO = Path(__file__).resolve().parent / ".rvc" / "repo"
PATCH = Path(__file__).resolve().parent / "rvc_onnx_export.patch"
MIN_TORCH = (2, 6)


def _require_patched_repo() -> None:
    if not RVC_REPO.is_dir():
        sys.exit(f"RVC clone missing at {RVC_REPO}. Follow the setup steps at the top of this file.")
    try:
        done = subprocess.run(
            ["git", "-C", str(RVC_REPO), "apply", "--reverse", "--check", "--ignore-whitespace", str(PATCH)],
            capture_output=True,
        )
    except FileNotFoundError:
        sys.exit("git was not found. Install Git and make sure it is on PATH; it is needed to check the RVC clone.")
    if done.returncode != 0:
        sys.exit(
            "The RVC clone is not patched. Run:\n"
            f"    git -C {RVC_REPO} apply --ignore-whitespace {PATCH}"
        )


def _require_safe_torch(version: str) -> None:
    # Compare numbers, not strings: "2.10" is newer than "2.6".
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None or (int(match.group(1)), int(match.group(2))) < MIN_TORCH:
        wanted = ".".join(map(str, MIN_TORCH))
        sys.exit(
            f"torch {version} is too old: this exporter needs torch {wanted} or newer. Older versions load "
            "model files in an unsafe way that can run code hidden in a .pth file. Reinstall with the setup "
            "steps at the top of this file."
        )


def _output_dir(voice_id: str) -> Path:
    override = os.environ.get("AIVOICE_EXPORT_DIR")
    return Path(override) if override else ROOT / "data" / "pretrained" / "rvc" / voice_id


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an RVC v2 .pth voice to ONNX for the AI Voice backend.")
    parser.add_argument("pth", type=Path)
    parser.add_argument("voice_id")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--name", help='the display name; write it as --name=<name> when it may start with "-"')
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # Checked after parsing so that --help works anywhere.
    _require_patched_repo()
    try:
        import torch
    except ImportError:
        sys.exit("torch is not installed in this Python. Run this script with tools/.rvc/venv/Scripts/python.")
    _require_safe_torch(torch.__version__)
    sys.path.insert(0, str(RVC_REPO))
    from infer.modules.onnx.export import export_onnx

    try:
        cpt = torch.load(args.pth, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Refusing to load {args.pth}: it is not a plain weights file ({exc}).")
    if cpt.get("version", "v1") != "v2":
        sys.exit("Only RVC v2 models are supported.")
    if cpt.get("f0", 1) != 1:
        sys.exit("Only pitch (f0) models are supported.")
    sample_rate = int(cpt["config"][-1])

    out = _output_dir(args.voice_id)
    out.mkdir(parents=True, exist_ok=True)
    export_onnx(str(args.pth), str(out / "model.onnx"))
    if args.index:
        shutil.copy(args.index, out / "model.index")
    (out / "voice.json").write_text(
        json.dumps({"name": args.name or args.voice_id, "sample_rate": sample_rate})
    )
    print(f"Exported {args.voice_id} at {sample_rate} Hz to {out}")


if __name__ == "__main__":
    main()
