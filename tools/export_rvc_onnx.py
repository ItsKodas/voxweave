"""Export an RVC v2 .pth voice to ONNX for the AI Voice backend.

One-time setup (run from the repo root; needs the patch because the stock exporter at
this tag fails and bakes a fixed 200-frame length into the graph):
    git -c core.autocrlf=false clone --depth 1 --branch 2.2.231006 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI tools/.rvc/repo
    git -C tools/.rvc/repo apply --ignore-whitespace ../../rvc_onnx_export.patch
    uv venv --python 3.10 tools/.rvc/venv
    uv pip install --python tools/.rvc/venv torch==2.6.0 numpy==1.26.4 onnx scipy --extra-index-url https://download.pytorch.org/whl/cpu

Usage:
    tools/.rvc/venv/Scripts/python tools/export_rvc_onnx.py <voice.pth> <voice-id> [--index file.index] [--name "Display Name"]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RVC_REPO = Path(__file__).resolve().parent / ".rvc" / "repo"
PATCH = Path(__file__).resolve().parent / "rvc_onnx_export.patch"


def _require_patched_repo() -> None:
    if not RVC_REPO.is_dir():
        sys.exit(f"RVC clone missing at {RVC_REPO}. Follow the setup steps at the top of this file.")
    done = subprocess.run(
        ["git", "-C", str(RVC_REPO), "apply", "--reverse", "--check", "--ignore-whitespace", str(PATCH)],
        capture_output=True,
    )
    if done.returncode != 0:
        sys.exit(
            "The RVC clone is not patched. Run:\n"
            f"    git -C {RVC_REPO} apply --ignore-whitespace {PATCH}"
        )


_require_patched_repo()
sys.path.insert(0, str(RVC_REPO))

import torch  # noqa: E402
from infer.modules.onnx.export import export_onnx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pth", type=Path)
    parser.add_argument("voice_id")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()

    try:
        cpt = torch.load(args.pth, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Refusing to load {args.pth}: it is not a plain weights file ({exc}).")
    if cpt.get("version", "v1") != "v2":
        sys.exit("Only RVC v2 models are supported.")
    if cpt.get("f0", 1) != 1:
        sys.exit("Only pitch (f0) models are supported.")
    sample_rate = int(cpt["config"][-1])

    out = ROOT / "data" / "pretrained" / "rvc" / args.voice_id
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
