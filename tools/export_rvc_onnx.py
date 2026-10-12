"""Export an RVC v2 .pth voice to ONNX for the AI Voice backend.

Runs in its own venv (see README). Usage:
    python tools/export_rvc_onnx.py <voice.pth> <voice-id> [--index file.index] [--name "Display Name"]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RVC_REPO = Path(__file__).resolve().parent / ".rvc" / "repo"
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
