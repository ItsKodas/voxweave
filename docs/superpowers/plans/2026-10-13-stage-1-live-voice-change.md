# Stage 1: Live Voice Change Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Tauri desktop app that converts the live microphone into a chosen RVC voice in real time and sends it to a virtual mic and an optional headphone monitor, with speed/quality presets.

**Architecture:** A Tauri 2 shell (Rust + React/TypeScript) spawns and supervises a Python FastAPI backend on a localhost port. The backend owns all audio I/O (sounddevice, WASAPI) and runs RVC inference through ONNX Runtime on DirectML. A pure, thread-free `StreamProcessor` does chunking and crossfading; a thin `LiveSession` connects it to device streams.

**Tech Stack:** Python 3.12 (managed by `uv`), FastAPI, uvicorn, onnxruntime-directml, sounddevice, soxr, pyworld-prebuilt, faiss-cpu, pytest. Tauri 2, Rust, React 18, TypeScript, Vite, Vitest.

**Spec:** `docs/superpowers/specs/2026-10-13-ai-voice-design.md`

## Global Constraints

- Windows 11, AMD Radeon RX 7900 XTX. No CUDA. Windows inference uses ONNX Runtime with `DmlExecutionProvider`.
- The machine has two GPUs (integrated Radeon and the RX 7900 XTX). The DirectML `device_id` must be chosen explicitly and is read from `data/settings.json`.
- Backend Python is 3.12, created with `uv venv --python 3.12`. The system Python 3.10 has no current `onnxruntime-directml` wheel. Do not use it for the backend.
- All pipeline audio is mono float32 at 48000 Hz. Constant: `aivoice.config.SAMPLE_RATE = 48000`.
- **Fail closed:** the virtual mic never carries unconverted microphone audio. Any failure emits silence.
- The backend binds to `127.0.0.1` only.
- VoiceMeeter is already installed and is the virtual mic. Do not install or bundle an audio driver.
- `data/` and `spike/` are gitignored. Model files are never committed.
- Stage 1 excludes: recording, import, training, TTS, the `voices` manifest, WSL2 and ROCm checks.
- Run all commands from the repo root `D:\Projects\AI Voice` unless a step says otherwise. Backend tests: `backend/.venv/Scripts/python -m pytest backend/tests -v`.

## Review Focus

Failure modes the spec implies but does not spell out. Each has a test in the task named.

1. **Device callbacks deliver blocks of arbitrary size, not multiples of the chunk.** No sample may be dropped or duplicated. Test in Task 8.
2. **Input device is stereo or not 48 kHz.** Audio is downmixed to mono and streams open with WASAPI auto-convert. Tests in Task 5 and Task 8.
3. **Dead-silent input, such as a hardware-muted mic.** Pitch extraction returns all-unvoiced; output must be finite with no NaN. Test in Task 7.
4. **Bad selections on Start:** same device for virtual mic and monitor, an output-only device chosen as input, an unknown voice. Each returns HTTP 400 with a readable message and opens no stream. Tests in Task 9 and Task 10.
5. **Repeated or out-of-order control actions:** Start pressed twice, Stop when stopped, mode change when stopped. No crash and no leaked streams. Test in Task 9.

## File Structure

```
.gitignore
README.md
tools/export_rvc_onnx.py        one-off .pth to ONNX exporter (runs in its own venv)
spike/bench.py                  throwaway benchmark (gitignored)
backend/
  pyproject.toml
  aivoice/
    __init__.py
    __main__.py                 CLI entry: --port, --parent-pid
    config.py                   SAMPLE_RATE, data_dir(), load_settings()
    presets.py                  LiveParams, PRESETS, resolve()
    ringbuffer.py               RingBuffer
    router.py                   Router: fan-out with per-output gain
    devices.py                  Device, list_devices(), find_virtual_mic(), SoundDeviceStreams
    processor.py                StreamProcessor: chunk, convert, crossfade, fail closed
    live.py                     LiveSession: streams + worker thread + stats
    pretrained.py               PretrainedVoice, list_voices(), vec_model_path()
    setup_check.py              Check, run_checks()
    manager.py                  StartConfig, LiveManager
    api.py                      create_app(): HTTP + WebSocket routes
    engines/
      __init__.py
      base.py                   VoiceEngine protocol
      f0.py                     extract_f0(), shift_f0(), to_coarse()
      rvc.py                    make_session(), blend_index(), RvcEngine
  tests/                        one test file per module
app/                            Tauri project
  src/api.ts                    typed backend client
  src/useStats.ts               WebSocket stats hook
  src/SetupCheck.tsx
  src/LivePanel.tsx
  src/App.tsx
  src-tauri/src/lib.rs          spawn + supervise backend, tray, hotkey
data/                           runtime data (gitignored)
  settings.json                 {"dml_device_id": N}
  pretrained/vec-768-layer-12.onnx
  pretrained/rvc/<voice-id>/model.onnx, voice.json, model.index (optional)
```

Note on the spec's engine interface: the spec writes `convert(chunk, voice, params)`. Here the voice is bound when the engine is constructed, so the method is `convert(audio, sample_rate, pitch_shift, params)`. Swapping engines is unaffected.

---

### Task 1: Benchmark spike and model tooling

This task is a gate. It proves RVC runs fast enough on DirectML on this GPU before any app code exists. `spike/` is throwaway. `tools/export_rvc_onnx.py` is kept.

**Prerequisite from the human partner:** one RVC **v2** voice model `.pth` (with pitch, the normal kind) that they have the right to use, and its `.index` file if they have one. Ask for the file paths before starting if not already supplied.

**Files:**
- Create: `.gitignore`
- Create: `tools/export_rvc_onnx.py`
- Create: `spike/bench.py` (not committed)
- Create: `docs/superpowers/notes/2026-10-13-directml-benchmark.md`

**Interfaces:**
- Produces: `data/pretrained/vec-768-layer-12.onnx`; `data/pretrained/rvc/<voice-id>/model.onnx` with inputs `phone` (1,T,768 float32), `phone_lengths` (1 int64), `pitch` (1,T int64), `pitchf` (1,T float32), `ds` (1 int64), `rnd` (1,192,T float32) and output `audio`; `data/pretrained/rvc/<voice-id>/voice.json` = `{"name": str, "sample_rate": int}`; `data/settings.json` = `{"dml_device_id": int}`.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
data/
spike/
tools/.rvc/
backend/.venv/
__pycache__/
*.pyc
.pytest_cache/
node_modules/
app/dist/
app/src-tauri/target/
app/src-tauri/gen/
```

- [ ] **Step 2: Create the backend venv with the runtime libraries**

```bash
uv venv --python 3.12 backend/.venv
uv pip install --python backend/.venv onnxruntime-directml numpy soxr pyworld-prebuilt faiss-cpu sounddevice
```

Expected: both commands succeed. `pyworld-prebuilt` is used because `pyworld` has no Windows wheel for Python 3.12; it installs the same `pyworld` import name.

- [ ] **Step 3: Download the ContentVec model**

```bash
mkdir -p data/pretrained
curl -L -o data/pretrained/vec-768-layer-12.onnx https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx
```

Expected: a file of 377,655,729 bytes.

- [ ] **Step 4: Write the exporter `tools/export_rvc_onnx.py`**

The RVC project removed its ONNX exporter from `main`. It exists in tag `2.2.231006`.

**Security:** a `.pth` file is a Python pickle, and loading one the old way can run arbitrary code hidden in it. The exporter therefore pins PyTorch 2.6.0, where `torch.load` defaults to the safe `weights_only=True`. That covers the RVC exporter's own internal `torch.load` call too. A genuine RVC checkpoint holds only tensors, lists, strings and numbers, so it loads fine. If loading fails with a `weights_only` error, the file contains something else: do not work around it, reject the file.

```python
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
```

- [ ] **Step 5: Create the export venv and export the supplied voice**

```bash
git clone --depth 1 --branch 2.2.231006 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI tools/.rvc/repo
uv venv --python 3.10 tools/.rvc/venv
uv pip install --python tools/.rvc/venv torch==2.6.0 numpy==1.26.4 onnx --extra-index-url https://download.pytorch.org/whl/cpu
tools/.rvc/venv/Scripts/python tools/export_rvc_onnx.py "<path to voice.pth>" demo --index "<path to .index>" --name "Demo"
```

Omit `--index` if there is no index file. Never set `weights_only=False` or downgrade PyTorch to make a file load. Expected: `Exported demo at 40000 Hz to ...` (or 48000). If the import of `infer.modules.onnx.export` fails on a missing module, install that module into `tools/.rvc/venv` and rerun.

- [ ] **Step 6: Write `spike/bench.py`**

Window lengths match the three presets (context + chunk + crossfade in ms): 550, 1100, 2080.

```python
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort
import pyworld as pw

vec_path, model_path = sys.argv[1], sys.argv[2]
WINDOWS_MS = {"low_latency": (550, 120), "balanced": (1100, 250), "high_quality": (2080, 500)}
print("providers:", ort.get_available_providers())


def session(path, device_id):
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        path, so, providers=[("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    )


def median_ms(fn, warm=3, runs=20):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


for device_id in (0, 1):
    try:
        vec, net = session(vec_path, device_id), session(model_path, device_id)
    except Exception as exc:  # noqa: BLE001
        print(f"device {device_id}: failed to load: {exc}")
        continue
    print(f"device {device_id}: vec in={[(i.name, i.shape) for i in vec.get_inputs()]}")
    print(f"device {device_id}: net in={[(i.name, i.shape, i.type) for i in net.get_inputs()]}")
    for name, (window_ms, chunk_ms) in WINDOWS_MS.items():
        wav = (np.random.randn(16 * window_ms) * 0.1).astype(np.float32)
        feats = vec.run(None, {vec.get_inputs()[0].name: wav[None, None, :]})[0]
        phone = np.repeat(feats, 2, axis=1).astype(np.float32)
        frames = phone.shape[1]
        feed = {
            "phone": phone,
            "phone_lengths": np.array([frames], dtype=np.int64),
            "pitch": np.full((1, frames), 100, dtype=np.int64),
            "pitchf": np.full((1, frames), 220.0, dtype=np.float32),
            "ds": np.array([0], dtype=np.int64),
            "rnd": np.random.randn(1, 192, frames).astype(np.float32),
        }
        vec_ms = median_ms(lambda: vec.run(None, {vec.get_inputs()[0].name: wav[None, None, :]}))
        net_ms = median_ms(lambda: net.run(None, feed))
        x = wav.astype(np.float64)
        dio_ms = median_ms(lambda: pw.stonemask(x, *pw.dio(x, 16000, f0_floor=50, f0_ceil=1100, frame_period=10), 16000), runs=10)
        harvest_ms = median_ms(lambda: pw.harvest(x, 16000, f0_floor=50, f0_ceil=1100, frame_period=10), warm=1, runs=5)
        print(
            f"device {device_id} {name}: feats={feats.shape} vec={vec_ms:.1f} net={net_ms:.1f} "
            f"dio={dio_ms:.1f} harvest={harvest_ms:.1f} budget={chunk_ms / 2:.0f} ms"
        )
```

- [ ] **Step 7: Run the benchmark**

```bash
backend/.venv/Scripts/python spike/bench.py data/pretrained/vec-768-layer-12.onnx data/pretrained/rvc/demo/model.onnx
```

Expected: `providers` includes `DmlExecutionProvider`; the `vec` output `feats` has shape `(1, frames, 768)`; the `net` input names are exactly `phone, phone_lengths, pitch, pitchf, ds, rnd`. One `device_id` is much faster than the other; that one is the RX 7900 XTX.

- [ ] **Step 8: Apply the decision gate**

For the faster device, a preset passes when `vec + net + f0` is under its `budget` (half the chunk length), using `dio` for low_latency and balanced and `harvest` for high_quality.

- **balanced passes:** continue. Write `data/settings.json` as `{"dml_device_id": <faster id>}`.
- **balanced fails:** STOP. Report the numbers to the human partner. The spec's fallbacks (ROCm-for-Windows PyTorch, CPU pitch extractor only) need their decision.
- **Only low_latency or high_quality fails:** continue. Record it; Task 13 retunes that preset's numbers.
- **`feats` shape or input names differ from the expected values above:** record the real ones. Task 7's engine code must be adjusted to match before its real-model test will pass.

- [ ] **Step 9: Record results**

Write `docs/superpowers/notes/2026-10-13-directml-benchmark.md` containing the full benchmark output, the chosen `dml_device_id`, the model sample rate, and the pass/fail result per preset.

- [ ] **Step 10: Commit**

```bash
git add .gitignore tools/export_rvc_onnx.py docs/superpowers/notes/2026-10-13-directml-benchmark.md
git commit -m "chore: add RVC ONNX exporter and DirectML benchmark results"
```

---

### Task 2: Backend skeleton

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/aivoice/__init__.py` (empty)
- Create: `backend/aivoice/engines/__init__.py` (empty)
- Create: `backend/aivoice/config.py`
- Create: `backend/aivoice/api.py`
- Create: `backend/aivoice/__main__.py`
- Test: `backend/tests/test_config.py`, `backend/tests/test_health.py`

**Interfaces:**
- Produces: `config.SAMPLE_RATE: int = 48000`; `config.data_dir() -> Path` (honours env `AIVOICE_DATA`, else `<repo>/data`, creates it); `config.load_settings(data: Path) -> dict` returning `{"dml_device_id": int}` with default `0`; `api.create_app() -> FastAPI` with `GET /health -> {"status": "ok"}`.

- [ ] **Step 1: Write `backend/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "aivoice"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "numpy",
  "onnxruntime-directml",
  "sounddevice>=0.4.6",
  "soxr",
  "pyworld-prebuilt",
  "faiss-cpu",
  "psutil",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]

[tool.setuptools.packages.find]
include = ["aivoice*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install the package**

```bash
uv pip install --python backend/.venv -e "backend[dev]"
```

Expected: success.

- [ ] **Step 3: Write the failing tests**

`backend/tests/test_config.py`:

```python
import json

from aivoice import config


def test_sample_rate():
    assert config.SAMPLE_RATE == 48000


def test_data_dir_uses_env_and_creates(tmp_path, monkeypatch):
    target = tmp_path / "d"
    monkeypatch.setenv("AIVOICE_DATA", str(target))
    assert config.data_dir() == target
    assert target.is_dir()


def test_settings_default(tmp_path):
    assert config.load_settings(tmp_path) == {"dml_device_id": 0}


def test_settings_read(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"dml_device_id": 1}))
    assert config.load_settings(tmp_path)["dml_device_id"] == 1


def test_settings_corrupt_falls_back(tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert config.load_settings(tmp_path) == {"dml_device_id": 0}
```

`backend/tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from aivoice.api import create_app


def test_health():
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `backend/.venv/Scripts/python -m pytest backend/tests -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError` for `aivoice.config` / `aivoice.api`.

- [ ] **Step 5: Implement**

`backend/aivoice/config.py`:

```python
import json
import os
from pathlib import Path

SAMPLE_RATE = 48000
_DEFAULTS = {"dml_device_id": 0}


def data_dir() -> Path:
    root = os.environ.get("AIVOICE_DATA")
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_settings(data: Path) -> dict:
    settings = dict(_DEFAULTS)
    try:
        loaded = json.loads((data / "settings.json").read_text())
        if isinstance(loaded, dict):
            settings.update(loaded)
    except (OSError, ValueError):
        pass
    return settings
```

`backend/aivoice/api.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="AI Voice")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
```

`backend/aivoice/__main__.py`:

```python
import argparse
import os
import threading
import time

import psutil
import uvicorn

from aivoice.api import create_app


def _exit_when_parent_dies(pid: int) -> None:
    while psutil.pid_exists(pid):
        time.sleep(1.0)
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.parent_pid:
        threading.Thread(target=_exit_when_parent_dies, args=(args.parent_pid,), daemon=True).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add backend
git commit -m "feat: backend skeleton with config and health endpoint"
```

---

### Task 3: Presets

**Files:**
- Create: `backend/aivoice/presets.py`
- Test: `backend/tests/test_presets.py`

**Interfaces:**
- Produces: `LiveParams(chunk_ms: int, crossfade_ms: int, context_ms: int, f0_method: str, index_rate: float)` (frozen dataclass); `PRESETS: dict[str, LiveParams]` with keys `low_latency`, `balanced`, `high_quality`; `resolve(preset: str, overrides: dict | None = None) -> LiveParams` raising `ValueError` on any invalid value.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from aivoice.presets import PRESETS, LiveParams, resolve


def test_three_presets_exist():
    assert set(PRESETS) == {"low_latency", "balanced", "high_quality"}


def test_presets_order_by_chunk_size():
    assert PRESETS["low_latency"].chunk_ms < PRESETS["balanced"].chunk_ms < PRESETS["high_quality"].chunk_ms


def test_resolve_returns_preset():
    assert resolve("balanced") == PRESETS["balanced"]


def test_resolve_applies_overrides():
    params = resolve("balanced", {"chunk_ms": 300, "index_rate": 0.5})
    assert params.chunk_ms == 300 and params.index_rate == 0.5
    assert params.crossfade_ms == PRESETS["balanced"].crossfade_ms


def test_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve("nope")


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_ms": 10},
        {"chunk_ms": 5000},
        {"crossfade_ms": 2},
        {"chunk_ms": 100, "crossfade_ms": 200},
        {"context_ms": -1},
        {"context_ms": 9000},
        {"f0_method": "magic"},
        {"index_rate": 1.5},
        {"index_rate": -0.1},
        {"bogus": 1},
    ],
)
def test_invalid_overrides_rejected(overrides):
    with pytest.raises(ValueError):
        resolve("balanced", overrides)


def test_all_presets_are_valid():
    for name in PRESETS:
        assert isinstance(resolve(name), LiveParams)
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_presets.py -v`
Expected: FAIL with `ModuleNotFoundError: aivoice.presets`.

- [ ] **Step 3: Implement `backend/aivoice/presets.py`**

```python
from dataclasses import dataclass, fields, replace

F0_METHODS = ("dio", "harvest")


@dataclass(frozen=True)
class LiveParams:
    chunk_ms: int
    crossfade_ms: int
    context_ms: int
    f0_method: str
    index_rate: float


PRESETS: dict[str, LiveParams] = {
    "low_latency": LiveParams(120, 30, 400, "dio", 0.0),
    "balanced": LiveParams(250, 50, 800, "dio", 0.3),
    "high_quality": LiveParams(500, 80, 1500, "harvest", 0.6),
}


def _validate(p: LiveParams) -> None:
    if not 40 <= p.chunk_ms <= 2000:
        raise ValueError("chunk_ms must be between 40 and 2000")
    if not 5 <= p.crossfade_ms <= p.chunk_ms:
        raise ValueError("crossfade_ms must be between 5 and chunk_ms")
    if not 0 <= p.context_ms <= 5000:
        raise ValueError("context_ms must be between 0 and 5000")
    if p.f0_method not in F0_METHODS:
        raise ValueError(f"f0_method must be one of {F0_METHODS}")
    if not 0.0 <= p.index_rate <= 1.0:
        raise ValueError("index_rate must be between 0 and 1")


def resolve(preset: str, overrides: dict | None = None) -> LiveParams:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    params = PRESETS[preset]
    if overrides:
        known = {f.name for f in fields(LiveParams)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        params = replace(params, **overrides)
    _validate(params)
    return params
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_presets.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/presets.py backend/tests/test_presets.py
git commit -m "feat: speed/quality presets with validated overrides"
```

---

### Task 4: Ring buffer and router

**Files:**
- Create: `backend/aivoice/ringbuffer.py`, `backend/aivoice/router.py`
- Test: `backend/tests/test_ringbuffer.py`, `backend/tests/test_router.py`

**Interfaces:**
- Produces: `RingBuffer(capacity: int)` with `write(data: np.ndarray) -> None`, `read(n: int) -> np.ndarray` (always length `n`, zero-padded when short), property `available: int`, counters `underruns: int`, `overflows: int`. `Router(capacity: int)` with `add_output(name: str, gain: float = 1.0) -> None`, `set_gain(name, gain)`, `set_enabled(name, enabled: bool)`, `push(audio: np.ndarray) -> None`, `read(name: str, n: int) -> np.ndarray`, `buffered(name: str) -> int`, `underruns() -> int`, `has(name: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_ringbuffer.py`:

```python
import numpy as np

from aivoice.ringbuffer import RingBuffer


def test_fifo_order():
    rb = RingBuffer(10)
    rb.write(np.arange(4, dtype=np.float32))
    assert rb.available == 4
    assert rb.read(4).tolist() == [0, 1, 2, 3]
    assert rb.available == 0


def test_wraparound():
    rb = RingBuffer(5)
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    rb.read(2)
    rb.write(np.array([4, 5, 6], dtype=np.float32))
    assert rb.read(4).tolist() == [3, 4, 5, 6]


def test_short_read_pads_with_silence_and_counts_underrun():
    rb = RingBuffer(10)
    rb.write(np.ones(2, dtype=np.float32))
    assert rb.read(5).tolist() == [1, 1, 0, 0, 0]
    assert rb.underruns == 1


def test_no_underrun_counted_before_first_write():
    rb = RingBuffer(10)
    assert rb.read(4).tolist() == [0, 0, 0, 0]
    assert rb.underruns == 0


def test_overflow_drops_oldest():
    rb = RingBuffer(4)
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    rb.write(np.array([4, 5, 6], dtype=np.float32))
    assert rb.overflows == 1
    assert rb.read(4).tolist() == [3, 4, 5, 6]


def test_write_larger_than_capacity_keeps_newest():
    rb = RingBuffer(3)
    rb.write(np.arange(10, dtype=np.float32))
    assert rb.read(3).tolist() == [7, 8, 9]
```

`backend/tests/test_router.py`:

```python
import numpy as np
import pytest

from aivoice.router import Router


def test_fans_out_with_gain():
    r = Router(100)
    r.add_output("mic")
    r.add_output("monitor", gain=0.5)
    r.push(np.ones(4, dtype=np.float32))
    assert r.read("mic", 4).tolist() == [1, 1, 1, 1]
    assert r.read("monitor", 4).tolist() == [0.5, 0.5, 0.5, 0.5]


def test_disabled_output_gets_silence_and_stays_in_sync():
    r = Router(100)
    r.add_output("monitor")
    r.set_enabled("monitor", False)
    r.push(np.ones(4, dtype=np.float32))
    assert r.buffered("monitor") == 4
    assert r.read("monitor", 4).tolist() == [0, 0, 0, 0]


def test_set_gain():
    r = Router(100)
    r.add_output("monitor")
    r.set_gain("monitor", 0.25)
    r.push(np.ones(2, dtype=np.float32))
    assert r.read("monitor", 2).tolist() == [0.25, 0.25]


def test_has_and_unknown_output():
    r = Router(100)
    r.add_output("mic")
    assert r.has("mic") and not r.has("monitor")
    with pytest.raises(KeyError):
        r.read("monitor", 1)


def test_underruns_summed():
    r = Router(100)
    r.add_output("mic")
    r.push(np.ones(1, dtype=np.float32))
    r.read("mic", 5)
    assert r.underruns() == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_ringbuffer.py backend/tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`backend/aivoice/ringbuffer.py`:

```python
import threading

import numpy as np


class RingBuffer:
    """Thread-safe mono float32 FIFO. Reads never block: a short read is padded with silence."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._read = 0
        self._size = 0
        self._primed = False
        self._lock = threading.Lock()
        self.underruns = 0
        self.overflows = 0

    @property
    def available(self) -> int:
        with self._lock:
            return self._size

    def write(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32).ravel()
        with self._lock:
            if len(data) > self._cap:
                data = data[-self._cap :]
                self.overflows += 1
            excess = self._size + len(data) - self._cap
            if excess > 0:
                self._read = (self._read + excess) % self._cap
                self._size -= excess
                self.overflows += 1
            start = (self._read + self._size) % self._cap
            first = min(len(data), self._cap - start)
            self._buf[start : start + first] = data[:first]
            self._buf[: len(data) - first] = data[first:]
            self._size += len(data)
            self._primed = True

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            take = min(n, self._size)
            first = min(take, self._cap - self._read)
            out[:first] = self._buf[self._read : self._read + first]
            out[first:take] = self._buf[: take - first]
            self._read = (self._read + take) % self._cap
            self._size -= take
            if take < n and self._primed:
                self.underruns += 1
        return out
```

Note: `test_write_larger_than_capacity_keeps_newest` and `test_overflow_drops_oldest` each expect `overflows` to rise by one per write, which the two branches above satisfy because a truncated write into an empty buffer has `excess == 0`.

`backend/aivoice/router.py`:

```python
from dataclasses import dataclass

import numpy as np

from aivoice.ringbuffer import RingBuffer


@dataclass
class _Output:
    buffer: RingBuffer
    gain: float
    enabled: bool = True


class Router:
    """Fans one mono stream out to named outputs, each with its own buffer and gain."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._outputs: dict[str, _Output] = {}

    def add_output(self, name: str, gain: float = 1.0) -> None:
        self._outputs[name] = _Output(RingBuffer(self._capacity), gain)

    def has(self, name: str) -> bool:
        return name in self._outputs

    def set_gain(self, name: str, gain: float) -> None:
        self._outputs[name].gain = gain

    def set_enabled(self, name: str, enabled: bool) -> None:
        self._outputs[name].enabled = enabled

    def push(self, audio: np.ndarray) -> None:
        for out in self._outputs.values():
            out.buffer.write(audio * out.gain if out.enabled else np.zeros_like(audio))

    def read(self, name: str, n: int) -> np.ndarray:
        return self._outputs[name].buffer.read(n)

    def buffered(self, name: str) -> int:
        return self._outputs[name].buffer.available

    def underruns(self) -> int:
        return sum(o.buffer.underruns for o in self._outputs.values())
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_ringbuffer.py backend/tests/test_router.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/ringbuffer.py backend/aivoice/router.py backend/tests/test_ringbuffer.py backend/tests/test_router.py
git commit -m "feat: ring buffer and output router"
```

---

### Task 5: Devices

**Files:**
- Create: `backend/aivoice/devices.py`
- Test: `backend/tests/test_devices.py`

**Interfaces:**
- Produces: `Device(index: int, name: str, max_input: int, max_output: int)` (frozen dataclass); `DeviceError(ValueError)`; `list_devices(sd=None) -> list[Device]` (WASAPI devices only); `find_virtual_mic(devices) -> Device | None`; `get_device(devices, index) -> Device` raising `DeviceError`; `SoundDeviceStreams(sd=None)` with `input(device: int, samplerate: int, on_audio: Callable[[np.ndarray], None])` and `output(device: int, samplerate: int, fill: Callable[[int], np.ndarray])`, each returning a stream object with `start()`, `stop()`, `close()`, `active: bool`, `latency: float` (seconds).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from aivoice.devices import DeviceError, SoundDeviceStreams, find_virtual_mic, get_device, list_devices


class FakeSd:
    def __init__(self):
        self.opened = []

    def query_hostapis(self):
        return ({"name": "MME"}, {"name": "Windows WASAPI"})

    def query_devices(self):
        return [
            {"index": 0, "name": "Mic (MME)", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
            {"index": 1, "name": "Headset Mic", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0},
            {"index": 2, "name": "Headset", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
            {"index": 3, "name": "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", "hostapi": 1,
             "max_input_channels": 0, "max_output_channels": 8},
        ]

    def WasapiSettings(self, **kwargs):
        return ("wasapi", kwargs)

    def InputStream(self, **kwargs):
        self.opened.append(("in", kwargs))
        return kwargs

    def OutputStream(self, **kwargs):
        self.opened.append(("out", kwargs))
        return kwargs


def test_lists_only_wasapi_devices():
    devices = list_devices(FakeSd())
    assert [d.index for d in devices] == [1, 2, 3]
    assert devices[0].max_input == 1 and devices[1].max_output == 2


def test_finds_voicemeeter():
    assert find_virtual_mic(list_devices(FakeSd())).index == 3


def test_virtual_mic_absent():
    devices = [d for d in list_devices(FakeSd()) if d.index != 3]
    assert find_virtual_mic(devices) is None


def test_virtual_mic_must_be_an_output():
    sd = FakeSd()
    devices = [d for d in list_devices(sd) if d.max_output == 0]
    assert find_virtual_mic(devices) is None


def test_get_device_unknown_index():
    with pytest.raises(DeviceError, match="no longer available"):
        get_device(list_devices(FakeSd()), 99)


def test_input_stream_uses_mono_float32_and_auto_convert():
    sd = FakeSd()
    received = []
    stream = SoundDeviceStreams(sd).input(1, 48000, received.append)
    assert stream["device"] == 1 and stream["samplerate"] == 48000
    assert stream["channels"] == 1 and stream["dtype"] == "float32"
    assert stream["extra_settings"] == ("wasapi", {"auto_convert": True})
    stream["callback"](np.ones((4, 1), dtype=np.float32), 4, None, None)
    assert received[0].shape == (4, 1)


def test_output_stream_duplicates_mono_to_stereo():
    sd = FakeSd()
    stream = SoundDeviceStreams(sd).output(2, 48000, lambda n: np.full(n, 0.5, dtype=np.float32))
    assert stream["channels"] == 2
    assert stream["extra_settings"] == ("wasapi", {"auto_convert": True})
    out = np.zeros((3, 2), dtype=np.float32)
    stream["callback"](out, 3, None, None)
    assert out.tolist() == [[0.5, 0.5]] * 3
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_devices.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/aivoice/devices.py`**

```python
from dataclasses import dataclass
from typing import Callable

import numpy as np

VIRTUAL_MIC_NAMES = ("voicemeeter input", "voicemeeter aux input", "cable input")


class DeviceError(ValueError):
    pass


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    max_input: int
    max_output: int


def _sd(sd):
    if sd is None:
        import sounddevice

        return sounddevice
    return sd


def list_devices(sd=None) -> list[Device]:
    sd = _sd(sd)
    wasapi = next((i for i, api in enumerate(sd.query_hostapis()) if "WASAPI" in api["name"]), None)
    return [
        Device(d["index"], d["name"], d["max_input_channels"], d["max_output_channels"])
        for d in sd.query_devices()
        if d["hostapi"] == wasapi
    ]


def find_virtual_mic(devices: list[Device]) -> Device | None:
    for device in devices:
        if device.max_output > 0 and any(n in device.name.lower() for n in VIRTUAL_MIC_NAMES):
            return device
    return None


def get_device(devices: list[Device], index: int) -> Device:
    for device in devices:
        if device.index == index:
            return device
    raise DeviceError(f"Audio device {index} is no longer available")


class SoundDeviceStreams:
    """Opens WASAPI streams. auto_convert lets Windows resample devices that are not 48 kHz."""

    def __init__(self, sd=None):
        self._sd = _sd(sd)

    def input(self, device: int, samplerate: int, on_audio: Callable[[np.ndarray], None]):
        def callback(indata, frames, time, status):
            on_audio(indata.copy())

        return self._sd.InputStream(
            device=device, samplerate=samplerate, channels=1, dtype="float32", latency="low",
            callback=callback, extra_settings=self._sd.WasapiSettings(auto_convert=True),
        )

    def output(self, device: int, samplerate: int, fill: Callable[[int], np.ndarray]):
        def callback(outdata, frames, time, status):
            outdata[:] = fill(frames)[:, None]

        return self._sd.OutputStream(
            device=device, samplerate=samplerate, channels=2, dtype="float32", latency="low",
            callback=callback, extra_settings=self._sd.WasapiSettings(auto_convert=True),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_devices.py -v`
Expected: 7 passed.

- [ ] **Step 5: Check against the real machine**

Run: `backend/.venv/Scripts/python -c "from aivoice.devices import *; ds=list_devices(); [print(d) for d in ds]; print('virtual mic:', find_virtual_mic(ds))"`
Expected: a list of WASAPI devices and `virtual mic:` showing a VoiceMeeter Input device. If it prints `None`, add the exact lowercase name fragment shown in the list to `VIRTUAL_MIC_NAMES` and add a matching assertion to the tests.

- [ ] **Step 6: Commit**

```bash
git add backend/aivoice/devices.py backend/tests/test_devices.py
git commit -m "feat: WASAPI device listing and stream factory"
```

---

### Task 6: Stream processor

The pure core of live mode. No threads, no devices.

How it works: it keeps a rolling window of the last `context + chunk + crossfade` input samples. Each call converts the whole window, takes the last `chunk + crossfade` samples, crossfades the first `crossfade` samples with the tail saved from the previous call, emits `chunk` samples, and saves the last `crossfade` samples as the new tail. With an engine that returns its input unchanged, the output is exactly the input delayed by `crossfade` samples.

**Files:**
- Create: `backend/aivoice/engines/base.py`, `backend/aivoice/processor.py`
- Test: `backend/tests/test_processor.py`

**Interfaces:**
- Consumes: `presets.LiveParams`.
- Produces: `engines.base.VoiceEngine` protocol with `convert(audio: np.ndarray, sample_rate: int, pitch_shift: float, params: LiveParams) -> np.ndarray` (same length as `audio`, float32). `processor.EngineFailure(RuntimeError)`. `processor.StreamProcessor(engine, sample_rate: int, params: LiveParams, max_failures: int = 5)` with attributes `chunk: int`, `crossfade: int`, `window: int` (sample counts), `params`, `mode: str` (`"converted"` or `"muted"`), `pitch_shift: float`, `infer_ms: float`, `slow_chunks: int`, and method `process(block: np.ndarray) -> np.ndarray` (input and output both exactly `chunk` samples).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from aivoice.presets import LiveParams
from aivoice.processor import EngineFailure, StreamProcessor

SR = 48000
PARAMS = LiveParams(chunk_ms=100, crossfade_ms=20, context_ms=200, f0_method="dio", index_rate=0.0)


class Identity:
    def convert(self, audio, sample_rate, pitch_shift, params):
        return audio.copy()


class Raises:
    def convert(self, audio, sample_rate, pitch_shift, params):
        raise RuntimeError("boom")


class Flaky:
    def __init__(self, fail_on):
        self.calls = 0
        self.fail_on = fail_on

    def convert(self, audio, sample_rate, pitch_shift, params):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("boom")
        return audio.copy()


def run(proc, signal):
    blocks = signal.reshape(-1, proc.chunk)
    return np.concatenate([proc.process(b) for b in blocks])


def noise(chunks, proc):
    return np.random.default_rng(0).standard_normal(chunks * proc.chunk).astype(np.float32)


def test_sizes():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    assert (proc.chunk, proc.crossfade, proc.window) == (4800, 960, 9600 + 4800 + 960)


def test_identity_engine_is_input_delayed_by_crossfade():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    x = noise(10, proc)
    y = run(proc, x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.allclose(y[: proc.crossfade], 0.0)
    assert np.allclose(y[proc.crossfade :], x[: -proc.crossfade], atol=1e-5)


def test_engine_error_emits_silence_never_input():
    proc = StreamProcessor(Raises(), SR, PARAMS, max_failures=100)
    y = run(proc, noise(5, proc) + 1.0)
    assert np.all(y == 0.0)


def test_repeated_failures_raise():
    proc = StreamProcessor(Raises(), SR, PARAMS, max_failures=3)
    block = np.ones(proc.chunk, dtype=np.float32)
    proc.process(block)
    proc.process(block)
    with pytest.raises(EngineFailure):
        proc.process(block)


def test_failure_counter_resets_on_success():
    proc = StreamProcessor(Flaky({1, 2, 4, 5}), SR, PARAMS, max_failures=3)
    block = np.ones(proc.chunk, dtype=np.float32)
    for _ in range(6):
        proc.process(block)


def test_wrong_length_engine_output_is_treated_as_failure():
    class Short:
        def convert(self, audio, sample_rate, pitch_shift, params):
            return audio[:10]

    proc = StreamProcessor(Short(), SR, PARAMS, max_failures=100)
    assert np.all(proc.process(np.ones(proc.chunk, dtype=np.float32)) == 0.0)


def test_nan_engine_output_is_treated_as_failure():
    class Nan:
        def convert(self, audio, sample_rate, pitch_shift, params):
            return np.full_like(audio, np.nan)

    proc = StreamProcessor(Nan(), SR, PARAMS, max_failures=100)
    assert np.all(proc.process(np.ones(proc.chunk, dtype=np.float32)) == 0.0)


def test_muted_emits_silence_without_calling_engine():
    engine = Flaky(set())
    proc = StreamProcessor(engine, SR, PARAMS)
    proc.mode = "muted"
    assert np.all(proc.process(np.ones(proc.chunk, dtype=np.float32)) == 0.0)
    assert engine.calls == 0


def test_unmute_fades_in_from_silence():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    ones = np.ones(proc.chunk, dtype=np.float32)
    proc.process(ones)
    proc.process(ones)
    proc.mode = "muted"
    proc.process(ones)
    proc.mode = "converted"
    y = proc.process(ones)
    assert y[0] == pytest.approx(0.0, abs=1e-3)
    assert y[-1] == pytest.approx(1.0, abs=1e-5)


def test_wrong_block_size_rejected():
    proc = StreamProcessor(Identity(), SR, PARAMS)
    with pytest.raises(ValueError):
        proc.process(np.zeros(proc.chunk + 1, dtype=np.float32))


def test_pitch_shift_and_params_reach_engine():
    seen = {}

    class Spy:
        def convert(self, audio, sample_rate, pitch_shift, params):
            seen.update(sr=sample_rate, pitch=pitch_shift, params=params, n=len(audio))
            return audio

    proc = StreamProcessor(Spy(), SR, PARAMS)
    proc.pitch_shift = 7.0
    proc.process(np.zeros(proc.chunk, dtype=np.float32))
    assert seen == {"sr": SR, "pitch": 7.0, "params": PARAMS, "n": proc.window}
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_processor.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`backend/aivoice/engines/base.py`:

```python
from typing import Protocol

import numpy as np

from aivoice.presets import LiveParams


class VoiceEngine(Protocol):
    def convert(self, audio: np.ndarray, sample_rate: int, pitch_shift: float, params: LiveParams) -> np.ndarray:
        """Convert mono float32 audio. Must return float32 of the same length, time-aligned at the end."""
        ...
```

`backend/aivoice/processor.py`:

```python
import time

import numpy as np

from aivoice.engines.base import VoiceEngine
from aivoice.presets import LiveParams


class EngineFailure(RuntimeError):
    pass


class StreamProcessor:
    def __init__(self, engine: VoiceEngine, sample_rate: int, params: LiveParams, max_failures: int = 5):
        self.engine = engine
        self.sample_rate = sample_rate
        self.params = params
        self.chunk = sample_rate * params.chunk_ms // 1000
        self.crossfade = sample_rate * params.crossfade_ms // 1000
        self.window = sample_rate * params.context_ms // 1000 + self.chunk + self.crossfade
        self.mode = "converted"
        self.pitch_shift = 0.0
        self.infer_ms = 0.0
        self.slow_chunks = 0
        self._max_failures = max_failures
        self._failures = 0
        self._input = np.zeros(self.window, dtype=np.float32)
        self._tail = np.zeros(self.crossfade, dtype=np.float32)
        t = np.linspace(0.0, 1.0, self.crossfade, endpoint=False, dtype=np.float32)
        self._fade_in = np.sin(0.5 * np.pi * t) ** 2
        self._fade_out = 1.0 - self._fade_in

    def _silence(self) -> np.ndarray:
        self._tail[:] = 0.0
        return np.zeros(self.chunk, dtype=np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        if len(block) != self.chunk:
            raise ValueError(f"expected {self.chunk} samples, got {len(block)}")
        self._input = np.concatenate([self._input[self.chunk :], block.astype(np.float32)])
        if self.mode == "muted":
            return self._silence()
        try:
            start = time.perf_counter()
            out = self.engine.convert(self._input, self.sample_rate, self.pitch_shift, self.params)
            self.infer_ms = (time.perf_counter() - start) * 1000.0
            if out.shape != self._input.shape or not np.all(np.isfinite(out)):
                raise ValueError("engine returned invalid audio")
        except Exception as exc:  # noqa: BLE001 - any engine fault must fail closed
            self._failures += 1
            if self._failures >= self._max_failures:
                raise EngineFailure(f"Voice conversion failed {self._failures} times in a row: {exc}") from exc
            return self._silence()
        self._failures = 0
        if self.infer_ms > self.params.chunk_ms:
            self.slow_chunks += 1
        segment = out[-(self.chunk + self.crossfade) :].astype(np.float32)
        head = segment[: self.crossfade] * self._fade_in + self._tail * self._fade_out
        emitted = np.concatenate([head, segment[self.crossfade : self.chunk]])
        self._tail = segment[self.chunk :].copy()
        return emitted
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_processor.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/engines/base.py backend/aivoice/processor.py backend/tests/test_processor.py
git commit -m "feat: fail-closed stream processor with crossfade"
```

---

### Task 7: Pitch extraction and RVC engine

**Files:**
- Create: `backend/aivoice/engines/f0.py`, `backend/aivoice/engines/rvc.py`
- Test: `backend/tests/test_f0.py`, `backend/tests/test_rvc.py`, `backend/tests/test_rvc_real.py`

**Interfaces:**
- Consumes: `presets.LiveParams`; ONNX tensor names from Task 1.
- Produces: `f0.extract_f0(audio16k: np.ndarray, method: str, n_frames: int) -> np.ndarray` (float32 Hz, length `n_frames`, 0 for unvoiced); `f0.shift_f0(f0, semitones: float) -> np.ndarray`; `f0.to_coarse(f0) -> np.ndarray` (int64 in 1..255). `rvc.make_session(path, device_id: int)` returning an ONNX Runtime session; `rvc.blend_index(feats, index, big_npy, rate: float) -> np.ndarray`; `rvc.RvcEngine(model_path, vec_path, model_sample_rate: int, index_path=None, session_factory=None)` implementing `VoiceEngine`. `session_factory` is a `Callable[[str], session]`.

Pipeline, from the RVC project's own ONNX inference code: resample to 16 kHz, run ContentVec to get `(1, T, 768)` features, optionally blend with index neighbours, repeat each frame twice, extract pitch for `2T` frames, shift it, map to coarse pitch, run the synthesiser, resample its output back to the input rate, and fit to the input length keeping the end aligned.

- [ ] **Step 1: Write the failing pitch tests**

`backend/tests/test_f0.py`:

```python
import numpy as np
import pytest

from aivoice.engines.f0 import extract_f0, shift_f0, to_coarse


def sine(freq, seconds=1.0, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("method", ["dio", "harvest"])
def test_detects_220hz(method):
    f0 = extract_f0(sine(220), method, 100)
    assert f0.shape == (100,) and f0.dtype == np.float32
    voiced = f0[f0 > 0]
    assert len(voiced) > 50
    assert abs(np.median(voiced) - 220) < 5


@pytest.mark.parametrize("method", ["dio", "harvest"])
def test_silence_is_unvoiced_and_finite(method):
    f0 = extract_f0(np.zeros(16000, dtype=np.float32), method, 100)
    assert f0.shape == (100,)
    assert np.all(np.isfinite(f0)) and np.all(f0 == 0)


def test_frame_count_is_respected():
    assert extract_f0(sine(220), "dio", 37).shape == (37,)


def test_unknown_method():
    with pytest.raises(ValueError):
        extract_f0(sine(220), "magic", 10)


def test_shift_octave_doubles_and_keeps_unvoiced():
    f0 = np.array([0.0, 110.0, 220.0], dtype=np.float32)
    assert shift_f0(f0, 12).tolist() == pytest.approx([0.0, 220.0, 440.0])


def test_coarse_range():
    coarse = to_coarse(np.array([0.0, 50.0, 220.0, 1100.0, 5000.0], dtype=np.float32))
    assert coarse.dtype == np.int64
    assert coarse[0] == 1 and coarse[1] == 1 and coarse[3] == 255 and coarse[4] == 255
    assert 1 < coarse[2] < 255
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_f0.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/aivoice/engines/f0.py`**

```python
import numpy as np
import pyworld

F0_MIN = 50.0
F0_MAX = 1100.0
_MEL_MIN = 1127.0 * np.log(1.0 + F0_MIN / 700.0)
_MEL_MAX = 1127.0 * np.log(1.0 + F0_MAX / 700.0)


def _resize(f0: np.ndarray, n: int) -> np.ndarray:
    if len(f0) == n:
        return f0
    if len(f0) == 0:
        return np.zeros(n, dtype=f0.dtype)
    idx = np.round(np.linspace(0, len(f0) - 1, n)).astype(int)
    return f0[idx]


def extract_f0(audio16k: np.ndarray, method: str, n_frames: int) -> np.ndarray:
    x = audio16k.astype(np.float64)
    if method == "dio":
        f0, t = pyworld.dio(x, 16000, f0_floor=F0_MIN, f0_ceil=F0_MAX, frame_period=10)
        f0 = pyworld.stonemask(x, f0, t, 16000)
    elif method == "harvest":
        f0, _ = pyworld.harvest(x, 16000, f0_floor=F0_MIN, f0_ceil=F0_MAX, frame_period=10)
    else:
        raise ValueError(f"Unknown f0 method: {method}")
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    return _resize(f0, n_frames).astype(np.float32)


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    return (f0 * 2.0 ** (semitones / 12.0)).astype(np.float32)


def to_coarse(f0: np.ndarray) -> np.ndarray:
    mel = 1127.0 * np.log(1.0 + f0.astype(np.float64) / 700.0)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - _MEL_MIN) * 254.0 / (_MEL_MAX - _MEL_MIN) + 1.0
    return np.rint(np.clip(mel, 1.0, 255.0)).astype(np.int64)
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_f0.py -v`
Expected: all passed. If `harvest` on silence returns non-zero values, the `test_silence_is_unvoiced_and_finite[harvest]` assertion fails: in that case add an energy gate at the top of `extract_f0` (`if not np.any(np.abs(x) > 1e-6): return np.zeros(n_frames, dtype=np.float32)`) and rerun.

- [ ] **Step 5: Write the failing engine tests**

`backend/tests/test_rvc.py`:

```python
import numpy as np
import pytest

from aivoice.engines.rvc import RvcEngine, blend_index
from aivoice.presets import LiveParams

SR = 48000
PARAMS = LiveParams(100, 20, 200, "dio", 0.0)


class In:
    def __init__(self, name):
        self.name = name


class FakeVec:
    def get_inputs(self):
        return [In("source")]

    def run(self, _, feed):
        frames = feed["source"].shape[-1] // 320
        return [np.zeros((1, frames, 768), dtype=np.float32)]


class FakeNet:
    def __init__(self, model_sr):
        self.model_sr = model_sr
        self.feeds = []

    def get_inputs(self):
        return [In(n) for n in ("phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd")]

    def run(self, _, feed):
        self.feeds.append(feed)
        frames = feed["phone"].shape[1]
        return [np.full((1, 1, frames * self.model_sr // 100), 0.25, dtype=np.float32)]


def make(model_sr=40000):
    net = FakeNet(model_sr)
    engine = RvcEngine(
        "model.onnx", "vec.onnx", model_sr,
        session_factory=lambda path: FakeVec() if "vec" in path else net,
    )
    return engine, net


def sine(freq, n=SR):
    return (0.5 * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def test_output_matches_input_length_and_dtype():
    engine, _ = make()
    for n in (48000, 40321, 26400):
        out = engine.convert(sine(220, n), SR, 0.0, PARAMS)
        assert out.shape == (n,) and out.dtype == np.float32


def test_48k_model_also_fits_length():
    engine, _ = make(model_sr=48000)
    assert engine.convert(sine(220), SR, 0.0, PARAMS).shape == (48000,)


def test_feed_shapes_and_types():
    engine, net = make()
    engine.convert(sine(220), SR, 0.0, PARAMS)
    feed = net.feeds[0]
    frames = feed["phone"].shape[1]
    assert feed["phone"].shape == (1, frames, 768) and feed["phone"].dtype == np.float32
    assert frames == 2 * (16000 // 320)
    assert feed["phone_lengths"].tolist() == [frames] and feed["phone_lengths"].dtype == np.int64
    assert feed["pitch"].shape == (1, frames) and feed["pitch"].dtype == np.int64
    assert feed["pitchf"].shape == (1, frames) and feed["pitchf"].dtype == np.float32
    assert feed["ds"].tolist() == [0]
    assert feed["rnd"].shape == (1, 192, frames)


def test_pitch_shift_of_an_octave_doubles_pitch():
    engine, net = make()
    engine.convert(sine(220), SR, 0.0, PARAMS)
    engine.convert(sine(220), SR, 12.0, PARAMS)
    base, shifted = (f["pitchf"][0] for f in net.feeds)
    assert np.median(shifted[shifted > 0]) / np.median(base[base > 0]) == pytest.approx(2.0, rel=1e-4)


def test_silent_input_gives_finite_output():
    engine, net = make()
    out = engine.convert(np.zeros(SR, dtype=np.float32), SR, 0.0, PARAMS)
    assert np.all(np.isfinite(out))
    assert np.all(net.feeds[0]["pitch"] == 1)


def test_no_index_ignores_index_rate():
    engine, _ = make()
    params = LiveParams(100, 20, 200, "dio", 0.8)
    assert engine.convert(sine(220), SR, 0.0, params).shape == (SR,)


class FakeIndex:
    def search(self, x, k):
        return np.ones((len(x), k), dtype=np.float32), np.zeros((len(x), k), dtype=np.int64)


def test_blend_index_rate_zero_is_unchanged():
    feats = np.random.default_rng(0).standard_normal((5, 768)).astype(np.float32)
    big = np.ones((3, 768), dtype=np.float32)
    assert np.array_equal(blend_index(feats, FakeIndex(), big, 0.0), feats)


def test_blend_index_rate_one_is_neighbours():
    feats = np.zeros((5, 768), dtype=np.float32)
    big = np.full((3, 768), 2.0, dtype=np.float32)
    assert np.allclose(blend_index(feats, FakeIndex(), big, 1.0), 2.0)


def test_blend_index_handles_zero_distance():
    class Exact(FakeIndex):
        def search(self, x, k):
            return np.zeros((len(x), k), dtype=np.float32), np.zeros((len(x), k), dtype=np.int64)

    out = blend_index(np.zeros((2, 768), dtype=np.float32), Exact(), np.ones((1, 768), dtype=np.float32), 0.5)
    assert np.all(np.isfinite(out))
```

- [ ] **Step 6: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_rvc.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 7: Implement `backend/aivoice/engines/rvc.py`**

```python
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from aivoice.engines.f0 import extract_f0, shift_f0, to_coarse
from aivoice.presets import LiveParams


def make_session(path: str, device_id: int = 0):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False  # required by the DirectML provider
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), options,
        providers=[("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
    )


def blend_index(feats: np.ndarray, index, big_npy: np.ndarray, rate: float) -> np.ndarray:
    """Pull features toward their nearest neighbours in the trained voice's index."""
    if rate <= 0:
        return feats
    score, ix = index.search(feats, 8)
    weight = np.square(1.0 / np.maximum(score, 1e-9))
    weight /= weight.sum(axis=1, keepdims=True)
    neighbours = np.sum(big_npy[ix] * weight[:, :, None], axis=1)
    return (neighbours * rate + feats * (1.0 - rate)).astype(np.float32)


def _fit(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or pad at the start so the end (the newest audio) stays aligned."""
    if len(audio) >= n:
        return audio[-n:]
    return np.concatenate([np.zeros(n - len(audio), dtype=audio.dtype), audio])


class RvcEngine:
    def __init__(self, model_path, vec_path, model_sample_rate: int, index_path=None,
                 session_factory: Callable[[str], object] | None = None):
        factory = session_factory or make_session
        self._vec = factory(str(vec_path))
        self._net = factory(str(model_path))
        self._vec_input = self._vec.get_inputs()[0].name
        self._model_sr = model_sample_rate
        self._index = None
        self._big_npy = None
        if index_path is not None and Path(index_path).exists():
            import faiss

            self._index = faiss.read_index(str(index_path))
            self._big_npy = self._index.reconstruct_n(0, self._index.ntotal)

    def convert(self, audio: np.ndarray, sample_rate: int, pitch_shift: float, params: LiveParams) -> np.ndarray:
        n = len(audio)
        wav16 = soxr.resample(audio.astype(np.float32), sample_rate, 16000).astype(np.float32)
        feats = self._vec.run(None, {self._vec_input: wav16[None, None, :]})[0][0]
        if self._index is not None:
            feats = blend_index(feats.astype(np.float32), self._index, self._big_npy, params.index_rate)
        phone = np.repeat(feats, 2, axis=0)[None].astype(np.float32)
        frames = phone.shape[1]
        pitchf = shift_f0(extract_f0(wav16, params.f0_method, frames), pitch_shift)
        feed = {
            "phone": phone,
            "phone_lengths": np.array([frames], dtype=np.int64),
            "pitch": to_coarse(pitchf)[None],
            "pitchf": pitchf[None],
            "ds": np.array([0], dtype=np.int64),
            "rnd": np.random.randn(1, 192, frames).astype(np.float32),
        }
        out = np.asarray(self._net.run(None, feed)[0], dtype=np.float32).reshape(-1)
        if self._model_sr != sample_rate:
            out = soxr.resample(out, self._model_sr, sample_rate).astype(np.float32)
        return _fit(out, n)
```

- [ ] **Step 8: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_rvc.py -v`
Expected: 9 passed.

- [ ] **Step 9: Add the real-model test**

`backend/tests/test_rvc_real.py`:

```python
import json
from functools import partial

import numpy as np
import pytest

from aivoice import config
from aivoice.engines.rvc import RvcEngine, make_session
from aivoice.presets import PRESETS

DATA = config.data_dir()
VEC = DATA / "pretrained" / "vec-768-layer-12.onnx"
VOICES = sorted((DATA / "pretrained" / "rvc").glob("*/model.onnx")) if (DATA / "pretrained" / "rvc").exists() else []

pytestmark = pytest.mark.skipif(not VEC.exists() or not VOICES, reason="real models not downloaded")


def test_real_model_converts_on_directml():
    model = VOICES[0]
    meta = json.loads((model.parent / "voice.json").read_text())
    index = model.parent / "model.index"
    device_id = config.load_settings(DATA)["dml_device_id"]
    engine = RvcEngine(model, VEC, meta["sample_rate"], index if index.exists() else None,
                       session_factory=partial(make_session, device_id=device_id))
    sr = config.SAMPLE_RATE
    t = np.arange(sr) / sr
    voice_like = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    for preset in PRESETS.values():
        out = engine.convert(voice_like, sr, 0.0, preset)
        assert out.shape == voice_like.shape
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) > 1e-4
```

- [ ] **Step 10: Run the real-model test**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_rvc_real.py -v`
Expected: PASS (not skipped, since Task 1 downloaded the models). If it fails on tensor names or shapes, fix `RvcEngine` using the names recorded in the Task 1 benchmark notes, update the fakes in `test_rvc.py` to match, and rerun both files.

- [ ] **Step 11: Commit**

```bash
git add backend/aivoice/engines backend/tests/test_f0.py backend/tests/test_rvc.py backend/tests/test_rvc_real.py
git commit -m "feat: pitch extraction and RVC engine on DirectML"
```

---

### Task 8: Live session

Connects device streams to the processor. The worker thread is a thin loop around `pump()`, so tests drive `feed()` and `pump()` directly without threads.

**Files:**
- Create: `backend/aivoice/live.py`
- Test: `backend/tests/test_live.py`

**Interfaces:**
- Consumes: `StreamProcessor` (Task 6), `Router` (Task 4), `RingBuffer` (Task 4), a stream factory with the `SoundDeviceStreams` shape (Task 5).
- Produces: `LiveSession(processor, router, sample_rate: int, stream_factory, input_device: int, output_devices: dict[str, int])` with `start()`, `stop()`, `feed(indata: np.ndarray)`, `pump() -> bool`, `stats() -> dict`, attributes `running: bool`, `error: str | None`. `stats()` keys: `running`, `mode`, `error`, `infer_ms`, `latency_ms`, `in_level`, `out_level`, `underruns`, `slow_chunks`, `suggest_faster`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from aivoice.live import LiveSession
from aivoice.presets import LiveParams
from aivoice.processor import StreamProcessor
from aivoice.router import Router

SR = 48000
PARAMS = LiveParams(50, 10, 100, "dio", 0.0)


class Identity:
    def convert(self, audio, sample_rate, pitch_shift, params):
        return audio.copy()


class Raises:
    def convert(self, audio, sample_rate, pitch_shift, params):
        raise RuntimeError("boom")


class FakeStream:
    def __init__(self, kind, device):
        self.kind, self.device = kind, device
        self.active = False
        self.closed = False
        self.latency = 0.01

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


class FakeStreams:
    def __init__(self, fail_on=None):
        self.streams = []
        self.fail_on = fail_on

    def input(self, device, samplerate, on_audio):
        return self._make("in", device)

    def output(self, device, samplerate, fill):
        return self._make("out", device)

    def _make(self, kind, device):
        if device == self.fail_on:
            raise RuntimeError("device busy")
        stream = FakeStream(kind, device)
        self.streams.append(stream)
        return stream


def make(engine=None, streams=None, outputs=None, max_failures=5):
    proc = StreamProcessor(engine or Identity(), SR, PARAMS, max_failures=max_failures)
    router = Router(SR * 2)
    outputs = outputs or {"mic": 3}
    for name in outputs:
        router.add_output(name)
    streams = streams or FakeStreams()
    return LiveSession(proc, router, SR, streams, 1, outputs), proc, router, streams


def drain(session):
    while session.pump():
        pass


def test_arbitrary_block_sizes_lose_no_samples():
    session, proc, router, _ = make()
    x = np.random.default_rng(1).standard_normal(proc.chunk * 6).astype(np.float32)
    sizes = [1000, 37, 2048, 480, 5000, 1]
    pos = 0
    while pos < len(x):
        size = sizes[pos % len(sizes)]
        session.feed(x[pos : pos + size])
        pos += size
        drain(session)
    y = router.read("mic", len(x))
    assert np.allclose(y[proc.crossfade :], x[: -proc.crossfade], atol=1e-5)


def test_pump_without_enough_audio_does_nothing():
    session, proc, router, _ = make()
    session.feed(np.ones(proc.chunk - 1, dtype=np.float32))
    assert session.pump() is False
    assert router.buffered("mic") == 0


def test_stereo_input_is_downmixed():
    session, proc, router, _ = make()
    stereo = np.stack([np.ones(proc.chunk * 3), -np.ones(proc.chunk * 3)], axis=1).astype(np.float32)
    session.feed(stereo)
    drain(session)
    assert np.allclose(router.read("mic", proc.chunk * 3), 0.0)


def test_single_channel_2d_input_accepted():
    session, proc, router, _ = make()
    session.feed(np.ones((proc.chunk, 1), dtype=np.float32))
    assert session.pump() is True


def test_engine_failure_stops_session_with_error_and_silence():
    session, proc, router, _ = make(engine=Raises(), max_failures=2)
    session.feed(np.ones(proc.chunk * 4, dtype=np.float32))
    drain(session)
    assert session.error and "failed" in session.error
    assert np.all(router.read("mic", proc.chunk * 4) == 0.0)
    assert session.pump() is False


def test_start_opens_and_starts_all_streams():
    session, _, _, streams = make(outputs={"mic": 3, "monitor": 2})
    session.start()
    try:
        assert [(s.kind, s.device) for s in streams.streams] == [("in", 1), ("out", 3), ("out", 2)]
        assert all(s.active for s in streams.streams)
        assert session.running
    finally:
        session.stop()


def test_start_twice_does_not_open_more_streams():
    session, _, _, streams = make()
    session.start()
    session.start()
    try:
        assert len(streams.streams) == 2
    finally:
        session.stop()


def test_stop_closes_streams_and_is_idempotent():
    session, _, _, streams = make()
    session.start()
    session.stop()
    session.stop()
    assert all(s.closed and not s.active for s in streams.streams)
    assert not session.running


def test_stop_before_start_is_safe():
    session, _, _, _ = make()
    session.stop()


def test_failed_open_closes_already_opened_streams():
    streams = FakeStreams(fail_on=3)
    session, _, _, _ = make(streams=streams)
    try:
        session.start()
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert all(s.closed for s in streams.streams)
    assert not session.running


def test_disconnected_device_is_detected():
    session, _, _, streams = make()
    session.start()
    try:
        streams.streams[0].active = False
        assert session.check_devices() is False
        assert "disconnected" in session.error
    finally:
        session.stop()


def test_stats_shape():
    session, proc, _, _ = make()
    session.feed(np.full(proc.chunk, 0.5, dtype=np.float32))
    session.pump()
    stats = session.stats()
    assert set(stats) == {"running", "mode", "error", "infer_ms", "latency_ms", "in_level", "out_level",
                          "underruns", "slow_chunks", "suggest_faster"}
    assert stats["in_level"] == 0.5 and stats["mode"] == "converted"
    assert stats["latency_ms"] >= PARAMS.chunk_ms + PARAMS.crossfade_ms
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_live.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/aivoice/live.py`**

```python
import threading
import time

import numpy as np

from aivoice.processor import EngineFailure, StreamProcessor
from aivoice.ringbuffer import RingBuffer
from aivoice.router import Router

SLOW_CHUNK_LIMIT = 10


class LiveSession:
    def __init__(self, processor: StreamProcessor, router: Router, sample_rate: int, stream_factory,
                 input_device: int, output_devices: dict[str, int]):
        self.processor = processor
        self.router = router
        self.sample_rate = sample_rate
        self.running = False
        self.error: str | None = None
        self._factory = stream_factory
        self._input_device = input_device
        self._output_devices = output_devices
        self._inbuf = RingBuffer(sample_rate * 4)
        self._streams: list = []
        self._thread: threading.Thread | None = None
        self._failed = False
        self._in_level = 0.0
        self._out_level = 0.0

    def feed(self, indata: np.ndarray) -> None:
        mono = indata if indata.ndim == 1 else indata.mean(axis=1)
        self._in_level = float(np.max(np.abs(mono))) if len(mono) else 0.0
        self._inbuf.write(mono)

    def pump(self) -> bool:
        if self._failed or self._inbuf.available < self.processor.chunk:
            return False
        block = self._inbuf.read(self.processor.chunk)
        try:
            out = self.processor.process(block)
        except EngineFailure as exc:
            self.error = str(exc)
            self._failed = True
            self.router.push(np.zeros(self.processor.chunk, dtype=np.float32))
            return False
        self._out_level = float(np.max(np.abs(out)))
        self.router.push(out)
        return True

    def check_devices(self) -> bool:
        if all(s.active for s in self._streams):
            return True
        self.error = "An audio device was disconnected"
        self._failed = True
        return False

    def start(self) -> None:
        if self.running:
            return
        try:
            self._streams.append(self._factory.input(self._input_device, self.sample_rate, self.feed))
            for name, device in self._output_devices.items():
                self._streams.append(
                    self._factory.output(device, self.sample_rate, lambda n, name=name: self.router.read(name, n))
                )
            for stream in self._streams:
                stream.start()
        except Exception:
            self._close_streams()
            raise
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_check = time.monotonic()
        while self.running and not self._failed:
            if not self.pump():
                time.sleep(0.002)
            if time.monotonic() - last_check > 0.5:
                last_check = time.monotonic()
                self.check_devices()
        if self._failed:
            self._close_streams()
            self.running = False

    def _close_streams(self) -> None:
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - closing a dead device must not raise
                pass
        self._streams = []

    def stop(self) -> None:
        self.running = False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._close_streams()

    def stats(self) -> dict:
        p = self.processor
        stream_ms = sum(getattr(s, "latency", 0.0) for s in self._streams[:2]) * 1000.0
        buffered_ms = 1000.0 * self.router.buffered("mic") / self.sample_rate if self.router.has("mic") else 0.0
        return {
            "running": self.running,
            "mode": p.mode,
            "error": self.error,
            "infer_ms": round(p.infer_ms, 1),
            "latency_ms": round(p.params.chunk_ms + p.params.crossfade_ms + p.infer_ms + stream_ms + buffered_ms, 1),
            "in_level": self._in_level,
            "out_level": self._out_level,
            "underruns": self.router.underruns(),
            "slow_chunks": p.slow_chunks,
            "suggest_faster": p.slow_chunks >= SLOW_CHUNK_LIMIT,
        }
```

`_close_streams` keeps closed `FakeStream` objects reachable through the test's own `streams.streams` list, which is what the tests inspect.

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_live.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/live.py backend/tests/test_live.py
git commit -m "feat: live session wiring streams to the processor"
```

---

### Task 9: Pretrained voices, setup check, and live manager

**Files:**
- Create: `backend/aivoice/pretrained.py`, `backend/aivoice/setup_check.py`, `backend/aivoice/manager.py`
- Create: `backend/tests/__init__.py` (empty; lets test files import helpers from each other)
- Test: `backend/tests/test_pretrained.py`, `backend/tests/test_setup_check.py`, `backend/tests/test_manager.py`

**Interfaces:**
- Consumes: `devices.Device`, `devices.find_virtual_mic`, `devices.get_device`, `presets.resolve`, `StreamProcessor`, `Router`, `LiveSession`, `RvcEngine`, `make_session`, `config.load_settings`, `config.SAMPLE_RATE`.
- Produces: `pretrained.PretrainedVoice(id: str, name: str, model_path: Path, index_path: Path | None, sample_rate: int)`; `pretrained.list_voices(data: Path) -> list[PretrainedVoice]`; `pretrained.vec_model_path(data: Path) -> Path`. `setup_check.Check(id: str, ok: bool, message: str)`; `setup_check.run_checks(data: Path, devices: list[Device], providers: list[str]) -> list[Check]`. `manager.StartConfig(input_device: int, mic_device: int, voice: str, monitor_device: int | None = None, monitor_gain: float = 1.0, preset: str = "balanced", overrides: dict | None = None, pitch_shift: float = 0.0)`; `manager.LiveManager(data: Path, list_devices: Callable[[], list[Device]], engine_factory: Callable[[PretrainedVoice], VoiceEngine], stream_factory)` with `start(cfg: StartConfig)`, `stop()`, `set_mode(mode: str)`, `update(pitch_shift: float | None = None, monitor_enabled: bool | None = None, monitor_gain: float | None = None)`, `stats() -> dict`; `manager.default_engine_factory(data: Path) -> Callable[[PretrainedVoice], VoiceEngine]`. All user-facing failures raise `ValueError`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_pretrained.py`:

```python
import json

from aivoice.pretrained import list_voices, vec_model_path


def add_voice(data, voice_id, meta='{"name": "Demo", "sample_rate": 40000}', index=False, model=True):
    folder = data / "pretrained" / "rvc" / voice_id
    folder.mkdir(parents=True)
    if model:
        (folder / "model.onnx").write_bytes(b"x")
    if meta is not None:
        (folder / "voice.json").write_text(meta)
    if index:
        (folder / "model.index").write_bytes(b"x")
    return folder


def test_no_folder_means_no_voices(tmp_path):
    assert list_voices(tmp_path) == []


def test_lists_voice_with_and_without_index(tmp_path):
    add_voice(tmp_path, "a", index=True)
    add_voice(tmp_path, "b", meta=json.dumps({"name": "Bee", "sample_rate": 48000}))
    voices = list_voices(tmp_path)
    assert [(v.id, v.name, v.sample_rate) for v in voices] == [("a", "Demo", 40000), ("b", "Bee", 48000)]
    assert voices[0].index_path is not None and voices[1].index_path is None


def test_broken_voices_are_skipped_without_affecting_others(tmp_path):
    add_voice(tmp_path, "good")
    add_voice(tmp_path, "badjson", meta="{nope")
    add_voice(tmp_path, "nometa", meta=None)
    add_voice(tmp_path, "nomodel", model=False)
    add_voice(tmp_path, "nosr", meta='{"name": "x"}')
    assert [v.id for v in list_voices(tmp_path)] == ["good"]


def test_vec_model_path(tmp_path):
    assert vec_model_path(tmp_path) == tmp_path / "pretrained" / "vec-768-layer-12.onnx"
```

`backend/tests/test_setup_check.py`:

```python
from aivoice.devices import Device
from aivoice.setup_check import run_checks
from tests.test_pretrained import add_voice

MIC = Device(1, "Headset Mic", 1, 0)
VM = Device(3, "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", 0, 8)


def by_id(checks):
    return {c.id: c for c in checks}


def test_everything_missing(tmp_path):
    checks = by_id(run_checks(tmp_path, [MIC], ["CPUExecutionProvider"]))
    assert set(checks) == {"directml", "virtual_mic", "base_model", "voice"}
    assert not any(c.ok for c in checks.values())
    assert "VoiceMeeter" in checks["virtual_mic"].message
    assert "vec-768-layer-12.onnx" in checks["base_model"].message


def test_everything_present(tmp_path):
    add_voice(tmp_path, "demo")
    (tmp_path / "pretrained" / "vec-768-layer-12.onnx").write_bytes(b"x")
    checks = run_checks(tmp_path, [MIC, VM], ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert all(c.ok for c in checks)
```

Create `backend/tests/__init__.py` (empty) so `tests.test_pretrained` is importable.

`backend/tests/test_manager.py`:

```python
import numpy as np
import pytest

from aivoice.devices import Device
from aivoice.manager import LiveManager, StartConfig
from tests.test_live import FakeStreams, Identity
from tests.test_pretrained import add_voice

DEVICES = [
    Device(1, "Headset Mic", 1, 0),
    Device(2, "Headset", 0, 2),
    Device(3, "VoiceMeeter Input", 0, 8),
]


@pytest.fixture
def setup(tmp_path):
    add_voice(tmp_path, "demo")
    streams = FakeStreams()
    built = []

    def engine_factory(voice):
        built.append(voice.id)
        return Identity()

    manager = LiveManager(tmp_path, lambda: DEVICES, engine_factory, streams)
    yield manager, streams, built
    manager.stop()


def cfg(**kw):
    base = dict(input_device=1, mic_device=3, voice="demo")
    base.update(kw)
    return StartConfig(**base)


def test_start_and_stop(setup):
    manager, streams, built = setup
    manager.start(cfg())
    assert manager.stats()["running"] is True
    assert built == ["demo"]
    manager.stop()
    assert manager.stats() == {
        "running": False, "mode": "off", "error": None, "infer_ms": 0.0, "latency_ms": 0.0,
        "in_level": 0.0, "out_level": 0.0, "underruns": 0, "slow_chunks": 0, "suggest_faster": False,
    }
    assert all(s.closed for s in streams.streams)


def test_monitor_adds_second_output(setup):
    manager, streams, _ = setup
    manager.start(cfg(monitor_device=2, monitor_gain=0.5))
    assert [(s.kind, s.device) for s in streams.streams] == [("in", 1), ("out", 3), ("out", 2)]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(input_device=2), "not an input"),
        (dict(mic_device=1), "not an output"),
        (dict(monitor_device=1), "not an output"),
        (dict(monitor_device=3), "must be different"),
        (dict(input_device=99), "no longer available"),
        (dict(voice="ghost"), "Unknown voice"),
        (dict(preset="nope"), "Unknown preset"),
        (dict(overrides={"chunk_ms": 1}), "chunk_ms"),
        (dict(pitch_shift=40), "pitch_shift"),
        (dict(monitor_gain=9), "monitor_gain"),
    ],
)
def test_bad_start_requests_open_nothing(setup, kwargs, message):
    manager, streams, _ = setup
    with pytest.raises(ValueError, match=message):
        manager.start(cfg(**kwargs))
    assert streams.streams == []
    assert manager.stats()["running"] is False


def test_start_twice_restarts_cleanly(setup):
    manager, streams, _ = setup
    manager.start(cfg())
    first = list(streams.streams)
    manager.start(cfg(preset="low_latency"))
    assert all(s.closed for s in first)
    assert sum(1 for s in streams.streams if s.active) == 2


def test_stop_when_stopped_is_safe(setup):
    manager, _, _ = setup
    manager.stop()
    manager.stop()


def test_mode_change(setup):
    manager, _, _ = setup
    manager.start(cfg())
    manager.set_mode("muted")
    assert manager.stats()["mode"] == "muted"
    manager.set_mode("converted")
    assert manager.stats()["mode"] == "converted"
    with pytest.raises(ValueError, match="mode"):
        manager.set_mode("passthrough")


def test_mode_change_when_stopped_is_a_clear_error(setup):
    manager, _, _ = setup
    with pytest.raises(ValueError, match="not running"):
        manager.set_mode("muted")


def test_update_when_stopped_is_a_clear_error(setup):
    manager, _, _ = setup
    with pytest.raises(ValueError, match="not running"):
        manager.update(pitch_shift=2)


def test_update_pitch_and_monitor(setup):
    manager, _, _ = setup
    manager.start(cfg(monitor_device=2))
    manager.update(pitch_shift=5, monitor_enabled=False, monitor_gain=0.2)
    with pytest.raises(ValueError, match="pitch_shift"):
        manager.update(pitch_shift=100)


def test_engine_that_fails_to_load_reports_and_opens_nothing(tmp_path):
    add_voice(tmp_path, "demo")
    streams = FakeStreams()

    def broken(voice):
        raise RuntimeError("corrupt file")

    manager = LiveManager(tmp_path, lambda: DEVICES, broken, streams)
    with pytest.raises(ValueError, match="Could not load voice"):
        manager.start(cfg())
    assert streams.streams == []


def test_device_open_failure_is_a_clear_error(tmp_path):
    add_voice(tmp_path, "demo")
    streams = FakeStreams(fail_on=3)
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), streams)
    with pytest.raises(ValueError, match="Could not open audio devices"):
        manager.start(cfg())
    assert manager.stats()["running"] is False


def test_engine_is_warmed_up_before_streams_open(tmp_path):
    add_voice(tmp_path, "demo")
    order = []

    class Warm:
        def convert(self, audio, sample_rate, pitch_shift, params):
            order.append("convert")
            return audio

    class Streams(FakeStreams):
        def _make(self, kind, device):
            order.append("open")
            return super()._make(kind, device)

    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Warm(), Streams())
    manager.start(cfg())
    manager.stop()
    assert order[0] == "convert"
    assert np.all(np.array(order[1:3]) == "open")
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_pretrained.py backend/tests/test_setup_check.py backend/tests/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`backend/aivoice/pretrained.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path

VEC_FILE = "vec-768-layer-12.onnx"


@dataclass(frozen=True)
class PretrainedVoice:
    id: str
    name: str
    model_path: Path
    index_path: Path | None
    sample_rate: int


def vec_model_path(data: Path) -> Path:
    return data / "pretrained" / VEC_FILE


def list_voices(data: Path) -> list[PretrainedVoice]:
    root = data / "pretrained" / "rvc"
    if not root.is_dir():
        return []
    voices = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        model = folder / "model.onnx"
        try:
            meta = json.loads((folder / "voice.json").read_text())
            sample_rate = int(meta["sample_rate"])
            name = str(meta.get("name") or folder.name)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not model.is_file():
            continue
        index = folder / "model.index"
        voices.append(PretrainedVoice(folder.name, name, model, index if index.is_file() else None, sample_rate))
    return voices
```

`backend/aivoice/setup_check.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from aivoice.devices import Device, find_virtual_mic
from aivoice.pretrained import list_voices, vec_model_path


@dataclass(frozen=True)
class Check:
    id: str
    ok: bool
    message: str


def run_checks(data: Path, devices: list[Device], providers: list[str]) -> list[Check]:
    dml = "DmlExecutionProvider" in providers
    mic = find_virtual_mic(devices)
    vec = vec_model_path(data)
    voices = list_voices(data)
    return [
        Check("directml", dml, "GPU acceleration (DirectML) is available." if dml
              else "DirectML is not available. Reinstall onnxruntime-directml in the backend environment."),
        Check("virtual_mic", mic is not None, f"Virtual mic found: {mic.name}." if mic
              else "No virtual mic found. Start VoiceMeeter, then choose 'VoiceMeeter Output' as the "
                   "microphone in your other apps. Monitor-only use still works."),
        Check("base_model", vec.is_file(), "Base speech model is installed." if vec.is_file()
              else f"Base speech model is missing. Download vec-768-layer-12.onnx into {vec.parent}."),
        Check("voice", bool(voices), f"{len(voices)} voice model(s) installed." if voices
              else "No voice models installed. Export one with tools/export_rvc_onnx.py (see README)."),
    ]
```

`backend/aivoice/manager.py`:

```python
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from aivoice.config import SAMPLE_RATE, load_settings
from aivoice.devices import Device, get_device
from aivoice.live import LiveSession
from aivoice.presets import resolve
from aivoice.pretrained import PretrainedVoice, list_voices, vec_model_path
from aivoice.processor import StreamProcessor
from aivoice.router import Router

MODES = ("converted", "muted")
_IDLE = {
    "running": False, "mode": "off", "error": None, "infer_ms": 0.0, "latency_ms": 0.0,
    "in_level": 0.0, "out_level": 0.0, "underruns": 0, "slow_chunks": 0, "suggest_faster": False,
}


@dataclass
class StartConfig:
    input_device: int
    mic_device: int
    voice: str
    monitor_device: int | None = None
    monitor_gain: float = 1.0
    preset: str = "balanced"
    overrides: dict | None = None
    pitch_shift: float = 0.0


def _check_pitch(value: float) -> None:
    if not -24 <= value <= 24:
        raise ValueError("pitch_shift must be between -24 and 24 semitones")


def _check_gain(value: float) -> None:
    if not 0 <= value <= 2:
        raise ValueError("monitor_gain must be between 0 and 2")


def default_engine_factory(data: Path) -> Callable[[PretrainedVoice], object]:
    from aivoice.engines.rvc import RvcEngine, make_session

    device_id = int(load_settings(data)["dml_device_id"])

    def build(voice: PretrainedVoice):
        return RvcEngine(voice.model_path, vec_model_path(data), voice.sample_rate, voice.index_path,
                         session_factory=partial(make_session, device_id=device_id))

    return build


class LiveManager:
    def __init__(self, data: Path, list_devices: Callable[[], list[Device]], engine_factory, stream_factory,
                 sample_rate: int = SAMPLE_RATE):
        self._data = data
        self._list_devices = list_devices
        self._engine_factory = engine_factory
        self._stream_factory = stream_factory
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._session: LiveSession | None = None
        self._last_error: str | None = None

    def start(self, cfg: StartConfig) -> None:
        with self._lock:
            params = resolve(cfg.preset, cfg.overrides)
            _check_pitch(cfg.pitch_shift)
            _check_gain(cfg.monitor_gain)
            devices = self._list_devices()
            source = get_device(devices, cfg.input_device)
            if source.max_input < 1:
                raise ValueError(f"{source.name} is not an input device")
            outputs = {"mic": cfg.mic_device}
            if cfg.monitor_device is not None:
                if cfg.monitor_device == cfg.mic_device:
                    raise ValueError("Monitor and virtual mic must be different devices")
                outputs["monitor"] = cfg.monitor_device
            for index in outputs.values():
                device = get_device(devices, index)
                if device.max_output < 1:
                    raise ValueError(f"{device.name} is not an output device")
            voice = next((v for v in list_voices(self._data) if v.id == cfg.voice), None)
            if voice is None:
                raise ValueError(f"Unknown voice: {cfg.voice}")

            self._stop_locked()
            self._last_error = None
            try:
                engine = self._engine_factory(voice)
                processor = StreamProcessor(engine, self._sr, params)
                engine.convert(np.zeros(processor.window, dtype=np.float32), self._sr, 0.0, params)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Could not load voice {voice.name}: {exc}") from exc
            processor.pitch_shift = cfg.pitch_shift
            router = Router(self._sr * 2)
            router.add_output("mic")
            if "monitor" in outputs:
                router.add_output("monitor", cfg.monitor_gain)
            session = LiveSession(processor, router, self._sr, self._stream_factory, cfg.input_device, outputs)
            try:
                session.start()
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Could not open audio devices: {exc}") from exc
            self._session = session

    def _stop_locked(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()
            self._last_error = None

    def _require(self) -> LiveSession:
        if self._session is None or not self._session.running:
            raise ValueError("Live mode is not running")
        return self._session

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._require().processor.mode = mode

    def update(self, pitch_shift: float | None = None, monitor_enabled: bool | None = None,
               monitor_gain: float | None = None) -> None:
        session = self._require()
        if pitch_shift is not None:
            _check_pitch(pitch_shift)
            session.processor.pitch_shift = pitch_shift
        if session.router.has("monitor"):
            if monitor_gain is not None:
                _check_gain(monitor_gain)
                session.router.set_gain("monitor", monitor_gain)
            if monitor_enabled is not None:
                session.router.set_enabled("monitor", monitor_enabled)

    def stats(self) -> dict:
        session = self._session
        if session is None:
            return {**_IDLE, "error": self._last_error}
        stats = session.stats()
        if not stats["running"]:
            self._last_error = stats["error"]
            return {**_IDLE, "error": stats["error"]}
        return stats
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests -v`
Expected: all tests pass, including the earlier tasks.

- [ ] **Step 5: Commit**

```bash
git add backend/aivoice/pretrained.py backend/aivoice/setup_check.py backend/aivoice/manager.py backend/tests
git commit -m "feat: voice discovery, setup check, and live manager"
```

---

### Task 10: HTTP and WebSocket API

**Files:**
- Modify: `backend/aivoice/api.py` (replace whole file)
- Modify: `backend/aivoice/__main__.py` (replace `main`)
- Modify: `backend/tests/test_health.py` (replace whole file)
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `LiveManager`, `StartConfig`, `default_engine_factory`, `list_devices`, `find_virtual_mic`, `SoundDeviceStreams`, `PRESETS`, `list_voices`, `run_checks`, `config.data_dir`.
- Produces: `api.create_app(manager, data: Path, list_devices: Callable[[], list[Device]], providers: list[str]) -> FastAPI`. Routes:
  - `GET /health` returns `{"status": "ok"}`
  - `GET /devices` returns `{"devices": [{"index", "name", "max_input", "max_output"}], "virtual_mic": int | null}`
  - `GET /voices` returns `[{"id", "name", "has_index"}]`
  - `GET /presets` returns `{name: {"chunk_ms", "crossfade_ms", "context_ms", "f0_method", "index_rate"}}`
  - `GET /setup-check` returns `[{"id", "ok", "message"}]`
  - `POST /live/start` body = `StartConfig` fields; returns stats
  - `POST /live/stop` returns stats
  - `POST /live/mode` body `{"mode": "converted" | "muted"}`; returns stats
  - `PATCH /live/settings` body `{"pitch_shift"?, "monitor_enabled"?, "monitor_gain"?}`; returns stats
  - `GET /live/stats` returns stats
  - `WS /ws` sends stats JSON every 100 ms
  - Any `ValueError` becomes HTTP 400 `{"detail": "<message>"}`.

- [ ] **Step 1: Write the failing tests**

Replace `backend/tests/test_health.py`:

```python
from tests.test_api import make_client


def test_health(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
```

`backend/tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from aivoice.api import create_app
from aivoice.manager import LiveManager
from aivoice.presets import PRESETS
from tests.test_live import FakeStreams, Identity
from tests.test_manager import DEVICES
from tests.test_pretrained import add_voice


def make_client(tmp_path, providers=("DmlExecutionProvider",)):
    add_voice(tmp_path, "demo", index=True)
    streams = FakeStreams()
    manager = LiveManager(tmp_path, lambda: DEVICES, lambda v: Identity(), streams)
    return TestClient(create_app(manager, tmp_path, lambda: DEVICES, list(providers))), streams


START = {"input_device": 1, "mic_device": 3, "voice": "demo"}


def test_devices(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/devices").json()
    assert body["virtual_mic"] == 3
    assert body["devices"][0] == {"index": 1, "name": "Headset Mic", "max_input": 1, "max_output": 0}


def test_voices(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/voices").json() == [{"id": "demo", "name": "Demo", "has_index": True}]


def test_presets(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/presets").json()
    assert set(body) == set(PRESETS)
    assert body["balanced"]["chunk_ms"] == PRESETS["balanced"].chunk_ms


def test_setup_check(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/setup-check").json()
    assert {c["id"] for c in body} == {"directml", "virtual_mic", "base_model", "voice"}
    assert next(c for c in body if c["id"] == "base_model")["ok"] is False


def test_start_mode_settings_stop(tmp_path):
    client, streams = make_client(tmp_path)
    assert client.post("/live/start", json={**START, "monitor_device": 2}).json()["running"] is True
    assert client.post("/live/mode", json={"mode": "muted"}).json()["mode"] == "muted"
    assert client.patch("/live/settings", json={"pitch_shift": 3, "monitor_gain": 0.5}).status_code == 200
    assert client.get("/live/stats").json()["running"] is True
    assert client.post("/live/stop").json()["running"] is False
    assert all(s.closed for s in streams.streams)


def test_bad_requests_return_400_with_message_and_open_nothing(tmp_path):
    client, streams = make_client(tmp_path)
    cases = [
        ({**START, "monitor_device": 3}, "must be different"),
        ({**START, "input_device": 2}, "not an input"),
        ({**START, "voice": "ghost"}, "Unknown voice"),
        ({**START, "overrides": {"chunk_ms": 1}}, "chunk_ms"),
    ]
    for body, message in cases:
        res = client.post("/live/start", json=body)
        assert res.status_code == 400 and message in res.json()["detail"]
    assert streams.streams == []


def test_missing_field_is_422(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/live/start", json={"voice": "demo"}).status_code == 422


def test_mode_when_stopped_is_400(tmp_path):
    client, _ = make_client(tmp_path)
    res = client.post("/live/mode", json={"mode": "muted"})
    assert res.status_code == 400 and "not running" in res.json()["detail"]


def test_stop_when_stopped_is_ok(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/live/stop").status_code == 200


def test_websocket_streams_stats(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["running"] is False and first["mode"] == "off"
        client.post("/live/start", json=START)
        seen_running = False
        for _ in range(20):
            if ws.receive_json()["running"]:
                seen_running = True
                break
        assert seen_running
    client.post("/live/stop")
```

- [ ] **Step 2: Run to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_api.py backend/tests/test_health.py -v`
Expected: FAIL with `TypeError: create_app() takes 0 positional arguments`.

- [ ] **Step 3: Implement**

Replace `backend/aivoice/api.py`:

```python
import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aivoice.devices import Device, find_virtual_mic
from aivoice.manager import LiveManager, StartConfig
from aivoice.presets import PRESETS
from aivoice.pretrained import list_voices
from aivoice.setup_check import run_checks


class StartBody(BaseModel):
    input_device: int
    mic_device: int
    voice: str
    monitor_device: int | None = None
    monitor_gain: float = 1.0
    preset: str = "balanced"
    overrides: dict | None = None
    pitch_shift: float = 0.0


class ModeBody(BaseModel):
    mode: str


class SettingsBody(BaseModel):
    pitch_shift: float | None = None
    monitor_enabled: bool | None = None
    monitor_gain: float | None = None


def create_app(manager: LiveManager, data: Path, list_devices: Callable[[], list[Device]],
               providers: list[str]) -> FastAPI:
    app = FastAPI(title="AI Voice")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/devices")
    def devices() -> dict:
        found = list_devices()
        mic = find_virtual_mic(found)
        return {"devices": [asdict(d) for d in found], "virtual_mic": mic.index if mic else None}

    @app.get("/voices")
    def voices() -> list[dict]:
        return [{"id": v.id, "name": v.name, "has_index": v.index_path is not None} for v in list_voices(data)]

    @app.get("/presets")
    def presets() -> dict:
        return {name: asdict(p) for name, p in PRESETS.items()}

    @app.get("/setup-check")
    def setup_check() -> list[dict]:
        return [asdict(c) for c in run_checks(data, list_devices(), providers)]

    @app.post("/live/start")
    def start(body: StartBody) -> dict:
        manager.start(StartConfig(**body.model_dump()))
        return manager.stats()

    @app.post("/live/stop")
    def stop() -> dict:
        manager.stop()
        return manager.stats()

    @app.post("/live/mode")
    def mode(body: ModeBody) -> dict:
        manager.set_mode(body.mode)
        return manager.stats()

    @app.patch("/live/settings")
    def settings(body: SettingsBody) -> dict:
        manager.update(**body.model_dump())
        return manager.stats()

    @app.get("/live/stats")
    def stats() -> dict:
        return manager.stats()

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                await socket.send_json(manager.stats())
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app
```

In `backend/aivoice/__main__.py`, replace the `from aivoice.api import create_app` import and the last line of `main` so the file reads:

```python
import argparse
import os
import threading
import time

import psutil
import uvicorn

from aivoice import config
from aivoice.api import create_app
from aivoice.devices import SoundDeviceStreams, list_devices
from aivoice.manager import LiveManager, default_engine_factory


def _exit_when_parent_dies(pid: int) -> None:
    while psutil.pid_exists(pid):
        time.sleep(1.0)
    os._exit(0)


def build_app():
    import onnxruntime as ort

    data = config.data_dir()
    manager = LiveManager(data, list_devices, default_engine_factory(data), SoundDeviceStreams())
    return create_app(manager, data, list_devices, ort.get_available_providers())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.parent_pid:
        threading.Thread(target=_exit_when_parent_dies, args=(args.parent_pid,), daemon=True).start()
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests -v`
Expected: all pass.

- [ ] **Step 5: Smoke test the real server**

Run in one terminal: `backend/.venv/Scripts/python -m aivoice --port 8765`
Run in another: `curl -s http://127.0.0.1:8765/setup-check`
Expected: JSON with all four checks `"ok": true` (models from Task 1 are present and VoiceMeeter is installed). Then `curl -s http://127.0.0.1:8765/devices` lists real devices. Stop the server with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add backend
git commit -m "feat: HTTP and WebSocket API for live mode"
```

---

### Task 11: Tauri shell

The shell picks a free port, spawns the backend, restarts it if it dies, and exposes a tray menu and a global hotkey. It holds no audio logic. In stage 1 the backend is run from source at `backend/.venv`; packaging an installer is out of scope.

**Files:**
- Create: `app/` via the Tauri scaffolder
- Modify: `app/src-tauri/Cargo.toml`
- Modify: `app/src-tauri/tauri.conf.json`
- Modify: `app/src-tauri/src/lib.rs` (replace whole file)

**Interfaces:**
- Consumes: backend CLI `python -m aivoice --port <n> --parent-pid <pid>`.
- Produces: Tauri command `backend_port() -> u16`. Events emitted to the frontend: `live-toggle` (no payload, from hotkey Ctrl+Alt+V) and `live-mode` (payload string `"converted"`, `"muted"` or `"off"`, from the tray menu).

- [ ] **Step 1: Scaffold the project**

```bash
npm create tauri-app@latest app -- --template react-ts --manager npm --identifier gg.horizons.aivoice --yes
cd app && npm install
```

If the scaffolder ignores the flags and prompts, answer: project name `app`, identifier `gg.horizons.aivoice`, frontend language TypeScript, package manager npm, UI template React, flavor TypeScript.

- [ ] **Step 2: Add Rust dependencies**

Run from `app/src-tauri`:

```bash
cargo add tauri-plugin-global-shortcut@2
cargo add tauri@2 --features tray-icon
```

- [ ] **Step 3: Configure the window and content policy**

In `app/src-tauri/tauri.conf.json`, set these values (leave the rest of the generated file as is):

```json
{
  "productName": "AI Voice",
  "app": {
    "windows": [{ "title": "AI Voice", "width": 520, "height": 760, "resizable": true }],
    "security": { "csp": null }
  }
}
```

`csp: null` lets the page call the backend on `http://127.0.0.1:<port>`.

- [ ] **Step 4: Write `app/src-tauri/src/lib.rs`**

```rust
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{Emitter, Manager, RunEvent, State};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

struct Backend {
    port: u16,
    child: Arc<Mutex<Option<Child>>>,
    quitting: Arc<AtomicBool>,
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("no free port")
}

fn backend_dir() -> PathBuf {
    match std::env::var("AIVOICE_BACKEND_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("backend"),
    }
}

fn spawn_backend(port: u16) -> std::io::Result<Child> {
    let dir = backend_dir();
    let mut cmd = Command::new(dir.join(".venv").join("Scripts").join("python.exe"));
    cmd.current_dir(&dir).args([
        "-m",
        "aivoice",
        "--port",
        &port.to_string(),
        "--parent-pid",
        &std::process::id().to_string(),
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.spawn()
}

/// Keeps the backend alive. A restarted backend has no live session, so live mode comes back stopped.
fn supervise(port: u16, child: Arc<Mutex<Option<Child>>>, quitting: Arc<AtomicBool>) {
    thread::spawn(move || {
        while !quitting.load(Ordering::SeqCst) {
            match spawn_backend(port) {
                Ok(process) => *child.lock().unwrap() = Some(process),
                Err(err) => eprintln!("failed to start backend: {err}"),
            }
            loop {
                thread::sleep(Duration::from_millis(500));
                if quitting.load(Ordering::SeqCst) {
                    return;
                }
                let mut guard = child.lock().unwrap();
                let exited = match guard.as_mut() {
                    Some(process) => !matches!(process.try_wait(), Ok(None)),
                    None => true,
                };
                if exited {
                    *guard = None;
                    break;
                }
            }
            thread::sleep(Duration::from_secs(1));
        }
    });
}

#[tauri::command]
fn backend_port(state: State<Backend>) -> u16 {
    state.port
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = free_port();
    let child = Arc::new(Mutex::new(None));
    let quitting = Arc::new(AtomicBool::new(false));
    supervise(port, child.clone(), quitting.clone());

    let app = tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        let _ = app.emit("live-toggle", ());
                    }
                })
                .build(),
        )
        .manage(Backend { port, child: child.clone(), quitting: quitting.clone() })
        .invoke_handler(tauri::generate_handler![backend_port])
        .setup(|app| {
            app.global_shortcut()
                .register(Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyV))?;

            let converted = MenuItem::with_id(app, "converted", "Converted", true, None::<&str>)?;
            let muted = MenuItem::with_id(app, "muted", "Muted", true, None::<&str>)?;
            let off = MenuItem::with_id(app, "off", "Off", true, None::<&str>)?;
            let show = MenuItem::with_id(app, "show", "Show window", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&converted, &muted, &off, &show, &quit])?;
            TrayIconBuilder::new()
                .icon(app.default_window_icon().expect("window icon").clone())
                .tooltip("AI Voice")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => app.exit(0),
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    mode => {
                        let _ = app.emit("live-mode", mode.to_string());
                    }
                })
                .build(app)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |_handle, event| {
        if let RunEvent::Exit = event {
            quitting.store(true, Ordering::SeqCst);
            if let Some(mut process) = child.lock().unwrap().take() {
                let _ = process.kill();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_port_is_bindable() {
        let port = free_port();
        assert!(port > 0);
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    #[test]
    fn backend_dir_honours_env() {
        std::env::set_var("AIVOICE_BACKEND_DIR", "C:/x/backend");
        assert_eq!(backend_dir(), PathBuf::from("C:/x/backend"));
        std::env::remove_var("AIVOICE_BACKEND_DIR");
    }
}
```

- [ ] **Step 5: Run the Rust tests**

Run from `app/src-tauri`: `cargo test`
Expected: 2 passed. If the Tauri 2 API has moved (for example an `Emitter` or tray import path), fix the imports per the compiler's suggestions; the behaviour above is the requirement.

- [ ] **Step 6: Verify supervision by hand**

Run from `app`: `npm run tauri dev`. Then in PowerShell:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*aivoice*' | Select-Object ProcessId, CommandLine
```

Expected: one `python.exe -m aivoice --port <n> --parent-pid <pid>`. Kill it with `Stop-Process -Id <ProcessId>`; within about two seconds the same query shows a new process on the same port. Close the app window and quit from the tray; the query then returns nothing. The tray icon shows Converted, Muted, Off, Show window, Quit.

- [ ] **Step 7: Commit**

```bash
git add app
git commit -m "feat: Tauri shell that supervises the backend with tray and hotkey"
```

---

### Task 12: Frontend

**Files:**
- Create: `app/src/api.ts`, `app/src/useStats.ts`, `app/src/SetupCheck.tsx`, `app/src/LivePanel.tsx`, `app/src/test-setup.ts`
- Modify: `app/src/App.tsx` (replace), `app/src/App.css` (replace), `app/vite.config.ts`, `app/package.json` (add `test` script)
- Test: `app/src/SetupCheck.test.tsx`, `app/src/LivePanel.test.tsx`

**Interfaces:**
- Consumes: every route from Task 10; Tauri command `backend_port`; events `live-toggle` and `live-mode` from Task 11.
- Produces: `api` object with `health()`, `devices()`, `voices()`, `presets()`, `setupCheck()`, `start(req)`, `stop()`, `mode(mode)`, `settings(patch)`, plus `wsUrl()`; types `Device`, `Voice`, `Params`, `Check`, `Stats`, `StartRequest`.

- [ ] **Step 1: Install test tooling**

Run from `app`:

```bash
npm install -D vitest jsdom @testing-library/react @testing-library/user-event @testing-library/jest-dom
```

Add to `app/package.json` scripts: `"test": "vitest run"`.

In `app/vite.config.ts`, change the first import to `import { defineConfig } from "vitest/config";` and add this key inside the config object returned to `defineConfig`:

```ts
  test: { environment: "jsdom", globals: true, setupFiles: "./src/test-setup.ts" },
```

`app/src/test-setup.ts`:

```ts
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 2: Write `app/src/api.ts`**

```ts
import { invoke } from "@tauri-apps/api/core";

export type Device = { index: number; name: string; max_input: number; max_output: number };
export type Voice = { id: string; name: string; has_index: boolean };
export type Params = {
  chunk_ms: number;
  crossfade_ms: number;
  context_ms: number;
  f0_method: "dio" | "harvest";
  index_rate: number;
};
export type Check = { id: string; ok: boolean; message: string };
export type Mode = "converted" | "muted";
export type Stats = {
  running: boolean;
  mode: Mode | "off";
  error: string | null;
  infer_ms: number;
  latency_ms: number;
  in_level: number;
  out_level: number;
  underruns: number;
  slow_chunks: number;
  suggest_faster: boolean;
};
export type StartRequest = {
  input_device: number;
  mic_device: number;
  voice: string;
  monitor_device: number | null;
  monitor_gain: number;
  preset: string;
  overrides: Partial<Params> | null;
  pitch_shift: number;
};
export type SettingsPatch = { pitch_shift?: number; monitor_enabled?: boolean; monitor_gain?: number };

let port: number | null = null;

async function origin(): Promise<string> {
  if (port === null) port = await invoke<number>("backend_port");
  return `127.0.0.1:${port}`;
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`http://${await origin()}${path}`, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<{ status: string }>("GET", "/health"),
  devices: () => call<{ devices: Device[]; virtual_mic: number | null }>("GET", "/devices"),
  voices: () => call<Voice[]>("GET", "/voices"),
  presets: () => call<Record<string, Params>>("GET", "/presets"),
  setupCheck: () => call<Check[]>("GET", "/setup-check"),
  start: (req: StartRequest) => call<Stats>("POST", "/live/start", req),
  stop: () => call<Stats>("POST", "/live/stop"),
  mode: (mode: Mode) => call<Stats>("POST", "/live/mode", { mode }),
  settings: (patch: SettingsPatch) => call<Stats>("PATCH", "/live/settings", patch),
  wsUrl: async () => `ws://${await origin()}/ws`,
};
```

- [ ] **Step 3: Write `app/src/useStats.ts`**

```ts
import { useEffect, useState } from "react";
import { api, Stats } from "./api";

export const IDLE: Stats = {
  running: false, mode: "off", error: null, infer_ms: 0, latency_ms: 0,
  in_level: 0, out_level: 0, underruns: 0, slow_chunks: 0, suggest_faster: false,
};

/** Live stats over WebSocket. Reconnects forever, so a restarted backend shows up as stopped. */
export function useStats(): Stats {
  const [stats, setStats] = useState<Stats>(IDLE);
  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = async () => {
      socket = new WebSocket(await api.wsUrl());
      socket.onmessage = (event) => setStats(JSON.parse(event.data));
      socket.onclose = () => {
        setStats((prev) => ({ ...IDLE, error: prev.error }));
        if (!closed) timer = setTimeout(connect, 1000);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, []);
  return stats;
}
```

- [ ] **Step 4: Write the failing component tests**

`app/src/SetupCheck.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SetupCheck } from "./SetupCheck";

vi.mock("./api", () => ({ api: { setupCheck: vi.fn() } }));
import { api } from "./api";

describe("SetupCheck", () => {
  it("shows nothing when every check passes", async () => {
    vi.mocked(api.setupCheck).mockResolvedValue([{ id: "directml", ok: true, message: "fine" }]);
    const { container } = render(<SetupCheck />);
    await vi.waitFor(() => expect(api.setupCheck).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("lists only the failing checks with their messages", async () => {
    vi.mocked(api.setupCheck).mockResolvedValue([
      { id: "directml", ok: true, message: "fine" },
      { id: "voice", ok: false, message: "No voice models installed." },
    ]);
    render(<SetupCheck />);
    expect(await screen.findByText("No voice models installed.")).toBeInTheDocument();
    expect(screen.queryByText("fine")).not.toBeInTheDocument();
  });
});
```

`app/src/LivePanel.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LivePanel } from "./LivePanel";
import { IDLE } from "./useStats";

vi.mock("./api", () => ({
  api: {
    devices: vi.fn(), voices: vi.fn(), presets: vi.fn(),
    start: vi.fn(), stop: vi.fn(), mode: vi.fn(), settings: vi.fn(),
  },
}));
import { api } from "./api";

const PRESETS = {
  low_latency: { chunk_ms: 120, crossfade_ms: 30, context_ms: 400, f0_method: "dio", index_rate: 0 },
  balanced: { chunk_ms: 250, crossfade_ms: 50, context_ms: 800, f0_method: "dio", index_rate: 0.3 },
  high_quality: { chunk_ms: 500, crossfade_ms: 80, context_ms: 1500, f0_method: "harvest", index_rate: 0.6 },
} as const;

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.devices).mockResolvedValue({
    devices: [
      { index: 1, name: "Headset Mic", max_input: 1, max_output: 0 },
      { index: 2, name: "Headset", max_input: 0, max_output: 2 },
      { index: 3, name: "VoiceMeeter Input", max_input: 0, max_output: 8 },
    ],
    virtual_mic: 3,
  });
  vi.mocked(api.voices).mockResolvedValue([{ id: "demo", name: "Demo", has_index: true }]);
  vi.mocked(api.presets).mockResolvedValue(PRESETS as never);
  vi.mocked(api.start).mockResolvedValue({ ...IDLE, running: true, mode: "converted" });
  vi.mocked(api.stop).mockResolvedValue(IDLE);
  vi.mocked(api.mode).mockResolvedValue({ ...IDLE, running: true, mode: "muted" });
  vi.mocked(api.settings).mockResolvedValue({ ...IDLE, running: true, mode: "converted" });
});

async function ready() {
  const button = await screen.findByRole("button", { name: "Start" });
  await vi.waitFor(() => expect(button).toBeEnabled());
  return button;
}

describe("LivePanel", () => {
  it("preselects the virtual mic and starts with the chosen settings", async () => {
    render(<LivePanel stats={IDLE} />);
    await userEvent.click(await ready());
    expect(api.start).toHaveBeenCalledWith({
      input_device: 1, mic_device: 3, voice: "demo", monitor_device: null, monitor_gain: 1,
      preset: "balanced", overrides: null, pitch_shift: 0,
    });
  });

  it("only offers inputs as the microphone and outputs as destinations", async () => {
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.getByLabelText("Microphone").querySelectorAll("option")).toHaveLength(1);
    expect(screen.getByLabelText("Virtual mic").querySelectorAll("option")).toHaveLength(2);
  });

  it("sends the monitor device when monitoring is on", async () => {
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.click(screen.getByLabelText("Listen to output"));
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].monitor_device).toBe(2);
  });

  it("sends advanced overrides only when changed", async () => {
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.click(screen.getByText("Advanced"));
    const chunk = screen.getByLabelText("Chunk (ms)");
    await userEvent.clear(chunk);
    await userEvent.type(chunk, "300");
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].overrides).toEqual({ chunk_ms: 300 });
  });

  it("shows the backend's error message when start fails", async () => {
    vi.mocked(api.start).mockRejectedValue(new Error("Monitor and virtual mic must be different devices"));
    render(<LivePanel stats={IDLE} />);
    await userEvent.click(await ready());
    expect(await screen.findByRole("alert")).toHaveTextContent("must be different");
  });

  it("shows Stop and Mute while running, plus latency", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted", latency_ms: 312.4, infer_ms: 40 }} />);
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(api.stop).toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Mute" }));
    expect(api.mode).toHaveBeenCalledWith("muted");
    expect(screen.getByText(/312 ms/)).toBeInTheDocument();
  });

  it("recommends a faster preset when the GPU cannot keep up", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted", suggest_faster: true }} />);
    expect(await screen.findByText(/try a faster preset/i)).toBeInTheDocument();
  });

  it("surfaces a session error from stats", async () => {
    render(<LivePanel stats={{ ...IDLE, error: "An audio device was disconnected" }} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("disconnected");
  });

  it("explains when no voices are installed", async () => {
    vi.mocked(api.voices).mockResolvedValue([]);
    render(<LivePanel stats={IDLE} />);
    expect(await screen.findByText(/no voice models/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
  });
});
```

- [ ] **Step 5: Run to verify failure**

Run from `app`: `npm test`
Expected: FAIL, cannot resolve `./SetupCheck` and `./LivePanel`.

- [ ] **Step 6: Implement the components**

`app/src/SetupCheck.tsx`:

```tsx
import { useEffect, useState } from "react";
import { api, Check } from "./api";

export function SetupCheck() {
  const [failing, setFailing] = useState<Check[]>([]);
  useEffect(() => {
    api.setupCheck().then((checks) => setFailing(checks.filter((c) => !c.ok))).catch(() => setFailing([]));
  }, []);
  if (failing.length === 0) return null;
  return (
    <section className="setup">
      <h2>Setup needed</h2>
      <ul>
        {failing.map((c) => (
          <li key={c.id}>{c.message}</li>
        ))}
      </ul>
    </section>
  );
}
```

`app/src/LivePanel.tsx`:

```tsx
import { useEffect, useMemo, useState } from "react";
import { api, Device, Params, Stats, Voice } from "./api";

const PRESET_LABELS: Record<string, string> = {
  low_latency: "Low latency",
  balanced: "Balanced",
  high_quality: "High quality",
};

function Meter({ label, level }: { label: string; level: number }) {
  return (
    <label className="meter">
      {label}
      <progress max={1} value={Math.min(1, level)} />
    </label>
  );
}

export function LivePanel({ stats }: { stats: Stats }) {
  const [devices, setDevices] = useState<Device[]>([]);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [presets, setPresets] = useState<Record<string, Params>>({});
  const [input, setInput] = useState<number | null>(null);
  const [mic, setMic] = useState<number | null>(null);
  const [monitorOn, setMonitorOn] = useState(false);
  const [monitor, setMonitor] = useState<number | null>(null);
  const [monitorGain, setMonitorGain] = useState(1);
  const [voice, setVoice] = useState("");
  const [preset, setPreset] = useState("balanced");
  const [overrides, setOverrides] = useState<Partial<Params>>({});
  const [pitch, setPitch] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const inputs = useMemo(() => devices.filter((d) => d.max_input > 0), [devices]);
  const outputs = useMemo(() => devices.filter((d) => d.max_output > 0), [devices]);

  useEffect(() => {
    Promise.all([api.devices(), api.voices(), api.presets()])
      .then(([d, v, p]) => {
        setDevices(d.devices);
        setVoices(v);
        setPresets(p);
        const ins = d.devices.filter((x) => x.max_input > 0);
        const outs = d.devices.filter((x) => x.max_output > 0);
        setInput(ins[0]?.index ?? null);
        setMic(d.virtual_mic ?? outs[0]?.index ?? null);
        setMonitor(outs.find((x) => x.index !== d.virtual_mic)?.index ?? null);
        setVoice(v[0]?.id ?? "");
        setLoaded(true);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const start = () =>
    run(() =>
      api.start({
        input_device: input as number,
        mic_device: mic as number,
        voice,
        monitor_device: monitorOn ? monitor : null,
        monitor_gain: monitorGain,
        preset,
        overrides: Object.keys(overrides).length ? overrides : null,
        pitch_shift: pitch,
      }),
    );

  const base = presets[preset];
  const value = <K extends keyof Params>(key: K): Params[K] | "" =>
    (overrides[key] ?? base?.[key] ?? "") as Params[K] | "";
  const override = <K extends keyof Params>(key: K, v: Params[K]) =>
    setOverrides((prev) => {
      const next = { ...prev };
      if (base && base[key] === v) delete next[key];
      else next[key] = v;
      return next;
    });

  const canStart = !busy && input !== null && mic !== null && voice !== "";
  const shownError = error ?? stats.error;

  return (
    <section className="live">
      {shownError && <p role="alert" className="error">{shownError}</p>}
      {loaded && voices.length === 0 && <p className="hint">No voice models installed. See the README to add one.</p>}

      <label>
        Microphone
        <select value={input ?? ""} onChange={(e) => setInput(Number(e.target.value))} disabled={stats.running}>
          {inputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      <label>
        Virtual mic
        <select value={mic ?? ""} onChange={(e) => setMic(Number(e.target.value))} disabled={stats.running}>
          {outputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      <label className="row">
        <input
          type="checkbox"
          checked={monitorOn}
          disabled={stats.running && !monitorOn}
          onChange={(e) => {
            setMonitorOn(e.target.checked);
            if (stats.running) run(() => api.settings({ monitor_enabled: e.target.checked }));
          }}
        />
        Listen to output
      </label>
      {monitorOn && (
        <>
          <label>
            Listen on
            <select value={monitor ?? ""} onChange={(e) => setMonitor(Number(e.target.value))} disabled={stats.running}>
              {outputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
            </select>
          </label>
          <label>
            Listen volume
            <input
              type="range" min={0} max={2} step={0.05} value={monitorGain}
              onChange={(e) => {
                const gain = Number(e.target.value);
                setMonitorGain(gain);
                if (stats.running) api.settings({ monitor_gain: gain }).catch(() => undefined);
              }}
            />
          </label>
        </>
      )}
      <label>
        Voice
        <select value={voice} onChange={(e) => setVoice(e.target.value)} disabled={stats.running}>
          {voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
        </select>
      </label>
      <label>
        Speed / quality
        <select
          value={preset}
          disabled={stats.running}
          onChange={(e) => {
            setPreset(e.target.value);
            setOverrides({});
          }}
        >
          {Object.keys(presets).map((p) => <option key={p} value={p}>{PRESET_LABELS[p] ?? p}</option>)}
        </select>
      </label>
      <label>
        Pitch shift ({pitch > 0 ? "+" : ""}{pitch} semitones)
        <input
          type="range" min={-24} max={24} step={1} value={pitch}
          onChange={(e) => {
            const semitones = Number(e.target.value);
            setPitch(semitones);
            if (stats.running) api.settings({ pitch_shift: semitones }).catch(() => undefined);
          }}
        />
      </label>

      <details>
        <summary>Advanced</summary>
        <label>
          Chunk (ms)
          <input type="number" value={value("chunk_ms")} disabled={stats.running}
            onChange={(e) => override("chunk_ms", Number(e.target.value))} />
        </label>
        <label>
          Crossfade (ms)
          <input type="number" value={value("crossfade_ms")} disabled={stats.running}
            onChange={(e) => override("crossfade_ms", Number(e.target.value))} />
        </label>
        <label>
          Context (ms)
          <input type="number" value={value("context_ms")} disabled={stats.running}
            onChange={(e) => override("context_ms", Number(e.target.value))} />
        </label>
        <label>
          Pitch detector
          <select value={value("f0_method")} disabled={stats.running}
            onChange={(e) => override("f0_method", e.target.value as Params["f0_method"])}>
            <option value="dio">Fast (dio)</option>
            <option value="harvest">Accurate (harvest)</option>
          </select>
        </label>
        <label>
          Index blend
          <input type="number" min={0} max={1} step={0.05} value={value("index_rate")} disabled={stats.running}
            onChange={(e) => override("index_rate", Number(e.target.value))} />
        </label>
      </details>

      <div className="actions">
        {stats.running ? (
          <>
            <button onClick={() => run(() => api.stop())} disabled={busy}>Stop</button>
            {stats.mode === "muted" ? (
              <button onClick={() => run(() => api.mode("converted"))} disabled={busy}>Unmute</button>
            ) : (
              <button onClick={() => run(() => api.mode("muted"))} disabled={busy}>Mute</button>
            )}
          </>
        ) : (
          <button onClick={start} disabled={!canStart}>Start</button>
        )}
      </div>

      {stats.running && (
        <div className="readout">
          <p>
            Latency {Math.round(stats.latency_ms)} ms · Inference {Math.round(stats.infer_ms)} ms · Dropouts{" "}
            {stats.underruns}
          </p>
          <Meter label="Input" level={stats.in_level} />
          <Meter label="Output" level={stats.out_level} />
          {stats.suggest_faster && <p className="hint">The GPU is not keeping up. Try a faster preset.</p>}
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 7: Run to verify pass**

Run from `app`: `npm test`
Expected: 11 passed.

- [ ] **Step 8: Write `App.tsx` and styles**

Replace `app/src/App.tsx`:

```tsx
import { listen } from "@tauri-apps/api/event";
import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import "./App.css";
import { LivePanel } from "./LivePanel";
import { SetupCheck } from "./SetupCheck";
import { useStats } from "./useStats";

function useBackendReady(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let stop = false;
    const poll = async () => {
      while (!stop) {
        try {
          await api.health();
          setReady(true);
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 300));
        }
      }
    };
    poll();
    return () => {
      stop = true;
    };
  }, []);
  return ready;
}

function Main() {
  const stats = useStats();
  const latest = useRef(stats);
  latest.current = stats;

  useEffect(() => {
    const toggle = listen("live-toggle", () => {
      const s = latest.current;
      if (s.running) api.mode(s.mode === "muted" ? "converted" : "muted").catch(() => undefined);
    });
    const mode = listen<string>("live-mode", (event) => {
      if (event.payload === "off") api.stop().catch(() => undefined);
      else if (latest.current.running) api.mode(event.payload as "converted" | "muted").catch(() => undefined);
    });
    return () => {
      toggle.then((off) => off());
      mode.then((off) => off());
    };
  }, []);

  return (
    <>
      <SetupCheck />
      <LivePanel stats={stats} />
    </>
  );
}

export default function App() {
  const ready = useBackendReady();
  return (
    <main>
      <h1>AI Voice</h1>
      {ready ? <Main /> : <p>Starting the voice engine…</p>}
      <footer>Ctrl+Alt+V mutes and unmutes while live.</footer>
    </main>
  );
}
```

Replace `app/src/App.css`:

```css
:root { color-scheme: dark; font-family: "Segoe UI", system-ui, sans-serif; background: #15171c; color: #e8eaed; }
main { max-width: 440px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.3rem; margin: 0 0 12px; }
h2 { font-size: 1rem; margin: 0 0 6px; }
label { display: block; margin: 10px 0; font-size: 0.85rem; color: #aab0bb; }
label.row { display: flex; gap: 8px; align-items: center; color: #e8eaed; }
select, input[type="number"], input[type="range"] { display: block; width: 100%; margin-top: 4px; }
select, input[type="number"] { padding: 6px; background: #22252c; color: inherit; border: 1px solid #343843; border-radius: 6px; }
button { padding: 8px 18px; border: 0; border-radius: 6px; background: #4f7cff; color: #fff; font-weight: 600; cursor: pointer; }
button:disabled { opacity: 0.5; cursor: default; }
.actions { display: flex; gap: 8px; margin: 16px 0; }
.error { background: #4a1f24; color: #ffb4bb; padding: 8px 10px; border-radius: 6px; }
.hint { color: #f2c66d; font-size: 0.85rem; }
.setup { background: #2a2618; padding: 10px 12px; border-radius: 6px; margin-bottom: 12px; }
.setup ul { margin: 0; padding-left: 18px; font-size: 0.85rem; }
.meter progress { width: 100%; height: 8px; }
details { margin: 10px 0; }
summary { cursor: pointer; color: #aab0bb; font-size: 0.85rem; }
footer { margin-top: 20px; font-size: 0.75rem; color: #7d8491; }
```

- [ ] **Step 9: Type-check and re-run tests**

Run from `app`: `npx tsc --noEmit && npm test`
Expected: no type errors, 11 passed.

- [ ] **Step 10: Commit**

```bash
git add app
git commit -m "feat: live voice change UI with setup check and stats"
```

---

### Task 13: End-to-end check, preset tuning, and README

**Files:**
- Create: `README.md`
- Modify: `backend/aivoice/presets.py` (only if tuning changes numbers)
- Modify: `backend/tests/test_presets.py` (only if tuning changes numbers the tests pin)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Run every automated test**

```bash
backend/.venv/Scripts/python -m pytest backend/tests -v
cd app && npm test && npx tsc --noEmit && cd src-tauri && cargo test
```

Expected: all pass, with `test_rvc_real.py` passing rather than skipped.

- [ ] **Step 2: Manual end-to-end check with the human partner**

Run from `app`: `npm run tauri dev`. The human partner must do the listening. Confirm each item:

1. The window opens and shows no "Setup needed" box.
2. Virtual mic is preselected as a VoiceMeeter Input device.
3. Tick "Listen to output", pick the headset, choose Balanced, press Start. Speaking produces the target voice in the headphones.
4. In VoiceMeeter, the virtual input strip shows level. In another app (Discord or Windows Sound Recorder) with VoiceMeeter Output as the microphone, the converted voice is recorded.
5. The pitch slider audibly changes pitch while live.
6. Mute silences both outputs. Ctrl+Alt+V toggles it with the window unfocused. Tray "Off" stops the session.
7. Each of the three presets starts and is intelligible. Note the Latency readout and the Dropouts count for each after 30 seconds of speech.
8. Fail closed: while live, kill the backend from PowerShell (`Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*aivoice*' | ForEach-Object { Stop-Process -Id $_.ProcessId }`). The recording app receives silence, not the raw voice. Within a few seconds the app shows Start again, not running.
9. Unplug the microphone while live. The app shows "An audio device was disconnected" and returns to stopped.

- [ ] **Step 3: Tune presets from the measurements**

For any preset where Dropouts kept rising or the "Try a faster preset" hint appeared in item 7, raise its `chunk_ms` in `backend/aivoice/presets.py` in steps of 40 until 30 seconds of speech gives no new dropouts. If `high_quality` cannot keep up with `harvest`, change its `f0_method` to `"dio"`. Keep `low_latency.chunk_ms < balanced.chunk_ms < high_quality.chunk_ms`. Update the mock `PRESETS` in `app/src/LivePanel.test.tsx` only if a test pins a changed number. Rerun Step 1.

- [ ] **Step 4: Write `README.md`**

```markdown
# AI Voice

A personal Windows app for real-time voice conversion. Stage 1 delivers live voice change:
your microphone goes in, a chosen voice comes out on a virtual mic, with an optional headphone monitor.

Design: `docs/superpowers/specs/2026-10-13-ai-voice-design.md`

## Requirements

- Windows 11 with a DirectX 12 GPU
- [uv](https://docs.astral.sh/uv/), Node 20+, Rust
- VoiceMeeter (provides the virtual microphone)

## Setup

    uv venv --python 3.12 backend/.venv
    uv pip install --python backend/.venv -e "backend[dev]"
    mkdir -p data/pretrained
    curl -L -o data/pretrained/vec-768-layer-12.onnx https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx
    cd app && npm install

If the machine has more than one GPU, create `data/settings.json` with `{"dml_device_id": 1}` (or 0)
to pick the fast one.

## Adding a voice

Voices are RVC v2 models exported to ONNX. Only use voices you have the right to use.

One-time exporter setup:

    git clone --depth 1 --branch 2.2.231006 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI tools/.rvc/repo
    uv venv --python 3.10 tools/.rvc/venv
    uv pip install --python tools/.rvc/venv torch==2.6.0 numpy==1.26.4 onnx --extra-index-url https://download.pytorch.org/whl/cpu

Export a voice:

    tools/.rvc/venv/Scripts/python tools/export_rvc_onnx.py path/to/voice.pth my-voice --index path/to/voice.index --name "My Voice"

## Running

    cd app && npm run tauri dev

Pick your microphone, leave Virtual mic on VoiceMeeter Input, choose a voice and press Start.
In other apps, select **VoiceMeeter Output** as the microphone. Ctrl+Alt+V mutes and unmutes.

## Tests

    backend/.venv/Scripts/python -m pytest backend/tests -v
    cd app && npm test
    cd app/src-tauri && cargo test
```

- [ ] **Step 5: Commit**

```bash
git add README.md backend app
git commit -m "docs: README and tuned presets for stage 1"
```
