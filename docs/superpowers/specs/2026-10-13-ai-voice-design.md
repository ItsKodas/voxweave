# AI Voice: Design

Date: 2026-10-13
Status: Awaiting review

## Purpose

A personal Windows desktop app that lets the user:

1. Build a voice dataset, either by reading prompted sentences into the mic or by importing existing recordings, and train voice models from it.
2. Speak typed text in any trained voice (text to speech).
3. Convert their live microphone into a chosen voice in real time, keeping their own pitch contour, timing and delivery while changing the timbre.
4. Send the result to a virtual microphone for other apps, with an optional monitor to headphones.
5. Trade speed against quality in both TTS and live mode.

The app wraps existing open-source engines. It does not implement new models.

## Target machine and constraints

- Windows 11 Pro, AMD Radeon RX 7900 XTX (24 GB VRAM), 31 GB RAM.
- No NVIDIA GPU, so no CUDA. Windows inference uses DirectML through ONNX Runtime. Training uses ROCm inside WSL2 (Ubuntu, already installed).
- WSL2 cannot access Windows audio devices with usable latency, so all audio I/O and live inference run natively on Windows.
- VoiceMeeter is already installed and provides the virtual microphone. The app does not ship or install an audio driver.
- Already present: Python 3.10, Node 24, Rust/Cargo.
- Single user, single machine. No accounts, no network service, no cloud.

## Architecture

Three processes.

### 1. Tauri shell

Rust plus a web frontend (React and TypeScript).

- Owns the window, tray icon and a global hotkey.
- Spawns the Python backend on launch, monitors it, and restarts it if it exits.
- Talks to the backend over HTTP and one WebSocket on a localhost port chosen at launch.
- Contains no audio or model logic.

### 2. Python backend (Windows)

A FastAPI service. It is the only component that touches audio devices, models and the data folder.

Modules, each with one job:

| Module | Responsibility | Depends on |
|---|---|---|
| `devices` | Enumerate input/output devices, open streams (sounddevice, WASAPI) | none |
| `router` | Fan one audio stream out to the virtual mic and/or the monitor output, with per-output gain | `devices` |
| `live` | Realtime pipeline: capture, chunk, convert, crossfade, emit | `router`, `engines.rvc` |
| `presets` | Map a named speed/quality preset to concrete engine parameters | none |
| `dataset` | Clip storage, silence trimming, level and clipping checks, import and splitting | none |
| `transcribe` | Whisper transcripts for imported clips | `dataset` |
| `prompts` | Serve sentences for guided recording and track which are done | `dataset` |
| `voices` | Voice library and manifest: which voices exist and which models each has | none |
| `training` | Launch and monitor the WSL2 worker, parse progress, handle resume | `voices`, `dataset` |
| `engines.rvc` | RVC inference on DirectML | none |
| `engines.tts` | TTS inference: F5-TTS zero-shot and GPT-SoVITS fine-tuned | none |
| `api` | HTTP and WebSocket routes; thin layer over the modules above | all |

Engines sit behind small interfaces (`convert(chunk, voice, params)`, `synthesize(text, voice, params)`) so an engine can be replaced without touching `live`, `api` or the UI.

### 3. Training worker (WSL2)

A Python script run through `wsl.exe` by the `training` module.

- Reads a voice's dataset folder and writes model files back to the same voice folder through the shared Windows path.
- Runs RVC training and GPT-SoVITS fine-tuning on ROCm.
- Writes structured progress lines (JSON, one per line) to a log file. The backend tails the file and forwards progress over the WebSocket.
- Saves checkpoints so a failed or cancelled run can resume.

### Data layout

```
data/
  voices.json                 manifest
  voices/<voice-id>/
    clips/NNNN.wav            16 kHz mono (a 48 kHz original is kept alongside)
    clips/NNNN.txt            transcript
    models/rvc/               .onnx model and index
    models/gptsovits/         fine-tuned weights
    reference.wav             short reference clip for zero-shot TTS
    training/                 logs and checkpoints
  pretrained/                 downloaded base and third-party models
```

`voices.json` records, per voice: id, display name, total clean minutes, and which models exist with their training date.

## Features

### Recording

Guided session:

- Shows one sentence at a time from a bundled, phonetically balanced script of about 200 sentences.
- User records, the backend trims leading and trailing silence, then checks peak level and clipping.
- A clip that clips or is too quiet is rejected with the reason and the same sentence is offered again.
- Accepted clips are saved with their text. The user can replay, redo or skip.
- A progress bar shows total clean minutes, with markers at 10 minutes (usable) and 30 minutes (good).

Import:

- User drops in audio files.
- The backend resamples, converts to mono, splits on silence into clips of at most 15 seconds, and discards clips under 1 second.
- Whisper runs locally to produce a transcript for each clip. The user can correct transcripts in the UI.

Out of scope for version one: LLM-generated sentences, noise removal, multi-speaker separation.

### Training

- One action per engine per voice: Train RVC, Train GPT-SoVITS.
- Refused with a clear message if the voice has under 5 minutes of clean audio.
- UI shows stage, epoch, percentage and estimated time remaining, plus Cancel and Resume.
- On success the RVC model is exported to ONNX so Windows inference can load it, and the manifest is updated.
- One training job runs at a time. Live mode is stopped while training runs, since both need the GPU.

### Text to speech

- Inputs: voice, text, preset, output target (headphones, virtual mic, both), optional save to WAV.
- Presets:
  - Fast: F5-TTS zero-shot from `reference.wav`, low step count. Works for any voice with a reference clip, no training needed.
  - Balanced: F5-TTS with full step count.
  - Best: fine-tuned GPT-SoVITS. Only offered when that model exists for the voice.
- Long text is split into sentences and synthesised and played in sequence, so playback starts before the whole text is finished.

### Live voice change

- Controls: input device, virtual mic output device, monitor on/off with its own output device and volume, target voice, pitch shift in semitones, preset.
- Presets set chunk size, crossfade length, pitch extraction method and index blend:
  - Low latency: small chunks, fast pitch extractor, low index blend.
  - Balanced.
  - High quality: large chunks, accurate pitch extractor, higher index blend.
- An advanced panel exposes the underlying parameters for manual tuning.
- Live readout: measured end-to-end latency, per-chunk inference time, input and output level meters.
- Global hotkey and tray menu toggle between converted, muted and off.

## Failure handling

The governing rule: the virtual mic never carries the user's unconverted voice unless they explicitly chose passthrough.

| Failure | Behaviour |
|---|---|
| Backend exits | Shell restarts it. Live mode comes back stopped, not running. Virtual mic is silent meanwhile. |
| Conversion throws or times out on a chunk | That chunk is emitted as silence. Repeated failures stop live mode with an error. |
| Inference slower than real time | Underruns are counted. Past a threshold the UI recommends a faster preset. Output is silence for missed chunks, never raw mic. |
| Device unplugged | Live mode stops, UI prompts for a new device. |
| Virtual mic device not found | Setup check explains how to select VoiceMeeter's input. Monitor-only use still works. |
| Training fails | Log and checkpoints are kept, dataset is untouched, Resume is offered. |
| WSL2 or ROCm unavailable | Setup check names what is missing. Training is disabled. Live mode and zero-shot TTS still work. |
| Model file missing or corrupt | Voice is shown as needing retraining for that engine. Other voices are unaffected. |

A setup check runs on first launch and from settings. It verifies: DirectML provider loads, a virtual mic device is present, WSL2 distro is reachable, ROCm sees the GPU, and base models are downloaded.

## Testing

- Unit tests (pytest) with small WAV fixtures: silence trimming, clipping and level detection, import splitting, chunking and crossfade continuity, preset mapping, manifest reads and writes, training progress parsing.
- Offline pipeline test: a file is pushed through `live` with a stub engine that returns its input. Asserts output length matches, no chunk is dropped or duplicated, and crossfade seams have no discontinuity.
- Fail-closed test: a stub engine that throws must produce silence on the output, never the input signal.
- API tests against the FastAPI app with engines stubbed.
- Frontend: component tests for the recording flow and the live control panel against a mocked backend.
- Manual listening check at the end of each stage. Model quality and perceived latency are judged by ear.

## Build stages

Each stage gets its own implementation plan and ends in something usable.

1. Live voice change. Tauri shell, backend skeleton, `devices`, `router`, `live`, `presets`, `engines.rvc`, setup check. Uses a downloaded pretrained RVC voice. Proves DirectML inference on the 7900 XTX is fast enough, which is the largest technical risk.
2. Recording and import. `dataset`, `prompts`, `transcribe`, `voices`, and the recording UI.
3. Training. The WSL2 worker, `training`, ONNX export, progress UI. Proves ROCm training works, the second largest risk.
4. Text to speech. `engines.tts` with F5-TTS first, then the GPT-SoVITS fine-tuned path.

## Risks

- DirectML may be too slow for the low-latency preset, or an RVC component may not export cleanly to ONNX. Stage 1 starts with a throwaway benchmark of RVC inference on this GPU before any app code. If it fails, the fallback is ROCm-for-Windows PyTorch, then CPU for the pitch extractor only.
- ROCm in WSL2 on this card may need specific driver and kernel versions. Stage 3 starts with a standalone training run outside the app.
- GPT-SoVITS and F5-TTS inference on DirectML is less proven than RVC. If neither runs acceptably on Windows, TTS inference moves into the WSL2 worker and returns audio to the backend. TTS is not latency critical, so this is acceptable.

## Responsible use

The app is for the user's own voice and voices they have permission to use. The recording flow only trains from clips the user records or imports themselves.
