# AI Voice

A personal Windows app for real-time voice conversion.

**Stage 1 (this version) is live voice change only:** your microphone goes in, a chosen voice comes out on a virtual microphone, with an optional headphone monitor. Recording, training your own voices and text-to-speech are later stages and are not in the app yet.

Design: `docs/superpowers/specs/2026-10-13-ai-voice-design.md`

## Requirements

- Windows 11 with a DirectX 12 GPU
- [uv](https://docs.astral.sh/uv/), Node 20+, Rust
- [Git for Windows](https://git-scm.com/) (the commands below are written for its Git Bash shell)
- [VoiceMeeter](https://vb-audio.com/Voicemeeter/) (provides the virtual microphone; without it the app still works in listen-only mode, see Running)

## Setup

Run from the repo root, in **Git Bash**. All commands in this README are Git Bash commands unless a block says PowerShell; `mkdir -p`, `curl -L -o` and the forward-slash paths do not work as written in PowerShell or cmd.

```bash
uv venv --python 3.12 backend/.venv
uv pip install --python backend/.venv -e "backend[dev]"
mkdir -p data/pretrained
curl -L -o data/pretrained/vec-768-layer-12.onnx https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx
cd app && npm install
```

`vec-768-layer-12.onnx` is the ContentVec base speech model every voice needs.

### Choosing the GPU

By default the backend uses DirectML device 0. If the machine has two GPUs (for example a discrete card plus an integrated one) and device 0 is the slow one, create `data/settings.json`:

```json
{"dml_device_id": 1}
```

The value must be a whole number. If it is not, the backend still starts, the "Setup needed" box says so, and Start is refused until the file is fixed or deleted.

To tell which is which, try both values: the wrong (integrated) GPU is many times slower, which shows up as high Latency and rising Dropouts in the app. On the development machine (RX 7900 XTX plus integrated Radeon) device 0 is the fast one and the integrated GPU was 5 to 20 times slower; see `docs/superpowers/notes/2026-10-13-directml-benchmark.md` for the benchmark numbers.

## Adding a voice

The app needs at least one voice. Voices are RVC v2 models (with pitch) exported to ONNX by `tools/export_rvc_onnx.py`.

**A fresh clone has no voices at all**, so the Voice list is empty and Start stays disabled until you export one. If a voice called `bench` is present, it was created by the benchmark: it is a randomly initialised model that only produces noise, not a usable voice.

**Security:** `.pth` files are Python pickles and can run code when loaded. The exporter only accepts plain weight files (it loads with `weights_only=True` and refuses anything else), but treat any `.pth` from the internet with suspicion. Only add voices you have the right to use.

One-time exporter setup (from the repo root). The patch is required: the stock exporter at this tag fails and bakes a fixed 200-frame length into the graph, and the exporter refuses to run on an unpatched clone.

```bash
git -c core.autocrlf=false clone --depth 1 --branch 2.2.231006 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI tools/.rvc/repo
git -C tools/.rvc/repo apply --ignore-whitespace ../../rvc_onnx_export.patch
uv venv --python 3.10 tools/.rvc/venv
uv pip install --python tools/.rvc/venv torch==2.6.0 numpy==1.26.4 onnx scipy --extra-index-url https://download.pytorch.org/whl/cpu
```

Export a voice (the optional `.index` file enables the index blend):

```bash
tools/.rvc/venv/Scripts/python tools/export_rvc_onnx.py path/to/voice.pth my-voice --index path/to/voice.index --name "My Voice"
```

The result is written to `data/pretrained/rvc/my-voice/`. Restart the app to see it in the Voice list.

## Running

```bash
cd app && npm run tauri dev
```

This is the only way to run the app for now. The location of the backend is fixed to this checkout when the app is compiled, so a built installer would not work on another machine or after the folder is moved.

`npm run tauri dev` keeps running after you close the window, because closing only hides the app to the tray. To stop it, choose **Quit** in the tray menu or press Ctrl+C in the console.

The app starts the Python backend itself. If something required is missing, a "Setup needed" box lists what and how to fix it. If the backend does not come up within 15 seconds the window says so and lists the usual causes.

1. Choose your **Microphone**. The Windows default recording device is preselected.
2. Leave **Virtual mic** on a VoiceMeeter Input device.
3. Optionally tick **Listen to output** and pick your headphones under **Listen on**. The Windows default playback device is preselected.
4. Choose a **Voice** and a **Speed / quality** preset, then press **Start**.
5. In other apps (Discord, recorders), select **VoiceMeeter Output** as the microphone.

**Plugged something in?** The device lists are read when the backend starts. Press **Refresh devices** to reload them after connecting a headset or starting VoiceMeeter, or after "An audio device was disconnected". It is only available while live mode is stopped.

**Listen-only mode (no VoiceMeeter).** When no virtual mic is found, **Virtual mic** is set to **None (listen only)**; the app never picks your speakers for it. Tick **Listen to output**, choose headphones and press Start to hear the converted voice yourself. Other apps hear nothing until VoiceMeeter is installed and running (then press Refresh devices). Use headphones, not speakers, or the microphone will pick the output up again. You can also choose None yourself to try a voice without sending it anywhere.

**Backend log.** Everything the backend prints (startup errors, Python tracebacks, request log) is appended to `data/backend.log`. Look there first when the app reports that the voice engine failed to start or stopped unexpectedly. The file is started afresh once it grows past about 5 MB.

Presets (all use the fast pitch detector at the moment):

| Preset | Chunk | Feel |
|---|---|---|
| Low latency | 120 ms | fastest response, roughest quality |
| Balanced | 250 ms | default compromise |
| High quality | 500 ms | smoothest, most delay |

The Advanced section lets you override chunk, crossfade, context, pitch detector and index blend. The pitch slider (-24 to +24 semitones) and the listen volume can be changed while live.

Controls:

- **Ctrl+Alt+V** mutes and unmutes from anywhere, even with the window unfocused.
- **Tray menu:** Converted, Muted, Off (stops the session), Show window, Quit.
- **Closing the window hides it to the tray**; the session keeps running. Use **Quit** in the tray menu to exit (or Ctrl+C in the console that runs `npm run tauri dev`).

## Safety behaviour

If anything fails (backend crash, device unplugged, a conversion that cannot keep up), the virtual mic receives silence, never your real voice. Mute is strict: both the virtual mic and the monitor go silent. If the backend dies while live, the app says "The voice engine stopped unexpectedly. Live mode is off." and the reason is in `data/backend.log`.

The backend only listens on 127.0.0.1, only answers requests addressed to `127.0.0.1` or `localhost`, and refuses changes requested by web pages other than the app itself.

## First run checklist

A human needs to do this once; it has not been done yet. Start with `cd app && npm run tauri dev`.

1. The window opens and shows no "Setup needed" box.
2. **Virtual mic** is preselected as a VoiceMeeter Input device, **Microphone** as the Windows default recording device, and (after ticking Listen to output) **Listen on** as the Windows default playback device, not a VoiceMeeter device.
3. Tick **Listen to output**, pick your headphones under **Listen on**, choose Balanced and press **Start**. Speaking produces the target voice in the headphones.
4. In VoiceMeeter the virtual input strip shows level. In another app (Discord or Windows Sound Recorder) with VoiceMeeter Output as the microphone, the converted voice is recorded.
5. Moving the **Pitch shift** slider audibly changes pitch while live.
6. **Mute** silences both outputs. Ctrl+Alt+V toggles it with the window unfocused. Tray **Off** stops the session. Closing the window hides it to the tray and Quit exits.
7. Each of the three presets starts and is intelligible. Note the Latency readout and the Dropouts count for each after 30 seconds of speech. If Dropouts keep rising or a "Try a faster preset" hint appears, raise that preset's `chunk_ms` in `backend/aivoice/presets.py` in steps of 40 (keep low_latency < balanced < high_quality).
8. Fail closed: while live, kill the backend from PowerShell. The recording app receives silence, not the raw voice, and within a few seconds the app shows **Start** again with the message "The voice engine stopped unexpectedly. Live mode is off."

   ```powershell
   Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*aivoice*' | ForEach-Object { Stop-Process -Id $_.ProcessId }
   ```

9. Unplug the microphone while live. The app shows "An audio device was disconnected" and returns to stopped. Plug it back in and press **Refresh devices**: it is listed again and Start works.
10. Start and stop a session, then open `data/backend.log`: it contains the backend's startup lines and request log.

## Tests

```bash
backend/.venv/Scripts/python -m pytest backend/tests -v
cd app && npm test && npx tsc --noEmit
cd app/src-tauri && cargo test
```

`backend/tests/test_rvc_real.py` runs the real DirectML path and needs the GPU, the base model and a voice in `data/pretrained/rvc/`.

## Known limitations

- The accurate pitch detector (harvest) is slower than real time at every preset's window size on the development machine, not only the larger ones, so all three presets use the fast one (dio). Harvest is still selectable under Advanced but will not keep up live with any preset.
- The app only runs from this checkout with `npm run tauri dev`; there is no installer, because the backend path is fixed at compile time.
- **Refresh devices**, the default-device preselection and listen-only mode are covered by automated tests with fake devices but have not been tried on real hardware yet.
- Preset values are untuned starting points; they have not been adjusted by ear.
- The tray menu, the Ctrl+Alt+V hotkey and the checklist above have not been verified by a human yet.
- Windows only; DirectML GPU required (no CPU fallback).
