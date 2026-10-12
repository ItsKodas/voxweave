# AI Voice

A personal Windows app for real-time voice conversion.

**Stages 1 and 2 are built.** The app does live voice change: your microphone goes in, a chosen voice comes out on a virtual microphone, with an optional headphone monitor. It also builds voice datasets: in the **Voices** tab you can record guided sentences or drop in existing audio, which is cut into clips and transcribed on this computer. Training your own voices and text-to-speech are later stages and are not in the app yet; a dataset is not used by live mode until training exists.

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
```

Install the backend packages with two commands, in this order:

```bash
uv pip install --python backend/.venv -e "backend[dev,transcribe]"
```

```bash
uv pip install --python backend/.venv --no-deps faster-whisper
```

`faster-whisper` is installed with `--no-deps` because it depends on the plain `onnxruntime` package, and the `transcribe` extra already lists its other dependencies. **Never install the plain `onnxruntime` package into this environment:** it shares a module name with `onnxruntime-directml` and would overwrite it, removing GPU support from live mode (`backend/tests/test_rvc_real.py` would then fail). Details: `docs/superpowers/notes/2026-10-13-whisper-setup.md`.

Then the base model and the app packages:

```bash
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

The same file also holds `whisper_model` (see Building a voice dataset). The value of `dml_device_id` must be a whole number. If it is not, the backend still starts, the "Setup needed" box says so, and Start is refused until the file is fixed or deleted.

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

## Building a voice dataset

Open the **Voices** tab. A dataset is a set of short clips of one voice, each with its text. It is what training (a later stage) will learn from. About 10 minutes of clean speech is usable and 30 minutes is good.

**Create a voice.** Type a name (1 to 60 characters, on one line) under **New voice name** and press **Create voice**. The folder it is stored in always gets a safe id, so two voices may share a name. Select a voice in the list to work on it. **Delete** next to a voice asks first and then moves its folder to `data/trash/` (nothing is erased by the app; to bring a voice back, move its folder from `data/trash/` into `data/voices/`). Deleting is refused while a recording is in progress, and while an import into that voice is running.

**The minutes bar** at the top of a voice shows how much audio it holds, with marks at 10 minutes ("Enough for a usable voice.") and 30 minutes ("Plenty for a good voice.").

### Guided recording

Under **Record**, choose the **Microphone** (the Windows default is preselected). The app shows one of 200 English sentences at a time.

- **Record** starts a take; read the sentence aloud. A level meter and the length of the take are shown.
- **Stop** ends the take and checks it. If it is good the app says "Saved", plays it back so you can hear it, and offers **Next sentence** and **Redo**.
- **Cancel** throws the take away without saving anything.
- **Skip** moves to the next sentence you have not recorded yet, without recording this one.
- **Redo** records the same sentence again. The new take replaces the earlier one, and only once the new take has been saved.
- **Next sentence** moves on to the next sentence that has no clip yet. When every sentence has one, the tab says so; import more audio or redo clips from the list.

Silence at the start and end of a take is trimmed off before it is checked. A take is rejected, with nothing written to disk, for one of four reasons:

1. **Too quiet:** nothing was heard, or the loudest moment was very low. Check the microphone and speak closer to it.
2. **Too short:** less than 1 second of speech. Read the whole sentence before pressing Stop.
3. **Too long:** more than 20 seconds of speech. Keep each take under 20 seconds. (The recorder itself stops collecting audio at 30 seconds and the screen shows "The take is at its maximum length"; press Stop to finish.)
4. **Distorted:** the signal hit its limit repeatedly (clipping). Move back from the microphone or lower its level.

Switching to another tab during a take cancels it, so the microphone is never left running.

### Importing existing audio

Drop audio files anywhere on the app window while the Voices tab is open and a voice is selected. Accepted types: `.wav .flac .mp3 .ogg .opus .m4a .aac .wma`. You can drop up to 200 files at once, and a single file can be up to 60 minutes long.

Each file is cut at pauses into clips of at most 15 seconds (continuous speech with no pauses is cut at its quietest points; pieces under 1 second are dropped), then transcribed in English on this computer. Pieces that are too quiet are skipped and counted rather than transcribed. A progress line shows the file being worked on, and **Cancel import** stops the job. When it finishes you see how many clips were added, and every file that failed is listed by name with the reason (not an audio file, not found, a folder, no speech found, and so on); the other files still import. Only one import runs at a time, and if the voice is deleted during an import the job stops with a message.

**The first import downloads the speech model, about 500 MB, once.** There is no separate progress bar for it: the first file simply takes longer. It is stored in `data/pretrained/whisper/`.

**Check and correct the transcripts.** Automatic transcripts contain mistakes, and a wrong transcript teaches a wrong voice. In the **Clips** list each clip has its transcript in a box: edit the text and click away to save it (at most 500 characters). **Play** plays the clip (the app fetches the audio and plays it from memory) and **Delete** removes the clip. Clips are listed newest first, tagged "recorded" or with the name of the file they came from. Clip numbers are never reused, even after a deletion.

### Where the data lives

```
data/voices.json                       summary of all voices (rewritten on every change)
data/voices/<id>/voice.json            the voice's name, creation time and clip list; the source of truth
data/voices/<id>/clips/0001.orig.wav   the clip at 48 kHz, mono, 16-bit
data/voices/<id>/clips/0001.wav        the same clip at 16 kHz, for training
data/voices/<id>/clips/0001.txt        the transcript (a copy; voice.json wins)
data/trash/<id>-<timestamp>/           deleted voices
data/pretrained/whisper/               the downloaded speech model
```

A voice folder that is damaged (for example a corrupt `voice.json`) is left alone and simply not listed; the other voices are unaffected.

### Settings

`whisper_model` in `data/settings.json` chooses the speech model for imports. The default is `small.en` (English only). A name without `.en`, such as `small`, is multilingual and detects the language of each clip; the recording sentences stay English. Use a plain model name (letters, digits, `.`, `_`, `-`). A different model is downloaded on its first use. A value that is not a model name makes imports fail with a message.

```json
{"whisper_model": "small.en"}
```

### Behaviour notes

- Recording and live mode cannot run at the same time. Starting a recording while live mode runs is refused ("Stop live mode before recording") and so is the reverse ("Stop recording before starting live mode").
- **Refresh devices** on the Live tab is refused while recording ("Stop recording before refreshing devices").
- The backend answers requests under `/library` (which serves your recordings) only for the app itself; requests from other websites get a 403.

### Privacy

Recordings and imported audio stay on this computer. The only network use is the one-time download of the speech model from Hugging Face.

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

## Stage 2 first run checklist

A human needs to do this once; it has not been done yet. Start with `cd app && npm run tauri dev`, and open the **Voices** tab.

1. Create a voice. It appears in the list with 0.0 min, 0 clips.
2. Record a sentence and hear it played back. The minutes bar moves.
3. Trigger each rejection: press **Record** and stay silent (too quiet); speak very quietly (too quiet); shout into the microphone (distorted); press **Stop** at once (too quiet or too short). Each gives a clear message and no clip appears.
4. Try **Skip** and **Redo**: Skip shows a different sentence; Redo replaces the earlier take of the same sentence instead of adding a second clip.
5. Record several sentences and watch the minutes bar grow.
6. Drop a real speech file (a few minutes of talking) on the window. The first time the model downloads; then clips appear, at most 15 seconds each, with sensible English transcripts.
7. Drop a file that is not audio (a renamed text file). It is reported as failed with its name, and nothing breaks.
8. Edit a transcript, click away, and confirm the matching `data/voices/<id>/clips/NNNN.txt` file changed.
9. Play a clip from the list; delete a clip and confirm its three files are gone and its number is not reused by the next clip.
10. Delete a voice and find its folder in `data/trash/`.
11. Start live mode and try to record: it is refused. Stop live mode, start a recording and try to start live mode: refused too.

## Tests

```bash
backend/.venv/Scripts/python -m pytest backend/tests -v
```

```bash
cd app && npm test && npx tsc --noEmit && npm run build
```

```bash
cd app/src-tauri && cargo test
```

`backend/tests/test_rvc_real.py` runs the real DirectML path and needs the GPU, the base model and a voice in `data/pretrained/rvc/`; it also shows that the speech packages did not remove DirectML.

## Known limitations

- The accurate pitch detector (harvest) is slower than real time at every preset's window size on the development machine, not only the larger ones, so all three presets use the fast one (dio). Harvest is still selectable under Advanced but will not keep up live with any preset.
- The app only runs from this checkout with `npm run tauri dev`; there is no installer, because the backend path is fixed at compile time.
- **Refresh devices**, the default-device preselection and listen-only mode are covered by automated tests with fake devices but have not been tried on real hardware yet.
- Preset values are untuned starting points; they have not been adjusted by ear.
- The tray menu, the Ctrl+Alt+V hotkey and the checklist above have not been verified by a human yet.
- Transcripts are English only by default (`small.en`); see Settings to change the model.
- Transcription runs on the CPU (the speech engine has no AMD GPU support), so long imports take a while.
- The first import waits for the model download with no separate progress bar.
- Dataset files are not used by live mode until training exists.
- Recording, import, drag and drop, clip playback and transcript quality have been tested with fake devices and files only; see the Stage 2 checklist.
- Windows only; DirectML GPU required for live mode (no CPU fallback).
