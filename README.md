# AI Voice

A personal Windows app for real-time voice conversion.

**Stages 1, 2 and 3 are built.** The app does live voice change: your microphone goes in, a chosen voice comes out on a virtual microphone, with an optional headphone monitor. It builds voice datasets: in the **Voices** tab you can record guided sentences or drop in existing audio, which is cut into clips and transcribed on this computer. It also trains your own voice models (RVC) from such a dataset on the AMD GPU, natively on Windows, and a trained voice then appears in the Voice list on the Live tab. Text to speech is a later stage and is not in the app yet. Only synthetic test audio has been trained so far, so how good a trained voice sounds is untested (see Known limitations).

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

Open the **Voices** tab. A dataset is a set of short clips of one voice, each with its text. It is what training (see Training a voice) learns from. Training needs at least 5 minutes of clean speech; about 10 minutes is usable and 30 minutes is good.

**Create a voice.** Type a name (1 to 60 characters, on one line) under **New voice name** and press **Create voice**. The folder it is stored in always gets a safe id, so two voices may share a name. Select a voice in the list to work on it. **Delete** next to a voice asks first and then moves its folder to `data/trash/` (nothing is erased by the app; to bring a voice back, move its folder from `data/trash/` into `data/voices/`). Deleting is refused while a recording is in progress, while an import into that voice is running, and while that voice is training ("Stop training before deleting this voice"). Deleting a voice also removes the temporary training files kept for it under `trainer/rvc/` (see Training a voice); its trained model lives in the voice's own folder and goes to `data/trash/` with it.

**The minutes bar** at the top of a voice shows how much audio it holds, with marks at 10 minutes ("Enough for a usable voice.") and 30 minutes ("Plenty for a good voice.").

### Guided recording

Under **Record**, choose the **Microphone** (the Windows default is preselected). The list follows **Refresh devices** on the Live tab: after plugging in a microphone, press it there and the microphone appears here too, with your chosen microphone still selected if it is still connected. The list never changes during a take. The app shows one of 200 English sentences at a time.

- **Record** starts a take; read the sentence aloud. A level meter and the length of the take are shown.
- **Stop** ends the take and checks it. If it is good the app says "Saved", plays it back so you can hear it, and offers **Next sentence** and **Redo**.
- **Cancel** throws the take away without saving anything.
- **Skip** moves to the next sentence you have not recorded yet, without recording this one.
- **Redo** records the same sentence again. The new take replaces the earlier one, and only once the new take has been saved. The earlier take is not erased: its files are moved to `data/trash/<voice-id>/clips/`.
- **Next sentence** moves on to the next sentence that has no clip yet. When every sentence has one, the tab says so; import more audio or redo clips from the list.

Silence at the start and end of a take is trimmed off before it is checked. A take is rejected, with nothing written to disk, for one of four reasons:

1. **Too quiet:** nothing was heard, or the loudest moment was very low. Check the microphone and speak closer to it.
2. **Too short:** less than 1 second of speech. Read the whole sentence before pressing Stop.
3. **Too long:** more than 20 seconds of speech. Keep each take under 20 seconds. Once a take passes 15 seconds the screen warns "Takes over 20 seconds are rejected. Finish the sentence and press Stop." (The recorder itself stops collecting audio at 30 seconds and the screen shows "The take is at its maximum length"; press Stop to finish.)
4. **Distorted:** the signal hit its limit repeatedly (clipping). Move back from the microphone or lower its level.

Switching to another tab during a take cancels it, so the microphone is never left running.

If a take cannot be written to disk (the disk is full, or the folder is read-only or in use), the app says so with the reason, for example "Could not save the clip: No space left on device". Nothing is added to the voice, an earlier take of the same sentence is kept, and that take is gone: free some space and record the sentence again. The same kind of message appears if creating a voice, saving a transcript or deleting a clip fails.

### Importing existing audio

Drop audio files anywhere on the app window while the Voices tab is open and a voice is selected. Accepted types: `.wav .flac .mp3 .ogg .opus .m4a .aac .wma`. You can drop up to 200 files at once, and a single file can be up to 60 minutes long.

Each file is cut at pauses into clips of at most 15 seconds (continuous speech with no pauses is cut at its quietest points; pieces under 1 second are dropped), then transcribed in English on this computer. Pieces that are too quiet are skipped and counted rather than transcribed. A progress line shows the file being worked on, and **Cancel import** stops the job. When it finishes you see how many clips were added and how many pieces were skipped as too quiet. If any were skipped, the line adds "Quiet recordings can be made louder in an audio editor and imported again." (the app never changes the loudness of your audio itself). Every file that failed is listed by name with the reason (not an audio file, not found, a folder, no speech found, and so on); the other files still import. Only one import runs at a time, and if the voice is deleted during an import the job stops with a message.

**The first import downloads the speech model, about 500 MB, once.** There is no separate progress bar for it: the first file simply takes longer. It is stored in `data/pretrained/whisper/`. Once it is there, the model is loaded from disk only: later imports, and later starts of the app, make no network request.

**Check and correct the transcripts.** Automatic transcripts contain mistakes, and a wrong transcript teaches a wrong voice. In the **Clips** list each clip has its transcript in a box: edit the text and click away to save it (at most 500 characters). An emptied box is not saved: the old text comes back with "A transcript cannot be empty. Delete the clip if it is not usable." **Play** plays the clip (the app fetches the audio and plays it from memory). **Delete** asks first and then takes the clip out of the voice and moves its three files to `data/trash/<voice-id>/clips/`; nothing is erased by the app. There is no undo button, but the audio is still there: to get a clip back, drop its `.orig.wav` from the trash onto the window to import it again. Clips are listed newest first, tagged "recorded" or with the name of the file they came from. Clip numbers are never reused, even after a deletion.

### Where the data lives

```
data/voices.json                       summary of all voices (rewritten on every change)
data/voices/<id>/voice.json            the voice's name, creation time and clip list; the source of truth
data/voices/<id>/clips/0001.orig.wav   the clip at 48 kHz, mono, 16-bit
data/voices/<id>/clips/0001.wav        the same clip at 16 kHz, for training
data/voices/<id>/clips/0001.txt        the transcript (a copy; voice.json wins)
data/trash/<id>-<timestamp>/           deleted voices
data/trash/<id>/clips/                 deleted clips and replaced takes of a voice (same three files per clip)
data/pretrained/whisper/               the downloaded speech model
```

A voice folder that is damaged (for example a corrupt `voice.json`) is left alone and not listed; the other voices are unaffected. `data/backend.log` names the skipped folder and the reason (never the file's contents).

The app never empties `data/trash/`. If a file in the trash already has the name a deleted clip would get, the clip is stored under a name with a timestamp added, so nothing in the trash is overwritten. Delete things from the trash yourself when you are sure you do not need them.

### Settings

`whisper_model` in `data/settings.json` chooses the speech model for imports. The default is `small.en` (English only). A name without `.en`, such as `small`, is multilingual and detects the language of each clip; the recording sentences stay English. Use a plain model name (letters, digits, `.`, `_`, `-`). A different model is downloaded on its first use. A value that is not a model name makes imports fail with a message.

```json
{"whisper_model": "small.en"}
```

### Behaviour notes

- Recording and live mode cannot run at the same time. Starting a recording while live mode runs is refused ("Stop live mode before recording") and so is the reverse ("Stop recording before starting live mode"). Training is refused while either runs, and either is refused while training runs; see Training a voice.
- **Refresh devices** on the Live tab is refused while recording ("Stop recording before refreshing devices").
- The backend answers requests under `/library` (which serves your recordings) only for the app itself; requests from other websites get a 403. Each refused request is written to `data/backend.log` as one line starting with "Refused", with its path and the Origin, Referer and Sec-Fetch-Site headers it carried. If every action on the Voices tab fails with "Origin not allowed", that line shows what the app's window actually sent.

### Privacy

Recordings and imported audio stay on this computer. The only network use is the one-time download of the speech model from Hugging Face, on the first import (and once more if you change `whisper_model` to a model you do not have yet), and the downloads of `bash trainer/setup.sh` (see Training setup). After that the model is loaded from disk without contacting Hugging Face, and the download library's usage reporting is switched off. Training itself makes no network requests.

## Training setup

Training runs **natively on Windows** (not in WSL) on the AMD GPU, in its own Python environment under `trainer/`. The backend environment never gets PyTorch; training code only ever runs in `trainer/.venv`. This is a one-time setup, and it is not needed unless you want to train voices.

You need, besides the Requirements above:

- An AMD graphics driver, **AMD Adrenalin 26.2.2 or newer**. The training environment uses AMD's official PyTorch for Windows wheels (ROCm 7.2.1), which require it. It was developed and tested on an RX 7900 XTX with driver 32.0.31041.1004 (Adrenalin 26.8.1). A future driver can break the pairing: the environment check then says the GPU is not visible.
- `ffmpeg` and `ffprobe` on `PATH` (the RVC preprocessing step runs both). Open a new Git Bash window after installing them and check with `ffmpeg -version` and `ffprobe -version`.
- `git`, `curl` and `sha256sum` in Git Bash (they come with Git for Windows) and `uv`.
- The ONNX exporter environment in `tools/.rvc` from **Adding a voice** above. The last training step exports the model with it, and the training check fails without it.

Then, from the repo root in Git Bash:

```bash
bash trainer/setup.sh
```

It creates `trainer/.venv` (Python 3.12), installs AMD's seven ROCm 7.2.1 wheels (hash-pinned) and the packages in `trainer/requirements.txt`, clones RVC into `trainer/rvc` at commit `81eed5e8f68b6bed1789f682fe78cdd324495afc`, applies the two patches in `trainer/patches/`, and downloads the pretrained files listed in `trainer/pretrained.json` (about 590 MB). Every pretrained file is checked against its recorded SHA-256 and size before it is put in place, and an unfinished download (`*.part`) is removed and never used. Nothing is installed system-wide, and the caches stay inside `trainer/.cache`.

**Size.** The AMD wheels are about 2.1 GB to download. On disk the environment is about 6.4 GB (`trainer/.venv`) and `trainer/rvc` about 0.6 GB. On a clean machine uv's package cache under `trainer/.cache/uv` is a further cost of similar size (the spike measured 6.5 GB for it). These sizes are from the development machine, which reused an existing cache: setup took 100 seconds there and 7 seconds when re-run. On a clean machine the downloads dominate.

The script is safe to re-run: each step is skipped when its result is already correct, so it also repairs a damaged environment. It never touches your voices or recordings.

To check the environment without changing anything, run the same check the app runs before every training start. It prints one JSON object and exits 0 only if everything passed. Add `--fast` to skip the GPU check, which takes a few seconds:

```bash
trainer/.venv/Scripts/python trainer/verify.py
```

The checks are `venv`, `torch_gpu` (the ROCm build of PyTorch is installed and sees the GPU), `rvc_commit`, `rvc_patches`, `pretrained` (files present with the right sizes), `ffmpeg` (both programs on `PATH`) and `exporter`. When one fails, its message names the missing item and the fix. In the app, the **Train** box of a voice then shows the failing checks under "Training is not set up on this computer" with a **Check again** button, and Start is refused with the same list. Live mode and recording keep working either way.

## Training a voice

Training turns a voice's dataset into an RVC v2 model (40 kHz, with pitch) that the Live tab can use. Open the **Voices** tab, select a voice and use its **Train** box.

**Before you start.** The voice needs at least **5.0 minutes** of clean audio (the minutes bar shows it; with less, the box says "Record or import at least 5 minutes to train", and the backend refuses with "This voice has N minutes of clean audio. Training needs at least 5."). More audio gives a better voice; 10 to 30 minutes is the useful range. Training reads only the 48 kHz `NNNN.orig.wav` files, through hard links (or copies) in a staging folder, and never changes or deletes your recordings. Live mode must be off and no recording may be running, or Start is refused ("Stop live mode before training", "Stop recording before training"). Imports may run during training.

**Choices.**

- **Epochs** (100, 200 or 300; default 200). One epoch is one pass over all of the voice's audio. More epochs train longer and usually fit the voice more closely, up to a point. The backend accepts any whole number from 20 to 1000, but the screen offers these three. A checkpoint is saved every 5 epochs.
- **Batch size** (4, 8 or 16; default 8), under **Advanced**: how many slices are processed at once. In the spike, 8 was nearly as fast as 16 at about a third of the GPU memory, and 4 was slower per epoch. Leave it at 8 unless you have a reason. It cannot be changed when resuming.
- **Rough time** is shown on the screen: a fixed rule of thumb of about one second per epoch per minute of audio, rounded up to the next 5 minutes. Measured on the development machine at batch 8: 10 minutes of audio takes about 9.4 seconds per epoch, so about 31 minutes for 200 epochs; 30 minutes of audio is about 28 seconds per epoch, about 1 h 35 min for 200 epochs (extrapolated, not measured). The other stages add about 2 to 3 minutes for 10 minutes of audio (spike: 9 s slicing, 38 s pitch, 92 s features, 4 s index, 8 s export). The first training on a machine can take a few minutes longer while the GPU libraries compile their kernels.

Press **Start training** (**Train again** for a voice that already has a model; it asks first, and the current model is replaced only when the new run finishes).

**The stages**, as the screen names them, with one progress bar over all of them:

1. **Preparing**: checks the clips and stages them.
2. **Slicing audio**: RVC cuts the clips into short slices.
3. **Finding pitch**: the pitch of each slice, on the GPU.
4. **Extracting features**: the voice features of each slice, on the GPU.
5. **Training**: shown with "Epoch N of M" and a time estimate. Almost all of the time is spent here.
6. **Building index**: the search index used to blend in the trained voice at conversion time.
7. **Exporting**: converts the model for live mode and puts it in place.

While a run is active the box shows the stage, the bar and **Cancel**, and says "Live mode and recording are unavailable until training finishes." Starting live mode is refused with "Training is running. Wait for it to finish or cancel it before starting live mode", and recording with the same text ending "before recording". The box of any other voice says "Another voice is training". Only one training runs at a time.

### Cancel, Resume and Start over

- **Cancel** asks "Stop training? You can resume it later." The trainer stops at the end of the epoch it is in (usually a few seconds; it is ended by force after at most 90 seconds) and the box says "Training was stopped". Nothing is thrown away: the saved training state stays, and so does any model you already had.
- **Resume** (shown only when the stopped run can be resumed) continues from the last saved checkpoint, which is at most 5 epochs back, with the same epochs and batch size. It is refused if the clips changed since the run started (clips added, deleted or replaced), because old features must not be mixed with a new clip list: "The clips changed since this training run started. Start a new training run." It is not offered either when the saved state cannot be read. If a save was cut off in the middle (power loss, process killed), Resume uses the verified backup copy of the checkpoint instead of silently starting from scratch. A run stopped before its first checkpoint was saved resumes by training from epoch 1.
- **Start over** (shown in place of Start training after a run was stopped or failed) asks "Start over? The progress so far will be discarded." and then removes that voice's saved training state and begins again. A model already in the voice's folder is kept until the new one is finished.
- A failed run shows its reason on screen; the details are in `data/voices/<id>/training/worker.log`.

**Closing the app.** Hiding the window to the tray does not stop training. **Quit** in the tray menu (or Ctrl+C in the console running `npm run tauri dev`) does: the training worker is tied to the app and ends with it, so nothing keeps the GPU busy afterwards, and the app waits up to 10 seconds for it to stop. The next time the app starts, the voice shows "Training was interrupted when the app closed" and **Resume** continues from the last checkpoint (up to 5 epochs are repeated). The same happens after a backend crash; a worker left running by a crashed backend is found and ended.

**Where things are.**

```
data/voices/<id>/models/rvc/model.onnx      the trained model, used by live mode
data/voices/<id>/models/rvc/model.index     its search index (used by the Live tab's index blend)
data/voices/<id>/models/rvc/voice.json      its name and sample rate
data/voices/<id>/training/job.json          the record of the latest run: state, epochs, batch size, times
data/voices/<id>/training/progress.jsonl    the progress lines the screen follows
data/voices/<id>/training/worker.log        what the training worker did, with the output of each stage
data/voices/<id>/training/worker.out.log    the worker's own output and errors, for a crash
trainer/rvc/logs/voice-<id>/                temporary: checkpoints and working files of a run
trainer/rvc/assets/weights/voice-<id>.pth   temporary: the small model, until it is exported
```

The model is exported into a new folder and swapped in only after it passes a check, so a failed run never damages the model you already have.

**Disk use during a run.** Beyond the one-time environment (see Training setup), a run keeps a checkpoint pair of about 1.3 GB (`G_2333333.pth` 438 MB and `D_2333333.pth` 857 MB) under `trainer/rvc/logs/voice-<id>/`, plus a verified second copy of that pair (`.bak`) for crash safety, so keep about 2.6 GB free on that drive, plus the slices and features (hundreds of MB for 10 minutes of audio). All of it is deleted automatically when the run succeeds, and when the voice is deleted. It stays after Cancel, a failure or an interruption, because Resume needs it. The exported model is about 55 MB and its index about 90 MB for 10 minutes of audio.

**Network.** Training makes no network requests: the worker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Only `bash trainer/setup.sh` downloads anything.

**Using the trained voice.** When a run finishes, the box says "Training finished. This voice is now available on the Live tab." and shows "Trained model: ready". On the **Live** tab the voice is listed under **Voice** with the name you gave it; choose it like any exported voice (see Running). The Live voice list reloads when a run finishes while the Voices tab is open. If a run finished while you were on another tab and the voice is missing, restart the app.

**Retraining.** Add or fix clips and press **Train again**. The new model replaces the old one only when the run completes; cancelling or failing leaves the old model in use.

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

1. Open the **Voices** tab and confirm the voice list loads; if every action fails with a 403, see `data/backend.log` for the refused headers.
2. Create a voice. It appears in the list with 0.0 min, 0 clips.
3. Record a sentence and hear it played back. The minutes bar moves.
4. Trigger each rejection: press **Record** and stay silent (too quiet); speak very quietly (too quiet); shout into the microphone (distorted); press **Stop** at once (too quiet or too short). Each gives a clear message and no clip appears. Let one take run past 15 seconds and see the warning about takes over 20 seconds.
5. Try **Skip** and **Redo**: Skip shows a different sentence; Redo replaces the earlier take of the same sentence instead of adding a second clip, and the earlier take's three files are in `data/trash/<voice-id>/clips/`.
6. Record several sentences and watch the minutes bar grow.
7. Drop a real speech file (a few minutes of talking) on the window. The first time the model downloads; then clips appear, at most 15 seconds each, with sensible English transcripts. Restart the app, disconnect from the network and import again: it still works.
8. Drop a file that is not audio (a renamed text file). It is reported as failed with its name, and nothing breaks.
9. Edit a transcript, click away, and confirm the matching `data/voices/<id>/clips/NNNN.txt` file changed.
10. Play a clip from the list. Delete a clip: the app asks first; answer no and it stays. Delete it again and answer yes: find its three files in `data/trash/<voice-id>/clips/`, and confirm its number is not reused by the next clip.
11. Delete a voice and find its folder in `data/trash/`.
12. Start live mode and try to record: it is refused. Stop live mode, start a recording and try to start live mode: refused too.
13. Plug in (or unplug) a microphone, press **Refresh devices** on the Live tab, and confirm the **Microphone** list on the Voices tab changed to match.

## Stage 3 first run checklist

A human needs to do this once; it has not been done yet. Everything in stage 3 was tested with synthetic audio only. Start with the setup, then `cd app && npm run tauri dev` and open the **Voices** tab.

1. Run `bash trainer/setup.sh`. It ends with "done in N s", and `trainer/.venv/Scripts/python trainer/verify.py` prints `"ok": true`.
2. Select a voice with less than 5 minutes: the Train box says to record or import at least 5 minutes and offers no Start.
3. Select a voice with at least 5 minutes of real speech (record it, or import a real recording). If the box lists a failing check, fix it and press **Check again**.
4. Train a short run first: choose **100** epochs and press **Start training**. Watch the stages go through their names, the bar move, and "Epoch N of 100" with a time estimate. Live mode and recording are refused meanwhile. The GPU is busy (Task Manager, Performance, GPU).
5. Press **Cancel** during Training and answer yes. It stops within seconds and says "Training was stopped". Press **Resume**: it continues from a saved epoch, not from 1 (unless you cancelled before the first save at epoch 5).
6. Choose **Quit** in the tray menu mid-training, confirm no `python.exe` from `trainer/.venv` is left running, then reopen the app: the voice shows "Training was interrupted when the app closed" and **Resume** works.
7. Let that short run finish. The box says "Training finished" and the voice appears under **Voice** on the Live tab. Start live mode with it, listen on headphones and note how it sounds. Check that `trainer/rvc/logs/voice-<id>/` is gone and `data/voices/<id>/models/rvc/` holds `model.onnx` and `model.index`.
8. Cancel another run, then add a clip to the voice and press **Resume**: it is refused with the "clips changed" message, and **Start over** works.
9. Then the full run: **Train again** with 200 (or 300) epochs and let it finish, listen on the Live tab again and compare it with the short run. Note the total time against the estimate.
10. Delete a trained test voice and confirm `trainer/rvc/logs/voice-<id>/` and its `trainer/rvc/assets/weights/` file are gone.

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

`backend/tests/test_worker_real.py` runs the real training pipeline on the GPU and is skipped unless asked for. It needs the training environment, takes about 5 to 6 minutes, and must not run while anything else uses the GPU (live mode, another training run):

```bash
AIVOICE_RUN_TRAIN_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_worker_real.py -q -W error -s
```

It runs a 2-epoch training and converts audio with the result, cancels and resumes a run, and kills a worker outright and resumes. Last result: 3 passed in 5 min 32 s. Without the variable the full backend suite reports those 3 as skipped (last result: 757 passed, 3 skipped).

## Known limitations

- The accurate pitch detector (harvest) is slower than real time at every preset's window size on the development machine, not only the larger ones, so all three presets use the fast one (dio). Harvest is still selectable under Advanced but will not keep up live with any preset.
- The app only runs from this checkout with `npm run tauri dev`; there is no installer, because the backend path is fixed at compile time.
- **Refresh devices**, the default-device preselection and listen-only mode are covered by automated tests with fake devices but have not been tried on real hardware yet.
- Preset values are untuned starting points; they have not been adjusted by ear.
- The tray menu, the Ctrl+Alt+V hotkey and the checklist above have not been verified by a human yet.
- Transcripts are English only by default (`small.en`); see Settings to change the model.
- Transcription runs on the CPU (the speech engine has no AMD GPU support), so long imports take a while.
- The first import waits for the model download with no separate progress bar.
- **Only synthetic audio has been trained so far** (about 60 seconds of speech-like test audio, 2 to 8 epochs, in the spike and in the automated runs). How a model trained on a real voice sounds is unproven, and so is the stability of a run lasting an hour or more. The Stage 3 checklist is the first real test.
- **Text to speech is not built yet.** Stage 3 trains RVC voice conversion models only. GPT-SoVITS (text to speech) training is planned for stage 4 and the app does not train it.
- Training depends on two patches carried against a pinned upstream commit of RVC (`81eed5e`): one rewrites a convolution that is about 60 times slower than normal on this AMD ROCm build (training is 3.4 times faster with it), the other stops preprocessing from dropping audio before a pause. `bash trainer/setup.sh` applies them and refuses to continue if the clone has been edited.
- Training needs an AMD GPU with the driver named in Training setup and has only been run on an RX 7900 XTX. There is no CPU fallback.
- Training and live mode cannot run at the same time (both need the GPU), and neither can training and recording. Quitting the app interrupts training; it can be resumed.
- The training time estimate is a fixed rule of thumb, and the first epoch of a first-ever run is much slower while the GPU libraries compile.
- The Live tab's voice list reloads when a run finishes while the Voices tab is open, not when you come back to the Live tab later; restart the app if a trained voice is missing.
- Recording, import, drag and drop, clip playback and transcript quality have been tested with fake devices and files only; see the Stage 2 checklist.
- Windows only; DirectML GPU required for live mode (no CPU fallback).
