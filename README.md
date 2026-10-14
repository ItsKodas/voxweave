# Voxweave

A personal Windows app for real-time voice conversion.

**Stages 1 to 4 are built.** The app does live voice change: your microphone goes in, a chosen voice comes out on a virtual microphone, with an optional headphone monitor. It builds voice datasets: in the **Voices** tab you can record guided sentences or drop in existing audio, which is cut into clips and transcribed on this computer. It also trains your own voice models (RVC) from such a dataset on the AMD GPU, natively on Windows, and a trained voice then appears in the Voice list on the Live tab. The **Speak** tab turns typed text into speech in one of your voices (text to speech), from a single short clip of that voice. Only synthetic test audio has been trained so far, so how good a trained voice sounds is untested, and nobody has listened to any synthesised speech yet (see Known limitations).

Design: `docs/superpowers/specs/2026-10-13-ai-voice-design.md`

## Install

For using Voxweave without building it. Nothing has to be installed first: no Git, Node, Rust, Python or command line.

**You need:** Windows 10 or 11 (64-bit), a graphics card that supports DirectX 12 (any make), about 2 GB of free disk space, and an internet connection the first time the app starts.

1. Download `Voxweave_<version>_x64-setup.exe` from the [latest release](../../releases/latest).
2. Run it. It installs for your Windows account only and does not ask for administrator rights. **The installer is not signed**, so Windows SmartScreen may show "Windows protected your PC": choose **More info**, then **Run anyway**. Your browser may warn about the download as well. Each release lists the SHA-256 of its installer if you want to check the file. On a PC without Microsoft's WebView2 runtime (some Windows 10 installations) the installer downloads it first.
3. Start Voxweave. The first start shows **First-time setup** with a progress bar and the current step. It downloads Python (22 MB), the voice engine's packages (124 MB) and the base speech model (378 MB), about 525 MB in total, which take about 0.9 GB on disk. This needs an internet connection once. If it stops, the screen shows why and a **Retry** button; Retry continues where it stopped. When it has finished, the normal app appears, and later starts go straight to it.
4. Use the app as described under Running, from "Choose your **Microphone**" on. For other apps to hear the converted voice you need a virtual microphone ([VoiceMeeter](https://vb-audio.com/Voicemeeter/) or VB-CABLE); without one you can still listen to the converted voice yourself.

**What works in the installed app:** live mode, the Voices tab (guided recording and importing audio; the first import downloads a speech recognition model of a few hundred MB, see Building a voice dataset), **My voice** calibration, and voice packs.

**What does not work in the installed app yet: training and text to speech.** Both need the source checkout (Building from source, below) on a PC with an AMD GPU, and the app says so on the Train panel and on the Speak tab instead of offering them. So the installed app cannot make a voice model by itself. A voice is trained on such a PC, exported there as a voice pack, and brought here by dropping its `.voxweave` file on the Voices tab (see "Moving a trained voice to another computer" under Speaking text). Until there is a voice, Start on the Live tab stays disabled.

**Where things are.** Everything the installed app writes is in `%LOCALAPPDATA%\Voxweave` (paste that into the address bar of Explorer):

| Folder | Contents |
|---|---|
| `data\` | your voices, recordings and settings, and `backend.log`; the same layout as `data/` in the rest of this README |
| `runtime\` | the Python environment, the download cache and `setup.log`; it can be deleted, the next start builds it again |

Unless you choose another folder in the installer, the program itself is in the same folder (`Voxweave.exe`, `uninstall.exe`, `backend\`, `setup\`, `bin\`). The app window's own browser data is kept by Windows in `%LOCALAPPDATA%\gg.horizons.aivoice`.

**Updating.** Run the installer of a newer release over the old one. The first start after an update runs the setup again; it only downloads what changed.

**If the app says the voice engine failed to start,** press **Repair the installation** in that message: it runs the setup again, which replaces what is missing or damaged and keeps your voices. **If the setup fails with a message about a missing DLL,** install Microsoft's [Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) and press Retry; a setup whose packages do not load removes the environment it built, so Retry builds it again from the start. **If the "Setup needed" box says DirectML is not available,** follow its steps (quit, delete `%LOCALAPPDATA%\Voxweave\runtime`, start again). If live mode still does not start after that, the graphics card or its driver does not support DirectX 12, which no repair changes. **If the app says it cannot start because a file is missing,** install it again.

**Uninstalling.** Use "Installed apps" in Windows Settings, or `uninstall.exe` in the program folder. This removes the program and its shortcuts. It leaves `%LOCALAPPDATA%\Voxweave\data` (your voices and recordings) and `%LOCALAPPDATA%\Voxweave\runtime` (about 0.9 GB that can be downloaded again) where they are; nothing of yours is deleted automatically. To remove everything, delete the folder `%LOCALAPPDATA%\Voxweave` yourself after uninstalling. The uninstaller's "Delete the application data" box only removes the window's browser data in `%LOCALAPPDATA%\gg.horizons.aivoice`, not your voices.

**What has been tested, and what has not.** On the development PC: the installer was built (12 MB) and its contents were unpacked and compared with the sources, without running it. The first-run setup was run for real into an empty folder (41 seconds on a fast connection; 3 seconds when run again with everything in place), and the voice engine was started from the environment it built and answered. The setup, the shell's installed mode, the setup screen and the installed app's messages are covered by automated tests. **Not tested at all: running the installer.** Nobody has installed it, seen the installed app start, watched the First-time setup screen in the real window, updated, or uninstalled, on any PC. It has not been tried on a PC without developer tools, on Windows 10, or on a graphics card other than the development PC's AMD card. The release workflow on GitHub has never run. Treat the first release as a test.

## Building from source

Everything from here on is about running Voxweave from its source code. For now that is also the only way to train voices and to use text to speech.

### Requirements

- Windows 11 with a DirectX 12 GPU
- [uv](https://docs.astral.sh/uv/), Node 20+, Rust
- [Git for Windows](https://git-scm.com/) (the commands below are written for its Git Bash shell)
- [VoiceMeeter](https://vb-audio.com/Voicemeeter/) (provides the virtual microphone; without it the app still works in listen-only mode, see Running)

### Setup

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

#### Choosing the GPU

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

This runs the app from the source checkout: the location of the backend is fixed to this checkout when the app is compiled. To use the app on another PC, build the installer (see Building the installer and releasing) or download one (see Install).

`npm run tauri dev` keeps running after you close the window, because closing only hides the app to the tray. To stop it, choose **Quit** in the tray menu or press Ctrl+C in the console.

The app starts the Python backend itself. If something required is missing, a "Setup needed" box lists what and how to fix it. If the backend does not come up within 15 seconds the window says so and lists the usual causes.

1. Choose your **Microphone**. The Windows default recording device is preselected.
2. Leave **Virtual mic** on a VoiceMeeter Input device.
3. Optionally tick **Listen to output** and pick your headphones under **Listen on**. The Windows default playback device is preselected.
4. Choose a **Voice** and a **Speed / quality** preset, then press **Start**.
5. In other apps (Discord, recorders), select **VoiceMeeter Output** as the microphone.

**Plugged something in?** The device lists are read when the backend starts. Press **Refresh devices** to reload them after connecting a headset or starting VoiceMeeter, or after "An audio device was disconnected". It is only available while live mode is stopped.

**Listen-only mode (no VoiceMeeter).** When no virtual mic is found, **Virtual mic** is set to **None (listen only)**; the app never picks your speakers for it. Tick **Listen to output**, choose headphones and press Start to hear the converted voice yourself. Other apps hear nothing until VoiceMeeter is installed and running (then press Refresh devices). Use headphones, not speakers, or the microphone will pick the output up again. You can also choose None yourself to try a voice without sending it anywhere.

**Backend log.** Everything the backend prints (startup errors, Python tracebacks, request log) is appended to `data/backend.log` (in the installed app: `%LOCALAPPDATA%\Voxweave\data\backend.log`). Look there first when the app reports that the voice engine failed to start or stopped unexpectedly. The file is started afresh once it grows past about 5 MB.

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

Under **Record**, choose the **Microphone** (the Windows default is preselected). The list follows **Refresh devices** on the Live tab: after plugging in a microphone, press it there and the microphone appears here too, with your chosen microphone still selected if it is still connected. The list never changes during a take. The app shows one of 200 English sentences at a time. Above the text you can switch between **Sentences** and **Paragraphs**: there are also 40 paragraphs of two or three sentences (10 to 15 seconds each), which collect speech faster with fewer presses of Record. Each list keeps its own count, and Skip and Next stay within the list you chose. Paragraph takes are longer than the 5 to 10 seconds text to speech wants for its reference clip, so keep at least a few single sentences recorded as well.

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

- Recording and live mode cannot run at the same time. Starting a recording while live mode runs is refused ("Stop live mode before recording") and so is the reverse ("Stop recording before starting live mode"). Training is refused while either runs, and either is refused while training runs; see Training a voice. Text to speech takes part in the same rule; see Speaking text.
- **Refresh devices** on the Live tab is refused while recording ("Stop recording before refreshing devices").
- The backend answers requests under `/library` (which serves your recordings) only for the app itself; requests from other websites get a 403. Each refused request is written to `data/backend.log` as one line starting with "Refused", with its path and the Origin, Referer and Sec-Fetch-Site headers it carried. If every action on the Voices tab fails with "Origin not allowed", that line shows what the app's window actually sent.

### Privacy

Recordings and imported audio stay on this computer. The installed app downloads Python, its packages and the base speech model once, at its first start (see Install), from GitHub, the Python Package Index and Hugging Face. Apart from that, the only network use is the one-time download of the speech model from Hugging Face, on the first import (and once more if you change `whisper_model` to a model you do not have yet), and the downloads of `bash trainer/setup.sh` (see Training setup) and `bash tts/setup.sh` (see Text to speech setup). After that the model is loaded from disk without contacting Hugging Face, and the download library's usage reporting is switched off. Training and text to speech themselves make no network requests.

## Training setup

Training runs **natively on Windows** (not in WSL) on the AMD GPU, in its own Python environment under `trainer/`. The backend environment never gets PyTorch; training code only ever runs in `trainer/.venv`. This is a one-time setup, and it is not needed unless you want to train voices.

You need, besides the Requirements above:

- An AMD graphics driver, **AMD Adrenalin 26.2.2 or newer**. The training environment uses AMD's official PyTorch for Windows wheels (ROCm 7.2.1), which require it. It was developed and tested on an RX 7900 XTX with driver 32.0.31041.1004 (Adrenalin 26.8.1). A future driver can break the pairing: the environment check then says the GPU is not visible.
- `ffmpeg` and `ffprobe` on `PATH` (the RVC preprocessing step runs both). Open a new Git Bash window after installing them and check with `ffmpeg -version` and `ffprobe -version`.
- `git`, `curl` and `sha256sum` in Git Bash (they come with Git for Windows) and `uv`.
- **Python 3.12** where `uv` can find it. The setup script does not download Python (it runs `uv venv --python 3.12 --no-python-downloads`); if you have no 3.12, get one first with `uv python install 3.12`.
- The ONNX exporter environment in `tools/.rvc` from **Adding a voice** above. The last training step exports the model with it, and the training check fails without it.

Then, from the repo root in Git Bash:

```bash
bash trainer/setup.sh
```

It creates `trainer/.venv` (Python 3.12), installs AMD's seven ROCm 7.2.1 wheels (hash-pinned) and the exact package versions in `trainer/requirements.lock` (the versions training was proven with; `trainer/requirements.txt` holds the ranges they were resolved from and is used, with a printed notice, only when the lock file is missing), clones RVC into `trainer/rvc` at commit `81eed5e8f68b6bed1789f682fe78cdd324495afc`, applies the two patches in `trainer/patches/`, and downloads the pretrained files listed in `trainer/pretrained.json` (about 590 MB). Every pretrained file is checked against its recorded SHA-256 and size before it is put in place, and an unfinished download (`*.part`) is removed and never used. The silence samples are unpacked to a temporary folder and put in place in one step, so an interrupted setup never leaves a half-unpacked set. Nothing is installed system-wide, and the caches stay inside `trainer/.cache`.

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
- **Rough time** is shown on the screen: a fixed rule of thumb of about one second per epoch per minute of audio, rounded up to the next 5 minutes. Measured on the development machine at batch 8: 10 minutes of audio takes about 9.4 seconds per epoch, so about 31 minutes for 200 epochs; 30 minutes of audio is about 28 seconds per epoch, about 1 h 35 min for 200 epochs (extrapolated, not measured). The other stages add about 2 to 3 minutes for 10 minutes of audio (spike: 9 s slicing, 38 s pitch, 92 s features, 4 s index, 8 s export). The first training on a machine can take a few minutes longer while the GPU libraries compile their kernels: the first epoch of a first-ever run can take minutes, and the box shows "Estimating time…" until two epochs are done.

Press **Start training** (**Train again** for a voice that already has a model; it asks first, and the current model is replaced only when the new run finishes).

**The stages**, as the screen names them, with one progress bar over all of them:

1. **Preparing**: checks the clips and stages them.
2. **Slicing audio**: RVC cuts the clips into short slices.
3. **Finding pitch**: the pitch of each slice, on the GPU.
4. **Extracting features**: the voice features of each slice, on the GPU.
5. **Training**: shown with "Epoch N of M" and a time estimate. Almost all of the time is spent here.
6. **Building index**: the search index used to blend in the trained voice at conversion time.
7. **Exporting**: converts the model for live mode and puts it in place.

**Windows display language.** The epoch count, the time remaining and the checkpoint backups are read from the English log lines RVC prints. RVC translates some of those lines for a few Windows display languages without an English fallback (Chinese, Japanese, Korean and a few others). On such a system a run still completes and its model is exported, but without epoch counts, without a time estimate and without checkpoint backups.

While a run is active the box shows the stage, the bar and **Cancel**, and says "Live mode and recording are unavailable until training finishes." Starting live mode is refused with "Training is running. Wait for it to finish or cancel it before starting live mode", and recording with the same text ending "before recording". The box of any other voice says "Another voice is training". Only one training runs at a time.

### Cancel, Resume and Start over

- **Cancel** asks "Stop training? You can resume it later." The trainer stops at the end of the epoch it is in, so that it is never cut off in the middle of a save (usually a few seconds). The wait for the end of the epoch is one minute, or twice the length of the recent epochs when they are longer than that, and never more than 15 minutes; after it the trainer is ended anyway, and the app ends it by force 30 seconds later still if it has not gone. The box then says "Training was stopped". Nothing is thrown away: the saved training state stays, and so does any model you already had.
- **Resume** is offered for a run that was cancelled, interrupted or failed, from the moment its clips are sliced. It keeps the same epochs and batch size, the sliced audio, and whatever pitch and feature files were already made. If a checkpoint was saved, training continues from the last one, which is at most 5 epochs back. If the run stopped before its first checkpoint (during pitch, features or the first epochs), the box says "Resume keeps the prepared audio and starts training from the first epoch." A run stopped while the clips were still being sliced has nothing prepared yet, so it offers only Start over.
- Resume is refused if the clips changed since the run started (clips added, deleted or replaced), because old features must not be mixed with a new clip list: "The clips changed since this training run started. Start a new training run. Or put the clips back as they were and resume." Nothing is deleted by that refusal and Resume stays offered: once the clips are as they were, the same run continues; until then every Resume ends with the same message. Resume is no longer offered, and the backend refuses it too, only after a failure that a resume can never get past (the saved state cannot be read, the prepared audio is incomplete); then only Start over remains. If a save was cut off in the middle (power loss, process killed), Resume uses the verified backup copy of the checkpoint instead of silently starting from scratch.
- **Start over** (shown in place of Start training after a run was stopped or failed) asks "Start over? The progress so far will be discarded." and then removes that voice's saved training state and begins again. A model already in the voice's folder is kept until the new one is finished.
- A failed run shows its reason on screen; the details are in `data/voices/<id>/training/worker.log`.

**Closing the app.** Hiding the window to the tray does not stop training. **Quit** in the tray menu does: it ends the backend and the whole training process tree at once, without waiting for the end of an epoch. Nothing keeps the GPU busy afterwards. The next time the app starts, the voice shows "Training was interrupted when the app closed" and **Resume** is offered (up to 5 epochs are repeated). Because the trainer is not given time to finish what it is doing, the interruption can land in the middle of a save; Resume then continues from the previous checkpoint's backup copy, so up to one more save interval is repeated, and nothing is lost silently. When the backend exits by itself in the normal way (for example Ctrl+C in the console running `npm run tauri dev`), it first asks the trainer to stop, waits up to 5 seconds, and then ends whatever is still running. The same recovery applies after a backend crash: the worker is tied to the backend and ends when it is gone, and a worker that is still found running at the next start is ended before anything else.

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
data/trash/<id>/models-<UTC time>/          a model that a later training run replaced
```

The model is exported into a new folder and swapped in only after it passes a check, so a failed run never damages the model you already have. If the app is killed in the middle of that swap, the previous model is put back when the app next starts.

**Disk use during a run.** Beyond the one-time environment (see Training setup), a run keeps a checkpoint pair of about 1.3 GB (`G_2333333.pth` 438 MB and `D_2333333.pth` 857 MB) under `trainer/rvc/logs/voice-<id>/`, plus a verified second copy of that pair (`.bak`) for crash safety, so keep about 2.6 GB free on that drive, plus the slices and features (hundreds of MB for 10 minutes of audio). All of it is deleted automatically when the run succeeds, and when the voice is deleted. It stays after Cancel, a failure or an interruption, because Resume needs it. The exported model is about 55 MB and its index about 90 MB for 10 minutes of audio.

**Network.** Training makes no network requests: the worker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Only `bash trainer/setup.sh` downloads anything.

**Using the trained voice.** When a run finishes while its box is on screen, the box says "Training finished. This voice is now available on the Live tab."; later, and for a run that finished while you were elsewhere, it says "Trained model: ready". On the **Live** tab the voice is listed under **Voice** with the name you gave it; choose it like any exported voice (see Running). The app follows a running job whichever tab is shown, so the Live voice list reloads when the run finishes even if you are waiting on the Live tab, and it reloads again every time you come back to the Live tab. Your selection and the rest of the form are kept.

**Retraining.** Add or fix clips and press **Train again**. The new model replaces the old one only when the run completes; cancelling or failing leaves the old model in use. The replaced model is not erased: it is moved to `data/trash/<id>/models-<UTC time>/` (about 150 MB for 10 minutes of audio). Nothing empties the trash; delete those folders yourself when you no longer want them.

## Text to speech setup

Text to speech runs **natively on Windows** on the AMD GPU, in its own Python environment under `tts/`, with the F5-TTS model. The backend environment never gets PyTorch; the model only ever runs in `tts/.venv`, in a separate worker process the backend starts and stops. This is a one-time setup, and it is not needed unless you want the **Speak** tab.

**Text to speech needs this environment, with AMD's ROCm build of PyTorch, and has no other way to run.** There is no DirectML version of it and no CPU fallback: without `tts/.venv` and a GPU the ROCm build can use, nothing can be spoken. The design said that text to speech from a short clip would still work without ROCm; as built, that no longer holds. Live mode does not depend on any of this.

**Licence of the model: non-commercial.** The F5-TTS weights this setup downloads (`F5TTS_v1_Base`) are licensed **CC-BY-NC-4.0**: personal, non-commercial use only, with attribution. Speech made with them may not be sold or used for paid work. The code (F5-TTS, Vocos) and the Vocos weights are MIT. If that licence does not suit what you want to do, do not use the Speak tab; nothing else in the app depends on these files.

You need, besides the Requirements above:

- The same AMD graphics driver as for training (**AMD Adrenalin 26.2.2 or newer**; see Training setup). It was developed and tested on an RX 7900 XTX only.
- `curl` and `sha256sum` in Git Bash (they come with Git for Windows) and `uv`.
- **Python 3.12** where `uv` can find it (`uv python install 3.12` if you have none; the script does not download Python).

Then, from the repo root in Git Bash:

```bash
bash tts/setup.sh
```

It creates `tts/.venv` (Python 3.12), installs AMD's seven ROCm 7.2.1 wheels (hash-pinned, the same ones as the training environment), the exact package versions in `tts/requirements.lock`, and `f5-tts` 1.1.22 without its declared dependencies (the lock file holds the part of them that speaking needs). It then downloads the four model files listed in `tts/pretrained.json` from Hugging Face, at pinned revisions: the F5-TTS model (1.35 GB) and its vocabulary, and the Vocos vocoder (54 MB) and its configuration, **about 1.4 GB in all**. A download that fails is tried again up to three times. Every file is checked against its recorded size and SHA-256 before it is put in place, and an unfinished download (`*.part`) is removed and never used. **That is the only time the SHA-256 is checked.** When the app runs, the model files are checked by size only (the `models` check below): they are loaded as safetensors and with PyTorch's weights-only loader, neither of which runs code from the file, and hashing 1.4 GB at every start was judged not worth the wait. To have the hashes checked again, run `bash tts/setup.sh` again. The script prints the licence notice above before it downloads. Nothing is installed system-wide, nothing outside `tts/` is touched, and the caches stay inside `tts/.cache`.

**Size.** The AMD wheels are about 2.1 GB to download; if the training environment's package cache (`trainer/.cache/uv`) exists, it is reused and they are not downloaded again. On disk the environment is about 6.5 GB (`tts/.venv`, largely shared with that cache through hard links when both are on the same drive) and the models 1.4 GB. On the development machine, where the AMD wheels were already cached, setup took 43 seconds and 8 seconds when re-run; on a clean machine the downloads dominate.

The script is safe to re-run: each step is skipped when its result is already correct, so it also repairs a damaged environment. It never touches your voices or recordings.

To check the environment without changing anything, run the check the app runs. It prints one JSON object and exits 0 only if everything passed. Add `--fast` to skip the GPU check, which takes a few seconds:

```bash
backend/.venv/Scripts/python tts/verify.py
```

The checks are `venv`, `torch_gpu` (the ROCm build of PyTorch is installed and sees the GPU), `f5_package` and `models` (files present with the right sizes; sizes only, not SHA-256, see above). When one fails, its message names the missing item and the fix. The quick form of the check, which the app runs before it starts the voice engine, is given 20 seconds, and the full one 5 minutes; a check that runs out of time is stopped, together with every process it started, and counts as failed. In the app, the Speak tab then shows the failing checks under "Text to speech is not set up on this computer" with a **Check again** button. Live mode, recording and training keep working either way.

## Speaking text

Open the **Speak** tab, choose a voice, type or paste text, and press **Speak** (or Ctrl+Enter in the text box). The text is spoken in that voice on the outputs you chose. **Nobody has listened to the output yet**: everything in this section was measured by program on the development machine (see Known limitations and the Stage 4 checklist).

**Choosing a voice.** The list holds the voices from the Voices tab. A voice can be spoken as soon as it has **one clip between 3 and 12 seconds long with a transcript**; no training is needed for Fast and Balanced. A voice without such a clip, or one whose only such clips are too quiet to copy, is listed but greyed out, with the reason (for example "Clip 3 is too quiet to copy. Record a louder clip of 5 to 10 seconds.").

**The reference clip.** The model copies a voice from one short clip of it and that clip's transcript: it hears how the clip sounds and continues in the same voice with your text. The app picks the clip by itself: a recorded clip before an imported one, then the one closest to 8 seconds. A clip that is too quiet to copy (an average level below about -50 dBFS) is passed over for the next one in that order. **The clip cannot be chosen by hand yet**: to make the app use another clip, record a better one or delete the one it picks. If the voice engine refuses the clip after all, the message names it by the clip number the Voices tab shows. Everything about the result follows from that clip, so it is worth getting a good one:

- One clear sentence of **5 to 10 seconds**, spoken the way you want the speech to sound. The speed of the clip sets the speed of the speech, and its mood carries over.
- A quiet room and no background sound. The model copies the room as well as the voice: hiss or echo in the clip comes back in every sentence.
- No long pauses inside it, and nothing cut off at the start or the end.
- **The transcript must be exactly what is said.** For an imported clip, check the automatic transcript on the Voices tab and correct it. A wrong transcript makes the model stumble or say words from the clip.
- The easiest way: on the Voices tab, record one of the guided sentences at a natural pace. A recorded clip is always preferred over imported ones.

**Speed / quality.** All three use the same model; they differ in how many refinement steps each sentence gets. The times are from the development machine (RX 7900 XTX) for sentences of about 90 characters, each about 8 seconds of speech, with the sample clip that ships inside the `f5_tts` package (`infer/examples/basic/basic_ref_en.wav`) as the reference:

| Preset | What it does | First sound after pressing Speak | Each further sentence |
|---|---|---|---|
| Fast | 10 steps | 0.5 s | 0.5 s |
| Balanced | 16 steps (the default) | 0.8 s | 0.8 s |
| Best | 32 steps, then the voice's own trained model reshapes the result | 4 to 5 s | 0.6 to 1.6 s |

- **How often a word goes wrong.** In the measurements, a speech recogniser found at least one wrong or missing word in 10 of 360 sentences spoken at Fast and in 3 of 180 at Balanced (some of those are the recogniser's own mistakes). Fast was first built with 8 steps, which got a word wrong in one sentence out of four; it was raised to 10 for that reason. Nothing in the app detects a sentence that was spoken wrongly, so listen to anything you keep.
- **Best** is offered only for a voice that has a trained model (see Training a voice); otherwise the option is greyed out with "Best needs a trained model. Train this voice on the Voices tab." It speaks the text with the model at full steps and then passes each sentence through the voice's trained RVC model, the same one live mode uses. Its first sound takes 4 to 5 seconds on every request, not only the first: the trained model is loaded again each time (about 2 seconds) while the engine makes the first sentence beside it (1.6 seconds alone at 32 steps, longer while the other model is loading), and the sentence then goes through the trained model (about 0.6 seconds). Whether Best sounds more like the voice than Balanced has not been judged by anyone; the only trained model that exists so far was made from synthetic test sound.
- **The first request after the app starts takes longer**: the voice engine has to start and load the model, 11 to 17 seconds on the development machine, and up to about 35 seconds the very first time on a machine. The screen says "Getting ready. Starting the voice engine. This takes about 15 seconds, and up to half a minute the first time." meanwhile. The engine then stays loaded and is stopped again after 10 minutes without a request, when training starts, and when the app quits. The backend watches the 10 minutes itself, looking about every 30 seconds, so the engine goes after 10 to 10.5 idle minutes whichever tab is open and whether or not the window is showing; an engine that dies while idle is noticed the same way, and the next request starts a fresh one.
- Speech is made faster than it plays at every preset (about 8 seconds of speech in half a second at Fast, in under a second at Balanced, in about 2 seconds at Best), so playback does not have to wait once it has started.

**Outputs.** **Listen on** chooses where you hear it (your headphones or speakers; "Off" for none). Preselected is the listen device chosen on the Live tab when that is a real output (not the virtual mic), otherwise the Windows default output. **Send to virtual mic** also plays it into the virtual microphone, so that other apps (Discord, a recorder) hear it as if spoken; the box appears only when a virtual mic exists. When there is more than one (VoiceMeeter Input and VoiceMeeter Aux Input, or a virtual cable), a **Virtual mic** list below the box chooses which one; it starts on the one chosen on the Live tab.

**Feminizing a voice.** Under **Feminize** on a voice's page, three sliders set how much higher the pitch is (semitones), how much higher the resonances of the vocal tract are (percent; this is what keeps a raised voice from sounding like a sped-up recording), and how breathy it is. Nothing here uses anyone else's voice: the voice is split by the WORLD vocoder into pitch, resonances and breath, changed, and put back together.

- **Hear a sample** plays one of the voice's clips with the current settings (made in a second or two).
- **Use for the live preview** saves the settings for the voice **My voice, feminized (preview, no training)**, which is always at the end of the Live tab's voice list. It needs no trained model and works at once, but sounds more processed. Its pitch comes from these settings; the Live tab's pitch slider adds to it.
- **Make a feminized voice** makes a new voice, "<name> (feminized)", from every clip of this one, feminized at the best quality, with the same transcripts (about a quarter of a second per second of audio). Train that voice like any other; live mode then converts your real voice into it, which smooths what the processing leaves rough. The original voice and its recordings are not changed.

What the result sounds like depends on the person, so expect a few rounds of sample and adjust. How you speak still counts: the pitch movement and rhythm of your live speech pass straight through, so a little more variation in pitch helps more than any setting. Nobody has judged the result by ear yet.

**Moving a trained voice to another computer (voice packs).** Under **Share** on a voice's page, **Export voice** writes the trained model as one file, `<name>.voxweave`, into your Downloads folder (an existing file is never replaced: a second export is `<name>-2.voxweave`). The file holds the voice model, its index, and, if the box is ticked, one short clip with its transcript so that text to speech works on the other computer too. Your other recordings are never included. On the other computer, drop the file anywhere on the app window while the Voices tab is open: a new voice appears, ready on the Live tab. A pack is typically 150 to 500 MB. Importing reads only the files it expects from the pack, up to fixed sizes, and refuses a model that does not load; even so, only import packs from people you trust, and only share voices you have the right to share. Exporting waits while a training run is in progress.

**When a training run goes wrong.** Training normally runs in half precision, which is fast. On some recordings its numbers overflow: every loss in the log becomes `nan`, and nothing learned after that is a voice. The trainer watches for this. At the first such line it stops, throws away what was saved, and starts training again from the beginning in full precision, which is slower but steady; the progress line says so. If it happens in full precision too, the run fails with advice to remove clips that are not clean speech, and it cannot be resumed. As a last line of defence, live mode and text to speech check what a voice model returns: a model that answers with noise at full volume (which is what a run that went wrong produces) is never played. Live mode refuses to start with it, or plays silence and stops, and says the model is damaged and the voice should be trained again.

**Calibrating live mode to your own voice.** The voice models only learn the target voice, so nothing is trained on you. Instead, **My voice** on the Live tab makes one recording of about 18 seconds with the microphone chosen there: stay quiet for the first 3 seconds, then read the text on screen aloud. It ends by itself. From it the app keeps (in `data/calibration/profile.json`) your usual pitch, your speaking level, and how loud the room is when you say nothing. Live mode then uses it three ways:

- **Pitch.** When you choose a voice, the pitch slider is set to the whole number of semitones that moves your usual pitch onto that voice's usual pitch, measured from up to a minute of the voice's own clips. A line under the slider says so. Move the slider and your setting is kept until you choose another voice or calibrate again. A voice with no clips in the library (an exported model) cannot be measured: set its pitch by hand.
- **Quiet gate.** While the microphone only carries room noise, live mode outputs silence without running the model, so breathing and hum are not turned into voice-like sound. It opens as soon as any part of the newest chunk is louder than the threshold, and stays open for about 0.3 seconds after you stop.
- **Level.** A quiet voice is raised towards an ordinary speaking level (by at most 12 dB, never into clipping), a very loud one lowered.

**Use my calibration** on the Live tab switches the gate and the level off for a start; **Forget calibration** removes the profile. Calibrate again after changing microphone or room. If the take cannot be used (too little speech, or the room and your speech too close in level) the screen says why and the earlier calibration is kept. Calibration cannot run during live mode or a recording, and those cannot start during a calibration take. None of this has been judged by ear yet: the measurements are tested with synthetic sound.

**Why the virtual microphone has two names.** A virtual microphone is a pipe with two ends. This app plays into the sending end, which Windows lists under playback devices and VB-Audio calls "Input" (VoiceMeeter Input, CABLE Input). Other apps choose the receiving end as their microphone: **VoiceMeeter Output**, **VoiceMeeter Aux Output** or **CABLE Output**. The lists on the Live and Speak tabs name the microphone first ("VoiceMeeter Output (sent through VoiceMeeter Input)") and a line underneath says which microphone to choose in other apps. With VoiceMeeter, the program must be running and the strip this app plays into must have its bus switched on (B1 for VoiceMeeter Output, B2 for Aux), or the microphone stays silent. A virtual cable needs neither.

**A dedicated microphone for the app.** The app cannot add a microphone to Windows by itself (only a driver signed through Microsoft can), and it does not download or run an installer for you. On the Live tab, **Set up a dedicated Voxweave microphone** gives the steps for VB-CABLE, a free virtual cable from VB-Audio (https://vb-audio.com/Cable/): install it yourself, press **I have installed it**, and "CABLE Input" becomes the virtual mic. Other apps then choose **CABLE Output** as their microphone; you can rename it in Windows Sound settings. When a cable is present the app prefers it over VoiceMeeter. At least one of the two must be chosen. On the listen device the speech is played at the **listen volume of the Live tab**, as the backend last learned it: from the last time live mode was started, or from the slider being moved while live mode ran. Until live mode has been started once since the app opened, that is full volume, wherever the slider stands. **A Live listen volume of 0 makes the speech silent on the listen device** (the virtual mic still gets it), and a volume above about 1.12 is held at 1.12 for speech, because more would distort it. The virtual mic always gets the speech at full level. Every sentence is brought towards the same peak level (about 1 dB below full scale) before it is played or saved, so that sentences do not differ in loudness. A sentence is raised by at most four times (12 dB) on the way: one that comes out far quieter than speech normally does, for example with a very quiet reference clip, stays quieter than the others instead of having its noise turned up to full level.

**Keep as a WAV file.** Tick it before pressing Speak and the whole speech is also written to `data/tts/<UTC time>-<voice id>.wav` (mono, 48 kHz), with the same short pauses between sentences as in playback. The screen shows "Saved to ..." with the path. The file is written when playback has finished; a request that was stopped or failed saves nothing. An earlier file is never overwritten. Nothing removes these files; delete them yourself.

**Stop.** **Stop** (or Escape) silences the outputs at once and cancels the rest; in the measurements the engine was idle again 0.04 seconds later. The engine stays loaded, so the next request starts quickly. Pressing Speak while something is being spoken is refused ("Already speaking. Stop it first.").

**Long text.** A request can hold up to 5000 characters, counted as typed. The text is cut into pieces at sentence ends (`.`, `!`, `?`, a blank line), each piece is spoken on its own, and **playback starts as soon as the first piece is ready** while the rest is still being made; the screen shows "Speaking sentence N of M". A sentence too long for the model in one go is cut at a comma, semicolon or colon, and failing that between words. How long a piece may be depends on the reference clip (132 characters with the package's sample clip; a clip with more words per second allows more). There is a pause of 0.3 seconds between pieces. Because each piece is spoken separately, the tone of voice does not carry from one sentence into the next.

**Numbers.** The model reads most numbers correctly and two kinds wrongly, so the app rewrites those two, and only those, before the text reaches the model:

- **A whole number written with thousands separators** is written out in words: "1,200" is spoken as "one thousand two hundred", "2,500,000" as "two million five hundred thousand". This works up to 999,999,999,999. Decimals after such a number are read digit by digit: "1,200.50" is "one thousand two hundred point five zero".
- **A phone number** is read digit by digit with a short pause between its groups, and 0 is said as "zero": "555 0142" is "five five five, zero one four two", "+61 412 345 678" starts with "plus six one". A run of digits counts as a phone number when it has 7 to 15 digits and something marks it as one: it starts with a plus sign or a 0, has an area code in brackets, has hyphens or dots between its groups ("555-0142", "555-123-4567"), or is a 1300 or 1800 number written 4-3-3 ("1800 123 456"). Groups with only spaces between them and none of those marks are left alone, because they are usually just numbers standing next to each other ("In 2023 4521 people attended", "room 101 2345"); the one exception is a number like "555 0142" or "9555 0142", whose last four digits begin with 0. So write "555-123-4567" rather than "555 123 4567" if you want it read digit by digit.

Everything else is left exactly as typed and read the way the model reads it: plain numbers ("40", "1200", "1500000"), decimals ("3.5"), years, times, dates, percentages and version numbers. When it is unclear whether digits are a phone number, they are left alone: a date such as 2026-10-13, two years ("1987-2004"), two round numbers ("500-1000"), a decimal, anything that begins with a date ("5-10-2026 1200"), or groups that do not end the way a phone number does ("100 200 300"). One known limit the other way: numbers listed with commas and no spaces, such as "100,200,300", are read as one number (one hundred million two hundred thousand three hundred); put a space after each comma. A currency sign stays where it is ("$1,200" reaches the model as "$one thousand two hundred"). If a number is still read wrongly, write it in words. The 5000 character limit is on what you type: the written-out text may be longer, and is simply cut into more pieces.

**What goes wrong, and how it shows.** A sentence that comes out silent stops the request with "The voice engine produced no sound for sentence N" rather than playing nothing. If the engine crashes or stops answering, the request ends with a message, the engine is ended, and the next request starts a fresh one. If the output device disappears, the request ends with the device error. If the engine refuses the reference clip, the message names the clip by its number on the Voices tab and says what can be done (record a louder one, or delete it or record it again). In every case the screen returns to Speak, and the engine's own output is in `data/tts/worker.log` (the previous engine's in `data/tts/worker.prev.log`).

**One thing at a time.** Text to speech, live mode, recording and training share the GPU and the audio devices, so only one of them runs at a time:

- Speak is refused while live mode is running ("Stop live mode before speaking"), during a recording ("Stop recording before speaking") and during training ("Training is running. Wait for it to finish or cancel it before speaking").
- While something is being spoken, starting live mode, a recording or training is refused ("Stop the speech before starting live mode", and so on), and so are **Refresh devices** and deleting a voice.
- An engine that is loaded but not speaking blocks nothing. Starting training stops it first, because both need the GPU.
- Pressing Speak in the very second in which the backend is stopping an idle engine (after its 10 minutes) is refused with "The voice engine of an earlier request is still shutting down ... Try again in a moment." Press Speak again.

**Network.** Speaking makes no network requests: the engine runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and the text you type never leaves this computer. Only `bash tts/setup.sh` downloads anything.

**Where things are.**

```
data/tts/<UTC time>-<voice id>.wav   speech you asked to keep
data/tts/worker.log                  what the voice engine printed (worker.prev.log: the engine before it)
data/tts/tmp/                        temporary: the pieces of the request in progress; emptied when it ends and at start-up
tts/.venv, tts/models, tts/.cache    the environment, the model files and the engine's caches (from the setup)
```

Your recordings are only read. The reference clip is given to the engine as it is and never changed.

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
5. Press **Cancel** during Training and answer yes. It stops within seconds and says "Training was stopped". Press **Resume**: it continues from a saved epoch, not from 1. If you cancelled before the first save at epoch 5, Resume is still offered, the box says "Resume keeps the prepared audio and starts training from the first epoch", and it does not slice the clips again (the pitch and feature stages only fill in what is missing) before it trains from epoch 1.
6. Choose **Quit** in the tray menu mid-training, confirm no `python.exe` from `trainer/.venv` is left running, then reopen the app: the voice shows "Training was interrupted when the app closed" and **Resume** works.
7. Let that short run finish while you wait on the **Live** tab: the voice appears under **Voice** there without a restart. Back on the Voices tab the box says "Trained model: ready" ("Training finished" if you watched it finish there). Start live mode with it, listen on headphones and note how it sounds. Check that `trainer/rvc/logs/voice-<id>/` is gone and `data/voices/<id>/models/rvc/` holds `model.onnx` and `model.index`.
8. Cancel another run, then add a clip to the voice and press **Resume**: it is refused with the "clips changed" message and nothing is deleted. Delete that clip again and press **Resume**: the run continues. **Start over** works as well.
9. Then the full run: **Train again** with 200 (or 300) epochs and let it finish, listen on the Live tab again and compare it with the short run. Note the total time against the estimate.
10. Delete a trained test voice and confirm `trainer/rvc/logs/voice-<id>/` and its `trainer/rvc/assets/weights/` file are gone.

## Stage 4 first run checklist

A human needs to do this once; it has not been done yet. **Nobody has listened to any synthesised speech**: the automated runs judged it by length, level and a speech recogniser only. Start with the setup, then `cd app && npm run tauri dev` and open the **Speak** tab.

1. Run `bash tts/setup.sh`. It prints the non-commercial licence notice, ends with "done in N s", and `backend/.venv/Scripts/python tts/verify.py` prints `"ok": true`. The Speak tab shows no "not set up" box.
2. Give a voice a good reference clip: on the Voices tab, record one clear sentence of 5 to 10 seconds in a quiet room (or pick a voice that has such a clip and check its transcript word for word). On the Speak tab the voice can be chosen; a voice without a usable clip is greyed out with the reason.
3. Type one sentence, choose headphones under **Listen on**, and speak it at **Fast**, then **Balanced** (and **Best** if the voice has a trained model). The first request shows "Getting ready" for 11 to 17 seconds (up to about 35 the very first time on a machine). Listen each time: is every word there, does it sound like the voice, does each sentence start cleanly (no click, blip or stray syllable before it, no long wait) and end without its last word cut short? Note how often Fast gets a word wrong.
4. Paste a long paragraph (ten sentences or more) and press Speak at Balanced. Sound starts within about a second, long before the whole text could have been made, the screen counts "Speaking sentence N of M", and the pauses between sentences sound natural.
5. Press **Stop** (and, another time, Escape) half-way through. The sound ends at once, the screen returns to Speak, and speaking again works straight away.
6. Tick **Keep as a WAV file**, speak two sentences, and play the file named on screen (under `data/tts/`) in another program. Stop a kept request half-way: no file is written.
7. Tick **Send to virtual mic** and speak. In another app (Discord or Windows Sound Recorder) with VoiceMeeter Output as the microphone, the speech is heard or recorded.
8. For a voice with a trained model, speak the same sentences at **Balanced** and at **Best** and compare them for likeness to the real voice. This comparison decides whether Best is worth its extra seconds.
9. One thing at a time: with live mode running, Speak is unavailable and says why; while a long text is being spoken, Start on the Live tab and Record on the Voices tab are refused. Start a training run after speaking: Task Manager shows the text-to-speech `python.exe` gone before training begins.
10. Choose **Quit** in the tray menu while something is being spoken, and confirm no `python.exe` whose command line contains `tts_worker.py` is left running:

    ```powershell
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*tts_worker.py*' | Select-Object ProcessId, CommandLine
    ```
11. The idle unload: speak something, then leave the app alone for 11 minutes on another tab (and, another time, hidden to the tray). The command of step 10 shows no worker any more, and Task Manager shows the graphics memory given back. Speak again: "Getting ready" appears as for a first request and the speech plays.
12. Loudness and volume: speak a text of several sentences; every sentence should be equally loud, and comfortably so (also at Best). Then start live mode with **Listen to output** on, move the listen volume slider well down, stop live mode and speak again: the speech on the listen device is quieter by as much; the virtual mic (step 7) is not.
13. Numbers: speak "It costs 1,200 dollars. Call 555 0142 before noon. In 1987 it was 3.5 percent." The first is said as "one thousand two hundred", the phone number digit by digit with a short pause in the middle, and the year and the decimal as before.
14. A clip that is too quiet: give a new voice one clip recorded very quietly (if the recorder accepts it) or imported from a quiet file. On the Speak tab the voice is greyed out with "Clip 1 is too quiet to copy. Record a louder clip of 5 to 10 seconds."; after recording a normal clip it can be chosen.

## Building the installer and releasing

```bash
cd app && npm ci && npm run installer
```

This needs Node, Rust and an internet connection, and takes about three minutes. It writes `app/src-tauri/target/release/bundle/nsis/Voxweave_<version>_x64-setup.exe`, a per-user NSIS installer for 64-bit Windows, and prints its size and SHA-256. `scripts/build-installer.ps1` does the work:

- `scripts/fetch-uv.ps1` downloads the pinned release of [uv](https://github.com/astral-sh/uv) and refuses it unless the SHA-256 of the archive and of the `uv.exe` in it are the two committed in that script. A `uv.exe` that is already staged is checked against the same hash every time, not trusted. `uv.exe` is bundled, never committed.
- `scripts/stage-installer.ps1` collects what the installer bundles besides the app into `installer/staged/` (not in git): the backend source `backend/aivoice`, `backend/pyproject.toml`, the first-run setup `installer/bootstrap.py`, its pinned package list `installer/requirements.lock`, and `uv.exe`.
- `tauri build` runs with `app/src-tauri/tauri.installer.conf.json`, which adds those files as bundle resources. They are in a separate config because Tauri copies resources next to development binaries too, and fails when one is missing; the main config, which `npm run tauri dev` uses, has none.

**How the installed app differs from the checkout.** The installed app is a release build made by `npm run installer` (which compiles that fact in through the `VOXWEAVE_INSTALLER` variable) that finds its bundled files next to itself (`app/src-tauri/src/setup.rs`). A development build, which is every `npm run tauri dev`, is the source checkout and behaves as before. A release build is never the source checkout: one that was not made by `npm run installer`, or that has a bundled file missing, starts no voice engine and says why. So a plain `npm run tauri build` no longer gives a usable app; it never worked on another PC, and the path to this checkout that it had compiled in is a place another user of a PC could put a program. For the same reason release builds ignore `AIVOICE_BACKEND_DIR`. The installed app keeps everything under `%LOCALAPPDATA%\Voxweave` and at its first start runs `uv run --no-project --python 3.12 setup\bootstrap.py`, which builds `runtime\backend\.venv` from the lock with `uv pip sync --require-hashes`, checks that `onnxruntime-directml` is present and the plain `onnxruntime` is absent, downloads the base model with a SHA-256 check, and writes `runtime\setup.json` last. The setup runs again when the app version changes or when the environment or the model is missing. The backend then runs as `runtime\backend\.venv\Scripts\python.exe -m aivoice` with `PYTHONPATH` at the bundled source, `AIVOICE_DATA` at `data\` and `AIVOICE_INSTALLED=1`. Both Pythons run with `PYTHONNOUSERSITE=1` and `PYTHONSAFEPATH=1`. The base model is downloaded from one fixed revision of its Hugging Face repository.

**The Rust version** is pinned in `app/src-tauri/rust-toolchain.toml`; rustup installs it the first time `cargo` runs there. The installer and the release are built with `--locked`, so with exactly the crates in `Cargo.lock`.

**The package lock.** `installer/requirements.lock` pins the exact version and hashes of every package the installed app gets: the backend's dependencies, the `transcribe` extra and `faster-whisper`, compiled with `--no-emit-package onnxruntime` so that the plain `onnxruntime` is never installed (see Setup). No test packages. It was compiled with the versions of the development environment the backend suite passes in. `scripts/compile-lock.ps1` regenerates it (PowerShell); without `-Constraints` it takes the newest versions, which then need the backend suite run against them.

**The version** has one place of truth, `"version"` in `app/package.json`. `node scripts/version.mjs set 1.2.3` writes it and its copies (`package-lock.json`, `Cargo.toml`, `Cargo.lock`, `backend/pyproject.toml`); `node scripts/version.mjs check` fails if they differ. `tauri.conf.json` reads it from `package.json`.

**Releasing.** Set the version, commit, tag and push:

```bash
node scripts/version.mjs set 1.2.3
git commit -am "release: 1.2.3" && git tag v1.2.3 && git push --follow-tags
```

`.github/workflows/release.yml` then checks that the tag matches the version, runs the frontend, Rust and setup tests, builds the installer from a clean state on `windows-latest` (no caches; every action pinned to a commit, with `.github/dependabot.yml` proposing updates), and publishes a GitHub Release for the tag with the installer attached and its SHA-256 in the notes. `.github/workflows/ci.yml` runs the same tests on every push and pull request. The backend's Python suite is not run on GitHub: it has only ever been run on the development PC, and whether it passes on a runner without a GPU or audio device is unknown. **Neither workflow has run yet.** The installer is not code signed.

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

The first-run setup of the installed app has its own tests, which need nothing but a Python (they use a fake uv and a fake download server), and the version copies are checked by a script:

```bash
python -m unittest discover -s installer/tests
node scripts/version.mjs check
```

`backend/tests/test_rvc_real.py` runs the real DirectML path and needs the GPU, the base model and a voice in `data/pretrained/rvc/`; it also shows that the speech packages did not remove DirectML.

`backend/tests/test_worker_real.py` runs the real training pipeline on the GPU and is skipped unless asked for. It needs the training environment, takes about 10 minutes on a free GPU, and should not run while anything else uses the GPU (live mode, another training run):

```bash
AIVOICE_RUN_TRAIN_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_worker_real.py -q -W error -s
```

It runs a 2-epoch training and converts audio with the result, cancels and resumes a run, kills a worker outright and resumes, and drives the app's own training manager through a whole run and a cancelled one on a temporary data folder (a library voice of 5.2 minutes). Last result: 4 passed in 17 min 2 s, measured while another experiment was sharing the GPU; on a free GPU the first three took 5 min 32 s, and the fourth took 4 min 51 s run by itself. If anything of a test's experiment is left under `trainer/rvc` afterwards, the test run says so with a warning. Without the variable the full backend suite reports those 4 as skipped (last result: 827 passed, 4 skipped).

`backend/tests/test_tts_worker_real.py` and `backend/tests/test_tts_real.py` run the real text-to-speech model on the GPU and are skipped unless asked for. They need the text-to-speech environment, take about a minute and a half together, open no audio device, and should not run while anything else uses the GPU:

```bash
AIVOICE_RUN_TTS_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_tts_worker_real.py backend/tests/test_tts_real.py -q -W error -s
```

The first drives the worker process directly (12 tests). The second drives the app's own text-to-speech manager on a temporary data folder with a temporary voice whose one clip is the sample that ships inside the `f5_tts` package: three sentences at Fast and at Balanced, Stop in the middle of a request, Best with the stage 3 test model if it still exists, unloading, and last the idle unload done by the manager's own housekeeping thread with nobody asking for the status (with the idle time shortened to 2 seconds; it waits for the thread's next round, at most half a minute); a speech recogniser checks that each Balanced sentence says its text. Last result: 23 passed in 91 s. Without the variable the full backend suite reports them as skipped (last result: 1545 passed, 27 skipped). Timings and findings: `docs/superpowers/notes/2026-10-13-stage-4-first-real-run.md`.

## Known limitations

- The accurate pitch detector (harvest) is slower than real time at every preset's window size on the development machine, not only the larger ones, so all three presets use the fast one (dio). Harvest is still selectable under Advanced but will not keep up live with any preset.
- **The installer has never been run.** It was built and its contents checked, and the first-run setup was run for real into an empty folder, but nobody has installed the app, seen the installed app start, updated or uninstalled it, on any PC; see "What has been tested" under Install.
- The installed app has no training and no text to speech; both need the source checkout on a PC with an AMD GPU. Its voices come from voice packs.
- The installer is not code signed, so Windows SmartScreen warns about it.
- The installed app's Python packages need Microsoft's Visual C++ runtime. Most PCs have it; whether the setup's check and message for a PC without it are enough has not been tried.
- Starting the installed app twice starts two voice engines, as starting the source checkout twice does; nothing prevents it.
- **Refresh devices**, the default-device preselection and listen-only mode are covered by automated tests with fake devices but have not been tried on real hardware yet.
- Preset values are untuned starting points; they have not been adjusted by ear.
- The tray menu, the Ctrl+Alt+V hotkey and the checklist above have not been verified by a human yet.
- Transcripts are English only by default (`small.en`); see Settings to change the model.
- Transcription runs on the CPU (the speech engine has no AMD GPU support), so long imports take a while.
- The first import waits for the model download with no separate progress bar.
- **Only synthetic audio has been trained so far** (about 60 seconds of speech-like test audio, 2 to 8 epochs, in the spike and in the automated runs). How a model trained on a real voice sounds is unproven, and so is the stability of a run lasting an hour or more. The Stage 3 checklist is the first real test.
- **Nobody has listened to any synthesised speech yet.** Text to speech was checked by program only: that sound comes out, how long and how loud it is, how fast it is made, and that a speech recogniser hears the words. Whether it sounds like the voice, whether it sounds natural, and whether sentences start and end cleanly are unknown. No real voice has been recorded or trained either, and the app has never been driven by hand. The Stage 4 checklist is the first real test.
- **A sentence can come out with a wrong word, and nothing detects it.** With the model's own sample clip as the reference, a speech recogniser found a wrong or missing word in 10 of 360 sentences at Fast (10 steps) and 3 of 180 at Balanced. These figures come from one reference clip and one recogniser; with a real voice they may differ.
- **The "Best" preset is not what the design first planned.** The design named GPT-SoVITS, a second model fine-tuned on the voice. It was not built: it would have needed a third environment and weeks of uncertain work on this GPU. Best is F5-TTS at full steps followed by the voice's trained RVC model. Whether that is closer to the voice than Balanced is unproven.
- Text to speech is English only (nothing else was tried, and the sentence splitting assumes Western punctuation).
- Dates, abbreviations, symbols and most numbers are read the way the model happens to read them. Only the two kinds of number that were measured as misread are rewritten by the app: whole numbers with thousands separators ("1,200" used to come out as something like "the first 200") and phone numbers (garbled about half the time); see Numbers under Speaking text. How the model reads the rewritten text was checked with a speech recogniser only, at Balanced: "1,200", "2,500,000", "555 0142" and "0412 345 678" were each heard correctly in 5 of 5 tries. **Nobody has listened to it.** Negative numbers, currency signs, ordinals ("1,200th"), numbers of a trillion or more, thousands grouped with spaces or dots, and phone numbers written as one unbroken run without a leading 0 or plus sign are not handled. A full stop followed by a space always ends a piece, so "Dr. Smith" is spoken as two pieces with a pause between.
- The reference clip is chosen by the app; there is no way to pick it by hand yet (the design planned a manual override).
- The listen volume of speech follows the Live tab's slider only as far as the backend knows it (see Outputs under Speaking text): a slider moved while live mode is off is not seen until live mode is started.
- Each sentence is spoken on its own, so intonation does not carry from one sentence into the next, and there is a fixed 0.3 second pause between them.
- The start of each sentence is found by rules measured on one reference clip (the model begins a piece with what is left of the reference clip's last word and a pause, which the app cuts off). With another clip a blip before a sentence, a late start, or, in an unlucky case, a missing short first word are possible.
- There is no speed, pitch or emotion control for speech; the reference clip decides all three.
- Text to speech and live mode cannot run at the same time, and neither can text to speech and recording or training. Starting training stops the voice engine.
- The F5-TTS weights are licensed CC-BY-NC-4.0 (non-commercial). See Text to speech setup.
- Text to speech needs an AMD GPU with the driver named in Training setup and has only been run on an RX 7900 XTX. There is no CPU fallback (the spike measured the processor as far too slow to be one) and no DirectML fallback: text to speech needs AMD's ROCm build of PyTorch, so the design's statement that zero-shot text to speech still works without ROCm does not hold.
- Training depends on two patches carried against a pinned upstream commit of RVC (`81eed5e`): one rewrites a convolution that is about 60 times slower than normal on this AMD ROCm build (training is 3.4 times faster with it), the other stops preprocessing from dropping audio before a pause. `bash trainer/setup.sh` applies them and refuses to continue if the clone has been edited.
- Training needs an AMD GPU with the driver named in Training setup and has only been run on an RX 7900 XTX. There is no CPU fallback.
- Training and live mode cannot run at the same time (both need the GPU), and neither can training and recording. Quitting the app interrupts training; it can be resumed.
- The training time estimate is a fixed rule of thumb, and the first epoch of a first-ever run is much slower while the GPU libraries compile.
- Training reads its progress from RVC's English log lines; on a Windows display language that RVC translates without an English fallback (Chinese, Japanese, Korean and a few others) a run completes but shows no epoch counts or time estimate and makes no checkpoint backups. See Training a voice.
- A model replaced by retraining is kept in `data/trash/` and is never removed automatically.
- Recording, import, drag and drop, clip playback and transcript quality have been tested with fake devices and files only; see the Stage 2 checklist.
- Windows only; DirectML GPU required for live mode (no CPU fallback).
