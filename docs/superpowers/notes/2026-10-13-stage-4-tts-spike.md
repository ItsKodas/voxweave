# Stage 4 spike: text to speech in the user's voice on the RX 7900 XTX (ROCm PyTorch)

Date: 2026-10-13. Everything below was run on this machine inside `D:\Projects\AI Voice\spike\stage4\`.
Raw output of every measurement is in `logs/` (file named next to each table).

## 1. Verdict and recommendation

**F5-TTS works well on this GPU and should be the only synthesis engine.** `f5-tts` 1.1.22 (model F5TTS_v1_Base, Vocos
vocoder) runs on AMD's `torch 2.9.1+rocm7.2.1` wheels in half precision at 5 to 24 times faster than real time
(real-time factor 0.19 at 32 steps, 0.09 at 16, 0.05 at 8), with under 1 GB of VRAM allocated by torch (about 1.3 GB
adapter-wide), fully offline, and with no source patch to the package: three small start-up shims in the worker are
enough. The dilated-convolution slowdown from stage 3 **does not apply**: neither the F5 transformer nor the Vocos
mel vocoder contains a single dilated convolution, and the vocoder costs 7 ms per sentence. Synthesising sentence by
sentence gives first audio after 0.4 s (8 steps), 0.7 s (16) or 1.4 to 1.6 s (32) for any length of text, and
synthesis then stays far ahead of playback. Recommended presets: **Fast** = F5 at 8 steps, **Balanced** = F5 at 32
steps, **Best** = F5 at 32 steps followed by the voice's existing RVC model through the app's `RvcEngine`, one
sentence per call padded to one fixed 20 s window (about 0.5 s extra per sentence, 30 to 40 times real time). Do not
build GPT-SoVITS now: it was not installable cleanly (details in section 8) and fine-tuning it would need a second
patched training stack for a gain the RVC pass already targets. Things nobody has heard yet: all quality statements
here are word-error-rate and signal measurements, not listening tests; there are no real user recordings; and the
RVC model used was trained on synthetic audio. One decision is needed from the owner: the F5 model weights are
**CC-BY-NC-4.0 (non-commercial)**.

## 2. Environment recipe (Git Bash, copy-pasteable)

```bash
S="D:/Projects/AI Voice/spike/stage4"            # in the app: the TTS worker's folder
mkdir -p "$S"/cache/{uv,pip,hf,torch,xdg,mpl,numba,miopen/db,tmp} "$S"/models/{F5TTS_v1_Base,vocos-mel-24khz} && cd "$S"
export UV_CACHE_DIR="$S/cache/uv" PIP_CACHE_DIR="$S/cache/pip" HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
export XDG_CACHE_HOME="$S/cache/xdg" MPLCONFIGDIR="$S/cache/mpl" NUMBA_CACHE_DIR="$S/cache/numba"
export MIOPEN_USER_DB_PATH="$S/cache/miopen/db" MIOPEN_CUSTOM_CACHE_DIR="$S/cache/miopen" MIOPEN_LOG_LEVEL=3
export ROCM_SDK_TARGET_FAMILY=custom TEMP="$S/cache/tmp" TMP="$S/cache/tmp"
PY="$S/venv/Scripts/python.exe"

# 2.1 venv + AMD ROCm 7.2.1 SDK + torch (identical to trainer/setup.sh; the spike pointed UV_CACHE_DIR at
#     D:/Projects/AI Voice/spike/stage3/cache/uv for these two commands only, so nothing was downloaded again)
uv venv --python 3.12 --no-python-downloads venv        # CPython 3.12.14
B=https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1
uv pip install --python "$PY" \
  "$B/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl#sha256=f68989d48df71cbfc3cb68bf705dc37c0f56e9666feddb59a1a0f5ff7539fe1c" \
  "$B/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl#sha256=19e6ee67e13432b7c1e8a4077df795dcb1239546ae314700c4b4d97e5b4b8f63" \
  "$B/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl#sha256=c7fe0b0731af8896093ff69e11496830d3cb6a4aed73e895c60b7cbdc200be92" \
  "$B/rocm-7.2.1.tar.gz#sha256=9084902eaa69213a00a90784ad89e6e5fe73c702df0cc6cc3a70d777c7a6142b"
uv pip install --python "$PY" \
  "$B/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl#sha256=e88bf270163b48f7f27f7ea3db5ffb3be4ba107301933022bcb3c6ddedfeeabb" \
  "$B/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl#sha256=62835605995c7812a3224733abf08e222afddb7a487c493521e613ca46e578ab" \
  "$B/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl#sha256=b5c34d76a2f01fb97e1e0f224bd97fc3a58ec1ee1ef2376af2d72086e9aca34c"

# 2.2 F5-TTS: dependencies under a constraint that pins the ROCm build, then the package itself without its deps
printf 'torch==2.9.1+rocm7.2.1\ntorchaudio==2.9.1+rocm7.2.1\ntorchvision==0.24.1+rocm7.2.1\n' > constraints.txt
uv pip install --python "$PY" -c constraints.txt -r requirements_f5_infer.txt     # file is in this folder; 105 packages
uv pip install --python "$PY" --no-deps "f5-tts==1.1.22"
"$PY" -c "import torch;print(torch.__version__, torch.cuda.is_available())"       # -> 2.9.1+rocm7.2.1 True

# 2.3 model files (plain HTTPS, no Hugging Face client involved)
H=https://huggingface.co
curl -sSL --fail -o models/F5TTS_v1_Base/model_1250000.safetensors $H/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/model_1250000.safetensors
curl -sSL --fail -o models/F5TTS_v1_Base/vocab.txt                 $H/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/vocab.txt
curl -sSL --fail -o models/vocos-mel-24khz/config.yaml             $H/charactr/vocos-mel-24khz/resolve/main/config.yaml
curl -sSL --fail -o models/vocos-mel-24khz/pytorch_model.bin       $H/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin
```

Which release is current: PyPI `f5-tts` 1.1.22 and GitHub tag `1.1.22` (commit `9c614e9657089213efc6a7421b30630be138a3f5`,
published 2026-07-23) are the same release; `main` is three small commits ahead (last 2026-10-11). Requirements per
`pyproject.toml`: `torch>=2.0.0`, `torchaudio>=2.0.0`, no Python upper bound (3.12 is fine). The README's AMD
instructions cover Linux only.

`requirements_f5_infer.txt` (inference-only; full resolved list in `logs/pip_freeze.txt`, 123 packages):
`accelerate>=0.33.0 datasets ema_pytorch>=0.5.2 hydra-core>=1.3.0 librosa matplotlib pydub pypinyin rjieba
safetensors soundfile torchdiffeq tqdm>=4.65.0 transformers unidecode vocos wandb x_transformers>=1.31.14
huggingface_hub psutil`. Left out of the package's declared dependencies: `gradio`, `bitsandbytes`, `torchcodec`,
`transformers_stream_generator`, `cached_path`, `click`, `tomli` (web UI, 8-bit optimiser, `torchaudio.load` backend,
command line). Key resolved versions: numpy 2.5.4, transformers 5.19.0, librosa 1.0.0, vocos 0.1.0,
x-transformers 2.31.7, huggingface-hub 2.2.0, torchdiffeq 0.2.5.

Did pip keep the ROCm torch? Yes. A dry run of the **full** `f5-tts==1.1.22` dependency set under the same
constraint file also leaves torch untouched (139 packages added, only `filelock` and `fsspec` changed), so the
constraint file alone is sufficient protection; the `--no-deps` route is used only to keep the environment smaller.

### Downloads: URL, size, SHA-256

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/model_1250000.safetensors` (repo revision `84e5a410d9cead4de2f847e7c9369a6440bdfaca`) | 1348435761 | 670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38 |
| `.../SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/vocab.txt` (same content as the copy inside the wheel, CRLF line ends) | 13800 | 2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c |
| `https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin` (revision `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`) | 54365991 | 97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a |
| `.../charactr/vocos-mel-24khz/resolve/main/config.yaml` | 461 | da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc |
| `f5_tts-1.1.22-py3-none-any.whl` (PyPI) | 1321095 | f0505dfb5463645caa526bace346ed1c89bcc9acb9ef42fdffd56c2c4c0a09d1 |
| the seven AMD files | | as in the stage 3 notes, section 2.7 (hashes are in the commands above) |

Both weight files load with `weights_only=True` / safetensors; no pickle code is executed.

## 3. Licences

| Item | Licence | Plainly |
| --- | --- | --- |
| F5-TTS code (`SWivid/F5-TTS`) | MIT | Free to use, modify, ship. |
| **F5TTS_v1_Base weights** (`SWivid/F5-TTS` on Hugging Face) | **CC-BY-NC-4.0** (model card and `infer/SHARED.md`; trained on the Emilia dataset) | You may use, copy and adapt them **for non-commercial purposes only**, with attribution. A personal app for your own use is fine. Selling the app with these weights, or using the generated speech for paid work, is not covered. There is no commercial variant of these weights. |
| Vocos code and `charactr/vocos-mel-24khz` weights | MIT | Free to use. |
| GPT-SoVITS code; `lj1995/GPT-SoVITS` weights | MIT; model card says `license: mit` | Bundled third-party models (chinese-hubert-base, chinese-roberta-wwm-ext-large, ERes2NetV2, BigVGAN) keep their own upstream licences, which that card does not restate. |

## 4. Per-process environment variables (TTS worker)

| Variable | Value | Why |
| --- | --- | --- |
| `ROCM_SDK_TARGET_FAMILY` | `custom` | Same as stage 3: the ROCm helper breaks on the space in the path (cosmetic message otherwise). |
| `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR` | `<worker>\cache\miopen\db`, `<worker>\cache\miopen` | Keeps MIOpen off `C:\Users\<user>\.miopen`; must persist (first call 3.7 s with the cache, 10.8 to 17.3 s without). |
| `MIOPEN_LOG_LEVEL` | `3` | Silences warning floods. |
| `TEMP`, `TMP` | `<worker>\cache\tmp` | MIOpen lock files, and wandb's `tmp*wandb-media` / `tmp*wandb-artifacts` folders created at import if wandb is really imported. Also where the package's `preprocess_ref_audio_text` leaks one WAV per call (not used in the recommended design). |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` | Belt and braces; nothing contacts the network (section 5.6). |
| `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `NUMBA_CACHE_DIR` | folders under `<worker>\cache` | Nothing was written to `hf`, `torch` or `xdg` in the whole spike; `numba` 628 KB (librosa) and `mpl` 160 KB were. |
| `WANDB_MODE` | `disabled` | Only relevant if wandb is really imported. |
| `PYTHONIOENCODING`, `PYTHONUNBUFFERED` | `utf-8`, `1` | JSON lines with non-ASCII text, live progress. |
| `PYTHONWARNINGS` | `ignore::SyntaxWarning` | pydub 0.25.1 prints four SyntaxWarnings on first import under 3.12. |

The cwd does not matter (unlike RVC training). Do **not** set `PYTHONDONTWRITEBYTECODE` for this venv: uv installs
no `.pyc` files and imports are already the slowest part of start-up.

## 5. F5-TTS on ROCm: measurements

Reference voice for everything: the clip shipped in the package, `f5_tts/infer/examples/basic/basic_ref_en.wav`
(5.33 s, 24 kHz mono) with its transcript "Some call me nature, others call me mother nature." Test text: ten
sentences of a short story (70 to 95 characters each; `scripts/f5lib.py`). Real parameter names are the package's:
`nfe_step`, `cfg_strength=2.0`, `sway_sampling_coef=-1.0`, `speed=1.0`, `cross_fade_duration=0.15`, `ode_method="euler"`.
Output: **24 000 Hz, mono, float32**. Precision: the package itself selects **float16** on this card
(`load_checkpoint` picks fp16 when the device is `cuda` with capability major >= 7; this card reports `(11, 0)`).

### 5.1 Load and warm-up (`logs/01_smoke_first_run.log`, `02_warmup_*.log`, `04_bench_cuda_fp16.log`, `12_worker.log`)

| Step | Time |
| --- | ---: |
| `import torch` | 2.8 to 3.4 s |
| import of `f5_tts.infer.utils_infer` + `f5_tts.model` as shipped | 13 to 21 s (of which `transformers.pipeline` and its torchvision import about 9 s, `Trainer`/wandb about 3.5 s) |
| the same with the training/ASR-only modules stubbed (section 10, shim 2); includes `import torch` | 7.9 to 11 s in total |
| Vocos load to GPU | 0.2 to 0.6 s |
| F5 checkpoint load to GPU (1.35 GB safetensors, converted to fp16) | 2.2 to 2.9 s (4.2 s once with a busy disk) |
| reference clip preparation | 0.02 s |
| **first synthesis of a process**, MIOpen cache warm | 3.3 to 4.0 s (a 1.5 s job otherwise) |
| first synthesis ever (empty MIOpen cache) | 10.8 s and 17.3 s in two runs (vocoder part 4.2 to 5.8 s) |
| first call at a **new output length**, later in the same process, length never seen before | +0.3 to 0.45 s (one off; then cached on disk) |
| first call at a new length that is already in the disk cache | +0.02 s |
| repeat call, same length | no penalty (spread under 2 %) |
| **worker start to "ready"** including one warm-up synthesis | **14.8 to 19.3 s** with stubs, 24.1 s without |

Unlike RVC training, variable lengths are cheap here: the model has only 15 convolution layers and MIOpen needs a
fraction of a second per new length, once. A worker should run one warm-up synthesis before reporting ready.

### 5.2 Speed (GPU, fp16, median of three warm runs; `logs/04_bench_cuda_fp16.log`, `bench_cuda_default.json`)

"Time to first audio" = time until the first piece is fully synthesised when pieces are produced one after another.
"Package chunks" = the package's own `chunk_text` split (here `max_chars` = 178). RTF = synthesis time / audio time.

| Text | Split | Audio | `nfe_step` 8 | 16 | 32 | First audio 8 / 16 / 32 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 sentence (87 chars) | 1 call | 8.3 s | 0.41 s (RTF 0.049) | 0.78 s (0.094) | 1.55 s (0.187) | 0.41 / 0.78 / 1.55 s |
| 3 sentences (270 chars) | package chunks, 2 calls | 25.6 s | 1.11 s (0.043) | 2.53 s (0.099) | 5.02 s (0.196) | 0.58 / 1.31 / 2.59 s |
| 3 sentences | by sentence, 3 calls | 25.5 s | 1.13 s (0.044) | 2.23 s (0.087) | 4.41 s (0.173) | 0.37 / 0.72 / 1.44 s |
| 10 sentences (845 chars) | package chunks, 6 calls | 80.0 s | 3.58 s (0.045) | 7.92 s (0.099) | 13.12 s (0.164) | 0.59 / 1.31 / 2.11 s |
| 10 sentences | by sentence, 10 calls | 79.6 s | 3.82 s (0.048) | 7.29 s (0.092) | 14.49 s (0.182) | 0.38 / 0.74 / 1.55 s |

`nfe_step=7` (which, like 5, 6, 10, 12 and 16, uses the package's pruned step schedule; 8 and 32 use a uniform one)
measured 0.34 s for one sentence (RTF 0.041). The vocoder is 0.007 s per sentence at every setting. Timed runs
were within about 3 % of their median. One oddity, not investigated: the 3-sentence by-sentence case was slower at 7
steps (1.25 s) than at 8 (1.13 s), reproducibly.

Peak VRAM (`torch.cuda` counters; "adapter" = total minus free as torch reports it for this process's view):

| State | Allocated | Reserved | Adapter-wide |
| --- | ---: | ---: | ---: |
| during model load (fp32 copy before the fp16 conversion) | 1990 MiB | 2002 MiB | |
| loaded, idle | 704 MiB | 728 MiB | about 880 MiB |
| synthesis, any setting in the table above | 863 to 949 MiB | 892 to 996 MiB | 1280 to 1360 MiB |
| one 85 s call (8021 frames, section 7) | 1412 MiB | 1458 MiB | |

Step count does not change memory. Worker host memory (RSS): 1.2 to 1.6 GB with stubs, 1.4 to 1.7 GB without.

Intelligibility check (the app's Whisper small.en on the outputs, `logs/05_asr_bench.log`, `11_asr_refs_stream.log`):
word error rate 0 to 0.6 % at 16 and 32 steps on the 156-word text; 0 to 4 % at 7 and 8 steps, where two kinds of
error appear that the higher settings did not show in by-sentence mode: a dropped or changed word, and once two
words of the **reference transcript leaking** into the output ("Others calm"). Raw output peaks reach 1.0 to 1.37,
so the worker must limit or normalise before writing 16-bit audio.

### 5.3 Other precisions and the CPU (`logs/07_bench_fp32.log`, `07_bench_cpu.log`)

| Configuration | 8 steps | 16 | 32 | Note |
| --- | ---: | ---: | ---: | --- |
| GPU fp16 (package default here) | RTF 0.049 | 0.094 | 0.187 | |
| GPU fp32 (forced) | RTF 0.310 | 0.620 | 1.239 | 6.6 times slower; attention then runs through the "AOTriton backend" path. VRAM 1463 MiB idle. **The worker should force fp16 explicitly** instead of relying on the capability test. |
| CPU fp32, 8 threads | **RTF 86** | | | 392 s for 4.6 s of audio at 8 steps (another job was using the machine, but this is not a matter of tuning). The CPU is not a fallback. |

Because F5 works, Fish Speech and XTTS were not evaluated.

### 5.4 Where the time goes; the dilated-convolution check (`logs/03_profile_parts.log`)

Synchronised hooks on every leaf module, one sentence at 16 steps: `Linear` layers 75 % of leaf time, `LayerNorm`
9 %, the two grouped `Conv1d(1024, 1024, k=31, groups=16)` position-embedding layers 4.6 % (0.54 ms per call),
activations 9 %, all of Vocos about 1.4 %. The rest of the wall time is attention
(`F.scaled_dot_product_attention`), which is not a leaf module. Full convolution inventory of model plus vocoder:
12 x depthwise `Conv1d(512, k=7)`, 2 x grouped `Conv1d(1024, k=31, groups=16)`, 1 x `Conv1d(100, 512, k=7)`.
**Dilated layers: none** (`[m for m in modules if m.dilation[0] > 1]` is empty). So there is no before/after to
report and no phase-reshape patch to carry. The slowdown would return only if the BigVGAN variant
(`F5TTS_Base_bigvgan`, not the default, not installed) or a HiFi-GAN-style vocoder were chosen.

### 5.5 Things that failed and why

1. `import vocos` fails: `AttributeError: module 'torch.distributed' has no attribute 'ReduceOp'`. AMD's Windows
   wheels are built without `torch.distributed` (`torch.distributed.is_available()` is False), and `encodec`, which
   `vocos` imports at import time, uses `torch.distributed.ReduceOp.SUM` as a default argument. Fix: define the
   attribute before importing (shim 1, section 10). Nothing calls it.
2. `torchaudio.load` fails: `ImportError: TorchCodec is required for load_with_torchcodec`. torchaudio 2.9 has no
   other loader. This breaks the package's `infer_process`, `F5TTS.infer` (api.py) and `socket_server.py` as
   shipped. Not fixed by installing torchcodec (untested on this build, needs matching FFmpeg DLLs); instead the
   worker reads audio with `soundfile` and calls `CFM.sample` / `infer_batch_process` directly, which take a tensor.
3. `cached_path` is not installed, so `f5_tts.api` cannot be imported. Intended: the worker never uses that class.

### 5.6 Offline behaviour (`logs/06_offline_check.log`)

`scripts/offline_check.py` replaces `socket.connect`, `connect_ex`, `getaddrinfo` and `create_connection` with
functions that record and refuse, then loads both models from local paths, prepares the reference with a transcript,
synthesises, and runs the package's own streaming generator. Result: **0 network attempts**, both without any
offline variable and with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Calls that would contact the network if the
worker used them (found by reading the code):

| Call | What it fetches |
| --- | --- |
| `load_vocoder(is_local=False)` (the default) | `hf_hub_download("charactr/vocos-mel-24khz", ...)` twice |
| `F5TTS(...)` in `api.py`, `infer_cli`, `infer_gradio` with no `ckpt_file` | `cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")` |
| `preprocess_ref_audio_text(audio, ref_text="")` | downloads and runs `openai/whisper-large-v3-turbo` to transcribe the reference. The app always has the transcript, so always pass it. |
| `infer_gradio`, training (`wandb`) | not used |

## 6. Sentence streaming (`logs/08_stream.log`, `12_worker.log`)

**What the package offers.** There is no streaming inside one model call: the flow-matching sampler produces the
whole mel spectrogram of a piece at once. The package works on text chunks:

- `chunk_text(text, max_chars)` splits at `; : , . ! ?` followed by a space and packs pieces up to `max_chars`
  UTF-8 bytes, where `max_chars = ref_text_bytes / ref_seconds * (22 - ref_seconds) * speed` (178 for the shipped
  clip). So by default a long input becomes pieces of up to about 17 s that may **end in the middle of a sentence
  at a comma**.
- `infer_batch_process(..., streaming=False)` (what `infer_process` and the API class use) submits **all** chunks to
  a thread pool at once, waits for all of them, joins them with a 150 ms linear cross-fade and returns one array:
  nothing can play early. Measured: 3.2 s (8 steps) and 12.3 s (32 steps) until anything is returned for the
  10-sentence text.
- `infer_batch_process(..., streaming=True, chunk_size=2048)` (used by `socket_server.py`) synthesises the chunks
  one after another and yields each finished chunk in 2048-sample slices, with **no** cross-fade or gap handling.
  Measured first audio for the 10-sentence text: 0.50 s / 1.00 s / 2.01 s at 8 / 16 / 32 steps.

**Cleanest approach for the app:** split into sentences in the worker (then `chunk_text` only for a sentence longer
than `max_chars`), one model call per piece, emit each piece as soon as it is done. Measured for the 10-sentence text:

| | 8 steps | 16 | 32 |
| --- | ---: | ---: | ---: |
| time to first audio, by sentence | **0.39 s** | **0.72 s** | **1.44 s** |
| time to first audio, package chunks, streaming mode | 0.50 s | 1.00 s | 2.01 s |
| whole 79.6 s of audio ready after | 3.75 s | 7.01 s | 13.96 s |
| per sentence | 0.34 to 0.44 s | 0.64 to 0.78 s | 1.28 to 1.55 s |
| smallest lead of synthesis over playback | never behind | never behind | never behind |
| through the prototype worker (files written, edges trimmed) | 0.42 s | | 1.60 to 1.68 s |

A shorter first piece helps further if wanted: the first clause alone (47 chars, 4.5 s of audio) takes 0.30 / 0.59 /
1.16 s. Because synthesis is 5 to 20 times faster than playback, a player never runs dry after the first piece.

**Consistency across separately synthesised sentences** (32 steps, ten sentences, measured with librosa pYIN):

| | Same seed for every sentence | Different seed per sentence |
| --- | --- | --- |
| median pitch per sentence | 92.8 to 106.9 Hz (sd 4.2 Hz) | 82.4 to 93.6 Hz (sd 4.3 Hz) |
| level per sentence | -18.6 to -16.0 dBFS (sd 0.75 dB) | -18.9 to -15.8 dBFS (sd 0.81 dB) |
| speaking rate | 10.5 chars/s for every sentence (the duration is set by a formula, see below) | same |
| non-speech at the start of each piece | **1.0 to 1.9 s** | 1.1 to 2.1 s |
| non-speech at the end | 0 to 64 ms | 0 to 26 ms |

Each sentence is conditioned on the same reference clip, so the voice does not drift; pitch and loudness vary by
about a semitone and under 1 dB between sentences, which is ordinary sentence-to-sentence variation. What separate
synthesis cannot do is carry intonation across a sentence boundary (each sentence starts fresh). Whether that is
audible as "list reading" needs a listening test.

**Gap and cross-fade handling needed:**

1. **Trim the start of every piece.** The total length of a piece is fixed before sampling as
   `ref_frames / ref_text_bytes * gen_text_bytes / speed`; the shipped reference speaks slowly with a pause, so each
   piece is given more time than it needs and the model puts 1 to 2 s of quiet at the start. With the reference
   used untrimmed this is room tone at about -50 dBFS rather than digital silence, so trim on 20 ms frame levels
   relative to the piece's loudest frame (the prototype uses -30 dB), not on a sample threshold. Without trimming,
   the audible start is 1 to 2 s later than the numbers above.
2. **No cross-fade is needed** between pieces that start and end in a pause: after trimming, apply 10 ms fades and
   insert a fixed pause (the prototype uses 300 ms between sentences). A plain concatenation of untrimmed pieces
   also measured clean (WER 0.6 %), just with long pauses.
3. The package's 150 ms cross-fade exists because its chunks can end mid-sentence; with sentence pieces it is not used.
4. **End-of-piece truncation happens.** In the package-chunk runs at 7, 16 and 32 steps the last word of one chunk
   ("...along the beach.") was missing, because the fixed duration was slightly too short for that chunk. It did not
   happen in the by-sentence runs here, but the mechanism is the same; see open risks.
5. Peak-limit each piece (raw peaks up to 1.37).

## 7. Reference clip requirements (`logs/09_refs.log`, `11_asr_refs_stream.log`)

There are no real recordings, so longer reference clips of the same shipped voice were made by synthesis (32
steps) with exact transcripts, stored at 48 kHz like the app's clips. This measures mechanics, speed and limits,
not how much a longer real clip improves similarity. Target: one 87-character sentence.

| Reference | Handling | Reference actually used | Total frames | 8 steps | 32 steps | WER 8 / 32 | Peak alloc |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 3 s | raw | 3.14 s | 1024 | 0.33 s | 1.21 s | 0 / 0 % | 884 MiB |
| 5 s (shipped) | raw | 5.33 s | 1333 | 0.39 s | 1.58 s | 0 / 0 % | 906 MiB |
| 8 s | raw | 8.28 s | 1534 | 0.43 s | 1.69 s | 0 / 0 % | 920 MiB |
| 15 s | raw | 14.87 s | 2161 | 0.70 s | 2.75 s | 0 / 0 % | 965 MiB |
| 30 s | raw | 31.17 s | 3692 | 1.48 s | 5.94 s | **80 / 93 %** | 1073 MiB |
| 15 s | package `preprocess_ref_audio_text` | cut to 9.18 s, transcript **not** cut | 1335 | 0.40 s | 1.55 s | 0 / 6.7 % | 906 MiB |
| 30 s | package `preprocess_ref_audio_text` | cut to 9.18 s, transcript **not** cut | 1087 | 0.34 s | 1.30 s | **100 / 80 %** | 890 MiB |

Findings:

- **Cost grows with reference length**, because the reference is part of every call: 1.2 s per sentence with a 3 s
  clip, 1.7 s with 8 s, 2.75 s with 15 s (32 steps). A 48 kHz clip is resampled by the worker (torchaudio's
  `Resample` works without torchcodec); no error at any length.
- **Reference plus generated audio per call must stay under about 30 s** (the training clips' maximum). One call
  with the 5 s reference: 22 s total and 31 s total are clean (WER 0 %), 37.5 s total already has errors (4.7 %),
  52.6 s gives 28 %, 85.6 s gives 94 %. Nothing raises an error: the audio simply turns to babble. There is no
  hard limit in code below 65536 frames (about 11.6 minutes); memory is not the constraint (1.4 GB at 85 s).
- A 30 s reference leaves no room and fails silently; the package formula even yields a negative `max_chars` (-97).
- **Do not use the package's `preprocess_ref_audio_text` with app-supplied transcripts.** It cuts audio longer than
  12 s at a silence (or hard at 12 s) but keeps the whole transcript, so text and audio no longer match and the
  duration formula is wrong (5.05 s allotted for a sentence that needs 8 s). It also strips edge silence, writes a
  temp WAV per call that it never deletes, and is where the Whisper download hides.
- Very short and awkward inputs work: "Yes." (1.3 s), "Thank you." (0.9 s), a question, digits ("42 dollars and 50
  cents"), text without punctuation. Texts under 10 bytes get `speed` 0.3 from the package logic.

**What the app should supply:** one clip of **5 to 10 s** (hard range 3 to 12 s), a single take without long
pauses, clean, not clipped, with its exact transcript ending in punctuation; mono, any sample rate. Fluent, evenly
paced clips matter because the reference's characters-per-second sets the speaking rate and its pauses turn into
leading silence. The app can choose the clip automatically (length in range, highest level without clipping) and
cache the prepared tensor per voice. Limit each generated piece to `22 - ref_seconds` seconds of estimated audio,
which is what `max_chars` does.

## 8. The RVC pass (`logs/10_rvc_pass.log`, `10b_rvc_longest.log`, `10c_rvc_joins.log`, `10d_rvc_overlap_padded.log`)

Run with `backend/.venv/Scripts/python` and `PYTHONDONTWRITEBYTECODE=1`, importing `aivoice.engines.rvc.RvcEngine`
unchanged (`scripts/rvc_pass.py`), model `spike/stage3/handoff/data/pretrained/rvc/final3` (40 kHz, trained on
synthetic audio: quality is meaningless), ContentVec `data/pretrained/vec-768-layer-12.onnx`, preset
`high_quality` (dio pitch, index rate 0.6). Input: F5 output upsampled from 24 to 48 kHz with soxr, so that
`convert(audio, 48000, 0.0, params)` returns 48 kHz (calling it at 24 kHz would throw away everything the RVC
model generates above 12 kHz). Mechanics confirmed: output finite, same length as input, and Whisper still
transcribes the converted speech almost word for word even with this junk model.

**One call, fresh sessions per length:**

| Length | First call | Later calls | Speed |
| ---: | ---: | ---: | ---: |
| 5 s | 0.33 s | 0.15 s | 33x real time |
| 15 s | 0.60 s | 0.35 s | 43x |
| 20 s | 0.75 s | (0.52 s as a fixed padded window) | 38x |
| 30 s | 1.03 s | 0.71 s | 42x |
| 36 s | 1.25 s | | 29x (first call) |
| **37 s and longer** | **fails** | | `[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running MatMul node. Name:'MatMul_326' ... 80070057 The parameter is incorrect.` in the ContentVec session on DirectML |

**Longest buffer that converts correctly in one call: 36 s** (37, 38, 39, 40, 45, 50, 55 and 60 s all fail the same
way, in the ContentVec model, before the voice model runs). So "60 s in one call" is not possible; 60 s was
converted in chunks:

| Chunking of 60 s | Calls | First call | Later calls (median) | Total | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 s chunks, no context, no fade (naive) | 12 | 0.36 s | 0.150 s | 1.97 s | 31x |
| 5 s + 0.5 s context each side + 50 ms fade (6.05 s window) | 12 | 0.44 s | 0.171 s | 2.34 s | 26x |
| 10 s + 0.5 s context + 50 ms fade (11.05 s window) | 6 | 0.51 s | 0.269 s | 1.87 s | 32x |
| 10 s + 1 s context + 80 ms fade | 6 | 0.52 s | 0.302 s | 2.04 s | 29x |
| 2 s + 0.5 s context + 50 ms fade | 30 | 0.27 s | 0.092 s | 2.94 s | 20x |

**Session shape behaviour.** The same session pair given 15 s, then 5 s: the 5 s calls take 0.24 s instead of 0.15 s
(1.6 times slower), and a too-long input kills the request with the error above. The penalty is milder than the
5 to 8 times seen in live mode, but it is there, so the fixed-shape rule holds offline too.

**Artefacts at chunk boundaries.**

- Naive chunking is wrong: every chunk starts with about 20 ms of zeros (the engine's `_fit` pads the start because
  the model returns one 20 ms frame less than the input), so each join has a hole. Measured: median level step
  across the 11 joins 49.6 dB, largest sample jump 1.64 times the file's 99.9th percentile (controls: 0.6 dB, 0.43).
- With context and a cross-fade there are no clicks (largest sample jump at joins 0.02 to 0.04, below the controls'
  0.11 to 0.20). But the two renderings being cross-faded are **not phase-coherent**: for six loud positions, the
  same half second of input rendered from two different windows correlated at +0.12, +0.07, +0.02, +0.08, +0.04 and
  -0.24 (best within +-5 ms of shift: 0.09 to 0.49), and their 50/50 mix was 2 to 4 dB quieter than either. So a
  cross-fade that lands inside voiced speech produces a short dip or phasey moment. This is structural (the model's
  harmonic source restarts its phase in every window; live RVC tools align with SOLA for this reason), not a
  property of the test model. The measured level step at joins was still above the controls (median 4.6 dB for
  5 s chunks against 0.8 dB).

**How an offline converter should chunk** (recommendation):

1. Convert **one F5 piece (sentence) per call**. Pieces are at most about 17 s by construction and begin and end in
   a pause, so there is never a join inside speech and no cross-fade is needed.
2. Keep **one fixed shape**: copy the piece into a zero buffer of a constant length **W = 20 s at 48 kHz**
   (960 000 samples), convert, and cut the result back to the piece's length. Measured with 12 pieces of 3 to 15 s:
   first call 0.75 s, then **0.52 s per piece** (max 0.55 s) regardless of the piece's length; the padded region
   comes back at rms 0.0001. W = 30 s costs 0.75 s per piece and about twice the memory; 20 s is enough.
3. Create the session pair once per synthesis job (1.0 to 1.9 s) and run one zero-buffer call to warm it; drop it
   when the job ends, or keep it if "Best" is used repeatedly.
4. Only if a piece is longer than W: split it at its quietest 20 ms frame near the middle and convert the halves
   separately.
5. Memory: the DirectML sessions added about 1.7 GB of dedicated VRAM for 5 to 12 s windows, about 2.5 GB at
   W = 20 s, about 4.5 to 5 GB at 30 to 36 s (adapter-wide counter while other jobs were also using the card, so
   treat these as rough).

Cost of "Best" per sentence: about 1.5 s of F5 at 32 steps plus 0.52 s of RVC: first audio after roughly 2.1 s, and
about 0.25 of real time overall.

## 9. GPT-SoVITS: feasibility only (nothing installed, nothing run)

Source read: `RVC-Boss/GPT-SoVITS` main at `1b47ca07ef43a284e0b8bd7f29d3e1e3ff76082a` (2026-10-09), cloned to
`src/GPT-SoVITS`. Findings marked (checked) were confirmed by me in the clone or on PyPI; the rest come from a
reading pass over the repository and are unverified by execution.

- **Version.** Latest tagged release `20250606v2pro` (2025-06-06); `main` has since added v5 ("Release the new V5
  version", changelog 2026-10-05). Model generations in code: v1, v2, v3, v4, v2Pro, v2ProPlus, v5dev, v5turbo;
  the WebUI default is v2Pro. Licence MIT for code and, per the model card, weights (checked).
- **Requirements.** Tested table in the README (checked): Python 3.12 only with torch 2.7.1 on CUDA 11.8/12.8;
  **no torch 2.9, no AMD/ROCm on Windows** (the ROCm option in `install.sh` is Linux-only; `install.ps1` offers
  CU126, CU128, CPU). `requirements.txt` pins `numpy<2`, `librosa==0.10.2`, `transformers>=4.51,<5`, `peft<0.18`,
  `torchmetrics<=1.5`, `pydantic<=2.10.6`, `gradio<5`: incompatible with the F5 environment above, so it needs
  **its own third venv**.
- **Does it install cleanly on this setup? No.** What blocks a quick, clean install and run:
  1. `jieba_fast` is imported unconditionally on the inference path (`TextPreprocessor.py:13` -> `text/chinese.py:19`)
     and exists on PyPI **only as a source archive with a C extension** (`jieba_fast-0.53.tar.gz`; checked), so it
     needs a compiler or a stub. `pyopenjtalk` is also source-only (checked; Japanese only, loaded lazily).
     `requirements.txt` additionally forces a source build of `opencc`.
  2. `GPT_SoVITS/module/distrib.py:32` is `def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):`
     (checked): the same import-time crash as encodec on this torch build, in the repository's own code, on the
     path of every inference and of feature extraction.
  3. `torchaudio.load` is called in the inference path (`TTS.py:119`, `:871`) and in `2-get-sv.py`: fails without
     torchcodec on torchaudio 2.9.
  4. The SoVITS decoder (`module/modules.py` `ResBlock1`, dilation (1, 3, 5)), the v4/v5 vocoder (same generator)
     and v3's BigVGAN **are dilated-convolution stacks**: the 60 times slowdown from stage 3 applies to every
     inference call and to SoVITS training until the phase-reshape rewrite is ported.
  Per the instruction, I stopped here rather than patching a third-party tree to get one timing.
- **Zero-shot inference needs** (v2ProPlus, sizes from the Hugging Face API): GPT `s1v3.ckpt` 155 MB, SoVITS
  `s2Gv2ProPlus.pth` 200 MB, `chinese-hubert-base` 189 MB, `chinese-roberta-wwm-ext-large` 652 MB (loaded even for
  English), speaker-verification `pretrained_eres2netv2w24s4ep4.ckpt` 108 MB: about **1.3 GB**. Reference clip must
  be **3 to 10 s** (`TTS.py:917`, checked: it raises outside that range). Output 32 kHz (v1/v2/v2Pro), 24 kHz (v3),
  48 kHz (v4/v5). It has fragment and token-level streaming modes in `api_v2.py`.
- **Fine-tuning involves:** slicing, optional denoise, ASR to a `.list` file, three feature-extraction scripts
  (text/BERT, HuBERT + 32 kHz audio, semantic tokens; plus speaker vectors for v2Pro), then **two trainings**:
  SoVITS (`s2_train.py`, GAN, default 8 epochs) and GPT (`s1_train.py`, PyTorch Lightning, default 15 epochs). The
  repository gives no training times; the README says 1 minute of data is enough.
  - SoVITS training skips distributed mode on one GPU (`s2_train.py:58`, checked), so it needs only fix 2 above,
    plus the dilated-convolution rewrite to be usable, as in stage 3.
  - **GPT training is hard-blocked on this torch build**: `s1_train.py:120` selects
    `DDPStrategy(process_group_backend=... "gloo")` whenever `torch.cuda.is_available()`, with `devices=-1` (checked),
    and `AR/data/bucket_sampler.py:40` raises `RuntimeError("Requires distributed package to be available")`
    (checked). Both need patches, and whether Lightning itself runs without `torch.distributed` is unknown.
- **Honest estimate.** English zero-shot inference: one source patch, two start-up shims, the convolution rewrite,
  a third environment and 1.3 GB of models; perhaps two to three days to a first timing, with unknown quality
  against F5 for this voice. Fine-tuning inside the app: a second trainer comparable to the whole of stage 3
  (six stages, two models, new log parsers, resume/cancel logic), on a code base that upstream tests only on CUDA
  and that is moving quickly (v5 landed a week ago), with at least three more patches and three untested unknowns
  (Lightning without distributed support, mixed precision under Lightning on ROCm, GAN training speed after the
  rewrite). I would put it at two to three weeks with a real chance of a dead end. The F5 + RVC pass needs **no
  new training code at all**: the voice model already exists after stage 3, and the pass is 80 lines around an
  engine the app already ships. Recommendation: do not build GPT-SoVITS; revisit only if listening tests show the
  F5 + RVC "Best" preset is not good enough.

## 10. Recommendation

### 10.1 Presets

| Preset | Engine | Settings | First audio (10 sentences) | Speed | Extra VRAM |
| --- | --- | --- | ---: | ---: | ---: |
| **Fast** | F5-TTS zero-shot | `nfe_step=8`, by sentence | 0.4 s | about 20x real time | about 1.3 GB |
| **Balanced** | F5-TTS zero-shot | `nfe_step=32`, by sentence | 1.4 to 1.7 s | about 5.5x | about 1.3 GB |
| **Best** | F5-TTS + the voice's RVC model | `nfe_step=32`, then `RvcEngine.convert` per sentence at 48 kHz in a fixed 20 s window | about 2.1 s | about 4x | about 1.3 GB + about 2.5 GB |

All three use `cfg_strength=2.0`, `sway_sampling_coef=-1.0`, `speed=1.0`, fp16. Notes: 16 steps (0.72 s to first
audio, 11x) measured as accurate as 32 here and is the obvious middle setting if "Balanced" should feel quicker;
which of 8 / 16 / 32 sounds acceptable is a listening decision. "Best" is only "best" if listening confirms that
the RVC pass improves likeness without adding artefacts; it can be offered only for voices that have a trained
model. Output is 24 kHz from F5 (resample to 48 kHz for the app) and 48 kHz from the RVC pass.

### 10.2 Worker process

A long-lived process in its own venv, speaking JSON lines; a working prototype is `scripts/tts_worker.py` with the
driver `scripts/worker_client.py` (`logs/12_worker.log`).

- **Start:** `<venv>\Scripts\python.exe -u tts_worker.py` with the environment of section 4 plus `TTS_MODEL_DIR`.
  Kill with `taskkill /PID <pid> /T /F` (the PID from `Popen` is uv's launcher shim, as in stage 3).
- **stdin, one JSON object per line:**
  - `{"cmd":"set_reference","wav":PATH,"text":STR}`: 3 to 12 s clip, any sample rate, with its transcript.
  - `{"cmd":"synthesize","id":STR,"text":STR,"out_dir":PATH,"nfe":8|16|32,"speed":1.0,"seed":INT|null}`
  - `{"cmd":"cancel","id":STR}`
  - `{"cmd":"shutdown"}` (closing stdin does the same)
- **stdout, one JSON object per line** (nothing else is ever written there; library chatter goes to stderr):
  - `{"event":"ready","import_s","load_s","warmup_s","device","torch","dtype","vram_reserved_mb","sample_rate"}` once.
  - `{"event":"reference","seconds"}`
  - `{"event":"started","id","pieces":N}`, then per piece
    `{"event":"audio","id","index","path","samples","sample_rate","seconds","elapsed","text"}`; the file is a
    float32 WAV written to `<out_dir>/<id>_<index>.wav` through a `.part` rename, already edge-trimmed and
    peak-limited, so the backend can play or convert it the moment the event arrives. Progress = `index + 1` of
    `pieces`.
  - `{"event":"done","id","pieces","seconds","elapsed"}` or `{"event":"cancelled","id","index","elapsed"}`.
  - `{"event":"error","id","code","message"}` with `code` in `bad_request`, `bad_json`, `unknown_command`,
    `out_of_memory`, `internal`. Request errors leave the worker alive (tested: no reference, 31 s reference,
    unknown command). If the process dies, the backend sees EOF and reads stderr.
- **Cancel:** a reader thread sets a flag that a forward pre-hook on the transformer checks, so a request stops
  within one sampling step. Measured: **5 and 13 ms** from sending `cancel` to the `cancelled` event, and the next
  request ran normally. Files already announced stay on disk for the backend to delete.
- **The RVC pass belongs in the backend**, not in this worker: it uses `backend/.venv` (onnxruntime-directml) and
  `RvcEngine`, consumes each `audio` event's file, and emits the converted piece. That keeps torch out of the
  backend and DirectML out of the worker.
- **Stay loaded or load on demand?**

  | | Loaded and idle | Load on demand |
  | --- | --- | --- |
  | VRAM | about 0.9 GB adapter-wide (704 MiB allocated), 1.3 GB while synthesising | 0 when not in use |
  | RAM | 1.2 to 1.6 GB | 0 |
  | Delay before the first audio of a request | 0.4 to 1.6 s | **15 to 19 s** start-up (8 to 11 s imports, 3 to 4 s model load, 4 s warm-up), then the same |

  Recommendation: start the worker when the user opens the text-to-speech view (or on the first request), keep it
  for the session, stop it after some minutes idle and whenever training starts. It fits beside live conversion
  (24 GB card), but both will slow each other; see risks.
- **Share `trainer/.venv` or own environment? Own environment.** A dry run of installing the F5 dependency set
  into `trainer/.venv` resolves (55 packages added, trainer pins kept: numpy 1.26.4, transformers 4.49.0, librosa
  0.10.2; `logs/13_shared_env_dry_run.log`), and a throwaway venv built exactly that way did synthesise correctly
  (`logs/13_shared_env_smoke.log`). But it only works with the import stubs: as shipped, `import f5_tts.model`
  pulls in `datasets` -> `pyarrow` 26, which refuses numpy below 2 (`ImportError: pyarrow requires NumPy 2.0 or
  newer, found 1.26.4`). Sharing would also tie TTS upgrades to the trainer's pins and to a carefully verified
  training environment, for no real saving: with one shared uv cache the 2 GB of AMD wheels are hard links, and
  the F5-only packages add about 1.1 GB. A separate `tts/.venv` built by a `tts/setup.sh` modelled on
  `trainer/setup.sh` is the clean answer.

### 10.3 Workarounds (worker-side shims; no patch to any installed package)

```diff
+# 1. REQUIRED. AMD's Windows torch has no torch.distributed; `encodec` (imported by `vocos`) uses
+#    torch.distributed.ReduceOp.SUM as a default argument value at import time.
+import torch, torch.distributed as _dist
+if not hasattr(_dist, "ReduceOp"):
+    import enum
+    _dist.ReduceOp = enum.Enum("ReduceOp", "SUM PRODUCT MIN MAX BAND BOR BXOR")
+
+# 2. OPTIONAL (start-up 24 s -> 15 s; lets these five packages be left out of the venv; avoids wandb temp folders).
+#    f5_tts.model.__init__ imports Trainer, and utils_infer imports transformers.pipeline, even for inference.
+import sys, types
+class _Stub(types.ModuleType):
+    def __getattr__(self, name):
+        if name.startswith("__"):
+            raise AttributeError(name)
+        return type(name, (), {})
+for _n in ("wandb", "accelerate", "accelerate.utils", "ema_pytorch", "datasets", "transformers"):
+    sys.modules[_n] = _Stub(_n)
+
+from f5_tts.infer import utils_infer as U     # only after the shims
```

3. Never call `torchaudio.load` (needs torchcodec): read audio with `soundfile`, resample with
   `torchaudio.transforms.Resample`, and call `model.sample(...)` + `vocoder.decode(...)` (or
   `infer_batch_process((tensor, sr), ...)`) directly. `f5_tts.api.F5TTS`, `infer_process` and `socket_server.py`
   are unusable as shipped on this build.
4. Do not call `preprocess_ref_audio_text` (section 7); validate and prepare the reference in the worker.
5. Load models from explicit local paths (`load_vocoder("vocos", True, <dir>, device)`,
   `load_model(DiT, cfg.model.arch, <ckpt>, "vocos", <vocab>, "euler", True, device)`); pass `dtype=torch.float16`
   through `load_checkpoint` explicitly in the real worker.
6. Trim each piece's edges on frame levels, fade 10 ms, limit the peak (section 6).
7. Stage 3's `ROCM_SDK_TARGET_FAMILY=custom` and MIOpen cache variables; `TEMP`/`TMP` redirected.

Shim 2 was exercised only on the inference path used here; if a future package version touches one of the stubbed
names at import time it will fail loudly at start-up, not silently.

## 11. Open risks

1. **Nobody has listened to anything.** Every quality statement is a Whisper word error rate or a signal measure.
   Likeness to a real user's voice, naturalness at 8 against 32 steps, sentence-to-sentence prosody, and whether
   the RVC pass helps are all untested. A listening session on real recordings should come before building the UI
   around three presets.
2. **No real recordings were used.** The reference is the package's sample voice; the long reference clips were
   themselves synthetic; the RVC model was trained on synthetic audio. Real clips have room tone, breaths and
   varied pacing, which affect leading silence, speaking rate and truncation.
3. **Weights licence: CC-BY-NC-4.0.** Fine for personal use; a blocker for anything commercial.
4. **Silent failure modes.** Too-long reference, too-long piece, or a duration that is too short all produce
   wrong audio without any error (babble, a dropped last word, reference words leaking in at low step counts).
   The worker must enforce the limits itself. A cheap guard would be to transcribe each piece with the app's
   Whisper and retry on mismatch; not built, and it would add CPU time per sentence.
5. **Duration is a formula, not a prediction.** Speaking rate and leading silence come from the reference clip's
   characters per second. A slow or pause-heavy reference gives slow speech with long lead-ins (1 to 2 s here);
   a fast one risks cut-off endings. `speed` and a small duration margin are the levers; neither was tuned.
6. **English only was tested.** The model also covers Chinese; other languages need community checkpoints with
   their own licences. Numbers, abbreviations and names are read by the model without a text normaliser.
7. **fp16 is selected by a capability test** inside the package; fp32 is 6.6 times slower on this build. Force it.
8. **GPU sharing.** Another session was using the machine during part of this spike. With all CPU cores
   busy (my own CPU benchmark running), one F5 sentence at 32 steps took 3.6 s instead of 1.55 s: the worker needs a free core.
   The table in 5.2 was measured on a quiet machine. Live conversion plus TTS at once was not measured.
9. **The 36 s DirectML limit** in the ContentVec model was found by experiment, not explained; it may differ with
   another onnxruntime or driver version. The fixed 20 s window stays far below it.
10. **RVC level and tone across pieces** could not be judged with the synthetic model (its output level changed
    with the window length: rms 0.069 / 0.059 / 0.053 for 3 / 6 / 11 s windows). Check with a real model.
11. **Package drift.** `f5-tts` releases often (1.1.20 to 1.1.22 between April and July); the shims depend on
    import structure. Pin `f5-tts==1.1.22` and the dependency versions in `logs/pip_freeze.txt` (the recipe's
    requirement file is unpinned and resolved to very recent releases: numpy 2.5, transformers 5.19).
12. **Driver/ROCm coupling** as in stage 3. The worker's `ready` event should be the health check.
13. Not evaluated: batching several sentences in one model call (would raise throughput, not first-audio time);
    the `F5TTS_v1_Small` config (no official weights); `torch.compile`; the v1 "no zero init" variant; long soak.

## 12. Directories created (sizes at the end of the spike)

Everything is under `D:\Projects\AI Voice\spike\stage4\` (8.3 GB as `du` counts it; most of the venv is hard links
into uv caches, so the additional disk use is about 2.6 GB: 1.4 GB models + 1.1 GB cache).

| Path | Size | Contents |
| --- | ---: | --- |
| `venv` | 6.5 GB apparent | Python 3.12.14, ROCm torch, F5-TTS inference set. About 5.3 GB of it is hard-linked from `spike/stage3/cache/uv` (the AMD wheels, reused as instructed), the rest from `cache/uv` here |
| `cache/uv` | 1.1 GB | the non-AMD packages downloaded for this spike |
| `cache/miopen` | 477 KB | MIOpen kernel and find caches (keep) |
| `cache/numba`, `cache/mpl`, `cache/tmp` | 628 KB, 160 KB, 124 KB | librosa JIT cache, matplotlib font cache, lock files |
| `cache/hf`, `cache/torch`, `cache/xdg`, `cache/pip` | empty | nothing was ever written there |
| `models` | 1.4 GB | `F5TTS_v1_Base/` (checkpoint, vocab), `vocos-mel-24khz/` |
| `src/F5-TTS` | 3.7 MB | shallow clone of tag 1.1.22, for reading |
| `src/GPT-SoVITS` | 43 MB | shallow clone of main at 1b47ca07, for reading |
| `out/evidence` | 13 MB | 12 WAV files (below) |
| `logs` | 572 KB | one log per experiment, `pip_freeze.txt`, `bench_cuda_default.json`, `importtime.txt` |
| `scripts` | 89 KB | throwaway scripts named in this file |
| `requirements_f5_infer.txt`, `constraints.txt`, `NOTES.md` | | |

Removed again during the spike: `venv_shared_test` (the trainer-pins experiment) and about 110 MB of intermediate
WAV output.

Evidence files in `out/evidence/`:
`f5_1sentence_nfe8.wav`, `f5_1sentence_nfe16.wav`, `f5_1sentence_nfe32.wav` (same sentence at the three step
counts); `f5_3sentences_by_sentence_nfe32_untrimmed.wav` (shows the 1 to 2 s lead-ins);
`f5_10sentences_worker_nfe32_trimmed_300ms_gaps.wav` (the prototype worker's output joined, 83 s);
`shipped_reference_basic_ref_en.wav`; `refstudy_reference_clip_8s_48k.wav`, `refstudy_ref8s_nfe32_good.wav`,
`refstudy_ref30s_nfe32_BROKEN_reference_too_long.wav`,
`refstudy_ref30s_package_clipping_nfe32_BROKEN_transcript_mismatch.wav`;
`limit_6sentences_in_one_call_nfe32_DEGRADED.wav`; `f5_then_rvc_onecall_30s_48k_synthetic_test_model.wav`.

### State at the end

- **Processes:** no Python or `typeperf` process started by this spike is running. Three were stopped by hand
  during the work: two CPU benchmarks that ran too long, and `typeperf` samplers orphaned when an RVC test crashed
  on the 37 s limit. Other Python processes on the machine (a pytest run of `tests/test_worker_real.py` and an RVC
  training it started) belong to another session and were left alone.
- **Outside the spike folder:**
  - `C:\Users\koda\.miopen`: does not exist.
  - `%TEMP%\miopen-lockfiles\memory_gfx1100.lock`: one zero-byte file dated 08:34, before this spike began (09:09);
    not created here and left in place. The folder's own timestamp changed at 10:07 during the spike, which could
    be the other session's training or a metadata query run in this spike's venv without `TEMP` redirected; no
    file was added.
  - `C:\Users\koda\.cache\huggingface`: unchanged (still only `grounding-dino-tiny` and `faster-whisper-tiny.en`,
    nothing newer than the start of the spike). No `~/.cache/torch`, `~/.cache/wandb`, `~/wandb`, `~/nltk_data`.
  - Other new files in `%TEMP%` during the period (two WebP images, Visual Studio updater logs, `pytest-of-koda`)
    are not from this spike.
  - `spike/stage3/cache/uv` was used as the uv cache for the two AMD install commands (and for the removed test
    venv), as instructed; uv may have added index metadata there. It is 13 GB now, against 6.5 GB in the stage 3
    notes; the trainer set-up by other sessions also uses it, and I did not measure it before starting.
- **Project files:** nothing outside `spike/stage4/` was written by this spike. `backend/.venv` was used read-only
  with `PYTHONDONTWRITEBYTECODE=1`; `trainer/.venv` was only read by a uv dry run. `git status` shows modified
  tracked files under `app/`, `backend/`, `tools/` and `README.md`; those are another session's work in progress,
  not this spike's.
