#!/usr/bin/env bash
# Builds (or repairs) the native text-to-speech environment under tts/ on Windows (Git Bash).
#
#   bash tts/setup.sh
#
# Creates tts/.venv (Python 3.12 + AMD ROCm 7.2.1 PyTorch + F5-TTS 1.1.22), the model files listed in
# tts/pretrained.json (SHA-256 verified) under tts/models, and tts/.cache. Safe to re-run: every step is skipped
# when its result is already correct. The only files it removes are its own unfinished *.part downloads.
# It touches nothing outside tts/ (not trainer/, not backend/.venv, no system settings or drivers).
# Recipe and reasoning: docs/superpowers/notes/2026-10-13-stage-4-tts-spike.md, sections 2 and 3.
set -euo pipefail

T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"   # tts/ as D:/Projects/AI Voice/tts (works from any cwd)
cd "$T"
PY="$T/.venv/Scripts/python.exe"
MODELS="$T/models"
ROCM_BASE=https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1
START=$SECONDS

say() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv is not installed (https://docs.astral.sh/uv/)."
command -v curl >/dev/null || die "curl is not installed."
command -v sha256sum >/dev/null || die "sha256sum is not available."

# ---- caches live inside tts/.cache (nothing is written under the user profile or C:) ----
# The AMD wheels are about 2 GB; an existing uv cache that already holds them is reused for this run (nothing is
# moved or deleted): tts/.cache/uv, else the trainer's, else the stage 3 spike's.
UV_CACHE_DIR=""
for candidate in "$T/.cache/uv" "$T/../trainer/.cache/uv" "$T/../spike/stage3/cache/uv"; do
  if [ -d "$candidate" ]; then
    UV_CACHE_DIR="$(cd "$candidate" && pwd -W)"
    break
  fi
done
if [ -z "$UV_CACHE_DIR" ]; then
  mkdir -p "$T/.cache/uv"
  UV_CACHE_DIR="$T/.cache/uv"
elif [ "$UV_CACHE_DIR" != "$T/.cache/uv" ]; then
  say "tts/.cache/uv does not exist; using an existing uv cache for this run: $UV_CACHE_DIR (avoids re-downloading ~2 GB)"
fi
mkdir -p "$T/.cache/pip" "$T/.cache/hf" "$T/.cache/torch" "$T/.cache/xdg" "$T/.cache/miopen/db" "$T/.cache/numba" "$T/.cache/mpl" "$T/.cache/tmp"
export UV_CACHE_DIR
export PIP_CACHE_DIR="$T/.cache/pip" HF_HOME="$T/.cache/hf" TORCH_HOME="$T/.cache/torch" XDG_CACHE_HOME="$T/.cache/xdg"
export MIOPEN_USER_DB_PATH="$T/.cache/miopen/db" MIOPEN_CUSTOM_CACHE_DIR="$T/.cache/miopen" MIOPEN_LOG_LEVEL=3
export NUMBA_CACHE_DIR="$T/.cache/numba" MPLCONFIGDIR="$T/.cache/mpl" TEMP="$T/.cache/tmp" TMP="$T/.cache/tmp"
export ROCM_SDK_TARGET_FAMILY=custom   # the ROCm SDK helper breaks on the space in the project path; harmless otherwise

# ---- 1. venv ----
if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  say "venv: already present"
else
  say "venv: creating tts/.venv (Python 3.12)"
  uv venv --python 3.12 --no-python-downloads "$T/.venv"
fi

# ---- 2. ROCm SDK + PyTorch (AMD wheels), skipped when torch 2.9.1+rocm7.2.1 is already installed ----
installed_torch="$("$PY" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null | tr -d '\r' || true)"
if [ "$installed_torch" = "2.9.1+rocm7.2.1" ]; then
  say "torch: $installed_torch already installed"
else
  say "torch: installing AMD ROCm 7.2.1 SDK and PyTorch wheels (large; first run only)"
  uv pip install --python "$PY" \
    "$ROCM_BASE/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl#sha256=f68989d48df71cbfc3cb68bf705dc37c0f56e9666feddb59a1a0f5ff7539fe1c" \
    "$ROCM_BASE/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl#sha256=19e6ee67e13432b7c1e8a4077df795dcb1239546ae314700c4b4d97e5b4b8f63" \
    "$ROCM_BASE/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl#sha256=c7fe0b0731af8896093ff69e11496830d3cb6a4aed73e895c60b7cbdc200be92" \
    "$ROCM_BASE/rocm-7.2.1.tar.gz#sha256=9084902eaa69213a00a90784ad89e6e5fe73c702df0cc6cc3a70d777c7a6142b"
  uv pip install --python "$PY" \
    "$ROCM_BASE/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl#sha256=e88bf270163b48f7f27f7ea3db5ffb3be4ba107301933022bcb3c6ddedfeeabb" \
    "$ROCM_BASE/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl#sha256=62835605995c7812a3224733abf08e222afddb7a487c493521e613ca46e578ab" \
    "$ROCM_BASE/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl#sha256=b5c34d76a2f01fb97e1e0f224bd97fc3a58ec1ee1ef2376af2d72086e9aca34c"
fi

# ---- 3. Python packages and F5-TTS, always under a constraint that pins the three ROCm torch packages ----
# Without it a package asking for "torch" could replace the GPU build with a CPU one from PyPI.
# Both the constraint file and the requirement files are named relative to tts/ (the cwd, see the cd above): uv
# splits an absolute -c path at the space in the project path, and the same care is taken with -r.
CONSTRAINTS=".cache/constraints.txt"
printf 'torch==2.9.1+rocm7.2.1\ntorchaudio==2.9.1+rocm7.2.1\ntorchvision==0.24.1+rocm7.2.1\n' > "$CONSTRAINTS"
if [ -f requirements.lock ]; then
  say "packages: tts/requirements.lock (exact versions)"
  uv pip install --python "$PY" -c "$CONSTRAINTS" -r requirements.lock
else
  say "packages: NOTICE: tts/requirements.lock is missing, so the newest versions allowed by tts/requirements.txt are installed instead; they may differ from the versions that were tested"
  uv pip install --python "$PY" -c "$CONSTRAINTS" -r requirements.txt
fi
# F5-TTS without its declared dependencies (the lock holds the inference-only subset that is needed).
say "f5-tts: 1.1.22 (--no-deps)"
uv pip install --python "$PY" -c "$CONSTRAINTS" --no-deps "f5-tts==1.1.22"

# ---- 4. model files: download to .part, verify size and SHA-256, then rename ----
say "NOTICE: the F5-TTS weights (F5TTS_v1_Base) are licensed CC-BY-NC-4.0: personal, non-commercial use only, with attribution. They may not be sold or used for paid work. Code licences (F5-TTS, Vocos, and the Vocos weights) are MIT."
mkdir -p "$MODELS"
manifest="$("$PY" -I -c '
import json, sys
for e in json.load(open(sys.argv[1], encoding="utf-8")):
    print(e["url"], e["path"], e["bytes"], e["sha256"], sep="\t")
' "$T/pretrained.json" | tr -d '\r')"

while IFS=$'\t' read -r url rel bytes sha; do
  dest="$MODELS/$rel"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$bytes" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    say "model: $rel ok"
    continue
  fi
  say "model: downloading $rel ($bytes bytes)"
  rm -f "$dest.part"
  curl -sSL --fail --retry 3 --retry-delay 2 -o "$dest.part" "$url" || { rm -f "$dest.part"; die "download of $url failed."; }
  got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  if [ "$got" != "$sha" ] || [ "$(stat -c %s "$dest.part")" != "$bytes" ]; then
    rm -f "$dest.part"
    die "SHA-256/size mismatch for $rel (expected $sha, got $got). The file was NOT installed."
  fi
  mv -f "$dest.part" "$dest"
done <<< "$manifest"

# ---- 5. verify ----
say "verify"
"$PY" "$T/verify.py" || die "verification failed (see the JSON above)."
say "done in $((SECONDS - START)) s"
