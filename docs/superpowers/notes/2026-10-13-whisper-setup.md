# Whisper setup (stage 2, task 1)

## Install commands

```bash
uv pip install --python backend/.venv -e "backend[dev,transcribe]"
uv pip install --python backend/.venv --no-deps faster-whisper
```

`faster-whisper` is installed with `--no-deps` because it depends on the plain `onnxruntime`
package. That package shares a module name with `onnxruntime-directml` and would overwrite it,
removing `DmlExecutionProvider` and disabling GPU live mode. The `transcribe` extra lists the
other dependencies explicitly. A `--dry-run` of the extra install showed no `onnxruntime` in the plan.

## ONNX Runtime providers

- Before: `1.24.4 ['DmlExecutionProvider', 'CPUExecutionProvider']`
- After:  `1.24.4 ['DmlExecutionProvider', 'CPUExecutionProvider']`

`uv pip list` shows `onnxruntime-directml 1.24.4` and no plain `onnxruntime`.

## Installed versions

| Package | Version |
|---|---|
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.2 |
| huggingface-hub | 2.2.0 |
| tokenizers | 0.23.3 |
| av | 19.0.1 |
| tqdm | 4.70.1 |

## Model

- `small.en`, int8 on CPU
- Folder: `data/pretrained/whisper/models--Systran--faster-whisper-small.en/snapshots/d1d751a5f8271d482d14ca55d9e2deeebbae577f/`
- Load (including download): 15.5 s
- Transcribe 5 s of low-level noise: 0.9 s, result `'You'`

## Stage 1 suite

`backend/.venv/Scripts/python -m pytest backend/tests -q`: 198 passed, none skipped.
