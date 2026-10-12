import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


def _load_whisper(name: str, root):
    # Must be set before the Hugging Face hub is first imported (faster_whisper imports it): it reads this once.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from faster_whisper import WhisperModel

    # CPU with 8-bit weights: CTranslate2 has no AMD GPU backend, and this is not time critical.
    options = {"device": "cpu", "compute_type": "int8", "download_root": str(root)}
    try:
        # A model that is already on disk loads with no network request at all.
        return WhisperModel(name, local_files_only=True, **options)
    except Exception as exc:  # noqa: BLE001 - not downloaded yet, or only partly: the hub raises several types
        log.warning("The speech model %s is not fully on disk (%s); downloading it once", name, type(exc).__name__)
    return WhisperModel(name, **options)


class WhisperTranscriber:
    """Speech to text for imported clips. The model loads on first use (and downloads once)."""

    def __init__(self, model_name: Callable[[], str], download_root: Path | str, model_factory=None):
        self._name = model_name
        self._root = download_root
        self._factory = model_factory or _load_whisper
        self._model = None
        self._language: str | None = None

    def transcribe(self, audio16k: np.ndarray) -> str:
        if self._model is None:
            name = self._name()
            self._model = self._factory(name, self._root)
            self._language = "en" if name.endswith(".en") else None
        segments, _ = self._model.transcribe(
            np.asarray(audio16k, dtype=np.float32), language=self._language, beam_size=5,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
