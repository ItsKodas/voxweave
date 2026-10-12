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
