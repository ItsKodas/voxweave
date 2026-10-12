import os
import sys
import types

import numpy as np
import pytest

from aivoice.transcribe import WhisperTranscriber, _load_whisper


class Segment:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([Segment(" Hello there. "), Segment("How are you?")]), object()


def make(name="small.en"):
    built = []
    model = FakeModel()

    def factory(model_name, root):
        built.append((model_name, root))
        return model

    return WhisperTranscriber(lambda: name, "root", model_factory=factory), model, built


def test_joins_segments_into_one_clean_line():
    transcriber, _, _ = make()
    assert transcriber.transcribe(np.zeros(16000, dtype=np.float32)) == "Hello there. How are you?"


def test_model_is_loaded_lazily_and_once():
    transcriber, _, built = make()
    assert built == []
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    assert built == [("small.en", "root")]


def test_english_models_force_english_and_others_detect():
    transcriber, model, _ = make("small.en")
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    assert model.calls[0][1]["language"] == "en"
    transcriber, model, _ = make("small")
    transcriber.transcribe(np.zeros(100, dtype=np.float32))
    assert model.calls[0][1]["language"] is None


def test_audio_is_passed_as_float32():
    transcriber, model, _ = make()
    transcriber.transcribe(np.zeros(100, dtype=np.float64))
    assert model.calls[0][0].dtype == np.float32


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel: records how it was built, loads nothing."""

    built: list[tuple[str, dict]] = []
    missing = False  # True: the model is not on disk yet, so a local-only load fails

    def __init__(self, name, **kwargs):
        FakeWhisperModel.built.append((name, kwargs))
        if FakeWhisperModel.missing and kwargs.get("local_files_only"):
            raise RuntimeError("the model is not in the local cache")


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeWhisperModel.built = []
    FakeWhisperModel.missing = False
    # A stand-in module, so the real package (and with it the hub) is never imported by these tests.
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    return FakeWhisperModel


def test_a_downloaded_model_is_loaded_without_any_network_request(fake_whisper):
    model = _load_whisper("small.en", "some/root")
    assert isinstance(model, fake_whisper)
    assert len(fake_whisper.built) == 1
    name, kwargs = fake_whisper.built[0]
    assert name == "small.en" and kwargs["local_files_only"] is True
    assert kwargs["download_root"] == "some/root" and kwargs["device"] == "cpu" and kwargs["compute_type"] == "int8"


def test_a_missing_model_is_downloaded_by_a_second_load(fake_whisper):
    fake_whisper.missing = True
    model = _load_whisper("small.en", "some/root")
    assert isinstance(model, fake_whisper)
    assert len(fake_whisper.built) == 2
    assert fake_whisper.built[0][1]["local_files_only"] is True
    name, kwargs = fake_whisper.built[1]
    assert name == "small.en" and "local_files_only" not in kwargs
    assert kwargs["download_root"] == "some/root" and kwargs["device"] == "cpu" and kwargs["compute_type"] == "int8"


def test_hub_telemetry_is_turned_off_before_the_model_package_is_imported(fake_whisper, monkeypatch):
    seen = []

    class Watching:
        def __init__(self, name, **kwargs):
            seen.append(os.environ.get("HF_HUB_DISABLE_TELEMETRY"))

    module = types.ModuleType("faster_whisper")
    module.__getattr__ = lambda attr: seen.append(os.environ.get("HF_HUB_DISABLE_TELEMETRY")) or Watching
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    _load_whisper("small.en", "root")
    assert seen and set(seen) == {"1"}  # already set when the package was reached, and when the model was built


def test_an_existing_telemetry_setting_is_left_alone(fake_whisper, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "true")
    _load_whisper("small.en", "root")
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "true"
