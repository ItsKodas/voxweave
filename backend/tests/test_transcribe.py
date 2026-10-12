import numpy as np

from aivoice.transcribe import WhisperTranscriber


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
