import json

from aivoice.pretrained import list_voices, vec_model_path


def add_voice(data, voice_id, meta='{"name": "Demo", "sample_rate": 40000}', index=False, model=True):
    folder = data / "pretrained" / "rvc" / voice_id
    folder.mkdir(parents=True)
    if model:
        (folder / "model.onnx").write_bytes(b"x")
    if meta is not None:
        (folder / "voice.json").write_text(meta)
    if index:
        (folder / "model.index").write_bytes(b"x")
    return folder


def test_no_folder_means_no_voices(tmp_path):
    assert list_voices(tmp_path) == []


def test_lists_voice_with_and_without_index(tmp_path):
    add_voice(tmp_path, "a", index=True)
    add_voice(tmp_path, "b", meta=json.dumps({"name": "Bee", "sample_rate": 48000}))
    voices = list_voices(tmp_path)
    assert [(v.id, v.name, v.sample_rate) for v in voices] == [("a", "Demo", 40000), ("b", "Bee", 48000)]
    assert voices[0].index_path is not None and voices[1].index_path is None


def test_broken_voices_are_skipped_without_affecting_others(tmp_path):
    add_voice(tmp_path, "good")
    add_voice(tmp_path, "badjson", meta="{nope")
    add_voice(tmp_path, "nometa", meta=None)
    add_voice(tmp_path, "nomodel", model=False)
    add_voice(tmp_path, "nosr", meta='{"name": "x"}')
    assert [v.id for v in list_voices(tmp_path)] == ["good"]


def test_vec_model_path(tmp_path):
    assert vec_model_path(tmp_path) == tmp_path / "pretrained" / "vec-768-layer-12.onnx"
