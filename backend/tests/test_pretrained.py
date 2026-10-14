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


def add_library_voice(data, name, created, model=True, sample_rate=40000, index=False):
    from aivoice.library import Library
    voice = Library(data).create(name)
    folder = data / "voices" / voice.id / "models" / "rvc"
    folder.mkdir(parents=True)
    if model:
        (folder / "model.onnx").write_bytes(b"x")
    (folder / "voice.json").write_text(json.dumps({"name": "ignored", "sample_rate": sample_rate}))
    if index:
        (folder / "model.index").write_bytes(b"i")
    return voice


def test_library_voices_with_a_model_follow_the_pretrained_ones(tmp_path):
    add_voice(tmp_path, "zeta")
    first = add_library_voice(tmp_path, "Mine", None, sample_rate=48000, index=True)
    add_library_voice(tmp_path, "Untrained", None, model=False)
    voices = list_voices(tmp_path)
    assert [v.id for v in voices] == ["zeta", first.id]
    mine = voices[1]
    assert (mine.name, mine.sample_rate) == ("Mine", 48000)
    assert mine.model_path == tmp_path / "voices" / first.id / "models" / "rvc" / "model.onnx"
    assert mine.index_path == mine.model_path.with_name("model.index")


def test_library_voices_are_listed_oldest_first(tmp_path, monkeypatch):
    import itertools
    from aivoice import library as library_module
    ticks = itertools.count(1_700_000_000_000_000_000, 1_000_000_000)
    monkeypatch.setattr(library_module, "_now_ns", lambda: next(ticks))
    a = add_library_voice(tmp_path, "A", None)
    b = add_library_voice(tmp_path, "B", None)
    assert [v.id for v in list_voices(tmp_path)] == [a.id, b.id]


def test_pretrained_wins_an_id_collision(tmp_path):
    voice = add_library_voice(tmp_path, "Mine", None)
    add_voice(tmp_path, voice.id)
    found = [v for v in list_voices(tmp_path) if v.id == voice.id]
    assert len(found) == 1 and found[0].name == "Demo"


def test_damaged_library_folders_never_raise(tmp_path):
    good = add_library_voice(tmp_path, "Good", None)
    (tmp_path / "voices" / "broken").mkdir(parents=True)
    (tmp_path / "voices" / "broken" / "voice.json").write_text("{nope")
    bad = add_library_voice(tmp_path, "Bad", None)
    (tmp_path / "voices" / bad.id / "models" / "rvc" / "voice.json").write_text("{nope")
    assert [v.id for v in list_voices(tmp_path)] == [good.id]


def test_versions_kept_at_earlier_epochs_are_listed_after_their_voice_and_share_its_index(tmp_path):
    import json

    from aivoice.library import Library
    from aivoice.pretrained import base_voice_id

    library = Library(tmp_path)
    voice = library.create("Poki")
    models = tmp_path / "voices" / voice.id / "models"
    for folder, with_index in (("rvc", True), ("rvc-e150", False), ("rvc-e100", False), ("rvc-ebad", False)):
        (models / folder).mkdir(parents=True)
        (models / folder / "model.onnx").write_bytes(b"x")
        (models / folder / "voice.json").write_text(json.dumps({"name": "Poki", "sample_rate": 40000}))
        if with_index:
            (models / folder / "model.index").write_bytes(b"i")
    voices = list_voices(tmp_path)
    assert [(v.id, v.name) for v in voices] == [
        (voice.id, "Poki"), (f"{voice.id}@e100", "Poki (epoch 100)"), (f"{voice.id}@e150", "Poki (epoch 150)")]
    assert all(v.index_path == models / "rvc" / "model.index" for v in voices)
    assert voices[1].model_path == models / "rvc-e100" / "model.onnx"
    assert base_voice_id(f"{voice.id}@e100") == voice.id and base_voice_id(voice.id) == voice.id
