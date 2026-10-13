import json
import zipfile

import numpy as np
import pytest

from aivoice import tts_reference
from aivoice.library import Library
from aivoice.pretrained import list_voices
from aivoice.voicepack import EXTENSION, VoicePackError, export_voice, import_voice
from tests.test_pretrained import add_voice

SR = 48000
ACCEPT = lambda path: True  # noqa: E731 - the real check loads the model with onnxruntime


def speech(seconds: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def trained_voice(tmp_path, name="Ada", index=True, clips=True):
    library = Library(tmp_path)
    voice = library.create(name)
    folder = tmp_path / "voices" / voice.id / "models" / "rvc"
    folder.mkdir(parents=True)
    (folder / "model.onnx").write_bytes(b"MODEL" * 1000)
    (folder / "voice.json").write_text(json.dumps({"name": name, "sample_rate": 40000}))
    if index:
        (folder / "model.index").write_bytes(b"INDEX" * 500)
    if clips:
        library.add_clip(voice.id, speech(6.0), "A clear sentence of six seconds.", "recorded")
        library.add_clip(voice.id, speech(2.0), "Too short.", "recorded")
    return library, voice


def choose(library):
    return lambda voice: tts_reference.choose(voice, tts_reference.library_levels(library, voice.id))


def test_a_trained_voice_goes_out_as_one_file_and_comes_back_as_a_new_voice(tmp_path):
    library, voice = trained_voice(tmp_path / "here")
    pack = export_voice(tmp_path / "here", library, voice.id, tmp_path / "out", choose_reference=choose(library),
                        now=lambda: 1234.0)
    assert pack.name == "ada" + EXTENSION and pack.parent == tmp_path / "out"
    with zipfile.ZipFile(pack) as z:
        assert sorted(z.namelist()) == ["manifest.json", "model.index", "model.onnx", "reference.txt", "reference.wav"]
        manifest = json.loads(z.read("manifest.json"))
    assert manifest == {"format": "voxweave-voice", "version": 1, "name": "Ada", "sample_rate": 40000,
                        "has_index": True, "created": 1234, "reference": {"seconds": 6.0}}

    there = Library(tmp_path / "there")
    imported = import_voice(tmp_path / "there", there, pack, is_model=ACCEPT)
    assert imported.name == "Ada" and imported.models == ["rvc"] and imported.rvc_sample_rate == 40000
    folder = tmp_path / "there" / "voices" / imported.id / "models" / "rvc"
    assert (folder / "model.onnx").read_bytes() == b"MODEL" * 1000
    assert (folder / "model.index").read_bytes() == b"INDEX" * 500
    assert not (folder.parent / "rvc.part").exists()
    # The reference clip came along, so text to speech has something to copy; the other recordings stayed home.
    assert [(c.text, c.source, c.origin) for c in imported.clips] == [
        ("A clear sentence of six seconds.", "imported", "voice pack")]
    # Live mode finds the voice the way it finds any trained voice.
    assert [v.id for v in list_voices(tmp_path / "there")] == [imported.id]


def test_the_reference_clip_is_left_out_when_not_wanted_or_when_no_clip_will_do(tmp_path):
    library, voice = trained_voice(tmp_path / "a")
    pack = export_voice(tmp_path / "a", library, voice.id, tmp_path / "out", include_reference=False,
                        choose_reference=choose(library))
    assert "reference.wav" not in zipfile.ZipFile(pack).namelist()
    library, voice = trained_voice(tmp_path / "b", clips=False, index=False)
    pack = export_voice(tmp_path / "b", library, voice.id, tmp_path / "out2", choose_reference=choose(library))
    assert sorted(zipfile.ZipFile(pack).namelist()) == ["manifest.json", "model.onnx"]
    imported = import_voice(tmp_path / "c", Library(tmp_path / "c"), pack, is_model=ACCEPT)
    assert imported.clips == [] and imported.models == ["rvc"]
    assert not (tmp_path / "c" / "voices" / imported.id / "models" / "rvc" / "model.index").exists()


def test_exporting_never_replaces_a_file_and_never_changes_the_voice(tmp_path):
    library, voice = trained_voice(tmp_path / "a")
    before = sorted((p.relative_to(tmp_path), p.stat().st_size) for p in (tmp_path / "a").rglob("*") if p.is_file())
    names = [export_voice(tmp_path / "a", library, voice.id, tmp_path / "out").name for _ in range(3)]
    assert names == ["ada.voxweave", "ada-2.voxweave", "ada-3.voxweave"]
    assert not list((tmp_path / "out").glob("*.part"))
    assert before == sorted((p.relative_to(tmp_path), p.stat().st_size) for p in (tmp_path / "a").rglob("*") if p.is_file())


def test_an_exported_model_that_is_not_in_the_library_can_be_packed_too(tmp_path):
    add_voice(tmp_path, "demo", index=True)
    pack = export_voice(tmp_path, Library(tmp_path), "demo", tmp_path / "out")
    assert json.loads(zipfile.ZipFile(pack).read("manifest.json"))["name"] == "Demo"


def test_a_voice_without_a_model_cannot_be_exported(tmp_path):
    library = Library(tmp_path)
    voice = library.create("Empty")
    for voice_id in (voice.id, "nope", "../.."):
        with pytest.raises(VoicePackError, match="no trained model"):
            export_voice(tmp_path, library, voice_id, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def make_pack(path, manifest=None, members=None):
    manifest = {"format": "voxweave-voice", "version": 1, "name": "Ada", "sample_rate": 40000} if manifest is None else manifest
    with zipfile.ZipFile(path, "w") as z:
        if manifest is not False:
            z.writestr("manifest.json", manifest if isinstance(manifest, str) else json.dumps(manifest))
        for name, payload in (members if members is not None else {"model.onnx": b"m"}).items():
            z.writestr(name, payload)
    return path


def nothing_left(data):
    voices = data / "voices"
    return not voices.exists() or not [p for p in voices.iterdir() if p.is_dir()]


@pytest.mark.parametrize("manifest, message", [
    (False, "not a voice pack"),
    ("not json", "not a voice pack"),
    ([1, 2], "not a voice pack"),
    ({"format": "something-else", "version": 1, "name": "A", "sample_rate": 40000}, "not a voice pack"),
    ({"format": "voxweave-voice", "version": 2, "name": "A", "sample_rate": 40000}, "newer version"),
    ({"format": "voxweave-voice", "version": 1, "name": "", "sample_rate": 40000}, "no usable name"),
    ({"format": "voxweave-voice", "version": 1, "name": "a\nb", "sample_rate": 40000}, "no usable name"),
    ({"format": "voxweave-voice", "version": 1, "name": "x" * 500, "sample_rate": 40000}, "no usable name"),
    ({"format": "voxweave-voice", "version": 1, "name": "A", "sample_rate": "40000"}, "sample rate"),
    ({"format": "voxweave-voice", "version": 1, "name": "A", "sample_rate": 5}, "sample rate"),
    ({"format": "voxweave-voice", "version": 1, "name": "A", "sample_rate": True}, "sample rate"),
])
def test_a_pack_with_a_bad_manifest_is_refused_and_leaves_nothing(tmp_path, manifest, message):
    pack = make_pack(tmp_path / "p.voxweave", manifest)
    with pytest.raises(VoicePackError, match=message):
        import_voice(tmp_path / "data", Library(tmp_path / "data"), pack, is_model=ACCEPT)
    assert nothing_left(tmp_path / "data")


def test_files_that_are_not_packs_are_refused(tmp_path):
    data = tmp_path / "data"
    (tmp_path / "text.voxweave").write_text("hello")
    make_pack(tmp_path / "nomodel.voxweave", members={"other.bin": b"x"})
    for path in (tmp_path / "text.voxweave", tmp_path / "nomodel.voxweave", tmp_path / "missing.voxweave", tmp_path):
        with pytest.raises(VoicePackError):
            import_voice(data, Library(data), path, is_model=ACCEPT)
    assert nothing_left(data)


def test_a_model_the_runtime_cannot_load_is_refused_and_the_new_voice_is_taken_away_again(tmp_path):
    pack = make_pack(tmp_path / "p.voxweave")
    data = tmp_path / "data"
    with pytest.raises(VoicePackError, match="not a voice model"):
        import_voice(data, Library(data), pack)  # the real check: b"m" is not an ONNX graph
    assert Library(data).list() == []


def test_only_the_known_members_are_ever_read_whatever_else_the_archive_holds(tmp_path):
    # A hostile archive: names that point outside the folder, and files nobody asked for.
    pack = make_pack(tmp_path / "p.voxweave", members={
        "model.onnx": b"m", "../../evil.txt": b"x", "/abs.txt": b"x", "voice.json": b"{}", "models/rvc/model.onnx": b"other",
    })
    data = tmp_path / "data"
    voice = import_voice(data, Library(data), pack, is_model=ACCEPT)
    written = sorted(p.relative_to(data / "voices" / voice.id).as_posix()
                     for p in (data / "voices" / voice.id).rglob("*") if p.is_file())
    assert written == ["models/rvc/model.onnx", "models/rvc/voice.json", "voice.json"]
    assert (data / "voices" / voice.id / "models" / "rvc" / "model.onnx").read_bytes() == b"m"
    assert not (tmp_path / "evil.txt").exists() and not list(tmp_path.glob("**/evil.txt"))


def test_a_member_larger_than_its_limit_is_refused_even_when_the_archive_understates_it(tmp_path, monkeypatch):
    import aivoice.voicepack as voicepack

    monkeypatch.setitem(voicepack.LIMITS, "model.onnx", 1000)
    monkeypatch.setitem(voicepack.LIMITS, "model.index", 1000)
    data = tmp_path / "data"
    for members in ({"model.onnx": b"x" * 5000}, {"model.onnx": b"m", "model.index": b"x" * 5000}):
        pack = make_pack(tmp_path / "big.voxweave", members=members)
        with pytest.raises(VoicePackError, match="not a voice pack"):
            import_voice(data, Library(data), pack, is_model=ACCEPT)
        assert Library(data).list() == []


def test_a_reference_that_cannot_be_used_is_left_out_and_the_voice_still_imports(tmp_path):
    data = tmp_path / "data"
    for payload, text in ((b"not a wav", "words"), (b"", "words")):
        pack = make_pack(tmp_path / "p.voxweave", members={"model.onnx": b"m", "reference.wav": payload,
                                                             "reference.txt": text})
        voice = import_voice(data, Library(data), pack, is_model=ACCEPT)
        assert voice.clips == [] and voice.models == ["rvc"]


def test_two_imports_of_the_same_pack_make_two_voices(tmp_path):
    data = tmp_path / "data"
    pack = make_pack(tmp_path / "p.voxweave")
    first = import_voice(data, Library(data), pack, is_model=ACCEPT)
    second = import_voice(data, Library(data), pack, is_model=ACCEPT)
    assert first.id != second.id and len(Library(data).list()) == 2
