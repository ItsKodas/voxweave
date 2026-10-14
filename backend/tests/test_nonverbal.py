import numpy as np
import pytest

from aivoice.nonverbal import classify, strip_annotations, usable_as_reference

SR = 48000


def voiced(seconds, hz=180.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6))).astype(np.float32)


def burst(seconds, amp=0.3):
    return (amp * np.random.default_rng(0).standard_normal(int(seconds * SR))).astype(np.float32)


@pytest.mark.parametrize("text, kind", [
    ("Hahahaha", "laugh"), ("Ha ha! ", "laugh"), ("hehe", "laugh"), ("(laughs)", "laugh"), ("[Laughter]", "laugh"),
    ("(coughs)", "cough"), ("[coughing]", "cough"), ("*sigh*", "breath"), ("(gasps)", "breath"),
    ("Mmmmmmm", "hum"), ("aaaah", "hum"), ("Um...", "filler"), ("mm-hmm", "filler"), ("Uh, um.", "filler"),
    ("This is a normal sentence.", "speech"), ("Have a nice day!", "speech"),
])
def test_what_whisper_wrote_says_what_kind_of_sound_it_was(text, kind):
    assert classify(text, voiced(1.0), SR)[0] == kind


def test_invented_words_are_judged_by_the_sound():
    # Whisper writes "Thank you." for many non-speech sounds: a short unvoiced burst is a cough, a long one a breath,
    # a voiced one a hum.
    assert classify("Thank you.", burst(0.6), SR) == ("cough", "[cough]")
    assert classify("", burst(2.5, amp=0.05), SR) == ("breath", "[breath]")
    assert classify("you", voiced(1.0), SR) == ("hum", "[hum]")


def test_speech_with_a_laugh_in_it_stays_speech_but_is_marked_so_text_to_speech_skips_it():
    kind, text = classify("That is so funny (laughs) really.", voiced(2.0), SR)
    assert kind == "speech" and text == "That is so funny really. [laugh]"
    assert not usable_as_reference(kind, text)
    assert usable_as_reference("speech", "A clean sentence.")
    assert not usable_as_reference("laugh", "[laugh] haha")


def test_annotations_are_taken_out_of_the_transcript():
    assert strip_annotations("Hello (laughs) there [coughing]") == ("Hello there", ["laughs", "coughing"])
    assert strip_annotations("No annotations here.") == ("No annotations here.", [])


def test_no_speech_likelihood_marks_short_wordless_pieces():
    assert classify("Okay then", burst(0.5), SR, no_speech=0.9)[0] == "cough"
    assert classify("Okay then, let us start the meeting now.", voiced(2.0), SR, no_speech=0.9)[0] == "speech"


def test_clips_keep_their_kind_and_text_to_speech_never_picks_a_non_speech_clip(tmp_path):
    from aivoice import tts_reference
    from aivoice.library import Library, LibraryError

    library = Library(tmp_path)
    voice = library.create("Kai")
    laugh = library.add_clip(voice.id, voiced(6.0), "[laugh] hahaha", "imported", kind="laugh")
    speech = library.add_clip(voice.id, voiced(6.0, hz=150), "A clear sentence of six seconds.", "imported")
    assert Library(tmp_path).get(voice.id).clips[0].kind == "laugh"  # kept on disk
    chosen = tts_reference.choose(library.get(voice.id))
    assert chosen.index == speech.index
    library.set_kind(voice.id, speech.index, "noise")
    with pytest.raises(ValueError):
        tts_reference.choose(library.get(voice.id))  # nothing left that is plain speech
    with pytest.raises(LibraryError, match="kind"):
        library.set_kind(voice.id, laugh.index, "music")
    with pytest.raises(LibraryError, match="kind"):
        library.add_clip(voice.id, voiced(1.0), "x", "imported", kind="singing")


def test_clips_saved_before_kinds_existed_load_as_speech(tmp_path):
    import json

    from aivoice.library import Library

    library = Library(tmp_path)
    voice = library.create("Old")
    library.add_clip(voice.id, voiced(1.0), "Hello.", "recorded")
    path = tmp_path / "voices" / voice.id / "voice.json"
    data = json.loads(path.read_text())
    for clip in data["clips"]:
        clip.pop("kind")
    path.write_text(json.dumps(data))
    assert Library(tmp_path).get(voice.id).clips[0].kind == "speech"


def test_labelling_existing_clips_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from tests.test_api_voicepack import ORIGIN, rig as voicepack_rig  # noqa: F401

    from aivoice.api import create_app
    from aivoice.devices import Device
    from aivoice.importer import Importer
    from aivoice.library import Library
    from aivoice.manager import LiveManager
    from aivoice.recorder import Recorder
    from aivoice.studio import Studio
    from tests.test_live import FakeStreams, Identity
    from tests.test_recorder import CapturingStreams

    devices = [Device(1, "Mic", 1, 0)]
    library = Library(tmp_path)
    voice = library.create("Kai")
    library.add_clip(voice.id, voiced(1.0), "Hahahaha", "imported")
    library.add_clip(voice.id, voiced(2.0), "A normal sentence here.", "imported")
    kept = library.add_clip(voice.id, voiced(1.0), "Hehe", "imported", kind="speech")
    library.set_kind(voice.id, kept.index, "filler")  # set by hand: never changed again
    studio = Studio(library, Recorder(CapturingStreams()), Importer(library, None), lambda: devices, prompts=("One.",))
    manager = LiveManager(tmp_path, lambda: devices, lambda v: Identity(), FakeStreams())
    client = TestClient(create_app(manager, tmp_path, lambda: devices, [], allowed_hosts=["testserver"],
                                   studio=studio, library=library, importer=None))
    res = client.post(f"/library/voices/{voice.id}/label-sounds", headers=ORIGIN).json()
    assert res == {"labelled": {"laugh": 1}, "clips": 3}
    kinds = [c["kind"] for c in client.get(f"/library/voices/{voice.id}/clips").json()]
    assert sorted(kinds) == ["filler", "laugh", "speech"]
    first = client.get(f"/library/voices/{voice.id}/clips").json()[-1]
    changed = client.put(f"/library/voices/{voice.id}/clips/{first['index']}/kind", json={"kind": "cough"},
                         headers=ORIGIN)
    assert changed.status_code == 200 and changed.json()["kind"] == "cough"
    bad = client.put(f"/library/voices/{voice.id}/clips/{first['index']}/kind", json={"kind": "x"}, headers=ORIGIN)
    assert bad.status_code == 400
