"""RVC filelist.txt and config.json writer (trainer/filelist.py), a port of the logic in RVC's webui.py."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "trainer" / "filelist.py"


@pytest.fixture(scope="module")
def filelist():
    spec = importlib.util.spec_from_file_location("rvc_filelist", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def read_lines(root):
    return (root / "logs" / "voice-a" / "filelist.txt").read_text(encoding="utf8").split("\n")


@pytest.fixture
def root(tmp_path):
    """A fake RVC root whose path contains a space; segment 1_0 lacks its feature file."""
    rvc = tmp_path / "RVC root"
    exp = rvc / "logs" / "voice-a"
    for seg in ("0_0", "0_1", "1_0"):
        touch(exp / "0_gt_wavs" / f"{seg}.wav")
        touch(exp / "2a_f0" / f"{seg}.wav.npy")
        touch(exp / "2b-f0nsf" / f"{seg}.wav.npy")
        if seg != "1_0":
            touch(exp / "3_feature768" / f"{seg}.npy")
    touch(exp / "0_gt_wavs" / "9_9.wav")  # present in one folder only
    (rvc / "configs" / "v1").mkdir(parents=True)
    (rvc / "configs" / "v1" / "40k.json").write_text(
        json.dumps({"train": {"log_interval": 200, "epochs": 20000}, "speaker_info": {"a": 1}, "data": {"sr": 40000}}),
        encoding="utf8",
    )
    (rvc / "configs" / "v2").mkdir()
    (rvc / "configs" / "v2" / "48k.json").write_text(json.dumps({"train": {"log_interval": 200}, "tag": "v2-48k"}))
    return rvc


def identity(items):
    return None


def test_writes_real_and_mute_lines(filelist, root):
    count = filelist.write_filelist(root, "voice-a", shuffle=identity)
    lines = read_lines(root)
    assert count == 4 == len(lines)
    assert [line.split("|")[0].rsplit("/", 1)[1] for line in lines] == ["0_0.wav", "0_1.wav", "mute40k.wav", "mute40k.wav"]
    assert not any("1_0" in line or "9_9" in line for line in lines)


def test_exact_line_text(filelist, root):
    filelist.write_filelist(root, "voice-a", shuffle=identity)
    lines = read_lines(root)
    base = str(root)
    assert "\\" in base and " " in base  # a Windows tmp path with a space
    d = base.replace("\\", "\\\\")
    # The folder part of a real line is backslash-doubled; the mute lines use the root as is.
    expected_real = (
        f"{d}/logs/voice-a/0_gt_wavs/0_0.wav|{d}/logs/voice-a/3_feature768/0_0.npy|"
        f"{d}/logs/voice-a/2a_f0/0_0.wav.npy|{d}/logs/voice-a/2b-f0nsf/0_0.wav.npy|0"
    )
    expected_mute = (
        f"{base}/logs/mute/0_gt_wavs/mute40k.wav|{base}/logs/mute/3_feature768/mute.npy|"
        f"{base}/logs/mute/2a_f0/mute.wav.npy|{base}/logs/mute/2b-f0nsf/mute.wav.npy|0"
    )
    assert lines[0] == expected_real
    assert lines[2] == lines[3] == expected_mute


def test_speaker_id_is_written(filelist, root):
    filelist.write_filelist(root, "voice-a", speaker_id=3, shuffle=identity)
    assert all(line.endswith("|3") for line in read_lines(root))


def test_default_shuffle_keeps_all_lines(filelist, root):
    assert filelist.write_filelist(root, "voice-a") == 4
    lines = read_lines(root)
    assert len(lines) == 4 and len(set(lines)) == 3


def test_shuffle_is_injectable(filelist, root):
    filelist.write_filelist(root, "voice-a", shuffle=lambda items: items.reverse())
    assert read_lines(root)[-1].split("|")[0].endswith("0_0.wav")


def test_config_copied_from_v1_40k_without_speaker_info(filelist, root):
    filelist.write_filelist(root, "voice-a", shuffle=identity)
    cfg = json.loads((root / "logs" / "voice-a" / "config.json").read_text(encoding="utf8"))
    assert "speaker_info" not in cfg
    assert cfg["train"] == {"log_interval": 200, "epochs": 20000}
    assert cfg["data"] == {"sr": 40000}


def test_config_text_format(filelist, root):
    filelist.write_filelist(root, "voice-a", log_interval=7, shuffle=identity)
    text = (root / "logs" / "voice-a" / "config.json").read_text(encoding="utf8")
    assert text == (
        '{\n    "data": {\n        "sr": 40000\n    },\n    "train": {\n        "epochs": 20000,\n'
        '        "log_interval": 7\n    }\n}\n'
    )


def test_existing_config_is_kept_and_only_log_interval_changes(filelist, root):
    path = root / "logs" / "voice-a" / "config.json"
    path.write_text(json.dumps({"train": {"log_interval": 200, "mine": True}, "speaker_info": {}, "extra": 1}))
    filelist.write_filelist(root, "voice-a", log_interval=9, shuffle=identity)
    assert json.loads(path.read_text(encoding="utf8")) == {"train": {"log_interval": 9, "mine": True}, "extra": 1}


def test_existing_config_without_log_interval_argument_keeps_its_value(filelist, root):
    path = root / "logs" / "voice-a" / "config.json"
    path.write_text(json.dumps({"train": {"log_interval": 200, "mine": True}}))
    filelist.write_filelist(root, "voice-a", shuffle=identity)
    assert json.loads(path.read_text(encoding="utf8")) == {"train": {"log_interval": 200, "mine": True}}


def test_other_rates_use_the_v2_config_and_mute_file(filelist, root):
    filelist.write_filelist(root, "voice-a", sample_rate="48k", shuffle=identity)
    cfg = json.loads((root / "logs" / "voice-a" / "config.json").read_text(encoding="utf8"))
    assert cfg["tag"] == "v2-48k"
    assert read_lines(root)[-1].split("|")[0].endswith("mute48k.wav")


def test_nothing_matching_raises_value_error(filelist, root):
    for seg in ("0_0", "0_1"):
        (root / "logs" / "voice-a" / "3_feature768" / f"{seg}.npy").unlink()
    with pytest.raises(ValueError):
        filelist.write_filelist(root, "voice-a", shuffle=identity)
    assert not (root / "logs" / "voice-a" / "filelist.txt").exists()


def test_missing_experiment_folders_raise_value_error(filelist, root):
    with pytest.raises(ValueError):
        filelist.write_filelist(root, "no-such-voice", shuffle=identity)
