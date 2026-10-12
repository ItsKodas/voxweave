import json

from aivoice import config


def test_sample_rate():
    assert config.SAMPLE_RATE == 48000


def test_data_dir_uses_env_and_creates(tmp_path, monkeypatch):
    target = tmp_path / "d"
    monkeypatch.setenv("AIVOICE_DATA", str(target))
    assert config.data_dir() == target
    assert target.is_dir()


def test_settings_default(tmp_path):
    assert config.load_settings(tmp_path) == {"dml_device_id": 0}


def test_settings_read(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"dml_device_id": 1}))
    assert config.load_settings(tmp_path)["dml_device_id"] == 1


def test_settings_corrupt_falls_back(tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert config.load_settings(tmp_path) == {"dml_device_id": 0}
