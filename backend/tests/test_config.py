import json

import pytest

from aivoice import config


def test_sample_rate():
    assert config.SAMPLE_RATE == 48000


def test_data_dir_uses_env_and_creates(tmp_path, monkeypatch):
    target = tmp_path / "d"
    monkeypatch.setenv("AIVOICE_DATA", str(target))
    assert config.data_dir() == target
    assert target.is_dir()


@pytest.mark.parametrize("value, expected", [(None, False), ("", False), ("0", False), ("1", True)])
def test_installed_app_follows_the_variable_the_shell_sets(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AIVOICE_INSTALLED", raising=False)
    else:
        monkeypatch.setenv("AIVOICE_INSTALLED", value)
    assert config.installed_app() is expected


def test_settings_default(tmp_path):
    assert config.load_settings(tmp_path) == {"dml_device_id": 0}


def test_settings_read(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"dml_device_id": 1}))
    assert config.load_settings(tmp_path)["dml_device_id"] == 1


def test_settings_corrupt_falls_back(tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert config.load_settings(tmp_path) == {"dml_device_id": 0}


def test_whisper_model_default(tmp_path):
    assert config.whisper_model(tmp_path) == "small.en"


def test_whisper_model_from_settings(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"whisper_model": "base.en"}))
    assert config.whisper_model(tmp_path) == "base.en"


@pytest.mark.parametrize("bad", [5, "", "../evil", "a b", None, "x" * 80])
def test_whisper_model_rejects_odd_values(tmp_path, bad):
    (tmp_path / "settings.json").write_text(json.dumps({"whisper_model": bad}))
    with pytest.raises(ValueError, match="whisper_model"):
        config.whisper_model(tmp_path)
