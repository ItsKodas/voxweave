"""Guards of tools/export_rvc_onnx.py. The export itself needs torch and a .pth, so it is not run here."""
import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "export_rvc_onnx.py"


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("export_rvc_onnx", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # must have no side effects: no guard, no torch import
    return module


def test_help_works_without_the_rvc_clone_or_torch(tool, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tool, "RVC_REPO", tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", ["export_rvc_onnx.py", "--help"])
    with pytest.raises(SystemExit) as info:
        tool.main()
    assert info.value.code == 0
    assert "voice_id" in capsys.readouterr().out


def test_missing_clone_is_reported_after_argument_parsing(tool, monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "RVC_REPO", tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", ["export_rvc_onnx.py", "voice.pth", "my-voice"])
    with pytest.raises(SystemExit) as info:
        tool.main()
    assert "RVC clone missing" in str(info.value.code)


def test_missing_git_is_a_clear_message(tool, monkeypatch, tmp_path):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(tool, "RVC_REPO", tmp_path)
    monkeypatch.setattr(tool.subprocess, "run", no_git)
    with pytest.raises(SystemExit) as info:
        tool._require_patched_repo()
    assert "git was not found" in str(info.value.code)


def test_unpatched_clone_is_refused(tool, monkeypatch, tmp_path):
    class Done:
        returncode = 1

    monkeypatch.setattr(tool, "RVC_REPO", tmp_path)
    monkeypatch.setattr(tool.subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(SystemExit) as info:
        tool._require_patched_repo()
    assert "not patched" in str(info.value.code)


@pytest.mark.parametrize("version", ["2.6.0", "2.6.0+cpu", "2.10.1", "3.0.0", "2.7.0.dev20250101"])
def test_safe_torch_versions_are_accepted(tool, version):
    tool._require_safe_torch(version)


@pytest.mark.parametrize("version", ["2.5.1", "2.5.1+cpu", "1.13.1", "2.0.0", "unknown"])
def test_old_or_unreadable_torch_versions_are_refused(tool, version):
    with pytest.raises(SystemExit) as info:
        tool._require_safe_torch(version)
    message = str(info.value.code)
    assert "2.6" in message and "unsafe" in message


def test_output_goes_to_the_pretrained_folder_by_default(tool, monkeypatch):
    monkeypatch.delenv("AIVOICE_EXPORT_DIR", raising=False)
    assert tool._output_dir("my-voice") == tool.ROOT / "data" / "pretrained" / "rvc" / "my-voice"


def test_export_dir_variable_redirects_the_output(tool, monkeypatch, tmp_path):
    target = tmp_path / "voices" / "abc" / "models" / "rvc.new"
    monkeypatch.setenv("AIVOICE_EXPORT_DIR", str(target))
    assert tool._output_dir("my-voice") == target


def test_an_empty_export_dir_variable_is_ignored(tool, monkeypatch):
    monkeypatch.setenv("AIVOICE_EXPORT_DIR", "")
    assert tool._output_dir("my-voice") == tool.ROOT / "data" / "pretrained" / "rvc" / "my-voice"


def test_the_guards_still_run_when_the_export_dir_is_set(tool, monkeypatch, tmp_path):
    monkeypatch.setenv("AIVOICE_EXPORT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(tool, "RVC_REPO", tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", ["export_rvc_onnx.py", "voice.pth", "my-voice"])
    with pytest.raises(SystemExit) as info:
        tool.main()
    assert "RVC clone missing" in str(info.value.code)
    assert not (tmp_path / "out").exists()
