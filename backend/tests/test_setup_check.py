from aivoice.devices import Device
from aivoice.setup_check import run_checks
from tests.test_pretrained import add_voice

MIC = Device(1, "Headset Mic", 1, 0)
VM = Device(3, "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", 0, 8)


def by_id(checks):
    return {c.id: c for c in checks}


def test_everything_missing(tmp_path):
    checks = by_id(run_checks(tmp_path, [MIC], ["CPUExecutionProvider"]))
    assert set(checks) == {"directml", "virtual_mic", "base_model", "voice"}
    assert not any(c.ok for c in checks.values())
    assert "VoiceMeeter" in checks["virtual_mic"].message
    assert "vec-768-layer-12.onnx" in checks["base_model"].message


def test_everything_present(tmp_path):
    add_voice(tmp_path, "demo")
    (tmp_path / "pretrained" / "vec-768-layer-12.onnx").write_bytes(b"x")
    checks = run_checks(tmp_path, [MIC, VM], ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert all(c.ok for c in checks)
