"""Guards of the text-to-speech environment: the download manifest, the lock file and tts/verify.py (probes are injected)."""
import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VERIFY = REPO / "tts" / "verify.py"
MANIFEST = REPO / "tts" / "pretrained.json"
F5 = "https://huggingface.co/SWivid/F5-TTS/resolve/84e5a410d9cead4de2f847e7c9369a6440bdfaca/"
VOCOS = "https://huggingface.co/charactr/vocos-mel-24khz/resolve/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/"

# (path under tts/models, bytes, sha256): the download table in section 2 of docs/superpowers/notes/2026-10-13-stage-4-tts-spike.md
MODELS = {
    ("F5TTS_v1_Base/model_1250000.safetensors", 1348435761, "670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38"),
    ("F5TTS_v1_Base/vocab.txt", 13800, "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"),
    ("vocos-mel-24khz/pytorch_model.bin", 54365991, "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"),
    ("vocos-mel-24khz/config.yaml", 461, "da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc"),
}


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("tts_verify", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_matches_the_spike_notes():
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(entries) == 4
    for e in entries:
        assert set(e) == {"url", "path", "bytes", "sha256"}
        assert len(e["sha256"]) == 64 and set(e["sha256"]) <= set("0123456789abcdef")
        assert e["url"].endswith("/" + e["path"].split("/")[-1])
        assert e["url"].startswith(F5 if e["path"].startswith("F5TTS_v1_Base/") else VOCOS)
    assert {(e["path"], e["bytes"], e["sha256"]) for e in entries} == MODELS


def test_verify_uses_only_the_standard_library():
    tree = ast.parse(VERIFY.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add(node.module.split(".")[0])
    assert names, "no imports found"
    assert names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)


# ---- a healthy fake installation under tmp_path, and probes that answer like a healthy machine ----

@pytest.fixture
def root(tmp_path):
    t = tmp_path / "tts"
    (t / ".venv" / "Scripts").mkdir(parents=True)
    (t / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    entries = []
    for i, rel in enumerate(["F5TTS_v1_Base/a.safetensors", "vocos-mel-24khz/b.bin"]):
        data = b"x" * (10 + i)
        f = t / "models" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        entries.append({"url": F5 + rel, "path": rel, "bytes": len(data), "sha256": "0" * 64})
    (t / "pretrained.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


class Probes:
    def __init__(self):
        self.torch_out = "2.9.1+rocm7.2.1 True\n"
        self.f5_out = "1.1.22\n"
        self.f5_code = 0
        self.calls = []

    def run(self, args, **kwargs):
        self.calls.append(list(args))
        assert isinstance(args, list)
        if "importlib.metadata" in args[-1]:
            err = "PackageNotFoundError: f5-tts" if self.f5_code else ""
            return subprocess.CompletedProcess(args, self.f5_code, self.f5_out, err)
        return subprocess.CompletedProcess(args, 0, self.torch_out, "")


def run_checks(verify, root, probes, **kw):
    return verify.run_checks(root, run=probes.run, **kw)


def by_id(result):
    return {c["id"]: c for c in result["checks"]}


def test_healthy_installation_is_ok(verify, root):
    result = run_checks(verify, root, Probes())
    assert result["ok"] is True
    assert [c["id"] for c in result["checks"]] == ["venv", "torch_gpu", "f5_package", "models"]
    assert all(c["ok"] for c in result["checks"])


def test_fast_skips_torch_gpu_and_does_not_import_torch(verify, root):
    probes = Probes()
    result = run_checks(verify, root, probes, fast=True)
    assert result["ok"] is True
    assert "torch_gpu" not in by_id(result)
    assert not any("import torch" in c[-1] for c in probes.calls)


def test_missing_venv_names_setup(verify, root):
    (root / "tts" / ".venv" / "Scripts" / "python.exe").unlink()
    checks = by_id(run_checks(verify, root, Probes()))
    assert not checks["venv"]["ok"] and "tts/setup.sh" in checks["venv"]["message"]
    assert not checks["torch_gpu"]["ok"] and not checks["f5_package"]["ok"]  # nothing to run
    assert checks["models"]["ok"]


def test_wrong_model_size_names_the_file(verify, root):
    (root / "tts" / "models" / "vocos-mel-24khz" / "b.bin").write_bytes(b"short")
    c = by_id(run_checks(verify, root, Probes()))["models"]
    assert not c["ok"] and "vocos-mel-24khz/b.bin" in c["message"] and "tts/setup.sh" in c["message"]


def test_missing_model_file_names_the_file(verify, root):
    (root / "tts" / "models" / "F5TTS_v1_Base" / "a.safetensors").unlink()
    c = by_id(run_checks(verify, root, Probes()))["models"]
    assert not c["ok"] and "F5TTS_v1_Base/a.safetensors" in c["message"] and "tts/setup.sh" in c["message"]
    assert "vocos-mel-24khz/b.bin" not in c["message"]


def test_unreadable_manifest_is_a_failed_check(verify, root):
    (root / "tts" / "pretrained.json").write_text("not json", encoding="utf-8")
    c = by_id(run_checks(verify, root, Probes()))["models"]
    assert not c["ok"] and "pretrained.json" in c["message"]


def test_cpu_torch_means_the_gpu_build_was_replaced(verify, root):
    probes = Probes()
    probes.torch_out = "2.9.1+cpu False\n"
    c = by_id(run_checks(verify, root, probes))["torch_gpu"]
    assert not c["ok"] and "GPU build was replaced" in c["message"] and "2.9.1+cpu" in c["message"]
    assert "tts/setup.sh" in c["message"]


def test_rocm_torch_without_a_visible_gpu_reports_it(verify, root):
    probes = Probes()
    probes.torch_out = "2.9.1+rocm7.2.1 False\n"
    c = by_id(run_checks(verify, root, probes))["torch_gpu"]
    assert not c["ok"] and "no GPU" in c["message"]


def test_python_that_crashes_is_a_failed_check(verify, root):
    probes = Probes()
    original = probes.run

    def crash(args, **kw):
        if "import torch" in args[-1]:
            return subprocess.CompletedProcess(args, 1, "", "ImportError: DLL load failed")
        return original(args, **kw)

    c = by_id(verify.run_checks(root, run=crash))["torch_gpu"]
    assert not c["ok"] and "DLL load failed" in c["message"]


def test_f5_package_missing_names_the_fix(verify, root):
    probes = Probes()
    probes.f5_code = 1
    c = by_id(run_checks(verify, root, probes))["f5_package"]
    assert not c["ok"] and "f5-tts" in c["message"] and "tts/setup.sh" in c["message"]


def test_f5_package_at_another_version_fails(verify, root):
    probes = Probes()
    probes.f5_out = "1.1.20\n"
    c = by_id(run_checks(verify, root, probes))["f5_package"]
    assert not c["ok"] and "1.1.20" in c["message"] and "1.1.22" in c["message"] and "tts/setup.sh" in c["message"]


def test_f5_package_check_asks_the_metadata_and_does_not_import_it(verify, root):
    probes = Probes()
    run_checks(verify, root, probes)
    f5 = [c for c in probes.calls if "importlib.metadata" in c[-1]]
    assert len(f5) == 1 and "import f5_tts" not in f5[0][-1] and "f5-tts" in f5[0][-1]


def test_main_prints_one_json_object_and_exit_code(verify, root, capsys):
    code = verify.main(["--fast", "--root", str(root)])  # the fake venv python is an empty file, so it cannot run
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False
    assert by_id(out)["venv"]["ok"] and not by_id(out)["f5_package"]["ok"]


# ---- the lock file: exact versions of what requirements.txt pulls in, and nothing of AMD's ----

LOCK = REPO / "tts" / "requirements.lock"
REQUIREMENTS = REPO / "tts" / "requirements.txt"


def _entries(path):
    return [line.split("#")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.split("#")[0].strip()]


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_the_lock_file_pins_one_exact_version_per_package():
    pinned = {}
    for entry in _entries(LOCK):
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)", entry)
        assert match, f"not an exact pin: {entry!r}"
        assert _canonical(match.group(1)) not in pinned, f"pinned twice: {entry!r}"
        pinned[_canonical(match.group(1))] = match.group(2)
    assert pinned
    # AMD's wheels are installed by setup.sh from their URLs with SHA-256; a pin here could swap in a CPU build.
    assert not [name for name in pinned if name in ("torch", "torchaudio", "torchvision") or "rocm" in name]
    assert not [version for version in pinned.values() if "rocm" in version]
    assert "f5-tts" not in pinned  # installed separately with --no-deps
    wanted = {_canonical(re.match(r"[A-Za-z0-9._-]+", entry).group(0)) for entry in _entries(REQUIREMENTS)}
    assert wanted and wanted <= set(pinned), wanted - set(pinned)
    assert int(pinned["numpy"].split(".")[0]) >= 2


def test_setup_pins_f5_and_the_rocm_torch_and_prints_the_licence_notice():
    script = (REPO / "tts" / "setup.sh").read_text(encoding="utf-8")
    assert "f5-tts==1.1.22" in script and "--no-deps" in script
    assert "-c " in script and "torch==2.9.1+rocm7.2.1" in script
    assert script.count("sha256=") >= 7
    assert "--no-python-downloads" in script
    assert "CC-BY-NC-4.0" in script and "non-commercial" in script
