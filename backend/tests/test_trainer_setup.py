"""Guards of the training environment: the download manifest and trainer/verify.py (probes are injected)."""
import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VERIFY = REPO / "trainer" / "verify.py"
MANIFEST = REPO / "trainer" / "pretrained.json"
PINNED = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/"

# (path, bytes, sha256): the table in section 2.7 of docs/superpowers/notes/2026-10-13-stage-3-training-spike.md
MODELS = {
    ("assets/pretrained_v2/f0G40k.pth", 73106273, "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb"),
    ("assets/pretrained_v2/f0D40k.pth", 142875703, "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09"),
    ("assets/hubert_base/config.json", 1492, "0346950779dfb7f9316fa74ed846e2b8a22a08eedfdc5387b73f327cb1a4a7cf"),
    ("assets/hubert_base/preprocessor_config.json", 225, "7c1976a680fb7acc757cd36fb08eef878fa36c70b4c9d2d595df9c608bbbbf0e"),
    ("assets/hubert_base/pytorch_model.bin", 189206711, "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5"),
    ("assets/rmvpe/rmvpe.pt", 181184272, "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"),
}
MUTE = ("../.cache/mute.zip", 312273, "ee948e85213e4ed2f2ba2f8dfcee810bfd0b63131d91450e920bbe1cbd0321d0")


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("trainer_verify", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_matches_the_spike_notes():
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(entries) == 7
    for e in entries:
        assert set(e) == {"url", "path", "bytes", "sha256"}
        assert len(e["sha256"]) == 64 and set(e["sha256"]) <= set("0123456789abcdef")
        assert e["url"].startswith(BASE)
    assert {(e["path"], e["bytes"], e["sha256"]) for e in entries if e["path"] != MUTE[0]} == MODELS
    mute = [e for e in entries if e["path"] == MUTE[0]]
    assert [(e["path"], e["bytes"], e["sha256"]) for e in mute] == [MUTE]


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

PATCHES = ["rvc_rocm_dilated_conv.patch", "rvc_preprocess_keep_all_tails.patch", "rvc_train_fp32_env.patch"]
# The four files every filelist line for silence names (trainer/filelist.py), relative to rvc/logs/mute.
MUTE_FILES = ("0_gt_wavs/mute40k.wav", "3_feature768/mute.npy", "2a_f0/mute.wav.npy", "2b-f0nsf/mute.wav.npy")


@pytest.fixture
def root(tmp_path):
    t = tmp_path / "trainer"
    (t / ".venv" / "Scripts").mkdir(parents=True)
    (t / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    for rel in MUTE_FILES:
        (t / "rvc" / "logs" / "mute" / rel).parent.mkdir(parents=True)
        (t / "rvc" / "logs" / "mute" / rel).write_bytes(b"x")
    (t / "patches").mkdir()
    for p in PATCHES:
        (t / "patches" / p).write_text("diff", encoding="utf-8")
    entries = []
    for i, rel in enumerate(["assets/pretrained_v2/a.pth", "assets/rmvpe/b.pt", "../.cache/mute.zip"]):
        data = b"x" * (10 + i)
        f = t / "rvc" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        entries.append({"url": BASE + "main/" + rel, "path": rel, "bytes": len(data), "sha256": "0" * 64})
    (t / "pretrained.json").write_text(json.dumps(entries), encoding="utf-8")
    (tmp_path / "tools" / ".rvc" / "venv" / "Scripts").mkdir(parents=True)
    (tmp_path / "tools" / ".rvc" / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    (tmp_path / "tools" / ".rvc" / "repo").mkdir()
    return tmp_path


class Probes:
    def __init__(self):
        self.head = PINNED
        self.python_out = "2.9.1+rocm7.2.1 True\n"
        self.rejected_patch = None
        self.missing = ()
        self.calls = []

    def run(self, args, **kwargs):
        self.calls.append(list(args))
        assert isinstance(args, list)
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, self.head + "\n", "")
        if "apply" in args:
            bad = self.rejected_patch is not None and str(args[-1]).endswith(self.rejected_patch)
            return subprocess.CompletedProcess(args, 1 if bad else 0, "", "error: patch failed" if bad else "")
        return subprocess.CompletedProcess(args, 0, self.python_out, "")

    def which(self, name):
        return None if name in self.missing else f"C:/bin/{name}.exe"


def run_checks(verify, root, probes, **kw):
    return verify.run_checks(root, run=probes.run, which=probes.which, **kw)


def by_id(result):
    return {c["id"]: c for c in result["checks"]}


def test_healthy_installation_is_ok(verify, root):
    result = run_checks(verify, root, Probes())
    assert result["ok"] is True
    assert [c["id"] for c in result["checks"]] == [
        "venv", "torch_gpu", "rvc_commit", "rvc_patches", "pretrained", "ffmpeg", "exporter",
    ]
    assert all(c["ok"] for c in result["checks"])


def test_fast_skips_torch_gpu_and_does_not_start_python(verify, root):
    probes = Probes()
    result = run_checks(verify, root, probes, fast=True)
    assert result["ok"] is True
    assert "torch_gpu" not in by_id(result)
    assert not any("-c" in c for c in probes.calls)


def test_missing_venv_names_setup(verify, root):
    (root / "trainer" / ".venv" / "Scripts" / "python.exe").unlink()
    result = run_checks(verify, root, Probes())
    checks = by_id(result)
    assert result["ok"] is False
    assert not checks["venv"]["ok"] and "trainer/setup.sh" in checks["venv"]["message"]
    assert not checks["torch_gpu"]["ok"]  # nothing to run


def test_wrong_pretrained_size_names_the_file(verify, root):
    (root / "trainer" / "rvc" / "assets" / "rmvpe" / "b.pt").write_bytes(b"short")
    c = by_id(run_checks(verify, root, Probes()))["pretrained"]
    assert not c["ok"] and "assets/rmvpe/b.pt" in c["message"] and "trainer/setup.sh" in c["message"]


def test_missing_pretrained_file_names_the_file(verify, root):
    (root / "trainer" / "rvc" / "assets" / "pretrained_v2" / "a.pth").unlink()
    c = by_id(run_checks(verify, root, Probes()))["pretrained"]
    assert not c["ok"] and "assets/pretrained_v2/a.pth" in c["message"]


@pytest.mark.parametrize("missing", MUTE_FILES)
def test_each_of_the_four_silence_files_the_filelist_names_is_required(verify, root, missing):
    (root / "trainer" / "rvc" / "logs" / "mute" / missing).unlink()
    c = by_id(run_checks(verify, root, Probes()))["pretrained"]
    assert not c["ok"] and f"logs/mute/{missing}" in c["message"] and "trainer/setup.sh" in c["message"]
    others = [rel for rel in MUTE_FILES if rel != missing]
    assert not any(f"logs/mute/{rel}" in c["message"] for rel in others)


def test_the_silence_files_checked_are_the_ones_the_filelist_writes(verify, tmp_path):
    spec = importlib.util.spec_from_file_location("trainer_filelist", REPO / "trainer" / "filelist.py")
    filelist = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(filelist)
    rvc = tmp_path / "rvc"
    exp = rvc / "logs" / "voice-x"
    for folder, name in (("0_gt_wavs", "0_0.wav"), ("3_feature768", "0_0.npy"), ("2a_f0", "0_0.wav.npy"),
                         ("2b-f0nsf", "0_0.wav.npy")):
        (exp / folder).mkdir(parents=True)
        (exp / folder / name).write_bytes(b"x")
    (rvc / "configs" / "v1").mkdir(parents=True)
    (rvc / "configs" / "v1" / "40k.json").write_text(json.dumps({"train": {}}), encoding="utf-8")
    filelist.write_filelist(rvc, "voice-x", shuffle=lambda lines: None)
    mute_line = (exp / "filelist.txt").read_text(encoding="utf-8").splitlines()[-1]
    root = Path(rvc / "logs" / "mute").as_posix()
    named = [Path(part).as_posix().removeprefix(root + "/") for part in mute_line.split("|")[:4]]
    assert tuple(named) == MUTE_FILES == verify.MUTE_FILES


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_ffmpeg_and_ffprobe_are_both_required(verify, root, missing):
    probes = Probes()
    probes.missing = (missing,)
    c = by_id(run_checks(verify, root, probes))["ffmpeg"]
    assert not c["ok"] and missing in c["message"]


def test_wrong_commit_fails(verify, root):
    probes = Probes()
    probes.head = "0" * 40
    c = by_id(run_checks(verify, root, probes))["rvc_commit"]
    assert not c["ok"] and PINNED in c["message"]


def test_rejected_patch_is_named(verify, root):
    probes = Probes()
    probes.rejected_patch = "rvc_preprocess_keep_all_tails.patch"
    c = by_id(run_checks(verify, root, probes))["rvc_patches"]
    assert not c["ok"] and "rvc_preprocess_keep_all_tails.patch" in c["message"]
    assert "rvc_rocm_dilated_conv.patch" not in c["message"]


def test_cpu_torch_means_the_gpu_build_was_replaced(verify, root):
    probes = Probes()
    probes.python_out = "2.9.1+cpu False\n"
    c = by_id(run_checks(verify, root, probes))["torch_gpu"]
    assert not c["ok"] and "GPU build was replaced" in c["message"] and "2.9.1+cpu" in c["message"]


def test_rocm_torch_without_a_visible_gpu_reports_it(verify, root):
    probes = Probes()
    probes.python_out = "2.9.1+rocm7.2.1 False\n"
    c = by_id(run_checks(verify, root, probes))["torch_gpu"]
    assert not c["ok"] and "no GPU" in c["message"]


def test_python_that_crashes_is_a_failed_check(verify, root):
    probes = Probes()
    original = probes.run

    def crash(args, **kw):
        if "-c" in args:
            return subprocess.CompletedProcess(args, 1, "", "ImportError: DLL load failed")
        return original(args, **kw)

    c = by_id(verify.run_checks(root, run=crash, which=probes.which))["torch_gpu"]
    assert not c["ok"] and "DLL load failed" in c["message"]


def test_missing_exporter_environment_fails(verify, root):
    (root / "tools" / ".rvc" / "venv" / "Scripts" / "python.exe").unlink()
    c = by_id(run_checks(verify, root, Probes()))["exporter"]
    assert not c["ok"] and "tools/.rvc" in c["message"]


def test_main_prints_one_json_object_and_exit_code(verify, root, capsys, monkeypatch):
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    code = verify.main(["--fast", "--root", str(root)])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False
    assert not by_id(out)["ffmpeg"]["ok"]


# ---- the lock file: exact versions of what requirements.txt pulls in, and nothing of AMD's ----

LOCK = REPO / "trainer" / "requirements.lock"
REQUIREMENTS = REPO / "trainer" / "requirements.txt"


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
    wanted = {_canonical(re.match(r"[A-Za-z0-9._-]+", entry).group(0)) for entry in _entries(REQUIREMENTS)}
    assert wanted and wanted <= set(pinned), wanted - set(pinned)
    assert pinned["numpy"].startswith("1.")  # RVC at the pinned commit does not run on numpy 2


def test_setup_installs_from_the_lock_file_and_says_so_when_it_falls_back():
    script = (REPO / "trainer" / "setup.sh").read_text(encoding="utf-8")
    assert '-r "$T/requirements.lock"' in script and '-r "$T/requirements.txt"' in script
    assert script.index('-r "$T/requirements.lock"') < script.index("NOTICE") < script.index('-r "$T/requirements.txt"')
