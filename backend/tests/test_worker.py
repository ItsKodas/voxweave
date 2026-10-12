"""The training worker (trainer/worker.py), driven by a fake stage runner inside a fake trainer/rvc tree.

Nothing here needs the training environment, a GPU or the RVC clone. The fake runner does what each real stage
does to the disk (segments, features, checkpoints, the small model, the index, the exported folder) and feeds the
worker log lines copied from the spike notes. Time is a fake clock; no test sleeps.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "trainer" / "worker.py"
EXP = "voice-abc"
CLIPS_CHANGED = "The clips changed since this training run started. Start a new training run."
STATE_UNREADABLE = "The saved training state could not be read"
windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows process handling")


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location("aivoice_training_worker", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- fakes

class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class Killed(Exception):
    """Raised inside a fake stage when the worker asks for the stage to stop."""


def write_checkpoint(path, epoch):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("epoch", str(epoch))


def read_checkpoint(path):
    with zipfile.ZipFile(path) as z:
        return int(z.read("epoch"))


def truncate(path):
    Path(path).write_bytes(b"PK\x03\x04 the trainer was killed while writing this")


def write_model(path, payload=b"\0"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", payload * (1024 * 1024 + 4096))


class Call:
    def __init__(self, runner, stage, argv, cwd, env, on_line, should_stop):
        self.runner, self.stage, self.argv, self.cwd, self.env = runner, stage, argv, cwd, env
        self._on_line, self._should_stop = on_line, should_stop
        self.exp_dir = runner.rvc / "logs" / EXP

    def line(self, text):
        self._on_line(text)
        self.idle()

    def idle(self):
        if self._should_stop():
            raise Killed()

    def arg(self, flag):
        return self.argv[self.argv.index(flag) + 1]


class FakeRunner:
    """Stands in for the real subprocess runner. `stages[name]` replaces what a stage does; `at` is told about
    points inside the fake trainer ("epoch_start:3", "saving:5", "epoch_done:3")."""

    def __init__(self, rvc, clock):
        self.rvc, self.clock = rvc, clock
        self.calls, self.killed, self.stages, self.points = [], [], {}, []
        self.at = lambda point: None
        self.epoch_seconds, self.ticks = 10.0, 1
        self.in_save, self.damage = False, set()

    def __call__(self, stage, argv, cwd, env, on_line, should_stop):
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv), "argv must be a list of strings"
        call = Call(self, stage, argv, cwd, env, on_line, should_stop)
        self.calls.append(call)
        try:
            return self.stages.get(stage, getattr(self, stage))(call)
        except Killed:
            self.killed.append(stage)
            return 1

    def ran(self):
        return [c.stage for c in self.calls]

    def point(self, name):
        self.points.append(name)
        self.at(name)

    # what each real stage does, in miniature
    def preprocess(self, c):
        staged = sorted(os.listdir(c.argv[2]))
        for folder in ("0_gt_wavs", "1_16k_wavs"):
            (c.exp_dir / folder).mkdir(exist_ok=True)
        c.line("[Data slicing] Started")
        c.line(f"[Data slicing] Pending: {len(staged)} | Processes: 8")
        for i, name in enumerate(staged):
            for folder in ("0_gt_wavs", "1_16k_wavs"):
                (c.exp_dir / folder / f"{i}_0.wav").write_bytes(b"seg")
            c.line(f"[Data slicing] Progress: {i + 1}/{len(staged)} | {name}")
        c.line(f"[Data slicing] Worker completed | Success: {len(staged)} | Failed: 0")
        c.line("[Data slicing] Completed")
        return 0

    def _segments(self, c):
        return sorted(p.name for p in (c.exp_dir / "1_16k_wavs").iterdir())

    def pitch(self, c):
        names = self._segments(c)
        for folder in ("2a_f0", "2b-f0nsf"):
            (c.exp_dir / folder).mkdir(exist_ok=True)
        c.line(f"[F0 extraction] Pending: {len(names)}")
        c.line("Loading RMVPE model")
        done = 0
        for name in names:
            for folder in ("2a_f0", "2b-f0nsf"):
                (c.exp_dir / folder / (name + ".npy")).write_bytes(b"f0")
            done += 1
            c.line(f"[F0 extraction] Progress: {done}/{len(names)} | Success: {done} | Skipped: 0 | {name}")
        c.line(f"[F0 extraction] Completed | Success: {done} | Skipped: 0 | Failed: 0")
        return 0

    def features(self, c):
        names = self._segments(c)
        (c.exp_dir / "3_feature768").mkdir(exist_ok=True)
        c.line(f"[HuBERT features] Device: cuda:0 | Pending: {len(names)} | Skipped: 0")
        for i, name in enumerate(names):
            (c.exp_dir / "3_feature768" / name.replace(".wav", ".npy")).write_bytes(b"fea")
            c.line(f"[HuBERT features] Progress: {i + 1}/{len(names)} | Success: {i + 1} | Failed: 0 | {name} "
                   "| (44, 768)")
        c.line(f"[HuBERT features] Completed | Success: {len(names)} | Skipped: 0 | Failed: 0")
        return 0

    def train(self, c):
        exp, total, every = c.arg("-e"), int(c.arg("-te")), int(c.arg("-se"))
        g, d = c.exp_dir / "G_2333333.pth", c.exp_dir / "D_2333333.pth"
        c.line(f"INFO:{exp}:Training precision selected by device rules: torch.float16")
        fresh = False
        try:  # like train.py: discriminator first, then generator, any error falls back to the pretrained base
            start = read_checkpoint(d)
            c.line(f"INFO:{exp}:Loaded checkpoint './logs\\{exp}\\D_2333333.pth' (epoch {start})")
            start = read_checkpoint(g)
            c.line(f"INFO:{exp}:Loaded checkpoint './logs\\{exp}\\G_2333333.pth' (epoch {start})")
        except Exception:
            start, fresh = 1, True
            c.line(f"INFO:{exp}:Loaded generator pretrained model: assets/pretrained_v2/f0G40k.pth")
            c.line(f"INFO:{exp}:Loaded discriminator pretrained model: assets/pretrained_v2/f0D40k.pth")
        for epoch in range(start, total + 1):
            self.point(f"epoch_start:{epoch}")
            if fresh and epoch == start:
                c.line(f"INFO:{exp}:Training epoch: {epoch} [0%]")
                c.line(f"INFO:{exp}:[0, 0.0001]")
                c.line(f"INFO:{exp}:loss_disc=5.255, loss_gen=3.198, loss_fm=10.025,loss_mel=25.093, loss_kl=9.000")
            for _ in range(self.ticks):
                self.clock.now += self.epoch_seconds / self.ticks
                c.idle()
            if epoch % every == 0:
                self.point(f"saving:{epoch}")
                self.in_save = True
                c.line(f"INFO:{exp}:Saving model and optimizer state at epoch {epoch} to ./logs\\{exp}\\G_2333333.pth")
                write_checkpoint(g, epoch)
                if ("G", epoch) in self.damage:
                    truncate(g)
                c.line(f"INFO:{exp}:Saving model and optimizer state at epoch {epoch} to ./logs\\{exp}\\D_2333333.pth")
                write_checkpoint(d, epoch)
                self.in_save = False
            c.line(f"INFO:{exp}:====> Epoch: {epoch} [2026-10-13 06:42:58] | (0:00:10.0)")
            self.point(f"epoch_done:{epoch}")
        c.line(f"INFO:{exp}:Training completed; saving the final model")
        write_model(self.rvc / "assets" / "weights" / f"{exp}.pth")
        c.line(f"INFO:{exp}:Saving final checkpoint: Success")
        return 0

    def index(self, c):
        exp = c.argv[2]
        name = f"added_IVF225_Flat_nprobe_1_{exp}_v2.index"
        c.line("[Index training] Feature shape: (8788, 768) | IVF count: 225")
        c.line("[Index training] Write progress: 1/2")
        (c.exp_dir / name).write_bytes(b"index")
        (c.exp_dir / name.replace("added", "trained")).write_bytes(b"trained")
        (self.rvc / "assets" / "indices" / f"{exp}_{name}").write_bytes(b"index")
        c.line(f"[Index training] Index built successfully: {name}")
        return 0

    def export(self, c):
        out = Path(c.env["AIVOICE_EXPORT_DIR"])
        (out / "model.onnx").write_bytes(b"\1" * (1024 * 1024 + 1))
        (out / "model.index").write_bytes(Path(c.arg("--index")).read_bytes())
        (out / "voice.json").write_text(json.dumps({"name": c.arg("--name"), "sample_rate": 40000}))
        c.line(f"Exported {c.argv[3]} at 40000 Hz to {out}")
        return 0


class Rig:
    def __init__(self, worker, tmp_path, monkeypatch):
        self.worker = worker
        self.trainer, self.tools = tmp_path / "repo with space" / "trainer", tmp_path / "repo with space" / "tools"
        self.rvc = self.trainer / "rvc"
        for folder in ("train", "configs/v1", "logs/mute/0_gt_wavs", "assets/pretrained_v2"):
            (self.rvc / folder).mkdir(parents=True)
        (self.rvc / "train" / "train.py").write_text("")
        (self.rvc / "configs" / "v1" / "40k.json").write_text(json.dumps({"train": {"log_interval": 200}}))
        (self.rvc / "logs" / "mute" / "0_gt_wavs" / "mute40k.wav").write_bytes(b"mute")
        self.tools.mkdir(parents=True)
        monkeypatch.setattr(worker, "TRAINER_DIR", self.trainer)
        monkeypatch.setattr(worker, "TOOLS_DIR", self.tools)
        self.voice = tmp_path / "data" / "voices" / "abc"
        self.clips = self.voice / "clips"
        self.clips.mkdir(parents=True)
        for n in (1, 2, 3):
            self.add_clip(n)
        self.job = self.voice / "training"
        self.exp = self.rvc / "logs" / EXP
        self.weights = self.rvc / "assets" / "weights" / f"{EXP}.pth"
        self.model = self.voice / "models" / "rvc"
        self.clock = Clock()
        self.runner = FakeRunner(self.rvc, self.clock)
        self.parent_alive = lambda pid: True
        self.recordings = self.clips_snapshot()

    def add_clip(self, n):
        (self.clips / f"{n:04d}.orig.wav").write_bytes(b"RIFF48k" + bytes([n]) * 50)
        (self.clips / f"{n:04d}.wav").write_bytes(b"RIFF16k" + bytes([n]) * 20)
        (self.clips / f"{n:04d}.txt").write_text(f"secret transcript {n}")

    def args(self, *extra, epochs=10, batch_size=8, save_every=5, experiment=EXP):
        return self.worker.parse_args([
            "--voice-dir", str(self.voice), "--job-dir", str(self.job), "--experiment", experiment,
            "--epochs", str(epochs), "--batch-size", str(batch_size), "--save-every", str(save_every),
            "--name", "My Voice", *extra,
        ])

    def run(self, *extra, **settings):
        return self.worker.run(self.args(*extra, **settings), runner=self.runner, clock=self.clock,
                               kill_tree=lambda pid: None, parent_alive=lambda pid: self.parent_alive(pid))

    def progress(self):
        return [json.loads(line) for line in (self.job / "progress.jsonl").read_text("utf-8").splitlines()]

    def final(self):
        return self.progress()[-1]

    def log(self):
        return (self.job / "worker.log").read_text("utf-8")

    def cancel(self):
        (self.job / "cancel").write_text("")

    def cancel_at(self, point):
        self.runner.at = lambda p: self.cancel() if p == point else None

    def clips_snapshot(self):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(self.clips.iterdir())}

    def tree(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def rig(worker, tmp_path, monkeypatch):
    rig = Rig(worker, tmp_path, monkeypatch)
    yield rig
    # Whatever a test put the worker through, every recording is still there, byte for byte, with its old time.
    now = rig.clips_snapshot()
    assert {name: now.get(name) for name in rig.recordings} == rig.recordings


def interrupt_training(rig, point="epoch_start:7", **settings):
    """A first run that is cancelled inside training (by default after the epoch 5 checkpoint and its backup)."""
    rig.cancel_at(point)
    assert rig.run(**settings) == 2
    rig.runner.at = lambda p: None
    rig.runner.calls.clear(), rig.runner.killed.clear(), rig.runner.points.clear()


# ---------------------------------------------------------------- the whole run

def test_a_full_run_trains_exports_and_leaves_the_recordings_untouched(rig):
    before = rig.clips_snapshot()
    assert rig.run() == 0
    assert rig.runner.ran() == ["preprocess", "pitch", "features", "train", "index", "export"]
    final = rig.final()
    assert (final["done"], final["ok"], final["error"], final["cancelled"]) == (True, True, None, False)
    assert final["percent"] == 100 and final["resumable"] is False  # nothing is left to resume
    assert json.loads((rig.model / "voice.json").read_text()) == {"name": "My Voice", "sample_rate": 40000}
    assert rig.clips_snapshot() == before


# ---------------------------------------------------------------- 1. prepare

def test_prepare_stages_only_the_48k_originals_as_hard_links(rig):
    seen = {}

    def preprocess(c):
        staging = Path(c.argv[2])
        seen["names"] = sorted(os.listdir(staging))
        seen["linked"] = all(os.path.samefile(staging / n, rig.clips / n) for n in seen["names"])
        return FakeRunner.preprocess(rig.runner, c)

    rig.runner.stages["preprocess"] = preprocess
    assert rig.run() == 0
    assert seen == {"names": ["0001.orig.wav", "0002.orig.wav", "0003.orig.wav"], "linked": True}
    assert Path(rig.runner.calls[0].argv[2]) == rig.exp / "staging"


def test_prepare_copies_when_hard_links_are_not_possible(rig, monkeypatch):
    def no_links(src, dst):
        raise OSError("links are not supported here")

    monkeypatch.setattr(rig.worker.os, "link", no_links)
    seen = {}

    def preprocess(c):
        staging = Path(c.argv[2])
        seen["files"] = {n: (staging / n).read_bytes() for n in sorted(os.listdir(staging))}
        seen["linked"] = any(os.path.samefile(staging / n, rig.clips / n) for n in seen["files"])
        return FakeRunner.preprocess(rig.runner, c)

    rig.runner.stages["preprocess"] = preprocess
    before = rig.clips_snapshot()
    assert rig.run() == 0
    assert seen["files"] == {f"{n:04d}.orig.wav": (rig.clips / f"{n:04d}.orig.wav").read_bytes() for n in (1, 2, 3)}
    assert seen["linked"] is False
    assert rig.clips_snapshot() == before


def test_a_voice_without_clips_fails_clearly_and_deletes_nothing(rig):
    for path in rig.clips.glob("*.orig.wav"):
        path.unlink()
    rig.recordings = rig.clips_snapshot()
    rig.exp.mkdir(parents=True)
    (rig.exp / "G_2333333.pth").write_bytes(b"an earlier run")
    assert rig.run() == 1
    assert rig.final()["error"] == "This voice has no recorded clips to train on."
    assert rig.final()["resumable"] is False
    assert rig.runner.calls == []
    assert (rig.exp / "G_2333333.pth").read_bytes() == b"an earlier run"


def test_a_fresh_run_removes_the_previous_experiment_and_its_model(rig):
    rig.exp.mkdir(parents=True)
    (rig.exp / "leftover.txt").write_text("old")
    (rig.exp / "0_gt_wavs").mkdir()
    (rig.exp / "0_gt_wavs" / "99_0.wav").write_bytes(b"old segment")
    rig.weights.parent.mkdir(parents=True)
    write_model(rig.weights, b"\7")
    seen = {}

    def preprocess(c):
        seen["leftover"] = (rig.exp / "leftover.txt").exists() or (rig.exp / "0_gt_wavs" / "99_0.wav").exists()
        seen["weights"] = rig.weights.exists()
        return FakeRunner.preprocess(rig.runner, c)

    rig.runner.stages["preprocess"] = preprocess
    assert rig.run() == 0
    assert seen == {"leftover": False, "weights": False}
    assert (rig.rvc / "logs" / "mute" / "0_gt_wavs" / "mute40k.wav").exists()


@pytest.mark.parametrize("name", ["../abc", "mute", "a/b", "a\\b", "", ".", "voice abc", "C:evil"])
def test_an_unsafe_experiment_name_is_refused_before_anything_is_touched(rig, name):
    assert rig.run(experiment=name) == 1
    assert "experiment name" in rig.final()["error"]
    assert rig.runner.calls == []
    assert (rig.rvc / "logs" / "mute" / "0_gt_wavs" / "mute40k.wav").exists()


def test_a_missing_training_environment_is_a_clear_failure(rig):
    (rig.rvc / "train" / "train.py").unlink()
    (rig.rvc / "train").rmdir()
    assert rig.run() == 1
    assert "trainer/setup.sh" in rig.final()["error"]


# ---------------------------------------------------------------- 2. stage commands

def test_each_stage_gets_the_exact_command_directory_and_environment_from_the_spike(rig):
    assert rig.run(epochs=10, batch_size=6, save_every=5) == 0
    py = str(rig.trainer / ".venv" / "Scripts" / "python.exe")
    exp_dir = f"{rig.rvc.as_posix()}/logs/{EXP}"
    calls = {c.stage: c for c in rig.runner.calls}
    assert calls["preprocess"].argv == [
        py, "train/preprocess.py", f"{exp_dir}/staging", "40000", "8", exp_dir, "False", "3.7"]
    assert calls["pitch"].argv == [py, "train/dataset/extract_f0.py", "cuda", "1", "0", "0", exp_dir, "true"]
    assert calls["features"].argv == [
        py, "train/dataset/extract_hubert_feature.py", "cuda:0", "1", "0", "0", exp_dir, "v2", "true"]
    assert calls["train"].argv == [
        py, "train/train.py", "-e", EXP, "-sr", "40k", "-f0", "1", "-bs", "6", "-g", "0", "-te", "10", "-se", "5",
        "-pg", "assets/pretrained_v2/f0G40k.pth", "-pd", "assets/pretrained_v2/f0D40k.pth",
        "-l", "1", "-c", "0", "-sw", "0", "-v", "v2"]
    assert calls["index"].argv == [py, "train/train_index.py", EXP, "v2", "assets/indices", "16", "single"]
    cache = rig.trainer / ".cache"
    for stage in ("preprocess", "pitch", "features", "train", "index"):
        env = calls[stage].env
        assert Path(calls[stage].cwd) == rig.rvc
        assert {k: env[k] for k in (
            "PYTHONPATH", "PYTHONSAFEPATH", "RVC_CUDA_GRAPH", "ROCM_SDK_TARGET_FAMILY", "MIOPEN_LOG_LEVEL",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONIOENCODING", "PYTHONUNBUFFERED")} == {
            "PYTHONPATH": str(rig.rvc), "PYTHONSAFEPATH": "1", "RVC_CUDA_GRAPH": "0",
            "ROCM_SDK_TARGET_FAMILY": "custom", "MIOPEN_LOG_LEVEL": "3", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        assert env["MIOPEN_USER_DB_PATH"] == str(cache / "miopen" / "db")
        assert env["MIOPEN_CUSTOM_CACHE_DIR"] == str(cache / "miopen" / "cache")
        for key in ("HF_HOME", "TORCH_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
            assert Path(env[key]).parent == cache
        assert env["PATH"] == os.environ["PATH"]  # ffmpeg and ffprobe are found through the inherited PATH
    assert (cache / "miopen" / "db").is_dir() and (cache / "miopen" / "cache").is_dir()


def test_the_filelist_is_written_before_training_starts(rig):
    seen = {}

    def train(c):
        seen["lines"] = (rig.exp / "filelist.txt").read_text("utf8").splitlines()
        seen["config"] = json.loads((rig.exp / "config.json").read_text("utf8"))
        return FakeRunner.train(rig.runner, c)

    rig.runner.stages["train"] = train
    assert rig.run() == 0
    assert len(seen["lines"]) == 3 + 2  # one segment per clip plus the two silence lines
    assert seen["config"] == {"train": {"log_interval": 200}}


# ---------------------------------------------------------------- 3. output checks

def fail_with(rig, stage, behaviour):
    rig.runner.stages[stage] = behaviour
    assert rig.run() == 1
    final = rig.final()
    assert final["done"] and not final["ok"] and not final["cancelled"]
    assert rig.runner.ran()[-1] == stage, "no later stage may run"
    return final["error"]


def test_preprocess_that_fails_a_file_fails_the_job_although_it_exits_zero(rig):
    def preprocess(c):
        FakeRunner.preprocess(rig.runner, c)
        c.line("[Data slicing][Failed] D:/x/staging/0002.orig.wav")
        c.line("Traceback (most recent call last):")
        c.line("[Data slicing][Failed] D:/x/staging/0003.orig.wav")
        return 0

    error = fail_with(rig, "preprocess", preprocess)
    assert "2 clips" in error and "0002.orig.wav" in error and "0003" not in error
    assert rig.final()["resumable"] is False, "the same clips would fail the same way"


def test_preprocess_that_produces_no_segments_fails(rig):
    def preprocess(c):
        c.line("[Data slicing] Completed")
        return 0

    assert "no audio" in fail_with(rig, "preprocess", preprocess)


@pytest.mark.parametrize("stage, line, expected", [
    ("pitch", "[F0 extraction] Completed | Success: 1 | Skipped: 0 | Failed: 2", "2 segments"),
    ("features", "[HuBERT features] Completed | Success: 2 | Skipped: 0 | Failed: 1", "1 segment"),
])
def test_pitch_or_features_reporting_failures_fails_the_job(rig, stage, line, expected):
    def behaviour(c):
        getattr(FakeRunner, stage)(rig.runner, c)
        c.line(line)
        return 0

    assert expected in fail_with(rig, stage, behaviour)


def test_segments_missing_their_features_fail_instead_of_being_dropped_silently(rig):
    def features(c):
        FakeRunner.features(rig.runner, c)
        (rig.exp / "3_feature768" / "1_0.npy").unlink()
        return 0

    rig.runner.stages["features"] = features
    assert rig.run() == 1
    assert "1 of 3 segments" in rig.final()["error"]
    assert "train" not in rig.runner.ran()


def test_training_that_exits_zero_without_the_model_fails(rig):
    def train(c):
        c.line(f"INFO:{EXP}:Saving final checkpoint: Traceback (most recent call last): FileNotFoundError")
        return 0

    assert "did not write the voice model" in fail_with(rig, "train", train)


@pytest.mark.parametrize("damage", ["small", "truncated"])
def test_a_final_model_that_is_too_small_or_cut_short_fails(rig, damage):
    def train(c):
        if damage == "small":
            rig.weights.write_bytes(b"x" * 1000)
        else:
            write_model(rig.weights)
            rig.weights.write_bytes(rig.weights.read_bytes()[:-200])  # still over 1 MB, but cut short
        return 0

    assert "did not write the voice model" in fail_with(rig, "train", train)


def test_an_index_stage_that_leaves_no_index_fails(rig):
    def index(c):
        c.line("[Index training] Training index")
        return 0

    assert "search index" in fail_with(rig, "index", index)


def test_a_non_zero_exit_reports_the_last_twenty_lines(rig):
    def pitch(c):
        for i in range(30):
            c.line(f"line {i}")
        return 3

    error = fail_with(rig, "pitch", pitch)
    assert "exit code 3" in error and rig.final()["resumable"] is True
    assert error.splitlines()[-20:] == [f"line {i}" for i in range(10, 30)]
    assert "line 9" not in error.splitlines()


# ---------------------------------------------------------------- 4. progress and ETA

def test_every_progress_line_has_the_documented_shape_and_percent_never_goes_down(rig):
    assert rig.run() == 0
    lines = rig.progress()
    order = ["prepare", "preprocess", "pitch", "features", "train", "index", "export"]
    bands = {"prepare": (0, 1), "preprocess": (1, 3), "pitch": (3, 7), "features": (7, 13), "train": (13, 97),
             "index": (97, 98), "export": (98, 100)}
    for line in lines:
        assert set(line) >= {"time", "stage", "stage_index", "stages", "percent", "epoch", "epochs", "eta_seconds",
                             "message"}
        assert line["stage_index"] == order.index(line["stage"]) + 1 and line["stages"] == 7
        low, high = bands[line["stage"]]
        assert low <= line["percent"] <= high
        assert isinstance(line["message"], str) and line["message"]
        if line["stage"] != "train":
            assert line["epoch"] is None and line["epochs"] is None and line["eta_seconds"] is None
    assert [line["stage"] for line in lines] == sorted((line["stage"] for line in lines), key=order.index)
    percents = [line["percent"] for line in lines]
    assert percents == sorted(percents)
    assert [line.get("done", False) for line in lines] == [False] * (len(lines) - 1) + [True]
    for stage in order:
        assert any(line["stage"] == stage for line in lines)
    assert lines[0]["time"] == 1_000_000.0 and lines[-1]["time"] == rig.clock.now


def test_stage_progress_moves_within_the_stage_band(rig):
    assert rig.run() == 0
    pitch = [line["percent"] for line in rig.progress() if line["stage"] == "pitch"]
    assert pitch[0] == 3 and pitch[-1] == 7 and len(set(pitch)) >= 3


def test_training_percent_is_epochs_completed_over_total(rig):
    assert rig.run(epochs=10) == 0
    by_epoch = {line["epoch"]: line["percent"] for line in rig.progress()
                if line["stage"] == "train" and line["epoch"]}
    assert by_epoch[5] == pytest.approx(13 + 84 * 5 / 10)
    assert by_epoch[10] == pytest.approx(97)
    assert all(line["epochs"] == 10 for line in rig.progress() if line["stage"] == "train")


def test_eta_is_the_mean_of_the_last_ten_epochs_times_the_epochs_left(rig):
    durations = {n: 10.0 for n in range(1, 31)}
    durations.update({1: 500.0, 2: 40.0, 3: 40.0})  # the slow warm-up epochs must age out of the average

    def at(point):
        kind, _, n = point.partition(":")
        if kind == "epoch_start":
            rig.runner.epoch_seconds = durations[int(n)]

    rig.runner.at = at
    assert rig.run(epochs=30, save_every=50) == 0
    eta = {}
    for line in rig.progress():
        if line["stage"] == "train" and line["epoch"]:
            eta.setdefault(line["epoch"], line["eta_seconds"])
    assert eta[1] is None  # one finished epoch: its length includes start-up, so there is nothing to average
    assert eta[2] == pytest.approx(40.0 * 28)
    assert eta[4] == pytest.approx((40 + 40 + 10) / 3 * 26)
    assert eta[13] == pytest.approx(10.0 * 17)  # epochs 4 to 13 only
    assert eta[30] == 0


# ---------------------------------------------------------------- 5. checkpoint backup

def test_a_saved_checkpoint_pair_is_backed_up_after_its_epoch_ends(rig):
    seen = {}

    def at(point):
        if point in ("saving:5", "epoch_done:5", "epoch_done:6", "saving:10"):
            seen[point] = {p.name: read_checkpoint(p) for p in rig.exp.glob("*.bak")}

    rig.runner.at = at
    rig.runner.stages["export"] = lambda c: 1  # keep the experiment folder for inspection
    assert rig.run(epochs=10, save_every=5) == 1
    pair = {"G_2333333.pth.bak": 5, "D_2333333.pth.bak": 5}
    assert seen == {"saving:5": {}, "epoch_done:5": pair, "epoch_done:6": pair, "saving:10": pair}
    assert "[worker] checkpoint of epoch 5 backed up in " in rig.log()
    assert not list(rig.exp.glob("*.tmp"))


def test_the_backup_is_never_copied_while_a_save_is_in_progress(rig, monkeypatch):
    copies = []
    real_copy = rig.worker.copy_durable

    def spy(src, dst):
        copies.append((Path(src).name, Path(dst).name, rig.runner.in_save, rig.runner.points[-1]))
        real_copy(src, dst)

    monkeypatch.setattr(rig.worker, "copy_durable", spy)
    rig.runner.stages["export"] = lambda c: 1
    rig.run(epochs=12, save_every=4)
    assert copies and all(in_save is False for _, _, in_save, _ in copies)
    assert [(src, point) for src, _, _, point in copies] == [
        ("G_2333333.pth", "saving:4"), ("D_2333333.pth", "saving:4"),
        ("G_2333333.pth", "saving:8"), ("D_2333333.pth", "saving:8")]
    assert all(dst.endswith(".bak.tmp") for _, dst, _, _ in copies), "written to a temp name, then renamed"


def test_a_damaged_checkpoint_never_replaces_a_good_backup(rig):
    rig.runner.damage.add(("G", 10))  # the generator written at epoch 10 comes out cut short
    rig.runner.stages["export"] = lambda c: 1
    rig.run(epochs=20, save_every=5)
    assert read_checkpoint(rig.exp / "G_2333333.pth.bak") == 15
    rig.runner.damage.add(("G", 15))
    rig.run(epochs=20, save_every=5)
    assert read_checkpoint(rig.exp / "G_2333333.pth.bak") == 5
    assert read_checkpoint(rig.exp / "D_2333333.pth.bak") == 5, "the pair is replaced together or not at all"
    assert not list(rig.exp.glob("*.tmp"))
    assert "backup" in rig.log().lower()


def test_a_backup_that_cannot_be_written_does_not_stop_training(rig, monkeypatch):
    def full_disk(src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(rig.worker, "copy_durable", full_disk)
    assert rig.run() == 0


# ---------------------------------------------------------------- 6. resume

def test_resume_refuses_when_a_clip_was_added(rig):
    interrupt_training(rig)
    rig.add_clip(4)
    before = rig.tree(rig.exp)
    assert rig.run("--resume") == 1
    assert rig.final()["error"] == CLIPS_CHANGED and rig.final()["resumable"] is False
    assert rig.runner.calls == []
    assert rig.tree(rig.exp) == before


def test_resume_refuses_when_a_clip_was_deleted_or_rerecorded(rig):
    interrupt_training(rig)
    clip = rig.clips / "0002.orig.wav"
    clip.write_bytes(clip.read_bytes() + b"more")
    rig.recordings = rig.clips_snapshot()
    assert rig.run("--resume") == 1
    assert rig.final()["error"] == CLIPS_CHANGED


def test_resume_refuses_when_the_dataset_record_is_missing_or_unreadable(rig):
    interrupt_training(rig)
    (rig.exp / "dataset.json").write_text("{not json")
    assert rig.run("--resume") == 1
    assert rig.final()["error"] == CLIPS_CHANGED
    (rig.exp / "dataset.json").unlink()
    assert rig.run("--resume") == 1
    assert rig.final()["error"] == CLIPS_CHANGED


def test_resume_refuses_when_the_batch_size_changed_but_not_when_epochs_did(rig):
    interrupt_training(rig, batch_size=8)
    assert rig.run("--resume", batch_size=4) == 1
    assert rig.final()["error"] == CLIPS_CHANGED
    assert rig.run("--resume", batch_size=8, epochs=20) == 0


def test_the_dataset_record_holds_the_fingerprint_and_settings(rig):
    interrupt_training(rig)
    record = json.loads((rig.exp / "dataset.json").read_text("utf8"))
    assert len(record["fingerprint"]) == 64
    assert record["settings"]["batch_size"] == 8 and record["settings"]["sample_rate"] == "40k"
    assert "epochs" not in record["settings"]


def test_resume_skips_finished_stages_and_continues_from_the_checkpoint(rig):
    interrupt_training(rig)
    first_run_lines = len(rig.progress())
    assert rig.run("--resume") == 0
    assert rig.runner.ran() == ["pitch", "features", "train", "index", "export"]  # no preprocess
    assert rig.runner.points[0] == "epoch_start:5"  # RVC repeats the number of the saved epoch
    assert "epoch_start:1" not in rig.runner.points
    lines = rig.progress()
    assert len(lines) > first_run_lines, "the progress file is appended to on resume"
    resumed = lines[first_run_lines:]
    assert [line["percent"] for line in resumed] == sorted(line["percent"] for line in resumed)
    assert rig.final()["ok"] is True


def test_resume_reruns_a_stage_that_was_interrupted_from_a_clean_slate(rig):
    def pitch(c):
        (c.exp_dir / "2a_f0").mkdir()
        (c.exp_dir / "2a_f0" / "0_0.wav.npy").write_bytes(b"half-written")
        rig.cancel()
        c.idle()

    rig.runner.stages["pitch"] = pitch
    assert rig.run() == 2
    del rig.runner.stages["pitch"]
    rig.runner.calls.clear()
    seen = {}

    def pitch_again(c):
        seen["stale"] = (c.exp_dir / "2a_f0" / "0_0.wav.npy").exists()
        return FakeRunner.pitch(rig.runner, c)

    rig.runner.stages["pitch"] = pitch_again
    assert rig.run("--resume") == 0
    assert seen == {"stale": False}
    assert rig.runner.ran() == ["pitch", "features", "train", "index", "export"]
    assert "epoch_start:1" in rig.runner.points  # no checkpoint was ever saved, so training starts normally


def test_resume_before_any_checkpoint_was_saved_trains_from_the_start(rig):
    interrupt_training(rig, point="epoch_start:3")  # the first save would have been at epoch 5
    assert not list(rig.exp.glob("*.pth*"))
    assert rig.run("--resume") == 0
    assert rig.runner.points[0] == "epoch_start:1"


def test_resume_with_an_unreadable_checkpoint_falls_back_to_the_backup(rig):
    interrupt_training(rig, point="epoch_start:12", epochs=20)  # saved and backed up at 5 and 10
    write_checkpoint(rig.exp / "G_2333333.pth.bak", 5), write_checkpoint(rig.exp / "D_2333333.pth.bak", 5)
    truncate(rig.exp / "D_2333333.pth")  # the trainer was killed while saving epoch 10
    assert rig.run("--resume", epochs=20) == 0
    assert rig.runner.ran() == ["pitch", "features", "train", "train", "index", "export"]
    assert rig.runner.killed == ["train"], "the trainer that fell back to the pretrained base is stopped"
    first_try = rig.runner.points.index("epoch_start:1")
    assert rig.runner.points[first_try + 1] == "epoch_start:5", "stopped before it trained or saved anything"
    assert rig.final()["ok"] is True


def test_a_half_resumed_trainer_counts_as_not_resumed(rig):
    interrupt_training(rig, point="epoch_start:12", epochs=20)
    truncate(rig.exp / "G_2333333.pth")  # the discriminator loads, the generator does not
    assert rig.run("--resume", epochs=20) == 0
    assert rig.runner.ran().count("train") == 2 and rig.runner.killed == ["train"]


def test_a_trainer_that_did_not_resume_is_caught_on_any_display_language(rig):
    interrupt_training(rig)

    def train(c):  # the epoch lines are translated by RVC; the loss line never is
        c.line(f"INFO:{EXP}:Loaded generator pretrained model: assets/pretrained_v2/f0G40k.pth")
        c.line(f"INFO:{EXP}:loss_disc=5.255, loss_gen=3.198, loss_fm=10.025,loss_mel=25.093, loss_kl=9.000")
        raise AssertionError("the trainer was left running although it did not resume")

    rig.runner.stages["train"] = train
    assert rig.run("--resume") == 1
    assert rig.final()["error"] == STATE_UNREADABLE and rig.runner.killed == ["train", "train"]
    assert read_checkpoint(rig.exp / "G_2333333.pth") == 5 and read_checkpoint(rig.exp / "G_2333333.pth.bak") == 5


def test_resume_fails_when_the_backup_is_unreadable_too_and_deletes_nothing(rig):
    interrupt_training(rig, point="epoch_start:12", epochs=20)
    for name in ("G_2333333.pth", "D_2333333.pth", "G_2333333.pth.bak", "D_2333333.pth.bak"):
        truncate(rig.exp / name)
        (rig.exp / name).write_bytes((rig.exp / name).read_bytes() + name.encode())
    before = rig.tree(rig.exp)
    assert rig.run("--resume", epochs=20) == 1
    assert rig.final()["error"] == STATE_UNREADABLE and rig.final()["resumable"] is False
    assert rig.runner.ran() == ["pitch", "features", "train", "train"] and rig.runner.killed == ["train", "train"]
    after = rig.tree(rig.exp)
    checkpoints = [n for n in before if ".pth" in n]
    assert len(checkpoints) == 4 and {n: after[n] for n in checkpoints} == {n: before[n] for n in checkpoints}
    assert not rig.weights.exists()
    assert rig.exp.is_dir()


def test_resume_fails_when_the_checkpoint_is_unreadable_and_there_is_no_backup(rig):
    interrupt_training(rig)
    truncate(rig.exp / "G_2333333.pth")
    (rig.exp / "G_2333333.pth.bak").unlink()
    before = rig.tree(rig.exp)
    assert rig.run("--resume") == 1
    assert rig.final()["error"] == STATE_UNREADABLE
    assert rig.runner.ran().count("train") == 1
    assert rig.tree(rig.exp) == before | {"filelist.txt": rig.tree(rig.exp)["filelist.txt"]}


def test_resume_fails_when_the_checkpoints_are_gone(rig):
    interrupt_training(rig)
    for path in rig.exp.glob("*.pth*"):
        path.unlink()
    assert rig.run("--resume") == 1
    assert rig.final()["error"] == STATE_UNREADABLE
    assert rig.runner.killed == ["train"], "it must not train from scratch in silence"
    assert not rig.weights.exists() and not list(rig.exp.glob("*.pth*"))


def test_a_trainer_that_did_not_resume_never_overwrites_the_backup(rig):
    interrupt_training(rig, point="epoch_start:12", epochs=20)
    truncate(rig.exp / "G_2333333.pth"), truncate(rig.exp / "D_2333333.pth")
    truncate(rig.exp / "G_2333333.pth.bak")  # the backup pair cannot resume either
    assert rig.run("--resume", epochs=20, save_every=1) == 1
    assert read_checkpoint(rig.exp / "D_2333333.pth.bak") == 10
    assert not any(p.startswith("saving") for p in rig.runner.points)


def test_resume_after_training_finished_goes_straight_to_the_index_and_export(rig):
    rig.runner.stages["index"] = lambda c: (rig.cancel(), c.idle())
    assert rig.run() == 2
    del rig.runner.stages["index"]
    rig.runner.calls.clear()
    assert rig.run("--resume") == 0
    assert rig.runner.ran() == ["pitch", "features", "index", "export"]


def test_resume_retrains_when_the_final_model_was_cut_short(rig):
    interrupt_training(rig, point="epoch_done:10")  # cancelled just as training finished
    rig.weights.write_bytes(rig.weights.read_bytes()[:-500])  # ...and killed while the small model was written
    assert rig.run("--resume") == 0
    assert rig.runner.ran() == ["pitch", "features", "train", "index", "export"]


# ---------------------------------------------------------------- 7. cancel

def assert_cancelled(rig, code, message="Training was cancelled"):
    assert code == 2
    final = rig.final()
    assert (final["done"], final["ok"], final["cancelled"], final["error"]) == (True, False, True, None)
    assert final["message"] == message and final["resumable"] is True
    assert not (rig.job / "cancel").exists(), "the cancel file is removed at exit"


def test_cancel_during_a_short_stage_kills_it_at_once(rig):
    def features(c):
        c.line("[HuBERT features] Device: cuda:0 | Pending: 3 | Skipped: 0")
        rig.cancel()
        c.idle()
        raise AssertionError("the stage was not stopped")

    rig.runner.stages["features"] = features
    assert_cancelled(rig, rig.run())
    assert rig.runner.killed == ["features"] and rig.runner.ran()[-1] == "features"
    assert rig.exp.is_dir() and not rig.model.exists()


def test_a_stale_cancel_file_is_removed_at_start(rig):
    rig.job.mkdir(parents=True)
    rig.cancel()
    assert rig.run() == 0


def test_cancel_during_training_waits_for_the_end_of_the_epoch(rig):
    rig.cancel_at("epoch_start:3")
    assert_cancelled(rig, rig.run())
    assert rig.runner.killed == ["train"]
    assert rig.runner.points[-1] == "epoch_start:3"
    assert f"[train] INFO:{EXP}:====> Epoch: 3 " in rig.log(), "killed right after the epoch 3 line"
    assert "Stopping" in " ".join(line["message"] for line in rig.progress())
    assert rig.exp.is_dir() and not rig.weights.exists()


def test_cancel_during_a_save_waits_until_the_save_and_its_epoch_are_done(rig):
    rig.cancel_at("saving:5")
    assert_cancelled(rig, rig.run())
    assert rig.runner.in_save is False, "never killed between a save line and the end of its epoch"
    assert read_checkpoint(rig.exp / "G_2333333.pth") == 5 and read_checkpoint(rig.exp / "D_2333333.pth") == 5
    assert "====> Epoch: 5 " in rig.log() and "epoch_done:5" not in rig.runner.points


def test_cancel_gives_up_waiting_after_sixty_seconds(rig):
    rig.runner.epoch_seconds, rig.runner.ticks = 400.0, 20  # an idle check every 20 s
    started = {}

    def at(point):
        if point == "epoch_start:2":
            rig.cancel()
            started["at"] = rig.clock.now

    rig.runner.at = at
    assert_cancelled(rig, rig.run())
    assert rig.runner.killed == ["train"]
    assert 60 <= rig.clock.now - started["at"] <= 80
    assert "====> Epoch: 2 " not in rig.log()


def test_cancel_at_the_last_epoch_lets_the_trainer_write_the_model(rig):
    rig.cancel_at("epoch_start:10")
    assert_cancelled(rig, rig.run(epochs=10))
    assert rig.runner.killed == [] and rig.weights.exists()
    assert rig.runner.ran()[-1] == "train", "but no further stage starts"


def test_cancel_removes_a_half_exported_model_and_keeps_the_old_one(rig):
    rig.model.mkdir(parents=True)
    (rig.model / "model.onnx").write_bytes(b"previous model")

    def export(c):
        (Path(c.env["AIVOICE_EXPORT_DIR"]) / "model.onnx").write_bytes(b"half")
        rig.cancel()
        c.idle()

    rig.runner.stages["export"] = export
    assert_cancelled(rig, rig.run())
    assert rig.tree(rig.voice / "models") == {"rvc/model.onnx": b"previous model"}


# ---------------------------------------------------------------- 8. parent watchdog

def test_the_worker_stops_when_the_app_disappears(rig):
    asked = []

    def features(c):
        c.line("[HuBERT features] Device: cuda:0 | Pending: 3 | Skipped: 0")
        rig.parent_alive = lambda pid: asked.append(pid) and False
        c.idle()
        raise AssertionError("the stage was not stopped")

    rig.runner.stages["features"] = features
    assert_cancelled(rig, rig.run("--parent-pid", "4242"), message="The app was closed")
    assert asked == [4242] and rig.runner.killed == ["features"]


def test_without_a_parent_pid_nothing_is_watched(rig):
    rig.parent_alive = lambda pid: False
    assert rig.run() == 0


def test_process_alive_sees_a_running_and_a_finished_process(worker):
    assert worker.process_alive(os.getpid()) is True
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    assert worker.process_alive(done.pid) is False


# ---------------------------------------------------------------- 9. killing a tree, and the real runner

def test_kill_process_tree_runs_taskkill_on_the_whole_tree_without_a_window(worker, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    worker.kill_process_tree(4321)
    assert seen["argv"] == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert seen["kwargs"].get("shell", False) is False
    assert seen["kwargs"]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FakeProcess:
    pid = 777

    def __init__(self, lines, code=0):
        self.stdout, self._code, self.killed = iter(lines), code, False

    def poll(self):
        return self._code

    def wait(self, timeout=None):
        return self._code

    def kill(self):
        self.killed = True


def test_the_real_runner_starts_the_stage_as_a_list_with_merged_utf8_output(worker, tmp_path):
    seen, lines = {}, []

    def popen(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        return FakeProcess(["one\n", "two\r\n", "three"], code=5)

    runner = worker.make_runner(kill_tree=lambda pid: seen.setdefault("killed", pid), popen=popen)
    code = runner("pitch", ["py", "a b.py"], tmp_path, {"A": "1"}, lines.append, lambda: False)
    assert code == 5 and lines == ["one", "two", "three"] and "killed" not in seen
    kwargs = seen["kwargs"]
    assert seen["argv"] == ["py", "a b.py"] and kwargs.get("shell", False) is False
    assert kwargs["cwd"] == str(tmp_path) and kwargs["env"] == {"A": "1"}
    assert kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert (kwargs["encoding"], kwargs["errors"], kwargs["bufsize"]) == ("utf-8", "replace", 1)
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)


def test_the_real_runner_kills_the_tree_once_when_asked_to_stop(worker, tmp_path):
    killed, lines = [], []
    runner = worker.make_runner(kill_tree=killed.append, popen=lambda argv, **kw: FakeProcess(["a\n", "b\n", "c\n"], 1))
    runner("train", ["py"], tmp_path, {}, lines.append, lambda: len(lines) >= 2)
    assert killed == [777]


def test_the_real_runner_kills_the_tree_when_the_worker_itself_fails(worker, tmp_path):
    killed = []

    class Running(FakeProcess):
        def poll(self):
            return None

    def explode(line):
        raise RuntimeError("bug in the worker")

    runner = worker.make_runner(kill_tree=killed.append, popen=lambda argv, **kw: Running(["a\n"]))
    with pytest.raises(RuntimeError):
        runner("train", ["py"], tmp_path, {}, explode, lambda: False)
    assert killed == [777]


def wait_until_gone(worker, pid):
    deadline = time.monotonic() + 20
    while worker.process_alive(pid):
        assert time.monotonic() < deadline, f"process {pid} is still running"


GRANDCHILD = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "print('child', child.pid, flush=True)\n"
)


@windows_only
def test_stopping_a_real_stage_ends_its_whole_process_tree(worker, tmp_path):
    lines = []
    runner = worker.make_runner(worker.kill_process_tree)
    code = runner("train", [sys.executable, "-c", GRANDCHILD + "time.sleep(300)\n"], tmp_path, dict(os.environ),
                  lines.append, lambda: bool(lines))
    assert code != 0
    wait_until_gone(worker, int(lines[0].split()[1]))


@windows_only
def test_children_die_with_the_worker_even_when_it_is_killed_outright(worker, tmp_path):
    script = (
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('w', {str(MODULE)!r})\n"
        "w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n"
        "assert w.bind_children_to_this_process()\n" + GRANDCHILD + "os._exit(0)\n"  # no clean-up of any kind
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
    assert done.stdout.startswith("child "), done.stderr
    wait_until_gone(worker, int(done.stdout.split()[1]))


# ---------------------------------------------------------------- 10. export

def test_the_export_runs_the_exporter_with_its_own_python_into_a_new_folder(rig):
    seen = {}

    def export(c):
        seen["old model during export"] = rig.model.exists()
        return FakeRunner.export(rig.runner, c)

    rig.runner.stages["export"] = export
    assert rig.run() == 0
    call = rig.runner.calls[-1]
    index = rig.exp / f"added_IVF225_Flat_nprobe_1_{EXP}_v2.index"
    assert call.argv == [
        str(rig.tools / ".rvc" / "venv" / "Scripts" / "python.exe"), str(rig.tools / "export_rvc_onnx.py"),
        str(rig.weights), EXP, "--index", str(index), "--name", "My Voice"]
    assert Path(call.cwd) == rig.tools.parent
    assert call.env["AIVOICE_EXPORT_DIR"] == str(rig.voice / "models" / "rvc.new")
    assert "PYTHONPATH" not in call.env or call.env["PYTHONPATH"] == os.environ.get("PYTHONPATH")
    assert seen == {"old model during export": False}
    assert sorted(p.name for p in (rig.voice / "models").iterdir()) == ["rvc"]
    assert sorted(p.name for p in rig.model.iterdir()) == ["model.index", "model.onnx", "voice.json"]


def test_a_new_model_replaces_the_previous_one(rig):
    rig.model.mkdir(parents=True)
    (rig.model / "model.onnx").write_bytes(b"previous model")
    (rig.model / "extra.txt").write_text("old")
    assert rig.run() == 0
    assert sorted(p.name for p in (rig.voice / "models").iterdir()) == ["rvc"]
    assert sorted(p.name for p in rig.model.iterdir()) == ["model.index", "model.onnx", "voice.json"]
    assert (rig.model / "model.onnx").stat().st_size > 1024 * 1024


def broken_export(kind):
    def export(c):
        out = Path(c.env["AIVOICE_EXPORT_DIR"])
        FakeRunner.export(c.runner, c)
        if kind == "exit code":
            return 1
        if kind == "small onnx":
            (out / "model.onnx").write_bytes(b"tiny")
        if kind == "bad json":
            (out / "voice.json").write_text("{")
        if kind == "no sample rate":
            (out / "voice.json").write_text(json.dumps({"name": "x"}))
        if kind == "no index":
            (out / "model.index").unlink()
        return 0

    return export


@pytest.mark.parametrize("kind", ["exit code", "small onnx", "bad json", "no sample rate", "no index"])
def test_a_failed_export_leaves_the_previous_model_and_the_training_output_untouched(rig, kind):
    rig.model.mkdir(parents=True)
    (rig.model / "model.onnx").write_bytes(b"previous model")
    (rig.model / "voice.json").write_text('{"name": "old", "sample_rate": 40000}')
    before = rig.tree(rig.voice / "models")
    rig.runner.stages["export"] = broken_export(kind)
    assert rig.run() == 1
    assert rig.final()["ok"] is False and rig.final()["error"]
    assert rig.tree(rig.voice / "models") == before
    assert rig.weights.exists() and rig.exp.is_dir(), "kept so that a resume can export again"
    del rig.runner.stages["export"]
    rig.runner.calls.clear()
    assert rig.run("--resume") == 0
    assert rig.runner.ran() == ["pitch", "features", "export"]


def test_a_swap_that_cannot_complete_puts_the_previous_model_back(rig, monkeypatch):
    rig.model.mkdir(parents=True)
    (rig.model / "model.onnx").write_bytes(b"previous model")
    real_replace = os.replace

    def replace(src, dst):
        if Path(src).name == "rvc.new":
            raise PermissionError("the folder is in use")
        real_replace(src, dst)

    monkeypatch.setattr(rig.worker.os, "replace", replace)
    assert rig.run() == 1
    assert rig.tree(rig.voice / "models") == {"rvc/model.onnx": b"previous model"}


def test_a_swap_interrupted_by_a_crash_is_repaired_at_the_next_start(rig):
    old = rig.voice / "models" / "rvc.old"
    old.mkdir(parents=True)
    (old / "model.onnx").write_bytes(b"previous model")
    rig.runner.stages["preprocess"] = lambda c: 1
    assert rig.run() == 1
    assert rig.tree(rig.voice / "models") == {"rvc/model.onnx": b"previous model"}


# ---------------------------------------------------------------- 11. clean-up

def test_success_removes_the_experiment_the_small_model_and_the_index_link(rig):
    other = rig.rvc / "assets" / "indices" / f"{EXP}0_added_IVF1_Flat_nprobe_1_{EXP}0_v2.index"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"another voice")
    assert rig.run() == 0
    assert not rig.exp.exists() and not rig.weights.exists()
    assert [p.name for p in (rig.rvc / "assets" / "indices").iterdir()] == [other.name]
    assert (rig.rvc / "logs" / "mute" / "0_gt_wavs" / "mute40k.wav").exists()


def test_failure_keeps_the_experiment_and_the_small_model(rig):
    rig.runner.stages["index"] = lambda c: 1
    assert rig.run() == 1
    assert rig.weights.exists() and (rig.exp / "G_2333333.pth").exists() and (rig.exp / "dataset.json").exists()


# ---------------------------------------------------------------- 12. logging

def test_every_output_line_is_logged_with_its_stage_and_no_transcript(rig):
    assert rig.run() == 0
    log = rig.log()
    assert "[preprocess] [Data slicing] Progress: 1/3 | 0001.orig.wav\n" in log
    assert "[pitch] Loading RMVPE model\n" in log
    assert "[features] [HuBERT features] Completed | Success: 3 | Skipped: 0 | Failed: 0\n" in log
    assert f"[train] INFO:{EXP}:loss_disc=5.255, loss_gen=3.198, loss_fm=10.025,loss_mel=25.093, loss_kl=9.000\n" in log
    assert "[index] [Index training] Write progress: 1/2\n" in log
    assert "[export] Exported " in log
    assert "secret transcript" not in log and "secret transcript" not in (rig.job / "progress.jsonl").read_text()


def test_a_fresh_run_starts_the_progress_file_again(rig):
    assert rig.run() == 0
    first = len(rig.progress())
    assert rig.run() == 0
    lines = rig.progress()
    assert len(lines) == first and [line.get("done", False) for line in lines].count(True) == 1


# ---------------------------------------------------------------- 13. robustness

@pytest.mark.parametrize("stage", ["preprocess", "train", "export"])
def test_an_exception_anywhere_still_ends_with_a_final_line(rig, stage):
    def explode(c):
        raise RuntimeError("the runner blew up")

    rig.runner.stages[stage] = explode
    assert rig.run() == 1
    final = rig.final()
    assert (final["done"], final["ok"], final["cancelled"]) == (True, False, False)
    assert "the runner blew up" in final["error"] and final["resumable"] is True
    assert final["stage"] == stage
    assert "Traceback" in rig.log()
    assert not (rig.job / "cancel").exists()


def test_a_failure_before_any_stage_still_ends_with_a_final_line(rig, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk trouble")

    monkeypatch.setattr(rig.worker.os, "link", explode)
    monkeypatch.setattr(rig.worker.shutil, "copy2", explode)
    assert rig.run() == 1
    assert "disk trouble" in rig.final()["error"] and rig.final()["stage"] == "prepare"


def test_the_command_line_matches_the_documented_interface(worker):
    args = worker.parse_args([
        "--voice-dir", "v", "--job-dir", "j", "--experiment", "voice-1", "--epochs", "200", "--batch-size", "8",
        "--save-every", "5", "--resume", "--parent-pid", "12", "--name", "A Name"])
    assert (args.epochs, args.batch_size, args.save_every, args.resume, args.parent_pid, args.name) == (
        200, 8, 5, True, 12, "A Name")
    assert worker.parse_args(["--voice-dir", "v", "--job-dir", "j", "--experiment", "e"]).resume is False


def test_the_worker_imports_only_the_standard_library_and_its_two_helpers():
    import ast

    names = set()
    for node in ast.walk(ast.parse(MODULE.read_text("utf8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names <= set(sys.stdlib_module_names)
