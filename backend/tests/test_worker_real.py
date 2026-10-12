"""Real end-to-end runs of trainer/worker.py on the GPU. Opt-in: they take minutes.

    AIVOICE_RUN_TRAIN_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_worker_real.py -q -W error -s

Skipped unless that variable is set and `trainer/verify.py --fast` passes. The voice lives in pytest's tmp_path
(never in data/voices). The experiment folder is under trainer/rvc/logs with the prefix "test-real-"; whatever
happens, the fixture ends every process that mentions the experiment and removes its folder, small model and
index links.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import psutil
import pytest

from aivoice import config
from aivoice.audiofile import write_wav

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "trainer"
RVC = TRAINER / "rvc"
PYTHON = TRAINER / ".venv" / "Scripts" / "python.exe"
VEC = config.data_dir() / "pretrained" / "vec-768-layer-12.onnx"
STAGES = ["prepare", "preprocess", "pitch", "features", "train", "index", "export"]
RUN_TIMEOUT = 45 * 60
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _environment_ready():
    if os.environ.get("AIVOICE_RUN_TRAIN_TEST") != "1":
        return "set AIVOICE_RUN_TRAIN_TEST=1 to run the real training tests"
    try:
        done = subprocess.run([sys.executable, str(TRAINER / "verify.py"), "--fast"], capture_output=True, text=True,
                              timeout=120, creationflags=NO_WINDOW)
        report = json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return f"trainer/verify.py --fast could not be run: {exc}"
    if not report.get("ok"):
        return "trainer/verify.py --fast failed: " + "; ".join(c["message"] for c in report["checks"] if not c["ok"])
    return None


_SKIP = _environment_ready()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


# ---------------------------------------------------------------- a synthetic voice

def _walk(rng, n, low, high, rate_hz, sr):
    points = rng.uniform(low, high, max(4, int(n / sr * rate_hz) + 4))
    return np.interp(np.linspace(0, len(points) - 1, n), np.arange(len(points)), points)


def speech_like(rng, seconds, sr=48000):
    """A voiced buzz with a wandering pitch, three moving formants and syllable-like gating (after the spike's
    make_dataset.py, with the formants applied to each harmonic instead of through scipy filters)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = _walk(rng, n, 100, 250, 1.5, sr) * (1 + 0.02 * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    formants = [(_walk(rng, n, 300, 900, 3, sr), 90, 1.0), (_walk(rng, n, 900, 2400, 3, sr), 120, 0.6),
                (_walk(rng, n, 2400, 3400, 2, sr), 180, 0.3)]
    y = np.zeros(n)
    for h in range(1, 40):
        freq = f0 * h
        gain = sum(g / (1 + ((freq - centre) / width) ** 2) for centre, width, g in formants)
        y += np.where(freq < 9000, np.sin(h * phase) * gain / h, 0.0)
    y += 0.01 * rng.standard_normal(n)
    envelope, pos = np.zeros(n), 0
    while pos < n:
        on, off = int(rng.uniform(0.12, 0.35) * sr), int(rng.uniform(0.03, 0.12) * sr)
        if rng.random() < 0.08:
            off += int(rng.uniform(0.2, 0.45) * sr)
        envelope[pos:pos + on] = rng.uniform(0.5, 1.0)
        pos += on + off
    window = np.hanning(int(0.03 * sr))
    y *= np.convolve(envelope, window / window.sum(), mode="same")
    return (y / (np.abs(y).max() + 1e-9) * rng.uniform(0.5, 0.9)).astype(np.float32)


def make_voice(folder, total_seconds=60.0, seed=0):
    clips = folder / "clips"
    clips.mkdir(parents=True)
    rng, done, number = np.random.default_rng(seed), 0.0, 0
    while done < total_seconds:
        seconds = min(float(rng.uniform(3, 8)), max(3.0, total_seconds - done))
        number += 1
        write_wav(clips / f"{number:04d}.orig.wav", speech_like(rng, seconds), 48000)
        (clips / f"{number:04d}.txt").write_text("a transcript the trainer must never read", encoding="utf-8")
        done += seconds
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(clips.iterdir())}


# ---------------------------------------------------------------- running the worker

def processes_of(experiment):
    """Every python.exe whose command line mentions the experiment (the worker, its launcher, every stage)."""
    found = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name, argv = (proc.info["name"] or "").lower(), proc.info["cmdline"] or []
            if "python" in name and any(experiment in part for part in argv):
                found.append(proc)
        except psutil.Error:
            pass
    return found


def wait_until_none_left(experiment, seconds=30):
    _, alive = psutil.wait_procs(processes_of(experiment), timeout=seconds)
    return [(p.pid, p.cmdline()[:3]) for p in alive if p.is_running()] + [
        (p.pid, "appeared late") for p in processes_of(experiment)]


class Real:
    def __init__(self, tmp_path):
        self.experiment = "test-real-" + uuid.uuid4().hex[:8]
        self.voice = tmp_path / "voices" / "real"
        self.job = self.voice / "training"
        self.exp = RVC / "logs" / self.experiment
        self.weights = RVC / "assets" / "weights" / f"{self.experiment}.pth"
        self.recordings = make_voice(self.voice)

    def command(self, epochs, *extra):
        return [str(PYTHON), "-P", str(TRAINER / "worker.py"), "--voice-dir", str(self.voice),
                "--job-dir", str(self.job), "--experiment", self.experiment, "--epochs", str(epochs),
                "--batch-size", "4", "--save-every", "1", "--parent-pid", str(os.getpid()), "--name=Real test voice",
                *extra]

    def start(self, epochs, *extra):
        self.job.mkdir(parents=True, exist_ok=True)
        out = open(self.job / "worker.out.log", "ab")
        try:
            return subprocess.Popen(self.command(epochs, *extra), stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        finally:
            out.close()

    def progress(self):
        path = self.job / "progress.jsonl"
        if not path.exists():
            return []
        lines = []
        for raw in path.read_text("utf-8").splitlines():
            try:
                lines.append(json.loads(raw))
            except ValueError:
                pass  # the line being written right now
        return lines

    def log(self):
        return (self.job / "worker.log").read_text("utf-8", errors="replace")

    def wait_for(self, process, condition, what, seconds=RUN_TIMEOUT):
        """Wait until condition(progress lines) holds. Fails at once if the worker ends first."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            lines = self.progress()
            if condition(lines):
                return lines
            if process.poll() is not None:
                pytest.fail(f"the worker ended (exit code {process.returncode}) before {what}:\n{self.diagnosis()}")
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        pytest.fail(f"timed out waiting for {what}:\n{self.diagnosis()}")

    def diagnosis(self):
        tail = "\n".join(self.log().splitlines()[-40:]) if (self.job / "worker.log").exists() else "(no worker.log)"
        out = (self.job / "worker.out.log").read_text("utf-8", errors="replace")[-2000:]
        return f"last progress line: {self.progress()[-1:]}\nworker.log tail:\n{tail}\nworker.out.log:\n{out}"

    def clean_up(self):
        for proc in processes_of(self.experiment):
            try:
                proc.kill()
            except psutil.Error:
                pass
        wait_until_none_left(self.experiment, 20)
        shutil.rmtree(self.exp, ignore_errors=True)
        self.weights.unlink(missing_ok=True)
        for link in (RVC / "assets" / "indices").glob(f"{self.experiment}_*"):
            link.unlink(missing_ok=True)

    def left_under_trainer(self):
        return [str(p) for p in (self.exp, self.weights, *(RVC / "assets" / "indices").glob(f"{self.experiment}_*"))
                if p.exists()]


@pytest.fixture
def real(tmp_path, request):
    rig = Real(tmp_path)
    try:
        yield rig
        now = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes())
               for p in sorted((rig.voice / "clips").iterdir())}
        assert now == rig.recordings, "the recordings must be untouched"
    finally:
        rig.clean_up()
        keep = os.environ.get("AIVOICE_TRAIN_TEST_KEEP")  # a folder to copy the run's logs to, for the notes
        if keep:
            Path(keep).mkdir(parents=True, exist_ok=True)
            for name in ("progress.jsonl", "worker.log", "worker.out.log"):
                if (rig.job / name).exists():
                    (Path(keep) / f"{request.node.name}.{name}").write_bytes((rig.job / name).read_bytes())


def assert_finished_ok(real, lines):
    final = lines[-1]
    assert final.get("done") is True and final.get("ok") is True, real.diagnosis()
    assert final["error"] is None and final["cancelled"] is False and final["percent"] == 100
    assert sorted(p.name for p in (real.voice / "models").iterdir()) == ["rvc"]
    model = real.voice / "models" / "rvc"
    assert sorted(p.name for p in model.iterdir()) == ["model.index", "model.onnx", "voice.json"]
    assert (model / "model.onnx").stat().st_size > 1024 * 1024
    assert json.loads((model / "voice.json").read_text()) == {"name": "Real test voice", "sample_rate": 40000}
    assert real.left_under_trainer() == [], "the experiment, small model and index link are removed after success"
    assert "a transcript the trainer must never read" not in real.log()


def assert_model_converts(real):
    from aivoice.engines.rvc import RvcEngine, make_session
    from aivoice.presets import PRESETS

    model = real.voice / "models" / "rvc"
    device_id = config.load_settings(config.data_dir())["dml_device_id"]
    engine = RvcEngine(model / "model.onnx", VEC, 40000, model / "model.index",
                       session_factory=partial(make_session, device_id=device_id))
    sr = config.SAMPLE_RATE
    audio = speech_like(np.random.default_rng(7), 1.0, sr)
    out = engine.convert(audio, sr, 0.0, next(iter(PRESETS.values())))
    assert out.shape == audio.shape and np.all(np.isfinite(out)) and np.max(np.abs(out)) > 1e-4


# ---------------------------------------------------------------- the tests

def test_a_real_two_epoch_run_trains_exports_and_converts(real):
    process = real.start(2)
    code = process.wait(timeout=RUN_TIMEOUT)
    lines = real.progress()
    assert code == 0, real.diagnosis()
    assert_finished_ok(real, lines)
    percents = [line["percent"] for line in lines]
    assert percents == sorted(percents)
    assert [s for s in STAGES if any(line["stage"] == s for line in lines)] == STAGES
    epochs = [line["epoch"] for line in lines if line["stage"] == "train"]
    assert 1 in epochs and 2 in epochs
    assert wait_until_none_left(real.experiment) == []
    assert_model_converts(real)


def test_a_real_run_can_be_cancelled_and_resumed(real):
    process = real.start(6)
    real.wait_for(process, lambda lines: any(line["stage"] == "train" and (line["epoch"] or 0) >= 1 for line in lines),
                  "the first finished epoch")
    (real.job / "cancel").write_text("")
    asked = time.monotonic()
    assert process.wait(timeout=180) == 2, real.diagnosis()
    print(f"\ncancel took {time.monotonic() - asked:.1f} s")
    final = real.progress()[-1]
    assert final["done"] is True and final["cancelled"] is True and final["ok"] is False
    assert final.get("resumable") is True
    assert wait_until_none_left(real.experiment) == [], "no trainer process may remain after a cancel"
    assert not (real.job / "cancel").exists()
    assert (real.exp / "G_2333333.pth").is_file() and (real.exp / "D_2333333.pth").is_file()
    cancelled_at = final["epoch"]
    first_run_lines = len(real.progress())

    process = real.start(6, "--resume")
    code = process.wait(timeout=RUN_TIMEOUT)
    lines = real.progress()
    assert code == 0, real.diagnosis()
    assert_finished_ok(real, lines)
    resumed = lines[first_run_lines:]
    assert not any(line["message"] == "Slicing the clips" for line in resumed), "slicing is not done twice"
    assert any(line["message"].startswith("Continuing from epoch") for line in resumed), "the resumed event"
    assert real.log().count("Loaded checkpoint") >= 2
    assert f"Continuing from epoch {cancelled_at}" in [line["message"] for line in resumed]
    assert "epoch_start" not in real.log() and "Loaded generator pretrained model" not in real.log().split(
        "stop requested")[-1], "the resumed run must not start again from the pretrained base"
    assert wait_until_none_left(real.experiment) == []
    assert_model_converts(real)


def test_killing_the_worker_itself_leaves_no_stage_running_and_a_resume_works(real):
    process = real.start(6)
    real.wait_for(process, lambda lines: any(line["stage"] == "train" and (line["epoch"] or 0) >= 2 for line in lines),
                  "the second finished epoch")
    deadline = time.monotonic() + 60
    while not all((real.exp / name).is_file() for name in ("G_2333333.pth.bak", "D_2333333.pth.bak")):
        assert time.monotonic() < deadline, "no checkpoint backup appeared"
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.5)
    before = processes_of(real.experiment)
    workers = [p for p in before if any(a.endswith("worker.py") for a in p.cmdline())]
    stages = [p for p in before if p not in workers]
    assert workers and stages, [p.cmdline()[:3] for p in before]
    real_worker = max(workers, key=lambda p: p.create_time())  # the launcher shim started first; this is Python
    print(f"\nbefore the kill: {len(workers)} worker processes, {len(stages)} stage processes")
    real_worker.kill()  # TerminateProcess on that one process only, as a crash would be
    assert wait_until_none_left(real.experiment, 60) == [], "the job object must end every stage with the worker"
    process.wait(timeout=30)
    assert not real.progress()[-1].get("done"), "a killed worker cannot write a final line"

    process = real.start(6, "--resume")
    code = process.wait(timeout=RUN_TIMEOUT)
    assert code == 0, real.diagnosis()
    assert_finished_ok(real, real.progress())
    assert "Loaded checkpoint" in real.log()
    assert wait_until_none_left(real.experiment) == []
