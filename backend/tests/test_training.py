import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aivoice import training
from aivoice.library import Library
from aivoice.training import TrainingError, TrainingManager

PASS = {"ok": True, "checks": [{"id": "venv", "ok": True, "message": "trainer Python found"}]}
FAIL = {"ok": False, "checks": [
    {"id": "venv", "ok": True, "message": "trainer Python found"},
    {"id": "torch_gpu", "ok": False, "message": "torch sees no GPU"},
    {"id": "ffmpeg", "ok": False, "message": "ffmpeg not found on PATH"},
]}
CLIPS_CHANGED = "The clips changed since this training run started. Start a new training run."


class FakeHandle:
    def __init__(self, pid):
        self.pid = pid
        self.code = None
        self.kills = 0
        self.kill_error = None

    def poll(self):
        return self.code

    def kill_tree(self):
        self.kills += 1
        if self.kill_error is not None:
            raise self.kill_error
        self.code = 1


class Rig:
    """A manager on a temporary data folder, with every outside effect faked and recorded."""

    def __init__(self, tmp_path):
        self.data = tmp_path / "data"
        self.trainer = tmp_path / "AI Voice" / "trainer"
        self.library = Library(self.data)
        self.launches = []  # (argv, log_path)
        self.handles = []
        self.launch_error = None
        self.report = PASS
        self.verify_calls = []  # the fast flag of each call
        self.alive = set()
        self.workers = {}  # pid -> experiment it is really a worker for
        self.killed = []
        self.clock = 1000.0
        self.manager = None

    def launch(self, argv, log_path):
        if self.launch_error is not None:
            raise self.launch_error
        self.launches.append((argv, log_path))
        handle = FakeHandle(4000 + len(self.handles))
        self.handles.append(handle)
        return handle

    def verify(self, fast):
        self.verify_calls.append(fast)
        return self.report

    def build(self):
        self.manager = TrainingManager(
            self.data, self.library, self.trainer, launch=self.launch, verify=self.verify,
            pid_alive=lambda pid: pid in self.alive, now=lambda: self.clock,
            is_worker=lambda pid, experiment: self.workers.get(pid) == experiment, kill_pid=self.killed.append)
        return self.manager

    def voice(self, name="Narrator", minutes=6.0):
        voice = self.library.create(name)
        path = self.data / "voices" / voice.id / "voice.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["clips"] = [{"index": 1, "text": "", "seconds": minutes * 60.0, "source": "recorded"}]
        raw["next_index"] = 2
        path.write_text(json.dumps(raw), encoding="utf-8")
        return voice.id

    def job_dir(self, voice_id):
        return self.data / "voices" / voice_id / "training"

    def record(self, voice_id):
        return json.loads((self.job_dir(voice_id) / "job.json").read_text(encoding="utf-8"))

    def write_record(self, voice_id, **fields):
        record = {"voice_id": voice_id, "experiment": f"voice-{voice_id}", "epochs": 200, "batch_size": 8,
                  "state": "failed", "started": 500.0, "finished": 600.0, "worker_pid": None, "error": None, **fields}
        self.job_dir(voice_id).mkdir(parents=True, exist_ok=True)
        (self.job_dir(voice_id) / "job.json").write_text(json.dumps(record), encoding="utf-8")

    def experiment_dir(self, voice_id):
        return self.trainer / "rvc" / "logs" / f"voice-{voice_id}"

    def checkpoint(self, voice_id, suffix=""):
        folder = self.experiment_dir(voice_id)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("G_2333333.pth", "D_2333333.pth"):
            (folder / (name + suffix)).write_bytes(b"x")

    def progress(self, voice_id, *lines, raw=None):
        with open(self.job_dir(voice_id) / "progress.jsonl", "ab") as f:
            for line in lines:
                f.write((json.dumps(line) + "\n").encode("utf-8"))
            if raw is not None:
                f.write(raw)

    def finish(self, voice_id, code, **final):
        if final:
            self.progress(voice_id, {**line(percent=final.pop("percent", 50.0)), "done": True, "ok": False,
                                     "error": None, "cancelled": False, **final})
        self.handles[-1].code = code


def line(stage="train", stage_index=5, percent=40.0, epoch=None, epochs=None, eta=None, message="Training"):
    return {"time": 1.0, "stage": stage, "stage_index": stage_index, "stages": 7, "percent": percent,
            "epoch": epoch, "epochs": epochs, "eta_seconds": eta, "message": message}


@pytest.fixture
def rig(tmp_path):
    rig = Rig(tmp_path)
    rig.build()
    return rig


# 1. Validation


def test_start_refuses_in_the_documented_order_and_leaves_nothing_behind(rig):
    short = rig.voice("Short", minutes=4.96)
    rig.report = FAIL
    m = rig.manager
    with pytest.raises(TrainingError, match="Unknown voice"):
        m.start("no-such-voice", epochs=5, batch_size=3)
    with pytest.raises(TrainingError, match="Unknown voice"):
        m.start("../escape")
    with pytest.raises(TrainingError, match="between 20 and 1000"):
        m.start(short, epochs=19, batch_size=3)
    with pytest.raises(TrainingError, match="between 20 and 1000"):
        m.start(short, epochs=1001)
    with pytest.raises(TrainingError, match="between 20 and 1000"):
        m.start(short, epochs=True)
    with pytest.raises(TrainingError, match="4, 8 or 16"):
        m.start(short, batch_size=3)
    with pytest.raises(TrainingError) as too_short:
        m.start(short)
    assert str(too_short.value) == "This voice has 4.9 minutes of clean audio. Training needs at least 5."
    assert rig.verify_calls == []  # the slow environment check only runs once everything else is fine

    ready = rig.voice("Ready")
    with pytest.raises(TrainingError) as broken:
        m.start(ready)
    assert "torch sees no GPU" in str(broken.value) and "ffmpeg not found on PATH" in str(broken.value)
    assert "trainer Python found" not in str(broken.value)

    assert rig.launches == []
    assert not rig.job_dir(short).exists() and not rig.job_dir(ready).exists()
    assert m.running is False and m.status()["state"] == "idle"


def test_exactly_five_minutes_is_enough(rig):
    rig.manager.start(rig.voice(minutes=5.0))
    assert rig.manager.running is True


def test_a_second_start_is_refused_before_the_minutes_check(rig):
    first, short = rig.voice("One"), rig.voice("Short", minutes=1.0)
    rig.manager.start(first)
    with pytest.raises(TrainingError, match="Training is already running"):
        rig.manager.start(short)
    with pytest.raises(TrainingError, match="Training is already running"):
        rig.manager.start(first)
    assert len(rig.launches) == 1


# 2. Launch


def test_start_launches_the_worker_and_writes_the_job_record(rig):
    voice = rig.voice("Dad's voice")
    status = rig.manager.start(voice, epochs=120, batch_size=16)

    (argv, log_path), = rig.launches
    voice_dir = rig.data / "voices" / voice
    assert argv == [
        str(rig.trainer / ".venv" / "Scripts" / "python.exe"), "-P", str(rig.trainer / "worker.py"),
        "--voice-dir", str(voice_dir), "--job-dir", str(voice_dir / "training"),
        "--experiment", f"voice-{voice}", "--epochs", "120", "--batch-size", "16", "--save-every", "5",
        "--parent-pid", str(os.getpid()), "--name=Dad's voice",
    ]
    assert all(isinstance(a, str) for a in argv)
    assert log_path == voice_dir / "training" / "worker.log"
    assert rig.record(voice) == {
        "voice_id": voice, "experiment": f"voice-{voice}", "epochs": 120, "batch_size": 16, "state": "running",
        "started": 1000.0, "finished": None, "worker_pid": 4000, "error": None}
    assert status["state"] == "running" and status["voice_id"] == voice
    assert status["epochs"] == 120 and status["percent"] == 0.0 and status["started"] == 1000.0
    assert rig.manager.running is True
    assert set(status) == {"state", "voice_id", "stage", "stage_index", "stages", "epoch", "epochs", "percent",
                           "eta_seconds", "message", "error", "started", "finished", "can_resume"}


def test_defaults_are_200_epochs_and_batch_8(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    assert (rig.record(voice)["epochs"], rig.record(voice)["batch_size"]) == (200, 8)


def test_a_worker_that_cannot_be_started_is_a_readable_error_and_keeps_the_last_record(rig):
    voice = rig.voice()
    rig.write_record(voice, state="cancelled")
    rig.build()
    rig.launch_error = FileNotFoundError(2, "The system cannot find the file specified")
    with pytest.raises(TrainingError, match="Could not start training"):
        rig.manager.start(voice)
    assert rig.record(voice)["state"] == "cancelled"
    assert rig.manager.running is False


def test_a_record_that_cannot_be_saved_stops_the_worker_it_just_started(rig, monkeypatch):
    voice = rig.voice()

    def full_disk(path, obj):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(training, "_atomic_json", full_disk)
    with pytest.raises(TrainingError, match="No space left"):
        rig.manager.start(voice)
    assert rig.handles[0].kills == 1
    assert rig.manager.running is False


# 3. Progress


def test_status_follows_the_progress_file_incrementally(rig):
    voice = rig.voice()
    m = rig.manager
    m.start(voice)
    assert m.status()["stage"] is None  # no progress file yet

    rig.progress(voice, line("preprocess", 2, 2.5, message="Slicing clips"))
    s = m.status()
    assert (s["stage"], s["stage_index"], s["stages"], s["percent"], s["message"]) == ("preprocess", 2, 7, 2.5, "Slicing clips")
    assert s["epoch"] is None and s["eta_seconds"] is None and s["epochs"] == 200

    full = json.dumps(line("train", 5, 30.0, epoch=40, epochs=200, eta=3600.5, message="Epoch 40 of 200"))
    rig.progress(voice, raw=b"this is not json\n[1, 2]\n\xff\xfe\n" + full[:25].encode())
    assert m.status()["stage"] == "preprocess"  # garbage skipped, the half-written line is left for later

    rig.progress(voice, raw=full[25:].encode() + b"\n")
    s = m.status()
    assert (s["stage"], s["epoch"], s["percent"], s["eta_seconds"], s["message"]) == ("train", 40, 30.0, 3600.5, "Epoch 40 of 200")
    assert s["state"] == "running"


def test_status_only_reads_what_was_appended(rig, monkeypatch):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.progress(voice, line(percent=10.0))
    rig.manager.status()
    seen = []
    real = training.json.loads
    monkeypatch.setattr(training.json, "loads", lambda text, *a, **k: (seen.append(text), real(text, *a, **k))[1])
    rig.manager.status()
    rig.progress(voice, line(percent=11.0))
    rig.manager.status()
    assert len(seen) == 1


def test_wrongly_typed_progress_values_are_ignored(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.progress(voice, line("pitch", 3, 5.0), {"stage": 7, "stage_index": "x", "percent": "lots", "epoch": [1],
                                                "eta_seconds": "soon", "message": None, "stages": None})
    s = rig.manager.status()
    assert (s["stage"], s["stage_index"], s["percent"], s["stages"], s["message"]) == ("pitch", 3, 5.0, 7, "Training")


def test_a_fresh_run_never_shows_the_previous_runs_last_lines(rig):
    voice = rig.voice()
    rig.write_record(voice, state="done")
    rig.progress(voice, {**line(percent=100.0, message="Finished"), "done": True, "ok": True, "error": None, "cancelled": False})
    rig.build()
    assert rig.manager.status(voice)["percent"] == 100.0
    s = rig.manager.start(voice)
    assert (s["state"], s["percent"], s["message"]) == ("running", 0.0, None)
    assert rig.manager.status()["state"] == "running"


def test_status_survives_a_voice_folder_that_disappears_while_training(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.library.delete(voice)
    assert rig.manager.status()["state"] == "running"
    rig.handles[0].code = 1
    # The record can no longer be saved, which must not raise; a voice that is gone is then forgotten.
    assert rig.manager.status(voice)["state"] == "idle"
    assert rig.manager.running is False
    assert rig.manager.status()["state"] == "idle"


def test_status_never_raises(rig, monkeypatch):
    voice = rig.voice()
    rig.manager.start(voice)

    def boom():
        raise RuntimeError("poll failed")

    rig.handles[0].poll = boom
    assert rig.manager.status()["state"] == "running"
    assert rig.manager.status(12345)["state"] == "idle"
    assert rig.manager.status("../x")["state"] == "idle"
    monkeypatch.setattr(rig.manager, "_snapshot", lambda *a: 1 / 0)
    assert rig.manager.status()["state"] == "idle"


# 4. Exit


def test_a_successful_exit_is_done(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.clock = 5000.0
    rig.finish(voice, 0, ok=True, percent=100.0)
    s = rig.manager.status()
    assert (s["state"], s["finished"], s["error"], s["percent"], s["can_resume"]) == ("done", 5000.0, None, 100.0, False)
    assert rig.record(voice)["state"] == "done" and rig.record(voice)["finished"] == 5000.0
    assert rig.manager.running is False


def test_a_cancelled_exit_is_cancelled(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.finish(voice, 2, cancelled=True, error="The app was closed")
    assert rig.manager.status()["state"] == "cancelled"
    assert rig.record(voice)["state"] == "cancelled"


def test_a_failed_exit_carries_the_workers_error(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.finish(voice, 1, error="Preprocessing failed for 2 clips, first 0003.orig.wav")
    s = rig.manager.status()
    assert (s["state"], s["error"]) == ("failed", "Preprocessing failed for 2 clips, first 0003.orig.wav")
    assert rig.record(voice)["error"] == s["error"] and s["finished"] == 1000.0 and s["eta_seconds"] is None


def test_an_exit_without_a_final_line_is_an_unexpected_stop(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.progress(voice, line(percent=12.0), raw=b'{"done": true, "ok": tr')
    rig.handles[0].code = -1073741819
    s = rig.manager.status()
    assert s["state"] == "failed"
    assert s["error"] == "The training process stopped unexpectedly. See training/worker.log."
    assert s["percent"] == 12.0


def test_the_final_line_is_read_even_when_it_lands_just_before_the_exit(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    handle = rig.handles[0]

    def poll():  # the worker writes its last line and exits between two status calls
        rig.finish(voice, 0, ok=True)
        handle.poll = lambda: 0
        return 0

    handle.poll = poll
    assert rig.manager.status()["state"] == "done"


def test_a_worker_that_hangs_after_its_final_line_is_stopped(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.progress(voice, {**line(percent=100.0), "done": True, "ok": True, "error": None, "cancelled": False})
    assert rig.manager.status()["state"] == "running"
    rig.clock += 29
    assert rig.manager.status()["state"] == "running" and rig.handles[0].kills == 0
    rig.clock += 1
    assert rig.manager.status()["state"] == "done" and rig.handles[0].kills == 1


# 5. Cancel


def test_cancel_asks_the_worker_then_kills_it_after_90_seconds(rig):
    voice = rig.voice()
    m = rig.manager
    m.start(voice)
    rig.progress(voice, line(percent=33.0))
    s = m.cancel()
    assert s["state"] == "cancelling" and (rig.job_dir(voice) / "cancel").is_file()
    assert rig.record(voice)["state"] == "cancelling"
    assert m.running is True

    rig.clock += 60
    assert m.cancel()["state"] == "cancelling"  # asking twice does not restart the 90 seconds
    rig.clock += 29
    assert m.status()["state"] == "cancelling" and rig.handles[0].kills == 0
    rig.clock += 1
    s = m.status()
    assert s["state"] == "cancelled" and rig.handles[0].kills == 1 and s["finished"] == 1090.0
    assert rig.record(voice)["state"] == "cancelled"
    assert not (rig.job_dir(voice) / "cancel").exists()
    assert m.running is False and s["percent"] == 33.0


def test_cancel_with_nothing_running(rig):
    voice = rig.voice()
    with pytest.raises(TrainingError, match="Training is not running"):
        rig.manager.cancel()
    rig.manager.start(voice)
    rig.finish(voice, 0, ok=True)
    with pytest.raises(TrainingError, match="Training is not running"):
        rig.manager.cancel()


def test_a_worker_that_stops_by_itself_after_cancel_is_cancelled_without_a_kill(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.manager.cancel()
    rig.clock += 10
    rig.finish(voice, 2, cancelled=True)
    assert rig.manager.status()["state"] == "cancelled" and rig.handles[0].kills == 0


def test_a_worker_that_dies_during_cancel_without_a_final_line_is_cancelled(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.manager.cancel()
    rig.handles[0].code = 1
    s = rig.manager.status()
    assert (s["state"], s["error"]) == ("cancelled", None)


def test_a_run_that_finishes_while_being_cancelled_is_done(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.manager.cancel()
    rig.finish(voice, 0, ok=True)
    assert rig.manager.status()["state"] == "done"


def test_a_kill_that_fails_is_retried_and_the_job_stays_cancelling(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.manager.cancel()
    rig.clock += 90
    rig.handles[0].kill_error = RuntimeError("access denied")
    assert rig.manager.status()["state"] == "cancelling" and rig.manager.running is True
    rig.handles[0].kill_error = None
    assert rig.manager.status()["state"] == "cancelled" and rig.handles[0].kills == 3  # status, running, status


# 6. Resume


@pytest.mark.parametrize("state", ["failed", "cancelled", "interrupted"])
def test_resume_continues_a_stopped_run_with_its_settings(rig, state):
    voice = rig.voice()
    rig.write_record(voice, state=state, epochs=300, batch_size=4, error="it broke" if state == "failed" else None)
    rig.checkpoint(voice)
    rig.progress(voice, {**line(percent=44.0), "done": True, "ok": False, "error": "it broke", "cancelled": False})
    rig.build()
    assert rig.manager.status(voice)["can_resume"] is True
    assert rig.manager.status(voice)["percent"] == 44.0

    s = rig.manager.start(voice, resume=True)
    argv = rig.launches[0][0]
    assert argv[argv.index("--epochs") + 1] == "300" and argv[argv.index("--batch-size") + 1] == "4"
    assert argv[argv.index("--save-every") + 2] == "--resume"
    assert (s["state"], s["error"], s["can_resume"], s["started"], s["finished"]) == ("running", None, False, 1000.0, None)
    assert rig.manager.status()["state"] == "running"  # the old final line is not read as this run's

    rig.progress(voice, line(percent=45.0))
    assert rig.manager.status()["percent"] == 45.0


def test_resume_may_change_the_epochs_but_not_the_batch_size(rig):
    voice = rig.voice()
    rig.write_record(voice, state="cancelled", epochs=300, batch_size=4)
    rig.checkpoint(voice)
    rig.build()
    with pytest.raises(TrainingError, match="batch size"):
        rig.manager.start(voice, batch_size=8, resume=True)
    assert rig.launches == []
    rig.manager.start(voice, epochs=400, batch_size=4, resume=True)
    argv = rig.launches[0][0]
    assert argv[argv.index("--epochs") + 1] == "400" and argv[argv.index("--batch-size") + 1] == "4"


def test_resume_uses_the_backup_checkpoint_when_only_that_exists(rig):
    voice = rig.voice()
    rig.write_record(voice, state="interrupted")
    rig.checkpoint(voice, suffix=".bak")
    rig.build()
    assert rig.manager.status(voice)["can_resume"] is True
    rig.manager.start(voice, resume=True)
    assert "--resume" in rig.launches[0][0]


def test_resume_is_refused_when_there_is_nothing_to_resume(rig):
    message = "There is no interrupted training run to resume for this voice"
    never, finished, no_checkpoint, half = rig.voice("a"), rig.voice("b"), rig.voice("c"), rig.voice("d")
    rig.write_record(finished, state="done")
    rig.checkpoint(finished)
    rig.write_record(no_checkpoint, state="failed")
    rig.write_record(half, state="failed")
    rig.experiment_dir(half).mkdir(parents=True)
    (rig.experiment_dir(half) / "G_2333333.pth").write_bytes(b"x")
    rig.build()
    for voice in (never, finished, no_checkpoint, half):
        with pytest.raises(TrainingError, match=message):
            rig.manager.start(voice, resume=True)
        assert rig.manager.status(voice)["can_resume"] is False
    assert rig.launches == [] and rig.verify_calls == []
    assert not rig.job_dir(never).exists()


def test_a_run_whose_clips_changed_fails_with_the_workers_message_and_cannot_be_resumed_again(rig):
    voice = rig.voice()
    rig.write_record(voice, state="interrupted")
    rig.checkpoint(voice)
    rig.build()
    rig.manager.start(voice, resume=True)
    rig.finish(voice, 1, error=CLIPS_CHANGED)
    s = rig.manager.status()
    assert (s["state"], s["error"], s["can_resume"]) == ("failed", CLIPS_CHANGED, False)
    with pytest.raises(TrainingError, match="no interrupted training run"):
        rig.manager.start(voice, resume=True)
    rig.manager.start(voice)  # a fresh run is still allowed
    assert "--resume" not in rig.launches[-1][0]


def test_a_failed_run_with_a_checkpoint_can_be_resumed(rig):
    voice = rig.voice()
    rig.manager.start(voice)
    rig.checkpoint(voice)
    assert rig.manager.status()["can_resume"] is False  # not while it runs
    rig.finish(voice, 1, error="The GPU driver was reset")
    assert rig.manager.status()["can_resume"] is True


# 7. Recovery at start-up


def test_a_running_job_whose_worker_is_gone_becomes_interrupted(rig):
    voice, other = rig.voice("a"), rig.voice("b")
    rig.write_record(voice, state="running", finished=None, worker_pid=777)
    rig.write_record(other, state="cancelling", finished=None, worker_pid=None)
    (rig.job_dir(voice) / "cancel").write_bytes(b"")
    rig.checkpoint(voice)
    rig.clock = 2000.0
    rig.build()
    assert rig.killed == []
    for v in (voice, other):
        assert rig.record(v)["state"] == "interrupted" and rig.record(v)["finished"] == 2000.0
        assert rig.manager.status(v)["state"] == "interrupted"
    assert rig.manager.status(voice)["can_resume"] is True
    assert not (rig.job_dir(voice) / "cancel").exists()
    assert rig.manager.running is False


def test_a_worker_left_alive_by_a_dead_backend_is_killed_not_adopted(rig):
    voice = rig.voice()
    rig.write_record(voice, state="running", finished=None, worker_pid=777)
    rig.alive.add(777)
    rig.workers[777] = f"voice-{voice}"
    rig.build()
    assert rig.killed == [777]
    assert rig.record(voice)["state"] == "interrupted"
    assert rig.manager.running is False and rig.manager.status()["state"] == "interrupted"


def test_a_live_pid_that_is_not_this_voices_worker_is_never_killed(rig):
    voice = rig.voice()
    rig.write_record(voice, state="running", finished=None, worker_pid=777)
    rig.alive.add(777)
    rig.workers[777] = "voice-someone-else"  # the pid was reused after a restart of the PC
    rig.build()
    assert rig.killed == []
    assert rig.record(voice)["state"] == "interrupted"


def test_recovery_survives_damaged_records_and_failing_checks(rig, tmp_path):
    good, bad, weird, crash = rig.voice("a"), rig.voice("b"), rig.voice("c"), rig.voice("d")
    rig.write_record(good, state="running", finished=None, worker_pid=777)
    rig.job_dir(bad).mkdir()
    (rig.job_dir(bad) / "job.json").write_text("{not json", encoding="utf-8")
    rig.write_record(weird, state="running", finished=None, worker_pid="777; calc.exe", epochs="many")
    rig.write_record(crash, state="running", finished=None, worker_pid=888)
    (rig.data / "voices" / "Not A Voice").mkdir()
    (rig.data / "voices" / "stray.txt").write_text("x")
    rig.alive.update({777, 888})

    def is_worker(pid, experiment):
        raise OSError("access denied")

    m = TrainingManager(rig.data, rig.library, rig.trainer, launch=rig.launch, verify=rig.verify,
                        pid_alive=lambda pid: pid in rig.alive, now=lambda: rig.clock,
                        is_worker=is_worker, kill_pid=rig.killed.append)
    assert rig.killed == []
    assert rig.record(good)["state"] == "interrupted" and rig.record(crash)["state"] == "interrupted"
    assert rig.record(weird)["state"] == "interrupted"
    assert m.status(bad)["state"] == "idle"
    assert (rig.job_dir(bad) / "job.json").read_text(encoding="utf-8") == "{not json"
    assert m.status(weird)["epochs"] is None

    empty = TrainingManager(tmp_path / "nothing", Library(tmp_path / "nothing"), rig.trainer)
    assert empty.status()["state"] == "idle"


def test_a_finished_job_is_still_reported_after_a_restart(rig):
    voice = rig.voice()
    rig.write_record(voice, state="done", started=10.0, finished=20.0)
    rig.build()
    s = rig.manager.status()
    assert (s["state"], s["voice_id"], s["started"], s["finished"]) == ("done", voice, 10.0, 20.0)


# 8. One job at a time


def test_status_reports_per_voice_or_the_current_job(rig):
    a, b, c = rig.voice("a"), rig.voice("b"), rig.voice("c")
    m = rig.manager
    m.start(a)
    rig.clock = 1100.0
    rig.finish(a, 1, error="first failed")
    assert m.status()["voice_id"] == a

    rig.clock = 1200.0
    m.start(b)
    assert (m.status()["voice_id"], m.status()["state"]) == (b, "running")
    assert (m.status(a)["state"], m.status(a)["error"]) == ("failed", "first failed")
    idle = m.status(c)
    assert (idle["state"], idle["voice_id"], idle["can_resume"], idle["percent"]) == ("idle", c, False, 0.0)

    rig.clock = 1300.0
    rig.finish(b, 0, ok=True)
    assert (m.status()["voice_id"], m.status()["state"]) == (b, "done")
    m.start(c)
    assert m.status()["voice_id"] == c and len(rig.launches) == 3


# 9. Environment


def test_environment_reports_ready_and_caches_only_a_passing_report(rig):
    m = rig.manager
    rig.report = FAIL
    env = m.environment()
    assert env["ready"] is False and env["ok"] is False and env["checks"] == FAIL["checks"]
    rig.report = PASS  # the user fixed the setup
    env = m.environment()
    assert env == {**PASS, "ready": True}
    assert rig.verify_calls == [False, False]
    rig.report = FAIL
    assert m.environment()["ready"] is True and rig.verify_calls == [False, False]  # cached
    assert m.environment(refresh=True)["ready"] is False
    assert m.environment()["ready"] is False  # and the stale pass is gone
    assert rig.verify_calls == [False, False, False, False]


def test_start_uses_the_full_check_once_and_a_fast_recheck_after(rig):
    a, b = rig.voice("a"), rig.voice("b")
    rig.manager.start(a)
    rig.finish(a, 0, ok=True)
    rig.manager.status()
    rig.manager.start(b)
    assert rig.verify_calls == [False, True]
    rig.finish(b, 0, ok=True)
    rig.report = FAIL  # something was removed since the full check passed
    with pytest.raises(TrainingError, match="ffmpeg not found"):
        rig.manager.start(a)
    assert rig.manager.environment()["ready"] is False


@pytest.mark.parametrize("report", [None, "nope", {"ok": True}, {"ok": "yes", "checks": []}, {"checks": [1]}])
def test_a_nonsense_or_crashing_verify_is_a_failed_environment(rig, report):
    rig.report = report
    env = rig.manager.environment()
    assert env["ready"] is False and env["ok"] is False and env["checks"][0]["ok"] is False

    def crash(fast):
        raise RuntimeError("no python")

    rig.manager._verify = crash
    assert rig.manager.environment()["ready"] is False
    with pytest.raises(TrainingError, match="no python"):
        rig.manager.start(rig.voice())


def test_status_is_not_blocked_while_the_environment_is_checked(rig):
    voice, other = rig.voice("a"), rig.voice("b")
    seen = []

    def slow_verify(fast):
        seen.append(rig.manager.status()["state"])  # would deadlock or wait if start held the lock
        seen.append(rig.manager.running)
        with pytest.raises(TrainingError, match="already running"):
            rig.manager.start(other)
        with pytest.raises(TrainingError, match="training"):
            rig.manager.forget(voice)
        return PASS

    rig.manager._verify = slow_verify
    rig.manager.start(voice)
    assert seen == ["idle", True]
    assert len(rig.launches) == 1


# 10. Forget


def test_forget_removes_the_experiment_and_small_model_but_not_while_training(rig):
    voice, other = rig.voice("a"), rig.voice("b")
    rig.checkpoint(voice)
    rig.checkpoint(other)
    weights = rig.trainer / "rvc" / "assets" / "weights"
    weights.mkdir(parents=True)
    for v in (voice, other):
        (weights / f"voice-{v}.pth").write_bytes(b"x")

    rig.manager.start(voice)
    with pytest.raises(TrainingError, match="training"):
        rig.manager.forget(voice)
    assert rig.experiment_dir(voice).is_dir()
    rig.manager.forget(other)  # another voice may be forgotten meanwhile
    assert not rig.experiment_dir(other).exists() and not (weights / f"voice-{other}.pth").exists()

    rig.manager.cancel()
    with pytest.raises(TrainingError, match="training"):
        rig.manager.forget(voice)
    rig.finish(voice, 2, cancelled=True)
    rig.manager.forget(voice)
    assert not rig.experiment_dir(voice).exists() and not (weights / f"voice-{voice}.pth").exists()
    assert weights.is_dir() and (rig.data / "voices" / voice / "voice.json").is_file()
    assert rig.manager.status(voice)["state"] == "idle"
    rig.manager.forget(voice)  # nothing left: not an error


@pytest.mark.parametrize("bad", ["..", "../..", "a/b", "", None, "Voice", "mute"])
def test_forget_only_touches_experiment_folders_of_voice_ids(rig, bad):
    logs = rig.trainer / "rvc" / "logs"
    (logs / "mute").mkdir(parents=True)
    (logs / "voice-..").mkdir()
    if bad == "mute":
        rig.manager.forget(bad)  # a valid id: only logs/voice-mute would go, never logs/mute
    else:
        with pytest.raises(TrainingError, match="Unknown voice"):
            rig.manager.forget(bad)
    assert (logs / "mute").is_dir() and (logs / "voice-..").is_dir()


# Shutdown


def test_shutdown_stops_the_worker_and_marks_the_job_interrupted(rig):
    voice = rig.voice()
    rig.manager.shutdown()  # nothing running: nothing happens
    rig.manager.start(voice)
    rig.manager.shutdown()
    assert rig.handles[0].kills == 1
    assert rig.record(voice)["state"] == "interrupted" and rig.manager.running is False


# The real process helpers, without starting a real process


def test_default_launch_is_windowless_detached_from_the_console_and_never_a_shell_string(tmp_path, monkeypatch):
    calls = []

    class FakePopen:
        pid = 4242

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            self.waits = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            return 1

    ran = []
    monkeypatch.setattr(training.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(training.subprocess, "run", lambda argv, **kw: ran.append((argv, kw)))
    log_path = tmp_path / "worker.log"
    argv = ["C:\\AI Voice\\python.exe", "-P", "C:\\AI Voice\\worker.py", "--name=A B"]
    handle = training._launch_worker(argv, log_path)

    (popen_argv, kwargs), = calls
    assert popen_argv == argv and kwargs.get("shell", False) is False
    assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.STDOUT
    assert log_path.is_file()
    assert handle.pid == 4242 and handle.poll() is None

    handle.kill_tree()
    (kill_argv, kill_kwargs), = ran
    assert kill_argv == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert kill_kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW and kill_kwargs.get("shell", False) is False


def test_default_verify_runs_the_script_with_the_backends_python(tmp_path, monkeypatch):
    calls = []
    out = {"stdout": json.dumps(PASS), "code": 0}

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if out["stdout"] is None:
            raise subprocess.TimeoutExpired(argv, 1)
        return subprocess.CompletedProcess(argv, out["code"], stdout=out["stdout"], stderr="")

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    trainer = tmp_path / "AI Voice" / "trainer"
    m = TrainingManager(tmp_path / "data", Library(tmp_path / "data"), trainer)
    assert m.environment() == {**PASS, "ready": True}
    assert calls[0][0] == [sys.executable, str(trainer / "verify.py")]
    assert calls[0][1].get("shell", False) is False and calls[0][1]["creationflags"] == subprocess.CREATE_NO_WINDOW

    assert m.environment() == {**PASS, "ready": True} and len(calls) == 1  # cached
    out.update(stdout=json.dumps(FAIL), code=1)
    assert m.environment(refresh=True)["checks"] == FAIL["checks"]
    out.update(stdout="Traceback (most recent call last):\n  boom")
    assert m.environment()["ready"] is False
    out.update(stdout=None)
    assert m.environment()["ready"] is False
    assert m._verify(True)["ok"] is False and calls[-1][0][-1] == "--fast"


def test_recognising_a_worker_by_its_command_line():
    worker = Path("D:/Projects/AI Voice/trainer/worker.py")
    argv = ["D:\\Projects\\AI Voice\\trainer\\.venv\\Scripts\\python.exe", "-P", "d:\\projects\\AI Voice\\trainer\\WORKER.PY",
            "--voice-dir", "D:\\data\\voices\\dad-1a2b3c", "--experiment", "voice-dad-1a2b3c", "--epochs", "200"]
    match = training._is_worker_command
    assert match(argv, worker, "voice-dad-1a2b3c") is (os.name == "nt")  # the case-insensitive path is a Windows rule
    exact = [argv[0], "-P", str(worker), *argv[3:]]
    assert match(exact, worker, "voice-dad-1a2b3c") is True
    assert match(exact, worker, "voice-dad-1a2b3") is False
    assert match(exact, worker, "voice-mum-000000") is False
    assert match(["chrome.exe", "--type=renderer"], worker, "voice-dad-1a2b3c") is False
    assert match([argv[0], "other.py", "--experiment", "voice-dad-1a2b3c"], worker, "voice-dad-1a2b3c") is False
    assert match([str(worker), "--experiment"], worker, "voice-dad-1a2b3c") is False
    assert match([], worker, "voice-dad-1a2b3c") is False


def test_the_real_process_checks_on_this_process():
    assert training._pid_alive(os.getpid()) is True
    assert training._pid_alive(0) is False and training._pid_alive(-5) is False and training._pid_alive("12") is False
    worker = Path("D:/nowhere/trainer/worker.py")
    assert training._is_worker_process(os.getpid(), "voice-x", worker) is False  # pytest is not a worker
