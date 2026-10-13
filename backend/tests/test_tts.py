import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from aivoice import tts
from aivoice.audiofile import read_wav, write_wav
from aivoice.devices import Device
from aivoice.library import Library
from aivoice.tts import TtsError, TtsManager
from aivoice.tts_reference import NO_CLIP
from tests.test_training import FakeProc, FakeTools

PASS = {"ok": True, "checks": [{"id": "venv", "ok": True, "message": "TTS Python found"}]}
FAIL = {"ok": False, "checks": [
    {"id": "venv", "ok": True, "message": "TTS Python found"},
    {"id": "torch_gpu", "ok": False, "message": "torch sees no GPU"},
    {"id": "models", "ok": False, "message": "model files wrong"},
]}
DEVICES = [Device(1, "Speakers", 0, 2), Device(2, "CABLE Input", 0, 2), Device(3, "Microphone", 1, 0)]
TRANSCRIPT = "This is the reference sentence for the voice."
TEXT = "Hello there. How are you today? This is the last one."
PIECES = ["Hello there.", "How are you today?", "This is the last one."]
WAIT = 10  # seconds a test waits for another thread before failing; never reached when things work


def speech(seconds=1.0, lead=0.5, rate=24000, level=0.3):
    """What the worker writes for a piece: a quiet lead-in, then sound."""
    t = np.arange(int(seconds * rate)) / rate
    return np.concatenate([np.zeros(int(lead * rate), np.float32),
                           (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)])


class FakeWorker:
    """Stands in for the worker process: the manager's writes arrive as parsed commands, and the test (or the
    automatic replies) put protocol lines on what the manager reads as the worker's stdout."""

    def __init__(self, rig, pid):
        self.rig, self.pid = rig, pid
        self.code = None
        self.kills = 0
        self.kill_error = None
        self.on_kill = None
        self.commands = queue.Queue()
        self.seen = []  # every command, in order
        self.lines = queue.Queue()
        self.stdin, self.stdout = _In(self), _Out(self)
        self.active = None  # the synthesize command being served
        self.broken_pipe = False
        if rig.ready:
            self.emit(event="ready", sample_rate=24000)

    # what the manager calls
    def poll(self):
        return self.code

    def kill_tree(self):
        self.kills += 1
        self.rig.log.append("kill")
        if self.on_kill is not None:
            self.on_kill()
        if self.kill_error is not None:
            raise self.kill_error
        self.die()

    # what a test calls
    def emit(self, **event):
        self.lines.put(json.dumps(event).encode("ascii") + b"\n")

    def raw(self, line: bytes):
        self.lines.put(line)

    def die(self, code=1):
        if self.code is None:
            self.code = code
            self.lines.put(None)

    def expect(self, name):
        command = self.commands.get(timeout=WAIT)
        assert command["cmd"] == name, command
        return command

    def named(self, name):
        return [c for c in self.seen if c["cmd"] == name]

    def audio(self, command, index, audio=None):
        path = Path(command["out_dir"]) / f"{command['id']}_{index:03d}.wav"
        audio = self.rig.audio_for(index) if audio is None else audio
        if isinstance(audio, bytes):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        elif audio is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(path, audio, 24000)
        self.emit(event="audio", id=command["id"], index=index, path=str(path), sample_rate=24000,
                  text=command["pieces"][index])
        return path

    def done(self, command):
        self.active = None
        self.emit(event="done", id=command["id"], pieces=len(command["pieces"]))

    def cancelled(self, command):
        self.active = None
        self.emit(event="cancelled", id=command["id"], index=0)

    # the automatic replies
    def received(self, command):
        self.seen.append(command)
        self.rig.log.append(command["cmd"])
        self.commands.put(command)
        if command["cmd"] == "synthesize":
            self.active = command
        if not self.rig.auto:
            return
        if command["cmd"] == "set_reference":
            self.emit(event="reference", seconds=self.rig.reference_seconds)
        elif command["cmd"] == "synthesize":
            self.emit(event="started", id=command["id"], pieces=len(command["pieces"]))
            for index in range(len(command["pieces"])):
                self.audio(command, index)
            self.done(command)
        elif command["cmd"] == "cancel":
            if self.active is not None and self.active["id"] == command["id"]:
                self.cancelled(self.active)  # the real worker is silent for a job that is not running


class _In:
    def __init__(self, worker):
        self._worker = worker
        self.closed = False

    def write(self, data: bytes):
        if self._worker.broken_pipe or self._worker.code is not None:
            raise BrokenPipeError("the pipe is closed")
        for line in data.decode("utf-8").splitlines():
            self._worker.received(json.loads(line))

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _Out:
    def __init__(self, worker):
        self._worker = worker
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        line = self._worker.lines.get()
        if line is None:
            raise StopIteration
        return line

    def close(self):
        self.closed = True


class FakePlayer:
    def __init__(self, rig):
        self.rig = rig
        self.starts = []  # the devices of each session
        self.session = None
        self.pieces = queue.Queue()  # (session, audio) as enqueued
        self.played = []
        self.stops = 0
        self.finishes = 0
        self.error = None
        self.error_at_end = None  # a device lost in the last moment of playback
        self.start_error = None
        self.auto_finish = True
        self.ended = threading.Event()
        self.finished = threading.Event()  # finish() was called: every piece has been queued
        self.stopped = threading.Event()

    def start(self, devices, gains=None):
        if self.start_error is not None:
            raise ValueError(self.start_error)
        self.starts.append(dict(devices))
        self.session = object()
        self.ended.clear()
        self.rig.log.append("player.start")
        return self.session

    def enqueue(self, audio, pause_ms=300, session=None):
        self.rig.log.append("enqueue")
        self.played.append((session, audio))
        self.pieces.put((session, audio))

    def finish(self):
        self.finishes += 1
        self.finished.set()
        if self.auto_finish:
            self.ended.set()

    def stop(self):
        self.stops += 1
        self.rig.log.append("player.stop")
        self.ended.set()
        self.stopped.set()

    def wait(self, timeout=None):
        ended = self.ended.wait(timeout)
        if ended and self.error_at_end is not None:
            self.error = self.error_at_end
        return ended

    def status(self):
        return {"playing": not self.ended.is_set(), "queued_seconds": 0.0, "played_seconds": 0.0,
                "error": self.error}


class FakeConverter:
    def __init__(self):
        self.inputs = []
        self.error = None
        self.silent = False
        self.entered = threading.Event()
        self.gate = None  # an Event a test sets to let a conversion return
        self.on_convert = None

    def convert(self, piece):
        self.inputs.append(piece)
        self.entered.set()
        if self.on_convert is not None:
            self.on_convert()
        if self.gate is not None:
            assert self.gate.wait(WAIT)
        if self.error is not None:
            raise self.error
        if self.silent:
            return np.zeros_like(piece)
        return (piece * 0.5).astype(np.float32)


class Rig:
    """A manager on a temporary data folder, with the worker, the player, the converter, the environment check,
    the clock and the process tools all faked and recorded."""

    def __init__(self, tmp_path):
        self.data = tmp_path / "data"
        self.tts_root = tmp_path / "AI Voice" / "tts"
        self.library = Library(self.data)
        self.player = FakePlayer(self)
        self.converter = FakeConverter()
        self.rvc_calls = []
        self.rvc_error = None
        self.tools = FakeTools()
        self.launches = []  # (argv, log path)
        self.workers = []
        self.launched = queue.Queue()
        self.launch_error = None
        self.launching = threading.Event()
        self.launch_gate = None  # an Event a test sets to let a launch return
        self.report = PASS
        self.verify_calls = []
        self.clock = 1_760_000_000.0  # 2025-10-09T08:53:20Z
        self.log = []  # what happened, in order, across the fakes
        self.auto = True  # the worker answers by itself
        self.ready = True  # and reports ready as soon as it is started
        self.reference_seconds = 8.0
        self.audio_for = lambda index: speech()
        self.devices = list(DEVICES)
        self.manager = None

    def launch(self, argv, log_path):
        if self.launch_error is not None:
            raise self.launch_error
        self.launching.set()
        if self.launch_gate is not None:
            assert self.launch_gate.wait(WAIT)
        self.launches.append((argv, log_path))
        worker = FakeWorker(self, 5000 + len(self.workers))
        self.workers.append(worker)
        self.launched.put(worker)
        self.log.append("launch")
        return worker

    def verify(self, fast):
        self.verify_calls.append(fast)
        return self.report

    def rvc(self, voice_id):
        self.rvc_calls.append(voice_id)
        if self.rvc_error is not None:
            raise self.rvc_error
        return self.converter

    def build(self):
        self.manager = TtsManager(self.data, self.library, self.tts_root, self.player, lambda: self.devices,
                                  launch=self.launch, verify=self.verify, rvc_factory=self.rvc,
                                  now=lambda: self.clock, processes=self.tools, tick=0.005)
        return self.manager

    def voice(self, name="Narrator", seconds=8.0, text=TRANSCRIPT, model=False):
        voice = self.library.create(name)
        if seconds:
            noise = np.random.default_rng(0).uniform(-0.2, 0.2, int(seconds * 48000)).astype(np.float32)
            self.library.add_clip(voice.id, noise, text, "recorded")
        if model:
            folder = self.data / "voices" / voice.id / "models" / "rvc"
            folder.mkdir(parents=True)
            (folder / "model.onnx").write_bytes(b"model")
            (folder / "voice.json").write_text(json.dumps({"sample_rate": 40000}))
        return self.library.get(voice.id)

    def speak(self, voice=None, text=TEXT, preset="fast", mic_device=None, monitor_device=1, save=False):
        voice = voice or self.voice()
        return self.manager.speak(voice.id, text, preset, mic_device=mic_device, monitor_device=monitor_device,
                                  save=save)

    def worker(self):
        return self.launched.get(timeout=WAIT)

    def children(self, *procs):
        """What the next worker to be launched will be seen to have started."""
        self.tools.trees[5000 + len(self.workers)] = list(procs)

    def finish(self):
        assert self.manager.join(WAIT), "the request did not end"
        return self.manager.status()

    def tmp_folders(self):
        tmp = self.data / "tts" / "tmp"
        return sorted(p.name for p in tmp.iterdir()) if tmp.is_dir() else []


@pytest.fixture
def rig(tmp_path):
    rig = Rig(tmp_path)
    rig.build()
    yield rig
    for worker in rig.workers:
        worker.die()  # a request a test left waiting for its worker ends at once
    rig.manager.shutdown()


def nothing_started(rig):
    assert rig.launches == [] and rig.player.starts == [] and rig.tmp_folders() == []
    assert rig.manager.status()["state"] == "idle" and not rig.manager.busy and not rig.manager.is_active()


# ---------------------------------------------------------------- 1. validation

@pytest.mark.parametrize("text, message", [
    ("", "There is no text to speak."),
    ("   \n\t ", "There is no text to speak."),
    ("a" * 5001, "The text is too long (5001 characters). Keep it under 5000 characters."),
    ("... !!! \U0001F600", "There is nothing to say in that text"),
    (None, "There is no text to speak."),
])
def test_text_that_cannot_be_spoken_is_refused(rig, text, message):
    with pytest.raises(TtsError, match="^" + __import__("re").escape(message) + "$"):
        rig.speak(text=text)
    nothing_started(rig)


def test_five_thousand_characters_are_accepted(rig):
    rig.speak(text=("word " * 1000).strip())
    assert rig.finish()["error"] is None


def test_an_unknown_preset_is_refused(rig):
    with pytest.raises(TtsError, match="^Unknown preset: studio. Choose fast, balanced or best.$"):
        rig.speak(preset="studio")
    nothing_started(rig)


def test_an_unknown_voice_is_refused(rig):
    with pytest.raises(TtsError, match="^Unknown voice$"):
        rig.manager.speak("no-such-voice", TEXT, "fast", monitor_device=1)
    nothing_started(rig)


def test_a_voice_without_a_reference_clip_is_refused_with_the_reference_message(rig):
    with pytest.raises(TtsError) as refused:
        rig.speak(rig.voice(seconds=2.0))
    assert str(refused.value) == NO_CLIP
    nothing_started(rig)


def test_a_reference_clip_whose_audio_is_gone_is_refused(rig):
    voice = rig.voice()
    (rig.data / "voices" / voice.id / "clips" / "0001.orig.wav").unlink()
    with pytest.raises(TtsError, match="The audio of clip 1, this voice's reference clip, is missing"):
        rig.speak(voice)
    nothing_started(rig)


def test_best_is_refused_for_a_voice_without_a_trained_model(rig):
    with pytest.raises(TtsError, match="^Best needs this voice's trained model, and it has none. Train the voice "
                                       "first, or choose Fast or Balanced.$"):
        rig.speak(preset="best")
    nothing_started(rig)
    assert rig.rvc_calls == []


@pytest.mark.parametrize("mic, monitor, message", [
    (None, None, "Choose a virtual mic or a listen device"),
    (1, 1, "Monitor and virtual mic must be different devices"),
    (3, None, "Microphone is not an output device"),
    (2, 3, "Microphone is not an output device"),
    (None, 9, "Audio device 9 is no longer available"),
])
def test_output_devices_are_checked(rig, mic, monitor, message):
    with pytest.raises(TtsError, match="^" + message + "$"):
        rig.speak(mic_device=mic, monitor_device=monitor)
    nothing_started(rig)


def test_both_outputs_are_opened_under_the_names_the_player_knows(rig):
    rig.speak(mic_device=2, monitor_device=1)
    rig.finish()
    assert rig.player.starts == [{"mic": 2, "monitor": 1}]


def test_a_second_request_is_refused_while_one_is_speaking(rig):
    rig.auto = False
    voice = rig.voice()
    rig.speak(voice)
    assert rig.manager.busy and rig.manager.is_active()
    with pytest.raises(TtsError, match="^Already speaking. Stop it first.$"):
        rig.speak(voice)
    assert len(rig.player.starts) == 1


def test_a_failing_environment_is_refused_with_its_failing_checks(rig):
    rig.report = FAIL
    with pytest.raises(TtsError, match="^Text to speech is not set up: torch sees no GPU; model files wrong$"):
        rig.speak()
    nothing_started(rig)


def test_checks_run_in_the_documented_order(rig):
    """Everything is wrong at once; each fix reveals the next refusal."""
    rig.report = FAIL
    bare = rig.voice("Bare", seconds=2.0)

    def refusal(voice_id, text, preset, **devices):
        with pytest.raises(TtsError) as refused:
            rig.manager.speak(voice_id, text, preset, **devices)
        return str(refused.value)

    assert refusal("nobody", "", "studio") == "There is no text to speak."
    assert refusal("nobody", TEXT, "studio").startswith("Unknown preset")
    assert refusal("nobody", TEXT, "best") == "Unknown voice"
    assert refusal(bare.id, TEXT, "best") == NO_CLIP
    voice = rig.voice()
    assert refusal(voice.id, TEXT, "best").startswith("Best needs this voice's trained model")
    assert refusal(voice.id, TEXT, "fast") == "Choose a virtual mic or a listen device"
    assert refusal(voice.id, TEXT, "fast", monitor_device=1).startswith("Text to speech is not set up")
    assert rig.verify_calls == [False]  # the environment is only looked at when everything else is in order
    nothing_started(rig)


def test_an_output_that_cannot_be_opened_is_refused_and_nothing_is_left_running(rig):
    rig.player.start_error = "Could not open the audio output: device busy"
    with pytest.raises(TtsError, match="^Could not open the audio output: device busy$"):
        rig.speak()
    assert rig.launches == [] and rig.tmp_folders() == [] and not rig.manager.busy
    assert rig.manager.status()["state"] == "idle"


def test_characters_that_cannot_be_sent_as_text_are_removed_before_the_worker_sees_them(rig):
    voice = rig.voice()
    record = rig.data / "voices" / voice.id / "voice.json"  # only a hand-edited record can hold one
    record.write_text(record.read_text(encoding="utf-8").replace(
        TRANSCRIPT, "Reference " + chr(92) + "ud800sentence for the voice, long enough."), encoding="utf-8")
    rig.speak(voice, text="Hello " + chr(0xDC00) + "there.")
    assert rig.finish()["error"] is None
    worker = rig.workers[0]
    assert worker.named("synthesize")[0]["pieces"] == ["Hello there."]
    assert worker.named("set_reference")[0]["text"] == "Reference sentence for the voice, long enough."
    for command in worker.seen:
        json.dumps(command).encode("utf-8")  # every command survives the trip as UTF-8


# ---------------------------------------------------------------- 2. the worker is started once and reused

def test_the_worker_is_started_on_first_use_and_is_starting_until_it_reports_ready(rig):
    rig.auto = rig.ready = False
    assert rig.manager.status()["engine"] == "stopped"
    status = rig.speak()
    assert status["state"] == "preparing" and status["started"] == rig.clock
    worker = rig.worker()
    assert rig.manager.status()["engine"] == "starting" and rig.manager.status()["state"] == "preparing"
    assert worker.seen == []  # nothing is asked of a worker that has not loaded its model yet
    worker.emit(event="ready", sample_rate=24000)
    worker.expect("set_reference")
    assert rig.manager.status()["engine"] == "ready"


def test_the_worker_is_launched_from_the_tts_environment_with_an_argument_list(rig):
    rig.speak()
    rig.finish()
    (argv, log_path), = rig.launches
    assert argv == [str(rig.tts_root / ".venv" / "Scripts" / "python.exe"), "-u", str(rig.tts_root / "tts_worker.py"),
                    "--model-dir", str(rig.tts_root / "models"), "--parent-pid", str(os.getpid())]
    assert log_path == rig.data / "tts" / "worker.log"  # where the worker's stderr is drained to


def test_the_worker_is_reused_and_the_reference_is_set_only_when_the_clip_changes(rig):
    voice, other = rig.voice(), rig.voice("Other", seconds=6.0, text="Another voice says this sentence here.")
    for chosen in (voice, voice, other, other, voice):
        rig.speak(chosen)
        assert rig.finish()["error"] is None
    worker, = rig.workers
    references = worker.named("set_reference")
    assert [Path(r["wav"]) for r in references] == [
        rig.data / "voices" / v.id / "clips" / "0001.orig.wav" for v in (voice, other, voice)]
    assert [r["text"] for r in references] == [TRANSCRIPT, "Another voice says this sentence here.", TRANSCRIPT]
    assert len(worker.named("synthesize")) == 5


def test_the_reference_is_set_again_when_the_voices_clip_or_its_transcript_changes(rig):
    voice = rig.voice()
    rig.speak(voice)
    rig.finish()
    rig.library.update_text(voice.id, 1, "A corrected transcript for the same clip.")
    rig.speak(voice)
    rig.finish()
    noise = np.random.default_rng(1).uniform(-0.2, 0.2, 5 * 48000).astype(np.float32)
    rig.library.add_clip(voice.id, noise, "A second clip of five seconds.", "recorded")
    rig.library.delete_clip(voice.id, 1)
    rig.speak(voice)
    assert rig.finish()["error"] is None
    references = rig.workers[0].named("set_reference")
    assert [r["text"] for r in references] == [TRANSCRIPT, "A corrected transcript for the same clip.",
                                                "A second clip of five seconds."]
    assert Path(references[2]["wav"]).name == "0002.orig.wav"


def test_the_reference_clip_is_the_voices_own_file_and_is_left_untouched(rig):
    voice = rig.voice()
    clip = rig.data / "voices" / voice.id / "clips" / "0001.orig.wav"
    before = clip.read_bytes()
    rig.speak(voice, save=True)
    rig.finish()
    assert Path(rig.workers[0].named("set_reference")[0]["wav"]) == clip
    assert read_wav(clip)[1] == 48000 and clip.read_bytes() == before
    assert sorted(p.name for p in clip.parent.iterdir()) == ["0001.orig.wav", "0001.txt", "0001.wav"]


def test_a_refused_reference_fails_the_request_with_the_workers_words_and_is_asked_again_next_time(rig):
    rig.auto = False
    voice = rig.voice()
    rig.speak(voice)
    worker = rig.worker()
    command = worker.expect("set_reference")
    assert command["id"]  # so that a refusal can be matched to this request
    worker.emit(event="error", id=command["id"], code="bad_request",
                message="The reference clip 0001.orig.wav could not be read as a WAV file: bad header")
    status = rig.finish()
    assert status["error"] == "The reference clip 0001.orig.wav could not be read as a WAV file: bad header"
    assert status["state"] == "idle" and worker.kills == 0 and worker.named("synthesize") == []
    rig.speak(voice)
    worker.expect("set_reference")  # the worker dropped its reference, so nothing is taken as loaded


def test_a_worker_that_ends_or_hangs_while_loading_the_reference_fails_the_request(rig):
    rig.auto = False
    voice = rig.voice()
    rig.speak(voice)
    worker = rig.worker()
    worker.expect("set_reference")
    worker.die()  # a reference still queued when the worker ends gets no answer
    status = rig.finish()
    assert status["error"].startswith("The voice engine stopped unexpectedly while loading the reference clip.")
    assert status["engine"] == "failed" and status["state"] == "idle"
    rig.speak(voice)
    worker = rig.worker()
    worker.expect("set_reference")
    rig.clock += tts.PIECE_TIMEOUT + 1
    status = rig.finish()
    assert status["error"].startswith("The voice engine did not answer within 60 seconds while loading the "
                                      "reference clip")
    assert worker.kills == 1 and worker.named("synthesize") == [] and len(rig.launches) == 2


# ---------------------------------------------------------------- 3. pieces

def test_the_text_is_split_for_the_reference_and_sent_with_the_presets_steps(rig):
    voice = rig.voice(model=True)
    for preset in ("fast", "balanced", "best"):
        rig.speak(voice, preset=preset)
        assert rig.finish()["error"] is None
    jobs = rig.workers[0].named("synthesize")
    assert [j["nfe"] for j in jobs] == [8, 16, 32]
    limit = tts.tts_audio.max_chars_for(TRANSCRIPT, 8.0)
    for job in jobs:
        assert job["pieces"] == PIECES and job["max_chars"] == limit and 0.5 <= job["speed"] <= 2.0
        assert tts.JOB_ID.fullmatch(job["id"])
        assert Path(job["out_dir"]) == rig.data / "tts" / "tmp" / job["id"]
    assert len({j["id"] for j in jobs}) == 3


def test_long_text_is_cut_to_the_limit_of_the_reference_clip(rig):
    rig.speak(text="word " * 200)
    rig.finish()
    job, = rig.workers[0].named("synthesize")
    assert len(job["pieces"]) > 5 and all(len(p.encode("utf-8")) <= job["max_chars"] for p in job["pieces"])


def test_the_limit_follows_the_length_the_worker_measured_for_the_reference(rig):
    rig.reference_seconds = 11.0  # the worker decoded a longer clip than the library recorded
    rig.speak(text="word " * 200)
    assert rig.finish()["error"] is None
    job, = rig.workers[0].named("synthesize")
    limit = tts.tts_audio.max_chars_for(TRANSCRIPT, 11.0)
    assert job["max_chars"] == limit < tts.tts_audio.max_chars_for(TRANSCRIPT, 8.0)
    assert all(len(p.encode("utf-8")) <= limit for p in job["pieces"])
    assert rig.manager.status()["pieces"] == len(job["pieces"])


def test_each_piece_is_trimmed_finished_and_queued_and_its_file_deleted(rig):
    rig.speak()
    status = rig.finish()
    assert status["error"] is None and status["state"] == "idle"
    assert (status["piece"], status["pieces"]) == (3, 3)
    assert len(rig.player.played) == 3 and rig.player.finishes == 1 and rig.player.stops == 0
    for session, audio in rig.player.played:
        assert session is rig.player.session  # late audio of another request could never join this one
        assert audio.dtype == np.float32 and audio.ndim == 1
        # 1.5 s at 24 kHz with half a second of lead-in becomes about 1 s at 48 kHz
        assert 0.95 * 48000 < len(audio) < 1.15 * 48000
        assert abs(audio[0]) < 0.01 and abs(audio[-1]) < 0.01 and 0.25 < np.abs(audio).max() <= 0.97
    assert rig.tmp_folders() == []


def test_the_end_of_the_reference_and_the_pause_after_it_are_not_played(rig):
    """The real model's piece often begins with the dying end of the reference clip's last word, then a second
    of room tone, then the sentence (first real run of the manager). Neither the blip nor the wait is played."""
    rate = 24000
    blip = speech(0.1, lead=0.0, level=0.06)  # 14 dB below the sentence: above what trimming calls quiet
    room = np.random.default_rng(1).uniform(-0.001, 0.001, rate).astype(np.float32)
    rig.audio_for = lambda index: np.concatenate([blip, room, speech(1.0, lead=0.0)])
    rig.speak()
    assert rig.finish()["error"] is None
    assert len(rig.player.played) == 3
    for _, audio in rig.player.played:
        assert 0.95 * 48000 < len(audio) < 1.15 * 48000  # the sentence alone, not 2.1 s
        assert np.abs(audio[:4800]).max() > 0.25  # and it starts with the sentence


def test_playback_of_the_first_piece_starts_before_later_pieces_are_synthesised(rig):
    rig.auto = False
    rig.speak()
    worker = rig.worker()
    worker.expect("set_reference")
    worker.emit(event="reference", seconds=8.0)
    job = worker.expect("synthesize")
    worker.emit(event="started", id=job["id"], pieces=3)
    assert rig.manager.status()["piece"] == 0
    first = worker.audio(job, 0)
    session, audio = rig.player.pieces.get(timeout=WAIT)
    assert session is rig.player.session and len(audio) > 0
    assert rig.player.finishes == 0
    worker.audio(job, 1)
    rig.player.pieces.get(timeout=WAIT)
    status = rig.manager.status()
    assert status["state"] == "speaking" and status["pieces"] == 3 and status["piece"] in (1, 2)
    assert not first.exists()
    worker.audio(job, 2)
    worker.done(job)
    assert rig.finish()["piece"] == 3


def test_the_request_stays_speaking_until_playback_has_ended(rig):
    rig.player.auto_finish = False
    rig.speak()
    for _ in range(3):
        rig.player.pieces.get(timeout=WAIT)
    assert not rig.manager.join(0.05)
    assert rig.manager.status()["state"] == "speaking" and rig.manager.busy
    rig.player.ended.set()
    assert rig.finish()["state"] == "idle"


def test_a_silent_piece_fails_the_request_and_is_never_played(rig):
    rig.audio_for = lambda index: np.zeros(24000, np.float32) if index == 1 else speech()
    rig.speak(save=True)
    status = rig.finish()
    assert status["error"] == "The voice engine produced no sound for sentence 2"
    assert status["state"] == "idle" and status["saved_path"] is None
    assert len(rig.player.played) == 1 and rig.player.stops == 1
    assert rig.tmp_folders() == [] and list((rig.data / "tts").glob("*.wav")) == []
    assert rig.workers[0].kills == 0  # the worker did nothing wrong that a restart would cure


def test_a_piece_that_is_only_a_click_of_room_tone_counts_as_silent(rig):
    rig.audio_for = lambda index: speech(seconds=0.05)
    rig.speak()
    assert rig.finish()["error"] == "The voice engine produced no sound for sentence 1"
    assert rig.player.played == []


@pytest.mark.parametrize("content", [None, b"this is not a wav file"])
def test_a_missing_or_unreadable_piece_fails_the_request(rig, content):
    rig.auto = False
    rig.speak()
    worker = rig.worker()
    worker.expect("set_reference")
    worker.emit(event="reference", seconds=8.0)
    job = worker.expect("synthesize")
    if content is None:
        worker.emit(event="audio", id=job["id"], index=0, path=str(Path(job["out_dir"]) / f"{job['id']}_000.wav"))
    else:
        worker.audio(job, 0, content)
    worker.expect("cancel")
    worker.cancelled(job)
    status = rig.finish()
    assert status["error"].startswith("Sentence 1 from the voice engine could not be read")
    assert rig.player.played == [] and rig.tmp_folders() == []


def test_a_piece_is_only_ever_read_from_the_requests_own_folder(rig, tmp_path):
    outside = tmp_path / "elsewhere.wav"
    write_wav(outside, speech(), 24000)
    rig.auto = False
    rig.speak()
    worker = rig.worker()
    worker.expect("set_reference")
    worker.emit(event="reference", seconds=8.0)
    job = worker.expect("synthesize")
    worker.emit(event="audio", id=job["id"], index=0, path=str(outside))
    worker.expect("cancel")
    worker.cancelled(job)
    assert rig.finish()["error"].startswith("Sentence 1 from the voice engine could not be read")
    assert rig.player.played == [] and outside.exists()


def test_pieces_out_of_order_fail_the_request(rig):
    rig.auto = False
    rig.speak()
    worker = rig.worker()
    worker.expect("set_reference")
    worker.emit(event="reference", seconds=8.0)
    job = worker.expect("synthesize")
    worker.audio(job, 1)
    assert "out of order" in rig.finish()["error"]
    assert rig.player.played == []


def test_a_job_reported_done_with_pieces_missing_fails(rig):
    rig.auto = False
    rig.speak(save=True)
    worker = rig.worker()
    worker.expect("set_reference")
    worker.emit(event="reference", seconds=8.0)
    job = worker.expect("synthesize")
    worker.audio(job, 0)
    worker.done(job)
    status = rig.finish()
    assert status["error"] == "The voice engine finished before every sentence was spoken"
    assert status["saved_path"] is None and rig.player.stops == 1


def test_best_sends_every_piece_through_the_voices_model_at_48_khz(rig):
    voice = rig.voice(model=True)
    rig.speak(voice, preset="best")
    status = rig.finish()
    assert status["error"] is None and status["preset"] == "best"
    assert rig.rvc_calls == [voice.id]
    assert len(rig.converter.inputs) == 3
    for (_, played), given in zip(rig.player.played, rig.converter.inputs):
        assert given.dtype == np.float32 and 0.95 * 48000 < len(given) < 1.15 * 48000  # trimmed, at 48 kHz
        assert 0.1 < np.abs(played).max() < 0.2  # what was played is what the converter returned (half the level)


def test_fast_and_balanced_never_load_the_voices_model(rig):
    voice = rig.voice(model=True)
    rig.speak(voice, preset="balanced")
    rig.finish()
    assert rig.rvc_calls == [] and rig.converter.inputs == []


@pytest.mark.parametrize("fault, message", [
    ("error", "Voice conversion failed for sentence 1: the session is gone"),
    ("silent", "Voice conversion produced no sound for sentence 1"),
    ("load", "Could not load this voice's trained model: model.onnx is damaged"),
])
def test_a_failed_rvc_pass_fails_the_request_and_plays_nothing(rig, fault, message):
    if fault == "error":
        rig.converter.error = RuntimeError("the session is gone")
    elif fault == "silent":
        rig.converter.silent = True
    else:
        rig.rvc_error = RuntimeError("model.onnx is damaged")
    rig.speak(rig.voice(model=True), preset="best")
    status = rig.finish()
    assert status["error"] == message and status["state"] == "idle"
    assert rig.player.played == [] and rig.tmp_folders() == []


def test_the_default_rvc_pass_uses_live_modes_engine_factory_and_the_high_quality_preset(rig, monkeypatch):
    from aivoice import manager as live_manager
    from aivoice.presets import PRESETS

    built = []

    class Engine:
        def convert(self, audio, sample_rate, pitch_shift, params):
            built.append(("convert", len(audio), sample_rate, pitch_shift, params))
            return audio

    def factory(data):
        def build(voice):
            built.append(("engine", data, voice))
            return Engine()
        return build

    monkeypatch.setattr(live_manager, "default_engine_factory", factory)
    voice = rig.voice(model=True)
    folder = rig.data / "voices" / voice.id / "models" / "rvc"
    (folder / "model.index").write_bytes(b"index")
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES)
    converter = manager._default_rvc(voice.id)
    (_, data, described), warm = built
    assert data == rig.data and described.id == voice.id and described.sample_rate == 40000
    assert described.model_path == folder / "model.onnx" and described.index_path == folder / "model.index"
    assert warm == ("convert", 20 * 48000, 48000, 0.0, PRESETS["high_quality"])  # warmed with one silent window
    assert len(converter.convert(np.ones(4800, np.float32))) == 4800


# ---------------------------------------------------------------- 4. saving

def test_save_writes_the_finished_pieces_under_data_tts_and_reports_the_path(rig):
    voice = rig.voice()
    rig.speak(voice, save=True)
    status = rig.finish()
    path = rig.data / "tts" / f"20251009T085320Z-{voice.id}.wav"
    assert status["saved_path"] == str(path) and status["error"] is None
    audio, rate = read_wav(path)
    played = sum(len(a) for _, a in rig.player.played)
    assert rate == 48000 and len(audio) == played + 2 * int(0.3 * 48000)  # the pieces with the pauses between
    assert sorted(p.name for p in (rig.data / "tts").iterdir()) == [path.name, "tmp", "worker.log"] or \
        sorted(p.name for p in (rig.data / "tts").iterdir()) == [path.name, "tmp"]
    assert rig.tmp_folders() == []


def test_nothing_is_saved_unless_asked(rig):
    rig.speak()
    assert rig.finish()["saved_path"] is None
    assert list((rig.data / "tts").glob("*.wav")) == []


def test_two_saves_in_the_same_second_do_not_overwrite_each_other(rig):
    voice = rig.voice()
    rig.speak(voice, save=True)
    first = rig.finish()["saved_path"]
    rig.speak(voice, text="Something else entirely.", save=True)
    second = rig.finish()["saved_path"]
    assert first != second and Path(second).name == f"20251009T085320Z-{voice.id}-2.wav"
    assert len(read_wav(first)[0]) > len(read_wav(second)[0])


def test_a_save_that_finds_no_free_name_fails_instead_of_overwriting(rig, monkeypatch):
    monkeypatch.setattr(tts, "SAVE_NAMES", 3)
    voice = rig.voice()
    root = rig.data / "tts"
    root.mkdir(parents=True)
    taken = [root / f"20251009T085320Z-{voice.id}{suffix}.wav" for suffix in ("", "-2", "-3")]
    for path in taken:
        path.write_bytes(b"an earlier result")
    rig.speak(voice, save=True)
    status = rig.finish()
    assert status["error"] == ("The speech was played but could not be saved: too many results were saved in the "
                               "same second")
    assert status["saved_path"] is None and all(p.read_bytes() == b"an earlier result" for p in taken)
    assert sorted(root.glob("*.wav")) == sorted(taken)


def test_a_save_that_fails_is_reported_and_leaves_no_file(rig, monkeypatch):
    def full(path, pieces, pause_ms=300):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tts, "save_wav", full)
    rig.speak(save=True)
    status = rig.finish()
    assert status["error"] == "The speech was played but could not be saved: No space left on device"
    assert status["saved_path"] is None and status["state"] == "idle"


def test_a_stopped_request_saves_nothing_and_leaves_no_temporary_files(rig):
    rig.player.auto_finish = False
    rig.speak(save=True)
    for _ in range(3):
        rig.player.pieces.get(timeout=WAIT)  # everything is synthesised; it is still playing
    status = rig.manager.stop()
    assert status["state"] == "idle" and status["saved_path"] is None and status["error"] is None
    assert list((rig.data / "tts").glob("*.wav")) == [] and rig.tmp_folders() == []


# ---------------------------------------------------------------- 5. stop

def mid_job(rig, voice=None, **options):
    """A request whose first piece is playing while the worker is busy with the second."""
    rig.auto = False
    rig.speak(voice, **options)
    worker = rig.worker() if not rig.workers or rig.workers[-1].code is not None else rig.workers[-1]
    if not worker.named("synthesize"):
        worker.expect("set_reference")
        worker.emit(event="reference", seconds=8.0)
    job = worker.expect("synthesize")
    worker.emit(event="started", id=job["id"], pieces=len(job["pieces"]))
    worker.audio(job, 0)
    rig.player.pieces.get(timeout=WAIT)
    return worker, job


def test_stop_silences_the_player_first_then_cancels_the_job_and_keeps_the_worker(rig):
    worker, job = mid_job(rig, save=True)
    leftover = Path(job["out_dir"]) / f"{job['id']}_001.wav.part"  # what a cancelled worker can leave behind
    leftover.write_bytes(b"half a piece")
    del rig.log[:]
    stopping = threading.Thread(target=rig.manager.stop)
    stopping.start()
    assert worker.expect("cancel") == {"cmd": "cancel", "id": job["id"]}
    assert rig.log[0] == "player.stop" and rig.log[-1] == "cancel"  # silence first, then the worker
    status = rig.manager.status()
    assert status["state"] == "stopping" and rig.manager.busy  # not idle until the worker has let go of the job
    with pytest.raises(TtsError, match="Already speaking"):
        rig.speak(rig.library.list()[0])
    assert worker.named("synthesize") == [job]  # and so no new job was sent to a worker that would refuse it
    worker.cancelled(job)
    stopping.join(WAIT)
    status = rig.manager.status()
    assert status["state"] == "idle" and status["error"] is None and status["saved_path"] is None
    assert status["engine"] == "ready" and worker.kills == 0 and "kill" not in rig.log
    assert rig.tmp_folders() == [] and list((rig.data / "tts").glob("*.wav")) == []
    rig.auto = True
    rig.speak(rig.library.list()[0])
    assert rig.finish()["error"] is None and len(rig.launches) == 1
    assert len(worker.named("set_reference")) == 1  # the clip is still loaded


def test_stop_returns_idle(rig):
    worker, job = mid_job(rig)
    rig.auto = True  # the worker answers the cancel
    status = rig.manager.stop()
    assert status["state"] == "idle" and not rig.manager.busy and rig.player.stops >= 1


def test_stop_is_heard_at_once_even_while_the_request_is_busy_converting(rig):
    rig.converter.gate = threading.Event()
    rig.player.stopped = threading.Event()
    rig.speak(rig.voice(model=True), preset="best")
    assert rig.converter.entered.wait(WAIT)  # the request's thread is inside the RVC pass
    stopping = threading.Thread(target=rig.manager.stop)
    stopping.start()
    assert rig.player.stopped.wait(WAIT)  # by stop() itself: the thread that would do it is busy
    assert rig.player.stops == 1 and rig.manager.status()["state"] == "stopping"
    rig.converter.gate.set()
    stopping.join(WAIT)
    assert rig.manager.status()["state"] == "idle" and rig.player.played == []  # the converted piece is dropped


def test_a_worker_found_dead_when_the_next_request_reaches_it_fails_that_request_and_is_replaced(rig):
    voice = rig.voice()
    rig.speak(voice)
    rig.finish()
    worker, = rig.workers
    worker.lines.put(None)  # its pipe ended, but the launcher process still looks alive
    rig.speak(voice)
    status = rig.finish()
    assert status["error"].startswith("The voice engine stopped unexpectedly while idle.")
    assert worker.kills == 1 and status["engine"] == "failed" and len(worker.named("synthesize")) == 1
    rig.speak(voice)
    assert rig.finish()["error"] is None and len(rig.launches) == 2


def test_stop_when_idle_is_harmless(rig):
    assert rig.manager.stop()["state"] == "idle"
    assert rig.player.stops == 0 and rig.launches == []
    rig.speak()
    rig.finish()
    assert rig.manager.stop()["state"] == "idle"
    assert rig.workers[0].named("cancel") == [] and rig.workers[0].kills == 0


def test_late_events_of_a_stopped_job_never_reach_the_next_one(rig):
    worker, old = mid_job(rig)
    late = Path(old["out_dir"]) / f"{old['id']}_001.wav"
    stopping = threading.Thread(target=rig.manager.stop)
    stopping.start()
    worker.expect("cancel")
    worker.audio(old, 1)  # finished just before the cancel took effect
    worker.cancelled(old)
    stopping.join(WAIT)
    assert not late.exists()
    old_session, played_before = rig.player.session, len(rig.player.played)
    rig.speak(rig.library.list()[0], text="A new request.")
    new = worker.expect("synthesize")
    worker.emit(event="audio", id=old["id"], index=1, path=str(late))  # a straggler with the old job's id
    worker.emit(event="done", id=old["id"], pieces=3)
    worker.audio(new, 0)
    worker.done(new)
    assert rig.finish()["error"] is None
    fresh = rig.player.played[played_before:]
    assert len(fresh) == 1 and fresh[0][0] is rig.player.session is not old_session


def test_stop_while_the_reference_is_loading_waits_for_the_workers_answer(rig):
    rig.auto = False
    voice = rig.voice()
    rig.speak(voice)
    worker = rig.worker()
    worker.expect("set_reference")
    stopping = threading.Thread(target=rig.manager.stop)
    stopping.start()
    assert not rig.manager.join(0.05)  # the answer is still owed: a later request must not mistake it for its own
    worker.emit(event="reference", seconds=8.0)
    stopping.join(WAIT)
    assert rig.manager.status()["state"] == "idle" and worker.named("cancel") == []
    rig.speak(voice)
    worker.expect("synthesize")  # the clip that did load is not loaded twice


def test_stop_while_the_worker_is_starting_leaves_it_to_finish_loading(rig):
    rig.auto = rig.ready = False
    voice = rig.voice()
    rig.speak(voice)
    worker = rig.worker()
    assert rig.manager.stop()["state"] == "idle"
    assert worker.kills == 0 and rig.manager.status()["engine"] == "starting"
    rig.speak(voice)
    worker.emit(event="ready")
    worker.expect("set_reference")
    assert len(rig.launches) == 1 and rig.manager.status()["engine"] == "ready"


def test_a_worker_that_does_not_answer_a_cancel_is_stopped(rig):
    worker, job = mid_job(rig)
    stopping = threading.Thread(target=rig.manager.stop)
    stopping.start()
    worker.expect("cancel")
    rig.clock += tts.CANCEL_TIMEOUT + 1
    stopping.join(WAIT)
    status = rig.manager.status()
    assert status["state"] == "idle" and status["engine"] == "failed" and worker.kills == 1
    assert "did not answer within 10 seconds" in status["error"]


# ---------------------------------------------------------------- 6. the worker dies or hangs

def test_a_worker_that_dies_mid_request_fails_it_with_the_end_of_its_log_and_is_replaced(rig):
    child = FakeProc(7001)
    rig.children(child)
    worker, job = mid_job(rig)
    rig.launches[0][1].write_text("loading\nTraceback (most recent call last):\nRuntimeError: HIP out of memory\n")
    worker.die()
    status = rig.finish()
    assert status["state"] == "idle" and status["engine"] == "failed"
    assert status["error"].startswith("The voice engine stopped unexpectedly while synthesising.")
    assert "RuntimeError: HIP out of memory" in status["error"]
    assert rig.player.stops == 1 and rig.tmp_folders() == []
    assert rig.tools.count("kill") == 1 and not child.alive  # what it had started is ended and checked
    rig.auto = True
    rig.speak(rig.library.list()[0])
    assert rig.finish()["error"] is None
    assert len(rig.launches) == 2 and rig.manager.status()["engine"] == "ready"
    assert len(rig.workers[1].named("set_reference")) == 1  # a fresh worker has no reference


def test_an_internal_error_fails_the_request_and_the_worker_is_killed_and_checked(rig):
    child = FakeProc(7002)
    rig.children(child)
    worker, job = mid_job(rig)
    rig.launches[0][1].write_text("ValueError: tensor shape\n")
    worker.emit(event="error", id=job["id"], code="internal", message="ValueError: tensor shape")
    status = rig.finish()
    assert status["error"].startswith("The voice engine failed: ValueError: tensor shape.")
    assert "Last output of the voice engine" in status["error"]
    assert worker.kills == 1 and status["engine"] == "failed" and status["state"] == "idle"
    assert not child.alive and rig.tools.calls[-1] == ("kill", 7002)  # ended and checked after the tree kill
    assert worker.named("cancel") == []  # the job had already ended with that error
    assert worker.stdin.closed
    rig.auto = True
    rig.speak(rig.library.list()[0])
    assert rig.finish()["error"] is None and len(rig.launches) == 2


def test_a_worker_that_does_not_deliver_a_piece_within_a_minute_is_treated_as_dead(rig):
    worker, job = mid_job(rig)
    rig.clock += tts.PIECE_TIMEOUT - 1
    assert not rig.manager.join(0.05) and worker.kills == 0
    rig.clock += 2
    status = rig.finish()
    assert status["error"].startswith("The voice engine did not answer within 60 seconds while synthesising")
    assert worker.kills == 1 and status["engine"] == "failed" and rig.player.stops == 1
    assert rig.tmp_folders() == []


def test_each_piece_gets_its_own_minute(rig):
    worker, job = mid_job(rig)
    rig.clock += tts.PIECE_TIMEOUT - 1
    worker.audio(job, 1)
    rig.player.pieces.get(timeout=WAIT)
    rig.clock += tts.PIECE_TIMEOUT - 1
    worker.audio(job, 2)
    worker.done(job)
    assert rig.finish()["error"] is None and worker.kills == 0


def test_a_slow_rvc_pass_does_not_eat_the_next_pieces_minute(rig):
    rig.auto = False
    rig.converter.on_convert = lambda: setattr(rig, "clock", rig.clock + tts.PIECE_TIMEOUT - 1)
    rig.speak(rig.voice(model=True), preset="best")
    worker = rig.worker()
    worker.expect("set_reference")
    worker.emit(event="reference", seconds=8.0)
    job = worker.expect("synthesize")
    worker.audio(job, 0)
    rig.player.pieces.get(timeout=WAIT)  # converted (59 s on the clock) and queued
    rig.clock += tts.PIECE_TIMEOUT - 1  # the worker's minute for the next piece starts now
    assert not rig.manager.join(0.05) and worker.kills == 0
    rig.converter.on_convert = None
    worker.audio(job, 1)
    worker.audio(job, 2)
    worker.done(job)
    assert rig.finish()["error"] is None


def test_the_previous_workers_log_is_kept_when_a_new_worker_starts(rig):
    voice = rig.voice()
    rig.speak(voice)
    rig.finish()
    log_path = rig.launches[0][1]
    log_path.write_text("the first worker's last words")
    rig.workers[0].die()
    rig.speak(voice)
    rig.finish()
    assert (log_path.parent / "worker.prev.log").read_text() == "the first worker's last words"
    log_path.write_text("the second worker's last words")
    rig.manager.unload()
    rig.speak(voice)
    rig.finish()
    assert (log_path.parent / "worker.prev.log").read_text() == "the second worker's last words"
    assert sorted(p.name for p in log_path.parent.glob("worker*")) == ["worker.prev.log"]  # the fake writes none


def test_status_is_not_held_up_while_the_worker_is_being_launched(rig):
    rig.launch_gate = threading.Event()
    rig.speak()
    assert rig.launching.wait(WAIT)
    seen = []
    poll = threading.Thread(target=lambda: seen.append(rig.manager.status()))
    poll.start()
    poll.join(WAIT)
    assert not poll.is_alive(), "status() waited for the launch"
    assert seen[0]["engine"] == "starting" and seen[0]["state"] == "preparing"
    assert rig.manager.is_active()
    rig.launch_gate.set()
    assert rig.finish()["error"] is None and rig.manager.status()["engine"] == "ready"


@pytest.mark.parametrize("ending", ["stop", "unload", "shutdown"])
def test_a_worker_whose_launch_outlives_its_request_is_not_left_running(rig, ending):
    rig.children(FakeProc(7009))
    rig.launch_gate = threading.Event()
    rig.speak()
    assert rig.launching.wait(WAIT)
    ender = threading.Thread(target=getattr(rig.manager, ending))
    ender.start()
    ender.join(0.05)
    rig.launch_gate.set()  # the process exists only now, after the request was stopped
    ender.join(WAIT)
    assert not ender.is_alive()
    worker, = rig.workers
    assert worker.kills == 1 and worker.code is not None and worker.seen == []
    assert ("kill", 7009) in rig.tools.calls  # its tree was listed before the kill and swept after it
    status = rig.manager.status()
    assert status["engine"] == "stopped" and status["state"] == "idle" and status["error"] is None


def test_a_launch_that_fails_leaves_the_engine_failed_and_not_starting(rig):
    rig.launch_error = OSError(5, "Access is denied")
    rig.speak()
    status = rig.finish()
    assert status["engine"] == "failed" and status["error"] == "Could not start the voice engine: Access is denied"
    rig.launch_error = None
    rig.speak(rig.library.list()[0])
    assert rig.finish()["error"] is None and rig.manager.status()["engine"] == "ready"


def test_a_worker_that_does_not_start_within_two_minutes_is_treated_as_dead(rig):
    rig.auto = rig.ready = False
    rig.speak()
    worker = rig.worker()
    rig.clock += tts.START_TIMEOUT - 1
    assert not rig.manager.join(0.05) and rig.manager.status()["engine"] == "starting"
    rig.clock += 2
    status = rig.finish()
    assert status["error"].startswith("The voice engine did not answer within 120 seconds while starting")
    assert worker.kills == 1 and status["engine"] == "failed" and worker.seen == []


def test_a_worker_that_exits_before_it_is_ready_is_a_start_failure_with_the_end_of_its_log(rig):
    rig.auto = rig.ready = False
    rig.speak()
    worker = rig.worker()
    rig.launches[0][1].write_text("usage: tts_worker.py\ntts_worker.py: error: unrecognized arguments: --x\n")
    worker.die(2)
    status = rig.finish()
    assert status["error"].startswith("The voice engine stopped unexpectedly while starting.")
    assert "unrecognized arguments: --x" in status["error"]
    assert status["engine"] == "failed" and status["state"] == "idle"
    assert worker.kills == 0  # it had already exited: nothing is killed, its processes are only checked


def test_a_start_up_error_reported_by_the_worker_is_shown(rig):
    rig.auto = rig.ready = False
    rig.speak()
    worker = rig.worker()
    worker.emit(event="error", id=None, code="internal",
                message="The voice engine could not start. FileNotFoundError: Model files are missing")
    worker.die(1)
    status = rig.finish()
    assert status["error"].startswith("The voice engine could not start. FileNotFoundError: Model files are missing")
    assert status["engine"] == "failed"


def test_a_worker_that_cannot_be_launched_fails_the_request(rig):
    rig.launch_error = FileNotFoundError(2, "The system cannot find the file specified")
    rig.speak()
    status = rig.finish()
    assert status["error"] == "Could not start the voice engine: The system cannot find the file specified"
    assert status["engine"] == "failed" and status["state"] == "idle" and rig.player.stops == 1


def test_a_line_that_is_not_json_is_logged_and_ignored(rig, caplog):
    rig.auto = False
    rig.speak()
    worker = rig.worker()
    worker.expect("set_reference")
    with caplog.at_level(logging.WARNING, logger="aivoice.tts"):
        worker.raw(b"Loading model weights... 100%\n")
        worker.raw(b"[1, 2, 3]\n")
        worker.raw(b"\xff\xfe not text\n")
        worker.emit(event="reference", seconds=8.0)
        job = worker.expect("synthesize")
    assert sum("not a protocol line" in r.message for r in caplog.records) == 3
    rig.auto = True
    for index in range(3):
        worker.audio(job, index)
    worker.done(job)
    assert rig.finish()["error"] is None


def test_request_errors_leave_the_worker_running(rig):
    for code, message, shown in [
        ("bad_request", "piece 0 is too long for this reference clip", "piece 0 is too long for this reference clip"),
        ("piece_failed", "piece 1 produced no usable audio in two attempts",
         "The voice engine produced no sound for sentence 2"),
        ("out_of_memory", "OutOfMemoryError: HIP", "The graphics card ran out of memory while speaking. Close other "
                                                    "programs that use it and try again."),
    ]:
        worker, job = mid_job(rig, rig.library.list()[0] if rig.library.list() else None)
        extra = {"index": 1} if code == "piece_failed" else {}
        worker.emit(event="error", id=job["id"], code=code, message=message, **extra)
        status = rig.finish()
        assert status["error"] == shown and status["engine"] == "ready" and status["state"] == "idle"
        assert worker.kills == 0 and worker.named("cancel") == [] and rig.tmp_folders() == []
    assert len(rig.launches) == 1


def test_a_kill_that_leaves_a_process_running_is_tried_again_and_blocks_a_new_worker_meanwhile(rig):
    stubborn = FakeProc(7003, killable=False)
    rig.children(stubborn)
    worker, job = mid_job(rig)
    worker.die()
    status = rig.finish()
    assert status["engine"] == "failed" and status["state"] == "idle"
    voice = rig.library.list()[0]
    with pytest.raises(TtsError, match="^The voice engine of an earlier request is still shutting down "
                                       r"\(process 7003\). Try again in a moment.$"):
        rig.speak(voice)
    assert len(rig.launches) == 1 and not rig.manager.busy
    kills = rig.tools.count("kill")
    rig.manager.status()
    assert rig.tools.count("kill") == kills  # not hammered on every poll
    stubborn.killable = True
    rig.clock += tts.KILL_RETRY
    assert rig.manager.status()["engine"] == "failed"
    assert not stubborn.alive
    rig.auto = True
    rig.speak(voice)
    assert rig.finish()["error"] is None and len(rig.launches) == 2


def test_a_worker_whose_pipe_breaks_is_treated_as_dead(rig):
    rig.auto = rig.ready = False
    rig.speak()
    worker = rig.worker()
    worker.broken_pipe = True
    worker.emit(event="ready")
    status = rig.finish()
    assert status["error"].startswith("The voice engine stopped unexpectedly")
    assert worker.kills == 1 and status["engine"] == "failed"


def test_a_worker_that_died_while_idle_is_replaced_without_failing_the_next_request(rig):
    voice = rig.voice()
    rig.speak(voice)
    rig.finish()
    rig.workers[0].die()
    rig.speak(voice)
    assert rig.finish()["error"] is None
    assert len(rig.launches) == 2 and rig.manager.status()["engine"] == "ready"


# ---------------------------------------------------------------- 7. unloading

def test_the_worker_is_unloaded_after_ten_idle_minutes_when_status_is_polled(rig):
    child = FakeProc(7004)
    rig.children(child)
    rig.speak()
    rig.finish()
    worker, = rig.workers
    rig.clock += tts.IDLE_UNLOAD - 1
    assert rig.manager.status()["engine"] == "ready" and worker.kills == 0
    rig.clock += 1
    status = rig.manager.status()
    assert status["engine"] == "stopped" and status["state"] == "idle" and status["error"] is None
    assert worker.kills == 1 and worker.stdin.closed
    assert not child.alive  # one the tree kill missed is ended from the list taken beforehand
    assert tts.IDLE_UNLOAD == 600


def test_unload_ends_and_checks_processes_the_tree_kill_missed(rig):
    orphan = FakeProc(7005)
    rig.children(orphan)
    rig.speak()
    rig.finish()
    worker, = rig.workers
    worker.on_kill = lambda: rig.tools.trees.pop(worker.pid)  # the tree can no longer be listed once it is gone
    rig.manager.unload()
    assert worker.kills == 1 and not orphan.alive and ("kill", 7005) in rig.tools.calls
    assert rig.manager.status()["engine"] == "stopped"


def test_the_worker_is_never_unloaded_while_a_request_is_running(rig):
    rig.player.auto_finish = False
    rig.speak()
    for _ in range(3):
        rig.player.pieces.get(timeout=WAIT)
    assert rig.player.finished.wait(WAIT)
    worker, = rig.workers
    rig.clock += tts.IDLE_UNLOAD + 5  # a long text is still playing
    status = rig.manager.status()
    assert status["engine"] == "ready" and status["state"] == "speaking" and worker.kills == 0
    rig.player.ended.set()
    assert rig.finish()["error"] is None
    assert rig.manager.status()["engine"] == "ready"  # the ten minutes count from the end of the request
    rig.clock += tts.IDLE_UNLOAD
    assert rig.manager.status()["engine"] == "stopped" and worker.kills == 1


def test_unload_stops_the_worker_and_the_next_request_starts_a_fresh_one(rig):
    voice = rig.voice()
    rig.speak(voice)
    rig.finish()
    assert rig.manager.unload() is None
    assert rig.workers[0].kills == 1 and rig.manager.status()["engine"] == "stopped"
    rig.manager.unload()  # nothing loaded: harmless
    assert rig.workers[0].kills == 1
    rig.speak(voice)
    assert rig.finish()["error"] is None and len(rig.launches) == 2


def test_unload_during_a_request_stops_it_first(rig):
    worker, job = mid_job(rig)
    rig.auto = True
    rig.manager.unload()
    status = rig.manager.status()
    assert status["state"] == "idle" and status["engine"] == "stopped" and not rig.manager.busy
    assert rig.log.index("player.stop") < rig.log.index("kill")
    assert worker.kills == 1 and rig.tmp_folders() == []


def test_unload_says_so_when_the_worker_cannot_be_stopped(rig):
    stubborn = FakeProc(7006, killable=False)
    rig.children(stubborn)
    rig.speak()
    rig.finish()
    worker, = rig.workers
    with pytest.raises(TtsError, match=r"^The voice engine could not be stopped \(process 7006 is still running\)"):
        rig.manager.unload()
    assert rig.manager.status()["engine"] == "failed"
    with pytest.raises(TtsError, match="still shutting down"):
        rig.speak(rig.library.list()[0])  # and nothing new is started beside it
    stubborn.killable = True
    rig.manager.unload()  # asked again, it tries again at once
    assert not stubborn.alive
    assert rig.manager.status()["engine"] == "stopped"


def test_shutdown_stops_the_request_and_the_worker_and_refuses_new_requests(rig):
    worker, job = mid_job(rig)
    rig.auto = True
    rig.manager.shutdown()
    assert worker.kills == 1 and not rig.manager.busy and rig.tmp_folders() == []
    assert rig.manager.status()["engine"] == "stopped"
    with pytest.raises(TtsError, match="^The app is closing$"):
        rig.speak(rig.library.list()[0])
    rig.manager.shutdown()
    assert len(rig.launches) == 1


def kill_in_progress(rig):
    """An idle-timeout kill started by a status poll on another thread, held half-way."""
    rig.speak()
    rig.finish()
    worker, = rig.workers
    entered, release = threading.Event(), threading.Event()

    def held():
        entered.set()
        assert release.wait(WAIT)

    worker.on_kill = held
    rig.clock += tts.IDLE_UNLOAD
    poll = threading.Thread(target=rig.manager.status)
    poll.start()
    assert entered.wait(WAIT)
    return worker, release, poll


def test_unload_waits_for_a_kill_another_thread_has_in_progress(rig):
    worker, release, poll = kill_in_progress(rig)
    outcome = []

    def unload():
        try:
            rig.manager.unload()
            outcome.append("unloaded")
        except Exception as exc:  # noqa: BLE001
            outcome.append(exc)

    unloading = threading.Thread(target=unload)
    unloading.start()
    unloading.join(0.05)
    assert outcome == []  # neither a false "could not be stopped" nor a return before the worker is gone
    release.set()
    unloading.join(WAIT)
    poll.join(WAIT)
    assert outcome == ["unloaded"] and worker.kills == 1 and worker.code is not None
    assert rig.manager.status()["engine"] == "stopped"


def test_shutdown_waits_for_a_kill_another_thread_has_in_progress(rig):
    worker, release, poll = kill_in_progress(rig)
    closing = threading.Thread(target=rig.manager.shutdown)
    closing.start()
    closing.join(0.05)
    assert closing.is_alive() and worker.code is None  # the worker is still there: shutdown has not returned
    release.set()
    closing.join(WAIT)
    poll.join(WAIT)
    assert not closing.is_alive() and worker.code is not None
    assert rig.manager.status()["engine"] == "stopped"


def test_a_kill_in_progress_that_never_ends_does_not_hold_unload_forever(rig, monkeypatch):
    monkeypatch.setattr(tts, "KILL_PATIENCE", 0.01)
    worker, release, poll = kill_in_progress(rig)
    with pytest.raises(TtsError, match="could not be stopped"):
        rig.manager.unload()
    release.set()
    poll.join(WAIT)
    assert rig.manager.status()["engine"] == "stopped"  # it did end; nothing is left marked as failed


def test_unload_reports_process_tools_that_fail_as_a_readable_error(rig):
    rig.children(FakeProc(7008))
    rig.speak()
    rig.finish()

    def broken(procs, timeout):
        raise OSError(5, "Access is denied")

    rig.tools.survivors = broken
    with pytest.raises(TtsError, match=r"^The voice engine could not be stopped \(process 5000 is still running\)"):
        rig.manager.unload()
    assert rig.manager.status()["engine"] == "failed"
    del rig.tools.survivors
    rig.manager.unload()
    assert rig.manager.status()["engine"] == "stopped"


def test_shutdown_never_raises(rig):
    rig.children(FakeProc(7007, killable=False))
    rig.speak()
    rig.finish()
    rig.workers[0].kill_error = RuntimeError("taskkill is missing")
    rig.manager.shutdown()
    assert rig.manager.status()["engine"] == "failed"


# ---------------------------------------------------------------- 8. playback errors

def test_a_lost_output_device_fails_the_request_with_the_players_words_and_cancels_the_job(rig):
    worker, job = mid_job(rig, save=True)
    rig.auto = True
    rig.player.error = "An audio device was disconnected"
    status = rig.finish()
    assert status["error"] == "An audio device was disconnected" and status["state"] == "idle"
    assert worker.named("cancel") == [{"cmd": "cancel", "id": job["id"]}]
    assert worker.kills == 0 and status["engine"] == "ready" and status["saved_path"] is None
    assert rig.tmp_folders() == []


def test_a_device_lost_while_the_last_piece_plays_fails_the_request(rig):
    rig.player.auto_finish = False
    rig.speak(save=True)
    for _ in range(3):
        rig.player.pieces.get(timeout=WAIT)
    rig.player.error_at_end = "An audio device stopped responding"
    rig.player.ended.set()
    status = rig.finish()
    assert status["error"] == "An audio device stopped responding" and status["saved_path"] is None


# ---------------------------------------------------------------- 9. status

def test_status_has_exactly_the_documented_keys(rig):
    keys = {"engine", "state", "voice_id", "preset", "piece", "pieces", "error", "saved_path", "started"}
    assert rig.manager.status() == {"engine": "stopped", "state": "idle", "voice_id": None, "preset": None,
                                    "piece": 0, "pieces": 0, "error": None, "saved_path": None, "started": None}
    voice = rig.voice()
    assert set(rig.speak(voice)) == keys
    status = rig.finish()
    assert set(status) == keys
    assert status == {"engine": "ready", "state": "idle", "voice_id": voice.id, "preset": "fast", "piece": 3,
                      "pieces": 3, "error": None, "saved_path": None, "started": rig.clock}


def test_a_new_request_clears_what_the_last_one_left_on_show(rig):
    rig.audio_for = lambda index: np.zeros(24000, np.float32)
    voice = rig.voice()
    rig.speak(voice)
    assert rig.finish()["error"] is not None
    rig.audio_for = lambda index: speech()
    assert rig.speak(voice)["error"] is None
    assert rig.finish()["error"] is None


def test_status_never_kills_while_holding_the_managers_lock(rig):
    rig.speak()
    rig.finish()
    worker, = rig.workers
    free = []

    def lock_is_free():
        def probe():
            got = rig.manager._lock.acquire(timeout=WAIT)
            free.append(got)
            if got:
                rig.manager._lock.release()

        other = threading.Thread(target=probe)
        other.start()
        other.join(1)  # the lock is an RLock: only another thread can tell whether this one holds it
        if other.is_alive():
            free.append(False)

    worker.on_kill = lock_is_free
    rig.clock += tts.IDLE_UNLOAD
    assert rig.manager.status()["engine"] == "stopped"
    assert free == [True]


def test_status_neither_reads_the_worker_nor_lists_processes_while_idle_or_busy(rig):
    worker, job = mid_job(rig)
    del rig.tools.calls[:]
    seen = len(worker.seen)
    for _ in range(50):
        assert rig.manager.status()["state"] == "speaking"
    assert rig.tools.calls == [] and len(worker.seen) == seen and rig.verify_calls == [False]


def test_status_never_raises(rig, monkeypatch):
    rig.speak()
    rig.finish()
    monkeypatch.setattr(rig.manager, "_service", lambda: 1 / 0)
    assert rig.manager.status()["engine"] == "ready"
    monkeypatch.setattr(rig.manager, "_snapshot", lambda: 1 / 0)
    assert rig.manager.status() == {"engine": "stopped", "state": "idle", "voice_id": None, "preset": None,
                                    "piece": 0, "pieces": 0, "error": None, "saved_path": None, "started": None}
    assert rig.manager.status_failures == 3


def test_a_voice_deleted_while_it_is_shown_does_not_break_status(rig):
    voice = rig.voice()
    rig.speak(voice)
    rig.finish()
    rig.library.delete(voice.id)
    assert rig.manager.status()["voice_id"] == voice.id
    with pytest.raises(TtsError, match="^Unknown voice$"):
        rig.speak(voice)


# ---------------------------------------------------------------- 10. start-up recovery

def test_temporary_files_of_an_earlier_run_are_removed_and_saved_files_are_kept(tmp_path):
    rig = Rig(tmp_path)
    root = rig.data / "tts"
    (root / "tmp" / "tts-0123abcd").mkdir(parents=True)
    (root / "tmp" / "tts-0123abcd" / "tts-0123abcd_000.wav").write_bytes(b"left over")
    (root / "tmp" / "stray.part").write_bytes(b"left over")
    saved = root / "20251009T085320Z-narrator-abc123.wav"
    saved.write_bytes(b"a saved result")
    (root / "worker.log").write_text("the last run's log")
    rig.build()
    assert rig.tmp_folders() == []
    assert saved.read_bytes() == b"a saved result"
    assert (root / "worker.log").exists()


def test_construction_works_without_a_tts_folder_and_creates_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.build()
    assert not (rig.data / "tts").exists()
    assert rig.manager.status()["engine"] == "stopped"


# ---------------------------------------------------------------- the environment

def test_environment_reports_ready_and_remembers_a_passing_check(rig):
    assert rig.manager.environment() == {**PASS, "ready": True}
    rig.manager.environment()
    assert rig.verify_calls == [False]
    rig.manager.environment(refresh=True)
    assert rig.verify_calls == [False, False]


def test_a_failing_environment_is_checked_again_each_time(rig):
    rig.report = FAIL
    assert rig.manager.environment()["ready"] is False
    rig.report = PASS
    assert rig.manager.environment()["ready"] is True
    assert rig.verify_calls == [False, False]


def test_environment_never_probes_the_gpu_when_told_not_to(rig):
    assert rig.manager.environment(probe=False) == {"ok": False, "ready": False, "unknown": True, "checks": []}
    assert rig.manager.environment(refresh=True, probe=False)["unknown"] is True
    assert rig.verify_calls == []
    rig.manager.environment()
    assert rig.manager.environment(refresh=True, probe=False) == {**PASS, "ready": True}
    assert rig.verify_calls == [False]


def test_environment_never_probes_the_gpu_while_a_request_is_running(rig):
    mid_job(rig)
    assert rig.verify_calls == [False]
    assert rig.manager.environment(refresh=True) == {**PASS, "ready": True}
    assert rig.verify_calls == [False]


def test_the_environment_is_not_checked_again_while_the_worker_is_loaded(rig):
    voice = rig.voice()
    for _ in range(3):
        rig.speak(voice)
        rig.finish()
    assert rig.verify_calls == [False]  # the running worker is the proof, and a check would delay every request
    rig.manager.unload()
    rig.speak(voice)
    rig.finish()
    assert rig.verify_calls == [False, True]  # before a new worker, only the quick checks are repeated
    rig.manager.unload()
    rig.report = FAIL
    with pytest.raises(TtsError, match="^Text to speech is not set up"):
        rig.speak(voice)
    assert len(rig.launches) == 2 and not rig.manager.busy
    assert rig.manager.status()["voice_id"] == voice.id  # what the last request left on show is still there


@pytest.mark.parametrize("report", [None, {"ok": True}, {"ok": True, "checks": [{"ok": "yes"}]}, "fine"])
def test_a_malformed_report_is_a_failed_environment(rig, report):
    rig.report = report
    result = rig.manager.environment()
    assert result["ready"] is False
    assert result["checks"][0]["message"] == "The text-to-speech environment check gave no usable result"


def test_a_check_that_raises_is_a_failed_environment(rig):
    rig.verify = lambda fast: 1 / 0
    rig.build()
    assert rig.manager.environment()["ready"] is False
    with pytest.raises(TtsError, match="^Text to speech is not set up: The text-to-speech environment check could "
                                       "not be run: division by zero$"):
        rig.speak()


def test_the_default_check_runs_verify_py_with_the_backends_python(rig, monkeypatch):
    calls = []

    def fake_run(argv, **options):
        calls.append((argv, options))
        return subprocess.CompletedProcess(argv, 0, stdout="noise\n" + json.dumps(PASS) + "\n", stderr="")

    monkeypatch.setattr(tts.subprocess, "run", fake_run)
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES)
    assert manager.environment() == {**PASS, "ready": True}
    assert manager._run_verify(True) == PASS
    (full, options), (fast, _) = calls
    assert full == [sys.executable, str(rig.tts_root / "verify.py")] and fast == full + ["--fast"]
    assert options["stdin"] == subprocess.DEVNULL and options["creationflags"] == tts._NO_WINDOW
    assert options["timeout"] and "shell" not in options


def test_a_check_that_prints_nothing_usable_is_a_failed_environment(rig, monkeypatch):
    monkeypatch.setattr(tts.subprocess, "run", lambda argv, **o: subprocess.CompletedProcess(
        argv, 1, stdout="", stderr="ModuleNotFoundError: no module named json"))
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES)
    report = manager.environment()
    assert report["ready"] is False and "ModuleNotFoundError" in report["checks"][0]["message"]


# ---------------------------------------------------------------- the real launch

class FakePopen:
    calls = []

    def __init__(self, argv, **options):
        FakePopen.calls.append((argv, options))
        self.pid = 4321
        self.stdin, self.stdout = object(), object()
        self.stderr_name = options["stderr"].name
        self.stderr_closed_at_launch = options["stderr"].closed

    def poll(self):
        return None


def test_the_default_launch_is_windowless_piped_and_keeps_stderr_out_of_the_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopen, "calls", [])
    monkeypatch.setattr(tts.subprocess, "Popen", FakePopen)
    log_path = tmp_path / "worker.log"
    log_path.write_text("the previous worker's output")
    argv = [r"D:\AI Voice\tts\.venv\Scripts\python.exe", "-u", r"D:\AI Voice\tts\tts_worker.py"]
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    cache = tmp_path / "AI Voice" / "tts" / ".cache"
    handle = tts._launch_worker(argv, log_path, FakeTools(), cache)
    (launched, options), = FakePopen.calls
    assert launched == argv and isinstance(launched, list) and "shell" not in options
    assert options["stdin"] == subprocess.PIPE and options["stdout"] == subprocess.PIPE
    # stderr goes to its own file, which also drains it: never merged into stdout, never an unread pipe
    assert options["stderr"] not in (subprocess.STDOUT, subprocess.PIPE, None)
    assert Path(options["stderr"].name) == log_path and options["stderr"].closed  # the backend keeps no handle
    assert log_path.read_bytes() == b""  # each worker starts its log afresh
    assert options["creationflags"] == tts._NO_WINDOW | tts._NEW_GROUP
    env = options["env"]
    assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUNBUFFERED"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"  # synthesis never uses the network
    assert env["ROCM_SDK_TARGET_FAMILY"] == "custom" and env["WANDB_MODE"] == "disabled"
    assert "PYTHONDONTWRITEBYTECODE" not in env and env["PATH"] == os.environ["PATH"]
    assert env["MIOPEN_CUSTOM_CACHE_DIR"] == str(cache / "miopen") and (cache / "miopen" / "db").is_dir()
    assert env["MIOPEN_USER_DB_PATH"] == str(cache / "miopen" / "db")
    assert env["TEMP"] == env["TMP"] == str(cache / "tmp") and (cache / "tmp").is_dir()
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
        assert Path(env[name]).parent == cache
    assert handle.pid == 4321 and handle.poll() is None
    assert handle.stdin is not None and handle.stdout is not None


def test_the_launched_handle_kills_its_tree_the_way_trainings_does(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopen, "calls", [])
    monkeypatch.setattr(tts.subprocess, "Popen", FakePopen)
    tools = FakeTools()
    stubborn = FakeProc(7010, killable=False)
    tools.trees[4321] = [FakeProc(4321), stubborn]
    handle = tts._launch_worker(["python"], tmp_path / "worker.log", tools)
    with pytest.raises(tts.KillFailed) as failed:
        handle.kill_tree()
    assert failed.value.survivors == [stubborn]
    assert [c[0] for c in tools.calls[:2]] == ["tree", "taskkill"]  # the list is taken before anything is killed
