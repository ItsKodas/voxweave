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

from aivoice import tts, tts_audio
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


class Counter:
    """Counts something another thread does, so that a test can wait until it has happened again instead of
    sleeping: "the request looked at the clock three more times and is still running"."""

    def __init__(self):
        self.count = 0
        self._changed = threading.Condition()

    def add(self):
        with self._changed:
            self.count += 1
            self._changed.notify_all()

    def more(self, n=3):
        with self._changed:
            target = self.count + n
            assert self._changed.wait_for(lambda: self.count >= target, WAIT), "nothing is looking any more"


class WatchedEvent(threading.Event):
    """An Event that shows when somebody has begun to wait for it."""

    def __init__(self):
        super().__init__()
        self.awaited = threading.Event()

    def wait(self, timeout=None):
        self.awaited.set()
        return super().wait(timeout)


class FakePacer:
    """Stands in for the Event the housekeeping thread waits on between its rounds. The thread only goes round
    when a test says so (tick), and a test knows when that round is over: the thread is waiting again."""

    def __init__(self):
        self.waits = []  # the seconds of each wait, in order
        self._go = queue.Queue()
        self._rounds = queue.Queue()
        self._set = threading.Event()

    # what the manager calls
    def wait(self, timeout=None):
        if self.waits:
            self._rounds.put(len(self.waits))  # back from a round
        self.waits.append(timeout)
        if self._set.is_set():
            return True
        self._go.get()
        return self._set.is_set()

    def set(self):
        self._set.set()
        self._go.put(None)

    def is_set(self):
        return self._set.is_set()

    # what a test calls
    def tick(self):
        """One housekeeping round, from start to finish."""
        self.begin()
        self._rounds.get(timeout=WAIT)

    def begin(self):
        """Starts a round and returns at once, for a round a test holds half-way."""
        self._go.put(None)

    def finished(self):
        self._rounds.get(timeout=WAIT)


class FakePlayer:
    def __init__(self, rig):
        self.rig = rig
        self.starts = []  # the devices of each session
        self.gains = []  # and its gains
        self.polls = Counter()  # status() calls: the request looking at the player
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
        self.gains.append(gains)
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
        self.polls.add()
        return {"playing": not self.ended.is_set(), "queued_seconds": 0.0, "played_seconds": 0.0,
                "error": self.error}


class FakeConverter:
    def __init__(self):
        self.inputs = []
        self.outputs = []
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
        self.outputs.append((piece * 0.5).astype(np.float32))
        return self.outputs[-1]


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
        self.looks = Counter()  # how often the manager has read the clock
        self.pacer = FakePacer()
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

    def now(self):
        self.looks.add()
        return self.clock

    def build(self):
        self.pacer = FakePacer()
        self.manager = TtsManager(self.data, self.library, self.tts_root, self.player, lambda: self.devices,
                                  launch=self.launch, verify=self.verify, rvc_factory=self.rvc,
                                  now=self.now, processes=self.tools, tick=0.005, pacer=self.pacer)
        return self.manager

    def still_running(self, looks=3):
        """True once the request has looked at the clock that many more times and has not ended: it had every
        chance to act on what the test just changed."""
        self.looks.more(looks)
        return self.manager.is_active()

    def voice(self, name="Narrator", seconds=8.0, text=TRANSCRIPT, model=False, level=0.2):
        voice = self.library.create(name)
        if seconds:
            self.clip(voice, seconds, text, level)
        if model:
            folder = self.data / "voices" / voice.id / "models" / "rvc"
            folder.mkdir(parents=True)
            (folder / "model.onnx").write_bytes(b"model")
            (folder / "voice.json").write_text(json.dumps({"sample_rate": 40000}))
        return self.library.get(voice.id)

    def clip(self, voice, seconds=8.0, text=TRANSCRIPT, level=0.2, source="recorded"):
        noise = np.random.default_rng(0).uniform(-level, level, int(seconds * 48000)).astype(np.float32)
        return self.library.add_clip(voice.id, noise, text, source)

    def speak(self, voice=None, text=TEXT, preset="fast", mic_device=None, monitor_device=1, save=False, **more):
        voice = voice or self.voice()
        return self.manager.speak(voice.id, text, preset, mic_device=mic_device, monitor_device=monitor_device,
                                  save=save, **more)

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
    keeper = rig.manager._keeper
    assert keeper is None or not keeper.is_alive(), "shutdown left the housekeeping thread running"


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


def test_a_refused_reference_fails_the_request_naming_the_clip_and_is_asked_again_next_time(rig):
    rig.auto = False
    voice = rig.voice()
    rig.speak(voice)
    worker = rig.worker()
    command = worker.expect("set_reference")
    assert command["id"]  # so that a refusal can be matched to this request
    worker.emit(event="error", id=command["id"], code="bad_request",
                message="The reference clip 0001.orig.wav could not be read as a WAV file: bad header")
    status = rig.finish()
    # the worker's words, with the clip as the Voices tab shows it instead of a file name, and what can be done
    assert status["error"] == ("Clip 1, this voice's reference clip, could not be read as a WAV file: bad header. "
                               "Delete that clip or record it again.")
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
    # Fast is 10 steps, not 8: at 8 a quarter of the sentences came out with a wrong word (stage 4 notes)
    assert [j["nfe"] for j in jobs] == [10, 16, 32]
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
        assert np.abs(audio).max() == pytest.approx(0.89, abs=0.005)  # every piece at the same level
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
    assert rig.player.finished.wait(WAIT)
    rig.player.polls.more(3)  # the request has looked at the player again and again: it is waiting for it
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
    assert len(rig.converter.outputs) == 3
    for (_, played), given, returned in zip(rig.player.played, rig.converter.inputs, rig.converter.outputs):
        assert given.dtype == np.float32 and 0.95 * 48000 < len(given) < 1.15 * 48000  # trimmed, at 48 kHz
        assert 0.25 < np.abs(given).max() < 0.35  # the converter is given the piece at the level it came in
        # What was played is what the converter returned (half the level), trimmed again and raised towards the
        # level every piece is played at: by four times, the most a piece is ever raised (0.15 becomes 0.6).
        assert 0.1 < np.abs(returned).max() < 0.2
        assert np.abs(played).max() == pytest.approx(0.6, abs=0.005)
        assert np.array_equal(played, tts_audio.finish(tts_audio.trim_edges(returned, 48000), 48000, level=0.89))


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
    assert rig.manager._job.stop.wait(WAIT)
    assert rig.still_running()  # the answer is still owed: a later request must not mistake it for its own
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
    assert rig.still_running() and worker.kills == 0
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
    assert rig.still_running() and worker.kills == 0
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
    job = rig.manager._job
    ender = threading.Thread(target=getattr(rig.manager, ending))
    ender.start()
    assert job.stop.wait(WAIT)  # the request has been told to stop while its worker is still being launched
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
    assert rig.still_running() and rig.manager.status()["engine"] == "starting"
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
    engine, = rig.manager._doomed
    engine.resting = worker.resting = WatchedEvent()  # shows when somebody begins to wait for this kill
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
    assert worker.resting.awaited.wait(WAIT)  # unload has got as far as waiting for the other thread's kill
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
    assert worker.resting.awaited.wait(WAIT)  # shutdown has got as far as waiting for the other thread's kill
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


class FakeCheck:
    """Stands in for the environment check's process (subprocess.Popen)."""
    calls = []  # (argv, options) of each check started
    waits = []  # the time limit each was given
    code, out, err = 0, "", ""
    hangs = False  # the check never finishes by itself

    def __init__(self, argv, **options):
        FakeCheck.calls.append((argv, options))
        self.pid = 6000 + len(FakeCheck.calls)
        self.returncode = None
        self.communicated = 0

    def communicate(self, timeout=None):
        self.communicated += 1
        FakeCheck.waits.append(timeout)
        if FakeCheck.hangs and self.returncode is None:
            raise subprocess.TimeoutExpired("verify.py", timeout)
        self.returncode = FakeCheck.code if self.returncode is None else self.returncode
        return FakeCheck.out, FakeCheck.err

    def poll(self):
        return self.returncode


@pytest.fixture
def check(monkeypatch):
    for name, value in (("calls", []), ("waits", []), ("code", 0), ("out", ""), ("err", ""), ("hangs", False)):
        monkeypatch.setattr(FakeCheck, name, value)
    monkeypatch.setattr(tts.subprocess, "Popen", FakeCheck)
    monkeypatch.setattr(tts.subprocess, "run", lambda *a, **k: pytest.fail("subprocess.run cannot end a tree"))
    return FakeCheck


def test_the_default_check_runs_verify_py_with_the_backends_python(rig, check):
    check.out = "noise\n" + json.dumps(PASS) + "\n"
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES, processes=rig.tools)
    assert manager.environment() == {**PASS, "ready": True}
    assert manager._run_verify(True) == PASS
    (full, options), (fast, _) = check.calls
    assert full == [sys.executable, str(rig.tts_root / "verify.py")] and fast == full + ["--fast"]
    assert options["stdin"] == subprocess.DEVNULL and options["creationflags"] == tts._NO_WINDOW
    assert options["stdout"] == subprocess.PIPE and options["stderr"] == subprocess.PIPE
    assert all(check.waits) and "shell" not in options
    assert rig.tools.calls == []  # a check that ends by itself is not killed


def test_a_check_that_prints_nothing_usable_is_a_failed_environment(rig, check):
    check.code, check.err = 1, "ModuleNotFoundError: no module named json"
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES, processes=rig.tools)
    report = manager.environment()
    assert report["ready"] is False and "ModuleNotFoundError" in report["checks"][0]["message"]


def test_the_quick_check_gets_twenty_seconds_and_the_full_one_five_minutes(rig, check):
    # the quick one runs while a request holds the audio and voices locks
    check.out = json.dumps(PASS)
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES, processes=rig.tools)
    manager._run_verify(True)
    manager._run_verify(False)
    assert check.waits == [20.0, 300.0] == [tts.FAST_VERIFY_TIMEOUT, tts.VERIFY_TIMEOUT]


def test_a_check_that_runs_out_of_time_has_its_whole_process_tree_ended(rig, check):
    # the Python of a virtual environment is a launcher; the real interpreter is its child
    check.hangs = True
    child = FakeProc(6101)
    rig.tools.trees[6001] = [FakeProc(6001), child]
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES, processes=rig.tools)
    report = manager._run_verify(True)
    assert report == {"ok": False, "checks": [{"id": "verify", "ok": False, "message": (
        "The text-to-speech environment check did not finish within 20 seconds and was stopped. Try again; if it "
        "keeps happening, restart the app.")}]}
    assert [c[0] for c in rig.tools.calls[:2]] == ["tree", "taskkill"]  # listed first, then ended as a tree
    assert ("taskkill", 6001) in rig.tools.calls and not child.alive
    assert check.waits == [20.0, tts.PIPE_WAIT]  # its output is let go of, with a limit of its own


def test_a_check_whose_tree_cannot_be_ended_is_reported_in_the_log_and_still_fails_cleanly(rig, check, caplog):
    check.hangs = True
    rig.tools.trees[6001] = [FakeProc(6001), FakeProc(6102, killable=False)]
    manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: DEVICES, processes=rig.tools)
    with caplog.at_level(logging.ERROR, logger="aivoice.tts"):
        report = manager._run_verify(False)
    assert report["ok"] is False and "did not finish within 300 seconds" in report["checks"][0]["message"]
    assert any("process 6001" in r.getMessage() and "6102" in r.getMessage() for r in caplog.records)


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


# ================================================================ the final fix wave

# ---------------------------------------------------------------- housekeeping: the manager's own clock (B1)
# The screen asks for the status only while a request is running, so nothing here may depend on being asked.
# rig.pacer.tick() is one round of the housekeeping thread, from start to finish; no test waits for real time.

def idle_worker(rig, *children):
    """A worker that has served a request and is idle, and the voice it spoke in."""
    if children:
        rig.children(*children)
    voice = rig.voice()
    rig.speak(voice)
    assert rig.finish()["error"] is None
    worker, = rig.workers
    return worker, voice


def housekeeping_kill_in_progress(rig):
    """The idle-timeout kill, started by the housekeeping thread and held half-way."""
    worker, voice = idle_worker(rig)
    entered, release = threading.Event(), threading.Event()

    def held():
        entered.set()
        assert release.wait(WAIT)

    worker.on_kill = held
    rig.clock += tts.IDLE_UNLOAD
    rig.pacer.begin()
    assert entered.wait(WAIT)
    engine, = rig.manager._doomed
    engine.resting = worker.resting = WatchedEvent()
    return worker, voice, release


def test_the_worker_is_unloaded_after_ten_idle_minutes_with_nobody_asking_for_the_status(rig):
    child = FakeProc(7020)
    worker, voice = idle_worker(rig, child)
    rig.clock += tts.IDLE_UNLOAD - 1
    rig.pacer.tick()
    assert worker.kills == 0 and worker.code is None
    rig.clock += 1
    rig.pacer.tick()
    # all of it before anybody asks: the worker is gone, checked, and its pipes closed
    assert worker.kills == 1 and worker.code is not None and worker.stdin.closed
    assert not child.alive  # one the tree kill missed is ended from the list taken beforehand
    assert rig.manager._engine is None and rig.manager._doomed == []
    status = rig.manager.status()
    assert status["engine"] == "stopped" and status["state"] == "idle" and status["error"] is None
    rig.speak(voice)
    assert rig.finish()["error"] is None and len(rig.launches) == 2  # and the next request starts a fresh one


def test_a_worker_that_died_while_idle_is_noticed_with_nobody_asking_for_the_status(rig, caplog):
    orphan = FakeProc(7021)
    worker, voice = idle_worker(rig, orphan)
    worker.die()
    with caplog.at_level(logging.WARNING, logger="aivoice.tts"):
        rig.pacer.tick()
    assert any("ended while idle" in r.getMessage() for r in caplog.records)
    assert not orphan.alive and worker.kills == 0 and worker.stdin.closed  # what it left running is ended
    assert rig.manager._doomed == []
    assert rig.manager.status()["engine"] == "failed"  # the screen can say that the engine stopped
    rig.speak(voice)
    assert rig.finish()["error"] is None and rig.manager.status()["engine"] == "ready"


def housekeepers():
    return [t for t in threading.enumerate() if t.name == "tts-housekeeping"]


def test_housekeeping_starts_with_the_first_worker_and_there_is_only_ever_one(rig):
    before = len(housekeepers())
    assert rig.manager._keeper is None  # nothing to look after: no thread
    rig.manager.status()
    with pytest.raises(TtsError):
        rig.speak(text="")
    assert rig.manager._keeper is None and len(housekeepers()) == before
    worker, voice = idle_worker(rig)
    keeper = rig.manager._keeper
    assert keeper.is_alive() and keeper.daemon and len(housekeepers()) == before + 1
    rig.manager.unload()
    rig.speak(voice)
    rig.finish()
    rig.workers[-1].die()
    rig.speak(voice)
    rig.finish()
    assert len(rig.launches) == 3 and rig.manager._keeper is keeper and len(housekeepers()) == before + 1


def test_housekeeping_looks_every_thirty_seconds_and_sooner_while_a_kill_is_owed(rig):
    stubborn = FakeProc(7022, killable=False)
    worker, voice = idle_worker(rig, stubborn)
    rig.pacer.tick()
    rig.pacer.tick()
    assert rig.pacer.waits == [30.0, 30.0, 30.0] and tts.HOUSEKEEPING == 30.0
    rig.clock += tts.IDLE_UNLOAD
    rig.pacer.tick()  # the idle timeout: the kill leaves 7022 running
    assert worker.kills == 1 and stubborn.alive and rig.manager._doomed != []
    assert rig.pacer.waits[-1] == tts.KILL_RETRY  # so it looks again as soon as another try is due
    kills = rig.tools.count("kill")
    rig.pacer.tick()  # the clock has not moved: not hammered
    assert rig.tools.count("kill") == kills
    stubborn.killable = True
    rig.clock += tts.KILL_RETRY
    rig.pacer.tick()
    assert not stubborn.alive and rig.manager._doomed == [] and worker.kills == 1
    assert rig.pacer.waits[-1] == 30.0
    assert rig.manager.status()["engine"] == "stopped"


def test_housekeeping_never_kills_while_holding_the_managers_lock_and_nothing_else_is_held_up(rig):
    worker, voice, release = housekeeping_kill_in_progress(rig)
    try:
        # While the kill is in progress on the housekeeping thread, everything that only takes the lock answers.
        seen = []
        asking = threading.Thread(target=lambda: seen.extend(
            [rig.manager.status(), rig.manager.is_active(), rig.manager.busy, rig.manager.join(0)]))
        asking.start()
        asking.join(WAIT)
        assert not asking.is_alive(), "the manager's lock is held during a kill"
        # being stopped, as planned: not "failed", which the screen would report as the engine having died
        assert seen[0]["engine"] == "stopped" and seen[0]["state"] == "idle"
        assert seen[1:] == [False, False, True]
        assert worker.kills == 1  # the status poll did not start a second kill of the same worker
    finally:
        release.set()
    rig.pacer.finished()
    assert worker.kills == 1 and worker.code is not None
    assert rig.manager.status()["engine"] == "stopped"


def test_speak_during_a_housekeeping_kill_is_refused_and_works_once_the_worker_is_gone(rig):
    worker, voice, release = housekeeping_kill_in_progress(rig)
    try:
        with pytest.raises(TtsError, match="^The voice engine of an earlier request is still shutting down "
                                           r"\(process 5000\). Try again in a moment.$"):
            rig.speak(voice)
        assert len(rig.launches) == 1 and not rig.manager.busy and worker.kills == 1
    finally:
        release.set()
    rig.pacer.finished()
    rig.speak(voice)
    assert rig.finish()["error"] is None
    assert len(rig.launches) == 2 and worker.kills == 1 and rig.workers[1].kills == 0


def test_unload_during_a_housekeeping_kill_waits_for_it_and_does_not_kill_a_second_time(rig):
    worker, voice, release = housekeeping_kill_in_progress(rig)
    outcome = []

    def unload():
        try:
            rig.manager.unload()
            outcome.append("unloaded")
        except Exception as exc:  # noqa: BLE001
            outcome.append(exc)

    unloading = threading.Thread(target=unload)
    unloading.start()
    try:
        assert worker.resting.awaited.wait(WAIT)  # unload is waiting for the housekeeping thread's kill
        assert outcome == [] and worker.kills == 1 and worker.code is None
    finally:
        release.set()
    unloading.join(WAIT)
    rig.pacer.finished()
    assert outcome == ["unloaded"] and worker.kills == 1 and worker.code is not None
    assert rig.manager.status()["engine"] == "stopped"


def test_shutdown_during_a_housekeeping_kill_waits_for_it_and_the_thread_ends(rig):
    worker, voice, release = housekeeping_kill_in_progress(rig)
    keeper = rig.manager._keeper
    closing = threading.Thread(target=rig.manager.shutdown)
    closing.start()
    try:
        assert worker.resting.awaited.wait(WAIT)
        assert closing.is_alive() and keeper.is_alive() and worker.code is None
    finally:
        release.set()
    closing.join(WAIT)
    assert not closing.is_alive() and not keeper.is_alive()  # neither a process nor a thread is left
    assert worker.kills == 1 and worker.code is not None and rig.manager._doomed == []
    assert rig.manager.status()["engine"] == "stopped"


def test_shutdown_waits_for_a_housekeeping_round_in_progress_before_it_returns(rig, monkeypatch):
    worker, voice = idle_worker(rig)
    entered, release = threading.Event(), threading.Event()
    service = rig.manager._service

    def slow():
        entered.set()
        assert release.wait(WAIT)
        service()

    monkeypatch.setattr(rig.manager, "_service", slow)
    keeper = rig.manager._keeper
    joining, join = threading.Event(), keeper.join
    keeper.join = lambda timeout=None: (joining.set(), join(timeout))[1]
    rig.pacer.begin()
    assert entered.wait(WAIT)
    closing = threading.Thread(target=rig.manager.shutdown)
    closing.start()
    try:
        assert joining.wait(WAIT), "shutdown returned without waiting for the housekeeping thread"
        assert closing.is_alive() and keeper.is_alive()
    finally:
        release.set()
    closing.join(WAIT)
    assert not closing.is_alive() and not keeper.is_alive() and worker.kills == 1


def test_shutdown_stops_the_housekeeping_thread_and_none_is_started_afterwards(rig):
    worker, voice = idle_worker(rig)
    keeper = rig.manager._keeper
    assert keeper.is_alive()
    rounds = len(rig.pacer.waits)
    rig.manager.shutdown()
    assert not keeper.is_alive() and worker.kills == 1
    assert len(rig.pacer.waits) == rounds  # it did not go round again on its way out
    with pytest.raises(TtsError, match="^The app is closing$"):
        rig.speak(voice)
    rig.manager.shutdown()  # twice is harmless
    assert rig.manager._keeper is keeper and not keeper.is_alive()


def test_shutdown_before_any_worker_leaves_no_thread(rig):
    rig.manager.shutdown()
    assert rig.manager._keeper is None and rig.pacer.is_set()
    with pytest.raises(TtsError, match="^The app is closing$"):
        rig.speak()
    assert rig.manager._keeper is None and rig.launches == []


def test_shutdown_while_the_first_worker_is_being_launched_leaves_neither_thread_nor_process(rig):
    rig.launch_gate = threading.Event()
    rig.speak()
    assert rig.launching.wait(WAIT)
    job, keeper = rig.manager._job, rig.manager._keeper
    assert keeper.is_alive()  # started with the launch: the worker is looked after from its first moment
    closing = threading.Thread(target=rig.manager.shutdown)
    closing.start()
    assert job.stop.wait(WAIT)
    rig.launch_gate.set()
    closing.join(WAIT)
    assert not closing.is_alive() and not keeper.is_alive()
    worker, = rig.workers
    assert worker.kills == 1 and worker.code is not None and rig.manager._doomed == []


def test_housekeeping_leaves_a_request_and_its_worker_alone_and_stop_still_works(rig):
    worker, job = mid_job(rig)
    seen = len(worker.seen)
    rig.pacer.tick()
    assert worker.kills == 0 and len(worker.seen) == seen and rig.manager.is_active()
    assert rig.manager.status()["state"] == "speaking"
    rig.auto = True
    status = rig.manager.stop()
    rig.pacer.tick()
    assert status["state"] == "idle" and worker.kills == 0 and rig.manager.status()["engine"] == "ready"


def test_the_idle_time_counts_from_the_end_of_the_last_request(rig):
    worker, job = mid_job(rig)
    rig.manager._last_used -= tts.IDLE_UNLOAD  # the request has been running for longer than the idle time
    rig.pacer.tick()
    assert worker.kills == 0
    rig.auto = True
    rig.manager.stop()
    rig.pacer.tick()  # just ended: the ten minutes start now
    assert worker.kills == 0 and rig.manager.status()["engine"] == "ready"
    rig.clock += tts.IDLE_UNLOAD
    rig.pacer.tick()
    assert worker.kills == 1


def test_a_housekeeping_round_that_fails_does_not_end_the_thread(rig, monkeypatch, caplog):
    worker, voice = idle_worker(rig)
    service = rig.manager._service
    monkeypatch.setattr(rig.manager, "_service", lambda: 1 / 0)
    with caplog.at_level(logging.ERROR, logger="aivoice.tts"):
        rig.pacer.tick()
        rig.pacer.tick()
    assert rig.manager.housekeeping_failures == 2
    assert len([r for r in caplog.records if "housekeeping failed" in r.getMessage()]) == 1  # logged once
    monkeypatch.setattr(rig.manager, "_service", service)
    rig.clock += tts.IDLE_UNLOAD
    rig.pacer.tick()
    assert worker.kills == 1 and rig.manager._keeper.is_alive()


class PickyThread(threading.Thread):
    """A Thread that cannot be started when it has one of the refused names."""
    refused = ()

    def start(self):
        if self.name in PickyThread.refused:
            raise RuntimeError("can't start new thread")
        super().start()


def test_a_housekeeping_thread_that_cannot_be_started_does_not_fail_the_request(rig, monkeypatch, caplog):
    monkeypatch.setattr(PickyThread, "refused", ("tts-housekeeping",))
    monkeypatch.setattr(tts.threading, "Thread", PickyThread)
    with caplog.at_level(logging.WARNING, logger="aivoice.tts"):
        worker, voice = idle_worker(rig)
    assert rig.manager._keeper is None
    assert any("housekeeping thread" in r.getMessage() for r in caplog.records)
    rig.clock += tts.IDLE_UNLOAD
    assert rig.manager.status()["engine"] == "stopped" and worker.kills == 1  # the screen's poll still does it
    monkeypatch.setattr(PickyThread, "refused", ())
    rig.speak(voice)
    rig.finish()
    assert rig.manager._keeper.is_alive()  # and the next worker gets its thread


# ---------------------------------------------------------------- the reference clip (B2)

def test_a_reference_clip_that_is_too_quiet_is_passed_over_for_the_next_best_one(rig):
    voice = rig.voice(level=0.001)  # clip 1: eight seconds, the preferred length, but far too quiet to copy
    rig.clip(voice, seconds=4.0, text="A louder clip that a voice can be copied from.")
    voice = rig.library.get(voice.id)
    rig.speak(voice)
    assert rig.finish()["error"] is None
    command, = rig.workers[0].named("set_reference")
    assert Path(command["wav"]).name == "0002.orig.wav"
    assert command["text"] == "A louder clip that a voice can be copied from."


def test_a_voice_whose_clips_are_all_too_quiet_is_refused_before_anything_is_started(rig):
    voice = rig.voice(level=0.001)
    with pytest.raises(TtsError) as refused:
        rig.speak(voice)
    assert str(refused.value) == "Clip 1 is too quiet to copy. Record a louder clip of 5 to 10 seconds."
    nothing_started(rig)
    clips = rig.data / "voices" / voice.id / "clips"
    assert sorted(p.name for p in clips.iterdir()) == ["0001.orig.wav", "0001.txt", "0001.wav"]  # only read


@pytest.mark.parametrize("said, shown", [
    ("The reference clip 0001.orig.wav is too quiet to copy a voice from. Record it louder, or choose another clip",
     "Clip 1 is too quiet to copy. Record a louder clip of 5 to 10 seconds."),
    ("The reference clip 0001.orig.wav is too long: it is 12.4 seconds, and a reference must be 3 to 12 seconds",
     "Clip 1, this voice's reference clip, is too long: it is 12.4 seconds, and a reference must be 3 to 12 "
     "seconds. Delete that clip or record it again."),
    ("The reference clip 0001.orig.wav was not found",
     "Clip 1, this voice's reference clip, was not found. Delete that clip or record it again."),
    ("The reference clip 0001.orig.wav could not be read: Format not recognised. Use it or choose another clip.",
     "Clip 1, this voice's reference clip, could not be read: Format not recognised. Use it. Delete that clip or "
     "record it again."),
    ("Something about 0001.orig.wav went wrong",
     "Something about clip 1 went wrong. Delete that clip or record it again."),
    ("The reference cannot change while tts-1 is being synthesised",  # not about the clip: as the worker said it
     "The reference cannot change while tts-1 is being synthesised"),
])
def test_what_the_worker_says_about_a_reference_file_is_put_in_terms_of_the_clip(rig, said, shown):
    rig.auto = False
    rig.speak()
    worker = rig.worker()
    command = worker.expect("set_reference")
    worker.emit(event="error", id=command["id"], code="bad_request", message=said)
    status = rig.finish()
    assert status["error"] == shown
    assert "orig.wav" not in status["error"] and "choose another clip" not in status["error"]
    assert worker.kills == 0 and status["engine"] == "ready"  # a refused clip is no fault of the engine


def test_the_clip_number_in_a_refusal_is_the_one_the_voices_tab_shows(rig):
    voice = rig.voice(seconds=2.0)  # clip 1 is too short to be a reference
    rig.clip(voice, seconds=20.0)  # clip 2 too long
    rig.clip(voice, seconds=6.0, text="The third clip.")
    rig.auto = False
    rig.speak(rig.library.get(voice.id))
    worker = rig.worker()
    command = worker.expect("set_reference")
    assert Path(command["wav"]).name == "0003.orig.wav"
    worker.emit(event="error", id=command["id"], code="bad_request",
                message="The reference clip 0003.orig.wav is too quiet to copy a voice from. Record it louder, or "
                        "choose another clip")
    assert rig.finish()["error"] == "Clip 3 is too quiet to copy. Record a louder clip of 5 to 10 seconds."


# ---------------------------------------------------------------- the listen volume (B4)

def test_the_listen_volume_is_applied_to_the_listen_device_only(rig):
    rig.speak(mic_device=2, monitor_device=1, monitor_gain=0.35)
    rig.finish()
    assert rig.player.starts == [{"mic": 2, "monitor": 1}] and rig.player.gains == [{"monitor": 0.35}]


@pytest.mark.parametrize("asked, used", [(2.0, 1 / 0.89), (1.5, 1 / 0.89), (1.13, 1 / 0.89), (1.12, 1.12),
                                         (1.0, 1.0), (0.0, 0.0)])
def test_the_listen_volume_is_held_where_a_piece_at_full_level_does_not_clip(rig, asked, used):
    # every piece has its peak at 0.89, and the volume is applied after that: above 1.12 it would clip
    rig.speak(monitor_gain=asked)
    rig.finish()
    assert rig.player.gains == [{"monitor": pytest.approx(used)}]
    assert rig.player.gains[0]["monitor"] * tts_audio.PIECE_PEAK <= 1.0 + 1e-9


def test_without_a_listen_device_or_a_volume_nothing_is_turned_down(rig):
    voice = rig.voice()
    rig.speak(voice, mic_device=2, monitor_device=None, monitor_gain=0.35)
    rig.finish()
    rig.speak(voice)
    rig.finish()
    assert rig.player.gains == [{}, {"monitor": 1.0}]


@pytest.mark.parametrize("gain", [-0.1, 2.01, float("nan"), "loud", None, True])
def test_a_listen_volume_outside_live_modes_range_is_refused(rig, gain):
    with pytest.raises(TtsError, match="^monitor_gain must be between 0 and 2$"):
        rig.speak(monitor_gain=gain)
    nothing_started(rig)


# ---------------------------------------------------------------- one level for every piece (B5)

def test_pieces_of_different_raw_level_are_played_at_the_same_level(rig):
    # as the worker hands them over: anything from a quarter of full scale to full scale
    levels = [0.9, 0.5, 0.25]
    rig.audio_for = lambda index: speech(level=levels[index])
    rig.speak(save=True)
    status = rig.finish()
    assert status["error"] is None
    peaks = [float(np.abs(audio).max()) for _, audio in rig.player.played]
    assert peaks == pytest.approx([0.89] * 3, abs=0.005)
    saved, _ = read_wav(status["saved_path"])
    assert float(np.abs(saved).max()) == pytest.approx(0.89, abs=0.005)  # and saved as played


def test_a_piece_that_is_nearly_nothing_is_raised_four_times_at_most(rig):
    # room tone or a click that passed for sound, or a very quiet reference clip: never full scale into the mic
    levels = [0.3, 0.01, 0.1]
    rig.audio_for = lambda index: speech(level=levels[index])
    rig.speak()
    assert rig.finish()["error"] is None
    peaks = [float(np.abs(audio).max()) for _, audio in rig.player.played]
    assert peaks == pytest.approx([0.89, 0.04, 0.4], rel=0.02)


def test_a_piece_that_is_silent_is_still_refused_and_never_raised_to_the_level(rig):
    rig.audio_for = lambda index: speech(level=0.3 if index == 0 else 0.004)
    rig.speak()
    status = rig.finish()
    assert status["error"] == "The voice engine produced no sound for sentence 2"
    assert len(rig.player.played) == 1


# ---------------------------------------------------------------- one environment probe at a time (B6)

def test_a_second_caller_waits_for_the_environment_check_in_progress_and_takes_its_result(rig, monkeypatch):
    probes = []

    class WatchedProbe(tts._Probe):
        def __init__(self):
            super().__init__()
            self.done = WatchedEvent()
            probes.append(self)

    monkeypatch.setattr(tts, "_Probe", WatchedProbe)
    entered, release = threading.Event(), threading.Event()

    def slow(fast):
        rig.verify_calls.append(fast)
        entered.set()
        assert release.wait(WAIT)
        return PASS

    rig.verify = slow
    rig.build()
    results = []
    first = threading.Thread(target=lambda: results.append(rig.manager.environment()))
    first.start()
    assert entered.wait(WAIT)
    second = threading.Thread(target=lambda: results.append(rig.manager.environment(refresh=True)))
    second.start()
    try:
        assert probes[0].done.awaited.wait(WAIT), "the second caller did not wait for the check in progress"
        assert rig.verify_calls == [False] and results == []
    finally:
        release.set()
    first.join(WAIT)
    second.join(WAIT)
    assert results == [{**PASS, "ready": True}] * 2
    assert rig.verify_calls == [False] and len(probes) == 1  # one GPU process, not two
    rig.manager.environment(refresh=True)
    assert rig.verify_calls == [False, False]  # a later refresh checks again


def test_a_request_takes_the_result_of_the_environment_check_in_progress(rig, monkeypatch):
    probes = []

    class WatchedProbe(tts._Probe):
        def __init__(self):
            super().__init__()
            self.done = WatchedEvent()
            probes.append(self)

    monkeypatch.setattr(tts, "_Probe", WatchedProbe)
    entered, release = threading.Event(), threading.Event()

    def slow(fast):
        rig.verify_calls.append(fast)
        if len(rig.verify_calls) == 1:
            entered.set()
            assert release.wait(WAIT)
        return FAIL

    rig.verify = slow
    rig.build()
    voice = rig.voice()
    screen = threading.Thread(target=rig.manager.environment)
    screen.start()
    assert entered.wait(WAIT)
    refused = []

    def speak():
        try:
            rig.speak(voice)
        except TtsError as exc:
            refused.append(str(exc))

    speaking = threading.Thread(target=speak)
    speaking.start()
    try:
        assert probes[0].done.awaited.wait(WAIT), "the request did not wait for the check in progress"
    finally:
        release.set()
    screen.join(WAIT)
    speaking.join(WAIT)
    assert refused == ["Text to speech is not set up: torch sees no GPU; model files wrong"]
    assert rig.verify_calls == [False] and not rig.manager.busy and rig.launches == []


def test_a_check_in_progress_that_raises_does_not_leave_the_next_caller_waiting(rig):
    calls = []

    def broken(fast):
        calls.append(fast)
        raise KeyboardInterrupt  # not caught by the manager: the guard must still be let go of

    rig.verify = broken
    rig.build()
    with pytest.raises(KeyboardInterrupt):
        rig.manager.environment()
    assert rig.manager._probe is None and calls == [False]
    del rig.verify
    rig.manager._verify = rig.verify
    assert rig.manager.environment()["ready"] is True


# ---------------------------------------------------------------- saving (B8)

def test_a_save_whose_write_fails_half_way_leaves_nothing_behind(rig, monkeypatch):
    import wave

    real = wave.Wave_write.writeframes
    results = rig.data / "tts"

    def full(self, data):
        if Path(self._file.name).parent != results:  # a clip of the voice, a piece from the worker: as ever
            return real(self, data)
        real(self, data[:1000])  # some of the result reaches the disk
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(wave.Wave_write, "writeframes", full)
    rig.speak(save=True)
    status = rig.finish()
    assert status["error"] == "The speech was played but could not be saved: No space left on device"
    assert status["saved_path"] is None
    left = sorted(p.name for p in (rig.data / "tts").iterdir())
    assert left in (["tmp"], ["tmp", "worker.log"]), left  # neither a half file nor its temporary name


def test_a_save_is_written_under_another_name_and_moved_into_place(rig, monkeypatch):
    from aivoice import audiofile

    moves = []
    real = audiofile.replace_file

    def replace(src, dst):
        if Path(dst).parent == rig.data / "tts":
            moves.append((Path(src).name, Path(dst).name, Path(dst).exists()))
        real(src, dst)

    monkeypatch.setattr(audiofile, "replace_file", replace)
    voice = rig.voice()
    rig.speak(voice, save=True)
    status = rig.finish()
    name = Path(status["saved_path"]).name
    assert moves == [(name + ".tmp", name, False)]  # whole before it has its name; never over another file


# ---------------------------------------------------------------- loose ends of the launch (D3)

def test_unload_says_so_in_the_log_when_a_launch_is_still_in_flight(rig, monkeypatch, caplog):
    monkeypatch.setattr(tts, "STOP_WAIT", 0)  # the launch takes longer than unload is prepared to wait
    rig.children(FakeProc(7030))
    rig.launch_gate = threading.Event()
    rig.speak()
    assert rig.launching.wait(WAIT)
    with caplog.at_level(logging.WARNING, logger="aivoice.tts"):
        rig.manager.unload()  # returns: there is nothing it could end yet
    assert any("still being started" in r.getMessage() for r in caplog.records)
    assert rig.workers == [] and rig.manager.is_active()
    rig.launch_gate.set()
    assert rig.manager.join(WAIT)
    worker, = rig.workers
    assert worker.kills == 1 and worker.code is not None and ("kill", 7030) in rig.tools.calls
    assert rig.manager.status()["engine"] == "stopped" and rig.manager._doomed == []


def test_unload_says_nothing_about_a_launch_when_there_is_none(rig, caplog):
    idle_worker(rig)
    with caplog.at_level(logging.WARNING, logger="aivoice.tts"):
        rig.manager.unload()
    assert not any("still being started" in r.getMessage() for r in caplog.records)


def test_a_worker_whose_reader_thread_cannot_be_started_is_ended_and_the_request_fails(rig, monkeypatch):
    rig.children(FakeProc(7031))
    monkeypatch.setattr(PickyThread, "refused", ("tts-events",))
    monkeypatch.setattr(tts.threading, "Thread", PickyThread)
    voice = rig.voice()
    rig.speak(voice)
    status = rig.finish()
    assert status["error"] == "Could not start the voice engine: can't start new thread"
    worker, = rig.workers
    assert worker.kills == 1 and worker.code is not None and worker.seen == []  # ended, and never spoken to
    assert ("kill", 7031) in rig.tools.calls and rig.manager._doomed == []
    assert status["engine"] == "failed" and status["state"] == "idle"
    monkeypatch.setattr(PickyThread, "refused", ())
    rig.speak(voice)
    assert rig.finish()["error"] is None and len(rig.launches) == 2


class PipedPopen(FakePopen):
    """A launched process whose pipes can be seen to be closed."""

    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    def __init__(self, argv, **options):
        super().__init__(argv, **options)
        self.stdin, self.stdout = PipedPopen.Pipe(), PipedPopen.Pipe()
        PipedPopen.last = self


@pytest.mark.parametrize("fault", [RuntimeError("no handle"), KeyboardInterrupt()])
def test_a_launch_that_fails_after_the_process_exists_ends_its_tree_before_the_error_goes_on(tmp_path, monkeypatch,
                                                                                             fault):
    monkeypatch.setattr(FakePopen, "calls", [])
    monkeypatch.setattr(tts.subprocess, "Popen", PipedPopen)
    tools = FakeTools()
    child = FakeProc(7032)
    tools.trees[4321] = [FakeProc(4321), child]

    def no_handle(proc, process_tools):
        assert tools.calls == []  # nothing has been ended yet: the process exists and is nobody's
        raise fault

    monkeypatch.setattr(tts, "_Worker", no_handle)
    with pytest.raises(type(fault)):
        tts._launch_worker(["python"], tmp_path / "worker.log", tools)
    assert [c[0] for c in tools.calls[:2]] == ["tree", "taskkill"] and ("taskkill", 4321) in tools.calls
    assert not child.alive
    assert PipedPopen.last.stdin.closed and PipedPopen.last.stdout.closed


def test_a_launch_that_fails_and_cannot_end_its_process_names_it_in_the_log(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(FakePopen, "calls", [])
    monkeypatch.setattr(tts.subprocess, "Popen", PipedPopen)
    tools = FakeTools()
    tools.trees[4321] = [FakeProc(4321), FakeProc(7033, killable=False)]
    monkeypatch.setattr(tts, "_Worker", lambda proc, process_tools: 1 / 0)
    with caplog.at_level(logging.ERROR, logger="aivoice.tts"), pytest.raises(ZeroDivisionError):
        tts._launch_worker(["python"], tmp_path / "worker.log", tools)
    assert any("process 4321" in r.getMessage() and "7033" in r.getMessage() for r in caplog.records)


def test_a_launch_that_fails_before_the_process_exists_ends_nothing(tmp_path, monkeypatch):
    def refused(argv, **options):
        raise OSError(2, "The system cannot find the file specified")

    monkeypatch.setattr(tts.subprocess, "Popen", refused)
    tools = FakeTools()
    with pytest.raises(OSError):
        tts._launch_worker(["python"], tmp_path / "worker.log", tools)
    assert tools.calls == []


# ---------------------------------------------------------------- numbers (D4)

def test_the_numbers_the_model_misreads_are_written_out_for_the_worker_and_only_those(rig):
    text = "It costs 1,200 dollars in 1987. Call 555 0142 at 12:30. That is 3.5 percent of 40."
    rig.speak(text=text)
    assert rig.finish()["error"] is None
    command, = rig.workers[0].named("synthesize")
    assert command["pieces"] == ["It costs one thousand two hundred dollars in 1987.",
                                 "Call five five five, zero one four two at 12:30.",
                                 "That is 3.5 percent of 40."]
    assert all(len(p.encode("utf-8")) <= command["max_chars"] for p in command["pieces"])


def test_the_length_limit_is_on_the_text_as_typed_not_on_what_the_numbers_become(rig):
    typed = ("999,999,999,999 " * 60).strip()  # 959 characters as typed; over 7000 once written out
    rig.speak(text=typed)
    status = rig.finish()
    assert status["error"] is None
    command, = rig.workers[0].named("synthesize")
    assert sum(len(p) for p in command["pieces"]) > 7000 > 5000 and status["pieces"] == len(command["pieces"])
    assert all(len(p.encode("utf-8")) <= command["max_chars"] for p in command["pieces"])  # each still fits
    with pytest.raises(TtsError, match=r"^The text is too long \(5999 characters\)"):
        rig.speak(rig.library.list()[0], text=("1,200 " * 1000).strip())


def test_the_pieces_are_written_out_again_when_the_worker_measures_the_reference_differently(rig):
    rig.reference_seconds = 7.0  # the clip is on record as 8 s: the limit changes, so the text is split again
    rig.speak(text="It costs 1,200 dollars. " * 12)
    assert rig.finish()["error"] is None
    command, = rig.workers[0].named("synthesize")
    assert all("one thousand two hundred" in p and "1,200" not in p for p in command["pieces"])
    assert all(len(p.encode("utf-8")) <= command["max_chars"] for p in command["pieces"])
