"""The text-to-speech worker (tts/tts_worker.py) and its start-up shims (tts/shims.py).

Nothing here needs the TTS environment, torch, a GPU or a model: the protocol layer is driven with a fake engine
and in-memory streams, and the subprocess tests run the worker under this (the backend's) Python, with a tiny
engine that only makes noise on stdout or with no model files at all. No test sleeps: a test waits for the worker's next event, and where it must
know that a command without an answer (cancel) has been handled it sends a marker command and waits for its reply.
"""
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from aivoice.audiofile import read_wav

TTS_DIR = Path(__file__).resolve().parents[2] / "tts"
MODULE = TTS_DIR / "tts_worker.py"
SHIMS = TTS_DIR / "shims.py"
WAIT = 20.0  # seconds a test waits for an event before it fails; never reached when the worker behaves
RATE = 24000
TRANSCRIPT = "Some call me nature, others call me mother nature."
windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows process handling")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker():
    return load(MODULE, "aivoice_tts_worker")


# ---------------------------------------------------------------- fakes

def make_wav(path, seconds, rate=48000, width=2):
    samples = int(round(seconds * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x01" * (samples * width))
    return path


def tone(seconds=1.0, peak=0.3, rate=RATE):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (np.sin(2 * np.pi * 220.0 * t) * peak).astype(np.float32)


class OutOfMemoryError(RuntimeError):
    """Named like torch.OutOfMemoryError, which the tests cannot import."""


class FakeEngine:
    """Returns one second of tone per piece unless `plan` holds something else for the next call."""

    def __init__(self):
        self.loads = 0
        self.references = []
        self.calls = []  # (text, nfe, speed, seed)
        self.plan = []  # one entry per coming synthesize_piece call: a result, an exception, or a callable
        self.load_error = None
        self.reference_seconds = None  # overrides the clip's real length
        self.reference_error = None

    def load(self):
        self.loads += 1
        if self.load_error:
            raise self.load_error
        return {"device": "fake", "sample_rate": RATE}

    def set_reference(self, wav_path, text):
        self.references.append((wav_path, text))
        if self.reference_error:
            raise self.reference_error
        if self.reference_seconds is not None:
            return self.reference_seconds
        with wave.open(wav_path, "rb") as w:
            return w.getnframes() / w.getframerate()

    def synthesize_piece(self, text, nfe, speed, seed, should_stop):
        self.calls.append((text, nfe, speed, seed))
        step = self.plan.pop(0) if self.plan else None
        if step is None:
            return tone(), RATE
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(should_stop)
        return step


class Gate:
    """A piece that waits inside the engine until the test lets it go on."""

    def __init__(self, worker, result=None, obey_stop=True):
        self.worker, self.result, self.obey_stop = worker, result, obey_stop
        self.entered, self.release = threading.Event(), threading.Event()
        self.saw_stop = None

    def __call__(self, should_stop):
        self.entered.set()
        assert self.release.wait(WAIT), "the test never released the piece"
        self.saw_stop = should_stop()
        if self.saw_stop and self.obey_stop:
            raise self.worker.Stopped()
        return self.result if self.result is not None else (tone(), RATE)


class Stdin:
    """What the worker reads: the test adds lines, and close() is the end of input."""

    def __init__(self):
        self._lines = queue.Queue()

    def write(self, text):
        self._lines.put(text)

    def close(self):
        self._lines.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        line = self._lines.get()
        if line is None:
            raise StopIteration
        return line


class Stdout:
    """What the worker writes. Text becomes visible to the test only when the worker flushes, so a test that
    receives an event has also shown that its line was flushed."""

    def __init__(self):
        self._cond = threading.Condition()
        self._unflushed = ""
        self.text = ""
        self.events = []
        self.files_present = {}  # audio path -> (the file existed, its .part did not) at the moment of the event
        self.finished = False
        self._taken = 0

    def write(self, text):
        with self._cond:
            self._unflushed += text
        return len(text)

    def flush(self):
        with self._cond:
            text, self._unflushed = self._unflushed, ""
            assert text == "" or text.endswith("\n"), f"flushed a partial line: {text!r}"
            self.text += text
            for line in text.splitlines():
                event = json.loads(line)
                if event.get("event") == "audio":
                    path = event["path"]
                    self.files_present[path] = os.path.isfile(path) and not os.path.exists(path + ".part")
                self.events.append(event)
            self._cond.notify_all()

    def finish(self):
        with self._cond:
            self.finished = True
            self._cond.notify_all()

    def next(self):
        with self._cond:
            ok = self._cond.wait_for(lambda: len(self.events) > self._taken or self.finished, WAIT)
            assert ok, "no event arrived in time"
            assert len(self.events) > self._taken, "the worker ended without another event"
            self._taken += 1
            return self.events[self._taken - 1]

    def untaken(self):
        with self._cond:
            return self.events[self._taken:]


class Session:
    def __init__(self, worker, engine=None, start=True, stdout=None, **serve_args):
        self.worker = worker
        self.engine = engine or FakeEngine()
        self.stdin, self.stdout = Stdin(), stdout or Stdout()
        self.code = None
        self._serve_args = serve_args
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if start:
            assert self.next()["event"] == "ready"

    def _run(self):
        try:
            self.code = self.worker.serve(self.stdin, self.stdout, self.engine, **self._serve_args)
        finally:
            self.stdout.finish()

    def send(self, **command):
        self.stdin.write(json.dumps(command) + "\n")

    def next(self, event=None):
        got = self.stdout.next()
        if event is not None:
            assert got["event"] == event, got
        return got

    def error(self, code):
        got = self.next("error")
        assert got["code"] == code, got
        assert isinstance(got["message"], str) and got["message"]
        return got

    def sync(self):
        """Every command sent before this one has been handled when this returns."""
        self.send(cmd="marker")
        self.error("unknown_command")

    def reference(self, tmp_path, seconds=5.0, text=TRANSCRIPT):
        wav = make_wav(tmp_path / "reference.wav", seconds)
        self.send(cmd="set_reference", wav=str(wav), text=text)
        return self.next("reference")

    def synthesize(self, tmp_path, id="job", pieces=("One sentence to say.",), nfe=16, **more):
        self.send(cmd="synthesize", id=id, pieces=list(pieces), out_dir=str(tmp_path / "out"), nfe=nfe, **more)

    def run_job(self, tmp_path, **args):
        """Send a job and return its events up to and including the one that ends it."""
        self.synthesize(tmp_path, **args)
        events = []
        while not events or events[-1]["event"] not in ("done", "cancelled", "error"):
            events.append(self.next())
        return events

    def end(self, close=True):
        """Close the input (unless the test already ended the worker) and return the exit code."""
        if close:
            self.stdin.close()
        self._thread.join(WAIT)
        assert not self._thread.is_alive(), "the worker did not end"
        return self.code

    def close(self):
        self.stdin.close()
        self._thread.join(WAIT)


@pytest.fixture
def session(worker):
    made = []

    def start(engine=None, **args):
        made.append(Session(worker, engine, **args))
        return made[-1]

    yield start
    for one in made:
        one.close()


def names(events):
    return [e["event"] for e in events]


# ---------------------------------------------------------------- 1. start-up

def test_ready_is_emitted_once_after_the_engine_loaded(session):
    s = session(start=False)
    ready = s.next("ready")
    assert s.engine.loads == 1
    assert ready["device"] == "fake" and ready["sample_rate"] == RATE
    assert s.end() == 0
    assert names(s.stdout.events) == ["ready"]


def test_a_load_failure_is_an_internal_error_and_a_non_zero_exit(session):
    engine = FakeEngine()
    engine.load_error = RuntimeError("the GPU is not available")
    s = session(engine, start=False)
    error = s.error("internal")
    assert "the GPU is not available" in error["message"]
    assert s.end(close=False) != 0
    assert names(s.stdout.events) == ["error"]


# ---------------------------------------------------------------- 2. set_reference

def test_set_reference_passes_the_clip_and_transcript_to_the_engine(session, tmp_path):
    s = session()
    event = s.reference(tmp_path, seconds=5.0, text="  " + TRANSCRIPT + "\n")
    assert event["seconds"] == pytest.approx(5.0, abs=0.001)
    assert s.engine.references == [(str(tmp_path / "reference.wav"), TRANSCRIPT)]


@pytest.mark.parametrize("seconds", [3.0, 12.0])
def test_a_reference_at_either_end_of_the_range_is_accepted(session, tmp_path, seconds):
    s = session()
    assert s.reference(tmp_path, seconds=seconds)["seconds"] == pytest.approx(seconds, abs=0.001)


@pytest.mark.parametrize("seconds, word", [(2.9, "short"), (12.1, "long"), (31.2, "long")])
def test_a_reference_outside_3_to_12_seconds_is_refused(session, tmp_path, seconds, word):
    s = session()
    wav = make_wav(tmp_path / "clip.wav", seconds)
    s.send(cmd="set_reference", wav=str(wav), text=TRANSCRIPT)
    error = s.error("bad_request")
    assert word in error["message"] and "3 to 12 seconds" in error["message"]
    assert f"{seconds:.1f}" in error["message"]
    assert s.engine.references == []


def test_a_missing_reference_file_is_refused(session, tmp_path):
    s = session()
    s.send(cmd="set_reference", wav=str(tmp_path / "nothing.wav"), text=TRANSCRIPT)
    assert "nothing.wav" in s.error("bad_request")["message"]
    s.send(cmd="set_reference", wav=str(tmp_path), text=TRANSCRIPT)  # a folder is not a clip either
    s.error("bad_request")
    assert s.engine.references == []


@pytest.mark.parametrize("text", ["", "   \n"])
def test_an_empty_transcript_is_refused(session, tmp_path, text):
    s = session()
    s.send(cmd="set_reference", wav=str(make_wav(tmp_path / "clip.wav", 5.0)), text=text)
    assert "transcript" in s.error("bad_request")["message"]
    assert s.engine.references == []


def test_a_reference_that_is_not_a_wav_file_is_refused(session, tmp_path):
    s = session()
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"this is not audio at all")
    s.send(cmd="set_reference", wav=str(clip), text=TRANSCRIPT)
    assert "clip.wav" in s.error("bad_request")["message"]
    assert s.engine.references == []


@pytest.mark.parametrize("command", [{"text": TRANSCRIPT}, {"wav": 7, "text": TRANSCRIPT}, {"wav": "a.wav"},
                                     {"wav": "a.wav", "text": None}])
def test_set_reference_without_a_path_or_transcript_is_refused(session, command):
    s = session()
    s.send(cmd="set_reference", **command)
    s.error("bad_request")
    assert s.engine.references == []


def test_the_length_the_engine_reports_is_checked_too(session, tmp_path):
    engine = FakeEngine()
    engine.reference_seconds = 14.9  # what the engine decoded disagrees with the file header
    s = session(engine)
    s.send(cmd="set_reference", wav=str(make_wav(tmp_path / "clip.wav", 5.0)), text=TRANSCRIPT)
    assert "14.9" in s.error("bad_request")["message"]
    assert names(s.run_job(tmp_path)) == ["error"]  # and no reference is in force


def test_a_refused_reference_leaves_no_reference_in_force(session, tmp_path):
    """Otherwise the next request would silently speak in the previous voice."""
    s = session()
    s.reference(tmp_path)
    s.send(cmd="set_reference", wav=str(tmp_path / "gone.wav"), text=TRANSCRIPT)
    s.error("bad_request")
    events = s.run_job(tmp_path)
    assert names(events) == ["error"] and events[0]["code"] == "bad_request"
    assert s.engine.calls == []


# ---------------------------------------------------------------- 3. synthesize

def test_synthesize_before_any_reference_is_refused(session, tmp_path):
    s = session()
    events = s.run_job(tmp_path, id="early")
    assert names(events) == ["error"]
    assert events[0]["code"] == "bad_request" and events[0]["id"] == "early"
    assert "reference" in events[0]["message"]
    assert s.engine.calls == []


def test_a_job_emits_started_then_one_audio_per_piece_in_order_then_done(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    pieces = ["The first sentence.", "A second one, a little longer.", "Third."]
    s.engine.plan = [(tone(1.0), RATE), (tone(2.0), RATE), (tone(0.5), RATE)]
    events = s.run_job(tmp_path, id="j1", pieces=pieces, nfe=16, speed=1.0, seed=7)

    assert names(events) == ["started", "audio", "audio", "audio", "done"]
    assert events[0] == {"event": "started", "id": "j1", "pieces": 3}
    out = tmp_path / "out"
    for index, (event, text, seconds) in enumerate(zip(events[1:4], pieces, (1.0, 2.0, 0.5))):
        assert event["id"] == "j1" and event["index"] == index and event["text"] == text
        assert event["sample_rate"] == RATE and event["seconds"] == pytest.approx(seconds)
        assert event["samples"] == int(seconds * RATE)
        assert Path(event["path"]) == out / f"j1_{index:03d}.wav"
        assert event["elapsed"] >= 0
    done = events[4]
    assert done["id"] == "j1" and done["pieces"] == 3 and done["seconds"] == pytest.approx(3.5)
    assert s.engine.calls == [(text, 16, 1.0, 7) for text in pieces]
    assert sorted(p.name for p in out.iterdir()) == ["j1_000.wav", "j1_001.wav", "j1_002.wav"]


def test_each_file_exists_complete_when_its_event_is_emitted(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    events = s.run_job(tmp_path, pieces=["One sentence here.", "Another sentence here."])
    paths = [e["path"] for e in events if e["event"] == "audio"]
    assert len(paths) == 2
    assert s.stdout.files_present == {paths[0]: True, paths[1]: True}


def test_a_piece_is_a_wav_file_the_backend_reads_and_it_is_not_trimmed(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    audio = np.concatenate([np.zeros(RATE, np.float32), tone(1.0, peak=0.5), np.zeros(RATE // 2, np.float32)])
    s.engine.plan = [(audio, RATE)]
    event = s.run_job(tmp_path)[1]
    data, rate = read_wav(event["path"])
    assert rate == RATE and len(data) == len(audio) == event["samples"]
    assert np.allclose(data, audio, atol=1.0 / 32000)
    with wave.open(event["path"], "rb") as w:
        assert (w.getnchannels(), w.getsampwidth()) == (1, 2)


def test_a_piece_louder_than_full_scale_is_scaled_down_not_clipped(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    audio = tone(1.0, peak=1.37)
    s.engine.plan = [(audio, RATE)]
    event = s.run_job(tmp_path)[1]
    data, _ = read_wav(event["path"])
    assert float(np.abs(data).max()) == pytest.approx(1.0, abs=0.001)
    assert np.allclose(data, audio / 1.37, atol=1.0 / 16000)  # the shape survives: nothing was flattened


def test_a_quiet_piece_is_written_at_its_own_level(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    s.engine.plan = [(tone(1.0, peak=0.1), RATE)]
    data, _ = read_wav(s.run_job(tmp_path)[1]["path"])
    assert float(np.abs(data).max()) == pytest.approx(0.1, abs=0.001)


def test_without_a_seed_the_worker_picks_one_and_uses_it_for_every_piece(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    events = s.run_job(tmp_path, pieces=["One sentence here.", "Another sentence here."])
    seeds = {call[3] for call in s.engine.calls}
    assert len(seeds) == 1
    seed = seeds.pop()
    assert isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2 ** 31
    assert [e["seed"] for e in events if e["event"] == "audio"] == [seed, seed]


BAD_REQUESTS = {
    "no id": dict(id=None),
    "empty id": dict(id=""),
    "an id that is not a plain name": dict(id="../escape"),
    "an id with a path separator": dict(id="a\\b"),
    "no pieces": dict(pieces=[]),
    "pieces as text": dict(pieces="One sentence."),
    "an empty piece": dict(pieces=["Fine.", "   "]),
    "a piece that is not text": dict(pieces=["Fine.", 3]),
    "no nfe": dict(nfe=None),
    "an nfe that is not offered": dict(nfe=12),
    "nfe as text": dict(nfe="16"),
    "nfe as a boolean": dict(nfe=True),
    "speed zero": dict(speed=0),
    "speed far too high": dict(speed=5.0),
    "speed as text": dict(speed="fast"),
    "speed not a number": dict(speed=float("nan")),
    "a seed that is not a whole number": dict(seed=1.5),
    "no out_dir": dict(out_dir=None),
    "max_chars zero": dict(max_chars=0),
    "max_chars as text": dict(max_chars="178"),
}


@pytest.mark.parametrize("change", BAD_REQUESTS.values(), ids=BAD_REQUESTS.keys())
def test_a_malformed_synthesize_is_refused_and_nothing_starts(session, tmp_path, change):
    s = session()
    s.reference(tmp_path)
    command = dict(cmd="synthesize", id="job", pieces=["One sentence."], out_dir=str(tmp_path / "out"), nfe=16)
    command.update(change)
    s.send(**{k: v for k, v in command.items() if v is not None})
    s.error("bad_request")
    assert s.engine.calls == [] and not (tmp_path / "out").exists()
    assert names(s.run_job(tmp_path, id="next")) == ["started", "audio", "done"]  # and the worker is not left busy


def test_a_piece_longer_than_the_callers_limit_is_refused_whole(session, tmp_path):
    """The package would chunk it at a comma; the worker refuses instead, before anything is synthesised."""
    s = session()
    s.reference(tmp_path)
    pieces = ["Short enough.", "This one is forty bytes long, no more!!!", "x" * 41]
    assert [len(p) for p in pieces] == [13, 40, 41]
    events = s.run_job(tmp_path, id="long", pieces=pieces, max_chars=40)
    assert names(events) == ["error"] and events[0]["code"] == "bad_request" and events[0]["id"] == "long"
    assert "41" in events[0]["message"] and "40" in events[0]["message"] and "piece 2" in events[0]["message"]
    assert s.engine.calls == [] and not (tmp_path / "out").exists()
    assert names(s.run_job(tmp_path, id="fits", pieces=pieces[:2], max_chars=40)) == ["started", "audio", "audio",
                                                                                      "done"]


def test_the_limit_counts_utf8_bytes_not_characters(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    piece = "Zwölf Boxkämpfer jagen Viktor über den großen Deich."  # 52 characters, 56 bytes
    assert (len(piece), len(piece.encode("utf-8"))) == (52, 56)
    assert names(s.run_job(tmp_path, id="a", pieces=[piece], max_chars=55)) == ["error"]
    events = s.run_job(tmp_path, id="b", pieces=[piece], max_chars=56)
    assert names(events) == ["started", "audio", "done"] and events[1]["text"] == piece


def test_a_piece_that_would_push_one_model_call_past_30_seconds_is_refused(session, tmp_path, worker):
    """Reference plus generated audio over about 30 s turns to babble without any error (spike notes, section 7),
    so the worker applies the model's own duration rule itself, whatever limit the caller sent."""
    s = session()
    transcript = "Nineteen bytes long."
    assert len(worker.reference_text(transcript).encode("utf-8")) == 21  # 10 s / 21 bytes: 0.48 s per byte
    s.reference(tmp_path, seconds=10.0, text=transcript)
    fits = "a" * 40 + "."  # 10 + 41 * 10 / 21 = 29.5 s
    over = "a" * 42 + "."  # 10 + 43 * 10 / 21 = 30.5 s
    events = s.run_job(tmp_path, id="over", pieces=[fits, over])
    assert names(events) == ["error"] and events[0]["code"] == "bad_request"
    assert "piece 1" in events[0]["message"] and "30" in events[0]["message"]
    assert s.engine.calls == []
    assert names(s.run_job(tmp_path, id="fits", pieces=[fits])) == ["started", "audio", "done"]


def test_estimated_seconds_follows_the_packages_duration_rule(worker):
    # reference: 5.33 s, 51 bytes with the space the engine appends; a piece of 87 bytes gets 5.33 / 51 * 87 s
    reference = worker.reference_text(TRANSCRIPT)
    assert reference == TRANSCRIPT + " "
    assert worker.generated_length(5.33, reference, "x" * 87, 1.0) == pytest.approx(5.33 / 51 * 87)
    assert worker.generated_length(5.33, reference, "x" * 87, 2.0) == pytest.approx(5.33 / 51 * 87 / 2)
    # the package slows text under 10 bytes to speed 0.3, whatever speed was asked for
    assert worker.generated_length(5.33, reference, "Yes.", 1.0) == pytest.approx(5.33 / 51 * 4 / 0.3)
    assert worker.generated_length(5.33, reference, "x" * 10, 1.0) == pytest.approx(5.33 / 51 * 10)


def test_reference_text_ends_in_punctuation_and_a_space(worker):
    assert worker.reference_text("  Hello there  ") == "Hello there. "
    assert worker.reference_text("Is it?") == "Is it? "
    assert worker.reference_text("Done!") == "Done! "
    assert worker.reference_text("你好。") == "你好。"  # a multi-byte ending gets no space, as in the package


# ---------------------------------------------------------------- 4. pieces that come back unusable

UNUSABLE = {
    "empty": (np.zeros(0, np.float32), RATE),
    "all NaN": (np.full(RATE, np.nan, np.float32), RATE),
    "some NaN": (np.concatenate([tone(0.5), np.full(10, np.nan, np.float32)]), RATE),
    "infinite": (np.full(RATE, np.inf, np.float32), RATE),
    "shorter than a tenth of a second": (tone(0.09), RATE),
    "near-silent": (tone(1.0, peak=0.001), RATE),
    "nothing at all": None,
}


@pytest.mark.parametrize("bad", UNUSABLE.values(), ids=UNUSABLE.keys())
def test_an_unusable_piece_is_retried_once_with_another_seed(session, tmp_path, bad):
    s = session()
    s.reference(tmp_path)
    engine = s.engine
    if bad is None:
        engine.plan = [lambda should_stop: None]
    else:
        engine.plan = [bad]
    events = s.run_job(tmp_path, pieces=["First sentence.", "Second sentence."], seed=11)
    assert names(events) == ["started", "audio", "audio", "done"]
    assert [call[0] for call in engine.calls] == ["First sentence.", "First sentence.", "Second sentence."]
    first, retry, second = (call[3] for call in engine.calls)
    assert first == 11 and retry != 11 and second == 11
    assert events[1]["seed"] == retry and events[2]["seed"] == 11
    data, _ = read_wav(events[1]["path"])
    assert len(data) == RATE and float(np.abs(data).max()) > 0.2  # the retry's audio, not the bad attempt's


def test_a_piece_that_fails_twice_ends_the_job_and_earlier_pieces_stand(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    s.engine.plan = [None, (np.zeros(0, np.float32), RATE), (np.full(RATE, np.nan, np.float32), RATE)]
    events = s.run_job(tmp_path, id="bad", pieces=["Good sentence.", "Bad sentence.", "Never reached."])
    assert names(events) == ["started", "audio", "error"]
    error = events[2]
    assert error["code"] == "piece_failed" and error["id"] == "bad" and error["index"] == 1
    assert "piece 1" in error["message"]
    assert len(s.engine.calls) == 3  # one good piece, two attempts, and the third piece was never tried
    assert os.path.isfile(events[1]["path"])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["bad_000.wav"]
    assert names(s.run_job(tmp_path, id="after")) == ["started", "audio", "done"]


# ---------------------------------------------------------------- 5. cancel

def test_cancel_inside_a_piece_stops_it_through_should_stop(session, tmp_path, worker):
    s = session()
    s.reference(tmp_path)
    gate = Gate(worker)
    s.engine.plan = [None, gate]
    s.synthesize(tmp_path, id="c1", pieces=["First sentence.", "Second sentence.", "Third sentence."])
    assert names([s.next(), s.next()]) == ["started", "audio"]
    assert gate.entered.wait(WAIT)
    s.send(cmd="cancel", id="c1")
    s.sync()
    gate.release.set()
    cancelled = s.next("cancelled")
    assert cancelled["id"] == "c1" and cancelled["index"] == 1 and cancelled["elapsed"] >= 0
    assert gate.saw_stop is True
    assert len(s.engine.calls) == 2  # the third piece was never started
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["c1_000.wav"]
    # the worker takes the next request, with the same reference
    assert names(s.run_job(tmp_path, id="c2")) == ["started", "audio", "done"]
    assert s.end() == 0
    assert "done" not in [e["event"] for e in s.stdout.events if e.get("id") == "c1"]


def test_cancel_stops_before_the_next_piece_even_if_the_engine_finishes_the_current_one(session, tmp_path, worker):
    s = session()
    s.reference(tmp_path)
    gate = Gate(worker, obey_stop=False)  # an engine that never looks at should_stop
    s.engine.plan = [gate]
    s.synthesize(tmp_path, id="c1", pieces=["First sentence.", "Second sentence."])
    s.next("started")
    assert gate.entered.wait(WAIT)
    s.send(cmd="cancel", id="c1")
    s.sync()
    gate.release.set()
    assert s.next("cancelled")["index"] == 0
    assert len(s.engine.calls) == 1
    assert list((tmp_path / "out").iterdir()) == []  # the piece that was finished but never announced is removed
    assert names(s.run_job(tmp_path, id="c2")) == ["started", "audio", "done"]


def test_a_job_cancelled_during_its_last_piece_never_emits_done(session, tmp_path, worker):
    s = session()
    s.reference(tmp_path)
    gate = Gate(worker, obey_stop=False)
    s.engine.plan = [gate]
    s.synthesize(tmp_path, id="last", pieces=["The only sentence."])
    s.next("started")
    assert gate.entered.wait(WAIT)
    s.send(cmd="cancel", id="last")
    s.sync()
    gate.release.set()
    s.next("cancelled")
    assert s.end() == 0
    assert names(s.stdout.events).count("done") == 0


def test_an_engine_error_while_cancelling_is_reported_as_cancelled(session, tmp_path, worker):
    """Some library wraps the exception raised in the sampling hook; the job still ends as cancelled."""
    s = session()
    s.reference(tmp_path)
    gate = Gate(worker, obey_stop=False)

    def piece(should_stop):
        gate(should_stop)
        raise RuntimeError("wrapped")

    s.engine.plan = [piece]
    s.synthesize(tmp_path, id="c1")
    s.next("started")
    assert gate.entered.wait(WAIT)
    s.send(cmd="cancel", id="c1")
    s.sync()
    gate.release.set()
    s.next("cancelled")


def test_a_cancel_that_lands_while_a_piece_is_being_written_keeps_its_event_back(session, tmp_path, worker,
                                                                                 monkeypatch):
    """The narrowest gap: the engine has returned, the cancel arrives, and only then would `audio` be written."""
    writing, go_on = threading.Event(), threading.Event()
    real_write = worker.write_piece

    def slow_write(path, audio, rate):
        real_write(path, audio, rate)
        writing.set()
        assert go_on.wait(WAIT)

    monkeypatch.setattr(worker, "write_piece", slow_write)
    s = session()
    s.reference(tmp_path)
    s.synthesize(tmp_path, id="gap", pieces=["First sentence.", "Second sentence."])
    s.next("started")
    assert writing.wait(WAIT)
    s.send(cmd="cancel", id="gap")
    s.sync()
    go_on.set()
    assert s.next("cancelled")["index"] == 0
    assert len(s.engine.calls) == 1
    assert list((tmp_path / "out").iterdir()) == []
    assert s.end() == 0
    assert "audio" not in names(s.stdout.events) and "done" not in names(s.stdout.events)


def test_cancel_for_an_unknown_id_is_ignored(session, tmp_path, worker):
    s = session()
    s.reference(tmp_path)
    s.send(cmd="cancel", id="nobody")
    s.send(cmd="cancel")
    s.sync()
    gate = Gate(worker)
    s.engine.plan = [gate]
    s.synthesize(tmp_path, id="real", pieces=["First sentence.", "Second sentence."])
    s.next("started")
    assert gate.entered.wait(WAIT)
    s.send(cmd="cancel", id="someone-else")
    s.sync()
    gate.release.set()
    assert names([s.next(), s.next(), s.next()]) == ["audio", "audio", "done"]
    assert gate.saw_stop is False


def test_a_late_cancel_for_a_finished_job_does_not_touch_the_next_job(session, tmp_path, worker):
    s = session()
    s.reference(tmp_path)
    assert names(s.run_job(tmp_path, id="old")) == ["started", "audio", "done"]
    s.send(cmd="cancel", id="old")  # arrives after the job ended, before the next one
    gate = Gate(worker)
    s.engine.plan = [gate]
    s.synthesize(tmp_path, id="new", pieces=["First sentence.", "Second sentence."])
    s.next("started")
    assert gate.entered.wait(WAIT)
    s.send(cmd="cancel", id="old")  # and again while the next one is inside a piece
    s.sync()
    gate.release.set()
    assert names([s.next(), s.next(), s.next()]) == ["audio", "audio", "done"]
    assert gate.saw_stop is False
    assert s.end() == 0
    assert "cancelled" not in names(s.stdout.events)


# ---------------------------------------------------------------- 6. bad input lines

def test_malformed_json_is_reported_and_the_worker_keeps_running(session, tmp_path):
    s = session()
    s.stdin.write("{not json\n")
    assert s.error("bad_json")["id"] is None
    s.stdin.write("\n")  # an empty line is nothing at all
    s.stdin.write('["cmd", "shutdown"]\n')  # JSON, but not a command
    s.error("bad_json")
    s.reference(tmp_path)
    assert names(s.run_job(tmp_path)) == ["started", "audio", "done"]


def test_an_unknown_command_is_reported_and_the_worker_keeps_running(session, tmp_path):
    s = session()
    s.send(cmd="bogus", id="x")
    error = s.error("unknown_command")
    assert "bogus" in error["message"] and error["id"] == "x"
    s.send(id="y")
    s.error("unknown_command")
    s.reference(tmp_path)
    assert names(s.run_job(tmp_path)) == ["started", "audio", "done"]


# ---------------------------------------------------------------- 7. out of memory and other engine errors

@pytest.mark.parametrize("error", [OutOfMemoryError("HIP out of memory. Tried to allocate 2.00 GiB"), MemoryError(),
                                   RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")],
                         ids=["torch's error class by name", "MemoryError", "an older message"])
def test_out_of_memory_ends_the_job_and_the_worker_takes_the_next(session, tmp_path, error):
    s = session()
    s.reference(tmp_path)
    s.engine.plan = [None, error]
    events = s.run_job(tmp_path, id="big", pieces=["First sentence.", "Second sentence.", "Third sentence."])
    assert names(events) == ["started", "audio", "error"]
    assert events[2]["code"] == "out_of_memory" and events[2]["id"] == "big"
    assert len(s.engine.calls) == 2  # no retry, and the third piece was not tried
    assert names(s.run_job(tmp_path, id="after")) == ["started", "audio", "done"]


def test_out_of_memory_while_setting_the_reference(session, tmp_path):
    s = session()
    s.engine.reference_error = OutOfMemoryError("HIP out of memory")
    s.send(cmd="set_reference", wav=str(make_wav(tmp_path / "clip.wav", 5.0)), text=TRANSCRIPT)
    s.error("out_of_memory")
    s.engine.reference_error = None
    s.reference(tmp_path)
    assert names(s.run_job(tmp_path)) == ["started", "audio", "done"]


def test_any_other_engine_error_is_internal_and_the_worker_stays_up(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    s.engine.plan = [ValueError("tensor shapes do not match")]
    events = s.run_job(tmp_path, id="odd")
    assert names(events) == ["started", "error"]
    assert events[1]["code"] == "internal" and "tensor shapes do not match" in events[1]["message"]
    assert names(s.run_job(tmp_path, id="after")) == ["started", "audio", "done"]


def test_the_engine_can_refuse_a_request_itself(session, tmp_path, worker):
    s = session()
    s.engine.reference_error = worker.BadRequest("The reference clip is silent")
    s.send(cmd="set_reference", wav=str(make_wav(tmp_path / "clip.wav", 5.0)), text=TRANSCRIPT)
    assert s.error("bad_request")["message"] == "The reference clip is silent"


# ---------------------------------------------------------------- 8. stdout carries JSON lines only

def test_every_line_on_stdout_is_one_flushed_json_object(session, tmp_path, worker):
    s = session()
    s.reference(tmp_path, text="Ünïcödé transcript, with “quotes”.")
    s.stdin.write("garbage\n")
    s.next("error")
    s.run_job(tmp_path, pieces=["Ein Stück Text — mit Umlauten: äöü.", "And a second, plain one."])
    assert s.end() == 0
    out = s.stdout
    assert out._unflushed == ""  # nothing was left unflushed (a partial line would have failed in flush())
    lines = out.text.split("\n")
    assert lines[-1] == "" and len(lines) - 1 == len(out.events)
    for line in lines[:-1]:
        event = json.loads(line)
        assert isinstance(event, dict) and isinstance(event["event"], str)
        assert line.isascii()  # readable whatever encoding the pipe has
    assert [e["text"] for e in out.events if e["event"] == "audio"][0] == "Ein Stück Text — mit Umlauten: äöü."


def test_taking_over_stdout_points_print_at_stderr_and_keeps_the_original_for_the_protocol(worker, tmp_path,
                                                                                         monkeypatch):
    """The Python-level half, with the descriptor calls recorded instead of made (pytest owns descriptor 1 here).
    The descriptor-level half runs for real in the subprocess test below."""
    import io
    original = os.open(tmp_path / "original-stdout", os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0))
    redirected = []
    monkeypatch.setattr(worker.os, "dup", lambda fd: {1: original}[fd])
    monkeypatch.setattr(worker.os, "dup2", lambda source, target: redirected.append((source, target)))
    fake_out, fake_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)

    protocol = worker.take_over_stdout()
    try:
        assert redirected == [(2, 1)]  # descriptor 1 now leads to stderr
        assert sys.stdout is fake_err
        print("a library says hello")
        protocol.write('{"event": "ready"}\n')
        protocol.flush()
    finally:
        protocol.close()
    assert fake_err.getvalue() == "a library says hello\n" and fake_out.getvalue() == ""
    assert (tmp_path / "original-stdout").read_bytes() == b'{"event": "ready"}\n'


NOISY_DRIVER = r'''
import ctypes, importlib.util, os, sys
import numpy as np

spec = importlib.util.spec_from_file_location("tts_worker_under_test", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def chatter(where):
    """Everything a library can do to stdout, from Python down to the operating system."""
    print("PRINT from", where)
    sys.__stdout__.write("DUNDER from %s\n" % where)
    sys.__stdout__.flush()
    os.write(1, ("FD1 from %s\n" % where).encode())
    if os.name == "nt":
        crt = ctypes.CDLL("ucrtbase")
        crt.puts(("CRT from %s" % where).encode())
        crt.fflush(None)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong,
                                       ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
        text, written = ("WIN32 from %s\n" % where).encode(), ctypes.c_ulong()
        kernel32.WriteFile(kernel32.GetStdHandle(-11), text, len(text), ctypes.byref(written), None)


class Noisy:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def load(self):
        chatter("load")
        return {"device": "noisy", "model_dir": str(self.model_dir), "offline": os.environ.get("HF_HUB_OFFLINE")}

    def set_reference(self, wav_path, text):
        chatter("reference")
        return 5.0

    def synthesize_piece(self, text, nfe, speed, seed, should_stop):
        chatter("piece")
        t = np.arange(24000, dtype=np.float32) / 24000
        return (np.sin(2 * np.pi * 220.0 * t) * 0.3).astype(np.float32), 24000


worker.run(sys.argv[2:], engine_factory=Noisy)  # the entry point's own way of ending the process
'''


def test_library_output_at_every_level_goes_to_stderr_in_a_real_process(tmp_path):
    """main() keeps the real stdout for the protocol and points sys.stdout and the process's descriptor 1 at
    stderr, so print(), C stdio and direct handle writes from a library all land in the log."""
    driver = tmp_path / "driver.py"
    driver.write_text(NOISY_DRIVER, encoding="utf-8")
    wav = make_wav(tmp_path / "reference.wav", 5.0)
    model_dir = tmp_path / "models with a space"
    env = dict(os.environ, HF_HUB_OFFLINE="0", PYTHONDONTWRITEBYTECODE="1")
    with open(tmp_path / "stderr.log", "wb") as log:
        process = subprocess.Popen([sys.executable, "-u", str(driver), str(MODULE), "--model-dir", str(model_dir),
                                    "--parent-pid", str(os.getpid())],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env)
        guard = threading.Timer(120.0, process.kill)  # a hung worker fails the test instead of hanging it
        guard.start()
        try:
            def send(**command):
                process.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
                process.stdin.flush()

            raw = []

            def until(event):
                while True:
                    line = process.stdout.readline()
                    assert line, "the worker ended early: " + (tmp_path / "stderr.log").read_text(errors="replace")
                    raw.append(line)
                    if json.loads(line)["event"] == event:
                        return json.loads(line)

            ready = until("ready")
            send(cmd="set_reference", wav=str(wav), text=TRANSCRIPT)
            until("reference")
            send(cmd="synthesize", id="p1", pieces=["Größe matters."], out_dir=str(tmp_path / "out"), nfe=8)
            done = until("done")
            send(cmd="shutdown")
            raw += process.stdout.readlines()
            code = process.wait(timeout=60)
        finally:
            guard.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=60)
            process.stdin.close()
            process.stdout.close()

    events = [json.loads(line) for line in raw]
    assert names(events) == ["ready", "reference", "started", "audio", "done"]
    assert all(line.endswith(b"\n") and not line.endswith(b"\r\n") for line in raw)
    assert code == 0 and done["pieces"] == 1
    assert events[3]["text"] == "Größe matters." and os.path.isfile(events[3]["path"])
    assert ready["model_dir"] == str(model_dir)
    assert ready["offline"] == "1"  # the worker forces the offline switch before the engine loads
    logged = (tmp_path / "stderr.log").read_text(errors="replace")
    kinds = ["PRINT", "DUNDER", "FD1"] + (["CRT", "WIN32"] if os.name == "nt" else [])
    for where in ("load", "reference", "piece"):
        for kind in kinds:
            assert f"{kind} from {where}" in logged, (kind, where, logged)


def test_the_real_entry_point_reports_missing_model_files_and_exits_non_zero(tmp_path):
    """The real engine checks its files before it imports anything heavy, so this runs under any Python."""
    result = subprocess.run([sys.executable, str(MODULE), "--model-dir", str(tmp_path / "no models here")],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=120,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    lines = result.stdout.splitlines()
    assert len(lines) == 1, (result.stdout, result.stderr)
    event = json.loads(lines[0])
    assert event["event"] == "error" and event["code"] == "internal"
    assert "model_1250000.safetensors" in event["message"] and "setup.sh" in event["message"]
    assert result.returncode == 1


# ---------------------------------------------------------------- 9. ending

def test_shutdown_ends_the_worker_with_exit_code_0(session, tmp_path):
    s = session()
    s.reference(tmp_path)
    s.send(cmd="shutdown")
    s.synthesize(tmp_path, id="too-late")  # nothing after shutdown is acted on
    assert s.end(close=False) == 0
    assert names(s.stdout.events) == ["ready", "reference"]
    assert s.engine.calls == []


def test_end_of_input_ends_the_worker_with_exit_code_0(session):
    s = session()
    assert s.end() == 0
    assert names(s.stdout.events) == ["ready"]


def test_the_parent_disappearing_ends_the_worker_with_exit_code_0(session):
    alive = threading.Event()
    alive.set()
    asked = threading.Event()

    def parent_alive():
        asked.set()
        return alive.is_set()

    s = session(parent_alive=parent_alive, poll_seconds=0.001)
    assert asked.wait(WAIT)
    alive.clear()
    assert s.end(close=False) == 0  # the input was never closed: only the watchdog can have ended it
    assert names(s.stdout.events) == ["ready"]


@pytest.mark.parametrize("how", ["shutdown", "end of input", "parent gone"])
def test_ending_during_a_job_stops_it_and_finishes_nothing_further(session, tmp_path, worker, how):
    alive = threading.Event()
    alive.set()
    gate = Gate(worker)

    def stop_wait(thread):
        # serve() calls this once it has asked the running job to stop: only now is the piece let go
        gate.release.set()
        thread.join(WAIT)

    s = session(parent_alive=alive.is_set, poll_seconds=0.001, stop_wait=stop_wait)
    s.reference(tmp_path)
    s.engine.plan = [None, gate]
    s.synthesize(tmp_path, id="cut", pieces=["First sentence.", "Second sentence.", "Third sentence."])
    assert names([s.next(), s.next()]) == ["started", "audio"]
    assert gate.entered.wait(WAIT)
    if how == "shutdown":
        s.send(cmd="shutdown")
    elif how == "end of input":
        s.stdin.close()
    else:
        alive.clear()
    assert s.end(close=False) == 0
    assert gate.saw_stop is True  # the piece was told to stop from inside
    assert len(s.engine.calls) == 2  # and the third piece was never started
    assert names(s.stdout.events) == ["ready", "reference", "started", "audio", "cancelled"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["cut_000.wav"]


def test_an_engine_stuck_inside_a_piece_does_not_keep_the_worker_from_ending(session, tmp_path, worker):
    s = session(stop_wait=lambda thread: None)  # the grace period runs out at once
    s.reference(tmp_path)
    gate = Gate(worker)
    s.engine.plan = [gate]
    s.synthesize(tmp_path, id="stuck")
    s.next("started")
    assert gate.entered.wait(WAIT)
    s.send(cmd="shutdown")
    assert s.end(close=False) == 0  # serve() returned although the piece never came back
    gate.release.set()


def test_a_broken_stdout_ends_the_worker_quietly(session):
    """The backend is gone and so is the pipe: there is nobody to tell, and nothing to keep running for."""

    class Broken(Stdout):
        def write(self, text):
            if self.events:
                raise BrokenPipeError("the pipe is gone")
            return super().write(text)

    s = session(stdout=Broken())
    s.send(cmd="bogus")  # the reply cannot be written
    assert s.end(close=False) == 0  # the input was never closed
    assert names(s.stdout.events) == ["ready"]


def test_a_real_worker_ends_when_its_parent_process_is_gone(tmp_path):
    """--parent-pid in a real process: stdin stays open the whole time, so only the watchdog can end it."""
    driver = tmp_path / "driver.py"
    driver.write_text(NOISY_DRIVER, encoding="utf-8")
    parent = subprocess.Popen([sys.executable, "-c", "pass"])
    parent.wait(timeout=60)
    with open(tmp_path / "stderr.log", "wb") as log:
        process = subprocess.Popen([sys.executable, "-u", str(driver), str(MODULE), "--model-dir", str(tmp_path),
                                    "--parent-pid", str(parent.pid)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        try:
            code = process.wait(timeout=120)
            lines = process.stdout.read().splitlines()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=60)
            process.stdin.close()
            process.stdout.close()
    assert code == 0
    assert [json.loads(line)["event"] for line in lines] == ["ready"]
    assert "the app is gone" in (tmp_path / "stderr.log").read_text(errors="replace")


# ---------------------------------------------------------------- 10. one job at a time

def test_a_second_synthesize_while_one_runs_is_refused_and_the_first_goes_on(session, tmp_path, worker):
    s = session()
    s.reference(tmp_path)
    gate = Gate(worker)
    s.engine.plan = [gate]
    s.synthesize(tmp_path, id="first", pieces=["First sentence.", "Second sentence."])
    s.next("started")
    assert gate.entered.wait(WAIT)
    s.synthesize(tmp_path, id="second")
    error = s.error("bad_request")
    assert error["id"] == "second" and "first" in error["message"]
    s.send(cmd="set_reference", wav=str(make_wav(tmp_path / "other.wav", 6.0)), text="Another voice.")
    assert "first" in s.error("bad_request")["message"]  # nor may the voice change under a running job
    gate.release.set()
    events = [s.next(), s.next(), s.next()]
    assert names(events) == ["audio", "audio", "done"] and {e["id"] for e in events} == {"first"}
    assert [call[0] for call in s.engine.calls] == ["First sentence.", "Second sentence."]
    assert len(s.engine.references) == 1
    assert names(s.run_job(tmp_path, id="third")) == ["started", "audio", "done"]


def test_a_synthesize_sent_straight_after_set_reference_waits_for_it(session, tmp_path):
    s = session()
    wav = make_wav(tmp_path / "reference.wav", 5.0)
    s.send(cmd="set_reference", wav=str(wav), text=TRANSCRIPT)
    s.synthesize(tmp_path, id="eager")
    assert names([s.next(), s.next(), s.next(), s.next()]) == ["reference", "started", "audio", "done"]


# ---------------------------------------------------------------- environment, watchdog, shims

def test_the_environment_is_made_offline_and_kept_inside_the_tts_folder(worker, tmp_path):
    env = {"HF_HUB_OFFLINE": "0", "MIOPEN_LOG_LEVEL": "5", "HF_HOME": "D:/elsewhere/hf"}
    worker.apply_environment(env, tmp_path / ".cache")
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"  # forced, whatever was set
    assert env["MIOPEN_LOG_LEVEL"] == "5" and env["HF_HOME"] == "D:/elsewhere/hf"  # the launcher's choices stand
    assert env["WANDB_MODE"] == "disabled" and env["ROCM_SDK_TARGET_FAMILY"] == "custom"
    cache = tmp_path / ".cache"
    assert env["MIOPEN_USER_DB_PATH"] == str(cache / "miopen" / "db")
    assert env["MIOPEN_CUSTOM_CACHE_DIR"] == str(cache / "miopen")
    for name in ("TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
        assert Path(env[name]).parent == cache and Path(env[name]).is_dir()
    assert (cache / "miopen" / "db").is_dir()
    assert not (cache / "hf").exists()  # a folder the launcher placed elsewhere is not created here


def test_the_default_cache_is_the_one_setup_creates(worker):
    assert worker.CACHE_DIR == TTS_DIR / ".cache"


@windows_only
def test_process_alive_sees_a_running_and_a_finished_process(worker):
    assert worker.process_alive(os.getpid()) is True
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(timeout=60)
    assert worker.process_alive(done.pid) is False


def test_parse_args(worker):
    args = worker.parse_args(["--model-dir", "D:/Projects/AI Voice/tts/models", "--parent-pid", "4242"])
    assert args.model_dir == "D:/Projects/AI Voice/tts/models" and args.parent_pid == 4242
    assert worker.parse_args(["--model-dir", "m"]).parent_pid is None


@pytest.fixture
def fake_torch(monkeypatch):
    """A stand-in for AMD's Windows build: torch.distributed exists but has no ReduceOp."""
    torch = types.ModuleType("torch")
    torch.distributed = types.ModuleType("torch.distributed")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch.distributed)
    for name in ("wandb", "accelerate", "accelerate.utils", "ema_pytorch", "datasets", "transformers"):
        monkeypatch.delitem(sys.modules, name, raising=False)  # and put back as they were afterwards
    return torch


def test_shims_define_reduce_op_when_torch_has_none(fake_torch):
    shims = load(SHIMS, "aivoice_tts_shims")
    shims.apply(stub_training_imports=False)
    assert fake_torch.distributed.ReduceOp.SUM is not None  # encodec's default argument resolves
    assert {m.name for m in fake_torch.distributed.ReduceOp} >= {"SUM", "PRODUCT", "MIN", "MAX"}
    for name in shims.TRAINING_ONLY_MODULES:
        assert name not in sys.modules


def test_shims_leave_a_real_reduce_op_alone(fake_torch):
    real = object()
    fake_torch.distributed.ReduceOp = real
    shims = load(SHIMS, "aivoice_tts_shims")
    shims.apply()
    assert fake_torch.distributed.ReduceOp is real


def test_shims_stub_the_training_only_imports(fake_torch):
    shims = load(SHIMS, "aivoice_tts_shims")
    shims.apply()
    assert set(shims.TRAINING_ONLY_MODULES) == {"wandb", "accelerate", "accelerate.utils", "ema_pytorch", "datasets",
                                                "transformers"}
    import importlib
    for name in shims.TRAINING_ONLY_MODULES:
        assert importlib.import_module(name) is sys.modules[name]
    from accelerate.utils import DistributedDataParallelKwargs  # the kind of import f5_tts.model.trainer does
    from transformers import pipeline
    assert isinstance(DistributedDataParallelKwargs, type) and isinstance(pipeline, type)
    with pytest.raises(AttributeError):
        sys.modules["wandb"].__path__  # dunder lookups must fail, or import machinery mistakes it for a package
    stubs = {name: sys.modules[name] for name in shims.TRAINING_ONLY_MODULES}
    shims.apply()  # applying twice changes nothing
    assert {name: sys.modules[name] for name in shims.TRAINING_ONLY_MODULES} == stubs
