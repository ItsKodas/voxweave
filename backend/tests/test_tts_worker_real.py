"""A real run of tts/tts_worker.py: the real model on the GPU, started the way the backend's launcher starts it.
Opt-in: it needs the TTS environment and takes about a minute.

    AIVOICE_RUN_TTS_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_tts_worker_real.py -q -W error -s

Skipped unless that variable is set and `tts/verify.py --fast` passes. One worker serves every test in this file,
in order. Its caches go to pytest's temporary folder, except the GPU library's kernel cache, which is the real one
under tts/.cache/miopen so that its first-ever cost is paid once. Whatever happens, the worker's whole process
tree is ended when the file is done. Timings are printed (use -s) and written to AIVOICE_TTS_RESULTS if set.
"""
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import pytest
import soxr

from aivoice.audiofile import read_wav

ROOT = Path(__file__).resolve().parents[2]
TTS = ROOT / "tts"
PYTHON = TTS / ".venv" / "Scripts" / "python.exe"
EXAMPLES = TTS / ".venv" / "Lib" / "site-packages" / "f5_tts" / "infer" / "examples" / "basic"
SAMPLE = EXAMPLES / "basic_ref_en.wav"
WHISPER = ROOT / "data" / "pretrained" / "whisper"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
START_TIMEOUT = 300.0  # an empty kernel cache makes the first start slow
PIECE_TIMEOUT = 120.0
CHARS_PER_SECOND = 15.0

THREE = ["The lighthouse keeper climbed the narrow stairs every evening before the sun went down.",
         "He trimmed the wick, polished the great glass lens, and wrote the weather in a small blue book.",
         "On clear nights he could count the lights of seven fishing boats far out on the water."]
TEN = THREE + ["When the fog rolled in, he rang the bell by hand until his arms ached.",
               "Nobody had asked him to stay after the new electric lamp was installed.",
               "Still, he came back each night, because the sea had never learned to read a schedule.",
               "One winter a storm tore the roof from the boathouse and scattered the oars along the beach.",
               "He gathered them one by one in the morning and stacked them neatly by the door.",
               "The children from the village brought him bread, and he told them stories about whales.",
               "Years later, they still remembered the sound of that bell more clearly than his face."]


def _environment_ready():
    if os.environ.get("AIVOICE_RUN_TTS_TEST") != "1":
        return "set AIVOICE_RUN_TTS_TEST=1 to run the real text-to-speech worker test"
    try:
        done = subprocess.run([sys.executable, str(TTS / "verify.py"), "--fast"], capture_output=True, text=True,
                              timeout=120, creationflags=NO_WINDOW)
        report = json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return f"tts/verify.py --fast could not be run: {exc}"
    if not report.get("ok"):
        return "tts/verify.py --fast failed: " + "; ".join(c["message"] for c in report["checks"] if not c["ok"])
    return None


_SKIP = _environment_ready()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


def sample_transcript():
    """The English sample's transcript, from the package's own example configuration."""
    text = (EXAMPLES / "basic.toml").read_text(encoding="utf-8")
    assert "basic_ref_en.wav" in text
    return re.search(r'^ref_text\s*=\s*"(.+)"\s*$', text, re.M).group(1)


def worker_environment(scratch):
    """Section 4 of the spike notes, as the launcher will set it."""
    cache = scratch / "cache"
    for name in ("hf", "torch", "xdg", "mpl", "numba", "tmp"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    miopen = TTS / ".cache" / "miopen"
    (miopen / "db").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "ROCM_SDK_TARGET_FAMILY": "custom",
        "MIOPEN_USER_DB_PATH": str(miopen / "db"), "MIOPEN_CUSTOM_CACHE_DIR": str(miopen), "MIOPEN_LOG_LEVEL": "3",
        "TEMP": str(cache / "tmp"), "TMP": str(cache / "tmp"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(cache / "hf"), "TORCH_HOME": str(cache / "torch"), "XDG_CACHE_HOME": str(cache / "xdg"),
        "MPLCONFIGDIR": str(cache / "mpl"), "NUMBA_CACHE_DIR": str(cache / "numba"),
        "WANDB_MODE": "disabled", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore::SyntaxWarning",
    })
    env.pop("PYTHONDONTWRITEBYTECODE", None)  # imports are the slowest part of start-up
    return env


class RealWorker:
    def __init__(self, scratch):
        self.scratch = scratch
        self.log_path = scratch / "worker-stderr.log"
        self.out_dir = scratch / "out"
        self.raw = []  # every line the worker wrote to stdout, as bytes
        self.results = {}
        self.peak_rss = 0
        self._lines = queue.Queue()
        self._log = open(self.log_path, "wb")
        self.started = time.perf_counter()
        self.process = subprocess.Popen(
            [str(PYTHON), "-u", str(TTS / "tts_worker.py"), "--model-dir", str(TTS / "models"),
             "--parent-pid", str(os.getpid())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, env=worker_environment(scratch),
            cwd=str(ROOT), creationflags=NO_WINDOW)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.process.stdout:
                self.raw.append(line)
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def log_tail(self):
        self._log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-3000:]

    def send(self, **command):
        self.process.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def next(self, timeout=PIECE_TIMEOUT):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(f"the worker wrote nothing for {timeout:.0f} s. Its log ends:\n{self.log_tail()}")
        if line is None:
            pytest.fail(f"the worker ended (exit code {self.process.poll()}). Its log ends:\n{self.log_tail()}")
        return json.loads(line)

    def expect(self, event, timeout=PIECE_TIMEOUT):
        got = self.next(timeout)
        assert got["event"] == event, f"{got}\n{self.log_tail()}"
        return got

    def tree(self):
        try:
            top = psutil.Process(self.process.pid)
            return [top] + top.children(recursive=True)
        except psutil.NoSuchProcess:
            return []

    def measure_memory(self):
        total = 0
        for process in self.tree():
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
        self.peak_rss = max(self.peak_rss, total)

    def set_reference(self, wav, text):
        self.send(cmd="set_reference", wav=str(wav), text=text)
        return self.expect("reference")

    def job(self, id, pieces, nfe, **more):
        """Run one request to its end. Returns (events, seconds from sending it to its first audio event)."""
        sent = time.perf_counter()
        self.send(cmd="synthesize", id=id, pieces=pieces, out_dir=str(self.out_dir), nfe=nfe, **more)
        events, first = [], None
        while not events or events[-1]["event"] not in ("done", "cancelled", "error"):
            events.append(self.next())
            if first is None and events[-1]["event"] == "audio":
                first = time.perf_counter() - sent
        self.measure_memory()
        return events, first

    def kill(self):
        for process in self.tree():
            try:
                process.kill()
            except psutil.Error:
                pass
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        self.kill()
        for stream in (self.process.stdin, self.process.stdout, self._log):
            try:
                stream.close()
            except OSError:
                pass


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    return tmp_path_factory.mktemp("tts-real")


@pytest.fixture(scope="module")
def reference(scratch):
    assert SAMPLE.is_file(), f"the package's sample clip is missing: {SAMPLE}"
    wav = scratch / "reference.wav"
    shutil.copyfile(SAMPLE, wav)
    return wav, sample_transcript()


@pytest.fixture(scope="module")
def real(scratch):
    worker = RealWorker(scratch)
    try:
        yield worker
    finally:
        worker.close()
        if worker.results:
            print("\nreal TTS worker run: " + json.dumps(worker.results, indent=2))
            target = os.environ.get("AIVOICE_TTS_RESULTS")
            if target:
                Path(target).write_text(json.dumps(worker.results, indent=2), encoding="utf-8")


def names(events):
    return [e["event"] for e in events]


def check_piece(event, text, index):
    """The file behind an audio event: readable by the backend, finite, audible, of a believable length."""
    assert event["index"] == index and event["text"] == text
    data, rate = read_wav(event["path"])
    assert rate == event["sample_rate"] == 24000
    assert len(data) == event["samples"] and np.isfinite(data).all()
    seconds = len(data) / rate
    assert seconds == pytest.approx(event["seconds"], abs=0.01)
    expected = len(text) / CHARS_PER_SECOND
    assert 0.3 * expected <= seconds <= 3.0 * expected, (seconds, expected, text)
    peak = float(np.abs(data).max())
    assert 0.05 < peak <= 1.0, peak
    assert float(np.sqrt(np.mean(np.square(data)))) > 0.005
    return data, rate


def per_piece(events):
    times = [e["elapsed"] for e in events if e["event"] == "audio"]
    return [round(b - a, 3) for a, b in zip([0.0] + times, times)]


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


# ---------------------------------------------------------------- the session, in order

def test_the_worker_reports_ready_with_the_model_in_half_precision(real):
    ready = real.expect("ready", START_TIMEOUT)
    real.results["start_to_ready_s"] = round(time.perf_counter() - real.started, 2)
    for key in ("import_s", "load_s", "warmup_s", "device", "torch", "dtype", "vram_reserved_mb", "sample_rate"):
        real.results[key] = ready[key]
    assert ready["dtype"] == "float16" and ready["sample_rate"] == 24000
    assert "rocm" in ready["torch"]
    assert "warm-up failed" not in real.log_tail()
    real.measure_memory()


def test_set_reference_reports_the_clips_length(real, reference):
    wav, text = reference
    with wave.open(str(wav), "rb") as w:
        seconds = w.getnframes() / w.getframerate()
    assert 3.0 <= seconds <= 12.0 and text == "Some call me nature, others call me mother nature."
    assert real.set_reference(wav, text)["seconds"] == pytest.approx(seconds, abs=0.01)


def test_three_pieces_at_8_steps(real):
    events, first = real.job("eight", THREE, 8, seed=7)
    assert names(events) == ["started", "audio", "audio", "audio", "done"], f"{events}\n{real.log_tail()}"
    assert events[0]["pieces"] == 3
    for index, (event, text) in enumerate(zip(events[1:4], THREE)):
        check_piece(event, text, index)
        assert event["seed"] == 7
    assert events[4]["pieces"] == 3
    real.results["first_audio_8_steps_s"] = round(first, 3)
    real.results["per_piece_8_steps_s"] = per_piece(events)
    real.results["audio_seconds_8_steps"] = [e["seconds"] for e in events[1:4]]


def test_a_second_request_at_16_steps_on_the_same_worker(real):
    events, first = real.job("sixteen", THREE, 16, seed=7)
    assert names(events) == ["started", "audio", "audio", "audio", "done"], f"{events}\n{real.log_tail()}"
    for index, (event, text) in enumerate(zip(events[1:4], THREE)):
        check_piece(event, text, index)
    real.results["first_audio_16_steps_s"] = round(first, 3)
    real.results["per_piece_16_steps_s"] = per_piece(events)
    real.sixteen = events[1:4]


def test_a_synthesised_piece_says_what_it_was_asked_to_say(real):
    if not list(WHISPER.rglob("model.bin")):
        pytest.skip("the backend's Whisper model is not installed under data/pretrained/whisper")
    from aivoice import config
    from aivoice.transcribe import WhisperTranscriber

    def local_only(name, root):  # as the backend loads it (processor, 8-bit weights), but never downloading
        from faster_whisper import WhisperModel
        return WhisperModel(name, device="cpu", compute_type="int8", download_root=str(root), local_files_only=True)

    event = real.sixteen[0]
    data, rate = read_wav(event["path"])
    transcriber = WhisperTranscriber(lambda: config.whisper_model(config.data_dir()), WHISPER, local_only)
    heard = transcriber.transcribe(soxr.resample(data, rate, 16000))
    wanted, got = words(event["text"]), words(heard)
    matched = 0
    for word in wanted:
        if word in got:
            got.remove(word)
            matched += 1
    real.results["whisper_heard"] = heard
    real.results["whisper_word_match"] = round(matched / len(wanted), 3)
    assert matched / len(wanted) >= 0.7, (heard, event["text"])


def test_cancel_right_after_started_stops_the_request_and_the_worker_goes_on(real):
    sent = time.perf_counter()
    real.send(cmd="synthesize", id="cut", pieces=TEN, out_dir=str(real.out_dir), nfe=32, seed=7)
    real.expect("started")
    cancelled_at = time.perf_counter()
    real.send(cmd="cancel", id="cut")
    events = []
    while not events or events[-1]["event"] not in ("done", "cancelled", "error"):
        events.append(real.next())
    real.results["cancel_to_cancelled_s"] = round(time.perf_counter() - cancelled_at, 3)
    real.results["cancel_request_total_s"] = round(time.perf_counter() - sent, 3)
    assert events[-1]["event"] == "cancelled", f"{events}\n{real.log_tail()}"
    announced = [e for e in events if e["event"] == "audio"]
    assert len(announced) < 10 and events[-1]["index"] == len(announced)
    real.results["cancelled_after_pieces"] = len(announced)
    # the worker is still usable, with the same reference
    events, _ = real.job("after-cancel", THREE[:1], 8, seed=7)
    assert names(events) == ["started", "audio", "done"], f"{events}\n{real.log_tail()}"
    # the same text, seed and steps as the very first piece of this worker: what that first piece paid extra
    real.results["first_piece_repeated_8_steps_s"] = events[1]["elapsed"]
    check_piece(events[1], THREE[0], 0)


def test_cancel_in_the_middle_of_a_piece_takes_effect_within_a_sampling_step(real):
    real.send(cmd="synthesize", id="mid", pieces=TEN, out_dir=str(real.out_dir), nfe=32, seed=7)
    real.expect("started")
    first = real.expect("audio")  # the second piece is being sampled now: about 1.5 s of work at 32 steps
    cancelled_at = time.perf_counter()
    real.send(cmd="cancel", id="mid")
    events = [first]
    while events[-1]["event"] not in ("done", "cancelled", "error"):
        events.append(real.next())
    waited = time.perf_counter() - cancelled_at
    real.results["cancel_mid_piece_to_cancelled_s"] = round(waited, 3)
    real.results["first_audio_32_steps_s"] = first["elapsed"]
    assert events[-1]["event"] == "cancelled", f"{events}\n{real.log_tail()}"
    announced = [e for e in events if e["event"] == "audio"]
    assert 1 <= len(announced) < 10 and events[-1]["index"] == len(announced)
    assert waited < 0.5, waited  # far less than a piece: it did not wait for the piece to finish
    assert sorted(p.name for p in real.out_dir.glob("mid_*")) == [f"mid_{i:03d}.wav" for i in range(len(announced))]
    events, _ = real.job("after-mid-cancel", THREE[:1], 16, seed=7)
    assert names(events) == ["started", "audio", "done"], f"{events}\n{real.log_tail()}"
    check_piece(events[1], THREE[0], 0)


def test_references_outside_3_to_12_seconds_are_refused_and_clear_the_reference(real, reference, scratch):
    wav, text = reference
    with wave.open(str(wav), "rb") as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    size = params.sampwidth * params.nchannels

    def variant(name, seconds):
        wanted = int(seconds * params.framerate) * size
        data = (frames * (wanted // len(frames) + 1))[:wanted]
        with wave.open(str(scratch / name), "wb") as w:
            w.setparams(params)
            w.writeframes(data)
        return scratch / name

    for name, seconds, word in (("one-second.wav", 1.0, "too short"), ("twenty-seconds.wav", 20.0, "too long")):
        assert real.set_reference(wav, text)["seconds"] > 3.0
        real.send(cmd="set_reference", wav=str(variant(name, seconds)), text=text)
        error = real.expect("error")
        assert error["code"] == "bad_request" and word in error["message"], error
        events, _ = real.job("no-reference", THREE[:1], 8)  # the earlier reference is no longer in force
        assert names(events) == ["error"] and events[0]["code"] == "bad_request", events
    # and a good one puts the worker back to work
    real.set_reference(wav, text)
    events, _ = real.job("again", THREE[:1], 8, seed=7)
    assert names(events) == ["started", "audio", "done"], f"{events}\n{real.log_tail()}"


def test_a_reference_too_quiet_to_copy_is_refused(real, reference, scratch):
    wav, text = reference
    data, rate = read_wav(wav)
    rms = float(np.sqrt(np.mean(np.square(data))))
    quiet = scratch / "quiet.wav"
    with wave.open(str(quiet), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.round(data * (0.0005 / rms) * 32767.0).astype("<i2").tobytes())
    real.results["sample_rms"] = round(rms, 4)
    real.send(cmd="set_reference", wav=str(quiet), text=text)
    error = real.expect("error")
    assert error["code"] == "bad_request" and "too quiet" in error["message"], error


@pytest.mark.parametrize("ending", ["?", "!"])
def test_a_transcript_ending_in_a_question_or_exclamation_mark_works(real, reference, ending):
    wav, text = reference
    real.set_reference(wav, text.rstrip(".") + ending)
    events, _ = real.job("ending-" + {"?": "question", "!": "exclamation"}[ending], THREE[:1], 16, seed=7)
    assert names(events) == ["started", "audio", "done"], f"{events}\n{real.log_tail()}"
    check_piece(events[1], THREE[0], 0)
    real.results.setdefault("audio_seconds_by_transcript_ending", {})[ending] = events[1]["seconds"]


def test_shutdown_ends_the_process_with_exit_code_0_and_stdout_held_only_json(real):
    real.measure_memory()
    real.results["peak_rss_mb"] = round(real.peak_rss / 2 ** 20)
    tree = real.tree()
    assert tree
    real.send(cmd="shutdown")
    code = real.process.wait(timeout=60)
    _, alive = psutil.wait_procs(tree, timeout=30)
    assert code == 0, real.log_tail()
    assert alive == [], f"still running after shutdown: {alive}"
    assert real._lines.get(timeout=30) is None  # the pipe closed with nothing more on it
    assert len(real.raw) > 20
    for line in real.raw:
        assert line.endswith(b"\n")
        event = json.loads(line)
        assert isinstance(event, dict) and isinstance(event["event"], str), line
    leftovers = sorted(p.name for p in real.out_dir.iterdir() if p.suffix != ".wav")
    assert leftovers == [], leftovers
    log = real.log_tail()
    real.results["log"] = real.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    assert "Traceback" not in log, log
