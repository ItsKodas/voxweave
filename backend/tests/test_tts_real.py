"""A real run of text to speech from end to end: the real TtsManager, the real worker and model on the GPU, a
temporary data folder and a temporary voice. Opt-in: it needs the TTS environment and takes about two minutes.

    AIVOICE_RUN_TTS_TEST=1 backend/.venv/Scripts/python -m pytest backend/tests/test_tts_real.py -q -W error -s

Skipped unless that variable is set and `tts/verify.py --fast` passes. No audio device is opened: the player is a
stand-in that records what it is given and "plays" it instantly, and the device list is made up. The voice's one
clip is the sample that ships inside the f5_tts package, with its transcript. The data folder is pytest's; the
real data/ is only read (the feature model and the DirectML setting, for the Best preset). The worker is started
by the manager's own launcher, so its caches and temporary files are the app's, under tts/.cache.

One manager and one worker serve every test in this file, in order. Whatever happens, every worker process this
file started is ended when it is done. Timings are printed (use -s) and written to AIVOICE_TTS_MANAGER_RESULTS
if that is set.
"""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
import soxr

from aivoice import config, tts_audio, tts_text
from aivoice.audiofile import read_wav
from aivoice.devices import Device
from aivoice.library import Library
from aivoice.pretrained import VEC_FILE
from aivoice.tts import TtsManager
from tests.test_tts_worker_real import (CHARS_PER_SECOND, NO_WINDOW, ROOT, SAMPLE, TEN, THREE, TTS, WHISPER, _SKIP,
                                        sample_transcript, words)

pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

SPIKE_MODEL = ROOT / "spike" / "stage3" / "handoff" / "data" / "pretrained" / "rvc" / "final3"
DEVICES = [Device(1, "Speakers", 0, 2), Device(2, "CABLE Input", 0, 2), Device(3, "Microphone", 1, 0)]
REQUEST_TIMEOUT = 400.0  # a first start with an empty kernel cache takes half a minute by itself
RATE = 48000
PAUSE = RATE * 300 // 1000  # the silence save_wav puts between pieces


class RecordingPlayer:
    """Stands in for aivoice.playback.Player: opens nothing, keeps what it is given, and has "played" everything
    the moment finish() is called."""

    def __init__(self):
        self.sessions = []  # the devices of each session
        self.pieces = []  # (when, session, audio) as enqueued
        self.stops = []  # when stop() was called
        self._session = None
        self._ended = threading.Event()

    def start(self, devices, gains=None):
        self.sessions.append(dict(devices))
        self._session = object()
        self._ended.clear()
        return self._session

    def enqueue(self, audio, pause_ms=300, session=None):
        if session is not None and session is not self._session:
            return
        self.pieces.append((time.perf_counter(), session, np.array(audio, copy=True)))

    def finish(self):
        self._ended.set()

    def stop(self):
        self.stops.append(time.perf_counter())
        self._ended.set()

    def wait(self, timeout=None):
        return self._ended.wait(timeout)

    def status(self):
        return {"playing": not self._ended.is_set(), "queued_seconds": 0.0, "played_seconds": 0.0, "error": None}


class TimedConverter:
    """The real converter, with the length and the time of each pass noted."""

    def __init__(self, inner, rig):
        self._inner, self._rig = inner, rig

    def convert(self, piece):
        began = time.perf_counter()
        out = self._inner.convert(piece)
        self._rig.conversions.append((len(piece), len(out), time.perf_counter() - began))
        if len(self._rig.conversions) == 1:  # both models are on the graphics card now
            self._rig.results["gpu_dedicated_mb_during_best"] = gpu_dedicated_mb()
        return out


def gpu_dedicated_mb():
    """Dedicated graphics memory in use on the busiest adapter, all programs together, or None if Windows does
    not say. A measurement for the notes only: nothing is asserted on it."""
    try:
        done = subprocess.run(["typeperf", r"\GPU Adapter Memory(*)\Dedicated Usage", "-sc", "1"],
                              capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW)
        rows = [line for line in done.stdout.splitlines() if line.startswith('"') and "PDH-CSV" not in line]
        return round(max(float(cell.strip('"')) for cell in rows[0].split(",")[1:]) / 2 ** 20)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def worker_processes():
    """The worker processes this test process has started: its descendants that run tts_worker.py."""
    found = []
    for process in psutil.Process().children(recursive=True):
        try:
            if any("tts_worker.py" in part for part in process.cmdline()):
                found.append(process)
        except psutil.Error:
            pass
    return found


class Rig:
    def __init__(self, scratch):
        self.data = scratch / "data"
        self.library = Library(self.data)
        self.player = RecordingPlayer()
        self.conversions = []  # (samples in, samples out, seconds) of each RVC pass
        self.converter_load_s = []
        self.results = {}
        self.seen = []  # every worker process seen, so that none is lost at the end
        self.manager = TtsManager(self.data, self.library, TTS, self.player, lambda: list(DEVICES),
                                  rvc_factory=self._rvc)
        voice = self.library.create("Package sample")
        sample, rate = read_wav(SAMPLE)
        self.transcript = sample_transcript()
        self.library.add_clip(voice.id, soxr.resample(sample, rate, RATE, quality="HQ"), self.transcript, "recorded")
        self.voice = self.library.get(voice.id)

    def _rvc(self, voice_id):
        """The manager's own way of loading the voice's model (TtsManager._default_rvc), timed."""
        began = time.perf_counter()
        converter = TimedConverter(self.manager._default_rvc(voice_id), self)
        self.converter_load_s.append(time.perf_counter() - began)
        return converter

    def workers(self):
        now = worker_processes()
        self.seen.extend(p for p in now if p not in self.seen)
        return now

    def follow(self):
        """Polls the status as the screen does until the request has ended. Returns every status seen."""
        seen, deadline = [], time.perf_counter() + REQUEST_TIMEOUT
        while True:
            status = self.manager.status()
            seen.append(status)
            if status["state"] == "idle":
                return seen
            assert time.perf_counter() < deadline, f"the request did not end: {status}\n{self.log()}"
            self.manager.join(0.1)

    def speak(self, text, preset, **options):
        """One request to its end. Returns (final status, statuses seen, the pieces given to the player as
        (seconds after the request, audio), seconds the request took)."""
        options.setdefault("monitor_device", 1)
        before = len(self.player.pieces)
        began = time.perf_counter()
        first = self.manager.speak(self.voice.id, text, preset, **options)
        seen = [first] + self.follow()
        took = time.perf_counter() - began
        pieces = [(when - began, audio) for when, _, audio in self.player.pieces[before:]]
        return seen[-1], seen, pieces, took

    def log(self):
        try:
            return (self.data / "tts" / "worker.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            return "(no worker log)"

    def leftovers(self):
        tmp = self.data / "tts" / "tmp"
        return sorted(p.name for p in tmp.iterdir()) if tmp.is_dir() else []

    def close(self):
        self.workers()
        try:
            self.manager.shutdown()
        finally:
            for process in self.seen:
                try:
                    process.kill()
                except psutil.Error:
                    pass
            psutil.wait_procs(self.seen, timeout=30)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    rig = Rig(tmp_path_factory.mktemp("tts-manager-real"))
    try:
        yield rig
    finally:
        rig.close()
        if rig.results:
            print("\nreal text-to-speech run: " + json.dumps(rig.results, indent=2))
            target = os.environ.get("AIVOICE_TTS_MANAGER_RESULTS")
            if target:
                Path(target).write_text(json.dumps(rig.results, indent=2), encoding="utf-8")


def check_pieces(pieces, texts):
    """What the player was given: one piece per sentence, each mono float32 at 48 kHz, audible, finished the way
    the pipeline finishes a piece, and of a believable length for its sentence."""
    assert len(pieces) == len(texts)
    for (_, audio), text in zip(pieces, texts):
        assert audio.dtype == np.float32 and audio.ndim == 1 and np.isfinite(audio).all()
        seconds = len(audio) / RATE  # true only if the piece is at 48 kHz: at 24 kHz it would be half as long
        expected = len(text) / CHARS_PER_SECOND
        assert 0.5 * expected <= seconds <= 2.0 * expected, (seconds, expected, text)
        peak = float(np.abs(audio).max())
        assert 0.05 < peak <= 0.97 + 1e-6, peak
        assert float(np.sqrt(np.mean(np.square(audio)))) > 0.005
        assert not tts_audio.is_silent(audio, RATE)
        assert abs(float(audio[0])) < 0.01 and abs(float(audio[-1])) < 0.01  # faded in and out
        # It starts with the sentence: not with what was left of the reference clip's last word and then the
        # second of room tone the model puts before a sentence (the first real run played both).
        frame = RATE * 20 // 1000
        level = tts_audio._frame_db(audio, frame)
        loud = level >= level.max() - 30.0
        assert loud[:3].any() and all(loud[i:i + 20].any() for i in range(15)), text


def gaps(pieces):
    times = [when for when, _ in pieces]
    return [round(b - a, 3) for a, b in zip([0.0] + times, times)]


def seconds_of(pieces):
    return [round(len(audio) / RATE, 2) for _, audio in pieces]


# ---------------------------------------------------------------- the session, in order

def test_the_environment_check_run_by_the_manager_passes(rig):
    began = time.perf_counter()
    report = rig.manager.environment()
    rig.results["environment_check_s"] = round(time.perf_counter() - began, 2)
    assert report["ready"] and report["ok"], report
    assert rig.manager.status()["engine"] == "stopped" and rig.workers() == []
    rig.results["gpu_dedicated_mb_before"] = gpu_dedicated_mb()


def test_the_three_sentences_are_three_pieces_for_this_reference(rig):
    assert rig.voice.clips[0].seconds == pytest.approx(5.33, abs=0.01)
    limit = tts_audio.max_chars_for(rig.transcript, rig.voice.clips[0].seconds)
    rig.results["max_chars"] = limit
    assert tts_text.split(tts_text.clean(" ".join(THREE)), limit) == THREE
    assert len(tts_text.split(tts_text.clean(" ".join(TEN)), limit)) == 10


def test_fast_starts_the_worker_and_plays_three_pieces_in_order_and_saves_them(rig):
    status, seen, pieces, took = rig.speak(" ".join(THREE), "fast", save=True)
    assert status["error"] is None, f"{status}\n{rig.log()}"
    assert (status["piece"], status["pieces"], status["engine"]) == (3, 3, "ready")
    assert [s["engine"] for s in seen if s["state"] != "idle"][0] == "starting"
    assert {"preparing", "speaking"} <= {s["state"] for s in seen}
    check_pieces(pieces, THREE)
    assert rig.player.sessions[-1] == {"monitor": 1}
    rig.results["cold_first_piece_s"] = round(pieces[0][0], 2)  # includes starting the worker
    rig.results["cold_request_s"] = round(took, 2)
    rig.results["fast_audio_seconds"] = seconds_of(pieces)
    workers = rig.workers()
    assert workers, "no worker process is running after a request"
    rig.worker_pids = sorted(p.pid for p in workers)
    rig.results["worker_processes"] = len(workers)
    rig.results["gpu_dedicated_mb_worker_loaded"] = gpu_dedicated_mb()
    # the saved result: under the temporary data folder, the pieces as played with the pauses between them
    saved = Path(status["saved_path"])
    assert saved.parent == rig.data / "tts" and saved.name.endswith(f"-{rig.voice.id}.wav")
    audio, rate = read_wav(saved)
    assert rate == RATE and len(audio) == sum(len(a) for _, a in pieces) + 2 * PAUSE
    assert rig.leftovers() == []


def test_a_second_fast_request_reuses_the_worker_and_its_reference(rig):
    status, seen, pieces, took = rig.speak(" ".join(THREE), "fast", mic_device=2)
    assert status["error"] is None, f"{status}\n{rig.log()}"
    check_pieces(pieces, THREE)
    assert rig.player.sessions[-1] == {"mic": 2, "monitor": 1}
    assert all(s["engine"] == "ready" for s in seen)
    assert sorted(p.pid for p in rig.workers()) == rig.worker_pids
    assert status["saved_path"] is None and len(list((rig.data / "tts").glob("*.wav"))) == 1
    rig.results["fast_first_piece_s"] = round(pieces[0][0], 3)
    rig.results["fast_between_pieces_s"] = gaps(pieces)
    rig.results["fast_request_s"] = round(took, 2)
    assert pieces[0][0] < 5.0, pieces[0][0]  # a loaded worker answers in well under a second; 5 s is a hang


def test_balanced_plays_three_pieces_on_the_same_worker(rig):
    status, _, pieces, took = rig.speak(" ".join(THREE), "balanced")
    assert status["error"] is None, f"{status}\n{rig.log()}"
    check_pieces(pieces, THREE)
    assert sorted(p.pid for p in rig.workers()) == rig.worker_pids
    rig.results["balanced_first_piece_s"] = round(pieces[0][0], 3)
    rig.results["balanced_between_pieces_s"] = gaps(pieces)
    rig.results["balanced_request_s"] = round(took, 2)
    rig.results["balanced_audio_seconds"] = seconds_of(pieces)
    rig.balanced = pieces


def test_each_piece_says_its_own_sentence(rig):
    if not list(WHISPER.rglob("model.bin")):
        pytest.skip("the backend's Whisper model is not installed under data/pretrained/whisper")
    from aivoice.transcribe import WhisperTranscriber

    def local_only(name, root):  # as the backend loads it (processor, 8-bit weights), but never downloading
        from faster_whisper import WhisperModel
        return WhisperModel(name, device="cpu", compute_type="int8", download_root=str(root), local_files_only=True)

    transcriber = WhisperTranscriber(lambda: config.whisper_model(config.data_dir()), WHISPER, local_only)
    scores = []
    for (_, audio), text in zip(rig.balanced, THREE):
        heard = transcriber.transcribe(soxr.resample(audio, RATE, 16000))
        wanted, got = words(text), words(heard)
        matched = 0
        for word in wanted:
            if word in got:
                got.remove(word)
                matched += 1
        scores.append(round(matched / len(wanted), 3))
        rig.results.setdefault("whisper_heard", []).append(heard)
        assert matched / len(wanted) >= 0.7, (heard, text)
    rig.results["whisper_word_match"] = scores


def test_stop_in_the_middle_of_a_request_is_idle_within_two_seconds_and_the_next_request_works(rig):
    before = len(rig.player.pieces)
    rig.manager.speak(rig.voice.id, " ".join(TEN), "balanced", monitor_device=1, save=True)
    deadline = time.perf_counter() + REQUEST_TIMEOUT
    while len(rig.player.pieces) == before:  # the first piece is out: the worker is busy with the second
        assert rig.manager.is_active() and time.perf_counter() < deadline, f"{rig.manager.status()}\n{rig.log()}"
        rig.manager.join(0.01)
    began = time.perf_counter()
    status = rig.manager.stop()
    waited = time.perf_counter() - began
    rig.results["stop_to_idle_s"] = round(waited, 3)
    rig.results["player_silenced_after_s"] = round(rig.player.stops[-1] - began, 4)
    assert status["state"] == "idle" and status["error"] is None, status
    assert waited < 2.0, waited
    assert not rig.manager.is_active()
    assert status["saved_path"] is None and len(list((rig.data / "tts").glob("*.wav"))) == 1  # nothing was saved
    assert rig.leftovers() == []
    queued = len(rig.player.pieces) - before
    assert 1 <= queued < 10 and status["pieces"] == 10
    rig.results["stopped_after_pieces"] = queued
    assert status["engine"] == "ready" and sorted(p.pid for p in rig.workers()) == rig.worker_pids
    # the next request, on the same worker; nothing of the stopped one turns up in it
    status, _, pieces, _ = rig.speak(" ".join(THREE), "fast")
    assert status["error"] is None, f"{status}\n{rig.log()}"
    check_pieces(pieces, THREE)
    assert len(rig.player.pieces) == before + queued + 3
    assert sorted(p.pid for p in rig.workers()) == rig.worker_pids
    rig.results["first_piece_after_stop_s"] = round(pieces[0][0], 3)


def test_best_refuses_a_voice_without_a_trained_model(rig):
    with pytest.raises(ValueError, match="Best needs this voice's trained model"):
        rig.manager.speak(rig.voice.id, " ".join(THREE), "best", monitor_device=1)
    assert not rig.manager.is_active()


def test_best_converts_each_piece_with_the_voices_model_and_keeps_its_length(rig):
    real_data = config.data_dir()
    vec = real_data / "pretrained" / VEC_FILE
    if not (SPIKE_MODEL / "model.onnx").is_file() or not (SPIKE_MODEL / "voice.json").is_file():
        pytest.skip(f"the stage 3 spike's test model is gone: {SPIKE_MODEL}")
    if not vec.is_file():
        pytest.skip(f"the feature model is not installed: {vec}")
    # The model where a trained voice keeps it, the feature model where the backend looks for it, and the same
    # graphics card as the real settings name. Copies: the real files are only read.
    folder = rig.data / "voices" / rig.voice.id / "models" / "rvc"
    folder.mkdir(parents=True)
    for name in ("model.onnx", "model.index", "voice.json"):
        if (SPIKE_MODEL / name).is_file():
            shutil.copyfile(SPIKE_MODEL / name, folder / name)
    (rig.data / "pretrained").mkdir(exist_ok=True)
    shutil.copyfile(vec, rig.data / "pretrained" / VEC_FILE)
    (rig.data / "settings.json").write_text(json.dumps({"dml_device_id": config.dml_device_id(real_data)}))
    assert rig.library.get(rig.voice.id).rvc_sample_rate == 40000

    for run in ("best", "best_again"):
        del rig.conversions[:]
        status, _, pieces, took = rig.speak(" ".join(THREE), "best")
        assert status["error"] is None, f"{status}\n{rig.log()}"
        check_pieces(pieces, THREE)
        assert len(rig.conversions) == 3
        for (samples_in, samples_out, _), (_, audio) in zip(rig.conversions, pieces):
            assert samples_out == samples_in  # the conversion keeps the piece's length to the sample
            assert 0.8 * samples_in <= len(audio) <= samples_in  # and trimming its edges afterwards takes little
        rig.results[f"{run}_first_piece_s"] = round(pieces[0][0], 3)
        rig.results[f"{run}_between_pieces_s"] = gaps(pieces)
        rig.results[f"{run}_request_s"] = round(took, 2)
        rig.results[f"{run}_rvc_pass_s"] = [round(s, 3) for _, _, s in rig.conversions]
        rig.results[f"{run}_rvc_load_s"] = round(rig.converter_load_s[-1], 2)
        rig.results[f"{run}_audio_seconds"] = seconds_of(pieces)
    assert sorted(p.pid for p in rig.workers()) == rig.worker_pids


def test_unload_leaves_no_worker_process(rig):
    workers = rig.workers()
    assert sorted(p.pid for p in workers) == rig.worker_pids
    began = time.perf_counter()
    rig.manager.unload()
    rig.results["unload_s"] = round(time.perf_counter() - began, 2)
    _, alive = psutil.wait_procs(workers, timeout=10)
    assert alive == [] and rig.workers() == []
    assert rig.manager.status()["engine"] == "stopped"
    assert rig.manager.status_failures == 0
    assert rig.leftovers() == []
    rig.results["gpu_dedicated_mb_after_unload"] = gpu_dedicated_mb()
    rig.results["worker_log"] = rig.log()[-1500:]
