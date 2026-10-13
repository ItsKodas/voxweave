"""The text-to-speech worker: F5-TTS loaded once, one sentence-sized piece per model call, JSON lines both ways.

Start:   tts/.venv/Scripts/python.exe -u tts/tts_worker.py --model-dir tts/models [--parent-pid N]

stdin, one JSON object per line:
  {"cmd":"set_reference","wav":PATH,"text":STR}       a 3 to 12 second clip (16-bit WAV) and its transcript
  {"cmd":"synthesize","id":STR,"pieces":[STR,...],"out_dir":PATH,"nfe":8|16|32,
   "speed":1.0,"seed":INT|null,"max_chars":INT}       speed, seed and max_chars are optional
  {"cmd":"cancel","id":STR}
  {"cmd":"shutdown"}                                  closing stdin does the same

stdout, one JSON object per line and nothing else, each line flushed:
  {"event":"ready", ...what the engine reports about itself...}                 once, after the model loaded
  {"event":"reference","seconds":S}
  {"event":"started","id","pieces":N}
  {"event":"audio","id","index","path","samples","sample_rate","seconds","elapsed","seed","text"}   per piece
  {"event":"done","id","pieces","seconds","elapsed"}
  {"event":"cancelled","id","index","elapsed"}        index = how many pieces had been announced
  {"event":"error","id","code","message"}             code: bad_request, bad_json, unknown_command, out_of_memory,
                                                      internal, piece_failed (which also carries "index")

A piece is written to <out_dir>/<id>_<index>.wav as mono 16-bit PCM (what the backend's read_wav reads) through a
`.part` rename, untrimmed; a piece louder than full scale is scaled down to fit, never clipped. The file is
complete before its event is written. Only an error at start-up ends the worker; every other error leaves it
running for the next request.

The model does not raise when it is asked for something it cannot do: it returns babble, drops the last word, or
returns near-silence. So this layer refuses what is known to go wrong (a reference outside 3 to 12 seconds, a piece
longer than the caller's limit, a call that would run past 30 seconds of audio) and treats an empty, non-finite,
very short or near-silent result as a failed piece: one retry with another seed, then `piece_failed`.

Threads: this one reads commands off a queue and is the only one that decides anything about them; a reader thread
only moves lines from stdin to that queue; an engine thread runs set_reference and synthesis in the order they were
accepted. Which job is active, its cancel flag and every write to stdout are guarded by one lock, and whatever ends
a job is decided and written inside that lock, so a job whose cancel was accepted can never report `done`, and a
cancel is matched against the job that is active at that moment, by id.

The protocol layer below uses only the standard library and numpy, so the tests drive serve() with a fake engine
under the backend's Python. torch and f5_tts are imported only inside F5Engine.load().
"""
import argparse
import ctypes
import importlib.util
import io
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
import traceback
import wave
from pathlib import Path

import numpy as np

TTS_DIR = Path(__file__).resolve().parent
CACHE_DIR = TTS_DIR / ".cache"  # the folder tts/setup.sh creates

MIN_REFERENCE_SECONDS, MAX_REFERENCE_SECONDS = 3.0, 12.0
MAX_CALL_SECONDS = 30.0  # reference plus generated audio in one model call; beyond it the output turns to babble
STEP_COUNTS = (8, 16, 32)
MIN_SPEED, MAX_SPEED = 0.5, 2.0
SHORT_PIECE_BYTES, SHORT_PIECE_SPEED = 10, 0.3  # the package's own rule for very short text
MIN_PIECE_SECONDS = 0.1
SILENT_PEAK = 0.005  # the level below which the backend calls a piece silent
POLL_SECONDS = 0.5  # how often the parent is checked while no command arrives
STOP_GRACE_SECONDS = 5.0  # how long a running piece gets to notice that the worker is ending
REPLACE_ATTEMPTS = 8
JOB_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # it becomes part of a file name
SENTENCE_ENDS = ".!?。！？…"


class BadRequest(ValueError):
    """The request cannot be carried out as sent. The message is shown to the user."""


class Stopped(Exception):
    """Raised by an engine inside synthesize_piece when should_stop() became true."""


class PieceFailed(Exception):
    def __init__(self, index, problem):
        super().__init__(f"piece {index} produced no usable audio in two attempts: {problem}")
        self.index = index


def log(*parts):
    print("[tts-worker]", *parts, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- rules shared by the protocol and the engine

def reference_text(text):
    """The transcript as the model is given it: ending in sentence punctuation, then a space (the package's rule;
    the piece is appended straight after it)."""
    text = text.strip()
    if not text:
        raise BadRequest("The reference transcript is empty")
    if text[-1] not in SENTENCE_ENDS:
        text += "."
    if len(text[-1].encode("utf-8")) == 1:
        text += " "
    return text


def generated_length(reference_length, reference_text, piece, speed):
    """How much audio the model is asked to generate for a piece, in the unit of reference_length (seconds or mel
    frames). This is the package's duration rule: the length is fixed before sampling from the reference's own
    length per byte of text, it is not predicted."""
    piece_bytes = len(piece.encode("utf-8"))
    if piece_bytes < SHORT_PIECE_BYTES:
        speed = SHORT_PIECE_SPEED
    return reference_length / len(reference_text.encode("utf-8")) * piece_bytes / speed


def check_reference_seconds(seconds, name):
    if not math.isfinite(seconds) or not MIN_REFERENCE_SECONDS <= seconds <= MAX_REFERENCE_SECONDS:
        problem = "too short" if seconds < MIN_REFERENCE_SECONDS else "too long"
        raise BadRequest(f"The reference clip {name} is {problem}: it is {seconds:.1f} seconds, and a reference "
                         f"must be {MIN_REFERENCE_SECONDS:g} to {MAX_REFERENCE_SECONDS:g} seconds")


def wav_seconds(path):
    name = Path(path).name
    try:
        with wave.open(str(path), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        raise BadRequest(f"The reference clip {name} could not be read as a WAV file: {exc}") from exc
    if rate <= 0:
        raise BadRequest(f"The reference clip {name} has no sample rate")
    return frames / rate


def is_out_of_memory(exc):
    """torch.OutOfMemoryError cannot be named here (torch is not imported), so it is recognised by name."""
    if isinstance(exc, MemoryError) or any(cls.__name__ == "OutOfMemoryError" for cls in type(exc).__mro__):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def usable_audio(result):
    """(audio, sample rate, None) for a piece worth playing, or (None, 0, what is wrong with it)."""
    try:
        audio, rate = result
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        rate = int(rate)
    except (TypeError, ValueError):
        return None, 0, "the engine returned no audio"
    if rate <= 0:
        return None, 0, "the engine returned no sample rate"
    if audio.size == 0:
        return None, 0, "the audio is empty"
    if not np.isfinite(audio).all():
        return None, 0, "the audio is not a number (NaN or infinity)"
    if audio.size / rate < MIN_PIECE_SECONDS:
        return None, 0, f"the audio is only {audio.size / rate:.3f} seconds long"
    if float(np.abs(audio).max()) < SILENT_PEAK:
        return None, 0, "the audio is near-silent"
    return audio, rate, None


def write_piece(path, audio, rate):
    """Mono 16-bit PCM, written under another name first so the backend never sees half a file."""
    peak = float(np.abs(audio).max())
    if peak > 1.0:  # raw peaks reach 1.37: keep the shape and let the backend set the level
        audio = audio / peak
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    part = Path(str(path) + ".part")
    try:
        with wave.open(str(part), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        for attempt in range(1, REPLACE_ATTEMPTS + 1):
            try:
                os.replace(part, path)
                break
            except PermissionError:  # Windows refuses while antivirus or an indexer has the file open
                if attempt == REPLACE_ATTEMPTS:
                    raise
                time.sleep(0.05 * attempt)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- the protocol

class _Job:
    def __init__(self, id, pieces, out_dir, nfe, speed, seed):
        self.id, self.pieces, self.out_dir, self.nfe, self.speed, self.seed = id, pieces, out_dir, nfe, speed, seed
        self.cancel = threading.Event()  # set under the server's lock, only while this job is the active one


def _whole_number(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _job_from(request):
    """A job from a synthesize command, or BadRequest. Everything that can be checked without the reference."""
    id = request.get("id")
    if not isinstance(id, str) or not JOB_ID.fullmatch(id):
        raise BadRequest("synthesize needs an id of 1 to 64 letters, digits, dots, dashes or underscores")
    pieces = request.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        raise BadRequest("synthesize needs pieces: a list with at least one piece of text")
    for index, piece in enumerate(pieces):
        if not isinstance(piece, str) or not piece.strip():
            raise BadRequest(f"piece {index} is not text, or is empty")
    out_dir = request.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir.strip():
        raise BadRequest("synthesize needs out_dir: the folder the pieces are written to")
    nfe = request.get("nfe")
    if not _whole_number(nfe) or nfe not in STEP_COUNTS:
        raise BadRequest("synthesize needs nfe: 8, 16 or 32")
    speed = request.get("speed", 1.0)
    if (isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed)
            or not MIN_SPEED <= speed <= MAX_SPEED):
        raise BadRequest(f"speed must be a number from {MIN_SPEED:g} to {MAX_SPEED:g}")
    seed = request.get("seed")
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 31)
    elif not _whole_number(seed):
        raise BadRequest("seed must be a whole number or null")
    max_chars = request.get("max_chars")
    if max_chars is not None:
        if not _whole_number(max_chars) or max_chars < 1:
            raise BadRequest("max_chars must be a whole number of at least 1")
        for index, piece in enumerate(pieces):
            size = len(piece.encode("utf-8"))
            if size > max_chars:  # the package would cut it at a comma and join the parts; refuse instead
                raise BadRequest(f"piece {index} is {size} bytes of text, over the limit of {max_chars} for one "
                                 "piece with this reference clip")
    return _Job(id, list(pieces), Path(os.path.abspath(out_dir)), nfe, float(speed), seed % 2 ** 31)


class _Server:
    def __init__(self, stdout, engine):
        self._stdout, self._engine = stdout, engine
        self._lock = threading.RLock()  # guards _active, each job's cancel flag, and every write to stdout
        self._active = None  # the job that was accepted and has not ended yet
        self._reference = None  # {"seconds", "text"}; touched only by the engine thread
        self._work = queue.Queue()
        self.closing = threading.Event()  # the worker is ending, or stdout is gone
        self.thread = threading.Thread(target=self._work_loop, name="tts-engine", daemon=True)

    # ---- writing events

    def emit(self, event):
        line = json.dumps(event, default=str) + "\n"  # ASCII only: valid whatever encoding the pipe is read with
        with self._lock:
            try:
                self._stdout.write(line)
                self._stdout.flush()
            except (OSError, ValueError) as exc:  # the backend is gone: nobody to tell, nothing to stay for
                if not self.closing.is_set():
                    log("stdout is gone, ending:", exc)
                self.closing.set()

    def error(self, code, message, id=None, **more):
        self.emit({"event": "error", "id": id, "code": code, "message": message, **more})

    def _error_from(self, exc, id):
        if isinstance(exc, BadRequest):
            return {"event": "error", "id": id, "code": "bad_request", "message": str(exc)}
        if isinstance(exc, PieceFailed):
            return {"event": "error", "id": id, "code": "piece_failed", "message": str(exc), "index": exc.index}
        log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        code = "out_of_memory" if is_out_of_memory(exc) else "internal"
        return {"event": "error", "id": id, "code": code, "message": f"{type(exc).__name__}: {str(exc)[:500]}"}

    # ---- commands (the serving thread)

    def handle(self, line):
        """Act on one line of input. False when it asked the worker to shut down."""
        line = line.strip()
        if not line:
            return True
        try:
            request = json.loads(line)
        except ValueError as exc:
            self.error("bad_json", f"That line is not JSON: {exc}")
            return True
        if not isinstance(request, dict):
            self.error("bad_json", "A command must be a JSON object")
            return True
        command, id = request.get("cmd"), request.get("id")
        if not isinstance(id, str):
            id = None
        if command == "shutdown":
            return False
        if command == "cancel":
            self._cancel(id)
        elif command == "set_reference":
            self._accept_reference(request, id)
        elif command == "synthesize":
            self._accept_job(request, id)
        else:
            self.error("unknown_command", f"Unknown command: {command!r}", id)
        return True

    def _cancel(self, id):
        with self._lock:
            if self._active is not None and self._active.id == id:
                self._active.cancel.set()
            else:
                log(f"cancel for {id!r} ignored: it is not the running job")

    def _accept_reference(self, request, id):
        wav, text = request.get("wav"), request.get("text")
        with self._lock:
            if not isinstance(wav, str) or not wav.strip() or not isinstance(text, str):
                self.error("bad_request", "set_reference needs wav (a path) and text (the clip's transcript)", id)
            elif self._active is not None:
                self.error("bad_request", f"The reference cannot change while {self._active.id} is being "
                                          "synthesised", id)
            else:
                self._work.put(("reference", (wav, text, id)))

    def _accept_job(self, request, id):
        try:
            job = _job_from(request)
        except BadRequest as exc:
            self.error("bad_request", str(exc), id)
            return
        with self._lock:
            if self._active is not None:
                self.error("bad_request", f"Already synthesising {self._active.id}: one request at a time", job.id)
            else:
                self._active = job
                self._work.put(("job", job))

    def stop(self):
        """Ask whatever is running to stop, and let nothing that is still queued start."""
        with self._lock:
            self.closing.set()
            if self._active is not None:
                self._active.cancel.set()
        self._work.put(None)

    # ---- the engine thread

    def _work_loop(self):
        while True:
            item = self._work.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == "reference":
                    if not self.closing.is_set():
                        self._set_reference(*payload)
                else:
                    self._run_job(payload)
            except Exception as exc:  # noqa: BLE001 - a bug here must not silently end the thread that does the work
                self.emit(self._error_from(exc, None))

    def _set_reference(self, wav, text, id):
        self._reference = None  # a refused clip must not leave the previous voice in force
        try:
            path = Path(wav)
            if not path.is_file():
                raise BadRequest(f"The reference clip {path.name} was not found")
            text = text.strip()
            model_text = reference_text(text)
            check_reference_seconds(wav_seconds(path), path.name)
            seconds = float(self._engine.set_reference(str(path), text))
            check_reference_seconds(seconds, path.name)  # what the engine decoded is what the model is given
        except Exception as exc:  # noqa: BLE001 - every failure is reported and the worker stays up
            self.emit(self._error_from(exc, id))
            return
        self._reference = {"seconds": seconds, "text": model_text}
        self.emit({"event": "reference", "seconds": round(seconds, 3)})

    def _check_job(self, job):
        reference = self._reference
        if reference is None:
            raise BadRequest("Set a reference clip before synthesising: no reference is loaded")
        for index, piece in enumerate(job.pieces):
            total = reference["seconds"] + generated_length(reference["seconds"], reference["text"], piece, job.speed)
            if total > MAX_CALL_SECONDS:
                raise BadRequest(f"piece {index} is too long for this reference clip: together they come to about "
                                 f"{total:.0f} seconds in one go, and the voice engine stays intelligible only up "
                                 f"to {MAX_CALL_SECONDS:g}")
        try:
            job.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BadRequest(f"The output folder {job.out_dir} could not be created: {exc}") from exc

    def _emit_unless_cancelled(self, job, event):
        with self._lock:
            if job.cancel.is_set():
                return False
            self.emit(event)
            return True

    def _synthesize(self, job, index, text):
        """One piece worth playing, as (audio, sample rate, the seed that gave it)."""
        problem = None
        for seed in (job.seed, (job.seed + 1) % 2 ** 31):
            if job.cancel.is_set():
                raise Stopped()
            result = self._engine.synthesize_piece(text, job.nfe, job.speed, seed, job.cancel.is_set)
            if job.cancel.is_set():
                raise Stopped()
            audio, rate, problem = usable_audio(result)
            if problem is None:
                return audio, rate, seed
            log(f"{job.id} piece {index} with seed {seed}: {problem}")
        raise PieceFailed(index, problem)

    def _run_job(self, job):
        began = time.perf_counter()
        announced, seconds = 0, 0.0

        def elapsed():
            return round(time.perf_counter() - began, 3)

        try:
            if self.closing.is_set():
                raise Stopped()
            self._check_job(job)
            if not self._emit_unless_cancelled(job, {"event": "started", "id": job.id, "pieces": len(job.pieces)}):
                raise Stopped()
            for index, text in enumerate(job.pieces):
                audio, rate, seed = self._synthesize(job, index, text)
                path = job.out_dir / f"{job.id}_{index:03d}.wav"
                write_piece(path, audio, rate)
                length = len(audio) / rate
                if not self._emit_unless_cancelled(job, {
                        "event": "audio", "id": job.id, "index": index, "path": str(path), "samples": len(audio),
                        "sample_rate": rate, "seconds": round(length, 3), "elapsed": elapsed(), "seed": seed,
                        "text": text}):
                    path.unlink(missing_ok=True)  # nobody was told about it, so nobody would delete it
                    raise Stopped()
                announced, seconds = announced + 1, seconds + length
            end = {"event": "done", "id": job.id, "pieces": announced, "seconds": round(seconds, 3)}
        except Stopped:
            end = None
        except Exception as exc:  # noqa: BLE001 - reported below; the worker stays up for the next request
            end = self._error_from(exc, job.id)
        with self._lock:
            # Decided and written in one step with the cancel flag, so a job whose cancel was accepted never
            # reports anything else, and the next request is accepted from the moment its ending is on stdout.
            if end is None or job.cancel.is_set():
                end = {"event": "cancelled", "id": job.id, "index": announced}
            if end["event"] != "error":
                end["elapsed"] = elapsed()
            self._active = None
            self.emit(end)


def _read_lines(stdin, commands):
    try:
        for line in stdin:
            commands.put(line)
    except (OSError, ValueError) as exc:
        log("stdin could not be read, ending:", exc)
    finally:
        commands.put(None)  # end of input: the backend closed the pipe or went away


def serve(stdin, stdout, engine, parent_alive=None, *, poll_seconds=POLL_SECONDS, stop_wait=None) -> int:
    """Run the protocol until shutdown, end of input, a broken stdout, or parent_alive() answering False.

    `engine` has load() (may return a dict of facts for the `ready` event), set_reference(wav_path, text) -> the
    clip's seconds, and synthesize_piece(text, nfe, speed, seed, should_stop) -> (float audio, sample rate), which
    should raise Stopped as soon as should_stop() is true. `stop_wait(thread)` waits for the engine thread when
    the worker ends (by default up to STOP_GRACE_SECONDS). Returns the process's exit code."""
    server = _Server(stdout, engine)
    try:
        facts = engine.load()
    except Exception as exc:  # noqa: BLE001 - whatever stopped the model loading is the answer to give
        log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        server.error("internal", f"The voice engine could not start. {type(exc).__name__}: {str(exc)[:1000]}")
        return 1
    server.emit({"event": "ready", **{k: v for k, v in (facts if isinstance(facts, dict) else {}).items()
                                      if k != "event"}})
    commands = queue.Queue()
    threading.Thread(target=_read_lines, args=(stdin, commands), name="tts-stdin", daemon=True).start()
    server.thread.start()
    while not server.closing.is_set():
        try:
            line = commands.get(timeout=poll_seconds)
        except queue.Empty:
            if parent_alive is not None and not parent_alive():
                log("the app is gone, ending")
                break
            continue
        if line is None or not server.handle(line):
            break
    server.stop()
    if stop_wait is None:
        server.thread.join(STOP_GRACE_SECONDS)
    else:
        stop_wait(server.thread)
    return 0


# ---------------------------------------------------------------- the real engine

def _load_sibling(name):
    spec = importlib.util.spec_from_file_location("aivoice_tts_" + name, TTS_DIR / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class F5Engine:
    """F5-TTS v1 Base with the Vocos vocoder on the GPU, in half precision, from local files only.

    It never uses torchaudio.load (needs torchcodec on this build), the package's F5TTS class or infer_process
    (both use torchaudio.load, and chunk long text at commas), or preprocess_ref_audio_text (cuts the clip but not
    its transcript, and downloads Whisper when the transcript is empty). See the spike notes, sections 5.5 to 7."""

    SAMPLE_RATE, HOP = 24000, 256  # F5TTS_v1_Base's mel settings
    TARGET_RMS = 0.1  # the package raises a quiet reference to this level and lowers the result by as much
    FILES = ("F5TTS_v1_Base/model_1250000.safetensors", "F5TTS_v1_Base/vocab.txt",
             "vocos-mel-24khz/config.yaml", "vocos-mel-24khz/pytorch_model.bin")

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self._reference = None
        self._should_stop = None

    def load(self):
        began = time.perf_counter()
        missing = [name for name in self.FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Model files are missing under {self.model_dir}: {', '.join(missing)}. "
                                    "Run tts/setup.sh.")
        import warnings
        warnings.filterwarnings("ignore", category=SyntaxWarning)  # pydub's, on first import under Python 3.12

        _load_sibling("shims").apply()
        import soundfile
        import torch
        import torchaudio
        from importlib.resources import files
        from omegaconf import OmegaConf
        from f5_tts.infer import utils_infer
        from f5_tts.model import DiT
        from f5_tts.model.utils import convert_char_to_pinyin, seed_everything

        self._torch, self._torchaudio, self._soundfile = torch, torchaudio, soundfile
        self._to_model_text, self._seed_everything = convert_char_to_pinyin, seed_everything
        imported = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot use the GPU (torch.cuda.is_available() is False). The processor is "
                               "not a fallback: it is hundreds of times too slow.")

        self._vocoder = utils_infer.load_vocoder("vocos", True, str(self.model_dir / "vocos-mel-24khz"), "cuda")
        config = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
        # Half precision is forced. The package picks it by a capability test that could one day answer otherwise,
        # and single precision is 6.6 times slower on this card. load_model() looks load_checkpoint up by name.
        package_load_checkpoint = utils_infer.load_checkpoint

        def load_checkpoint_half(model, ckpt_path, device, dtype=None, use_ema=True):
            return package_load_checkpoint(model, ckpt_path, device, dtype=torch.float16, use_ema=use_ema)

        utils_infer.load_checkpoint = load_checkpoint_half
        try:
            self._model = utils_infer.load_model(
                DiT, config.model.arch, str(self.model_dir / "F5TTS_v1_Base" / "model_1250000.safetensors"),
                "vocos", str(self.model_dir / "F5TTS_v1_Base" / "vocab.txt"), "euler", True, "cuda")
        finally:
            utils_infer.load_checkpoint = package_load_checkpoint
        dtype = next(self._model.parameters()).dtype
        if dtype != torch.float16:
            raise RuntimeError(f"The model loaded as {dtype}, not half precision")
        # Called before every pass through the transformer, so a stop takes effect within one sampling step.
        self._model.transformer.register_forward_pre_hook(self._before_step)
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        self._warm_up()
        return {"import_s": round(imported - began, 2), "load_s": round(loaded - imported, 2),
                "warmup_s": round(time.perf_counter() - loaded, 2), "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "dtype": str(dtype).replace("torch.", ""),
                "vram_reserved_mb": round(torch.cuda.memory_reserved() / 2 ** 20), "sample_rate": self.SAMPLE_RATE}

    def _before_step(self, module, args):
        if self._should_stop is not None and self._should_stop():
            raise Stopped()

    def _warm_up(self):
        """One short synthesis, so that the first request does not pay for the GPU library's first call (3 to 4
        seconds, 10 to 17 with an empty kernel cache). The reference is noise; the result is thrown away."""
        torch = self._torch
        try:
            noise = torch.randn(1, 3 * self.SAMPLE_RATE) * self.TARGET_RMS
            self._reference = {"audio": noise.to("cuda"), "text": "This is a warm up. ", "rms": self.TARGET_RMS,
                               "frames": noise.shape[-1] // self.HOP}
            self.synthesize_piece("A short sentence to warm the engine up.", 8, 1.0, 0, lambda: False)
        except Exception as exc:  # noqa: BLE001 - a failed warm-up costs the first request some seconds, no more
            if is_out_of_memory(exc):
                raise
            log("warm-up failed (the first request will be slower):", f"{type(exc).__name__}: {exc}")
        finally:
            self._reference = None

    def _release_after(self, exc):
        if is_out_of_memory(exc):
            self._torch.cuda.empty_cache()

    def set_reference(self, wav_path, text):
        torch, name = self._torch, Path(wav_path).name
        self._reference = None
        try:
            data, rate = self._soundfile.read(str(wav_path), dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001 - libsndfile's own error type
            raise BadRequest(f"The reference clip {name} could not be read: {exc}") from exc
        seconds = len(data) / rate
        check_reference_seconds(seconds, name)
        model_text = reference_text(text)
        try:
            audio = torch.from_numpy(data.mean(axis=1)[None].copy())  # mono, shaped 1 x samples
            rms = float(torch.sqrt(torch.mean(torch.square(audio))))
            if not math.isfinite(rms) or rms < 1e-4:
                raise BadRequest(f"The reference clip {name} is silent")
            if rms < self.TARGET_RMS:
                audio = audio * (self.TARGET_RMS / rms)
            if rate != self.SAMPLE_RATE:
                audio = self._torchaudio.transforms.Resample(rate, self.SAMPLE_RATE)(audio)
            self._reference = {"audio": audio.to("cuda"), "text": model_text, "rms": rms,
                               "frames": audio.shape[-1] // self.HOP}
        except Exception as exc:
            self._release_after(exc)
            raise
        return seconds

    def synthesize_piece(self, text, nfe, speed, seed, should_stop):
        torch, reference = self._torch, self._reference
        if reference is None:
            raise BadRequest("No reference clip is loaded")
        frames = reference["frames"]
        duration = frames + int(generated_length(frames, reference["text"], text, speed))
        if duration * self.HOP / self.SAMPLE_RATE > MAX_CALL_SECONDS:
            raise BadRequest(f"That piece is too long for this reference clip (more than {MAX_CALL_SECONDS:g} "
                             "seconds in one go)")
        self._seed_everything(int(seed))
        self._should_stop = should_stop
        try:
            with torch.inference_mode():
                generated, _ = self._model.sample(
                    cond=reference["audio"], text=self._to_model_text([reference["text"] + text]),
                    duration=duration, steps=nfe, cfg_strength=2.0, sway_sampling_coef=-1.0)
                del _
                mel = generated.to(torch.float32)[:, frames:, :].permute(0, 2, 1)
                wave_out = self._vocoder.decode(mel)
                if reference["rms"] < self.TARGET_RMS:
                    wave_out = wave_out * (reference["rms"] / self.TARGET_RMS)
                audio = wave_out.squeeze().cpu().numpy()
        except Exception as exc:
            self._model.transformer.clear_cache()  # sample() clears it only when it runs to the end
            self._release_after(exc)
            raise
        finally:
            self._should_stop = None
        return audio.astype(np.float32, copy=False), self.SAMPLE_RATE


# ---------------------------------------------------------------- the process

def apply_environment(env, cache_dir=CACHE_DIR):
    """The settings this process must have whatever started it. Offline is forced: synthesis never uses the
    network. The rest only fills in what the launcher left out, so that nothing lands in the user's profile."""
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    cache = Path(cache_dir)
    defaults = {"WANDB_MODE": "disabled", "ROCM_SDK_TARGET_FAMILY": "custom", "MIOPEN_LOG_LEVEL": "3"}
    folders = {"MIOPEN_USER_DB_PATH": cache / "miopen" / "db", "MIOPEN_CUSTOM_CACHE_DIR": cache / "miopen",
               "HF_HOME": cache / "hf", "TORCH_HOME": cache / "torch", "XDG_CACHE_HOME": cache / "xdg",
               "MPLCONFIGDIR": cache / "mpl", "NUMBA_CACHE_DIR": cache / "numba"}
    for name, value in defaults.items():
        env.setdefault(name, value)
    for name, folder in folders.items():
        if name not in env:
            env[name] = str(folder)
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log(f"could not create {folder}: {exc}")


def take_over_stdout():
    """Keep the real stdout for the protocol and send everything else that writes to stdout to stderr.

    The engine's libraries print while they are imported and while they work. sys.stdout covers Python's print();
    native libraries write to file descriptor 1 or to the process's standard output handle directly, so that is
    pointed at stderr as well (on Windows the C runtime's dup2 also replaces the standard handle). The protocol
    keeps a private copy of the original, which child processes do not inherit."""
    try:
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        pass
    protocol_fd = os.dup(1)
    try:
        os.dup2(2, 1)
    except OSError:  # started without a stderr: library output has nowhere to go but away
        sink = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink, 1)
        os.close(sink)
    sys.stdout = sys.stderr if sys.stderr is not None else open(os.devnull, "w")
    # Buffered underneath (a raw pipe may take only part of a write); every event is flushed as it is written.
    return os.fdopen(protocol_fd, "w", encoding="ascii", errors="backslashreplace", newline="\n")


_parent_handles = {}  # pid -> open handle, kept so that a recycled PID is never mistaken for the original process


def process_alive(pid):
    """True while the process with this PID is running. Copied from trainer/procutil.py (process_alive and
    wait_for_exit with no wait), because this worker may import nothing from trainer/."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True
    synchronize, wait_timeout, access_denied = 0x00100000, 0x00000102, 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    handle = _parent_handles.get(pid)
    if handle is None:
        handle = kernel32.OpenProcess(synchronize, 0, int(pid))
        if not handle:
            return ctypes.get_last_error() == access_denied  # denied: it exists but belongs to someone else
        _parent_handles[pid] = handle
    return kernel32.WaitForSingleObject(handle, 0) == wait_timeout


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Text to speech with F5-TTS, driven by JSON lines on stdin.")
    parser.add_argument("--model-dir", required=True, help="the folder tts/setup.sh downloads the models to")
    parser.add_argument("--parent-pid", type=int, default=None, help="end when this process is gone")
    return parser.parse_args(argv)


def main(argv=None, engine_factory=F5Engine) -> int:
    args = parse_args(argv)
    protocol = take_over_stdout()  # before anything that could import a library
    apply_environment(os.environ)
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    parent_alive = (lambda: process_alive(args.parent_pid)) if args.parent_pid else None
    return serve(stdin, protocol, engine_factory(args.model_dir), parent_alive)


def run(argv=None, engine_factory=F5Engine):
    """main(), then end the process at once with its exit code. Never returns.

    Not sys.exit(): the reader thread is still blocked reading stdin, which makes an ordinary interpreter shutdown
    die with a fatal error (exit code 0xC0000005) when the worker ended for any reason but end of input; and if a
    piece is still inside the GPU library, or that library's own shutdown hangs, an ordinary exit would leave
    this process on the GPU. Every protocol line is already flushed when it is written."""
    code = main(argv, engine_factory)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError, AttributeError):
            pass
    os._exit(code)


if __name__ == "__main__":
    run()
