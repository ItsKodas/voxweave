"""Tests of installer/bootstrap.py with a fake uv and a fake downloader. Standard library only, so they run with
any Python 3.12 (python -m unittest discover -s installer/tests) and under pytest alike."""
import errno
import hashlib
import http.server
import io
import json
import sys
import tempfile
import re
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bootstrap  # noqa: E402

MODEL = bytes(range(256)) * 40  # 10240 bytes
SPEC = bootstrap.ModelSpec("https://example.invalid/model.onnx", hashlib.sha256(MODEL).hexdigest(), len(MODEL))
GOOD_RUNTIME = {"directml": "1.24.4", "plain": None, "providers": ["DmlExecutionProvider", "CPUExecutionProvider"]}


class FakeUv:
    """Stands in for every process the bootstrap starts: uv itself and the environment's Python."""

    def __init__(self, paths, runtime=None, python_version="3.12", fail=None):
        self.paths = paths
        self.runtime = dict(GOOD_RUNTIME if runtime is None else runtime)
        self.python_version = python_version
        self.fail = fail  # the uv subcommand that fails, e.g. "sync"
        self.calls = []
        self.envs = []
        self.state_seen = []

    def uv_calls(self, *words):
        return [c for c in self.calls if c[0] == str(self.paths.uv) and list(c[1:1 + len(words)]) == list(words)]

    def __call__(self, argv, env, tick=None):
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv), argv
        self.calls.append(argv)
        self.envs.append(env)
        self.state_seen.append(self.paths.state.exists())
        if tick is not None:
            tick()
        if argv[0] == str(self.paths.uv):
            if argv[1] == "venv":
                if self.fail == "venv":
                    return 2, "error: no network\n"
                assert not self.paths.venv.exists(), "the old environment must be gone before a new one is made"
                self.paths.python.parent.mkdir(parents=True)
                self.paths.python.write_bytes(b"python")
                return 0, ""
            if argv[1:3] == ["pip", "sync"]:
                if self.fail == "sync":
                    return 1, "line one\nerror: could not reach pypi.org\n"
                return 0, "Installed 46 packages\n"
            raise AssertionError(f"unexpected uv call {argv}")
        if argv[0] == str(self.paths.python):
            if "sys.version_info" in argv[-1]:
                return (0, self.python_version + "\n") if self.python_version else (1, "broken")
            return 0, "noise\n" + json.dumps(self.runtime) + "\n"
        raise AssertionError(f"unexpected process {argv}")


class FakeWeb:
    """fetch(url, offset) -> (stream, start, total). Can ignore ranges, cut a response short or serve other bytes."""

    def __init__(self, content=MODEL, honour_range=True, cut_after=None, errors=0):
        self.content = content
        self.honour_range = honour_range
        self.cut_after = cut_after  # bytes served by the first response before the connection drops
        self.errors = errors  # how many requests fail outright first
        self.offsets = []

    def __call__(self, url, offset):
        assert url == SPEC.url
        self.offsets.append(offset)
        if self.errors > 0:
            self.errors -= 1
            raise OSError("connection refused")
        start = offset if self.honour_range else 0
        body = self.content[start:]
        if self.cut_after is not None:
            body, self.cut_after = body[:self.cut_after], None
            return _Dropping(body), start, len(self.content)
        return io.BytesIO(body), start, len(self.content)


class _Dropping(io.BytesIO):
    def read(self, size=-1):
        data = super().read(size)
        if not data:
            raise OSError("connection reset")
        return data


class BootstrapCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "root"
        self.resources = base / "resources"
        (self.resources / "bin").mkdir(parents=True)
        (self.resources / "bin" / "uv.exe").write_bytes(b"uv")
        (self.resources / "setup").mkdir()
        (self.resources / "setup" / "requirements.lock").write_bytes(b"numpy==2.5.4\n")
        self.paths = bootstrap.Paths(self.root, self.resources)
        self.uv = FakeUv(self.paths)
        self.web = FakeWeb()
        self.out = io.StringIO()
        self.sleeps = []

    def make(self, **over):
        args = dict(version="0.1.0", model=SPEC, run=self.uv, fetch=self.web, out=self.out,
                    sleep=self.sleeps.append, free_bytes=lambda path: 10 ** 12)
        args.update(over)
        return bootstrap.Bootstrap(self.paths, **args)

    def run_setup(self, **over):
        self.out.seek(0)
        self.out.truncate()
        code = self.make(**over).run()
        return code, self.lines()

    def lines(self):
        return [json.loads(line) for line in self.out.getvalue().splitlines()]

    def test_a_fresh_run_builds_everything_and_writes_the_state_last(self):
        code, lines = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(lines[-1], {"done": True, "ok": True, "error": None})
        self.assertEqual(len(self.uv.uv_calls("venv")), 1)
        self.assertEqual(self.uv.uv_calls("venv")[0][2:], ["--python", "3.12", str(self.paths.venv)])
        sync = self.uv.uv_calls("pip", "sync")
        self.assertEqual(len(sync), 1)
        self.assertEqual(sync[0][3:], ["--python", str(self.paths.python), "--require-hashes", "--compile-bytecode",
                                       str(self.paths.lock)])
        self.assertEqual(self.paths.model.read_bytes(), MODEL)
        self.assertFalse(self.paths.model.with_name(self.paths.model.name + ".part").exists())
        state = json.loads(self.paths.state.read_text())
        self.assertEqual(state["version"], "0.1.0")
        self.assertEqual(state["lock_sha256"], hashlib.sha256(b"numpy==2.5.4\n").hexdigest())
        self.assertNotIn(True, self.uv.state_seen, "setup.json must be written after every process has run")

    def test_progress_lines_have_a_step_a_message_and_a_rising_percent(self):
        _, lines = self.run_setup()
        progress = lines[:-1]
        self.assertTrue(progress)
        for line in progress:
            self.assertEqual(set(line), {"step", "percent", "message"})
            self.assertIsInstance(line["message"], str)
        percents = [line["percent"] for line in progress]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual(percents[-1], 100)
        self.assertEqual([s for s in dict.fromkeys(line["step"] for line in progress)],
                         ["environment", "packages", "verify", "model", "finish"])

    def test_a_second_run_creates_and_downloads_nothing(self):
        self.run_setup()
        self.uv.calls.clear()
        self.web.offsets.clear()
        code, lines = self.run_setup()
        self.assertEqual(code, 0)
        self.assertTrue(lines[-1]["ok"])
        self.assertEqual(self.uv.uv_calls("venv"), [])
        self.assertEqual(self.web.offsets, [])
        # The package step still runs: uv finds everything in place within a second and repairs what is not.
        self.assertEqual(len(self.uv.uv_calls("pip", "sync")), 1)

    def test_every_uv_call_keeps_its_caches_and_python_under_the_root(self):
        self.run_setup()
        for env in self.uv.envs:
            self.assertEqual(env["UV_CACHE_DIR"], str(self.paths.root / "runtime" / "uv-cache"))
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(self.paths.root / "runtime" / "python"))
            self.assertEqual(env["UV_PYTHON_PREFERENCE"], "only-managed")
            self.assertEqual(env["UV_NO_CONFIG"], "1")
            self.assertEqual(env["UV_PYTHON_INSTALL_REGISTRY"], "0")
            self.assertNotIn("VIRTUAL_ENV", env)

    def test_python_ignores_the_users_site_packages_and_the_current_folder(self):
        self.run_setup()
        for env in self.uv.envs:
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(env["PYTHONSAFEPATH"], "1")

    def test_an_environment_with_the_wrong_python_is_rebuilt(self):
        self.run_setup()
        (self.paths.venv / "leftover.txt").write_text("old")
        self.uv.python_version = ""
        self.uv.calls.clear()
        original = self.uv.__call__

        def healed(argv, env, tick=None):
            result = original(argv, env, tick)
            if argv[1:2] == ["venv"]:
                self.uv.python_version = "3.12"
            return result

        code, _ = self.run_setup(run=healed)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.uv.uv_calls("venv")), 1)
        self.assertFalse((self.paths.venv / "leftover.txt").exists())

    def test_a_half_finished_download_is_continued(self):
        part = self.paths.model.with_name(self.paths.model.name + ".part")
        part.parent.mkdir(parents=True)
        part.write_bytes(MODEL[:4000])
        code, _ = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(self.web.offsets, [4000])
        self.assertEqual(self.paths.model.read_bytes(), MODEL)
        self.assertFalse(part.exists())

    def test_a_server_that_ignores_the_range_starts_the_file_again(self):
        part = self.paths.model.with_name(self.paths.model.name + ".part")
        part.parent.mkdir(parents=True)
        part.write_bytes(b"x" * 4000)
        self.web.honour_range = False
        code, _ = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(self.paths.model.read_bytes(), MODEL)

    def test_a_dropped_connection_is_retried_from_where_it_stopped(self):
        self.web.cut_after = 3000
        code, lines = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(self.web.offsets, [0, 3000])
        self.assertEqual(self.paths.model.read_bytes(), MODEL)
        self.assertEqual(len(self.sleeps), 1)

    def test_a_download_that_never_works_fails_with_a_readable_error(self):
        self.web.errors = 99
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertFalse(lines[-1]["ok"])
        self.assertIn("base speech model", lines[-1]["error"])
        self.assertIn("connection refused", lines[-1]["error"])
        self.assertIn("internet", lines[-1]["error"])
        self.assertEqual(len(self.web.offsets), bootstrap.DOWNLOAD_ATTEMPTS)
        self.assertFalse(self.paths.model.exists())
        self.assertFalse(self.paths.state.exists())

    def test_a_download_with_the_wrong_content_is_refused_and_removed(self):
        self.web.content = b"y" * len(MODEL)
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("checksum", lines[-1]["error"])
        self.assertFalse(self.paths.model.exists())
        self.assertFalse(self.paths.model.with_name(self.paths.model.name + ".part").exists())
        self.assertFalse(self.paths.state.exists())
        self.assertEqual(self.web.offsets, [0, 0], "one fresh download is tried after a bad checksum")

    def test_an_installed_model_with_the_wrong_content_is_downloaded_again(self):
        self.paths.model.parent.mkdir(parents=True)
        self.paths.model.write_bytes(b"damaged")
        code, _ = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(self.paths.model.read_bytes(), MODEL)

    def test_a_full_disk_during_the_download_is_a_disk_problem_and_is_not_retried(self):
        with mock.patch.object(bootstrap, "_write_block", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("disk is full", lines[-1]["error"])
        self.assertNotIn("internet", lines[-1]["error"])
        self.assertEqual(self.web.offsets, [0], "it is not tried five times like a network problem")
        self.assertEqual(self.sleeps, [])
        self.assertFalse(self.paths.state.exists())

    def test_any_other_error_writing_the_file_is_reported_as_such_and_not_retried(self):
        with mock.patch.object(bootstrap, "_write_block", side_effect=PermissionError(errno.EACCES, "Access is denied")):
            code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("could not be written", lines[-1]["error"])
        self.assertIn("Access is denied", lines[-1]["error"])
        self.assertEqual(self.web.offsets, [0])

    def test_the_log_is_only_appended_to_never_started_afresh(self):
        self.paths.runtime.mkdir(parents=True)
        self.paths.log.write_text("earlier\n" + "x" * 3_000_000, encoding="utf-8")
        self.run_setup()
        text = self.paths.log.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("earlier\n"), "rotating the log is the app's job alone")
        self.assertIn("pip", text)

    def test_a_longer_part_file_than_the_model_is_discarded(self):
        part = self.paths.model.with_name(self.paths.model.name + ".part")
        part.parent.mkdir(parents=True)
        part.write_bytes(MODEL + b"extra")
        code, _ = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(self.web.offsets, [0])

    def test_a_complete_part_file_is_checked_without_a_request(self):
        part = self.paths.model.with_name(self.paths.model.name + ".part")
        part.parent.mkdir(parents=True)
        part.write_bytes(MODEL)
        code, _ = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(self.web.offsets, [])
        self.assertEqual(self.paths.model.read_bytes(), MODEL)

    def test_the_plain_onnxruntime_package_fails_the_setup(self):
        self.uv.runtime["plain"] = "1.24.4"
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("onnxruntime", lines[-1]["error"])
        self.assertFalse(self.paths.state.exists())

    def test_an_environment_that_fails_its_check_is_removed_so_that_retry_rebuilds_it(self):
        good = dict(self.uv.runtime)
        self.uv.runtime = {"broken": True}  # what a half-installed package looks like: the imports fail
        original = self.uv.__call__

        def failing_imports(argv, env, tick=None):
            code, text = original(argv, env, tick)
            if argv[0] == str(self.paths.python) and "sys.version_info" not in argv[-1]:
                return 1, "ImportError: DLL load failed while importing onnxruntime_pybind11_state"
            return code, text

        code, lines = self.run_setup(run=failing_imports)
        self.assertEqual(code, 1)
        self.assertIn("DLL load failed", lines[-1]["error"])
        self.assertIn("Retry builds the environment again", lines[-1]["error"])
        self.assertFalse(self.paths.venv.exists(), "the environment that failed is gone")
        self.assertFalse(self.paths.state.exists())

        self.uv.runtime = good
        self.uv.calls.clear()
        code, _ = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.uv.uv_calls("venv")), 1, "the retry made a new environment")
        self.assertEqual(len(self.uv.uv_calls("pip", "sync")), 1)

    def test_a_second_failed_check_in_a_row_also_empties_the_download_cache(self):
        """The environment's files are links into uv's cache, so a damaged file there would come straight back."""
        self.uv.runtime = {**GOOD_RUNTIME, "directml": None}
        cached = self.paths.cache / "archive-v0" / "numpy.whl"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"damaged")
        self.run_setup()
        self.assertTrue(cached.exists(), "one failure keeps the cache: a retry should not download everything again")
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertFalse(self.paths.cache.exists(), "the second failure in a row removes it")
        self.assertIn("downloads the packages again", lines[-1]["error"])
        self.uv.runtime = dict(GOOD_RUNTIME)
        self.assertEqual(self.run_setup()[0], 0)
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"fine")
        self.uv.runtime = {**GOOD_RUNTIME, "directml": None}
        self.run_setup()
        self.assertTrue(cached.exists(), "a success in between starts the count again")

    def test_every_kind_of_failed_check_removes_the_environment(self):
        for runtime in ({**GOOD_RUNTIME, "plain": "1.24.4"}, {**GOOD_RUNTIME, "directml": None},
                        {**GOOD_RUNTIME, "providers": ["CPUExecutionProvider"]}):
            self.uv.runtime = runtime
            code, lines = self.run_setup()
            self.assertEqual(code, 1)
            self.assertIn("Retry", lines[-1]["error"])
            self.assertFalse(self.paths.venv.exists(), runtime)

    def test_a_missing_directml_package_fails_the_setup(self):
        self.uv.runtime["directml"] = None
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("onnxruntime-directml", lines[-1]["error"])
        self.assertFalse(self.paths.state.exists())

    def test_a_runtime_without_the_directml_provider_fails_the_setup(self):
        self.uv.runtime["providers"] = ["CPUExecutionProvider"]
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("DirectML", lines[-1]["error"])

    def test_a_failing_package_install_reports_the_end_of_uvs_output(self):
        self.uv.fail = "sync"
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("could not reach pypi.org", lines[-1]["error"])
        self.assertIn("packages", lines[-1]["error"])
        self.assertFalse(self.paths.state.exists())
        self.assertEqual(self.web.offsets, [])

    def test_a_failed_run_is_completed_by_the_next_one(self):
        self.uv.fail = "sync"
        self.run_setup()
        self.uv.fail = None
        self.uv.calls.clear()
        code, _ = self.run_setup()
        self.assertEqual(code, 0)
        self.assertEqual(self.uv.uv_calls("venv"), [], "the environment of the failed run is kept")
        self.assertTrue(self.paths.state.exists())

    def test_too_little_disk_space_is_refused_before_anything_is_downloaded(self):
        code, lines = self.run_setup(free_bytes=lambda path: 10 ** 6)
        self.assertEqual(code, 1)
        self.assertIn("free space", lines[-1]["error"])
        self.assertEqual(self.uv.calls, [])
        self.assertEqual(self.web.offsets, [])

    def test_disk_space_is_not_asked_for_once_everything_is_installed(self):
        self.run_setup()
        code, _ = self.run_setup(free_bytes=lambda path: 10 ** 6)
        self.assertEqual(code, 0)

    def test_an_unexpected_error_is_reported_not_raised(self):
        def explode(argv, env, tick=None):
            raise RuntimeError("boom")

        code, lines = self.run_setup(run=explode)
        self.assertEqual(code, 1)
        self.assertEqual(lines[-1]["done"], True)
        self.assertIn("boom", lines[-1]["error"])

    def test_a_missing_bundled_file_is_reported(self):
        (self.resources / "bin" / "uv.exe").unlink()
        code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("uv.exe", lines[-1]["error"])

    def test_nothing_is_written_outside_the_root(self):
        before = sorted(p.relative_to(self.resources).as_posix() for p in self.resources.rglob("*"))
        self.run_setup()
        after = sorted(p.relative_to(self.resources).as_posix() for p in self.resources.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(sorted(p.name for p in self.root.parent.iterdir()), ["resources", "root"])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["data", "runtime"])

    def test_a_second_setup_at_the_same_time_is_refused(self):
        self.paths.runtime.mkdir(parents=True)
        with bootstrap.single_run(self.paths.runtime / "setup.lock"):
            code, lines = self.run_setup()
        self.assertEqual(code, 1)
        self.assertIn("already running", lines[-1]["error"])
        code, _ = self.run_setup()
        self.assertEqual(code, 0)


class PathsCase(unittest.TestCase):
    def test_layout(self):
        paths = bootstrap.Paths(Path("C:/Users/x/AppData/Local/Voxweave"), Path("C:/app"))
        root = Path("C:/Users/x/AppData/Local/Voxweave").resolve()
        self.assertEqual(paths.venv, root / "runtime" / "backend" / ".venv")
        self.assertEqual(paths.python, root / "runtime" / "backend" / ".venv" / "Scripts" / "python.exe")
        self.assertEqual(paths.model, root / "data" / "pretrained" / "vec-768-layer-12.onnx")
        self.assertEqual(paths.state, root / "runtime" / "setup.json")
        self.assertEqual(paths.uv, Path("C:/app").resolve() / "bin" / "uv.exe")
        self.assertEqual(paths.lock, Path("C:/app").resolve() / "setup" / "requirements.lock")

    def test_a_drive_root_is_refused(self):
        with self.assertRaises(bootstrap.SetupError):
            bootstrap.Paths(Path("C:/"), Path("C:/app"))


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    content = MODEL
    ranges = True
    seen = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append(self.headers.get("Range"))
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/model")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        header = self.headers.get("Range")
        if header and self.ranges:
            start = int(header.split("=")[1].split("-")[0])
            body = self.content[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(self.content) - 1}/{len(self.content)}")
        else:
            body = self.content
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class OpenUrlCase(unittest.TestCase):
    """The real downloader against a local web server."""

    def setUp(self):
        _RangeHandler.seen = []
        _RangeHandler.ranges = True
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def test_from_the_start(self):
        stream, start, total = bootstrap.open_url(self.base + "/model", 0)
        with stream:
            self.assertEqual((start, total, stream.read()), (0, len(MODEL), MODEL))

    def test_from_an_offset_through_a_redirect(self):
        stream, start, total = bootstrap.open_url(self.base + "/redirect", 1000)
        with stream:
            self.assertEqual((start, total, stream.read()), (1000, len(MODEL), MODEL[1000:]))
        self.assertEqual(_RangeHandler.seen, ["bytes=1000-", "bytes=1000-"])

    def test_a_server_without_ranges_answers_from_the_start(self):
        _RangeHandler.ranges = False
        stream, start, total = bootstrap.open_url(self.base + "/model", 1000)
        with stream:
            self.assertEqual((start, total, stream.read()), (0, len(MODEL), MODEL))

    def test_the_whole_setup_through_the_real_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = Path(tmp) / "resources"
            (resources / "bin").mkdir(parents=True)
            (resources / "bin" / "uv.exe").write_bytes(b"uv")
            (resources / "setup").mkdir()
            (resources / "setup" / "requirements.lock").write_bytes(b"numpy==2.5.4\n")
            paths = bootstrap.Paths(Path(tmp) / "root", resources)
            part = paths.model.with_name(paths.model.name + ".part")
            part.parent.mkdir(parents=True)
            part.write_bytes(MODEL[:5000])
            spec = bootstrap.ModelSpec(self.base + "/model", SPEC.sha256, SPEC.size)
            out = io.StringIO()
            code = bootstrap.Bootstrap(paths, version="0.1.0", model=spec, run=FakeUv(paths), out=out,
                                       free_bytes=lambda path: 10 ** 12).run()
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(paths.model.read_bytes(), MODEL)
            self.assertEqual(_RangeHandler.seen, ["bytes=5000-"])


class LockCase(unittest.TestCase):
    """The committed installer/requirements.lock: what every installed app gets."""

    def setUp(self):
        text = (Path(__file__).resolve().parents[1] / "requirements.lock").read_text(encoding="utf-8")
        self.entries = {}  # name -> (version, number of hashes)
        name = None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pinned = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)", line)
            if pinned:
                name = re.sub(r"[-_.]+", "-", pinned.group(1)).lower()
                self.assertNotIn(name, self.entries, "a package is listed twice")
                self.entries[name] = [pinned.group(2), 0]
            else:
                self.assertRegex(line, r"^--hash=sha256:[0-9a-f]{64}( \\)?$", "only pins and their hashes belong here")
                self.assertIsNotNone(name)
                self.entries[name][1] += 1

    def test_every_package_is_pinned_with_at_least_one_hash(self):
        self.assertGreater(len(self.entries), 30)
        for name, (version, hashes) in self.entries.items():
            self.assertTrue(version, name)
            self.assertGreater(hashes, 0, f"{name} has no hash")

    def test_the_plain_onnxruntime_is_not_in_the_lock_and_directml_is(self):
        self.assertNotIn("onnxruntime", self.entries)
        self.assertIn("onnxruntime-directml", self.entries)
        self.assertIn("faster-whisper", self.entries)

    def test_no_test_packages_are_installed(self):
        for name in ("pytest", "httpx"):
            self.assertNotIn(name, self.entries)


class MainCase(unittest.TestCase):
    def test_the_committed_model_pin_is_complete(self):
        self.assertRegex(bootstrap.MODEL.sha256, r"^[0-9a-f]{64}$")
        self.assertGreater(bootstrap.MODEL.size, 100_000_000)

    def test_the_model_is_downloaded_from_one_fixed_revision(self):
        self.assertRegex(bootstrap.MODEL.url,
                         r"^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/vec-768-layer-12\.onnx$")

    def test_arguments(self):
        args = bootstrap.parse_args(["--root", "C:/r", "--resources", "C:/app", "--app-version", "0.1.0"])
        self.assertEqual((args.root, args.resources, args.app_version), ("C:/r", "C:/app", "0.1.0"))

    def test_the_real_process_runner_returns_the_exit_code_and_output_and_ticks(self):
        ticks = []
        code, text = bootstrap.run_process(
            [sys.executable, "-c", "import sys, time; print('hello'); sys.stderr.write('oops'); time.sleep(0.5); "
                                   "sys.exit(3)"],
            None, tick=lambda: ticks.append(1), interval=0.1)
        self.assertEqual(code, 3)
        self.assertIn("hello", text)
        self.assertIn("oops", text)
        self.assertTrue(ticks)


if __name__ == "__main__":
    unittest.main()
