"""Windows process helpers for the training worker. Standard library only.

- process_alive / wait_for_exit: is a PID still running (the handle is kept, so a recycled PID is never mistaken
  for the original process).
- kill_process_tree: `taskkill /PID n /T /F`. The PID a venv launcher gives back is only a shim around the real
  Python, so ending that one process would leave the trainer and its data-loader workers running.
- bind_children_to_this_process: a job object that ends every child when this process ends, however it ends.
- unlink_ignoring_read_only: remove one name of a read-only file without touching the file's attributes.
- make_runner: the real stage runner (start a command, pass on its output lines, end its tree on request).
"""
import ctypes
import os
import queue
import subprocess
import threading
import time

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_kernel32_dll = None


def _kernel32():
    global _kernel32_dll
    if _kernel32_dll is not None:
        return _kernel32_dll
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, dword, void_p = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p
    kernel32.OpenProcess.restype, kernel32.OpenProcess.argtypes = handle, [dword, ctypes.c_int, dword]
    kernel32.WaitForSingleObject.restype, kernel32.WaitForSingleObject.argtypes = dword, [handle, dword]
    kernel32.GetCurrentProcess.restype, kernel32.GetCurrentProcess.argtypes = handle, []
    kernel32.CreateJobObjectW.restype, kernel32.CreateJobObjectW.argtypes = handle, [void_p, ctypes.c_wchar_p]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.SetInformationJobObject.argtypes = [handle, ctypes.c_int, void_p, dword]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [handle, handle]
    kernel32.CloseHandle.restype, kernel32.CloseHandle.argtypes = ctypes.c_int, [handle]
    kernel32.CreateFileW.restype = handle
    kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, dword, dword, void_p, dword, dword, handle]
    kernel32.SetFileInformationByHandle.restype = ctypes.c_int
    kernel32.SetFileInformationByHandle.argtypes = [handle, ctypes.c_int, void_p, dword]
    _kernel32_dll = kernel32
    return kernel32


_process_handles = {}  # pid -> open handle, kept so that a recycled PID is never mistaken for the original process


def process_alive(pid):
    """True while the process with this PID is running."""
    return not wait_for_exit(pid, 0)


def wait_for_exit(pid, seconds):
    """Wait up to `seconds` for the process with this PID to end. True when it is gone."""
    if os.name != "nt":
        deadline = time.monotonic() + seconds
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
    synchronize, wait_timeout, access_denied = 0x00100000, 0x00000102, 5
    kernel32 = _kernel32()
    handle = _process_handles.get(pid)
    if handle is None:
        handle = kernel32.OpenProcess(synchronize, 0, int(pid))
        if not handle:
            return ctypes.get_last_error() != access_denied  # denied: it exists but belongs to someone else
        _process_handles[pid] = handle
    return kernel32.WaitForSingleObject(handle, int(max(0.0, seconds) * 1000)) != wait_timeout


def unlink_ignoring_read_only(path):
    """Windows: remove one name of a file although the file is read-only, without changing the file.

    A staged hard link shares its attributes with the recording it links to, so clearing the read-only attribute
    to delete the link would alter the recording. This asks Windows to drop the name and leave the file alone."""
    delete_access, share_all, open_existing = 0x00010000, 0x7, 3
    backup_semantics, open_reparse_point = 0x02000000, 0x00200000
    disposition_info_ex, delete_posix_ignore_read_only = 21, 0x1 | 0x2 | 0x10
    kernel32 = _kernel32()
    handle = kernel32.CreateFileW(str(path), delete_access, share_all, None, open_existing,
                                  backup_semantics | open_reparse_point, None)
    if not handle or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        flags = ctypes.c_ulong(delete_posix_ignore_read_only)
        if not kernel32.SetFileInformationByHandle(handle, disposition_info_ex, ctypes.byref(flags),
                                                   ctypes.sizeof(flags)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


def kill_process_tree(pid):
    """End a process and everything it started. The PID a venv launcher gives back is only a shim around the real
    Python, so ending that one process would leave the trainer and its data-loader workers running."""
    if os.name != "nt":
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        return
    try:
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass  # the caller still ends the process it started itself


_lifetime_job = None


def bind_children_to_this_process():
    """Windows: put this process in a job object that ends every process in it when its last handle closes.

    The handle is held only by this process, so if the worker dies in any way at all (even killed outright, with
    no chance to clean up) Windows ends every stage it started. Returns True when the job is in place."""
    global _lifetime_job
    if os.name != "nt":
        return False
    if _lifetime_job is not None:
        return True

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", ctypes.c_ulong), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_ulong),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_ulong),
                    ("SchedulingClass", ctypes.c_ulong)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kill_on_job_close, extended_limit_information = 0x2000, 9
    job = None
    try:
        kernel32 = _kernel32()
        job = kernel32.CreateJobObjectW(None, None)
        if job:
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = kill_on_job_close
            if (kernel32.SetInformationJobObject(job, extended_limit_information, ctypes.byref(limits),
                                                 ctypes.sizeof(limits))
                    and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())):
                _lifetime_job = job  # never closed on purpose: it closes when this process ends
                return True
            kernel32.CloseHandle(job)  # a job that is not in force must not be left open
    except (OSError, AttributeError):
        pass
    return False


def make_runner(kill_tree=kill_process_tree, popen=subprocess.Popen, poll_seconds=0.25, drain_seconds=30.0):
    """The real stage runner: start the command, pass each output line to on_line, and end the whole process tree
    as soon as should_stop() says so (it is asked after every line and every poll_seconds while the stage is quiet)."""

    def runner(stage, argv, cwd, env, on_line, should_stop):
        process = popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                        creationflags=NO_WINDOW)
        lines = queue.Queue()

        def pump():
            try:
                for line in process.stdout:
                    lines.put(line)
            except (OSError, ValueError):
                pass
            finally:
                lines.put(None)

        reader = threading.Thread(target=pump, name="stage-output", daemon=True)
        reader.start()
        killed, give_up_at = False, None
        try:
            while True:
                try:
                    line = lines.get(timeout=poll_seconds)
                except queue.Empty:
                    line = ""
                if line is None:
                    break  # the output pipe closed: every process that held it has ended
                if line:
                    on_line(line.rstrip("\r\n"))
                if not killed and should_stop():
                    kill_tree(process.pid)
                    killed = True
                if give_up_at is None and (killed or (not line and process.poll() is not None)):
                    give_up_at = time.monotonic() + drain_seconds  # ended or killed, yet the pipe is still open
                elif give_up_at is not None and time.monotonic() > give_up_at:
                    break  # something still holds the pipe; do not wait for it for ever
            if not killed:
                return process.wait()
            if process.poll() is None:
                process.kill()
            try:
                return process.wait(timeout=drain_seconds)
            except subprocess.TimeoutExpired:
                return 1
        finally:
            if process.poll() is None:  # only reached when something above raised
                kill_tree(process.pid)
                try:
                    process.kill()
                    process.wait(timeout=drain_seconds)
                except (OSError, subprocess.SubprocessError):
                    pass
            reader.join(timeout=1.0)
            if not reader.is_alive():  # closing under a reader that is still blocked could hang
                try:
                    process.stdout.close()
                except (OSError, AttributeError):
                    pass

    return runner
