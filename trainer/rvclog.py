"""Parsers for the log lines of RVC's training pipeline. Standard library only; no function here raises.

Line shapes are documented in docs/superpowers/notes/2026-10-13-stage-3-training-spike.md (section 5).
Every pattern is anchored at the start of the message, so a path or experiment name that happens to
contain a keyword (an experiment called "Epoch", a file called "Failed.wav") cannot match.
"""
import re

# train.py prints "LEVEL:<experiment>:<message>" to stdout and "<time>\t<experiment>\tLEVEL\t<message>" to train.log.
_STDOUT_PREFIX = re.compile(r"^[A-Z]+:[^:\t]*:(.*)$", re.DOTALL)
_FILE_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+\t[^\t]*\t[A-Z]+\t(.*)$", re.DOTALL)

_EPOCH_DONE = re.compile(r"^====> Epoch: (\d{1,9})(?!\d)")
_EPOCH_PROGRESS = re.compile(r"^Training epoch: (\d{1,9}) \[(\d{1,3})%\]")
_SAVED = re.compile(r"^Saving model and optimizer state at epoch (\d{1,9}) to (.+)$")
_RESUMED = re.compile(r"^Loaded checkpoint '.*' \(epoch (\d{1,9})\)$")
_FINAL = re.compile(r"^Saving final checkpoint: (.*)$")
_LOSS_START = re.compile(r"^loss_[a-z]+=")

_STAGE_TAGS = {
    "preprocess": "[Data slicing]",
    "pitch": "[F0 extraction]",
    "features": "[HuBERT features]",
    "index": "[Index training]",
}
_PROGRESS = re.compile(r"^(?:Write )?[Pp]rogress: (\d{1,9})/(\d{1,9})(?!\d)")
_COMPLETED = re.compile(r"^(?:Worker )?[Cc]ompleted \| Success: (\d{1,9})\b.*?\bFailed: (\d{1,9})(?!\d)")
_FAILED_FILE = re.compile(r"^\[Failed\] (.+)$")

_FATAL = re.compile(r"Traceback \(most recent call last\)|\b\w*Error\b|out of memory", re.IGNORECASE)
_HARMLESS = re.compile(r"^MIOpen\(HIP\): Error \[EvaluateInvokers\]")


def _message(line: str) -> str:
    """The message part of a train.py line in either form (or the line itself when it has no prefix)."""
    line = line.rstrip("\r\n")
    for prefix in (_FILE_PREFIX, _STDOUT_PREFIX):
        match = prefix.match(line)
        if match:
            return match.group(1)
    return line


def _loss_values(message: str):
    values = {}
    for part in message.split(","):
        name, sep, number = part.strip().partition("=")
        if not sep or not name.startswith("loss_"):
            return None
        try:
            values[name] = float(number)
        except ValueError:
            return None
    return values or None


def parse_train_line(line):
    """Classify one line of train.py output (stdout or train.log). Returns a dict or None."""
    try:
        if not isinstance(line, str):
            return None
        message = _message(line)
        if match := _EPOCH_DONE.match(message):
            return {"kind": "epoch_done", "epoch": int(match.group(1))}
        if match := _EPOCH_PROGRESS.match(message):
            return {"kind": "epoch_progress", "epoch": int(match.group(1)), "percent": int(match.group(2))}
        if match := _SAVED.match(message):
            return {"kind": "saved", "epoch": int(match.group(1)), "path": match.group(2).strip()}
        if match := _RESUMED.match(message):
            return {"kind": "resumed", "epoch": int(match.group(1))}
        if match := _FINAL.match(message):
            return {"kind": "final_saved", "ok": match.group(1).rstrip().endswith("Success")}
        if _LOSS_START.match(message):
            values = _loss_values(message)
            if values:
                return {"kind": "loss", "values": values}
        return None
    except Exception:  # a parser must never take the worker down
        return None


def parse_stage_line(stage, line):
    """Classify one line of a pipeline stage ("preprocess", "pitch", "features", "index"). Returns a dict or None."""
    try:
        tag = _STAGE_TAGS.get(stage)
        if tag is None or not isinstance(line, str):
            return None
        line = line.rstrip("\r\n")
        if not line.startswith(tag):
            return None
        rest = line[len(tag):]
        if stage == "preprocess" and (match := _FAILED_FILE.match(rest)):
            return {"kind": "failed_file", "path": match.group(1)}
        rest = rest.lstrip(" ")
        if match := _PROGRESS.match(rest):
            return {"kind": "progress", "done": int(match.group(1)), "total": int(match.group(2))}
        if match := _COMPLETED.match(rest):
            return {"kind": "completed", "success": int(match.group(1)), "failed": int(match.group(2))}
        return None
    except Exception:
        return None


def is_fatal_noise(line):
    """True when a line looks like a real failure (an Error, a traceback, out of memory).

    Known harmless lines that contain the word Error (the MIOpen EvaluateInvokers message) return False.
    Callers should classify training lines with parse_train_line first: a line such as
    "INFO:Error:Training epoch: 1 [0%]" mentions the word only through the experiment name.
    """
    try:
        if not isinstance(line, str):
            return False
        line = line.rstrip("\r\n")
        if _HARMLESS.match(line):
            return False
        return bool(_FATAL.search(line))
    except Exception:
        return False
