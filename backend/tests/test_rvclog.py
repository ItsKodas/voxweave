"""Parsers for RVC's training and stage log lines (trainer/rvclog.py). Sample lines are copied from real runs."""
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "trainer" / "rvclog.py"


@pytest.fixture(scope="module")
def rvclog():
    spec = importlib.util.spec_from_file_location("rvclog", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def both_forms(message, exp="spike1"):
    """The same message as train.py prints it to stdout, as it appears in train.log, and bare."""
    return [
        f"INFO:{exp}:{message}",
        f"2026-10-13 06:42:42,025\t{exp}\tINFO\t{message}",
        message,
    ]


@pytest.mark.parametrize("line", both_forms("====> Epoch: 1 [2026-10-13 06:42:58] | (0:01:29.686441)"))
def test_epoch_done(rvclog, line):
    assert rvclog.parse_train_line(line) == {"kind": "epoch_done", "epoch": 1}


@pytest.mark.parametrize("line", both_forms("Training epoch: 12 [29%]"))
def test_epoch_progress(rvclog, line):
    assert rvclog.parse_train_line(line) == {"kind": "epoch_progress", "epoch": 12, "percent": 29}


@pytest.mark.parametrize("line", both_forms("Saving model and optimizer state at epoch 1 to ./logs\\spike1\\G_17.pth"))
def test_saved(rvclog, line):
    assert rvclog.parse_train_line(line) == {"kind": "saved", "epoch": 1, "path": "./logs\\spike1\\G_17.pth"}


@pytest.mark.parametrize(
    "line", both_forms("loss_disc=5.083, loss_gen=3.019, loss_fm=12.336,loss_mel=30.665, loss_kl=9.000")
)
def test_loss(rvclog, line):
    assert rvclog.parse_train_line(line) == {
        "kind": "loss",
        "values": {"loss_disc": 5.083, "loss_gen": 3.019, "loss_fm": 12.336, "loss_mel": 30.665, "loss_kl": 9.0},
    }


@pytest.mark.parametrize("line", both_forms("Loaded checkpoint './logs\\spike1\\G_2333333.pth' (epoch 2)"))
def test_resumed(rvclog, line):
    assert rvclog.parse_train_line(line) == {"kind": "resumed", "epoch": 2}


@pytest.mark.parametrize("line", both_forms("Saving final checkpoint: Success"))
def test_final_saved_ok(rvclog, line):
    assert rvclog.parse_train_line(line) == {"kind": "final_saved", "ok": True}


def test_final_saved_failure_is_not_ok(rvclog):
    line = "INFO:spike1:Saving final checkpoint: Traceback (most recent call last): FileNotFoundError: assets/weights"
    assert rvclog.parse_train_line(line) == {"kind": "final_saved", "ok": False}


def test_trailing_newline_is_ignored(rvclog):
    line = "INFO:spike1:====> Epoch: 3 [2026-10-13 06:42:58] | (0:01:29.6)\r\n"
    assert rvclog.parse_train_line(line) == {"kind": "epoch_done", "epoch": 3}


@pytest.mark.parametrize(
    "line",
    [
        "INFO:spike1:Training precision selected by device rules: torch.float16",
        "INFO:spike1:Loaded generator pretrained model: assets/pretrained_v2/f0G40k.pth",
        "INFO:spike1:<All keys matched successfully>",
        "INFO:spike1:Loaded discriminator pretrained model: assets/pretrained_v2/f0D40k.pth",
        "INFO:spike1:[5, 0.0001]",
        "INFO:spike1:Saving checkpoint spike1_e1: Success",
        "INFO:spike1:Training completed; saving the final model",
        "2026-10-13 06:42:42,027\tspike1\tINFO\t[5, 0.0001]",
        "[Data slicing] Progress: 3/31 | 0002.orig.wav",
        "",
    ],
)
def test_other_train_lines_are_none(rvclog, line):
    assert rvclog.parse_train_line(line) is None


@pytest.mark.parametrize("exp", ["Epoch", "Failed", "Saving", "epoch_done", "Epoch 9 Success", "loss_disc"])
def test_experiment_name_with_keyword_does_not_fool_the_parser(rvclog, exp):
    assert rvclog.parse_train_line(f"INFO:{exp}:[5, 0.0001]") is None
    assert rvclog.parse_train_line(f"2026-10-13 06:42:42,027\t{exp}\tINFO\tTraining precision selected") is None
    assert rvclog.parse_train_line(f"INFO:{exp}:====> Epoch: 4 [2026-10-13 06:42:58] | (0:01:29)") == {
        "kind": "epoch_done",
        "epoch": 4,
    }


def test_path_containing_keyword_does_not_fool_the_parser(rvclog):
    assert rvclog.parse_train_line("INFO:x:Loaded generator pretrained model: C:/Epoch: 5/(epoch 7)") is None
    assert rvclog.parse_train_line("INFO:x:Loaded checkpoint 'a' (epoch 7)") == {"kind": "resumed", "epoch": 7}


# --- stages -----------------------------------------------------------------


def test_preprocess_lines(rvclog):
    stage = rvclog.parse_stage_line
    assert stage("preprocess", "[Data slicing] Progress: 3/31 | 0002.orig.wav") == {
        "kind": "progress",
        "done": 3,
        "total": 31,
    }
    assert stage("preprocess", "[Data slicing] Worker completed | Success: 2 | Failed: 1\r\n") == {
        "kind": "completed",
        "success": 2,
        "failed": 1,
    }
    assert stage("preprocess", "[Data slicing][Failed] D:/Projects/AI Voice/spike/stage3/dataset/appstyle/0001.txt") == {
        "kind": "failed_file",
        "path": "D:/Projects/AI Voice/spike/stage3/dataset/appstyle/0001.txt",
    }
    for line in (
        "[Data slicing] Started",
        "[Data slicing] Pending: 31 | Processes: 8",
        "[Data slicing] Completed",
        "RuntimeError: Failed to load audio: ffmpeg error (see stderr output for detail)",
    ):
        assert stage("preprocess", line) is None


def test_pitch_lines(rvclog):
    stage = rvclog.parse_stage_line
    assert stage("pitch", "[F0 extraction] Progress: 5/58 | Success: 5 | Skipped: 0 | 11_0.wav") == {
        "kind": "progress",
        "done": 5,
        "total": 58,
    }
    assert stage("pitch", "[F0 extraction] Completed | Success: 58 | Skipped: 0 | Failed: 0") == {
        "kind": "completed",
        "success": 58,
        "failed": 0,
    }
    assert stage("pitch", "[F0 extraction] Pending: 58") is None
    assert stage("pitch", "Loading RMVPE model") is None


def test_feature_lines(rvclog):
    stage = rvclog.parse_stage_line
    assert stage("features", "[HuBERT features] Progress: 10/58 | Success: 10 | Failed: 0 | 12_3.wav | (44, 768)") == {
        "kind": "progress",
        "done": 10,
        "total": 58,
    }
    assert stage("features", "[HuBERT features] Completed | Success: 56 | Skipped: 0 | Failed: 2") == {
        "kind": "completed",
        "success": 56,
        "failed": 2,
    }
    assert stage("features", "[HuBERT features] Device: cuda:0 | Pending: 58 | Skipped: 0") is None


def test_index_lines(rvclog):
    stage = rvclog.parse_stage_line
    assert stage("index", "[Index training] Write progress: 1/2") == {"kind": "progress", "done": 1, "total": 2}
    assert stage("index", "[Index training] Feature shape: (8788, 768) | IVF count: 225") is None
    assert stage("index", "[Index training] Index built successfully: added_IVF225_Flat_nprobe_1_spike1_v2.index") is None


def test_stage_tag_must_match_the_stage(rvclog):
    assert rvclog.parse_stage_line("pitch", "[HuBERT features] Progress: 10/58 | Success: 10") is None
    assert rvclog.parse_stage_line("nonsense", "[F0 extraction] Progress: 5/58") is None


def test_failed_file_path_with_keywords_is_taken_whole(rvclog):
    line = "[Data slicing][Failed] C:/Progress: 1/2/Completed | Success: 9 | Failed: 9.wav"
    assert rvclog.parse_stage_line("preprocess", line) == {
        "kind": "failed_file",
        "path": "C:/Progress: 1/2/Completed | Success: 9 | Failed: 9.wav",
    }


def test_file_name_in_progress_line_does_not_fool_the_parser(rvclog):
    line = "[F0 extraction] Progress: 5/58 | Success: 5 | Skipped: 0 | Completed | Success: 1 | Failed: 7.wav"
    assert rvclog.parse_stage_line("pitch", line) == {"kind": "progress", "done": 5, "total": 58}


def test_stage_lines_are_not_train_lines_and_vice_versa(rvclog):
    assert rvclog.parse_train_line("[F0 extraction] Completed | Success: 58 | Skipped: 0 | Failed: 0") is None
    assert rvclog.parse_stage_line("pitch", "INFO:Epoch:====> Epoch: 1 [x] | (y)") is None


# --- noise and garbage ---------------------------------------------------------

MIOPEN = (
    "MIOpen(HIP): Error [EvaluateInvokers] Koda:C:/develop/TheRock/rocm-libraries/projects/miopen/src/conv/"
    "solver_finders.cpp:303: Invalid elapsed time detected in EvaluateInvokers, failed condition: elapsed <= 0"
)


def test_miopen_evaluate_invokers_line_is_not_fatal(rvclog):
    assert rvclog.is_fatal_noise(MIOPEN) is False
    assert rvclog.is_fatal_noise(MIOPEN + "\r\n") is False


def test_real_errors_are_still_fatal(rvclog):
    assert rvclog.is_fatal_noise("RuntimeError: HIP out of memory") is True
    assert rvclog.is_fatal_noise("Traceback (most recent call last):") is True
    assert rvclog.is_fatal_noise("MIOpen(HIP): Error [Something] else") is True
    assert rvclog.is_fatal_noise("INFO:spike1:Training epoch: 1 [0%]") is False
    assert rvclog.is_fatal_noise("MIOpen(HIP): Warning [IsEnoughWorkspace] [EvaluateInvokers] Solver <GemmBwdRest>") is False


GARBAGE = [
    "", " ", "\x00\x01\x02\xff\xfe", "\udcff\udcfe", "\t\t\t", ":::", "INFO::", "x" * 10000, "Epoch: " * 2000,
    "====> Epoch: " + "9" * 5000, "Training epoch: 1 [" + "9" * 5000 + "%]", "loss_disc=" + "9" * 3000,
    "Loaded checkpoint ' (epoch ", "[Data slicing] Progress: /", "\n\n",
]


@pytest.mark.parametrize("line", GARBAGE)
def test_garbage_never_raises(rvclog, line):
    rvclog.parse_train_line(line)
    for stage in ("preprocess", "pitch", "features", "index", "other"):
        rvclog.parse_stage_line(stage, line)
    assert isinstance(rvclog.is_fatal_noise(line), bool)


def test_non_string_input_never_raises(rvclog):
    assert rvclog.parse_train_line(None) is None
    assert rvclog.parse_stage_line("pitch", None) is None
    assert rvclog.is_fatal_noise(None) is False
