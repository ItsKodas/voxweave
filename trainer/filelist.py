"""Writes RVC's filelist.txt and config.json for one experiment. Standard library only.

A faithful port of the filelist/config logic in RVC's webui.py run_train_model() (single speaker, v2 features,
pitch model), as proven by spike/stage3/scripts/make_filelist.py. Text is written in text mode exactly as the
spike script does (so line endings are the platform's).
"""
import json
import os
import random
from pathlib import Path


def _stems(folder):
    try:
        return {name.split(".")[0] for name in os.listdir(folder)}
    except OSError:
        return set()


def write_filelist(
    rvc_root,
    experiment,
    sample_rate="40k",
    speaker_id=0,
    log_interval=None,
    shuffle=random.shuffle,
):
    """Write logs/<experiment>/filelist.txt and config.json under rvc_root; return the number of filelist lines.

    Raises ValueError when no segment is present in all four feature folders.
    """
    now_dir = os.path.abspath(str(rvc_root))
    exp_dir = "%s/logs/%s" % (now_dir, experiment)
    gt, fea, f0, f0nsf = (exp_dir + "/" + d for d in ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf"))
    names = _stems(gt) & _stems(fea) & _stems(f0) & _stems(f0nsf)
    if not names:
        raise ValueError("No clip has all of its audio, pitch and feature files; nothing to train on.")

    def doubled(folder):
        return folder.replace("\\", "\\\\")

    lines = [
        "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
        % (doubled(gt), n, doubled(fea), n, doubled(f0), n, doubled(f0nsf), n, speaker_id)
        for n in sorted(names)
    ]
    for _ in range(2):
        lines.append(
            "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature768/mute.npy|"
            "%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
            % (now_dir, sample_rate, now_dir, now_dir, now_dir, speaker_id)
        )
    shuffle(lines)

    cfg_path = Path(exp_dir) / "config.json"
    # webui quirk: v2 at 40k uses configs/v1/40k.json (there is no v2/40k.json)
    version_dir = "v1" if sample_rate == "40k" else "v2"
    source = Path(now_dir) / "configs" / version_dir / ("%s.json" % sample_rate)
    with open(cfg_path if cfg_path.exists() else source, encoding="utf8") as f:
        cfg = json.load(f)
    cfg.pop("speaker_info", None)
    if log_interval is not None:
        cfg["train"]["log_interval"] = int(log_interval)

    with open(exp_dir + "/filelist.txt", "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
    return len(lines)
