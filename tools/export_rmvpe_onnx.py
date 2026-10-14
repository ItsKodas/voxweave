"""Export the RMVPE pitch model to ONNX for live mode.

RMVPE is the neural pitch tracker RVC uses to prepare training data. The trainer already has its weights
(trainer/rvc/assets/rmvpe/rmvpe.pt, SHA-256 checked by trainer/setup.sh). This script turns them into an ONNX graph
the backend can run with onnxruntime-directml, so live mode tracks pitch the way the models learned it.

The graph takes the STFT magnitude of 16 kHz audio (1, 513, frames; frames a multiple of 32) and returns the pitch
salience (1, frames, 360). The mel filterbank, the log and the network are inside the graph; the STFT itself is done
by the backend in numpy (a hann window of 1024, hop 160, reflect padding of 512), exactly as RVC does it.

Two steps, from the repository root. The trainer's environment has librosa (for the mel filterbank) but no onnx;
the exporter's environment (tools/.rvc/venv, see README "Adding a voice") has onnx but no librosa:

    trainer/.venv/Scripts/python tools/export_rmvpe_onnx.py --mel-basis data/pretrained/rmvpe_mel.npy
    tools/.rvc/venv/Scripts/python tools/export_rmvpe_onnx.py data/pretrained/rmvpe.onnx --mel-basis data/pretrained/rmvpe_mel.npy
    backend/.venv/Scripts/python tools/check_rmvpe_onnx.py data/pretrained/rmvpe.onnx

The checkpoint is a plain state dict and is loaded with weights_only=True: nothing in it can run code.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RVC = ROOT / "trainer" / "rvc"
sys.path.insert(0, str(RVC))



def _load_e2e():
    """RVC's RMVPE network class. Its module imports librosa and a CUDA helper at the top for parts not used here,
    so those are stood in for when they are missing."""
    import types

    try:
        import librosa  # noqa: F401
    except ImportError:
        fake = types.ModuleType("librosa")
        fake.filters = types.SimpleNamespace(mel=None)
        fake.util = types.SimpleNamespace(normalize=None, pad_center=None, tiny=None)
        sys.modules.update({"librosa": fake, "librosa.filters": fake.filters, "librosa.util": fake.util})
    from infer.rmvpe import E2E

    return E2E

N_FFT, HOP, N_MELS, FMIN, FMAX, SR, CLAMP = 1024, 160, 128, 30, 8000, 16000, 1e-5


class MagnitudeToSalience(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, basis: np.ndarray):
        super().__init__()
        self.register_buffer("basis", torch.from_numpy(basis).float())
        self.net = net

    def forward(self, magnitude):
        mel = torch.log(torch.clamp(torch.matmul(self.basis, magnitude), min=CLAMP))
        return self.net(mel)


def numpy_magnitude(audio16k: np.ndarray) -> np.ndarray:
    """The same STFT magnitude as the backend computes, for the check below."""
    window = np.hanning(N_FFT + 1)[:-1]  # periodic hann, as scipy get_window("hann", fftbins=True)
    padded = np.pad(audio16k, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = 1 + (len(padded) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(padded[index] * window, axis=1)).T[None].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--mel-basis", type=Path, required=True)
    args = parser.parse_args()
    if args.out is None:  # step 1: the filterbank, with librosa
        from librosa.filters import mel as mel_filters

        basis = mel_filters(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=True).astype(np.float32)
        args.mel_basis.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.mel_basis, basis)
        print(f"written {args.mel_basis} {basis.shape}")
        return
    net = _load_e2e()(4, 1, (2, 2))
    state = torch.load(RVC / "assets" / "rmvpe" / "rmvpe.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    model = MagnitudeToSalience(net, np.load(args.mel_basis)).eval()

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)
    magnitude = numpy_magnitude(audio)[:, :, :192]  # 192 frames: a multiple of 32
    example = torch.from_numpy(magnitude)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (example,), str(args.out), input_names=["magnitude"], output_names=["salience"],
                      dynamic_axes={"magnitude": {2: "frames"}, "salience": {1: "frames"}}, opset_version=17,
                      dynamo=False)

    # What the PyTorch model gives for two lengths, saved so the graph can be checked with the backend's onnxruntime
    # (the trainer's environment has none): backend/.venv/Scripts/python tools/check_rmvpe_onnx.py <out>
    with torch.no_grad():
        checks = {f"m{f}": magnitude[:, :, :f] for f in (192, 96)}
        checks.update({f"s{f}": model(torch.from_numpy(magnitude[:, :, :f])).numpy() for f in (192, 96)})
    np.savez(args.out.with_suffix(".check.npz"), **checks)
    print(f"written {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
