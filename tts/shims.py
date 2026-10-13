"""Start-up shims that let f5-tts 1.1.22 import on AMD's ROCm PyTorch for Windows without patching any package.

Call apply() after the worker has taken stdout for itself and before anything from f5_tts or vocos is imported
(spike notes, sections 5.5 and 10.3).
"""
import enum
import sys
import types

# f5_tts.model imports its Trainer, and f5_tts.infer.utils_infer imports transformers.pipeline, even when only
# synthesising. Nothing on the synthesis path uses them; importing them costs about nine seconds of start-up and
# makes wandb create temporary folders.
TRAINING_ONLY_MODULES = ("wandb", "accelerate", "accelerate.utils", "ema_pytorch", "datasets", "transformers")


class _Stub(types.ModuleType):
    """Stands in for a module that is imported but never used: any name looked up in it is an empty class."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)  # the import machinery must not take this for a package or a real module
        return type(name, (), {})


def apply(stub_training_imports: bool = True) -> None:
    """Install the shims. Safe to call more than once.

    1. Required: AMD's Windows wheels are built without torch.distributed, and `encodec` (imported by `vocos`) uses
       torch.distributed.ReduceOp.SUM as a default argument value at import time. Nothing ever calls it.
    2. Optional: stand-ins for the training-only modules above. If a later package version touches one of these
       names at import time, start-up fails loudly instead of synthesis going quietly wrong.
    """
    import torch.distributed as distributed

    if not hasattr(distributed, "ReduceOp"):
        distributed.ReduceOp = enum.Enum("ReduceOp", "SUM PRODUCT MIN MAX BAND BOR BXOR")
    if stub_training_imports:
        for name in TRAINING_ONLY_MODULES:
            if not isinstance(sys.modules.get(name), _Stub):
                sys.modules[name] = _Stub(name)
