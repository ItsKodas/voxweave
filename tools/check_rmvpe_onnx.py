"""Checks an exported RMVPE graph against the outputs the export saved next to it (see export_rmvpe_onnx.py)."""
import sys
from pathlib import Path

import numpy as np
import onnxruntime

graph = Path(sys.argv[1])
saved = np.load(graph.with_suffix(".check.npz"))
session = onnxruntime.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
for frames in (192, 96):
    got = session.run(None, {"magnitude": saved[f"m{frames}"]})[0]
    error = float(np.max(np.abs(got - saved[f"s{frames}"])))
    print(f"frames {frames}: max difference {error:.2e}")
    if error > 1e-3:
        raise SystemExit("The exported graph does not match the model")
print("the graph matches the model")
