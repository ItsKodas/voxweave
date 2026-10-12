# DirectML benchmark (Task 1 spike)

Date: 2026-10-13. Machine: Windows 11, AMD RX 7900 XTX + integrated AMD Radeon. onnxruntime-directml 1.24.4, Python 3.12.

## IMPORTANT: the model is randomly initialised

No trained voice `.pth` was supplied, so the benchmark network (`data/pretrained/rvc/bench/model.onnx`, voice id `bench`) is a **randomly initialised** standard RVC v2 40 kHz network (`SynthesizerTrnMsNSFsidM`, config `[1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3,7,11], [[1,3,5]]*3, [10,10,2,2], 512, [16,16,4,4], 109, 256, 40000]`, no weights loaded), exported with the same settings as RVC's `infer/modules/onnx/export.py` (opset 13, same inputs, dynamic axes). Inference speed depends only on architecture and shapes, so timings equal those of a trained voice; the audio output is noise. `tools/export_rvc_onnx.py` (the kept tool) was verified only to import (`--help`), not run end to end.

## Result

- **dml_device_id = 0** (the RX 7900 XTX; device 1, the integrated GPU, is 5-20x slower). Written to `data/settings.json`.
- Model sample rate (bench voice): 40000 Hz.
- ContentVec input: `source` `[1, 1, 'sample_length']`; output `feats` shape `(1, frames, 768)` (frames = samples16k // 320: 27 / 54 / 103 for the three windows, before the x2 repeat).
- Net inputs exactly: `phone [1,T,768] f32, phone_lengths [1] i64, pitch [1,T] i64, pitchf [1,T] f32, ds [1] i64, rnd [1,192,T] f32`; output `audio` (`(1,1,T*400)` at 40 kHz).

### Per-preset gate (device 0, fresh sessions per preset; ms)

| preset | vec | net | f0 | total | budget | result |
|---|---|---|---|---|---|---|
| low_latency (dio) | 5.3 | 20.5 | 1.8 | 27.6 | 60 | PASS |
| balanced (dio) | 6.8 | 35.4 | 5.6 | 47.8 | 125 | PASS |
| high_quality (harvest) | 8.3 | 46.2 | 465.7 | 520.2 | 250 | FAIL (with dio: 67.6, would pass) |

high_quality fails only because of `harvest`; Task 13 retunes that preset (e.g. dio, or a lighter pitch extractor).

### Full output (fresh session per preset, the methodology used for the gate)

```
providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
device 0: vec in=[('source', [1, 1, 'sample_length'])]
device 0: net in=[('phone', [1, 'phone_dynamic_axes_1', 768], 'tensor(float)'), ('phone_lengths', [1], 'tensor(int64)'), ('pitch', [1, 'pitch_dynamic_axes_1'], 'tensor(int64)'), ('pitchf', [1, 'pitchf_dynamic_axes_1'], 'tensor(float)'), ('ds', [1], 'tensor(int64)'), ('rnd', [1, 192, 'rnd_dynamic_axes_1'], 'tensor(float)')]
device 0 low_latency: feats=(1, 27, 768) vec=5.3 net=20.5 dio=1.8 harvest=133.5 budget=60 ms
device 0 balanced: feats=(1, 54, 768) vec=6.8 net=35.4 dio=5.6 harvest=235.3 budget=125 ms
device 0 high_quality: feats=(1, 103, 768) vec=8.3 net=46.2 dio=13.1 harvest=465.7 budget=250 ms
device 1: vec in=[('source', [1, 1, 'sample_length'])]
device 1: net in=[('phone', [1, 'phone_dynamic_axes_1', 768], 'tensor(float)'), ('phone_lengths', [1], 'tensor(int64)'), ('pitch', [1, 'pitch_dynamic_axes_1'], 'tensor(int64)'), ('pitchf', [1, 'pitchf_dynamic_axes_1'], 'tensor(float)'), ('ds', [1], 'tensor(int64)'), ('rnd', [1, 192, 'rnd_dynamic_axes_1'], 'tensor(float)')]
device 1 low_latency: feats=(1, 27, 768) vec=48.9 net=385.8 dio=2.2 harvest=135.1 budget=60 ms
device 1 balanced: feats=(1, 54, 768) vec=84.5 net=745.5 dio=4.5 harvest=249.7 budget=125 ms
device 1 high_quality: feats=(1, 103, 768) vec=180.4 net=1382.5 dio=10.3 harvest=486.0 budget=250 ms
```

## Finding: ONNX Runtime DML sessions are shape-sensitive

The brief's bench.py reuses ONE session pair for all three window sizes. Output of that run:

```
providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
device 0: vec in=[('source', [1, 1, 'sample_length'])]
device 0: net in=[('phone', [1, 'phone_dynamic_axes_1', 768], 'tensor(float)'), ('phone_lengths', [1], 'tensor(int64)'), ('pitch', [1, 'pitch_dynamic_axes_1'], 'tensor(int64)'), ('pitchf', [1, 'pitchf_dynamic_axes_1'], 'tensor(float)'), ('ds', [1], 'tensor(int64)'), ('rnd', [1, 192, 'rnd_dynamic_axes_1'], 'tensor(float)')]
device 0 low_latency: feats=(1, 27, 768) vec=6.1 net=18.7 dio=2.9 harvest=130.2 budget=60 ms
device 0 balanced: feats=(1, 54, 768) vec=51.5 net=109.9 dio=5.7 harvest=262.7 budget=125 ms
device 0 high_quality: feats=(1, 103, 768) vec=57.6 net=123.7 dio=14.8 harvest=510.5 budget=250 ms
device 1: vec in=[('source', [1, 1, 'sample_length'])]
device 1: net in=[('phone', [1, 'phone_dynamic_axes_1', 768], 'tensor(float)'), ('phone_lengths', [1], 'tensor(int64)'), ('pitch', [1, 'pitch_dynamic_axes_1'], 'tensor(int64)'), ('pitchf', [1, 'pitchf_dynamic_axes_1'], 'tensor(float)'), ('ds', [1], 'tensor(int64)'), ('rnd', [1, 192, 'rnd_dynamic_axes_1'], 'tensor(float)')]
device 1 low_latency: feats=(1, 27, 768) vec=50.2 net=386.8 dio=2.9 harvest=122.0 budget=60 ms
device 1 balanced: feats=(1, 54, 768) vec=120.3 net=817.6 dio=5.7 harvest=248.3 budget=125 ms
device 1 high_quality: feats=(1, 103, 768) vec=219.8 net=1461.3 dio=12.3 harvest=534.0 budget=250 ms
```

In that run only the first shape the session sees (low_latency) is fast; every later, different shape runs ~8x (vec) / ~5x (net) slower (balanced would have FAILED: 51.5+109.9+5.7 = 167 ms > 125 ms). Verified: the same session re-run at the first shape stays fast (vec ~6 ms, 120 consecutive runs), and a session first used with 1100 ms gives vec 7 ms / net 29 ms, while the same session at 550 ms is then slow (50 / 100 ms). Fresh sessions per preset give the fast numbers above. **Consequence for Task 7+: the engine must run each session at one fixed input shape (pad windows to the preset's constant length; create/recreate sessions on preset change), never feed varying shapes to one session.** The gate was judged on the fixed-shape (fresh session) numbers because that is how the engine will run.

## Export problems found and what was changed (RVC tag 2.2.231006, export venv only)

The exporter at that tag does not produce a usable dynamic-length model out of the box. Changes made in the gitignored clone `tools/.rvc/repo` (also needed for any real voice export; diff below):

1. `models_onnx.py` `ResidualCouplingBlock.forward`: reverse branch did `x = flow(...)` but the flow layers return `(x, logdet)` tuples -> `AttributeError: 'tuple' object has no attribute 'split'` during export. Fixed to `x, _ = flow(...)`.
2. `attentions.py` relative-position attention: `int(length)`, `max()` and `if pad_length > 0` were baked at trace time as the dummy length 200, so the exported graph only worked with exactly 200 frames (Reshape failure at any other length, on CPU and DML alike). Rewrote `_get_relative_embeddings` (always pad by `length`, slice `[w+1, w+2*length)`) and removed the `int()` calls so lengths stay dynamic. Verified the re-exported model runs at T=110/200/220 on CPU and DML.
3. Installed `scipy` into the export venv (missing import).
4. uv quirk: `uv venv --python 3.10` failed with a broken managed-interpreter link; worked with the explicit path `C:/Users/koda/AppData/Roaming/uv/python/cpython-3.10.21-windows-x86_64-none/python.exe`.

```diff
diff --git a/infer/lib/infer_pack/attentions.py b/infer/lib/infer_pack/attentions.py
index 2cc745a..c152b54 100644
--- a/infer/lib/infer_pack/attentions.py
+++ b/infer/lib/infer_pack/attentions.py
@@ -306,23 +306,13 @@ class MultiHeadAttention(nn.Module):
         return ret
 
     def _get_relative_embeddings(self, relative_embeddings, length: int):
-        max_relative_position = 2 * self.window_size + 1
-        # Pad first before slice to avoid using cond ops.
-        pad_length: int = max(length - (self.window_size + 1), 0)
-        slice_start_position = max((self.window_size + 1) - length, 0)
-        slice_end_position = slice_start_position + 2 * length - 1
-        if pad_length > 0:
-            padded_relative_embeddings = F.pad(
-                relative_embeddings,
-                # commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
-                [0, 0, pad_length, pad_length, 0, 0],
-            )
-        else:
-            padded_relative_embeddings = relative_embeddings
-        used_relative_embeddings = padded_relative_embeddings[
-            :, slice_start_position:slice_end_position
+        # Dynamic-length friendly: always pad by `length`, then slice [w+1, w+2*length).
+        padded_relative_embeddings = F.pad(
+            relative_embeddings, [0, 0, length, length, 0, 0]
+        )
+        return padded_relative_embeddings[
+            :, self.window_size + 1 : self.window_size + 2 * length
         ]
-        return used_relative_embeddings
 
     def _relative_position_to_absolute_position(self, x):
         """
@@ -342,7 +332,7 @@ class MultiHeadAttention(nn.Module):
         x_flat = F.pad(
             x_flat,
             # commons.convert_pad_shape([[0, 0], [0, 0], [0, int(length) - 1]])
-            [0, int(length) - 1, 0, 0, 0, 0],
+            [0, length - 1, 0, 0, 0, 0],
         )
 
         # Reshape and slice out the padded elements.
@@ -361,9 +351,9 @@ class MultiHeadAttention(nn.Module):
         x = F.pad(
             x,
             # commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, int(length) - 1]])
-            [0, int(length) - 1, 0, 0, 0, 0, 0, 0],
+            [0, length - 1, 0, 0, 0, 0, 0, 0],
         )
-        x_flat = x.view([batch, heads, int(length**2) + int(length * (length - 1))])
+        x_flat = x.view([batch, heads, length * (2 * length - 1)])
         # add 0's in the beginning that will skew the elements after reshape
         x_flat = F.pad(
             x_flat,
diff --git a/infer/lib/infer_pack/models_onnx.py b/infer/lib/infer_pack/models_onnx.py
index ff60414..97308ef 100644
--- a/infer/lib/infer_pack/models_onnx.py
+++ b/infer/lib/infer_pack/models_onnx.py
@@ -149,7 +149,7 @@ class ResidualCouplingBlock(nn.Module):
                 x, _ = flow(x, x_mask, g=g, reverse=reverse)
         else:
             for flow in reversed(self.flows):
-                x = flow(x, x_mask, g=g, reverse=reverse)
+                x, _ = flow(x, x_mask, g=g, reverse=reverse)
         return x
 
     def remove_weight_norm(self):
```
