# ck_nvfp4_fused_matmul

Keeps comfy-kitchen's fused NVFP4 matmul enabled on non-NVIDIA devices, so a
checkpoint's `full_precision_matrix_mult: true` does not force the slow path.

## What it changes, and why each piece is needed

`full_precision_matrix_mult` in a checkpoint means "do not use the quantized matmul
for these layers". An exporter writes it whenever **it** had no fused kernel, so on a
machine that does have one the flag records the exporting machine as much as a
quality request.

Three separate things stood between the flag and the fused kernel. Fixing only the
first two was measured at **5%** — the flag is consulted in three places, not one.

| # | Target | What was blocking |
|---|---|---|
| 1 | `comfy.model_management.supports_nvfp4_compute` | Returned `False` off-NVIDIA, so `get_disabled_quant_formats` disabled `nvfp4` outright. |
| 2 | `comfy.ops.MixedPrecisionOp.can_use_quantized_matmul` | Honoured the flag unconditionally. |
| 3 | `comfy.ops.use_quantized_matmul` | Cleared `_full_precision_mm` but not `comfy_force_cast_weights`, which `Clip.__init__`'s `set_model_compute_dtype(torch.float32)` sets on every text encoder — and which zeroes `_use_quantized` before `_full_precision_mm` is ever read. |

Measured on `qwen3vl_32b_minimax_h3_nvfp4_awq` (350 nvfp4 layers, 12.19 GB of packed
weights), gfx1103:

| | ms/token | per-token weight traffic |
|---|---|---|
| flag honoured (upstream) | 1867 | read 12.19 GB, write 24.38 GB, read 24.38 GB back |
| this node | **515** | read 12.19 GB |

## Why a custom node and not a source edit

`hook_breaker_ac10a0.py` in the ComfyUI root names exactly one function it will not
let custom nodes hook — `comfy.model_management.cast_to` — and `main.py` calls
`save_functions()` before `init_extra_nodes` and `restore_functions()` after. A
blocklist of one is a statement that everything else is patchable by design.

All three targets are reached as attribute lookups, so a monkeypatch is live for
every call site:

```
ops.py:1848   comfy.model_management.supports_nvfp4_compute(device)
ops.py:1881   module.can_use_quantized_matmul(disabled)
sd.py:484     comfy.ops.use_quantized_matmul(self, device)
```

None of the three is on the blocklist. Keeping the fork out of ComfyUI's own files
also means a ComfyUI upgrade cannot half-apply it.

## The failure mode this is built against

A monkeypatch whose target gets renamed **does not raise** — it silently stops being
called. The symptom is not a crash; it is a 3.6x slowdown that reads as a hardware
regression. So:

- every patch records whether it actually installed, and one that cannot find what it
  expects is skipped and reported rather than half-applied;
- `verify()` re-checks that each wrapper is still the live object at its call site;
- the node logs which patches are active at import;
- `CK_NVFP4_FUSED_MATMUL_FAIL_LOUD=1` turns a failed install into an exception at
  import, instead of a warning.

## Scoping

Both overrides are scoped to formats with a real fused kernel behind them, so the
clearing cannot outrun what the device can execute. `int8_tensorwise` and the fp8
formats keep `full_precision_matrix_mult` meaning exactly what it says. There is a
test for that, and it is the one that would catch a leak.

## The benchmark does not load this

`benchmark_attn/benchmark_pipeline.py` puts `E:\ComfyUI` on `sys.path` and imports
`comfy.*` directly; it never calls `init_extra_nodes`, so it never imports this file.
It imports this node explicitly right after the `comfy.*` imports and **fails loudly
if the patches are not live** — without that, a missing patch produces 1867 ms/token
with no error anywhere, which looks like a slow GPU rather than a missing module.

`CK_NVFP4_REQUIRE=0` downgrades that failure to a warning, for deliberately measuring
the unpatched number.

## API

```python
from custom_nodes.ck_nvfp4_fused_matmul import verify, status

ok, live = verify()   # {name: installed-and-live}
status()              # {name: installed}
```

`NODE_CLASS_MAPPINGS` is empty and `nodes.py:2305` only reads it when present, so the
node exists purely for its import-time side effect.