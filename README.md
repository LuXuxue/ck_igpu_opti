# ck_igpu_opti

Two comfy-kitchen patches for an iGPU, installed by importing the package:
keeps the fused NVFP4 matmul enabled on a checkpoint that asks for the slow
path, and routes the VAE decoder's convolutions to the kitchen's fp16 conv3d.
Formerly `ck_nvfp4_fused_matmul` and `ck_vae_conv`.

| Part | Module | What it does | Switch (default) |
|---|---|---|---|
| NVFP4 matmul | `nvfp4_fused_matmul.py` | Ignores `full_precision_matrix_mult` on formats with a real fused kernel | `CK_NVFP4_FUSED_MATMUL` (1) |
| VAE conv | `vae_conv.py` | Decoder's large 3x3 convs on `comfy_kitchen.fp16_conv3d` | `COMFY_KITCHEN_VAE_CONV` (1) |

Both are inert where they do not apply: the matmul part changes nothing on a
device without the fused kernel, and the VAE part changes nothing for a VAE
that is not in fp16 (or bf16 with the extra switch below).

Scope by vendor: the matmul half answers `False` for `is_nvidia()` and keeps
the checkpoint's flag meaning what it says there (upstream already runs the
fused path on NVIDIA, and this override was only measured on HIP). The VAE half
needs the HIP extension's `ndhwc_to_nchw` and its `fp16_conv3d(zero_border=)`;
on a build without them `patch_vae_decode()` installs nothing.

## NVFP4 fused matmul

`full_precision_matrix_mult` in a checkpoint means "do not use the quantized
matmul for these layers". An exporter writes it whenever **it** had no fused
kernel, so on a machine that does have one the flag records the exporting
machine as much as a quality request.

Three separate things stood between the flag and the fused kernel. Fixing only
the first two was measured at **5%** — the flag is consulted in three places,
not one.

| # | Target | What was blocking |
|---|---|---|
| 1 | `comfy.model_management.supports_nvfp4_compute` | Returned `False` off-NVIDIA, so `get_disabled_quant_formats` disabled `nvfp4` outright. |
| 2 | `comfy.ops.MixedPrecisionOp.can_use_quantized_matmul` | Honoured the flag unconditionally. |
| 3 | `comfy.ops.use_quantized_matmul` | Cleared `_full_precision_mm` but not `comfy_force_cast_weights`, which `Clip.__init__`'s `set_model_compute_dtype(torch.float32)` sets on every text encoder — and which zeroes `_use_quantized` before `_full_precision_mm` is ever read. |

Measured on `qwen3vl_32b_minimax_h3_nvfp4_awq` (350 nvfp4 layers, 12.19 GB of
packed weights), gfx1103:

| | ms/token | per-token weight traffic |
|---|---|---|
| flag honoured (upstream) | 1867 | read 12.19 GB, write 24.38 GB, read 24.38 GB back |
| this patch | **515** | read 12.19 GB |

Both overrides are scoped to formats with a real fused kernel behind them, so
the clearing cannot outrun what the device can execute. `int8_tensorwise` and
the fp8 formats keep `full_precision_matrix_mult` meaning exactly what it says.

## VAE conv

The decode is ~15% of an SDXL run at 1024x1024 and ~85% of that is convolution,
on torch's conv2d — which is MIOpen here. `comfy_kitchen.fp16_conv3d` is a 3-D
fp16 NDHWC conv (a 2-D conv is the same thing with a depth of one) that reads
the spatial border as zeros, so the route pays one NCHW into NDHWC copy, the
kernel, and one transpose back. Those copies are fixed work per element while
the kernel's advantage grows with the shape's FLOPs, so only large shapes win.
Measured on the 6-WGP 780M, fp16, per call (2026-10):

| C | K | spatial | MIOpen NCHW | route | (MIOpen / route) |
|---|---|---|---|---|---|
| 512 | 512 | 512^2 | 372.7 ms | 244.4 | 1.52x |
| 256 | 256 | 1024^2 | 255.9 ms | 214.6 | 1.19x |
| 512 | 256 | 512^2 | 155.4 ms | 124.3 | 1.25x |
| 256 | 128 | 1024^2 | 171.8 ms | 117.9 | 1.46x |
| 256 | 256 | 768^2 | 140.0 ms | 120.8 | 1.16x |
| 512 | 512 | 384^2 | 100.9 ms | 126.2 | 0.80x (below the gate) |
| 512 | 512 | 256^2 | 26.8 ms | 57.4 | 0.47x (below the gate) |
| 128 | 128 | 1024^2 | 33.5 ms | 66.1 | 0.51x (below the gate) |

The wrapper therefore checks FLOPs per call: route past 1.2e12, and in the
6e11-1.2e12 band only for outputs of 256 channels or fewer. At 1024x1024 the
512-channel and 1024-spatial stages route; below 768x768 almost nothing does.
The last convolution (128 channels to 3) and the mid block's 1x1s stay on
MIOpen.

The diffusion model's convolutions are not in scope: on their shapes MIOpen
runs at 10-12 TFLOPS while the kernel reaches 6.4-8.8, so the route measured
0.49-0.87x even before its two copies.

Same-workflow A/B on the 6-WGP 780M, SDXL, VAE in fp16, 8 steps, two runs a
side: at 1024x1024 the decode goes 2515 -> 2217 ms (its convolutions 2095 ->
1821), so 1.13x; at 768x768 it lands inside the pipeline's run-to-run band.
A paired in-process ABBA (one VAE, the route toggled through the gate, medians
of three, 2026-10-27) is the cleaner number and gives 2962 -> 2113 ms, 1.40x,
`max|diff| = 0.01`. bf16 VAEs cast through the two copies and eat the margin
(1.05x against 1.41x), so they stay on MIOpen unless
`COMFY_KITCHEN_VAE_CONV_BF16=1`.

## Why a custom node and not a source edit

`hook_breaker_ac10a0.py` in the ComfyUI root names exactly one function it will
not let custom nodes hook — `comfy.model_management.cast_to` — and `main.py`
calls `save_functions()` before `init_extra_nodes` and `restore_functions()`
after. A blocklist of one is a statement that everything else is patchable by
design. Every target here is reached as an attribute lookup at call time, so a
monkeypatch is live for every call site, and keeping the fork out of ComfyUI's
own files means an upgrade cannot half-apply it.

## The failure mode this is built against

A monkeypatch whose target gets renamed **does not raise** — it silently stops
being called. The symptom is not a crash; it is a 3.6x slowdown that reads as a
hardware regression. So:

- every NVFP4 patch records whether it actually installed, and one that cannot
  find what it expects is skipped and reported rather than half-applied;
- `verify()` re-checks that each wrapper is still the live object at its call site;
- the node logs which patches are active at import;
- `CK_NVFP4_FUSED_MATMUL_FAIL_LOUD=1` turns a failed install into an exception
  at import instead of a warning.

## The benchmark does not load this

`benchmark_pipeline.py` puts `E:\ComfyUI` on `sys.path` and imports `comfy.*`
directly; it never calls `init_extra_nodes`, so it never imports this package.
It imports `ck_igpu_opti` explicitly right after the `comfy.*` imports and
**fails loudly if the patches are not live** — without that, a missing patch
produces 1867 ms/token with no error anywhere. `CK_NVFP4_REQUIRE=0` downgrades
that failure to a warning, for deliberately measuring the unpatched number.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `CK_NVFP4_FUSED_MATMUL` | 1 | 0 leaves the matmul override off |
| `CK_NVFP4_FUSED_MATMUL_FAIL_LOUD` | 0 | 1 raises at import instead of warning |
| `CK_NVFP4_REQUIRE` | 1 | benchmark harness: 0 tolerates a missing patch |
| `COMFY_KITCHEN_VAE_CONV` | 1 | 0 leaves the VAE route off |
| `COMFY_KITCHEN_VAE_CONV_BF16` | 0 | 1 also routes bf16 VAEs |
| `COMFY_KITCHEN_VAE_CONV_VERBOSE` | 0 | 1 prints each conv's routing decision |

## API

```python
from ck_igpu_opti import verify, status, patch_vae_decode

ok, live = verify()   # {name: installed-and-live}
status()              # {name: installed}
```

`NODE_CLASS_MAPPINGS` is empty: there are no graph nodes here. It still has to
be defined — `nodes.py` falls through to "IMPORT FAILED" for a package without
it or without `comfy_entrypoint`, even when the import itself succeeded.
