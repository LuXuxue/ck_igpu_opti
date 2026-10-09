"""Enable comfy-kitchen's fused NVFP4 matmul on non-NVIDIA devices.

WHY THIS EXISTS
---------------
`full_precision_matrix_mult: true` in a checkpoint means "do not use the quantized
matmul for these layers". An exporter writes it whenever IT had no fused kernel
available, so on this machine the flag records the exporting machine as much as it
records a quality request. For qwen3vl_32b_minimax_h3_nvfp4_awq -- 350 nvfp4 layers,
12.19 GB of packed weights -- honouring it costs the whole decode step its bandwidth:
the fallback dequantizes every weight to bf16 on every forward, so each token reads
12.19 GB packed, writes 24.38 GB and reads 24.38 GB back, where the fused kernel
reads 12.19 GB and stops.

There were three separate things standing between the flag and the fused kernel, and
fixing only some of them does nothing (measured: fixing the first two was worth 5%):

  1. `model_management.supports_nvfp4_compute` returned False off-NVIDIA, so
     `get_disabled_quant_formats` disabled nvfp4 outright.
  2. `MixedPrecisionOp.can_use_quantized_matmul` honoured the flag unconditionally.
  3. `use_quantized_matmul` cleared `_full_precision_mm` but not
     `comfy_force_cast_weights`, which `Clip.__init__`'s
     `set_model_compute_dtype(torch.float32)` sets on every text encoder and which
     zeroes `_use_quantized` before `_full_precision_mm` is ever read.

WHY A CUSTOM NODE AND NOT A SOURCE EDIT
---------------------------------------
`hook_breaker_ac10a0.py` in the ComfyUI root names exactly one function it refuses to
let custom nodes hook -- `comfy.model_management.cast_to` -- and `main.py` calls
`save_functions()` before `init_extra_nodes` and `restore_functions()` after. A
blocklist of one is a statement that everything else is patchable by design. The
three functions here are not on it, and all three are reached as attribute lookups:

  ops.py:1848   comfy.model_management.supports_nvfp4_compute(device)
  ops.py:1881   module.can_use_quantized_matmul(disabled)
  sd.py:484     comfy.ops.use_quantized_matmul(self, device)

so a monkeypatch is live for every call site. Keeping the fork out of ComfyUI's own
files means a ComfyUI upgrade cannot half-apply it.

THE FAILURE MODE THIS IS BUILT AGAINST
--------------------------------------
A monkeypatch whose target gets renamed does not raise; it silently stops being
called. The symptom is not a crash, it is a 3.8x slowdown that looks like a hardware
regression. So:

  - every patch checks the thing it is replacing still looks like what it expects,
    and refuses to install (loudly) if not;
  - `verify()` re-checks the installed patches and returns a report;
  - `CK_NVFP4_FUSED_MATMUL_FAIL_LOUD` turns a failed install into an exception at
    import, for the caller that would otherwise go on to report a wrong number.

NOT LOADED BY THE BENCHMARK, AND THAT MATTERS
----------------------------------------------
`benchmark_pipeline.py` puts E:\\ComfyUI on sys.path and imports `comfy.ops` /
`comfy.sd` directly. It never calls `init_extra_nodes`, so it never imports this
file. If the source edits are reverted and only this node exists, the benchmark
reports 1971 ms/token with no error anywhere -- which is the whole reason
`verify()` exists and the reason the benchmark harness has to import this explicitly.
See benchmark_attn/README.MD section 9.
"""
import logging
import os

from .patch import apply, status, verify

logger = logging.getLogger("ck_nvfp4_fused_matmul")


def _apply() -> None:
    try:
        report = apply()
    except Exception as e:  # noqa: BLE001
        # Never take ComfyUI down. An NVFP4 text encoder still runs, just slowly.
        logger.error("ck-nvfp4-fused-matmul: patch failed to install: %s", e, exc_info=True)
        if os.environ.get("CK_NVFP4_FUSED_MATMUL_FAIL_LOUD", "") == "1":
            raise
        return
    installed = [name for name, ok in report.items() if ok]
    skipped = [name for name, ok in report.items() if not ok]
    if installed:
        logger.info(
            "ck-nvfp4-fused-matmul: active for %s"
            + ("; NOT applied: %s" % ", ".join(skipped) if skipped else ""),
            ", ".join(installed),
        )
    else:
        logger.warning(
            "ck-nvfp4-fused-matmul: installed nothing -- nvfp4 linear will fall back "
            "to dequantize-and-matmul, which costs ~3.8x on this checkpoint. "
            "Run verify() for the per-patch reason."
        )


_apply()

# NODE_CLASS_MAPPINGS is optional (nodes.py:2305 only reads it if present) -- this
# node exists for its import-time side effect. Declared empty rather than omitted so
# ComfyUI's loader takes the V1 branch and logs the module as loaded.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["apply", "verify", "status", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]