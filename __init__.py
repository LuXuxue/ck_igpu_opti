"""comfy-kitchen's iGPU optimisations for ComfyUI, installed by importing this.

Two independent patches, both reached as attribute lookups at their call sites,
both inert on a workload they do not apply to:

  nvfp4_fused_matmul  keeps the fused NVFP4 matmul on a device that has the
                      kernel, so a checkpoint's ``full_precision_matrix_mult``
                      does not force dequantise-every-forward. Lifted from
                      ``ck_nvfp4_fused_matmul`` unchanged.
  vae_conv            routes the VAE decoder's large 3x3 convolutions to the
                      kitchen's fp16 conv3d. Lifted from ``ck_vae_conv``
                      unchanged.

Installing the node is the opt-in. ``CK_NVFP4_FUSED_MATMUL=0`` and
``COMFY_KITCHEN_VAE_CONV=0`` each turn one half off, for measuring what the
other one is worth on a real ComfyUI start.

``NODE_CLASS_MAPPINGS`` is empty on purpose: this package exists for its
import-time side effect, and ``nodes.py`` only reads the mapping if it is
present. It has to be present -- a package without it (or ``comfy_entrypoint``)
is skipped with "IMPORT FAILED" even though the import succeeded, which is what
the old ``ck_vae_conv`` package did.
"""
import logging
import os

from . import nvfp4_fused_matmul  # noqa: F401
from .nvfp4_fused_matmul import apply, status, verify  # noqa: F401

logger = logging.getLogger("ck_igpu_opti")

try:
    from . import vae_conv  # noqa: F401
    from .vae_conv import patch_vae_decode
except Exception:  # noqa: BLE001 - a broken VAE route must not disable the matmul one
    logger.warning("ck-igpu-opti: the VAE conv route is unavailable", exc_info=True)
    vae_conv = None

    def patch_vae_decode(*args, **kwargs):
        raise RuntimeError("the VAE conv route is unavailable; see the import log")


def _apply_nvfp4() -> None:
    """Install the NVFP4 overrides, or say why they could not be installed."""
    if os.environ.get("CK_NVFP4_FUSED_MATMUL", "1") == "0":
        logger.info("ck-igpu-opti: the NVFP4 matmul override is off "
                    "(CK_NVFP4_FUSED_MATMUL=0)")
        return
    try:
        report = apply()
    except Exception as e:  # noqa: BLE001
        # Never take ComfyUI down. An NVFP4 text encoder still runs, just slowly.
        logger.error("ck-igpu-opti: nvfp4 patch failed to install: %s", e, exc_info=True)
        if os.environ.get("CK_NVFP4_FUSED_MATMUL_FAIL_LOUD", "") == "1":
            raise
        return
    installed = [name for name, ok in report.items() if ok]
    skipped = [name for name, ok in report.items() if not ok]
    if installed:
        logger.info(
            "ck-igpu-opti: NVFP4 matmul override active for %s"
            + ("; NOT applied: %s" % ", ".join(skipped) if skipped else ""),
            ", ".join(installed),
        )
    else:
        logger.warning(
            "ck-igpu-opti: the NVFP4 matmul override installed nothing -- nvfp4 linear "
            "will fall back to dequantize-and-matmul, which costs ~3.8x on that "
            "checkpoint. Run verify() for the per-patch reason."
        )


_apply_nvfp4()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "apply",
    "verify",
    "status",
    "patch_vae_decode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
