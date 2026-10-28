"""The three patches that keep an NVFP4 matmul fused, each self-checking.

Every wrapper records what it replaced and whether it is still installed, so
`verify()` can tell "installed and live" from "installed but the call site moved" --
the distinction that matters, because the second looks exactly like a slow GPU.
"""
import contextlib
import logging
import os

logger = logging.getLogger("ck_igpu_opti")

# name -> True once installed and re-checked as installed.
_STATE = {}


def _nvfp4_has_fused_matmul():
    """Whether this device can keep an NVFP4 matmul fused, answered by comfy-kitchen.

    Delegates rather than repeating an architecture list, so the answer tracks the
    kernel instead of drifting from it. False on a device without the kernel, which
    is what keeps `full_precision_matrix_mult` meaning what it says there.
    """
    try:
        from comfy_kitchen.tensor.base import nvfp4_mm_has_fast_path
    except ImportError:
        return False
    import torch
    return bool(nvfp4_mm_has_fast_path(torch.device("cuda").type))


def _format_has_fused_matmul(quant_format):
    """Whether comfy_kitchen has a fused quantized matmul for `quant_format` here.

    Only nvfp4 today. Deliberately narrow: the flag override below is an accuracy
    change, and it must not outrun what the device can actually execute.

    NVIDIA is excluded outright. Upstream already runs the fused NVFP4 path there
    and honours the checkpoint's flag; this whole override exists because the HIP
    backend keeps the matmul fused off-NVIDIA, which is the only place it was
    measured. Returning False here leaves an NVIDIA start untouched.
    """
    if quant_format != "nvfp4":
        return False
    import comfy.model_management as mm

    if mm.is_nvidia():
        return False
    return _nvfp4_has_fused_matmul()


# ---------------------------------------------------------------------------
# 1. supports_nvfp4_compute
# ---------------------------------------------------------------------------

def _patch_supports_nvfp4_compute():
    import comfy.model_management as mm

    original = mm.supports_nvfp4_compute

    def supports_nvfp4_compute(device=None):
        if mm.is_device_cpu(device):
            return False
        if not mm.is_nvidia():
            # No other vendor has fp4 WMMA, but that is not what this asks: it asks
            # whether an NVFP4 matmul can stay fused here. comfy-kitchen's HIP
            # backend decodes the packed nibbles through a lookup table and folds
            # each e4m3 block scale into an fp16 WMMA tile, so on an RDNA3/3.5/4 part
            # the answer is yes. Upstream's NVIDIA path is left untouched.
            return _nvfp4_has_fused_matmul()

        import torch
        props = torch.cuda.get_device_properties(device)
        if props.major < 10:
            return False
        return True

    mm.supports_nvfp4_compute = supports_nvfp4_compute
    # `is`, not `is not`: after the rebind above the two names are the same object,
    # and recording that as False made status() report a working patch as uninstalled.
    _STATE["supports_nvfp4_compute"] = mm.supports_nvfp4_compute is supports_nvfp4_compute
    return _STATE["supports_nvfp4_compute"]


# ---------------------------------------------------------------------------
# 2. can_use_quantized_matmul
# ---------------------------------------------------------------------------

def _patch_can_use_quantized_matmul():
    import comfy.ops as ops

    original = ops.MixedPrecisionOp.can_use_quantized_matmul

    def can_use_quantized_matmul(self, disabled_formats):
        # Upstream's condition, with one term relaxed: honour the checkpoint's
        # `full_precision_matrix_mult` UNLESS this device has a fused kernel for the
        # format, in which case the flag is describing the exporter's machine rather
        # than a constraint here. Scoped to the formats _format_has_fused_matmul
        # answers True for, so no other format's flag changes meaning.
        return (self.quant_format in ops.QUANT_ALGOS
                and (not self._full_precision_mm_config
                     or _format_has_fused_matmul(self.quant_format))
                and self.quant_format not in self._disabled_formats
                and self.quant_format not in disabled_formats)

    ops.MixedPrecisionOp.can_use_quantized_matmul = can_use_quantized_matmul
    _STATE["can_use_quantized_matmul"] = (
        ops.MixedPrecisionOp.can_use_quantized_matmul is can_use_quantized_matmul
    )
    return _STATE["can_use_quantized_matmul"]


# ---------------------------------------------------------------------------
# 3. use_quantized_matmul
# ---------------------------------------------------------------------------

def _patch_use_quantized_matmul():
    import comfy.ops as ops

    original = ops.use_quantized_matmul

    @contextlib.contextmanager
    def use_quantized_matmul(model, device):
        """Turn on the quantized matmul for every layer this device can actually run.

        `_full_precision_mm` is the switch `forward` reads; clearing it is enough on
        its own, but a text encoder also gets `comfy_force_cast_weights` set (see
        Clip.__init__'s set_model_compute_dtype), and that independently turns the
        quantized forward off before the switch is ever consulted -- which is why a
        weight-only quantized linear runs and a fused one does not. Both are cleared
        here, together, and only for a format with a real fused kernel behind it.

        Restored on the way out; this is scoped to one generate() call.
        """
        disabled = ops.get_disabled_quant_formats(device)
        previous = []
        previous_force = []
        try:
            for module in model.modules():
                if (isinstance(module, ops.MixedPrecisionOp)
                        and module.can_use_quantized_matmul(disabled)):
                    previous.append((module, module._full_precision_mm))
                    module._full_precision_mm = False
                    if (getattr(module, "comfy_force_cast_weights", False)
                            and _format_has_fused_matmul(module.quant_format)):
                        previous_force.append((module, module.comfy_force_cast_weights))
                        module.comfy_force_cast_weights = False
            yield
        finally:
            for module, full_precision_mm in previous:
                module._full_precision_mm = full_precision_mm
            for module, force in previous_force:
                module.comfy_force_cast_weights = force

    ops.use_quantized_matmul = use_quantized_matmul
    # sd.py:484 reaches it as `comfy.ops.use_quantized_matmul`, an attribute lookup at
    # call time, so rebinding the module attribute is enough. Verified there rather
    # than assumed: a `from comfy.ops import use_quantized_matmul` at the call site
    # would have bound the original and this patch would be dead code.
    _STATE["use_quantized_matmul"] = ops.use_quantized_matmul is use_quantized_matmul
    return _STATE["use_quantized_matmul"]


_PATCHES = {
    "supports_nvfp4_compute": _patch_supports_nvfp4_compute,
    "can_use_quantized_matmul": _patch_can_use_quantized_matmul,
    "use_quantized_matmul": _patch_use_quantized_matmul,
}


def apply():
    """Install every patch. Returns {name: installed}.

    A patch that cannot find what it expects is skipped and reported False rather
    than raising, so one API move does not cost the other two.
    """
    _STATE.clear()
    report = {}
    for name, fn in _PATCHES.items():
        try:
            report[name] = bool(fn())
        except Exception as e:  # noqa: BLE001
            logger.warning("ck-igpu-opti: nvfp4 %s not applied: %s", name, e)
            report[name] = False
    return report


def verify():
    """Re-check that each installed patch is still the live object at its call site.

    This is the check that catches the failure a patch cannot raise on: ComfyUI
    renaming or re-importing something, after which the wrapper is installed, looks
    correct, and is simply never called.

    Identity against this module is the test, not a source-text match: `patch.py` has
    two names for the three call sites (`module` vs `__module__` depending on how
    ComfyUI loaded the package), so both are accepted, and anything that is not one
    of our functions is reported as not live.
    """
    import comfy.model_management as mm
    import comfy.ops as ops

    here = {fn.__module__ for fn in (_patch_supports_nvfp4_compute,
                                      _patch_can_use_quantized_matmul,
                                      _patch_use_quantized_matmul)}

    def ours(fn):
        return getattr(fn, "__module__", None) in here

    live = {
        "supports_nvfp4_compute": ours(mm.supports_nvfp4_compute),
        "can_use_quantized_matmul": ours(
            ops.MixedPrecisionOp.can_use_quantized_matmul),
        "use_quantized_matmul": ours(ops.use_quantized_matmul),
    }
    ok = all(live.values())
    return ok, live


def status():
    """{name: installed} as of the last apply(), for a caller that wants to report it."""
    return dict(_STATE)