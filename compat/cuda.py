"""CPU fallbacks for the CUDA-only calls in the profiling code.

``src/pruning_experiments.py`` was written for a Colab GPU runtime and calls
``torch.cuda.synchronize()`` unconditionally around its latency measurement,
and asks the profiler for ``ProfilerActivity.CUDA``. Both raise (or warn
loudly) on a machine without a GPU.

Rather than editing the experiment code, this module makes those two calls
degrade gracefully when no CUDA device is visible. When a GPU *is* present
nothing is patched at all, so GPU runs execute the original code path
unchanged.

Caveat for CPU runs: the ``memory_mb`` metric comes from the profiler's
``device_memory_usage`` counter, which is device memory. Without a GPU it is
always 0. CPU runs therefore reproduce every metric except GPU memory, and
``inference_ms`` is of course not comparable to the GPU numbers in the paper.
"""

from __future__ import annotations

import torch


def install_cuda_fallback() -> bool:
    """Patch CUDA-only entry points when no GPU is available.

    Returns ``True`` if a CUDA device is available (nothing patched).
    """
    if torch.cuda.is_available():
        print(f"[compat] CUDA available: {torch.cuda.get_device_name(0)}")
        return True

    print("[compat] no CUDA device; installing CPU fallbacks "
          "(memory_mb will be 0, timings are CPU timings)")

    def _noop_synchronize(*args, **kwargs) -> None:
        return None

    torch.cuda.synchronize = _noop_synchronize

    real_profile = torch.profiler.profile
    cuda_activity = torch.profiler.ProfilerActivity.CUDA

    def _cpu_only_profile(*args, **kwargs):
        activities = kwargs.get("activities")
        if activities is not None:
            kwargs["activities"] = [a for a in activities if a != cuda_activity]
        return real_profile(*args, **kwargs)

    # Only the public attribute is patched; torch.profiler.profiler.profile is
    # left alone so torch's own isinstance checks keep working.
    torch.profiler.profile = _cpu_only_profile
    return False
