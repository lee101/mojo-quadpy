"""Build and load the Mojo weighted-reduction kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_QUADPY_LIB") or os.path.join(ROOT, "dist", "libmojo-quadpy.so")
I = ctypes.c_int64


def _float64_buffer(array: np.ndarray, name: str) -> np.ndarray:
    """Return a contiguous float64 buffer without silently discarding data."""
    array = np.asarray(array)
    if array.dtype.kind != "f" or array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError(f"{name} must have a real floating-point dtype no wider than float64")
    return np.ascontiguousarray(array, dtype=np.float64)


def _build() -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise RuntimeError("mojo not found; run through pixi or set MOJO_QUADPY_LIB")
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], text=True,
                          capture_output=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_loaded: ctypes.CDLL | None = None


def weighted_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reduce rows of ``values[..., npoints]`` with one shared weight vector."""
    global _loaded
    vals = _float64_buffer(values, "values")
    w = _float64_buffer(weights, "weights")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if vals.ndim == 0 or vals.shape[-1] != w.size:
        raise ValueError("values' final axis must have one entry per quadrature point")
    if _loaded is None:
        _loaded = ctypes.CDLL(_build())
        _loaded.mq_weighted_sum.argtypes = [I, I, I, I, I]
        _loaded.mq_weighted_sum.restype = None
    result = np.empty(vals.shape[:-1] or (), dtype=np.float64)
    rows = vals.size // w.size
    # Avoid constructing C pointers from zero-length NumPy sentinel buffers.
    if rows == 0:
        return result
    flat_result = result.reshape(-1)
    _loaded.mq_weighted_sum(vals.ctypes.data, w.ctypes.data, flat_result.ctypes.data,
                             w.size, rows)
    return result
