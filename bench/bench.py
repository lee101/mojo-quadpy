"""Benchmark the Mojo reduction against NumPy's weighted reduction.

Run only as ``pixi run bench``; that task serializes benchmark jobs.
"""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from quadpy._lib import weighted_sum


def timeit(fn, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    rng = np.random.default_rng(0)
    print(f"machine: {platform.platform()} | {platform.processor() or 'unknown CPU'}")
    print(f"{'case':<34}{'Mojo':>12}{'NumPy':>12}{'ratio':>10}")
    print("-" * 68)
    for rows, n in [(1, 1_000_000), (64, 65_536), (512, 16_384)]:
        values = np.ascontiguousarray(rng.normal(size=(rows, n)))
        weights = np.ascontiguousarray(rng.normal(size=n))
        weighted_sum(values, weights)
        mojo = timeit(lambda: weighted_sum(values, weights))
        numpy = timeit(lambda: np.einsum("...n,n->...", values, weights))
        print(f"weighted reduction ({rows} x {n:,}){mojo * 1e3:>10.2f} ms{numpy * 1e3:>10.2f} ms{numpy / mojo:>8.2f}x")


if __name__ == "__main__":
    main()
