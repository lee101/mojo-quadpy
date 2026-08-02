"""Tensor-product Gauss rules on ``[-1, 1]^n``."""

from __future__ import annotations

import numpy as np

from ._scheme import CnScheme


def gauss_legendre(dim: int, n: int) -> CnScheme:
    """n points per coordinate on an n-dimensional reference cube."""
    if dim < 1 or n < 1:
        raise ValueError("dim and n must be positive")
    x, w = np.polynomial.legendre.leggauss(n)
    grids = np.meshgrid(*([x] * dim), indexing="ij")
    points = np.stack([g.ravel() for g in grids])
    wg = np.meshgrid(*([w] * dim), indexing="ij")
    return CnScheme("Tensor Gauss-Legendre", 2 * n - 1, points,
                    np.prod(np.stack(wg), axis=0).ravel(), "cube")
