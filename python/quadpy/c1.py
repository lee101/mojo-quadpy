"""One-dimensional rules on the reference interval ``[-1, 1]``."""

from __future__ import annotations

import numpy as np

from ._scheme import C1Scheme


def gauss_legendre(n: int) -> C1Scheme:
    """The n-point Gauss-Legendre rule, exact through degree ``2*n - 1``."""
    if n < 1:
        raise ValueError("n must be positive")
    points, weights = np.polynomial.legendre.leggauss(n)
    return C1Scheme("Gauss-Legendre", 2 * n - 1, points[None, :], weights, "line")


def gauss_lobatto(n: int) -> C1Scheme:
    """The n-point Gauss-Lobatto rule including both endpoints."""
    if n < 2:
        raise ValueError("n must be at least 2")
    p = np.polynomial.legendre.Legendre.basis(n - 1)
    interior = np.real_if_close(p.deriv().roots(), tol=1000) if n > 2 else np.empty(0)
    points = np.r_[-1.0, interior, 1.0]
    weights = 2.0 / (n * (n - 1) * p(points) ** 2)
    return C1Scheme("Gauss-Lobatto", 2 * n - 3, points[None, :], weights, "line")
