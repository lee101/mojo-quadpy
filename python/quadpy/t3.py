"""Symmetric tetrahedron rules on the unit reference tetrahedron."""

from __future__ import annotations

import numpy as np

from ._scheme import TnScheme


def get_good_scheme(degree: int) -> TnScheme:
    """Return a symmetric rule of at least the requested degree (1--2)."""
    if degree < 0 or degree > 2:
        raise ValueError("supported tetrahedron degrees are 0 through 2")
    if degree <= 1:
        return TnScheme("Centroid", 1, np.full((3, 1), 1/4), np.array([1/6]), "simplex")
    a, b = 0.5854101966249685, 0.1381966011250105
    bary = np.array([[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]])
    return TnScheme("Keast 2", 2, bary[:, 1:].T, np.full(4, 1/24), "simplex")
