"""Symmetric triangle rules on ``{(x, y): x, y >= 0, x + y <= 1}``."""

from __future__ import annotations

import numpy as np

from ._scheme import TnScheme


def get_good_scheme(degree: int) -> TnScheme:
    """Return a compact symmetric rule of at least the requested degree (1--3)."""
    if degree < 0 or degree > 3:
        raise ValueError("supported triangle degrees are 0 through 3")
    if degree <= 1:
        return TnScheme("Centroid", 1, np.array([[1/3], [1/3]]), np.array([1/2]), "simplex")
    if degree == 2:
        return TnScheme("Hammer-Marlowe-Stroud 2", 2,
                         np.array([[1/6, 2/3, 1/6], [1/6, 1/6, 2/3]]),
                         np.full(3, 1/6), "simplex")
    return TnScheme("Hammer-Marlowe-Stroud 3", 3,
                     np.array([[1/3, 1/5, 3/5, 1/5], [1/3, 1/5, 1/5, 3/5]]),
                     np.array([-27/96, 25/96, 25/96, 25/96]), "simplex")
