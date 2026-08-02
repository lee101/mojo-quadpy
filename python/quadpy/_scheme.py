"""Common quadrature scheme objects and affine-domain integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import weighted_sum


def _reduce(values, weights):
    values = np.asarray(values)
    if values.dtype.kind != "f" or values.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("integrand values must have a real floating-point dtype no wider than float64")
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim == 0:
        return values * weights.sum()
    if values.shape[-1] != weights.size:
        raise ValueError("integrand must return values with points on its final axis")
    return weighted_sum(values, weights)


@dataclass(frozen=True)
class QuadratureScheme:
    name: str
    degree: int
    points: np.ndarray
    weights: np.ndarray
    domain: str
    source: str | None = "Standard published quadrature rule"
    test_tolerance: float = 1.0e-14

    def __post_init__(self):
        object.__setattr__(self, "points", np.asarray(self.points, dtype=np.float64))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=np.float64))
        if self.points.ndim != 2 or self.points.shape[1] != self.weights.size:
            raise ValueError("points must have shape (dimension, number of weights)")

    @property
    def dim(self) -> int:
        return self.points.shape[0]

    def integrate(self, f, domain=None):
        """Integrate a vectorized callable on the reference or affine domain.

        ``domain`` is ``(a, b)`` for a line, bounds shaped ``(dim, 2)`` for a
        cube, or simplex vertices shaped ``(dim + 1, dim)``.
        """
        points, scale = self._mapped_points(domain)
        return _reduce(f(points), self.weights) * scale

    def show(self):
        """Return a compact description, matching upstream's inspection helper."""
        return f"{self.name}: degree {self.degree}, {self.weights.size} points"

    def _mapped_points(self, domain):
        if domain is None:
            return self.points, 1.0
        a = np.asarray(domain, dtype=np.float64)
        if self.domain == "line":
            if a.shape != (2,):
                raise ValueError("line domain must be (a, b)")
            return ((a[1] - a[0]) * self.points + a[0] + a[1]) / 2, abs(a[1] - a[0]) / 2
        if self.domain == "cube":
            if a.shape != (self.dim, 2):
                raise ValueError("cube domain must have shape (dim, 2)")
            half = (a[:, 1] - a[:, 0]) / 2
            return half[:, None] * self.points + a.mean(axis=1)[:, None], abs(np.prod(half))
        if a.shape != (self.dim + 1, self.dim):
            raise ValueError("simplex domain must have shape (dim + 1, dim)")
        edges = (a[1:] - a[0]).T
        return a[0, :, None] + edges @ self.points, abs(np.linalg.det(edges))


class C1Scheme(QuadratureScheme):
    pass


class CnScheme(QuadratureScheme):
    pass


class TnScheme(QuadratureScheme):
    pass
