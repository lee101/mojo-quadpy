"""Square rules; a compatibility convenience around :mod:`quadpy.cn`."""

from .cn import gauss_legendre as _gauss_legendre


def gauss_legendre(n: int):
    return _gauss_legendre(2, n)
