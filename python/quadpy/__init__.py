"""Fast, small numerical quadrature rules backed by a Mojo reduction kernel."""

from . import c1, c2, c3, cn, t2, t3
from ._scheme import C1Scheme, CnScheme, QuadratureScheme, TnScheme

__all__ = ["c1", "c2", "c3", "cn", "t2", "t3", "C1Scheme", "CnScheme", "TnScheme", "QuadratureScheme"]
