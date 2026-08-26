"""Polynomial exactness and independent NumPy reference checks for covered rules."""

import math

import numpy as np
import pytest

import quadpy
from quadpy._lib import weighted_sum


def line_monomial(k, a=-1.0, b=1.0):
    return (b ** (k + 1) - a ** (k + 1)) / (k + 1)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_gauss_legendre_exact_to_advertised_degree(n):
    scheme = quadpy.c1.gauss_legendre(n)
    for k in range(scheme.degree + 1):
        assert scheme.integrate(lambda x, k=k: x[0] ** k) == pytest.approx(line_monomial(k), abs=2e-14)


@pytest.mark.parametrize("n", [2, 3, 6])
def test_gauss_lobatto_exact_to_advertised_degree(n):
    scheme = quadpy.c1.gauss_lobatto(n)
    assert scheme.points[0, 0] == -1 and scheme.points[0, -1] == 1
    for k in range(scheme.degree + 1):
        assert scheme.integrate(lambda x, k=k: x[0] ** k) == pytest.approx(line_monomial(k), abs=3e-13)


def test_line_affine_mapping_and_batched_values():
    scheme = quadpy.c1.gauss_legendre(16)
    got = scheme.integrate(lambda x: np.stack([x[0] ** 2, np.exp(x[0])]), (2.0, 5.0))
    assert got == pytest.approx([39.0, math.e**5 - math.e**2], abs=2e-12)


def test_tensor_cube_matches_separable_numpy_reference():
    scheme = quadpy.cn.gauss_legendre(3, 5)
    got = scheme.integrate(lambda x: np.exp(x[0]) * (1 + x[1] ** 2) * np.cos(x[2]))
    x, w = np.polynomial.legendre.leggauss(5)
    reference = (w @ np.exp(x)) * (w @ (1 + x**2)) * (w @ np.cos(x))
    assert got == pytest.approx(reference, abs=2e-14)


def test_cube_affine_mapping():
    scheme = quadpy.c2.gauss_legendre(4)
    got = scheme.integrate(lambda x: x[0] * x[1] ** 2, [[2, 5], [-1, 3]])
    assert got == pytest.approx(98.0)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_triangle_rule_polynomial_exactness(degree):
    scheme = quadpy.t2.get_good_scheme(degree)
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            exact = math.factorial(i) * math.factorial(j) / math.factorial(i + j + 2)
            assert scheme.integrate(lambda x, i=i, j=j: x[0] ** i * x[1] ** j) == pytest.approx(exact, abs=2e-14)


def test_triangle_affine_mapping_and_validation():
    scheme = quadpy.t2.get_good_scheme(3)
    vertices = np.array([[1.0, 2.0], [4.0, 2.0], [1.0, 6.0]])
    assert scheme.integrate(lambda x: np.ones(x.shape[1]), vertices) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        scheme.integrate(lambda x: x[0], np.ones((2, 2)))


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_tetrahedron_rule_polynomial_exactness(degree):
    scheme = quadpy.t3.get_good_scheme(degree)
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            for k in range(degree + 1 - i - j):
                exact = math.factorial(i) * math.factorial(j) * math.factorial(k) / math.factorial(i + j + k + 3)
                assert scheme.integrate(lambda x, i=i, j=j, k=k: x[0]**i * x[1]**j * x[2]**k) == pytest.approx(exact, abs=3e-14)


def test_input_contracts_and_compatibility_modules():
    assert quadpy.c3.gauss_legendre(2).dim == 3
    assert quadpy.c2.gauss_legendre(2).points.shape == (2, 4)
    assert quadpy.c3.gauss_legendre(4).integrate(lambda x: x[0] * x[1] * x[2]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        quadpy.cn.gauss_legendre(0, 2)
    with pytest.raises(ValueError):
        quadpy.t2.get_good_scheme(4)


def test_weighted_sum_rejects_lossy_or_ambiguous_ffi_inputs():
    values = np.ones((2, 3), dtype=np.float64)
    weights = np.ones(3, dtype=np.float64)
    with pytest.raises(TypeError):
        weighted_sum(values.astype(np.complex128), weights)
    with pytest.raises(TypeError):
        weighted_sum(values.astype(np.int64), weights)
    with pytest.raises(ValueError):
        weighted_sum(values, np.ones((1, 3)))
    with pytest.raises(ValueError):
        weighted_sum(values[:, :0], np.array([], dtype=np.float64))
    empty = weighted_sum(np.empty((0, 3)), weights)
    assert empty.shape == (0,)


@pytest.mark.parametrize("rows,n", [(31, 65_539), (256, 32_771)])
def test_weighted_sum_simd_tail_and_parallel_threshold(rows, n):
    rng = np.random.default_rng(rows + n)
    values = rng.normal(size=(rows, n))
    weights = rng.normal(size=n)
    np.testing.assert_allclose(
        weighted_sum(values, weights),
        np.einsum("...n,n->...", values, weights),
        rtol=1e-12,
        atol=1e-12,
    )
