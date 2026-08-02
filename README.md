# mojo-quadpy

A small, standalone Mojo port of the compute-heavy part of numerical quadrature: evaluating a rule's weighted sum over large scalar or batched fields. It supplies practical standard rules for intervals, cubes, triangles, and tetrahedra behind a Python API modelled on quadpy's module names.

## Covered subset

| module | covered API | reference domain |
| --- | --- | --- |
| `quadpy.c1` | `gauss_legendre(n)`, `gauss_lobatto(n)` | `[-1, 1]` |
| `quadpy.cn` | `gauss_legendre(dim, n)` | `[-1, 1]^dim` |
| `quadpy.c2`, `quadpy.c3` | `gauss_legendre(n)` | square and cube conveniences |
| `quadpy.t2` | `get_good_scheme(degree)` for degree 0--3 | unit triangle |
| `quadpy.t3` | `get_good_scheme(degree)` for degree 0--2 | unit tetrahedron |

Every scheme exposes `points`, `weights`, `degree`, `source`, `test_tolerance`, `show()`, and `integrate(f, domain=None)`. Integrands are vectorized callables accepting points shaped `(dimension, npoints)` and return a real floating-point scalar or array whose final axis is `npoints`. `float16`, `float32`, and `float64` are accepted; complex, integer, object, and wider floating dtypes are rejected rather than silently losing information. Affine line, box, triangle, and tetrahedron domains are supported.

This deliberately does not yet cover quadpy's extensive named Stroud, Dunavant, Xiao-Gimbutas, sphere, disk, or pyramid catalogues, symbolic rule construction, or non-affine mappings. Numerical tests use independent NumPy tensor-product calculations and exact monomial integrals.

## Install and use

```bash
pixi install
pixi run build
pixi run test
```

```python
import numpy as np
import quadpy

rule = quadpy.t2.get_good_scheme(3)
triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
print(rule.integrate(lambda x: x[0] + x[1], triangle))  # 1.0

cube = quadpy.c2.gauss_legendre(8)
print(cube.integrate(lambda x: np.exp(x[0] + x[1])))
```

## How it works

Rule construction and affine mapping are concise NumPy code. The compute-bound final operation, a batched float64 weighted reduction, crosses `ctypes` once into `dist/libmojo-quadpy.so`. Python validates shape and dtype, makes owned contiguous buffers, and keeps them alive for the call; empty result sets stay in Python. Mojo rebuilds non-null pointers and performs a target-width SIMD reduction with four independent accumulators. Large batches use eight CPU workers; smaller batches stay serial to avoid launch overhead. No Mojo-side allocation occurs and the caller owns every buffer.

## Benchmark

Run `pixi run bench` to measure this checkout. Results below were measured on the build host; use the command above for your machine.

| case | Mojo | NumPy | result |
| --- | ---: | ---: | --- |
| weighted reduction (1 x 1,000,000) | 0.49 ms | 0.57 ms | 1.17x faster |
| weighted reduction (64 x 65,536) | 4.20 ms | 3.93 ms | 0.94x of NumPy |
| weighted reduction (512 x 16,384) | 9.29 ms | 11.07 ms | 1.19x faster |

Measured with `pixi run bench` on `Linux-6.8.0-136-generic-x86_64-with-glibc2.39 | x86_64`.

A GPU path is intentionally not included because this low-arithmetic-intensity reduction is designed for CPU-resident NumPy arrays.

## License

MIT
