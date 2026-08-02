"""Small C ABI for batched quadrature reductions over float64 buffers."""

from std.algorithm.functional import parallelize
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_MIN_ELEMENTS = 4_000_000
comptime PARALLEL_MIN_ROWS = 128
comptime PARALLEL_WORKERS = 8
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def weighted_sum_row(values: Ptr, weights: Ptr, dst: Ptr, n: Int, row: Int):
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var i = 0
    var base = row * n
    while i + 4 * W <= n:
        acc0 += values.load[width=W](base + i) * weights.load[width=W](i)
        acc1 += values.load[width=W](base + i + W) * weights.load[width=W](i + W)
        acc2 += values.load[width=W](base + i + 2 * W) * weights.load[width=W](i + 2 * W)
        acc3 += values.load[width=W](base + i + 3 * W) * weights.load[width=W](i + 3 * W)
        i += 4 * W
    while i + W <= n:
        acc0 += values.load[width=W](base + i) * weights.load[width=W](i)
        i += W
    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i < n:
        total += values[base + i] * weights[i]
        i += 1
    dst[row] = total


def weighted_sum(values: Ptr, weights: Ptr, dst: Ptr, n: Int, rows: Int):
    if rows >= PARALLEL_MIN_ROWS and n * rows >= PARALLEL_MIN_ELEMENTS:
        @parameter
        def compute_row(row: Int):
            weighted_sum_row(values, weights, dst, n, row)
        parallelize[compute_row](rows, PARALLEL_WORKERS)
    else:
        for row in range(rows):
            weighted_sum_row(values, weights, dst, n, row)


@export("mq_weighted_sum")
def mq_weighted_sum(values: Int, weights: Int, dst: Int, n: Int, rows: Int) abi("C"):
    weighted_sum(
        Ptr(unsafe_from_address=values), Ptr(unsafe_from_address=weights),
        Ptr(unsafe_from_address=dst), n, rows
    )
