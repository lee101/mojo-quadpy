"""Small C ABI for batched quadrature reductions over float64 buffers."""

from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float64]()
comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 6_000_000
comptime MAX_WORKERS = 8


def weighted_sum_row(values: Ptr, weights: Ptr, dst: Ptr, n: Int, row: Int):
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var i = 0
    var base = row * n
    while i + 4 * W <= n:
        acc0 += values.unsafe_load[width=W](base + i) * weights.unsafe_load[width=W](i)
        acc1 += values.unsafe_load[width=W](base + i + W) * weights.unsafe_load[width=W](i + W)
        acc2 += values.unsafe_load[width=W](base + i + 2 * W) * weights.unsafe_load[width=W](i + 2 * W)
        acc3 += values.unsafe_load[width=W](base + i + 3 * W) * weights.unsafe_load[width=W](i + 3 * W)
        i += 4 * W
    while i + W <= n:
        acc0 += values.unsafe_load[width=W](base + i) * weights.unsafe_load[width=W](i)
        i += W
    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i < n:
        total += values[unsafe_offset=base + i] * weights[unsafe_offset=i]
        i += 1
    dst[unsafe_offset=row] = total


def weighted_sum(values: Ptr, weights: Ptr, dst: Ptr, n: Int, rows: Int):
    if rows * n < PARALLEL_THRESHOLD or rows == 1:
        for row in range(rows):
            weighted_sum_row(values, weights, dst, n, row)
        return

    var workers = min(rows, MAX_WORKERS)
    initialize_runtime()

    @__parameter
    @__copy_capture(values, weights, dst, n, rows, workers)
    def work(worker: Int):
        var begin = worker * rows // workers
        var end = (worker + 1) * rows // workers
        for row in range(begin, end):
            weighted_sum_row(values, weights, dst, n, row)

    parallelize[work](workers, workers)


@export("mq_weighted_sum")
def mq_weighted_sum(values: Int, weights: Int, dst: Int, n: Int, rows: Int) abi("C"):
    weighted_sum(
        Ptr(unsafe_from_address=values), Ptr(unsafe_from_address=weights),
        Ptr(unsafe_from_address=dst), n, rows
    )
