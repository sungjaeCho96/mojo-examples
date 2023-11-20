import benchmark
from sys.intrinsics import strided_load
from math import div_ceil, min
from memory import memset_zero
from memory.unsafe import DTypePointer
from random import rand, random_float64
from sys.info import simdwidthof
from runtime.llcl import Runtime

fn matrix_getitem(self: object, i: object) raises -> object:
    return self.value[i]

fn matrix_setitem(self: object, i: object, value: object) raises -> object:
    self.value[i] = value
    return None

fn matrix_append(self: object, value: object) raises -> object:
    self.value.append(value)
    return None

fn matrix_init(rows: Int, cols: Int) raises -> object:
    let value = object([])
    return object(
        Attr("value", value),
        Attr("__getitem__", matrix_getitem),
        Attr("__setitem__", matrix_setitem),
        Attr("rows", rows),
        Attr("cols", cols),
        Attr("append", matrix_append)
    )

def matmul_untyped(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

def benchmark_matmul_untyped(M: Int, N: Int, K: Int, python_gflops: Float64):
    C = matrix_init(M, N)
    A = matrix_init(M, K)
    B = matrix_init(K, N)
    for i in range(M):
        c_row = object([])
        a_row = object([])
        b_row = object([])
        for j in range(N):
            c_row.append(0.0)
            a_row.append(random_float64(-5, 5))
            b_row.append(random_float64(-5, 5))
        C.append(c_row)
        A.append(a_row)
        B.append(b_row)
    
    @parameter
    fn test_fn():
        try:
            _ = matmul_untyped(C, A, B)
        except:
            pass
    
    let secs = benchmark.run[test_fn]().mean()
    _ = (A, B, C)
    let gflops = ((2*M*N*K)/secs) / 1e9
    let speedup: Float64 = gflops / python_gflops
    print(gflops, "GFLOP/s, a", speedup.value, "x speedup over Python")

def main():
    let python_gflops: Float64 = 0.00512225803915595
    benchmark_matmul_untyped(128, 128, 128, python_gflops)