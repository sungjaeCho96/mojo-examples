from algorithm import vectorize
from memory import memset_zero, stack_allocation
from common.types import BufferPtrFloat32, TensorF32, nelts

import math

@register_passable
struct Accumulator[T: DType, width: Int]:
    var data: DTypePointer[T]

    @always_inline
    fn __init__() -> Self:
        # allocate a DTypePointer on stack that doesn't need to be freed
        let data = stack_allocation[width, T]()
        memset_zero(data, width)
        return Self {data: data}

    @always_inline
    fn accumulate[_width: Int](inout self, val: SIMD[T, _width]) -> None:
        # This is a hack to make sure both SIMD ganve _width length.
        # SIMD[T, width] += SIMD[T, _width] is always an error.
        let newVal = self.data.simd_load[_width]() + val
        self.data.simd_store[_width](newVal)

    @always_inline
    fn total(self) -> SIMD[T, 1]:
        return self.data.simd_load[width]().reduce_add()

@always_inline
fn accum(inout a: TensorF32, b: TensorF32) -> None:
    let size = a.dim(0)

    @parameter
    fn _acc[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) + b.simd_load[_nelts](j))
    
    vectorize[nelts, _acc](size)
    
@always_inline
fn rmsnorm(inout o: BufferPtrFloat32, x: BufferPtrFloat32, weight: BufferPtrFloat32, size: Int) -> None:
    # Calculate sum of squares
    var tmp = Accumulator[DType.float32, nelts]()

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        tmp.accumulate(x.offset(j).simd_load[_nelts](0) ** 2)

    vectorize[nelts, _sum2](size)

    var ss: Float32 = tmp.total()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        let val = weight.simd_load[_nelts](j) * ss * x.simd_load[_nelts](j)
        o.offset(j).simd_store[_nelts](0, val)

    vectorize[nelts, _norm](size)