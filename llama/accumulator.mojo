from algorithm import vectorize
from memory import memset_zero, stack_allocation
from types import BufferPtrFloat32, TensorF32, nelts
from tensorutils import TensorSlice

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