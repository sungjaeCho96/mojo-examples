from algorithm import sum
from algorithm import vectorize, parallelize, unroll
from builtin import string
from math import round
from memory import memset_zero, memcpy, stack_allocation
from memory.buffer import Buffer
from memory.unsafe import DTypePointer
from random import rand
from runtime.llcl import num_cores
from sys import argv
from tensor import Tensor, TensorShape, TensorSpec

# The SIMD vector width
from sys.info import simdwidthof
import math
import os
import random
import time

var workers = 0

alias nelts = (4 * simdwidthof[DType.float32]())

alias PointerString = Pointer[UInt8]
alias BufferPtrType = DTypePointer[DType.uint8]
alias BufferPtrFloat32 = DTypePointer[DType.float32]
alias PointerStrings = Pointer[PointerString]
alias TensorF32 = Tensor[DType.float32]

"""
var : mutable
let : immutable
fn[] : Optional parameters and keyword parameters (https://docs.modular.com/mojo/programming-manual.html#optional-parameters-and-keyword-parameters)
"""
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

struct TesorSlice:
    # Provides a view into a tensor representing a 1D slice on ites first or first 2 dimensions.
    # Same function signatures as Tensor but without owning the data
    var _data: BufferPtrFloat32
    var _shape: TensorShape

    fn __init__(inout self, t: TensorF32, layer: Int) raises:
        let elements_per_layer = t.num_elements() // t.dim(0)
        self._data = t.data().offset(layer * elements_per_layer)
        if t.rank() == 2:
            self._shape = TensorShape(t.dim(1))
        elif t.rank() == 3:
            self._shape = TensorShape(t.dim(1), t.dim(2))
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn __init__(inout self, t: TensorF32, layer: Int, row: Int) raises:
        let elements_per_layer = t.num_elements() // t.dim(0)
        let elements_per_row = elements_per_layer // t.dim(1)
        self._data = t.data().offset(
            layer * elements_per_layer + row * elements_per_row
        )
        if t.rank() == 3:
            self._shape = TensorShape(t.dim(2))
        elif t.rank() == 1:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error(
                "Trying to slice a 1D Tensor by layer and row. This requires a 3D"
                " Tensor."
            )
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented")

    fn data(self) -> BufferPtrFloat32:
        return self._data

    fn shape(self) -> TensorShape:
        return self._shape

    fn num_elements(self) -> Int:
        return self._shape.num_elements()

    fn dim(self, idx: Int) -> Int:
        return self._shape[idx]

    fn rank(self) -> Int:
        return self._shape.rank()

    fn simd_load[nelts: Int](self, idx: Int) -> SIMD[DType.float32, nelts]:
        return self._data.simd_load[nelts](idx)

    fn simd_load[nelts: Int](self, *indices: Int) -> SIMD[DType.float32, nelts]:
        if len(VariadicList(indices)) > 2:
            print(
                "Warning: TensorSlice only supports 1D and 2D indexing. Results are"
                " unlikely to be correct."
            )
        return self.simd_load[nelts](indices[0] * self._shape[1] + indices[1])

    fn simd_load[nelts: Int](self, indices: StaticIntTuple[2]) -> SIMD[DType.float32, nelts]:
        return self._data.simd_load[nelts](indices[0] * self._shape[1] + indices[1])

    fn __getitem__(self, idx: Int) -> SIMD[DType.float32, 1]:
        return self._data.simd_load[1](idx)

    fn simd_store[nelts: Int](self, idx: Int, val: SIMD[DType.float32, nelts]):
        return self._data.simd_store[nelts](idx, val)

    fn __setitem__(self, idx: Int, val: SIMD[DType.float32, 1]):
        return self.simd_store[1](idx,val)

struct FileBuf:
    var data: BufferPtrType
    var offset: Int
    var size: Int

    fn __init__(inout self):
        self.data = BufferPtrType()
        self.offset = 0
        self.size = 0

    fn __del__(owned self):
        self.data.free()

    fn move_offset(inout self, size: Int) raises:
        let new_offset = self.offset + size

        if new_offset > self.size:
            raise Error("Resulting offset will be past the end of the FileBuf")
        if new_offset < 0:
            raise Error("Resulting offset will be before the beginning of the FileBuf")

        self.offset = new_offset

    fn bitcast_offset_f32(inout self, size: Int) raises -> BufferPtrFloat32:
        let ret = self.data.offset(self.offset).bitcast[DType.float32]()
        self.move_offset(size * sizeof[DType.float32]())

    fn get_offset(self) raises -> Int:
        if self.offset > self.size:
            raise Error("Offset is past the end of the FileBuf")
        if self.offset < 0:
            raise Error("Offset if before the beginning of the FileBuf")
        
        return self.offset

fn read_val_int(inout buf: FileBuf) raises -> Int:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    let data = buf.data.offset(buf.get_offset()).bitcast[DType.int32]()
    let result = data.load(0)
    buf.move_offset(4)

    return result.to_int()

fn read_val_float32(inout buf: FileBuf) raises -> Float32:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    let val = buf.data.offset(buf.get_offset()).bitcast[DType.float32]().load(0)
    buf.move_offset(4)

    return val

fn read_val_str(inout buf: FileBuf, slen: Int) raises -> PointerString:
    let str = PointerString.alloc(slen+1)
    for i in range(slen):
        str.store(i, buf.data.load(buf.get_offset()))
        buf.move_offset(1)
    str.store(slen, 0)

    return str

fn str_len(s: PointerString) -> Int:
    var len = 0
    while s[len] != 0:
        len += 1
    return len

fn str_concat(s1: PointerString, s2: PointerString) -> PointerString:
    let l1 = str_len(s1)
    let l2 = str_len(s2)
    let str = PointerString.alloc(l1 + l2 + 1)

    memcpy[UInt8](str, s1, l1)
    memcpy[UInt8](str.offset(l1), s2, l2)
    str.store(l1 + l2, 0)

    return str

fn str_to_ptr(s: String) -> PointerString:
    
