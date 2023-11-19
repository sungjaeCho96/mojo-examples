from algorithm import sum
from algorithm import vectorize, parallelize, unroll
from builtin import string
from math import round
from memory import memcpy
from memory.buffer import Buffer
from memory.unsafe import DTypePointer
from random import rand
from runtime.llcl import num_cores
from sys import argv

# The SIMD vector width
from sys.info import simdwidthof
import math
import os
import random
import time

# My Pacakges
from accumulator import Accumulator
from common.types import PointerString, BufferPtrType, BufferPtrFloat32, PointerStrings, TensorF32

var workers = 0

alias nelts = (4 * simdwidthof[DType.float32]())

"""
var : mutable
let : immutable
fn[] : Optional parameters and keyword parameters (https://docs.modular.com/mojo/programming-manual.html#optional-parameters-and-keyword-parameters)
"""

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