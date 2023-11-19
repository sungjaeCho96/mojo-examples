alias PointerString = Pointer[UInt8]
alias BufferPtrType = DTypePointer[DType.uint8]
alias BufferPtrFloat32 = DTypePointer[DType.float32]
alias PointerStrings = Pointer[PointerString]
alias TensorF32 = Tensor[DType.float32]
alias nelts = (4*simdwidthof[DType.float32]())