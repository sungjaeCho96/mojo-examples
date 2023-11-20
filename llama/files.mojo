from types import BufferPtrType, BufferPtrFloat32

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
        return ret

    fn get_offset(self) raises -> Int:
        if self.offset > self.size:
            raise Error("Offset is past the end of the FileBuf")
        if self.offset < 0:
            raise Error("Offset if before the beginning of the FileBuf")
        
        return self.offset

fn read_file(file_name: String, inout buf: FileBuf) raises:
    var f = open(file_name, "r")
    let data = f.read()
    f.close()

    let cp_size = data._buffer.size
    let cp_buf: BufferPtrType = BufferPtrType.alloc(cp_size)

    let data_ptr = data._as_ptr().bitcast[DType.uint8]()
    
    for i in range(cp_size):
        cp_buf.store(i, data_ptr.load(i))
    
    # don't free data
    _ = data
    buf.data = cp_buf
    buf.size = cp_size
    buf.offset = 0
    
    return None