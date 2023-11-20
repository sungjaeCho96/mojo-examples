from types import TensorF32, BufferPtrFloat32
from tensor import TensorShape
from tensorutils import TensorSlice
from files import FileBuf
from common import read_val_int

struct Config:
    var dim: Int
    var kv_dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var kv_mul: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int

    fn __init__(inout self):
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab_size = 0
        self.seq_len = 0
        self.kv_dim = 0
        self.kv_mul = 0
        self.head_size = 0

fn config_init(inout config: Config, inout buf: FileBuf, print_config: Int = 0) raises:
    config.dim = read_val_int(buf)
    config.hidden_dim = read_val_int(buf)
    config.n_layers = read_val_int(buf)
    config.n_heads = read_val_int(buf)
    config.n_kv_heads = read_val_int(buf)
    config.vocab_size = read_val_int(buf)
    config.seq_len = read_val_int(buf)
    config.head_size = config.dim // config.n_heads
    config.kv_dim = (config.n_kv_heads * config.dim) // config.n_heads
    config.kv_mul = config.n_heads // config.n_kv_heads

    if print_config:
        print("config: dim, hidden_dim", config.dim, config.hidden_dim)
        print("config: n_layers, n_heads", config.n_layers, config.n_heads)
        print("config: vocab_size, seq_len", config.vocab_size, config.seq_len)
        print("config: head_size", config.head_size)
        print("config: kv_dim, kv_mul", config.kv_dim, config.kv_mul)
    return None



struct RunState:
    var x: TensorF32 # activation at current time stamp (dim,)
    var xb: TensorF32 # same, but inside a residual branch (dim,)
    var xb2: TensorF32 # an additional buffer just for convenience (dim,)
    var hb: TensorF32 # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: TensorF32 # buffer for hidden dimension in the ffn (hidden_dim,)
    var q: TensorF32 # query (dim,)
    var k: TensorSlice # key (kv_dim,)
    var v: TensorSlice # value (kv_dim,)
    var att: TensorF32 # buffer for scores/attention values (n_heads, seq_len)
    var logits: TensorF32 # output logits
    var key_cache: TensorF32 # (layer, seq_len, dim)
    var value_cache: TensorF32 # (layer, seq_len, dim)

    fn __init__(inout self, config: Config) raises:
        self.x = TensorF32(config.dim)
        self.xb = TensorF32(config.dim)
        self.xb2 = TensorF32(config.dim)
        self.hb = TensorF32(config.hidden_dim)
        self.hb2 = TensorF32(config.hidden_dim)
        self.q = TensorF32(config.dim)
        self.att = TensorF32(config.n_heads, config.seq_len)
        self.logits = TensorF32(config.vocab_size)
        self.key_cache = TensorF32(config.n_layers, config.seq_len, config.kv_dim)
        self.value_cache = TensorF32(config.n_layers, config.seq_len, config.kv_dim)
        # So their updates flow to the caches, k and v are slices with shared memory.
        # Initialize with placeholders. The real tensors reference layer and position during forward pass.
        self.k = TensorSlice(TensorF32(TensorShape(1, config.kv_dim)), 1)
        self.v = TensorSlice(TensorF32(TensorShape(1, config.kv_dim)), 1)

struct TransformerWeights:
    var token_embedding_table: TensorF32
    var freq_cis_real: TensorF32
    var freq_cis_imag: TensorF32
    var rms_att_weight: TensorF32
    var wq: TensorF32
    var wk: TensorF32
    var wv: TensorF32
    var wo: TensorF32
    var rms_ffn_weight: TensorF32
    var w1: TensorF32
    var w3: TensorF32
    var w2: TensorF32
    var rms_final_weight: TensorF32
    var wcls: TensorF32

    fn __init__(
        inout self, config: Config, shared_weights: Int, inout buf: FileBuf
    ) raises:
        fn load_weights(inout buf: FileBuf, *dims: Int) raises -> TensorF32:
            # Ensure returned Tensor doesn't share a pointer with FileBuf
            let shape = TensorShape(dims)
            let result_data = BufferPtrFloat32.alloc(shape.num_elements())
            memcpy(
                result_data,
                buf.bitcast_offset_f32(shape.num_elements()),
                shape.num_elements(),
            )
            return TensorF32(result_data, shape)

        self.token_embedding_table = load_weights(buf, config.vocab_size, config.dim)
        self.rms_att_weight = load_weights(buf, config.n_layers, config.dim)
        self.wq = load_weights(buf, config.n_layers, config.dim, config.dim)
        self.wk = load_weights(buf, config.n_layers, config.kv_dim, config.dim)
        self.wv = load_weights(buf, config.n_layers, config.kv_dim, config.dim)
        self.wo = load_weights(buf, config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = load_weights(buf, config.n_layers, config.dim)
        self.w1 = load_weights(buf, config.n_layers, config.hidden_dim, config.dim)
        self.w2 = load_weights(buf, config.n_layers, config.dim, config.hidden_dim)
        self.w3 = load_weights(buf, config.n_layers, config.hidden_dim, config.dim)
        self.rms_final_weight = load_weights(buf, config.dim)
        # maybe need modifying for different model
        # config.head_size // 2 for stories and tinyllama-1.1
        self.freq_cis_real = load_weights(buf, config.seq_len, config.head_size // 2)
        self.freq_cis_imag = load_weights(buf, config.seq_len, config.head_size // 2)
        if shared_weights:
            self.wcls = self.token_embedding_table
        else:
            self.wcls = load_weights(buf, config.vocab_size, config.dim)