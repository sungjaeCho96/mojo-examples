from tensor import TensorShape
from algorithm import vectorize, parallelize
from memory import memset_zero
from random import rand
import math

from types import BufferPtrFloat32, TensorF32, nelts
from accumulator import Accumulator, accum
from config import Config, RunState, TransformerWeights

var workers = 0

struct TensorSlice:
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

    fn __getitem__(self, idx: Int) -> SIMD[DType.float32, 1]:
        return self._data.simd_load[1](idx)

    fn __setitem__(self, idx: Int, val: SIMD[DType.float32, 1]):
        return self.simd_store[1](idx,val)
        
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

    fn simd_store[nelts: Int](self, idx: Int, val: SIMD[DType.float32, nelts]):
        return self._data.simd_store[nelts](idx, val)

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

@always_inline
fn rmsnorm(inout o: TensorF32, x: TensorF32, weight: TensorF32):
    rmsnorm(o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))

@always_inline
fn rmsnorm(inout o: TensorF32, x: TensorF32, weight: TensorSlice):
    rmsnorm(o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))

@always_inline
fn softmax(inout x: TensorF32, start: Int, end: Int):
    var max_val: Float32 = -1e9

    @parameter
    fn _max[_nelts: Int](ii: Int):
        let val = x.simd_load[_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val
    
    vectorize[nelts, _max](end - start)

    var acc = Accumulator[DType.float32, nelts]()

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        let val = math.exp(x.simd_load[_nelts](start + ii) - max_val)
        x.simd_store[_nelts](start + ii, val)
        acc.accumulate(val)

    vectorize[nelts, _exp](end - start)

    var ssum = acc.total()
    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.simd_store[_nelts](start + ii, x.simd_load[_nelts](start + ii) / ssum)

    vectorize[nelts, _norm](end - start)

@always_inline
fn softmax(inout x: TensorF32) -> None:
    softmax(x, 0, x.dim(0))

@always_inline
fn batch_matmul[n: Int](
    C: StaticTuple[n, BufferPtrFloat32],
    A: BufferPtrFloat32,
    B: StaticTuple[n, BufferPtrFloat32],
    rows: Int,
    cols: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp = StaticTuple[n, Accumulator[DType.float32, nelts]]()
        @unroll
        for k in range(n):
            tmp[k] = Accumulator[DType.float32, nelts]()

        let row_offset = i * cols
        @parameter
        fn dot[_nelts: Int](j: Int):
            let a = A.simd_load[_nelts](j)
            @unroll
            for k in range(n):
                tmp[k].accumulate(a * B[k].simd_load[_nelts](row_offset + j))

        vectorize[nelts, dot](cols)

        @unroll
        for k in range(n):
            C[k].store(i, tmp[k].total())

    parallelize[compute_row](rows, workers)

fn matmul_dimension_checks(a: TensorShape, b: TensorShape) raises:
    if a[0] != b[1]:
        raise Error(
            "matmul dimension mismatch. A rows (dim 0) npt equal to B columns (dim 1)"
        )
    if b.rank() != 2:
        raise Error("mathmul expects B to be a 2D matrix")

@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorF32) raises:
    # B (d,n) @ A(n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[1](
        StaticTuple[1, BufferPtrFloat32](C.data()),
        A.data(),
        StaticTuple[1, BufferPtrFloat32](B.data()),
        B.dim(0),
        B.dim(1)
    )

@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorSlice) raises:
    # B (d,n) @ A(n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[1](
        StaticTuple[1, BufferPtrFloat32](C.data()),
        A.data(),
        StaticTuple[1, BufferPtrFloat32](B.data()),
        B.dim(0),
        B.dim(1)
    )

@always_inline
fn matmul(C: TensorSlice, A: TensorF32, B: TensorSlice) raises:
    # B (d,n) @ A(n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[1](
        StaticTuple[1, BufferPtrFloat32](C.data()),
        A.data(),
        StaticTuple[1, BufferPtrFloat32](B.data()),
        B.dim(0),
        B.dim(1)
    )

# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope_rotation_llama(
    inout state: RunState,
    freq_cls_real_row: TensorSlice,
    freq_cls_imag_row: TensorSlice,
    config: Config,
) -> None:
    # Stories model, llama2
    let head_size = config.head_size
    @parameter
    fn head_loop(i: Int):
        # Simple vectorization with (head_size // 2) steps gave junk transformers output.
        # Maybe because the nelt ranges and up overlapping between the steps.
        for j in range(0, config.head_size, 2):
            let fcr = freq_cls_real_row[j // 2]
            let fci = freq_cls_imag_row[j // 2]
            let q0 = state.q[i * head_size + j]
            let q1 = state.q[i * head_size + j + 1]
            state.q[i * head_size + j] = q0 * fcr - q1 * fci
            state.q[i * head_size + j + 1] = q0 * fci + q1 * fcr
            if i < config.n_kv_heads:
                let k0 = state.k[i * head_size + j]
                let k1 = state.k[i * head_size + j + 1]
                state.k[i * head_size + j] = k0 * fcr - k1 * fci
                state.k[i * head_size + j +1] = k0 * fci + k1 * fcr
    parallelize[head_loop](config.n_heads, workers)

@always_inline
fn transformers(
    token: Int,
    pos: Int,
    config: Config,
    inout state: RunState,
    weights: TransformerWeights,
) raises -> None:
    # A few convenience variables
    let dim = config.dim
    let hidden_dim = config.hidden_dim
    let head_size = config.head_size
    let kv_dim = config.kv_dim
    let kv_mul = config.kv_mul

    # Copy the token embedding into x
    let content_row = weights.token_embedding_table.data().offset(token * dim)
    memcpy[DType.float32](state.x.data(), content_row, dim)

    # Pluck out the "pos" row of freq_cls_real and freq_cls_imag
    let freq_cls_real_row = TensorSlice(weights.freq_cis_real, pos)
    let freq_cls_imag_row = TensorSlice(weights.freq_cis_imag, pos)

    # Forward all the layers
    for l in range(config.n_layers):
        # Attention rmsnorm
        rmsnorm(state.xb, state.x, TensorSlice(weights.rms_att_weight, l))
        # QKV matmuls for this position
        let loff = l * config.seq_len * config.kv_dim
        state.k = TensorSlice(state.key_cache, l, pos)
        state.v = TensorSlice(state.value_cache, l, pos)
        if kv_dim == dim:
            batch_matmul[3](
                StaticTuple[3, BufferPtrFloat32](
                    state.q.data(), state.k.data(), state.v.data()
                ),
                state.xb.data(),
                StaticTuple[3, BufferPtrFloat32](
                    TensorSlice(weights.wk, l).data(), TensorSlice(weights.wv, l).data()
                ),
                kv_dim,
                dim,
            )
        else:
            matmul(state.q, state.xb, TensorSlice(weights.wq, l))
            batch_matmul[2](
                StaticTuple[2, BufferPtrFloat32](state.k.data(), state.v.data()),
                state.xb.data(),
                StaticTuple[2, BufferPtrFloat32](
                    TensorSlice(weights.wk, l).data(), TensorSlice(weights.wv, l).data()
                ),
                kv_dim,
                dim,
            )
        
        # Apply RoPE rotation to the q and k vectors for each head
        rope_rotation_llama(state, freq_cls_real_row, freq_cls_imag_row, config)

        memset_zero(state.xb.data(), state.xb.num_elements())

        # Multihead attention. Iterate over all heads in parallel.
        @parameter
        fn loop_over_heads(h: Int):
            # Get the query vector for this head
            let q_offset = h * head_size

            # Index of attention scores for this head
            let att_offset = h * config.seq_len

            # Iterate over all timesteps, including the current one
            for t in range(pos + 1):
                # Starting index of the ket vector for this head and at this timestep
                let k_offset = loff + t * kv_dim + (h // kv_mul) * head_size
                # Calculate the attention score as the dot product of q and k
                var score: Float32 = 0.0

                @parameter
                fn score_fn[_nelts: Int](i: Int):
                    score += (
                        state.q.simd_load[_nelts](q_offset + i)
                        * state.key_cache.simd_load[_nelts](k_offset + i)
                    ).reduce_add()

                vectorize[nelts, score_fn](head_size)
                score /= math.sqrt[DType.float32, 1](head_size)

                # Save the score to the attention buffer
                state.att[att_offset + t] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(state.att, att_offset, att_offset + pos + 1)
            # Weighted sum of the values, store back into xb
            let xb_offset = h * head_size
            for t in range(pos + 1):
                # Starting index of the value vector for tbis head and at this timestep
                let v_offset = loff + t * kv_dim + (h // kv_mul) * head_size

                # Get the attention weight for this timestep
                let a = state.att[att_offset + t]

                # Accumulate the weighted value into xb
                @parameter
                fn xb_accumulate[_nelts: Int](i: Int):
                    let xbi = state.xb.simd_load[_nelts](
                        xb_offset + i
                    ) + a * state.value_cache.simd_load[_nelts](v_offset + i)
                    state.xb.simd_store[_nelts](xb_offset + i, xbi)
                
                vectorize[nelts, xb_accumulate](head_size)

        parallelize[loop_over_heads](config.n_heads, workers)
        # Final matrix multiplication to get the output of the attention
        matmul(state.xb2, state.xb, TensorSlice(weights.wo, l))
        # Residual connection back into x
        accum(state.x, state.xb2)
        # FFN rmsnorm
        rmsnorm(state.xb, state.x, TensorSlice(weights.rms_ffn_weight, l))

        # Calculate self.w1(x) and self.w3(x) for FFN
        batch_matmul[2](
            StaticTuple[2, BufferPtrFloat32](state.hb.data(), state.hb2.data()),
            state.xb.data(),
            StaticTuple[2, BufferPtrFloat32](
                TensorSlice(weights.w1, l).data(), TensorSlice(weights.w3, l).data()
            ),
            hidden_dim,
            dim,
        )

        @parameter
        fn silu[_nelts: Int](i: Int):
            let initial_hb = state.hb.simd_load[_nelts](i)
            # Apply SiLU activation function (silu(x) = x * sigmoid(x))
            let hbi = initial_hb * (1.0 / (1.0 + math.exp(-initial_hb)))
            # Elementwise multiply with w3(x)
            state.hb.simd_store[_nelts](i, hbi * state.hb2.simd_load[_nelts](i))

        vectorize[nelts, silu](hidden_dim)
        # Final matrix multiplication to get the output of the FFN
        matmul(state.xb, state.hb, TensorSlice(weights.w2, l))

        # Residual connection
        accum(state.x, state.xb)

    # Final rmsnorm
    rmsnorm(state.x, state.x, weights.rms_final_weight)

    # Classifier into logits
    matmul(state.logits, state.x, weights.wcls)

fn argmax(v: TensorF32) -> Int:
    # return argmax of v
    var max_i: Int = 0
    var max_p: Float32 = v[0]
    for i in range(v.dim(0)):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i

fn sample(prob: TensorF32) -> Int:
    let n = prob.dim(0)
    # Sample index from prob, they must sum to 1
    # get random value within (min, max) float32 range
    let r = rand[DType.float32](1)
    var cdf: Float32 = 0.0
    for i in range(n):
        cdf += prob[i]
        if r[0] < cdf:
            return i
    # Incase of rouding errors
    return n - 1