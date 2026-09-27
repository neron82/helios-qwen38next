// Dense causal GQA attention. See qwen_gqa.cuh for the contract and why dense comes first.
#include "qwen_gqa.cuh"

#include <cuda_fp16.h>

namespace helios { namespace attn {

namespace {

// Decode-occupancy targets for the split-KV kernel below, swept on an 82-SM RTX 3090 over 1k..64k
// of key range and n = 1 and n = 2; see gqa_split_kv_chunks for what they bound.
//   GQA_WARPS_PER_SM 8 because the kernel is compiled __launch_bounds__(32, 8), i.e. exactly 8
//     warps fit per SM, so a target above 8 buys no extra resident warps - it just splits the chunks
//     finer and pushes the grid past one full wave. Measured at n = 2, 32k: 0.19 ms at 8, 0.27 at
//     10, 0.25 at 12, 0.22 at 16.
//   GQA_MIN_KEYS 32 because a shorter chunk cannot pay for its own 12 KB of partial output: at 4k,
//     32 keys/chunk measures 0.039 ms against 0.062 at 64 and 0.105 at 128.
static constexpr int GQA_WARPS_PER_SM = 8;   // chunk-grid size: n * n_kv_heads * chunks warps
static constexpr int GQA_MIN_KEYS = 32;       // smallest chunk worth launching; also the short-context cap
static constexpr int GQA_HD = 256;          // 8 values per lane x 32 lanes

// One BLOCK per (row, q head), with the KV range split across the block's warps.
//
// The original form was one WARP per (row, q head), each walking the whole KV cache serially in an
// online-softmax loop. That is fine at prefill (n*n_q_heads warps) and hopeless at decode, where
// n = 1 gives 24 warps for the entire GPU - three blocks on an 82-SM card - and each of them walks
// the full context one position at a time. Measured decode fell from 37.9 tok/s at 430 tokens of
// context to 15.9 at 4.4k and 5.4 at 17.6k on the strength of that alone.
//
// Now every warp takes a contiguous slice of the keys, computes its own (max, sumexp, weighted-V)
// partial in registers, and the block combines the partials with the usual log-sum-exp merge. This
// is the flash-decoding split, minus the cross-block stage (the block already owns the row, so no
// second kernel or global scratch is needed). head_dim 256 is assumed: 8 channels per lane.
// Warps per (row, head). The right value depends entirely on how many rows there are:
//
//  * decode (n = 1) launches n * n_q_heads = 24 blocks, so the KV range must be split or the card sits
//    idle - 16 warps each taking nk/16 keys;
//  * prefill (n = 256) already launches 6144 blocks, so splitting buys nothing and costs a 16-way
//    log-sum-exp combine (16 KB of shared memory traffic) per block, paid even by the early causal
//    rows whose KV slice is a handful of keys.
//
// One warp per (row, head) is therefore right at prefill and wrong at decode, which is why this is
// templated and dispatched on n rather than fixed.
template <int GQA_WARPS>
__global__ void gqa_dense_split_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                       const half* __restrict__ v, half* __restrict__ out, int n,
                                       int n_q_heads, int n_kv_heads, int hd, int pos0,
                                       float scale) {
  const int bh = blockIdx.x;
  if (bh >= n * n_q_heads) return;
  const int row = bh / n_q_heads;
  const int h = bh % n_q_heads;
  const int group = n_q_heads / n_kv_heads;
  const int kvh = h / group;
  const int nk = pos0 + row + 1;                       // keys 0 .. pos0+row inclusive

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int per = (nk + GQA_WARPS - 1) / GQA_WARPS;
  const int t0 = warp * per;
  const int t1 = min(nk, t0 + per);

  const half* qh = q + ((size_t)row * n_q_heads + h) * hd;
  float o[8];
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = 0.f;
  float m = -INFINITY, l = 0.f;

  for (int t = t0; t < t1; t++) {
    const half* kh = k + ((size_t)t * n_kv_heads + kvh) * hd;
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int d = lane * 8 + i;
      dot += __half2float(qh[d]) * __half2float(kh[d]);
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    dot *= scale;

    const float mnew = fmaxf(m, dot);
    const float a = __expf(m - mnew), b = __expf(dot - mnew);
    l = l * a + b;
    const half* vh = v + ((size_t)t * n_kv_heads + kvh) * hd;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int d = lane * 8 + i;
      o[i] = fmaf(b, __half2float(vh[d]), o[i] * a);
    }
    m = mnew;
  }

  __shared__ float sm_m[GQA_WARPS], sm_l[GQA_WARPS], sm_o[GQA_WARPS][256];
  if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
#pragma unroll
  for (int i = 0; i < 8; i++) sm_o[warp][lane * 8 + i] = o[i];
  __syncthreads();

  float gm = -INFINITY;
#pragma unroll
  for (int w = 0; w < GQA_WARPS; w++) gm = fmaxf(gm, sm_m[w]);
  float gl = 0.f;
#pragma unroll
  for (int w = 0; w < GQA_WARPS; w++) gl += sm_l[w] * __expf(sm_m[w] - gm);

  half* oh = out + ((size_t)row * n_q_heads + h) * hd;
  const float inv = gl > 0.f ? 1.f / gl : 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < GQA_WARPS; w++)
      acc += sm_o[w][lane * 8 + i] * __expf(sm_m[w] - gm);
    oh[lane * 8 + i] = __float2half_rn(acc * inv);
  }
}

// ---------------------------------------------------------------------------
// Tensor-core (mma) flash-attention prefill - the alternative to gqa_dense_split_kernel.
//
// The scalar kernel above is limited by its arithmetic, not by bandwidth: per (row, head, key) it
// runs 256 scalar fp32 FMAs and a 5-step butterfly reduction, which is ~1 instruction per MAC
// against the tensor cores' 16x8x16 per instruction. The second limit is that one block owns one
// row, so every key is re-read from L2 once per row.
//
// This kernel removes both: one block owns MMA_BR query rows, so a staged K/V tile is read from
// memory once per 64 rows instead of once per row, and both matmuls run on
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (fp16 in, fp32 accumulate). The softmax stays
// online and entirely in registers: fp32 running max, sum and output accumulator per lane, rescaled
// by exp2(m_old - m_new) at every key tile, so the kernel is exact up to fp16 rounding of P and the
// fp32 accumulation order - no second pass over the KV cache, no scratch buffer.
//
// Shape: 4 warps, each owning 16 query rows of the 64-row block for the whole key walk. head_dim is
// fixed at 256 and head_dim/8 = 32 output n-tiles per lane, i.e. 128 fp32 accumulators per lane.
// Keys and queries are staged in shared memory with __syncthreads() (no cp.async, no swizzle) and
// the fragments are read with the same direct addressing indexer_score_mma_k proves out in attn.cu:
// for a contraction step over channels [16*ks, 16*ks+16) and lane split g = lane>>2,
// tig = lane&3, the A operand is q[row g][16*ks+2*tig] / [row g+8] / [+8] / [row g+8][+8] and the
// B operand is k[key 8*nt+g][16*ks+2*tig] / [+8], all as 32-bit (two fp16) loads. Scores land in
// (score[nt][0], score[nt][1]) = (row g, key 8*nt+2*tig) and (row g+8, ...) for the pair, which is
// what the P fragment repack below indexes.
//
// Gate: off unless HELIOS_ATTN_MMA=1 (see gqa_dense_f16). gqa_dense_split_kernel is untouched and
// stays the default; gqa_dense_f16_mma exports the tensor-core path for the parity test.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two fp32 -> one 32-bit A/B fragment register. The union keeps the result in a register; a
// reinterpret_cast through a local half2 would send it to local memory.
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  union { __half2 h; uint32_t u; } c;
  c.h = __floats2half2_rn(lo, hi);
  return c.u;
}

// Lanes 4g..4g+3 hold the same two rows (g and g+8) of the mma tile, split over 4 column pairs, so a
// row-wise softmax reduction is a 2-step butterfly over tig.
__device__ __forceinline__ float warp_max4(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
  return x;
}
__device__ __forceinline__ float warp_sum4(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  return x;
}

constexpr int MMA_D = 256;       // head_dim this kernel is specialised for
constexpr int MMA_BR = 64;       // query rows per block
constexpr int MMA_BC = 32;       // keys per key tile
constexpr int MMA_WARPS = 4;     // 4 warps x 16 rows
constexpr int MMA_THREADS = MMA_WARPS * 32;
// Shared-memory row strides. All three tiles are read as fragments by 32 lanes at once, and a
// power-of-two stride collapses a 32-bit fragment load onto 4 banks. The +8 halves of padding
// (16 bytes, so staged rows stay 16-byte aligned for the uint4 copies) put a Q/K fragment load on
// bank 4*g + tig, all 32 banks exactly once. V is read a half at a time (see the PV loop) and with
// this stride lands on 8*tig + g/2, 16 words over 16 banks.
constexpr int MMA_QS = MMA_D + 8;
constexpr int MMA_KS = MMA_D + 8;
constexpr int MMA_VS = MMA_D + 8;
constexpr int MMA_SMEM = (MMA_BR * MMA_QS + MMA_BC * MMA_KS + MMA_BC * MMA_VS) * (int)sizeof(half);

__global__ __launch_bounds__(MMA_THREADS, 1) void gqa_dense_mma_kernel(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ out, int n, int n_q_heads, int n_kv_heads, int pos0, float scale) {
  constexpr int QKNt = MMA_BC / 8;   // 4  score n-tiles per key tile
  constexpr int QKKs = MMA_D / 16;   // 16 contraction steps over head_dim
  constexpr int PVNT = MMA_D / 8;    // 32 output n-tiles (channel groups of 8)
  constexpr int PVKS = MMA_BC / 16;  // 2  contraction steps over keys

  extern __shared__ __align__(16) half smem[];
  half* q_s = smem;                                   // [MMA_BR][MMA_QS]
  half* k_s = q_s + MMA_BR * MMA_QS;                  // [MMA_BC][MMA_KS]
  half* v_s = k_s + MMA_BC * MMA_KS;                  // [MMA_BC][MMA_VS]

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tig = lane & 3;
  const int q0 = blockIdx.x * MMA_BR;                 // first query row of this block
  const int h = blockIdx.y;
  const int kvh = h / (n_q_heads / n_kv_heads);
  const int nrows = min(MMA_BR, n - q0);              // query rows in this block (tail is short)
  if (nrows <= 0) return;
  const int row0 = warp * 16;                         // this warp's first row in the block
  const int max_key = pos0 + q0 + nrows - 1;          // last key any row of this block can see
  const int n_tiles = max_key / MMA_BC + 1;
  const size_t kv_stride = (size_t)n_kv_heads * MMA_D;
  // FA-style: fold the softmax scale and log2(e) into the exp2, so scores stay raw.
  const float scale_l2 = scale * 1.4426950408889634074f;

  const int qrow0 = q0 + row0 + g, qrow1 = qrow0 + 8;  // this lane's two rows
  const bool vis0 = qrow0 < q0 + nrows, vis1 = qrow1 < q0 + nrows;
  const int pos_r0 = pos0 + qrow0, pos_r1 = pos0 + qrow1;   // last visible key of each row

  // Q is staged once and stays resident for the whole key walk; rows past n are zero filled so the
  // tensor cores never see them (their outputs are not stored).
  for (int i = tid; i < MMA_BR * (MMA_D / 8); i += MMA_THREADS) {
    const int r = i / (MMA_D / 8), d = (i % (MMA_D / 8)) * 8;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < nrows) val = *(const uint4*)(q + ((size_t)(q0 + r) * n_q_heads + h) * MMA_D + d);
    *(uint4*)&q_s[r * MMA_QS + d] = val;
  }

  float acc[PVNT][4];
#pragma unroll
  for (int i = 0; i < PVNT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  // Running max is kept in scaled log2 units so the exp2 below is one fma; l and acc stay fp32.
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  for (int kb = 0; kb < n_tiles; kb++) {
    const int k0 = kb * MMA_BC;
    // Keys past the block's last visible position are zero filled: their scores are masked out, but
    // a stale or non-finite cache entry multiplied by a zero probability would still poison the mma
    // accumulator (0 * inf = NaN).
    for (int i = tid; i < MMA_BC * (MMA_D / 8); i += MMA_THREADS) {
      const int t = i / (MMA_D / 8), d = (i % (MMA_D / 8)) * 8;
      const int key = k0 + t;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
      if (key <= max_key) {
        const size_t off = (size_t)key * kv_stride + (size_t)kvh * MMA_D + d;
        kv = *(const uint4*)(k + off);
        vv = *(const uint4*)(v + off);
      }
      *(uint4*)&k_s[t * MMA_KS + d] = kv;
      *(uint4*)&v_s[t * MMA_VS + d] = vv;
    }
    __syncthreads();

    // S = Q K^T over this warp's 16 rows and the tile's MMA_BC keys.
    float score[QKNt][4];
#pragma unroll
    for (int nt = 0; nt < QKNt; nt++) score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < QKKs; ks++) {
      const int o = ks * 16 + tig * 2;
      const uint32_t a[4] = {*(const uint32_t*)&q_s[(row0 + g) * MMA_QS + o],
                             *(const uint32_t*)&q_s[(row0 + g + 8) * MMA_QS + o],
                             *(const uint32_t*)&q_s[(row0 + g) * MMA_QS + o + 8],
                             *(const uint32_t*)&q_s[(row0 + g + 8) * MMA_QS + o + 8]};
#pragma unroll
      for (int nt = 0; nt < QKNt; nt++) {
        const int brow = nt * 8 + g;                  // this lane's key row inside the 8-key tile
        const uint32_t b[2] = {*(const uint32_t*)&k_s[brow * MMA_KS + o],
                                *(const uint32_t*)&k_s[brow * MMA_KS + o + 8]};
        mma_m16n8k16(score[nt], a, b);
      }
    }

    // Causal mask, per (row, key): -inf where the key is in the future of that row. A tile that ends
    // at or before the block's first row position is fully visible to every row of the block.
    const bool full = (q0 + MMA_BR <= n) && (k0 + MMA_BC - 1 <= pos0 + q0);
    float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < QKNt; nt++) {
      if (!full) {
        const int key0 = k0 + nt * 8 + 2 * tig, key1 = key0 + 1;
        score[nt][0] = (vis0 && key0 <= pos_r0) ? score[nt][0] : -INFINITY;
        score[nt][1] = (vis0 && key1 <= pos_r0) ? score[nt][1] : -INFINITY;
        score[nt][2] = (vis1 && key0 <= pos_r1) ? score[nt][2] : -INFINITY;
        score[nt][3] = (vis1 && key1 <= pos_r1) ? score[nt][3] : -INFINITY;
      }
      bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
      bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
    }
    bm0 = warp_max4(bm0);
    bm1 = warp_max4(bm1);

    const float nm0 = fmaxf(m0, bm0 * scale_l2);
    const float nm1 = fmaxf(m1, bm1 * scale_l2);
    // m == -inf means "no visible key yet": alpha 0 keeps the accumulator at 0 instead of forming
    // exp2(-inf - -inf) = NaN. That case is a whole tile of masked scores for that row.
    const float alpha0 = (m0 == -INFINITY) ? 0.f : exp2f(m0 - nm0);
    const float alpha1 = (m1 == -INFINITY) ? 0.f : exp2f(m1 - nm1);

    // P = exp2(score*scale*log2e - m), written straight into the PV A-fragment registers: the
    // m16n8k16 A operand wants row g over keys 16*pk+2*tig (+1) in reg 0, row g+8 in reg 1, and
    // the +8 key half in regs 2 and 3 - which is exactly how the score n-tiles are laid out.
    uint32_t pf[PVKS][4];
    float bl0 = 0.f, bl1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < QKNt; nt++) {
      const float p00 = score[nt][0] > -INFINITY ? exp2f(__fmaf_rn(score[nt][0], scale_l2, -nm0)) : 0.f;
      const float p01 = score[nt][1] > -INFINITY ? exp2f(__fmaf_rn(score[nt][1], scale_l2, -nm0)) : 0.f;
      const float p10 = score[nt][2] > -INFINITY ? exp2f(__fmaf_rn(score[nt][2], scale_l2, -nm1)) : 0.f;
      const float p11 = score[nt][3] > -INFINITY ? exp2f(__fmaf_rn(score[nt][3], scale_l2, -nm1)) : 0.f;
      bl0 += p00 + p01;
      bl1 += p10 + p11;
      const int pk = nt >> 1;
      if ((nt & 1) == 0) {
        pf[pk][0] = pack_f16x2(p00, p01);
        pf[pk][1] = pack_f16x2(p10, p11);
      } else {
        pf[pk][2] = pack_f16x2(p00, p01);
        pf[pk][3] = pack_f16x2(p10, p11);
      }
    }

    l0 = l0 * alpha0 + bl0;
    l1 = l1 * alpha1 + bl1;
    m0 = nm0;
    m1 = nm1;
#pragma unroll
    for (int nt = 0; nt < PVNT; nt++) {
      acc[nt][0] *= alpha0;
      acc[nt][1] *= alpha0;
      acc[nt][2] *= alpha1;
      acc[nt][3] *= alpha1;
    }

    // O += P V, contracting over the tile's keys. The B operand needs one channel for two adjacent
    // keys, and the staged V is [key][channel], so those are two 16-bit loads per register (the MMA_VS
    // pad keeps that pattern conflict free; a transposed stage would need a second pass over smem).
#pragma unroll
    for (int ks = 0; ks < PVKS; ks++) {
      // b0 = {V[key 2*tig], V[key 2*tig+1]} at this lane's channel, b1 = the +8 and +9 keys. The
      // staged tile is [key][channel], so a key step is a row step, not the +8 that the QK K-fragment
      // uses along channels. (Reading the channel offset here instead - keys 16..31 then accumulate
      // the wrong V rows, and it only shows once a key past 7 carries any probability.)
      const int krow = ks * 16 + 2 * tig;
#pragma unroll
      for (int nt = 0; nt < PVNT; nt++) {
        const half* vp = &v_s[krow * MMA_VS + nt * 8 + g];
        const uint32_t b[2] = {
            (uint32_t)__half_as_ushort(vp[0]) | ((uint32_t)__half_as_ushort(vp[MMA_VS]) << 16),
            (uint32_t)__half_as_ushort(vp[8 * MMA_VS]) |
                ((uint32_t)__half_as_ushort(vp[9 * MMA_VS]) << 16)};
        mma_m16n8k16(acc[nt], pf[ks], b);
      }
    }
    __syncthreads();   // tile consumed; the next iteration may overwrite it
  }

  // Normalise once per row. The row sums are only complete after the 4-lane reduction.
  l0 = warp_sum4(l0);
  l1 = warp_sum4(l1);
  const float inv0 = l0 > 0.f ? __frcp_rn(l0) : 0.f;
  const float inv1 = l1 > 0.f ? __frcp_rn(l1) : 0.f;
#pragma unroll
  for (int nt = 0; nt < PVNT; nt++) {
    const int d0 = nt * 8 + 2 * tig;                 // the two channels this lane holds
    if (vis0)
      *(uint32_t*)&out[((size_t)qrow0 * n_q_heads + h) * MMA_D + d0] =
          pack_f16x2(acc[nt][0] * inv0, acc[nt][1] * inv0);
    if (vis1)
      *(uint32_t*)&out[((size_t)qrow1 * n_q_heads + h) * MMA_D + d0] =
          pack_f16x2(acc[nt][2] * inv1, acc[nt][3] * inv1);
  }
}

// Rows per launch below which the tensor-core path is not dispatched: the engine's decode shapes
// (n = 1) give only n_q_heads blocks, and the scalar/split-KV kernels are the right answer there.
constexpr int GQA_MMA_MIN_ROWS = 32;

// Tensor-core prefill is the DEFAULT. Set HELIOS_ATTN_MMA=0 to fall back to the scalar kernel.
//
// It was first landed default-off because the mma path re-associates the attention accumulation
// (tensor-core tiles + exp2 softmax + a different row reduction than the scalar 5-step butterfly),
// and greedy decoding can turn a re-association into a different - equally valid - token. Measured
// before flipping: at every parity shape the two agree to the same accuracy as the scalar path
// against the fp64 CPU reference (mma max_err 3.8e-05 at n=3/pos0=597, 3.7e-05 at n=70/pos0=1000,
// 9.8e-05 at n=33/pos0=100, 3.6e-04 at n=128/pos0=0 - the same order as the scalar DENSE kernel's
// own 3.0e-05 / 3.1e-05 / 6.1e-05 / 2.4e-04), and a greedy 40-token generation is token-identical
// to the scalar path on a representative prompt. Re-association is inherent to any tensor-core
// attention and is not a correctness defect; the accuracy is the scalar path's. So the default is
// ON, and the win is kept: 7.7x on the prefill attention phase (957 -> 123 ms) and +45% end-to-end
// prefill (322 -> 466 tok/s) at 10.3k, with the gap WIDENING in context (the scalar phase grows
// ~95 ms/chunk with sequence length, the mma phase ~3.4 ms/chunk). Decode is unaffected - the
// dispatch below requires n >= 32, and decode (n=1) keeps the split-KV path.
static bool gqa_mma_enabled() {
  const char* e = getenv("HELIOS_ATTN_MMA");
  return e == nullptr || e[0] != '0';
}

}  // namespace

void gqa_dense_f16_mma(const void* q, const void* k, const void* v, void* out, int n, int n_q_heads,
                       int n_kv_heads, int head_dim, int pos0, float scale, Stream s) {
  if (n <= 0) return;
  if (head_dim != MMA_D) return;
  // The tiles need more than the 48 KB default. The opt-in is per (function, device) and idempotent,
  // so it is tracked per device - the engine runs two GPUs and the attribute does not follow the
  // current device from the device it was set on.
  static bool attr_set[64] = {false};
  int dev = 0;
  cuda_check(cudaGetDevice(&dev));
  if (dev >= 0 && dev < 64 && !attr_set[dev]) {
    cuda_check(cudaFuncSetAttribute(gqa_dense_mma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, MMA_SMEM));
    attr_set[dev] = true;
  }
  dim3 grid((unsigned)((n + MMA_BR - 1) / MMA_BR), (unsigned)n_q_heads);
  gqa_dense_mma_kernel<<<grid, MMA_THREADS, MMA_SMEM, s>>>(
      (const half*)q, (const half*)k, (const half*)v, (half*)out, n, n_q_heads, n_kv_heads, pos0,
      scale);
  cuda_check(cudaPeekAtLastError());
}

void gqa_dense_f16(const void* q, const void* k, const void* v, void* out, int n, int n_q_heads,
                   int n_kv_heads, int head_dim, int pos0, float scale, Stream s) {
  if (n <= 0) return;
  if (head_dim != 256) return;                    // the split kernel assumes 8 channels per lane
  // Tensor-core prefill is the default (n >= 32). The scalar kernel below is bit-for-bit unchanged,
  // so setting HELIOS_ATTN_MMA=0 recovers it exactly. See gqa_mma_enabled for why the default is on
  // and the accuracy it was verified to.
  if (gqa_mma_enabled() && n >= GQA_MMA_MIN_ROWS) {
    gqa_dense_f16_mma(q, k, v, out, n, n_q_heads, n_kv_heads, head_dim, pos0, scale, s);
    return;
  }
  // 16 warps per (row, head), at both ends of the size range. The obvious guess - that prefill should
  // stop splitting because it already launches 6144 blocks and pays a 16-way log-sum-exp combine per
  // block - is wrong, and measurably so: one warp per row gave 275.5 ms/layer-ish for the attention
  // phase against 235.5 for 16 warps, and 32 warps gave 275.9. At prefill the later causal rows walk
  // enough keys that the KV walk still dominates the combine, so the split keeps paying; N = 1 is not
  // special in the direction I assumed. Measured, not reasoned.
  // 16 warps per (row, head). Three alternatives have now been measured worse than this one at
  // prefill: 1 warp per row with one block per row (275.5 ms/step on the attn phase), 32 warps
  // (275.9), and 8 rows per block with one warp per row so the K/V reuse becomes intra-block (251.6
  // against 237.0). The last was the most principled idea - the kernel is L2-throughput bound at
  // 1.63 TB/s of logical KV reads, re-reading each key once per query head - and it still lost, because
  // with one warp per row the serial KV walk costs more than the L1 reuse saves. A later experiment
  // went further and put multiple query heads in one block so each K/V element is fetched once and
  // reused across the GQA group (this model: 24 Q heads over 2 KV heads, group 12); it was bitwise
  // identical for every config except an over-registered corner, but also measured no faster than
  // this one-block-per-(row,head) form, because the reuse the extra heads buy is already served by L2
  // and the extra live accumulators cost more occupancy than they save. So the L2 reuse is real but
  // not the binding constraint here; the phase is limited by the scalar fp32 dot and its 5-step
  // butterfly reduction, not by fetching K/V. A tensor-core (mma) formulation is the next lever, and
  // it is gqa_dense_f16_mma above: 3.31 ms against 86.35 ms for this kernel on the engine's prefill
  // chunk shape (n = 1024 at pos0 = 5120, 24 heads), a 26x kernel-level win that takes the measured
  // 10.3k-token attention phase from 948.7 ms to 121.0 ms and prefill from 322.9 to 470.1 tok/s. It
  // is the default (HELIOS_ATTN_MMA=0 falls back here); its re-association of the attention
  // accumulation is verified numerically equivalent to this kernel, see gqa_mma_enabled.
  gqa_dense_split_kernel<16><<<n * n_q_heads, 512, 0, s>>>(
      (const half*)q, (const half*)k, (const half*)v, (half*)out, n, n_q_heads, n_kv_heads,
      head_dim, pos0, scale);
}

// ---------------------------------------------------------------------------
// Split-KV decode attention.
//
// One warp per (row, kv head, kv chunk), carrying ALL `group` query heads that share that KV head.
// A second kernel combines the chunk partials in chunk order with the same expressions as the
// shared-memory combine in gqa_dense_split_kernel.
//
// The previous form here was one warp per (row, QUERY head, chunk) with GQA_SPLIT_KV pinned at 16.
// nsys put that at 6.30 ms/step - 22.3% of the decode step, at 4.8% of DRAM peak, and the only
// kernel in the step under 5% - for two independent reasons, both fixed below.
//
// 1. Occupancy frozen at 15%. 16 chunks x 2 kv heads x 24 query heads = 768 warps = 9.4 warps/SM
//    on an 82-SM card, at any context length: a 4k decode and a 32k decode launched the same grid,
//    so the 32k step paid 4x the serial KV walk for zero extra parallelism. The chunk count is now
//    chosen at launch (gqa_split_kv_chunks) from the SM count, the kv-head count and the key range
//    length, so the grid scales with the context instead of being a constant.
//
// 2. Twelve-fold K/V re-read. With 24 query heads over 2 kv heads, every KV byte was fetched once
//    per query head: 452 MB of L1<-L2 traffic per launch against 18.9 MB of unique KV (~1.07 TB/s
//    through L2) while DRAM sat at 45 GB/s of 936. One warp now holds all 12 heads of its KV head
//    at once, so K and V are read once per token and the 12 uses are register arithmetic.
//
// Numerics. Within a (query head, key) the float operation sequence is UNCHANGED - same 8-element
// fp32 dot in the same order, same 5-step butterfly, same running max, same sum, same weighted-V
// accumulation - so each head's per-chunk partial is bitwise what the one-head-per-warp form
// produced for that same chunk. What does change is the partition of the key range, which
// re-associates the online softmax across chunks; that is the same re-association the previous
// constant-16 form already performed relative to the unsplit kernel, and the attention parity test
// holds the result to the same bound against a double-precision CPU reference.
//
// Two arithmetic shortcuts in the inner loop are exact rather than approximate, and both are
// guarded so they cannot change a bit:
//   * the running max only ever moves UP, so on a step where it does not move the rescale factor is
//     __expf(0) == 1.0f exactly, and `o * 1.0f == o` / `l * 1.0f == l` are bit-preserving no-ops.
//     The 96-accumulator rescale is therefore skipped on those steps, which is almost all of them.
//     The branch is warp-uniform: after the butterfly every lane holds the same dot, hence the same
//     max.
//   * the 8-element dot keeps the source form `d += q * k`; under --use_fast_math that already
//     contracts to one fmaf, so spelling it out would change nothing.
// ---------------------------------------------------------------------------

static constexpr int GQA_SPLIT_MAX = 256;   // hard cap: bounds the partial buffer and the combine smem

static int gqa_sm_count() {
  static int sms = 0;
  if (!sms) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (sms <= 0) sms = 1;
  }
  return sms;
}

// Chunks per key range for a launch of `n` rows over `nk` keys. Both clamps are about not paying
// more than the parallelism is worth:
//   * the target is GQA_WARPS_PER_SM warps per SM; one warp covers one (row, kv head) per chunk, so
//     the grid is n * n_kv_heads * chunks warps;
//   * a chunk shorter than GQA_MIN_KEYS keys cannot pay for its own 12 KB of partial output, so the
//     count is capped at nk / GQA_MIN_KEYS. This is also the only term that depends on the key
//     range, and it is what makes the grid shrink at short context.
// The result is monotone non-decreasing in nk, which is what lets gqa_split_kv_bytes bound a launch
// from a context CAPACITY rather than from the current position.
static int gqa_split_kv_chunks(int n, int n_kv_heads, int nk) {
  const long long lanes = (long long)n * n_kv_heads;
  long long s = ((long long)GQA_WARPS_PER_SM * gqa_sm_count() + lanes - 1) / lanes;
  const long long by_keys = nk / GQA_MIN_KEYS;
  if (s > by_keys) s = by_keys;
  if (s > GQA_SPLIT_MAX) s = GQA_SPLIT_MAX;
  return s < 1 ? 1 : (int)s;
}

// GROUP query heads per warp. It is a template parameter because it is the size of the register
// array of accumulators (GROUP * 8 fp32 per lane). The instantiated values cover every GQA group in
// use; anything else falls back to GROUP = 1, i.e. the old one-query-head-per-warp shape, which is
// still correct because the head index is computed from the grid rather than assumed.
template <int GROUP>
__global__ __launch_bounds__(32, 8) void gqa_split_kv_kernel(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    float* __restrict__ partial, int n, int n_q_heads, int n_kv_heads, int pos0, int split,
    float scale, const int* __restrict__ row_pos, size_t kv_slot_stride) {
  const int chunk = blockIdx.x;
  const int tiles = (n_q_heads / n_kv_heads + GROUP - 1) / GROUP;   // GROUP-sized query-head tiles
  const int rk = blockIdx.y / tiles;                              // (row, kv head) index
  const int tile = blockIdx.y - rk * tiles;
  const int row = rk / n_kv_heads;
  const int kvh = rk - row * n_kv_heads;
  const int group = n_q_heads / n_kv_heads;
  const int j0 = tile * GROUP;                                     // first head of this tile
  const int nj = group - j0 < GROUP ? group - j0 : GROUP;          // == GROUP on every real dispatch
  // Per-row key count and per-row cache base. Both default to the single-sequence behaviour this
  // kernel always had (row r at pos0 + r over one shared cache), so a null row_pos with a zero
  // stride is exactly the old code. See gqa_dense_split_kv_decode for what the batched form is for.
  const int nk = (row_pos ? row_pos[row] : pos0 + row) + 1;        // keys 0 .. this row's pos incl
  const half* __restrict__ krow = k + (size_t)row * kv_slot_stride;
  const half* __restrict__ vrow = v + (size_t)row * kv_slot_stride;

  const int lane = threadIdx.x;
  // Same partition the 16-warp block used, with the warp count supplied by the launch instead of
  // frozen at 16: per = ceil(nk / split), chunk c owns [c*per, min(nk, ...)).
  const int per = (nk + split - 1) / split;
  const int t0 = chunk * per;
  const int t1 = min(nk, t0 + per);

  // Q for this tile's heads, converted to fp32 ONCE. The one-head-per-warp form re-read these from
  // memory on every key; here they are GROUP * 8 registers that amortise over the whole chunk.
  const half* qh = q + ((size_t)row * n_q_heads + kvh * group + j0) * GQA_HD + lane * 8;
  float qv[GROUP][8];
#pragma unroll
  for (int j = 0; j < GROUP; j++) {
    const bool live = j < nj;
    const half* src = qh + (size_t)j * GQA_HD;
#pragma unroll
    for (int i = 0; i < 8; i++) qv[j][i] = live ? __half2float(src[i]) : 0.f;
  }
  float o[GROUP][8];
#pragma unroll
  for (int j = 0; j < GROUP; j++)
#pragma unroll
    for (int i = 0; i < 8; i++) o[j][i] = 0.f;
  float m[GROUP], l[GROUP];
#pragma unroll
  for (int j = 0; j < GROUP; j++) { m[j] = -INFINITY; l[j] = 0.f; }

#pragma unroll 4
  for (int t = t0; t < t1; t++) {
    // K and V are read ONCE for all nj heads, which is the whole point: the one-head-per-warp form
    // fetched these same bytes nj times. 8 halves per lane is a 16-byte aligned uint4 and 32 lanes
    // cover the full 256-wide head, so each fetch is one fully coalesced 512-byte transaction.
    const size_t off = ((size_t)t * n_kv_heads + kvh) * GQA_HD + lane * 8;
    union { uint4 v; __half2 h[4]; } kr, vr;
    kr.v = *reinterpret_cast<const uint4*>(krow + off);
    vr.v = *reinterpret_cast<const uint4*>(vrow + off);
    float kf[8], vf[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float2 a = __half22float2(kr.h[i]);
      const float2 b = __half22float2(vr.h[i]);
      kf[2 * i] = a.x; kf[2 * i + 1] = a.y;
      vf[2 * i] = b.x; vf[2 * i + 1] = b.y;
    }

    float dot[GROUP];
#pragma unroll
    for (int j = 0; j < GROUP; j++) {
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) d += qv[j][i] * kf[i];
      dot[j] = d;
    }
    // 5-step butterfly, one per head, all in this one warp instead of nj warps' worth of it.
#pragma unroll
    for (int off = 16; off; off >>= 1)
#pragma unroll
      for (int j = 0; j < GROUP; j++) dot[j] += __shfl_xor_sync(0xffffffffu, dot[j], off);
#pragma unroll
    for (int j = 0; j < GROUP; j++) dot[j] *= scale;

    // Online softmax; see the header comment for why the rescale is conditional and exact.
    bool chg = false;
#pragma unroll
    for (int j = 0; j < GROUP; j++) chg |= (fmaxf(m[j], dot[j]) != m[j]);
    if (chg) {
#pragma unroll
      for (int j = 0; j < GROUP; j++) {
        const float mnew = fmaxf(m[j], dot[j]);
        const float a = __expf(m[j] - mnew);
        l[j] = l[j] * a;
#pragma unroll
        for (int i = 0; i < 8; i++) o[j][i] = o[j][i] * a;
        m[j] = mnew;
      }
    }
#pragma unroll
    for (int j = 0; j < GROUP; j++) {
      const float b = __expf(dot[j] - m[j]);
      l[j] = l[j] + b;
#pragma unroll
      for (int i = 0; i < 8; i++) o[j][i] = fmaf(b, vf[i], o[j][i]);
    }
  }

#pragma unroll
  for (int j = 0; j < GROUP; j++) {
    if (j >= nj) break;
    float* dst =
        partial + ((size_t)(row * n_q_heads + kvh * group + j0 + j) * split + chunk) * (2 + GQA_HD);
    if (lane == 0) { dst[0] = m[j]; dst[1] = l[j]; }
#pragma unroll
    for (int i = 0; i < 8; i++) dst[2 + lane * 8 + i] = o[j][i];
  }
}

// One warp per (row, query head, 32-channel slice): the 256-wide head is eight 32-wide slices, so a
// 24-head decode is 192 warps spread over the 82 SMs. The single-block-per-head form this replaces
// read exactly the same partials but from 24 blocks sitting on 24 SMs, which measured 10.8 us of
// the 66 us launch at a 9k context - one seventh of the phase spent on 29% of the card.
__global__ __launch_bounds__(32) void gqa_split_kv_combine(const float* __restrict__ partial,
                                                          half* __restrict__ out, int n,
                                                          int n_q_heads, int split) {
  const int bh = blockIdx.x >> 3;                       // (row, query head)
  if (bh >= n * n_q_heads) return;
  const int lane = threadIdx.x;
  const int d = (blockIdx.x & 7) * 32 + lane;           // this lane's one output channel
  const float* base = partial + (size_t)bh * split * (2 + GQA_HD);

  // EVERY lane reads all `split` (m, l) pairs itself, for the same reason the old combine did: a
  // shared-memory staging array promoted to registers by the unroller under constant bounds ends up
  // per-lane private, and the lanes that never wrote it combine against garbage. Here `split` is a
  // runtime value so that array cannot even be a register array; the reads are warp-uniform, so
  // they are L1 broadcasts, and the exp weights are staged once into shared memory.
  __shared__ float sm_w[GQA_SPLIT_MAX];
  float gm = -INFINITY;
  for (int c = 0; c < split; c++) gm = fmaxf(gm, base[(size_t)c * (2 + GQA_HD)]);
  float gl = 0.f;
  for (int c = 0; c < split; c++)
    gl += base[(size_t)c * (2 + GQA_HD) + 1] * __expf(base[(size_t)c * (2 + GQA_HD)] - gm);
  for (int c = lane; c < split; c += 32) sm_w[c] = __expf(base[c * (2 + GQA_HD)] - gm);
  __syncwarp();

  // Identical combine order and expressions to the shared-memory version in
  // gqa_dense_split_kernel, with warp w replaced by chunk c.
  float acc = 0.f;
  for (int c = 0; c < split; c++) acc += base[(size_t)c * (2 + GQA_HD) + 2 + d] * sm_w[c];

  const float inv = gl > 0.f ? 1.f / gl : 0.f;
  half* oh = out + (size_t)bh * GQA_HD;
  oh[d] = __float2half_rn(acc * inv);
}

size_t gqa_split_kv_bytes(int n, int n_q_heads, int n_kv_heads, int head_dim, int ctx_cap) {
  // Bound the largest grid the launcher can produce for up to `n` rows at any position up to
  // `ctx_cap`: gqa_split_kv_chunks is monotone in the key range, so the capacity value dominates
  // every real position, and n * chunks(n) is what the buffer is indexed by. The loop is over the
  // row count because the chunk count FALLS as n grows, which is what keeps this from being 32x
  // the decode size: at n = 1 the count is the per-SM target over n_kv_heads, at n = 32 it is a
  // thirty-second of that, and the product is flat. 4.8 MB at 262k context, against 12.7 MB for the
  // old fixed 16 over 32 rows - so the wider grid is paid for out of the existing allocation.
  size_t worst = 0;
  for (int r = 1; r <= n; r++) {
    const size_t slots = (size_t)r * (size_t)gqa_split_kv_chunks(r, n_kv_heads, ctx_cap);
    if (slots > worst) worst = slots;
  }
  return worst * (size_t)n_q_heads * (2 + head_dim) * sizeof(float);
}

void gqa_dense_split_kv_decode(const void* q, const void* k, const void* v, void* out, float* partial,
                                int n, int n_q_heads, int n_kv_heads, int head_dim, int pos0,
                                float scale, Stream s, const int* row_pos, size_t kv_slot_stride) {
  if (n <= 0) return;
  const int group = n_q_heads / n_kv_heads;
  const int split = gqa_split_kv_chunks(n, n_kv_heads, pos0 + n);
  // One warp per (row, kv head, chunk), GROUP query heads each. `tiles` is 1 for every instantiated
  // group; the grid only grows past n * n_kv_heads blocks for a group this kernel does not cover.
#define GQA_LAUNCH(G)                                                                 \
  do {                                                                                \
    const int tiles = (group + (G) - 1) / (G);                                        \
    dim3 grid((unsigned)split, (unsigned)(n * n_kv_heads * tiles), 1);                 \
    gqa_split_kv_kernel<G><<<grid, 32, 0, s>>>((const half*)q, (const half*)k,          \
                                                (const half*)v, partial, n, n_q_heads,  \
                                                n_kv_heads, pos0, split, scale,         \
                                                row_pos, kv_slot_stride);               \
  } while (0)
  if (group == 12) GQA_LAUNCH(12);
  else if (group == 16) GQA_LAUNCH(16);
  else if (group == 8) GQA_LAUNCH(8);
  else if (group == 6) GQA_LAUNCH(6);
  else if (group == 4) GQA_LAUNCH(4);
  else if (group == 2) GQA_LAUNCH(2);
  else GQA_LAUNCH(1);   // group 1 and every group this kernel does not instantiate
#undef GQA_LAUNCH
  gqa_split_kv_combine<<<(unsigned)(n * n_q_heads * 8), 32, 0, s>>>(partial, (half*)out, n,
                                                               n_q_heads, split);
}

}} // namespace helios::attn
