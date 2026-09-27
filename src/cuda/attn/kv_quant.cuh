#pragma once
// Quantized KV cache - a port of exllamav3's CacheLayer_quant (exllamav3/cache/quant.py) and the
// kernels behind it (exllamav3_ext/cache/q_cache_kernels.cuh).
//
// LAYOUT (per K tensor, per V tensor, per layer; helios keeps K and V in separate buffers):
//
//   quantized: uint32 q[token][gpt * bits]   packed codes
//               half    s[token][gpt]         one absmax scale per 32-value group
//   gpt = token_dim / 32, token_dim = n_kv_heads * head_dim
//
// so one token costs gpt * (4*bits + 2) bytes instead of 2 * token_dim: 224 B against 1024 B at
// the reference's CACHE_QUANT=3 on this model (2 kv heads x head_dim 256), a 4.57x reduction.
//
// ALGORITHM (verbatim from the reference, compand_a = 0 which is the reference server's default):
// values are grouped in runs of 32 along the token dimension, rotated by an unnormalized 32-point
// Hadamard H32 (scaled by 1/sqrt(32)), scaled to [-1, 1] by the group's absmax (stored as one half),
// and quantized linearly onto 2^bits centroids at ((2q+1)/2^bits - 1). The reference splits the
// H32 into an in-register H4 over the four values a lane owns plus a three-round 8-lane butterfly;
// the composite is exactly the Sylvester matrix H32[i][j] = (-1)^popcount(i & j) in the value order
// 4*lane + j, which is what this port applies directly (verified bit-exact against the reference
// kernels, see test_kv_quant).
//
// PACKING is by BIT PLANE, not a linear little-endian pack: the bitrate is split into descending
// powers of two (3 = a 2-bit plane then a 1-bit plane, 6 = 4 then 2), and each plane stores
// w bits per value in its own run of w uint32 words, with value 4*lane + j at bit
// (lane*4*w) + j*w of that run. That is the reference's layout and the consumer here reproduces it
// exactly, so a cache written here can be read by the reference kernels and vice versa.
//
// DEQUANTIZATION is done in integer arithmetic. H32 is symmetric and H32 * 1 = 32 * e_0, so
//
//   x = H32 * ((q - (m - 0.5)) * s * r32 / m) = (s * r32 / m) * (H32*q - 32*(m - 0.5)*e_0)
//
// and the whole H32 transform of the integer codes is exact in int32 (|H32*q| <= 32*255). That
// leaves one float multiply per value instead of a 32-float register-resident butterfly, which
// matters because this runs inside the mma prefill kernel's K/V staging loop and the split-KV
// decode kernel's inner loop. The result differs from the reference's float path only by the
// reference's own rounding (bounded by 1 fp16 ulp, asserted in the test).

#include "../cuda_shim.hpp"

#include <cstdlib>

namespace helios { namespace attn {

// One quantized tensor pair (packed codes + half scales). Both are [token][...] with token ==
// absolute position in the sequence, so a chunk starting at pos0 writes at row pos0.
struct KvQuant {
  const void* q = nullptr;
  const void* s = nullptr;
};

inline int kvq_groups(int token_dim) { return token_dim / 32; }

// Bytes one token occupies in one quantized tensor.
inline size_t kvq_row_bytes(int token_dim, int bits) {
  return (size_t)kvq_groups(token_dim) * ((size_t)bits * 4 + 2);
}
// Byte offset of a quantized tensor's half-scale array from its base. The codes of ALL ctx_rows
// tokens come first, then the scales, so this scales with the cache length - reading it as a
// per-token stride (kvq_row_bytes) puts K's scales on top of K's codes from token 1 on and runs
// the last rows off the end of the allocation.
inline size_t kvq_scale_offset(int ctx_rows, int token_dim, int bits) {
  return (size_t)ctx_rows * (size_t)kvq_groups(token_dim) * (size_t)bits * 4;
}

// Capacity of the KV cache in rows, published by the runner at init. The scale array of a
// quantized tensor sits after all of that tensor's codes, so both the write and the read paths
// need it; it is one call so the runner sets it and everything else reads it.
inline int& kvq_ctx_rows_ref() { static int rows = 0; return rows; }
inline int kvq_ctx_rows() { return kvq_ctx_rows_ref(); }

// HELIOS_KV_QUANT: 0 (the default) keeps the fp16 KV cache untouched; 2..8 selects the bitrate.
// Read once per process so the allocator and every kernel agree.
//
// The attention READ path is dequant STAGING, not a fused dequant inside the attention kernels: the
// cache is materialized into an fp16 scratch (attn_layer.cu) with kvq_dequant immediately before the
// call, and the UNCHANGED attention kernels are pointed at that scratch. The scratch is laid out
// exactly like the fp16 cache - [row][n_kv_heads][head_dim] fp16, row == absolute position - so the
// kernels cannot tell the two apart, and the attention arithmetic is bit-for-bit the code that was
// tuned and verified. That is deliberate: templating the mma kernel on the bitrate, and mapping lanes
// to quant groups in the scalar kernels, both produced wrong output (fluent wrong text) and were
// reverted. What this costs is one extra pass over the KV range per layer per call - a write of
// (pos0+n) * token_dim * 2 bytes, the same volume the attention kernel then reads - plus the scratch
// itself (2 * ctx * token_dim * 2 B, 512 MB at 262k on this model). Both are measured and reported in
// RESULTS.md; against the ~4.7 GiB of fp16 cache freed, the trade is strongly positive.
//
// The format is bit-exact against exllamav3 (test_kv_quant, golden vectors produced by compiling the
// reference's own q_cache_kernels.cuh). The path is LOSSY by construction: greedy output on a
// quantized cache is not expected to be token-identical to fp16, only to agree with it for a while
// and stay coherent.
inline int kvq_bits() {
  static int bits = [] {
    const char* e = getenv("HELIOS_KV_QUANT");
    if (!e || !*e) return 0;
    int b = atoi(e);
    if (b < 2 || b > 8) return 0;
    fprintf(stderr,
            "[kv-quant] HELIOS_KV_QUANT=%d: KV cache is the exllamav3 CacheLayer_quant layout; the "
            "attention read path dequantizes into an fp16 staging buffer, so the attention kernels "
            "are unchanged and the output is lossy (not token-identical to fp16). See RESULTS.md.\n",
            b);
    return b;
  }();
  return bits;
}

// Quantize rows [pos0, pos0+n) of the fp16 [n, token_dim] K and V projections into the cache.
// One launch does both tensors (the reference's paged kernel does the same), so a chunk costs one
// extra launch per layer rather than two.
void kvq_write(const void* k, const void* v, KvQuant kc, KvQuant vc, int n, int token_dim, int bits,
               int pos0, Stream s = 0);

// Plain dequantize-to-fp16, for tests and for any consumer that wants a materialized cache.
void kvq_dequant(const void* q, const void* scales, void* out_half, int n, int token_dim, int bits,
                 int pos0, Stream s = 0);

#ifdef __CUDACC__

namespace kvq_impl {

// 1/sqrt(32) - the reference's r32. Folding it into the stored scale is what makes the transform
// an involution: the scale is the absmax of H32*x/sqrt(32), and dequant applies H32 unscaled.
constexpr float kR32 = 0.17677669529663688110f;

// One bit plane's field for value (4*lane + j) of a group: the plane owns w bits per value across
// w uint32 words starting at word_base, value j of lane L at bit (L*4*w) + j*w. When packing, rem
// codes bits remain above this plane.
template <int w>
__device__ __forceinline__ uint32_t cq_plane_get(const uint32_t* __restrict__ gw, int word_base,
                                                 int lane, int j) {
  const int off = lane * 4 * w;
  return (gw[word_base + (off >> 5)] >> ((off & 31) + j * w)) & ((1u << w) - 1u);
}

template <int w>
__device__ __forceinline__ void cq_plane_put(uint32_t* __restrict__ gw, int word_base, int lane,
                                             int j, int rem, uint32_t code) {
  const int off = lane * 4 * w;
  const uint32_t field = (code >> rem) & ((1u << w) - 1u);
  gw[word_base + (off >> 5)] |= field << ((off & 31) + j * w);
}

// The bitrate split into descending powers of two. 3 -> {2, 1}, 6 -> {4, 2}, 8 -> {8}, and so on;
// a 4-bit-aligned field never straddles a word, which is what makes the unpack branch-free.
template <int bits>
__device__ __forceinline__ uint32_t cq_unpack(const uint32_t* __restrict__ gw, int lane, int j) {
  uint32_t acc = 0;
  int rem = bits, wb = 0;
  if constexpr (bits & 8) { acc = (acc << 8) | cq_plane_get<8>(gw, wb, lane, j); rem -= 8; wb += 8; }
  if constexpr (bits & 4) { acc = (acc << 4) | cq_plane_get<4>(gw, wb, lane, j); rem -= 4; wb += 4; }
  if constexpr (bits & 2) { acc = (acc << 2) | cq_plane_get<2>(gw, wb, lane, j); rem -= 2; wb += 2; }
  if constexpr (bits & 1) { acc = (acc << 1) | cq_plane_get<1>(gw, wb, lane, j); }
  return acc;
}

template <int bits>
__device__ __forceinline__ void cq_pack(uint32_t* __restrict__ gw, int lane, int j, uint32_t code) {
  int rem = bits, wb = 0;
  if constexpr (bits & 8) { cq_plane_put<8>(gw, wb, lane, j, rem - 8, code); rem -= 8; wb += 8; }
  if constexpr (bits & 4) { cq_plane_put<4>(gw, wb, lane, j, rem - 4, code); rem -= 4; wb += 4; }
  if constexpr (bits & 2) { cq_plane_put<2>(gw, wb, lane, j, rem - 2, code); rem -= 2; wb += 2; }
  if constexpr (bits & 1) { cq_plane_put<1>(gw, wb, lane, j, rem - 1, code); }
}

// H32 in the value order 4*lane + j: the in-register H4 the reference applies to the four values a
// lane owns, then the three 8-lane butterfly rounds. Kept in this exact order (and in fp32) so the
// codes this port writes are BIT-IDENTICAL to the reference's, not merely equivalent in MSE.
__device__ __forceinline__ void cq_hadamard32(float* __restrict__ v) {
#pragma unroll
  for (int lane = 0; lane < 8; lane++) {
    const int i = lane * 4;
    const float a = v[i], b = v[i + 1], c = v[i + 2], d = v[i + 3];
    const float s0 = a + b, d0 = a - b, s1 = c + d, d1 = c - d;
    v[i]     = s0 + s1;
    v[i + 1] = d0 + d1;
    v[i + 2] = s0 - s1;
    v[i + 3] = d0 - d1;
  }
#pragma unroll
  for (int i = 1; i < 8; i <<= 1) {
    // The reference's shuffle pairs are disjoint within a round, so the round is done in place on
    // the low lane of each pair; the high lane's sign is -1, which is where a - b comes from.
#pragma unroll
    for (int lane = 0; lane < 8; lane++) {
      if (lane & i) continue;
      const int o = lane ^ i;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int x = lane * 4 + j, y = o * 4 + j;
        const float a = v[x], b = v[y];
        v[x] = a + b;
        v[y] = a - b;
      }
    }
  }
}

}  // namespace kvq_impl

// Quantize 32 consecutive fp16 values into `bits` uint32 words plus one half scale.
// `in` and `out` may alias nothing; the 32 words are written whole.
template <int bits>
__device__ __forceinline__ void kvq_quant_group(const half* __restrict__ in, uint32_t* __restrict__ gw,
                                                half* __restrict__ out_scale) {
  constexpr int m = 1 << (bits - 1);
  constexpr float mf = (float)m;
  float v[32];
#pragma unroll
  for (int i = 0; i < 32; i++) v[i] = __half2float(in[i]);
  kvq_impl::cq_hadamard32(v);
#pragma unroll
  for (int i = 0; i < 32; i++) v[i] = __fmul_rn(v[i], kvq_impl::kR32);

  // Group absmax. The reference adds 1e-10 so an all-zero group cannot divide by zero.
  float s = fabsf(v[0]);
#pragma unroll
  for (int i = 1; i < 32; i++) s = fmaxf(s, fabsf(v[i]));
  s += 1e-10f;
  const float inv_s = 1.0f / s;

#pragma unroll
  for (int lane = 0; lane < 8; lane++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      // Midpoint grid: centroid q sits at (2q+1)/2^bits - 1, so round-toward-zero of v*2^(bits-1)
      // + 2^(bits-1) places v exactly on a centroid. The reference uses __float2int_rd, i.e. a
      // truncation, and the argument is never negative, so this is floor.
      // __fmul_rn, not `v * inv_s`: helios compiles with --use_fast_math, which lets this multiply
      // contract into the following fmaf as fmaf(v, inv_s * mf, mf). That is a different rounding,
      // and a code pushed across a centroid boundary is a full quantization step of error in one
      // value - enough to break the bit-identity with the reference that this path is verified on.
      const float t = v[lane * 4 + j] * inv_s;
      int qv = __float2int_rd(fmaf(t, mf, mf));
      qv = max(min(qv, (1 << bits) - 1), 0);
      kvq_impl::cq_pack<bits>(gw, lane, j, (uint32_t)qv);
    }
  }
  *out_scale = __float2half_rn(s);
}

// Dequantize one 32-value group into 32 fp16 values. Exact integer inverse of the transform.
template <int bits>
__device__ __forceinline__ void kvq_dequant_group(const uint32_t* __restrict__ gw, half scale,
                                                  half* __restrict__ out) {
  constexpr int m = 1 << (bits - 1);
  int t[32];
#pragma unroll
  for (int lane = 0; lane < 8; lane++) {
    const int i = lane * 4;
    const int a = (int)kvq_impl::cq_unpack<bits>(gw, lane, 0);
    const int b = (int)kvq_impl::cq_unpack<bits>(gw, lane, 1);
    const int c = (int)kvq_impl::cq_unpack<bits>(gw, lane, 2);
    const int d = (int)kvq_impl::cq_unpack<bits>(gw, lane, 3);
    const int s0 = a + b, d0 = a - b, s1 = c + d, d1 = c - d;
    t[i]     = s0 + s1;
    t[i + 1] = d0 + d1;
    t[i + 2] = s0 - s1;
    t[i + 3] = d0 - d1;
  }
  // Same three rounds as the quantizer, in int32: exact, and the codes are <= 2^bits - 1 so
  // |t| <= 32 * 255.
#pragma unroll
  for (int i = 1; i < 8; i <<= 1) {
#pragma unroll
    for (int lane = 0; lane < 8; lane++) {
      if (lane & i) continue;
      const int o = lane ^ i;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int x = lane * 4 + j, y = o * 4 + j;
        const int p = t[x], q = t[y];
        t[x] = p + q;
        t[y] = p - q;
      }
    }
  }
  // H32 * 1 = 32 * e_0, so the (m - 0.5) centroid offset only survives in component 0.
  t[0] -= 32 * m - 16;
  const float sm = __half2float(scale) * (kvq_impl::kR32 / (float)m);
#pragma unroll
  for (int i = 0; i < 32; i++) out[i] = __float2half_rn((float)t[i] * sm);
}

#endif  // __CUDACC__

}}  // namespace helios::attn
