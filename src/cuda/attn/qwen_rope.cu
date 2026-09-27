// Partial NEOX RoPE. See qwen_rope.cuh for the parameter derivation.
#include "qwen_rope.cuh"

#include <cuda_fp16.h>

namespace helios { namespace attn {

namespace {

// One warp per (row, head); the rotary block is at most 64 wide here, so a warp with each lane
// handling two rotation pairs is enough without shared memory.
__global__ void rope_qk_kernel(void* __restrict__ q, void* __restrict__ k, int rows, int n_q_heads,
                               int n_kv_heads, int head_dim, int rot, const int* __restrict__ pos,
                               float theta) {
  const int half_rot = rot >> 1;
  const int total = rows * (n_q_heads + n_kv_heads);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int row = idx / (n_q_heads + n_kv_heads);
  const int head = idx % (n_q_heads + n_kv_heads);
  const bool is_q = head < n_q_heads;
  const int h = is_q ? head : head - n_q_heads;
  half* base = (is_q ? (half*)q : (half*)k) + ((size_t)row * (is_q ? n_q_heads : n_kv_heads) + h) * head_dim;

  const float p = (float)pos[row];
  for (int i = 0; i < half_rot; i++) {
    const float inv = powf(theta, -2.0f * (float)i / (float)rot);
    const float ang = p * inv;
    // precise intrinsics: these angles feed every attention logit, and the cost is
    // negligible next to the surrounding GEMMs
    const float c = cosf(ang), s = sinf(ang);
    const float x1 = __half2float(base[i]);
    const float x2 = __half2float(base[i + half_rot]);
    base[i] = __float2half_rn(x1 * c - x2 * s);
    base[i + half_rot] = __float2half_rn(x1 * s + x2 * c);
  }
}

}  // namespace

void rope_qk_partial_neox(void* q, void* k, int rows, int n_q_heads, int n_kv_heads, int head_dim,
                          int rotary_dim, const int* pos, float theta, Stream s) {
  if (rows <= 0 || rotary_dim <= 0) return;
  const int total = rows * (n_q_heads + (k ? n_kv_heads : 0));
  const int threads = 128;
  rope_qk_kernel<<<(total + threads - 1) / threads, threads, 0, s>>>(q, k, rows, n_q_heads,
                                                                    k ? n_kv_heads : 0, head_dim,
                                                                    rotary_dim, pos, theta);
}

}} // namespace helios::attn
