// Quantized KV cache: quantize-on-write and materialize-to-fp16. See kv_quant.cuh for the layout
// and the algorithm, and for the reference (exllamav3_cache/quant.py + q_cache_kernels.cuh) this
// ports.
#include "kv_quant.cuh"

namespace helios { namespace attn {

namespace {

// One thread per 32-value group, both tensors in one launch: the first half of the grid quantizes
// K, the second half V. A group is 32 values, so a chunk of 1024 tokens on this model is
// 1024 * 16 * 2 = 32768 threads, i.e. 128 blocks of 256 - one launch, a few microseconds, once per
// layer per chunk.
template <int bits>
__global__ void kvq_write_kernel(const half* __restrict__ k_in, const half* __restrict__ v_in,
                                 uint32_t* __restrict__ kq, half* __restrict__ ks,
                                 uint32_t* __restrict__ vq, half* __restrict__ vs, int n_groups_per_tensor,
                                 int groups_per_token, int pos0) {
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= 2 * n_groups_per_tensor) return;
  const half* src;
  uint32_t* dst_q;
  half* dst_s;
  if (task < n_groups_per_tensor) {
    src = k_in; dst_q = kq; dst_s = ks;
  } else {
    task -= n_groups_per_tensor;
    src = v_in; dst_q = vq; dst_s = vs;
  }
  const int row = task / groups_per_token;
  const int g = task - row * groups_per_token;
  const int t = pos0 + row;
  // A thread owns exactly one group's `bits` words, so the words can be written whole after being
  // zeroed here - no cross-lane atomics, which is what the reference needs because it packs eight
  // lanes into one group.
  uint32_t* gw = dst_q + (size_t)t * groups_per_token * bits + (size_t)g * bits;
#pragma unroll
  for (int i = 0; i < bits; i++) gw[i] = 0;
  kvq_quant_group<bits>(src + (size_t)row * groups_per_token * 32 + (size_t)g * 32, gw,
                        dst_s + (size_t)t * groups_per_token + g);
}

template <int bits>
__global__ void kvq_dequant_kernel(const uint32_t* __restrict__ q, const half* __restrict__ s,
                                   half* __restrict__ out, int n_groups_per_tensor,
                                   int groups_per_token, int pos0) {
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= n_groups_per_tensor) return;
  const int row = task / groups_per_token;
  const int g = task - row * groups_per_token;
  const int t = pos0 + row;
  kvq_dequant_group<bits>(q + (size_t)t * groups_per_token * bits + (size_t)g * bits,
                          s[(size_t)t * groups_per_token + g],
                          out + (size_t)row * groups_per_token * 32 + (size_t)g * 32);
}

template <int bits>
void launch_write(const void* k, const void* v, KvQuant kc, KvQuant vc, int n, int token_dim,
                  int pos0, Stream s) {
  const int gpt = token_dim / 32;
  const int total = n * gpt;
  const int threads = 256;
  kvq_write_kernel<bits><<<(total * 2 + threads - 1) / threads, threads, 0, s>>>(
      (const half*)k, (const half*)v, (uint32_t*)kc.q, (half*)kc.s, (uint32_t*)vc.q, (half*)vc.s,
      total, gpt, pos0);
  cuda_check(cudaPeekAtLastError());
}

template <int bits>
void launch_dequant(const void* q, const void* s, void* out, int n, int token_dim, int pos0,
                    Stream st) {
  const int gpt = token_dim / 32;
  const int total = n * gpt;
  const int threads = 256;
  kvq_dequant_kernel<bits><<<(total + threads - 1) / threads, threads, 0, st>>>(
      (const uint32_t*)q, (const half*)s, (half*)out, total, gpt, pos0);
  cuda_check(cudaPeekAtLastError());
}

}  // namespace

#define KVQ_DISPATCH(fn, bits, ...)                    \
  switch (bits) {                                      \
    case 2: fn<2>(__VA_ARGS__); break;                 \
    case 3: fn<3>(__VA_ARGS__); break;                 \
    case 4: fn<4>(__VA_ARGS__); break;                 \
    case 5: fn<5>(__VA_ARGS__); break;                 \
    case 6: fn<6>(__VA_ARGS__); break;                 \
    case 7: fn<7>(__VA_ARGS__); break;                 \
    default: fn<8>(__VA_ARGS__); break;                \
  }

void kvq_write(const void* k, const void* v, KvQuant kc, KvQuant vc, int n, int token_dim, int bits,
               int pos0, Stream s) {
  if (n <= 0) return;
  KVQ_DISPATCH(launch_write, bits, k, v, kc, vc, n, token_dim, pos0, s)
}

void kvq_dequant(const void* q, const void* s, void* out, int n, int token_dim, int bits, int pos0,
                 Stream st) {
  if (n <= 0) return;
  KVQ_DISPATCH(launch_dequant, bits, q, s, out, n, token_dim, pos0, st)
}

#undef KVQ_DISPATCH

}}  // namespace helios::attn
