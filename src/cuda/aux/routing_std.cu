// Standard softmax router, ported from exllamav3_ext/routing.cu (routing_std_topk_kernel and the
// warp_reduce_best_f32 helper it uses). Raw pointers, explicit stream, no torch types.
#include "routing_std.cuh"
#include "helios_shim.cuh"

namespace helios { namespace aux {

namespace {

constexpr int MAX_ROUTER_THREADS = 512;

// Argmax over (key, payload, idx) triples across a warp. On sm_80+ the key is encoded as a
// monotonic unsigned int so the hardware max-reduce finds the winner in one instruction, then the
// winner's fields are fetched from the lowest tied lane.
__device__ __forceinline__ void warp_reduce_best_f32(float& key, float& payload, int& idx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned int ku = __float_as_uint(key);
  ku = (ku & 0x80000000u) ? ~ku : (ku | 0x80000000u);
  unsigned int m = __reduce_max_sync(0xffffffffu, ku);
  int src = __ffs(__ballot_sync(0xffffffffu, ku == m)) - 1;
  key = __shfl_sync(0xffffffffu, key, src);
  payload = __shfl_sync(0xffffffffu, payload, src);
  idx = __shfl_sync(0xffffffffu, idx, src);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float ok = __shfl_down_sync(0xffffffffu, key, offset);
    float op = __shfl_down_sync(0xffffffffu, payload, offset);
    int oi = __shfl_down_sync(0xffffffffu, idx, offset);
    if (ok > key) { key = ok; payload = op; idx = oi; }
  }
  key = __shfl_sync(0xffffffffu, key, 0);
  payload = __shfl_sync(0xffffffffu, payload, 0);
  idx = __shfl_sync(0xffffffffu, idx, 0);
#endif
}

// One block per row: iterative argmax with a shared-memory merge when the expert count exceeds a
// warp, then the softmax over the selected logits.
__global__ __launch_bounds__(MAX_ROUTER_THREADS) void routing_std_topk_kernel(
    const half* __restrict__ scores, int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights, int num_experts, int K) {
  const int row = blockIdx.x;
  const int t = threadIdx.x;
  const int lane_id = t % 32;
  const int warp_id = t / 32;
  const int num_warps = CEIL_DIVIDE(num_experts, 32);

  scores += (size_t)num_experts * row;
  topk_indices += (size_t)K * row;
  topk_weights += (size_t)K * row;

  extern __shared__ unsigned char sh[];
  float* sh_key = reinterpret_cast<float*>(sh);
  int* sh_idx = reinterpret_cast<int*>(sh_key + num_warps * K);
  float* max_red = reinterpret_cast<float*>(sh_idx + num_warps * K);

  const bool mask = t < num_experts;
  float logit = mask ? __half2float(scores[t]) : -1.0e30f;
  float max_logit = warp_reduce_max_f(logit);
  max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
  if (num_warps > 1) {
    if (lane_id == 0) max_red[warp_id] = max_logit;
    __syncthreads();
    max_logit = lane_id < num_warps ? max_red[lane_id] : -1.0e30f;
    max_logit = warp_reduce_max_f(max_logit);
    max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
  }

  float key = logit, payload = logit;
  int idx = mask ? t : -1;
  for (int k = 0; k < K; ++k) {
    float bk = key, bp = payload;
    int bi = idx;
    warp_reduce_best_f32(bk, bp, bi);
    if (lane_id == k) {
      sh_key[warp_id * K + k] = bk;
      sh_idx[warp_id * K + k] = bi;
    }
    if (idx == bi) key = -1.0e30f;
  }
  __syncthreads();

  int num_candidates = num_warps * K;
  while (num_candidates > 32) {
    const int stage_warps = CEIL_DIVIDE(num_candidates, 32);
    if (warp_id < stage_warps) {
      const int pos = t;
      key = pos < num_candidates ? sh_key[pos] : -1.0e30f;
      payload = key;
      idx = pos < num_candidates ? sh_idx[pos] : -1;
      for (int k = 0; k < K; ++k) {
        float bk = key, bp = payload;
        int bi = idx;
        warp_reduce_best_f32(bk, bp, bi);
        if (lane_id == k) {
          sh_key[warp_id * K + k] = bk;
          sh_idx[warp_id * K + k] = bi;
        }
        if (idx == bi) key = -1.0e30f;
      }
    }
    __syncthreads();
    num_candidates = stage_warps * K;
  }

  if (warp_id == 0) {
    key = lane_id < num_candidates ? sh_key[lane_id] : -1.0e30f;
    payload = key;
    idx = lane_id < num_candidates ? sh_idx[lane_id] : -1;
    for (int k = 0; k < K; ++k) {
      float bk = key, bp = payload;
      int bi = idx;
      warp_reduce_best_f32(bk, bp, bi);
      if (lane_id == k) {
        sh_key[k] = expf(bp - max_logit);      // softmax is over the selected top-k only
        sh_idx[k] = bi;
      }
      if (idx == bi) key = -1.0e30f;
    }
    __syncwarp();
    float e = lane_id < K ? sh_key[lane_id] : 0.0f;
    const float sum = warp_reduce_sum_first_k(e, K) + 1e-20f;
    e /= sum;
    if (lane_id < K) {
      topk_indices[lane_id] = (int64_t)sh_idx[lane_id];
      topk_weights[lane_id] = __float2half_rn(e);
    }
  }
}

}  // namespace

void routing_std_logits(const half* scores, int64_t* topk_indices, half* topk_weights, int bsz,
                        int num_experts, int K, Stream s) {
  if (bsz <= 0 || num_experts <= 0 || K <= 0) return;
  const int num_warps = (num_experts + 31) / 32;
  const size_t smem = ((size_t)num_warps * K * 2 + num_warps) * 4;   // sh_key, sh_idx, max_red
  const int threads = num_warps * 32 <= MAX_ROUTER_THREADS ? num_warps * 32 : MAX_ROUTER_THREADS;
  routing_std_topk_kernel<<<bsz, threads, smem, s>>>(scores, topk_indices, topk_weights,
                                                     num_experts, K);
}

}} // namespace helios::aux
