// MTP draft head for Qwen3.8-Flash-Next. See mtp.cuh for the contract and the upstream caveat.
#include "engine/mtp.cuh"

#include "core/device.hpp"
#include <math_constants.h>

#include "engine/glue.cuh"
#include "engine/glue2.cuh"
#include "cuda/aux/gr_mix.cuh"
#include "cuda/quant/exl3_gemm.cuh"

namespace helios {

namespace {

// out[r,h,d] = x[r,h,d] * rsqrt(mean_d(x^2) + eps) * (w[h,d] + 1)
__global__ void tap_norm_k(const float* __restrict__ x, const half* __restrict__ w,
                           float* __restrict__ out, int R, int H, int D, float eps) {
  const int r = blockIdx.x, h = blockIdx.y;
  const float* xr = x + ((size_t)r * H + h) * D;
  float* orow = out + ((size_t)r * H + h) * D;
  const half* wr = w + (size_t)h * D;
  float ss = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) { const float v = xr[d]; ss += v * v; }
  __shared__ float red[32];
  for (int off = 16; off > 0; off >>= 1) {
    ss += __shfl_down_sync(0xffffffffu, ss, off);
  }
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (threadIdx.x == 0) red[0] = rsqrtf(v / (float)D + eps);
  }
  __syncthreads();
  const float inv = red[0];
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    orow[d] = xr[d] * inv * (__half2float(wr[d]) + 1.0f);
  }
}

// streams[r,h,d] = hidden[r,h,d] + emb[r,d]
__global__ void add_emb_bcast_k(const half* __restrict__ hidden, const half* __restrict__ emb,
                                float* __restrict__ streams, int R, int H, int D) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = (size_t)R * H * D;
  if (i >= total) return;
  const int d = (int)(i % D);
  const int r = (int)(i / ((size_t)H * D));
  streams[i] = __half2float(hidden[i]) + __half2float(emb[(size_t)r * D + d]);
}

// streams[r,h,d] = mean_h(hidden[r,h,d]) + emb[r,d]  (the stream_tap = false alternative)
__global__ void add_emb_mean_k(const half* __restrict__ hidden, const half* __restrict__ emb,
                               float* __restrict__ streams, int R, int H, int D) {
  const int r = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= D) return;
  float acc = 0.f;
  for (int h = 0; h < H; h++) acc += __half2float(hidden[((size_t)r * H + h) * D + d]);
  acc /= (float)H;
  const float e = __half2float(emb[(size_t)r * D + d]);
  for (int h = 0; h < H; h++) streams[((size_t)r * H + h) * D + d] = acc + e;
}

// Single-block argmax over the fp32 logits (vocab is small enough for one block's grid-stride).
__global__ void argmax_f32_k(const float* __restrict__ x, int n, int* __restrict__ out) {
  float best = -CUDART_INF_F;
  int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = x[i];
    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
  }
  __shared__ float sb[32];
  __shared__ int si[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, best, off);
    const int oi = __shfl_down_sync(0xffffffffu, bi, off);
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
  }
  if (lane == 0) { sb[warp] = best; si[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    const int nw = (blockDim.x + 31) >> 5;
    best = (lane < nw) ? sb[lane] : -CUDART_INF_F;
    bi = (lane < nw) ? si[lane] : 0;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, best, off);
      const int oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) *out = bi;
  }
}

// Row-wise argmax: one block per row of a (rows x n) logits buffer, same tie rule as
// argmax_f32_k (lowest index wins). The spec path needs one int per verified position, not a
// vocab-sized row copied back per position.
__global__ void row_argmax_k(const float* __restrict__ x, int rows, int n, int* __restrict__ out) {
  const float* xr = x + (size_t)blockIdx.x * n;
  float best = -CUDART_INF_F;
  int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = xr[i];
    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
  }
  __shared__ float sb[32];
  __shared__ int si[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, best, off);
    const int oi = __shfl_down_sync(0xffffffffu, bi, off);
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
  }
  if (lane == 0) { sb[warp] = best; si[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    const int nw = (blockDim.x + 31) >> 5;
    best = (lane < nw) ? sb[lane] : -CUDART_INF_F;
    bi = (lane < nw) ? si[lane] : 0;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, best, off);
      const int oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) out[blockIdx.x] = bi;
  }
}


}  // namespace

void row_argmax(const float* logits, int rows, int vocab, int* out, cudaStream_t s) {
  if (rows <= 0) return;
  row_argmax_k<<<(unsigned)rows, 1024, 0, s>>>(logits, rows, vocab, out);
  HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

void mtp_tap_norm(const float* stack, const half* norm_w, float* out, int n, int H, int D, float eps,
                  cudaStream_t s) {
  if (n <= 0) return;
  dim3 grid((unsigned)n, (unsigned)H);
  tap_norm_k<<<grid, 256, 0, s>>>(stack, norm_w, out, n, H, D, eps);
  HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

bool mtp_scratch_init(MtpScratch& sc, const Config& cfg, int max_n, int device) {
  auto A = [&](size_t bytes) { return Engine::instance().gpu(device).alloc(bytes, 256); };
  const int D = cfg.hidden, H = cfg.hc_mult;
  sc.stack_ = (float*)A((size_t)max_n * H * D * 4);
  sc.normed_ = (half*)A((size_t)max_n * H * D * 2);
  sc.hidden_ = (half*)A((size_t)max_n * H * D * 2);
  sc.emb16_ = (half*)A((size_t)max_n * D * 2);
  sc.emb_norm_ = (half*)A((size_t)max_n * D * 2);
  sc.emb_fc_ = (half*)A((size_t)max_n * D * 2);
  sc.streams_ = (float*)A((size_t)max_n * H * D * 4);
  sc.tap_ = (float*)A((size_t)max_n * H * D * 4);
  sc.post_ = (float*)A((size_t)max_n * H * 4);
  sc.had_ = A((size_t)max_n * H * D * 2);
  sc.ids_dev_ = (int*)A((size_t)max_n * 4);
  sc.max_n = max_n;
  sc.ready = sc.stack_ && sc.streams_ && sc.had_ && sc.ids_dev_;
  if (!sc.ready) fprintf(stderr, "[mtp] scratch allocation failed\n");
  return sc.ready;
}

void mtp_draft_step(Model& m, MtpScratch& sc, AttnScratch& attn, MoeScratch& moe,
                    const float* trunk_streams, const int* ids, int n, int pos0, void* k_cache,
                    void* v_cache, float* tap_out, int* draft_ids, float* logits, bool stream_tap,
                    cudaStream_t s, bool compute_head, const half* emb16_in) {
  const Config& c = m.cfg;
  const int D = c.hidden, H = c.hc_mult;
  MTPWeights& w = m.mtp;

  // ---- input combine: streams = fc_hidden(tap_norm(trunk stack)) + fc_embedding(norm(embed)) ----
  mtp_tap_norm(trunk_streams, w.pre_fc_norm_hidden, sc.stack_, n, H, D, c.rms_eps, s);
  glue::cast_f32_f16(sc.stack_, sc.normed_, (size_t)n * H * D, s);
  {
    exl3::GroupWords hw{(const uint16_t*)w.fc_hidden.trellis, w.fc_hidden.suh, w.fc_hidden.svh,
                        w.fc_hidden.mul1};
    exl3::linear(sc.hidden_, sc.normed_, hw, n * H, D, D, w.fc_hidden.K, /*y_fp32=*/false, s, sc.had_);
  }
  // `emb16_in` is the pipeline handoff: the embedding table sits on layer 0's card, this head on the
  // last layer's card, and the caller has already gathered these n rows where the table lives. It
  // must NOT be gathered here in that case - a pointer from the other card's context is not
  // dereferenceable from this stream, and the failure would be an illegal access at best.
  if (!emb16_in) {
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.ids_dev_, ids, (size_t)n * 4, cudaMemcpyHostToDevice, s));
    glue::embed_gather(m.embed, sc.ids_dev_, n, sc.emb16_, D, s);
  }
  // pre_fc_norm_embedding is a plain RMSNorm with constant_bias 1.0: the same (w + 1) form as the
  // tap norm, taken over the hidden axis with a single "stream".
  glue::cast_f16_f32(sc.emb16_, sc.stack_, (size_t)n * D, s);
  mtp_tap_norm(sc.stack_, w.pre_fc_norm_embedding, sc.stack_, n, 1, D, c.rms_eps, s);
  glue::cast_f32_f16(sc.stack_, sc.emb_norm_, (size_t)n * D, s);
  {
    exl3::GroupWords ew{(const uint16_t*)w.fc_embedding.trellis, w.fc_embedding.suh,
                        w.fc_embedding.svh, w.fc_embedding.mul1};
    exl3::linear(sc.emb_fc_, sc.emb_norm_, ew, n, D, D, w.fc_embedding.K, /*y_fp32=*/false, s, sc.had_);
  }
  if (stream_tap) {
    add_emb_bcast_k<<<(unsigned)(((size_t)n * H * D + 255) / 256), 256, 0, s>>>(
        sc.hidden_, sc.emb_fc_, sc.streams_, n, H, D);
  } else {
    dim3 g((unsigned)n, (unsigned)((D + 255) / 256));
    add_emb_mean_k<<<g, 256, 0, s>>>(sc.hidden_, sc.emb_fc_, sc.streams_, n, H, D);
  }
  HELIOS_CUDA_CHECK(cudaPeekAtLastError());

  // ---- one full decoder block on the draft streams ----
  Layer& L = w.layer;
  // Both hc SITES carry the reference's post gate. GatedResidual's use_combine defaults to true
  // for a decoder block (only the logit mixer passes use_combine=False), and that gate is what
  // scales each sublayer's output per stream before it is added back into that stream. Skipping it
  // turns the block into a plain residual add, and because the block's OUTPUT STACK is the next
  // chain step's tap, the error propagates into every later draft - n=1 does not excuse it.
  aux::gr_mix(sc.streams_, L.hc_attn.norm, L.hc_attn.down, L.hc_attn.up, L.hc_attn.inject, n, H, D,
              c.hc_rank, c.rms_eps, sc.stack_, sc.post_, s);
  glue::cast_f32_f16(sc.stack_, sc.normed_, (size_t)n * D, s);
  attn_layer(L.attn, c, attn, sc.normed_, sc.stack_, k_cache, v_cache, n, pos0, s);
  aux::gr_apply(sc.streams_, sc.stack_, sc.post_, n, H, D, s);

  aux::gr_mix(sc.streams_, L.hc_mlp.norm, L.hc_mlp.down, L.hc_mlp.up, L.hc_mlp.inject, n, H, D,
              c.hc_rank, c.rms_eps, sc.stack_, sc.post_, s);
  glue::cast_f32_f16(sc.stack_, sc.normed_, (size_t)n * D, s);
  moe_layer(L.moe, c, moe, sc.normed_, sc.stack_, L.index, /*mtp=*/true, n, s);
  aux::gr_apply(sc.streams_, sc.stack_, sc.post_, n, H, D, s);

  // The draft chain's output IS the pre-mixer stack (it is the next step's tap).
  if (tap_out) {
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(tap_out, sc.streams_, (size_t)n * H * D * 4,
                                      cudaMemcpyDeviceToDevice, s));
  }

  // ---- its own combine-less mixer, then the shared head on the last row ----
  if (!compute_head) return;   // prefill's cache-fill only needs the attention above
  aux::gr_mix(sc.streams_, w.mixer.norm, w.mixer.down, w.mixer.up, nullptr, n, H, D, c.hc_rank,
              c.rms_eps, sc.stack_, nullptr, s);
  glue::cast_f32_f16(sc.stack_, sc.normed_, (size_t)n * D, s);
  {
    exl3::GroupWords hw{(const uint16_t*)m.lm_head.trellis, m.lm_head.suh, m.lm_head.svh,
                        m.lm_head.mul1};
    exl3::gemm(logits, sc.normed_ + (size_t)(n - 1) * D, hw, 1, c.vocab, D, m.lm_head.K,
               /*y_fp32=*/true, s, sc.had_);
  }
  if (draft_ids) {
    argmax_f32_k<<<1, 1024, 0, s>>>(logits, c.vocab, draft_ids);
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
  }
}

}  // namespace helios
