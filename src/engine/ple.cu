// PLE: host n-gram hashing, the ported ngram_dequant kernel, and the kernels this layer needs that
// have no upstream CUDA equivalent (the signed-sqrt gate, the dilated depthwise conv, the injection).
#include "engine/ple.cuh"
#include "core/device.hpp"

#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "engine/glue2.cuh"
#include "cuda/aux/norm.cuh"

namespace helios {

namespace {

constexpr int kNgramRowDim = 160;      // ple_embed_dim / ngram_heads
constexpr uint32_t kMul1 = 0x83DCD12Du;

// ---- n-gram row decode: ported verbatim from exllamav3_ext/ngram.cu (ngram_dequant_kernel) ----
__global__ __launch_bounds__(kNgramRowDim) void ngram_dequant_kernel(
    const int16_t* __restrict__ packed, const int32_t* __restrict__ heads,
    const half* __restrict__ bias, half* __restrict__ out, const int K, const int words) {
  const int r = blockIdx.x;
  const int i = threadIdx.x;

  extern __shared__ uint16_t sw[];
  if (i < words) sw[i] = (uint16_t)packed[(size_t)r * words + i];
  __syncthreads();

  const float scale = __half2float(__ushort_as_half(sw[0]));

  // stream bit m of element i lives at ring position ((i - m/K) mod ROW_DIM) * K + m%K
  uint32_t state = 0;
#pragma unroll 4
  for (int m = 0; m < 16; ++m) {
    int pos = i - m / K;
    if (pos < 0) pos += kNgramRowDim;
    const int sb = pos * K + m % K;
    const uint32_t bit = (sw[1 + (sb >> 4)] >> (sb & 15)) & 1u;
    state |= bit << m;
  }

  const uint32_t prod = state * kMul1;
  const float h = 1024.0f + (float)((prod & 0xff) + ((prod >> 8) & 0xff) + ((prod >> 16) & 0xff) +
                                    ((prod >> 24) & 0xff));
  const float k_inv = __half2float(__ushort_as_half(0x1eee));
  const float k_bias = __half2float(__ushort_as_half(0xc931));
  const float cb = __half2float(__float2half_rn(h * k_inv + k_bias));

  const float b = __half2float(bias[(size_t)heads[r] * kNgramRowDim + i]);
  out[(size_t)r * kNgramRowDim + i] = __float2half_rn(cb * scale + b);
}

// ---- per-(row,stream) query.key dot: the reference's bmm ----
// Per-stream dot product: out[r,h] = sum_d q[r,h,d] * k[r,h,d].
//
// The reduction must cover every thread in the block, not one warp: with a block of 256 the loop
// strides by 256, so each lane holds a partial over disjoint elements and a warp-local shuffle
// would silently drop 7/8 of the sum. Warp partials go to shared memory, then warp 0 finishes.
__global__ void ple_dot_kernel(const float* __restrict__ q, const float* __restrict__ k,
                               float* __restrict__ out, int heads, int dim) {
  const int r = blockIdx.x, h = blockIdx.y;
  const float* qr = q + ((size_t)r * heads + h) * dim;
  const float* kr = k + ((size_t)r * heads + h) * dim;
  float acc = 0.f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) acc = fmaf(qr[d], kr[d], acc);
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  __shared__ float warps[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarp = (blockDim.x + 31) >> 5;
  if (lane == 0) warps[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < nwarp) ? warps[lane] : 0.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) out[(size_t)r * heads + h] = v;   // one writer: no atomic needed
  }
}

// ---- gate: sigmoid(sign(x)*sqrt(max(|x|,1e-6))) * value, x = scale * dot ----
// One row of the output per (token, stream): row rh is token rh/heads, stream rh%heads. The value
// vector is per TOKEN ([n, dim]) and is broadcast across that token's streams, so indexing it by the
// flattened row - as an earlier revision did - reads past the end of the buffer for every stream
// after the first and silently drops 3/4 of the layer's contribution.
__global__ void ple_gate_kernel(const float* __restrict__ gate, const half* __restrict__ value,
                                float* __restrict__ gated, float scale, int heads, int dim) {
  const int rh = blockIdx.y;                 // flattened [token, stream] row
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  const int r = rh / heads;                  // token index into value
  const float x = gate[rh] * scale;
  const float a = sqrtf(fmaxf(fabsf(x), 1e-6f));
  const float ss = x > 0.0f ? a : (x < 0.0f ? -a : 0.0f);
  const float s = 1.0f / (1.0f + __expf(-ss));
  gated[(size_t)rh * dim + d] = s * __half2float(value[(size_t)r * dim + d]);
}

// ---- dilated depthwise causal conv + SiLU (fp16 in/out) ----
__global__ void ple_conv_kernel(const half* __restrict__ x, const half* __restrict__ w,
                                const half* __restrict__ state, half* __restrict__ out, int n,
                                int ch, int dilation, int taps, int win) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)n * ch) return;
  const int t = (int)(i / ch), d = (int)(i % ch);
  // Tap order matters: the reference runs F.conv1d, which is a CROSS-CORRELATION, over the padded
  // stream xt = [state(win) | x(n)]. So out[t] = sum_k w[k] * xt[t + dilation*k], pairing w[0] with
  // the OLDEST carried input (xt[0]) and w[taps-1] with the newest. Indexing x[t - dilation*k]
  // instead pairs the taps backwards, which silently reverses the filter.
  float acc = 0.f;
  for (int k = 0; k < taps; k++) {
    const int idx = t + dilation * k - win;          // position in the global input stream
    const float v = (idx >= 0) ? __half2float(x[(size_t)idx * ch + d])
                               : __half2float(state[(size_t)d * win + (win + idx)]);
    acc = fmaf(__half2float(w[(size_t)d * taps + k]), v, acc);
  }
  const half hv = __float2half_rn(acc);       // silu on the rounded fp16 value, as the reference
  out[i] = __float2half_rn(__half2float(hv) / (1.0f + __expf(-__half2float(hv))));
}

// State writeback: new[d][j] = (n + j < win) ? state[d][n + j] : x[j + n - win]. One block covers
// all win columns of 64 channels; the staging forces every read before any write, so the n < win
// case (decode) cannot race within a channel, and channels are independent.
__global__ void ple_conv_state_kernel(const half* __restrict__ x, half* __restrict__ state, int n,
                                      int ch, int win) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = threadIdx.y;
  __shared__ half tmp[64][9];
  if (j < win) {
    const int src = n + j - win;
    tmp[threadIdx.x][j] = (src >= 0) ? x[(size_t)src * ch + d] : state[(size_t)d * win + n + j];
  }
  __syncthreads();
  if (j < win) state[(size_t)d * win + j] = tmp[threadIdx.x][j];
}

// streams[r][h][d] += gated[r][h][d] + conv_out[r][h*dim + d]
__global__ void ple_inject_kernel(float* __restrict__ streams, const float* __restrict__ gated,
                                  const half* __restrict__ conv, long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  // gated is [row, H, D] and conv is [row, H*D] with the same linear order, so both index by i
  streams[i] += gated[i] + __half2float(conv[i]);
}

// ---- shard-bucketed gather -> caller row order ----
// The H2D side of the n-gram row gather groups rows by shard (one contiguous run of the pinned
// staging buffer per shard, hence one memcpy per shard), but everything downstream - the dequant
// kernel's `heads[r]`, and the [n, heads*dim] layout of `decoded` - is indexed by the caller's row
// i = token*heads + head. This puts the staged rows back where the caller expects them: one block
// per staged row, so each block reads 82 contiguous bytes and writes 82 contiguous bytes somewhere
// else in the buffer - 1.34 MB moved twice, which is a rounding error next to the H2D it feeds.
__global__ void ple_rows_scatter_kernel(int16_t* __restrict__ dst, const int16_t* __restrict__ src,
                                        const int* __restrict__ slot, int words) {
  const long r = blockIdx.x;                       // staged slot
  const int16_t* in = src + r * words;
  int16_t* out = dst + (long)slot[r] * words;
  for (int i = threadIdx.x; i < words; i += blockDim.x) out[i] = in[i];
}

}  // namespace

void ple_ngram_ids(const int64_t* history, int n_hist, int n_ctx, int64_t eos, int ngram_size,
                   int heads_per_ngram, const int64_t* multipliers, const int64_t* vocab_sizes,
                   const int64_t* offsets, int64_t* out_ids) {
  const int n_out = n_hist - n_ctx;
  if (n_out <= 0) return;
  const int nh = (ngram_size - 1) * heads_per_ngram;
  int64_t prev_eos = -1;
  for (int p = 0; p < n_hist; p++) {
    if (p > 0 && history[p - 1] == eos) prev_eos = p - 1;
    const int64_t seg_start = prev_eos + 1;
    uint64_t mixed = (uint64_t)history[p] * (uint64_t)multipliers[0];
    int k = 0;
    for (int s = 1; s < ngram_size; s++) {
      const int64_t src = (p - s >= seg_start) ? history[p - s] : eos;
      mixed ^= (uint64_t)src * (uint64_t)multipliers[s];
      const int lo = (s - 1) * heads_per_ngram;
      for (int h = lo; h < lo + heads_per_ngram; h++) {
        int64_t m = (int64_t)(mixed % (uint64_t)vocab_sizes[h]);
        if (m < 0) m += vocab_sizes[h];
        if (p >= n_ctx) out_ids[(size_t)(p - n_ctx) * nh + k] = m + offsets[h];
        k++;
      }
    }
  }
}

void ngram_dequant_f16(const void* packed, const int* heads, const half* bias, half* out, int U,
                       int K, int words, cudaStream_t s) {
  if (U <= 0) return;
  ngram_dequant_kernel<<<U, kNgramRowDim, (size_t)words * sizeof(uint16_t), s>>>(
      (const int16_t*)packed, (const int32_t*)heads, bias, out, K, words);
}

void ple_gate(const float* gate, const void* value, float* gated, float gate_scale, int rows,
              int heads, int dim, cudaStream_t s) {
  if (rows <= 0) return;
  // `gated` is [rows * heads, dim] - one row per (token, stream) - so the grid must cover rows*heads.
  // An earlier revision launched only `rows` row-blocks, writing just the first 1/heads of the
  // output and leaving the rest of the buffer untouched.
  dim3 grid((dim + 127) / 128, rows * heads);
  ple_gate_kernel<<<grid, 128, 0, s>>>(gate, (const half*)value, gated, gate_scale, heads, dim);
}

void ple_dilated_conv(const void* x, void* conv_state, const half* w, void* out, int n, int ch,
                      int dilation, cudaStream_t s) {
  if (n <= 0) return;
  const int taps = 4, win = (taps - 1) * dilation;
  const long total = (long)n * ch;
  ple_conv_kernel<<<(int)((total + 255) / 256), 256, 0, s>>>(
      (const half*)x, w, (const half*)conv_state, (half*)out, n, ch, dilation, taps, win);
  dim3 blk(64, win);
  ple_conv_state_kernel<<<(ch + 63) / 64, blk, 0, s>>>((const half*)x, (half*)conv_state, n, ch,
                                                       win);
}

bool ple_scratch_init(PleScratch& sc, const Config& cfg, int max_n, int device) {
  sc.max_n = max_n;
  sc.ch = cfg.hc_dim;
  sc.dim = cfg.hidden;
  const int rows = max_n * cfg.ngram_heads;
  auto A = [&](size_t bytes) { return Engine::instance().gpu(device).alloc(bytes, 256); };
  sc.key16 = A((size_t)max_n * sc.ch * 2);
  sc.value16 = A((size_t)max_n * sc.dim * 2);
  sc.key = A((size_t)max_n * cfg.hc_mult * sc.dim * 4);
  sc.query = A((size_t)max_n * cfg.hc_mult * sc.dim * 4);
  sc.gate = A((size_t)max_n * cfg.hc_mult * 4);
  sc.gated = A((size_t)max_n * cfg.hc_mult * sc.dim * 4);
  sc.normed = A((size_t)max_n * sc.ch * 2);
  sc.conv_out = A((size_t)max_n * sc.ch * 2);
  sc.rows = A((size_t)rows * 41 * 2);
  sc.row_stage = A((size_t)rows * 41 * 2);
  sc.row_slot = A((size_t)rows * 4);
  sc.row_ids = A((size_t)rows * 4);
  sc.head_ids = A((size_t)rows * 4);
  sc.decoded = A((size_t)rows * kNgramRowDim * 2);
  sc.conv_state_snap = A((size_t)sc.ch * (4 - 1) * cfg.ngram_size * 2);
  sc.conv_in_snap = A((size_t)max_n * sc.ch * 2);
  sc.bias_dev = A((size_t)cfg.ngram_heads * kNgramRowDim * 2);
  // The gather stages its rows through pinned memory so the per-shard H2D is a real DMA. Three
  // slots so a chunk can be staged while the previous chunks' copies are still in flight: 1.34 MB
  // of host RAM each at n = 1024, plus the 64 KB table that maps a staged run back to its row.
  for (int g = 0; g < PleScratch::kGatherSlots; g++) {
    HELIOS_CUDA_CHECK(cudaMallocHost(&sc.row_pin[g], (size_t)rows * 41 * 2));
    HELIOS_CUDA_CHECK(cudaMallocHost(&sc.row_slot_host[g], (size_t)rows * 4));
    HELIOS_CUDA_CHECK(cudaEventCreateWithFlags(&sc.row_pin_ev[g], cudaEventDisableTiming));
  }
  sc.row_shard_host = new int[rows];
  sc.shard_cnt = new int[cfg.ngram_shards];
  sc.shard_start = new int[cfg.ngram_shards];
  sc.shard_cur = new int[cfg.ngram_shards];
  sc.row_ids_host = new int[rows];
  sc.head_ids_host = new int[rows];
  sc.hash_host = new int64_t[rows];
  return sc.key16 && sc.decoded && sc.rows && sc.bias_dev && sc.hash_host && sc.row_pin[0] &&
         sc.row_slot_host[0] && sc.row_stage && sc.row_slot;
}

void ple_scratch_free(PleScratch& sc) {
  for (int g = 0; g < PleScratch::kGatherSlots; g++) {
    if (sc.row_pin[g]) cudaFreeHost(sc.row_pin[g]);
    if (sc.row_slot_host[g]) cudaFreeHost(sc.row_slot_host[g]);
    if (sc.row_pin_ev[g]) cudaEventDestroy(sc.row_pin_ev[g]);
  }
  delete[] sc.row_shard_host;
  delete[] sc.shard_cnt;
  delete[] sc.shard_start;
  delete[] sc.shard_cur;
  delete[] sc.row_ids_host;
  delete[] sc.head_ids_host;
  delete[] sc.hash_host;
  sc = PleScratch{};
}

void ple_layer(const PleWeights& w, const Config& cfg, PleScratch& sc, const half* embed,
               float* streams, const int64_t* history, int n_hist, int n_ctx, void* conv_state,
               int n, cudaStream_t s, bool capture) {
  const int H = cfg.hc_mult, D = cfg.hidden, ch = cfg.hc_dim, nh = cfg.ngram_heads;
  const int rows = n * nh;
  const int words = 1 + cfg.ngram_row_dim * cfg.ngram_K / 16;      // 41
  const size_t row_bytes = (size_t)words * 2;

  if (!sc.bias_ready && w.head_bias) {       // one-time: the bias lives on the host (mmap'd)
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.bias_dev, w.head_bias,
                                      (size_t)nh * kNgramRowDim * 2, cudaMemcpyHostToDevice, s));
    sc.bias_ready = true;
  }

  // 1) host hashing, then gather the rows the ids point at (shard = id / rows_per_shard).
  //
  // This used to be one 82-byte PAGEABLE cudaMemcpyAsync per row, 16,384 of them for a 1024-token
  // chunk, issued at layer 1 where the stream has nothing to hide them behind: the driver staged
  // every one of them. The table is 128 mmap'd shards, so the rows are instead bucketed by shard
  // here (the shard id falls out of the hash the host just computed), gathered into pinned memory
  // in shard order, and uploaded one contiguous run per shard - 125 transfers per 1024-token chunk
  // instead of 16,384, measured (the head offsets do not reach the last shards, so 125-126 of the
  // 128 buckets are non-empty). The staging buffer is in shard order, so a small scatter puts the
  // rows back in caller order before the dequant, which leaves `rows`, the row_ids/head_ids
  // uploads and the dequant itself untouched.
  ple_ngram_ids(history, n_hist, n_ctx, cfg.ngram_eos, cfg.ngram_size, cfg.heads_per_ngram,
                w.layer_multipliers, w.head_vocab_sizes, w.head_offsets, sc.hash_host);
  const int nshards = cfg.ngram_shards;
  const int64_t rps = cfg.ngram_rows_per_shard;
  memset(sc.shard_cnt, 0, sizeof(int) * nshards);
  for (int i = 0; i < rows; i++) {                  // row id -> (shard, local, head)
    const int64_t id = sc.hash_host[i];
    const int64_t shard = id / rps;
    sc.row_shard_host[i] = (int)shard;
    // An id outside the table falls back to shard 0, exactly as the per-row gather did.
    sc.shard_cnt[(shard >= 0 && shard < nshards) ? (int)shard : 0]++;
    sc.row_ids_host[i] = (int)(id - shard * rps);
    int h = nh - 1;
    while (h > 0 && id < w.head_offsets[h]) h--;
    sc.head_ids_host[i] = h;
  }
  // Reusing a slot whose copies have not landed yet would hand the DMA a buffer the host is
  // rewriting underneath it. With three slots this only ever waits on a chunk that is already
  // retired, and in the steady state the query says READY and nothing blocks.
  const int gslot = sc.row_pin_cur;
  sc.row_pin_cur = (gslot + 1) % PleScratch::kGatherSlots;
  if (cudaEventQuery(sc.row_pin_ev[gslot]) == cudaErrorNotReady)
    HELIOS_CUDA_CHECK(cudaEventSynchronize(sc.row_pin_ev[gslot]));
  char* pin = sc.row_pin[gslot];
  int* slot_host = sc.row_slot_host[gslot];
  int acc = 0;                                      // bucket offsets
  for (int sh = 0; sh < nshards; sh++) {
    sc.shard_start[sh] = acc;
    sc.shard_cur[sh] = acc;
    acc += sc.shard_cnt[sh];
  }
  for (int i = 0; i < rows; i++) {                  // gather into pinned, shard by shard
    const int64_t shard = sc.row_shard_host[i];
    const int64_t local = sc.row_ids_host[i];
    const int b = (shard >= 0 && shard < nshards) ? (int)shard : 0;
    const char* src = (shard >= 0 && shard < nshards) ? w.shards[shard] : w.shards[0];
    if (local >= 0 && local < rps) src += (size_t)local * row_bytes;
    const int j = sc.shard_cur[b]++;
    slot_host[j] = i;
    memcpy(pin + (size_t)j * row_bytes, src, row_bytes);
  }
  for (int sh = 0; sh < nshards; sh++) {            // one transfer per shard that is present
    const int c = sc.shard_cnt[sh];
    if (!c) continue;
    const size_t off = (size_t)sc.shard_start[sh] * row_bytes;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync((char*)sc.row_stage + off, pin + off,
                                      (size_t)c * row_bytes, cudaMemcpyHostToDevice, s));
  }
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.row_slot, slot_host, (size_t)rows * 4,
                                    cudaMemcpyHostToDevice, s));
  // The event covers every copy that reads this slot, so the next chunk to pick it up knows the
  // DMA is done with it.
  HELIOS_CUDA_CHECK(cudaEventRecord(sc.row_pin_ev[gslot], s));
  ple_rows_scatter_kernel<<<rows, 64, 0, s>>>((int16_t*)sc.rows, (const int16_t*)sc.row_stage,
                                              (const int*)sc.row_slot, words);
  HELIOS_CUDA_CHECK(cudaPeekAtLastError());
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.row_ids, sc.row_ids_host, (size_t)rows * 4,
                                    cudaMemcpyHostToDevice, s));
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.head_ids, sc.head_ids_host, (size_t)rows * 4,
                                    cudaMemcpyHostToDevice, s));
  ngram_dequant_f16(sc.rows, (const int*)sc.head_ids, (const half*)sc.bias_dev, (half*)sc.decoded,
                    rows, cfg.ngram_K, words, s);
  // decoded is [n*nh, 160] row-major, which is exactly the [n, 2560] n-gram embedding, and it is
  // the embedding the projections consume - NOT the token embedding. `embed` is the token
  // embedding of the current chunk and is not an input to this layer at all; an earlier revision
  // passed it here, which silently reduced the whole layer to a function of the current token and
  // discarded the n-gram table entirely.
  const half* emb = (const half*)sc.decoded;

  // 2) key = norm_key(key_proj(emb)); value = value_proj(emb); query = norm_query(streams)
  glue::gemm_nt_f16(sc.key16, emb, w.key_proj, n, ch, D, /*y_fp32=*/false, false, s);
  aux::rms_norm(sc.key16, aux::kHalf, w.norm_key, aux::kHalf, sc.key, aux::kFloat, n * H, D,
                cfg.rms_eps, 1.0f, 1.0f, false, H, s);
  glue::gemm_nt_f16(sc.value16, emb, w.value_proj, n, D, D, false, false, s);
  aux::rms_norm(streams, aux::kFloat, w.norm_query, aux::kHalf, sc.query, aux::kFloat, n * H, D,
                cfg.rms_eps, 1.0f, 1.0f, false, H, s);

  // 3) per-stream dot -> gate transform -> grouped norm
  HELIOS_CUDA_CHECK(cudaMemsetAsync(sc.gate, 0, (size_t)n * H * 4, s));
  {
    dim3 grid(n, H);
    ple_dot_kernel<<<grid, 256, 0, s>>>((const float*)sc.query, (const float*)sc.key,
                                        (float*)sc.gate, H, D);
  }
  ple_gate((const float*)sc.gate, sc.value16, (float*)sc.gated, 1.0f / sqrtf((float)D), n, H, D, s);
  aux::rms_norm(sc.gated, aux::kFloat, w.norm_conv, aux::kHalf, sc.normed, aux::kHalf, n * H, D,
                cfg.rms_eps, 1.0f, 1.0f, false, H, s);

  // Speculative verify only: keep the pre-conv state and this chunk's conv input so a partial
  // accept can rewind the 9-column running conv over the accepted prefix (Runner::ple_replay).
  if (capture) {
    const size_t cs = (size_t)ch * (4 - 1) * cfg.ngram_size * 2;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.conv_state_snap, conv_state, cs, cudaMemcpyDeviceToDevice, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.conv_in_snap, sc.normed, (size_t)n * ch * 2,
                                      cudaMemcpyDeviceToDevice, s));
  }
  // 4) dilated conv (dilation = ngram_size), then inject gated + conv into all four streams
  ple_dilated_conv(sc.normed, conv_state, w.conv, sc.conv_out, n, ch, cfg.ngram_size, s);
  const long total = (long)n * H * D;
  if (getenv("HELIOS_PLE")) {
    auto sr = [&](const float* q, long cnt) {
      std::vector<float> t(cnt);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), q, cnt * 4, cudaMemcpyDeviceToHost, s));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      double a = 0;
      for (float v : t) a += (double)v * v;
      return sqrt(a / cnt);
    };
    fprintf(stderr, "[ple] streams before=%.6f (total=%ld)\n", sr(streams, total), total);
  }
  ple_inject_kernel<<<(int)((total + 255) / 256), 256, 0, s>>>(streams, (const float*)sc.gated,
                                                               (const half*)sc.conv_out, total);
  if (getenv("HELIOS_PLE")) {
    std::vector<float> t((size_t)total);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), streams, (size_t)total * 4,
                                      cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    double a = 0;
    for (float v : t) a += (double)v * v;
    fprintf(stderr, "[ple] streams after=%.6f\n", sqrt(a / total));
  }
  if (getenv("HELIOS_PLE")) {
    auto hr = [&](const void* p, size_t cnt, bool bf) {
      std::vector<unsigned short> t(cnt);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), p, cnt * 2, cudaMemcpyDeviceToHost, s));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      double a = 0;
      for (size_t i = 0; i < cnt; i++) {
        const float f = bf ? __bfloat162float(__ushort_as_bfloat16(t[i]))
                           : __half2float(*(half*)&t[i]);
        a += (double)f * f;
      }
      return sqrt(a / cnt);
    };
    auto fr = [&](const void* p, size_t cnt) {
      std::vector<float> t(cnt);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), p, cnt * 4, cudaMemcpyDeviceToHost, s));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      double a = 0;
      for (float v : t) a += (double)v * v;
      return sqrt(a / cnt);
    };
    fprintf(stderr, "[ple] rows=%d hash0=%lld hash1=%lld row0=%d head0=%d\n", rows,
            (long long)sc.hash_host[0], (long long)sc.hash_host[1], sc.row_ids_host[0],
            sc.head_ids_host[0]);
    fprintf(stderr, "[ple] decoded=%.6f key16=%.6f value16=%.6f gate=%.6f gated=%.6f "
                    "normed=%.6f conv_out=%.6f key=%.6f query=%.6f\n",
            hr(sc.decoded, (size_t)rows * kNgramRowDim, false), hr(sc.key16, (size_t)n * ch, false),
            hr(sc.value16, (size_t)n * sc.dim, false), fr(sc.gate, (size_t)n * H),
            fr(sc.gated, (size_t)n * H * sc.dim), hr(sc.normed, (size_t)n * ch, false),
            hr(sc.conv_out, (size_t)n * ch, false), fr(sc.key, (size_t)n * ch),
            fr(sc.query, (size_t)n * ch));
    {
      std::vector<float> gv((size_t)n * H);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(gv.data(), sc.gate, gv.size() * 4,
                                        cudaMemcpyDeviceToHost, s));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      double mean = 0;
      int pos = 0;
      for (float v : gv) { mean += v; if (v > 0) pos++; }
      mean /= gv.size();
      fprintf(stderr, "[ple] gate mean=%.2f pos=%d/%zu scaled_mean=%.4f\n", mean, pos, gv.size(),
              mean * cfg.hidden);
    }
  }

}

}  // namespace helios
