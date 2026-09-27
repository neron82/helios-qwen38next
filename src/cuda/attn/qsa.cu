#include "qsa.cuh"
#include "qwen_rope.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../../engine/glue.cuh"
#include "../../engine/glue2.cuh"
#include "../aux/norm.cuh"
#include <algorithm>
#include <cmath>

namespace helios { namespace attn {

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

namespace {

// Mean-pool one block of `cr` raw keys: fp32 accumulate, cast to fp16. Mirrors the reference's
// `raw_k.view(bsz, nb, cr, dk).float().mean(dim=2).to(dtype)`.
__global__ void qsa_pool_one_kernel(const half* block, half* out, int cr, int dk) {
  const int d = threadIdx.x;
  if (d >= dk) return;
  float acc = 0.f;
  for (int j = 0; j < cr; j++) acc += __half2float(block[(size_t)j * dk + d]);
  out[d] = __float2half_rn(acc / (float)cr);
}

// Score every completed block for ONE query: score[b] = sum_h relu(q_h . k_b) * scale.
// One WARP per block, one block per pooled entry.
//
// This used to launch 128 threads for a 128-element dot product, so every thread computed exactly
// one product and then paid a full block reduction to combine it: five shuffles, a shared-memory
// staging step and two __syncthreads PER HEAD, four times over, to combine four numbers per thread.
// Worse, all n_heads dot the SAME key vector, so that vector was re-read from memory once per head.
// A single warp removes the shared memory and both barriers entirely, and hoisting the key load out
// of the head loop reads it once. head_dim is a multiple of 32 here (128), so the lane->element
// mapping is a clean 4 elements per lane.
__global__ __launch_bounds__(32) void qsa_score_kernel(const half* __restrict__ q, int n_heads,
                                 int head_dim, const half* __restrict__ pooled, int n_blocks,
                                 float scale, float* __restrict__ score) {
  const int b = blockIdx.x;
  if (b >= n_blocks) return;
  const half* kh = pooled + (size_t)b * head_dim;
  const int lane = threadIdx.x;                 // blockDim is 32: the block IS one warp
  constexpr int kMaxPerLane = 8;                 // head_dim up to 256
  float kv[kMaxPerLane];
  int cnt = 0;
  for (int t = lane; t < head_dim; t += 32)
    if (cnt < kMaxPerLane) kv[cnt++] = __half2float(kh[t]);

  float acc = 0.f;
  for (int h = 0; h < n_heads; h++) {
    const half* qh = q + (size_t)h * head_dim;
    float d = 0.f;
    int j = 0;
    for (int t = lane; t < head_dim; t += 32, j++)
      if (j < kMaxPerLane) d += __half2float(qh[t]) * kv[j];
    // One warp means the entire cross-lane reduction is five shuffles: no shared memory, no barriers.
    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    acc += fmaxf(d, 0.f);
  }
  if (lane == 0) score[b] = acc * scale;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------------------------

size_t qsa_scratch_bytes(int n, int n_blocks, int n_heads, int head_dim, int n_sel, int hidden) {
  size_t b = 0;
  b += (size_t)n * (hidden > 0 ? hidden : n_heads * head_dim) * 2;   // Hadamard scratch for the gemm
  b += (size_t)n * (n_heads + 1) * head_dim * 2;   // raw projection output (includes the key)
  b += (size_t)n * n_heads * head_dim * 2;          // q (normed + roped)
  b += (size_t)n * head_dim * 2;                    // raw_k
  b += (size_t)n * n_sel * 4;                       // sel
  b += (size_t)n * n_blocks * 4;                    // score
  b += (size_t)n * 4;                               // pos
  return b;
}

size_t qsa_layer_bytes(int max_ctx, int head_dim, int compress_ratio) {
  const size_t nb = (size_t)(max_ctx / compress_ratio) + 2;
  return nb * head_dim * 2 + nb * 4 + (size_t)compress_ratio * head_dim * 2;
}

// ---------------------------------------------------------------------------------------------
// project: fused qk projection, q norm+rope, raw_k extraction
// ---------------------------------------------------------------------------------------------

void qsa_project(const void* x, int n, const void* qk_w, const void* qk_suh, const void* qk_svh,
                 int qk_mul1, int qk_bits, int hidden, const void* q_ln_w,
                 const void* k_ln_w, int n_heads,
                 int head_dim, int rotary_dim, float rope_theta, float rms_eps, QsaScratch& sc,
                 Stream s) {
  if (n <= 0) return;
  const int qk_out = (n_heads + 1) * head_dim;
  sc.n = n;

  // mul1 comes from the checkpoint, like every other GroupWords call site. It was hardcoded 0 here,
  // but `index_qk_proj.mul1` IS present in this quant's config - so the dequant path was indexing
  // the scale arrays as if the tensor carried none, and compute-sanitizer caught the resulting
  // read as a wild pointer 124 TB before any allocation inside the input Hadamard.
  exl3::GroupWords gw{(const uint16_t*)qk_w, (const half*)qk_suh, (const half*)qk_svh, qk_mul1};
  // The gemm writes its input Hadamard transform into the `a_had` argument - n*hidden halves.
  // Passing sc.qk there (sized for the output only) overflowed into q / raw_k / sel / score.
  exl3::gemm(sc.qk, x, gw, n, qk_out, hidden, qk_bits, /*y_fp32=*/false, s, sc.had);

  // q = first n_heads*head_dim of each row, k = the tail head_dim of each row.
  const half* src = (const half*)sc.qk;
  const int stride = qk_out;
  // Split q (first n_heads*head_dim per row) and the raw key (the tail head_dim per row) out of the
  // fused projection. This MUST be a device-side copy: the first version dereferenced sc.qk / sc.q /
  // sc.raw_k on the HOST, which is a segfault on a device pointer. It is the reason the engine produced
  // no output at any QSA stage.
  {
    const size_t q_row = (size_t)n_heads * head_dim;              // halves per row, the q part
    const size_t k_off = q_row;                                    // raw key starts after q
    const size_t row_bytes = (size_t)qk_out * 2;
    for (int i = 0; i < n; i++) {
      const char* src_row = (const char*)sc.qk + (size_t)i * row_bytes;
      cuda_check(cudaMemcpyAsync((char*)sc.q + (size_t)i * q_row * 2, src_row, q_row * 2,
                                 cudaMemcpyDeviceToDevice, s));
      cuda_check(cudaMemcpyAsync((char*)sc.raw_k + (size_t)i * head_dim * 2, src_row + k_off * 2,
                                 (size_t)head_dim * 2, cudaMemcpyDeviceToDevice, s));
    }
  }
  // q_layernorm: RMSNorm with constant_bias 1.0, over head_dim, per (token, head).
  aux::rms_norm(sc.q, aux::kHalf, q_ln_w, aux::kHalf, sc.q, aux::kHalf, n * n_heads, head_dim,
                 rms_eps, /*constant_bias=*/1.0f, 1.0f, false, 1, s);
  // Rope q at the query position. The buffer is [n, n_heads, head_dim]; rope wants [rows, n_heads, hd].
  rope_qk_partial_neox(sc.q, nullptr, n, n_heads, 0, head_dim, rotary_dim, sc.pos, rope_theta, s);
  (void)k_ln_w;
}

// ---------------------------------------------------------------------------------------------
// pool_update: fold raw keys into completed blocks (incremental, cacheable)
// ---------------------------------------------------------------------------------------------

void qsa_pool_update(QsaLayerState& st, const half* new_raw_k, int n, const void* k_ln_w,
                    int head_dim, int rotary_dim, int compress_ratio, float rope_theta, float rms_eps,
                    Stream s) {
  if (n <= 0) return;
  half* tail = (half*)st.tail_raw;
  int done = 0;
  while (done < n) {
    const int room = compress_ratio - st.n_tail;
    const int take = std::min(room, n - done);
    cuda_check(cudaMemcpyAsync(tail + (size_t)st.n_tail * head_dim, new_raw_k + (size_t)done * head_dim,
                               (size_t)take * head_dim * 2, cudaMemcpyDeviceToDevice, s));
    st.n_tail += take;
    done += take;
    if (st.n_tail == compress_ratio) {
      half* dst = (half*)st.pooled_k + (size_t)st.n_blocks * head_dim;
      qsa_pool_one_kernel<<<1, 256, 0, s>>>(tail, dst, compress_ratio, head_dim);
      // k_layernorm then rope at the block START position (block j covers keys [j*cr, (j+1)*cr)).
      aux::rms_norm(dst, aux::kHalf, k_ln_w, aux::kHalf, dst, aux::kHalf, 1, head_dim, rms_eps,
                     /*constant_bias=*/1.0f, 1.0f, false, 1, s);
      const int bstart = st.n_blocks * compress_ratio;
      cuda_check(cudaMemcpyAsync(st.block_pos + st.n_blocks, &bstart, sizeof(int),
                                 cudaMemcpyHostToDevice, s));
      rope_qk_partial_neox(dst, nullptr, 1, 1, 0, head_dim, rotary_dim, st.block_pos + st.n_blocks,
                           rope_theta, s);
      st.n_blocks++;
      st.n_tail = 0;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// select: score + top-k over blocks, decode path (n == 1)
// ---------------------------------------------------------------------------------------------

// Top-k select. Each of the K = ceil(n_sel/32) warps keeps a running top-(n_sel/K) in registers, then
// the warps merge through shared memory and one warp does the final insertion sort. Streaming: a warp
// scans the whole score array (strided) and only touches its candidate set on improvement, so the cost
// is O(n_blocks) loads regardless of n_sel.
// Top-k over block scores. Single block.
//
// Correctness note that cost me a test cycle: each warp scans a disjoint stripe and keeps its own top-c.
// A block that is NOT in its stripe's top-c has at least c blocks in the same stripe scoring higher, so
// its global rank exceeds c. Therefore the union of the per-warp sets contains the true top-n_sel iff
// c >= n_sel. With c = ceil(n_sel/nwarps) the union is far too small - at n_sel=8, nwarps=32 that is
// c=1 and only 2 candidates survive. So c is n_sel, and the warp count is chosen to fit shared memory
// (each slot is a float + an int = 8 bytes; ~8000 slots is 64 KB, safe on a 100 KB SM).
__global__ void qsa_topk_kernel(const float* __restrict__ score, int n_blocks, int n_sel,
                                int* __restrict__ out) {
  extern __shared__ float sm[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int max_warps = (int)(blockDim.x >> 5);
  const int W = max_warps < 1 ? 1 : max_warps;
  const int c = n_sel;                      // per-warp candidate depth; see note above
  const int slots = W * c;
  float* bs = sm;                           // [slots], each warp's slice sorted ASCENDING
  int*   bi = (int*)(sm + slots);
  __shared__ int scnt[32];
  for (int i = lane; i < c; i += 32) { bs[warp * c + i] = -INFINITY; bi[warp * c + i] = -1; }
  if (lane == 0) scnt[warp] = 0;
  __syncthreads();

  // Warp-aggregated insertion into a per-warp ASCENDING top-c list, with a fill count.
  //
  // Two bugs lived here, both caught by the parity test (the first version returned blocks 0 and 32
  // regardless of the scores):
  //  1. All 32 lanes of a warp hold different candidates but share ONE list. A per-lane insert races.
  //     The winner is now chosen warp-wide (butterfly max + ballot) and only that lane writes.
  //  2. Inserting into an all--inf list by walking down from slot c-1 always lands on c-1 and
  //     overwrites the previous value, so exactly one entry survived per warp. A count (scnt) is
  //     needed to know how much of the list is real, and insertion must shift from the fill point up.
  // The iteration count is UNIFORM across the warp; deriving it from a per-lane bound makes the loop
  // divergent and the __any_sync below then hangs.
  const int rounds = (n_blocks + W * 32 - 1) / (W * 32);
  for (int r = 0; r < rounds; r++) {
    const int b = (r * W + warp) * 32 + lane;
    float v = (b < n_blocks) ? score[b] : -INFINITY;
    for (;;) {
      const int base = warp * c;
      const int cnt = scnt[warp];
      const float floor_v = (cnt < c) ? -INFINITY : bs[base];
      if (!__any_sync(0xffffffffu, v > floor_v)) break;
      float best = v;
      for (int o = 16; o; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
      const unsigned mask = __ballot_sync(0xffffffffu, best == v);
      const int src = __ffs(mask) - 1;
      if (lane == src) {
        int p = 0;
        while (p < cnt && bs[base + p] < best) p++;
        const int last = (cnt < c) ? cnt : c - 1;
        for (int i = last; i > p; i--) {
          bs[base + i] = bs[base + i - 1];
          bi[base + i] = bi[base + i - 1];
        }
        bs[base + p] = best;
        bi[base + p] = b;
        if (cnt < c) scnt[warp] = cnt + 1;
        v = -INFINITY;
      }
    }
  }
  __syncthreads();

  // k-way merge of the W sorted-descending lists: each round takes the largest unconsumed head across
  // all lists and advances that list's cursor.
  __shared__ int cur[32];
  if (threadIdx.x == 0) {
    for (int w = 0; w < W; w++) cur[w] = 0;
    __syncwarp();
    for (int r = 0; r < n_sel; r++) {
      int bw = -1; float best = -INFINITY;
      for (int w = 0; w < W; w++) {
        if (cur[w] < c) {
          const float v = bs[w * c + c - 1 - cur[w]];   // descending from the weakest end
          if (v > best) { best = v; bw = w; }
        }
      }
      if (bw < 0) { out[r] = -1; continue; }
      out[r] = bi[bw * c + c - 1 - cur[bw]];
      cur[bw]++;

    }
  }


}

void qsa_select(const QsaLayerState& st, QsaScratch& sc, int n_heads, int head_dim, int n_sel,
                float scale, Stream s) {
  if (sc.n != 1 || st.n_blocks <= 0 || n_sel <= 0) return;
  float* score = (float*)sc.score;
  qsa_score_kernel<<<st.n_blocks, 32, 0, s>>>((const half*)sc.q, n_heads, head_dim,
                                                (const half*)st.pooled_k, st.n_blocks, scale, score);
  // Warps so that W * n_sel slots fit in shared memory. The 48 KB DEFAULT dynamic-smem limit applies
  // unless the kernel opts in via cudaFuncSetAttribute, and a 64 KB request fails the launch with
  // "invalid argument" - which the next unrelated cuda_check in the attention layer reports, making it
  // look like an activation-kernel bug. At n_sel = 512 this needs W <= 12 to stay under 48 KB; the unit
  // test used n_sel = 8 and never hit the limit, which is why it passed and the engine did not.
  // A dynamic shared-memory request of EXACTLY 49152 B (the 48 KB default cap) is rejected: the
  // usable default is one block LESS than the cap on sm_86. The failed launch left a sticky
  // cudaErrorInvalidValue that every LATER cuda_check inherited, so the fault surfaced as
  // "invalid argument" in an unrelated activation kernel two stages downstream, and every
  // cudaDeviceSynchronize in between reported success. Keep the request strictly under the cap.
  constexpr size_t kSmemUsable = 48 * 1024 - 1024;   // 47 KB, safely under
  const size_t slot = sizeof(float) + sizeof(int);
  int W = (int)(kSmemUsable / (slot * (size_t)(n_sel > 0 ? n_sel : 1)));
  if (W > 32) W = 32;
  if (W < 1) W = 1;
  const size_t sh = (size_t)W * n_sel * slot;
  if (sh > kSmemUsable) { fprintf(stderr, "[qsa] select: %zu B shared > usable\n", sh); return; }
  qsa_topk_kernel<<<1, W * 32, sh, s>>>(score, st.n_blocks, n_sel, (int*)sc.sel);
  // (An ascending sort of the selected block ids was tried here to give the gather near-sequential
  // memory access. Measured NEUTRAL: 15.94 / 15.87 tok/s with it against 15.67 / 15.88 without, at 15k
  // context. So the gather order is not what makes sparse attention slower than dense here.)
}

// ---------------------------------------------------------------------------------------------

namespace {
// Expand each query's selected blocks into the exact token indices it may attend to.
//
// Deterministic by construction: query q writes into a FIXED slot range [q*cap, q*cap + count), with the
// count precomputed by qsa_count_kernel. The obvious atomicAdd(n_out, 1) version is wrong here - the
// interleaving is not run-to-run stable, so the attention kernel's summation order would vary and greedy
// decoding would stop being reproducible. This is the same class of problem as the grid/autotune
// experiments that diverged text.
// Expand each query's selected blocks into the exact token indices it may attend to.
//
// ONE thread per query, serial. The output is at most n_sel*cr + cr = 2052 entries, so a serial loop is
// free, and a single thread makes the result trivially deterministic and run-to-run identical - which
// matters because greedy decoding must reproduce.
//
// The previous version was a 256-thread kernel that used threadIdx as a striding variable over the
// blocks while ALSO storing the per-query count in tok_off; launched with one thread it walked only
// block 0 and then overwrote the count with a block id, which made the sparse kernel attend to a
// garbage range. The engine produced no output at all, which is how it surfaced.
__global__ void qsa_expand_kernel(const int* __restrict__ sel, int n_sel, int pos, int cr, int cap,
                                  int* __restrict__ tok_idx, int* __restrict__ tok_off) {
  const int q = blockIdx.x;
  if (threadIdx.x != 0 || q != 0) return;   // n_rows == 1 for the decode path
  const int nb_q = (pos + 1) / cr;
  int o = 0;
  for (int i = 0; i < n_sel && o + cr <= cap; i++) {
    if (sel[i] < 0) break;
    if (sel[i] >= nb_q) continue;           // block starts at or beyond the query's position
    const int b = sel[i];
    for (int r = 0; r < cr; r++) {
      const int idx = b * cr + r;
      if (idx > pos) continue;              // causality within the last block
      tok_idx[o] = idx;
      o++;
    }
  }
  for (int idx = nb_q * cr; idx <= pos && o < cap; idx++) {   // incomplete tail, always visible
    tok_idx[o] = idx;
    o++;
  }
  tok_off[0] = o;                           // in-band count, read by gqa_sparse_decode
}
}  // namespace

void qsa_expand(const QsaLayerState& st, const QsaScratch& sc, int n_rows, int n_sel, int cr, int pos,
                int cap, void* tok_idx, void* tok_block, void* tok_off, Stream s) {
  if (n_rows <= 0) return;
  qsa_expand_kernel<<<n_rows, 1, 0, s>>>((const int*)sc.sel, n_sel, pos, cr, cap, (int*)tok_idx,
                                         (int*)tok_off);
  if (getenv("QSA_EXP_DBG")) {
    cudaError_t e = cudaStreamSynchronize(s);

  }
}


// ---------------------------------------------------------------------------------------------
// Sparse GQA decode attention
// ---------------------------------------------------------------------------------------------

namespace {

// Phase 1: one warp per (chunk, row, head) writes its (m, l, o) partial over W gathered tokens.
// Phase 2 combines over chunks. Two-phase because a single block with W warps splitting the gathered
// range would work, but grid.x chunks each writing the same output row overwrite one another.
__global__ void gqa_sparse_partial_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                         const half* __restrict__ v, const int* __restrict__ tok_idx,
                                         const int* __restrict__ tok_off, float* __restrict__ part,
                                         int n, int n_q_heads, int n_kv_heads, int hd, float scale) {
  const int W = 16;
  const int chunk = blockIdx.x;
  const int bh = blockIdx.y;
  if (bh >= n * n_q_heads) return;
  const int row = bh / n_q_heads;
  const int h = bh % n_q_heads;
  const int kvh = h / (n_q_heads / n_kv_heads);
  const int lane = threadIdx.x & 31;

  const int nvis = tok_off[0];
  const int t0 = chunk * W;
  const int t1 = min(nvis, t0 + W);

  const half* qh = q + ((size_t)row * n_q_heads + h) * hd;
  float o[8];
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = 0.f;
  float m = -INFINITY, l = 0.f;

  for (int t = t0; t < t1; t++) {
    const int pos = tok_idx[t];
    const half* kh = k + ((size_t)pos * n_kv_heads + kvh) * hd;
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
    const half* vh = v + ((size_t)pos * n_kv_heads + kvh) * hd;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int d = lane * 8 + i;
      o[i] = fmaf(b, __half2float(vh[d]), o[i] * a);
    }
    m = mnew;
  }

  float* dst = part + ((size_t)bh * gridDim.x + chunk) * (2 + hd);
  if (lane == 0) { dst[0] = m; dst[1] = l; }
#pragma unroll
  for (int i = 0; i < 8; i++) dst[2 + lane * 8 + i] = o[i];
}

// Combine the per-chunk (m, l, o) partials.
//
// Back to the DENSE kernel's proven shape: one block per (row, head), 16 warps, each streaming a
// contiguous slice of the gathered indices, combined through shared memory. The previous two-stage
// reduce-then-combine was wrong in a subtle way - stage A's group slot holds only the 8 components its
// own lane wrote, while stage B indexed the slot at ITS lane's offset, so lanes read each other's
// (mostly zero) data. Reproducing the dense structure sidesteps the transposition entirely: with 16
// warps per block each warp owns one lane's components and the shared combine reads sm_o[w][...] the
// same way the dense kernel does.
template <int W>
__global__ void gqa_sparse_combine_kernel(const float* __restrict__ part, half* __restrict__ out,
                                          int nchunks, int n, int n_q_heads, int hd) {
  const int bh = blockIdx.x;
  if (bh >= n * n_q_heads) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const float* base = part + (size_t)bh * nchunks * (2 + hd);
  const int nvis = nchunks;                       // chunk count is fixed; empty ones have m = -inf

  __shared__ float sm_m[W], sm_l[W], sm_o[W][256];
  // Each warp folds its strided subset of chunks for THIS lane's 8 components.
  float m = -INFINITY, l = 0.f;
  float o[8];
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = 0.f;
  for (int c = warp; c < nvis; c += W) {
    const float* p = base + (size_t)c * (2 + hd);
    const float cm = p[0], cl = p[1];
    if (cm == -INFINITY) continue;
    if (m == -INFINITY) {
      m = cm; l = cl;
#pragma unroll
      for (int i = 0; i < 8; i++) o[i] = p[2 + lane * 8 + i];
    } else {
      const float mnew = fmaxf(m, cm);
      const float a = __expf(m - mnew), b = __expf(cm - mnew);
      l = l * a + cl * b;
#pragma unroll
      for (int i = 0; i < 8; i++) o[i] = fmaf(b, p[2 + lane * 8 + i], o[i] * a);
    }
  }
  if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
#pragma unroll
  for (int i = 0; i < 8; i++) sm_o[warp][lane * 8 + i] = o[i];
  __syncthreads();

  float gm = -INFINITY;
#pragma unroll
  for (int w = 0; w < W; w++) gm = fmaxf(gm, sm_m[w]);
  float gl = 0.f;
#pragma unroll
  for (int w = 0; w < W; w++) if (sm_m[w] != -INFINITY) gl += sm_l[w] * __expf(sm_m[w] - gm);
  half* oh = out + (size_t)bh * hd;
  const float inv = gl > 0.f ? 1.f / gl : 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int d = lane * 8 + i;
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < W; w++)
      if (sm_m[w] != -INFINITY) acc += sm_o[w][d] * __expf(sm_m[w] - gm);
    oh[d] = __float2half_rn(acc * inv);
  }
}

}  // namespace

void gqa_sparse_decode(const void* q, const void* k, const void* v, const int* tok_idx,
                       const int* tok_off, int cap, void* out, float* part, float* grp, int n,
                       int n_q_heads, int n_kv_heads, int head_dim, float scale, Stream s) {
  if (n <= 0 || head_dim != 256 || cap <= 0) return;
  const int W = 16;
  const int nchunks = (cap + W - 1) / W;
  dim3 grid((unsigned)nchunks, (unsigned)(n * n_q_heads), 1);
  gqa_sparse_partial_kernel<<<grid, 32, 0, s>>>((const half*)q, (const half*)k, (const half*)v,
                                                tok_idx, tok_off, part, n, n_q_heads, n_kv_heads,
                                                head_dim, scale);
  gqa_sparse_combine_kernel<W><<<(unsigned)(n * n_q_heads), W * 32, 0, s>>>(part, (half*)out, nchunks,
                                                                          n, n_q_heads, head_dim);

}

}} // namespace helios::attn
