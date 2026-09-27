// GatedResidual (qwen4_exp hyper-connections). See gr_mix.cuh for the exact math and why it follows
// hyperconnections.py::_mix_ref rather than exllamav3's fused kernel layout.
//
// One block per row: the 4 streams of a token are only 10240 floats, so they fit in shared memory
// and both projections (down: rank x H*D, up: H*D x rank) read them without recomputing the norm.
// This is the correctness-first path; the upstream fuses the norm into the fp16 weights and runs the
// second projection as a half-GEMM, which is a speed change to make once parity is proven.
#include "gr_mix.cuh"
#include "gr_mix_tc.cuh"

#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace helios { namespace aux {

// HELIOS_MIXER_TC=1 runs the tensor-core mixer (gr_mix_tc.cu) instead of the seven fp32 kernels
// below. It is a DIFFERENT NUMERICAL PATH, not a faster spelling of this one: it narrows the two A
// operands to fp16 and accumulates in fp32, so it is not bit-identical to the fp32 reference this
// file implements and that test_aux checks. That is why it is off by default and why the fp32
// kernels are left exactly as they are - they are the oracle. Rows below the threshold keep the
// fp32 path regardless of the flag: its decode-shaped kernels are latency-bound, and the TC
// kernels tile 64 rows at a time. Measured crossover and error are in RESULTS.md.
static int g_tc_force = -1, g_tc_min_r_force = 0;      // gr_force_tc(); -1 = follow the env

bool gr_mixer_tc()
{
  if (g_tc_force >= 0) return g_tc_force != 0;
  static int v = -1;
  if (v < 0) { const char* e = getenv("HELIOS_MIXER_TC"); v = (!e || atoi(e)) ? 1 : 0; }
  return v != 0;
}

int gr_mixer_tc_min_r()
{
  if (g_tc_force >= 0) return g_tc_min_r_force;
  static int v = -1;
  if (v < 0) { const char* e = getenv("HELIOS_MIXER_TC_MIN_R"); v = e ? atoi(e) : 256; }
  return v;
}

void gr_force_tc(bool on, int min_r) { g_tc_force = on ? 1 : 0; g_tc_min_r_force = min_r; }

bool gr_mixer_tc_active(int R, int H, int D, int rank)
{
  return gr_mixer_tc() && gr_mix_tc_applicable(R, H, D, rank, gr_mixer_tc_min_r());
}

namespace {

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x * sigmoidf_(x); }

// Split into three kernels so the work actually fills the GPU.
//
// The original form launched ONE BLOCK PER ROW. That is fine at prefill (R = chunk size) and
// catastrophic at decode, where R = 1: a single 256-thread block then had to stream the whole
// `down` and `up` pair (2 x 6.5 MB for this model) on one SM. Measured cost was 2.8 ms per call,
// 136 ms per token for each of the two sites - 85 % of the entire decode step.
//
// Now:
//   gr_prep : one block per row, but only the cheap parts (per-stream rms, the rank-sized t).
//   gr_up   : grid (R, ceil(H*D/256)), one thread per output element - this is the 109 MFLOP /
//             13 MB part and it now fills the machine.
//   gr_fin  : grid (R, ceil(D/256)), the H-way reduction into `mixed`.
//   gr_post : the site gate, R*H elements.
// `t` and the gated streams are caller-independent temporaries held in a cached device buffer.

// Per-stream RMS over D and the (w+1) scale, in two parallel kernels.
//
// The single-kernel form summed each stream with only H (=4) threads looping D (=2560) times - 2560
// serial FMAs per lane - from R (=1) block, i.e. one SM. Measured 49.11 us per call, 38% of the whole
// hc mix, for ~100 KB of traffic (~2 GB/s). Split it: a (R, H) grid does the sums with full block
// reductions, then a grid-stride pass applies the scale.
__global__ void gr_hsum_kernel(const float* __restrict__ streams, int H, int D,
                               float* __restrict__ sums) {
  const int r = blockIdx.x, h = blockIdx.y;
  const float* x = streams + ((size_t)r * H + h) * D;
  float acc = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) acc = fmaf(x[d], x[d], acc);
  __shared__ float red[32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    const int nw = (int)(blockDim.x >> 5);
    acc = lane < nw ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) sums[r * H + h] = acc;
  }
}

__global__ void gr_scale_kernel(const float* __restrict__ streams, const half* __restrict__ norm_raw,
                                const float* __restrict__ sums, int R, int H, int D, float eps,
                                float* __restrict__ normed) {
  const int HD = H * D;
  const size_t total = (size_t)R * HD;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (size_t)gridDim.x * blockDim.x) {
    const int r = (int)(i / (size_t)HD);
    const int h = (int)((i % (size_t)HD) / (size_t)D);
    // norm_raw is [H*D] - one weight row shared by every batch row - so the index wraps. Indexing
    // it with the global i instead read past the end for any R > 1 (R = 1, i.e. decode, was fine;
    // a 57-token prefill chunk was not).
    normed[i] = streams[i] * rsqrtf(sums[r * H + h] / (float)D + eps) *
                (__half2float(norm_raw[i % (size_t)HD]) + 1.0f);
  }
}

// t[r][i] = silu((1/H) * sum_j normed[r][j] * down[i][j]) - a tall-skinny GEMM, (R x HD) x (HD x rank).
//
// The previous form was one block per (r, i), which tiles NEITHER operand: every block re-read both
// normed[r] and down[i] in full. ncu at prefill (R=256) measured 391.4 MB of DRAM traffic and
// 2.26 ms for a pair of matrices totalling 17 MB - about 2.9 TB/s of L2 traffic, so the stage was
// bandwidth-bound on re-reads, not short of parallelism (97.4% warps active).
//
// Now a block owns a DTR x DTI tile of the output and walks HD in DBK-deep chunks, staging both
// operand tiles in shared memory. Each operand is then read from L2 once per tile of the OTHER
// axis - DTR=8 gives 32 row-tiles and DTI=32 gives 10 col-tiles, so L2 traffic falls from ~6.55 GB
// to ~315 MB, a 21x cut. The `down` row stride is padded by 2 halfs so that the 32 threads reading
// consecutive output columns land in 32 distinct smem banks (stride 130 halfs = 65 words, bank =
// (li*65) % 32 = li), and staging maps threadIdx to consecutive k so the global reads coalesce.
//
// DBK=128 over 256 is a wash on throughput (321.6/321.3/321.2 against 320.0/319.4/320.1) and is
// taken for the shared memory: 12,416 B against 24,704 B, so twice as many blocks stay resident.
// Block size 64 and 128 were also measured and are worse (246-299), so the "more blocks will raise
// the 38.5% warps" reading is wrong - it is smem per block, not block count, that was the limit.
//
// NOTE this reassociates the sum: the old form did a strided sum plus a butterfly tree, this one
// accumulates j = 0..HD-1 in order. That is a reassociation within a single length-10240 dot
// product (~1e-6 relative), not a change of algorithm. `test_aux` checks the whole gr_mix chain at
// the production shape (R=256) against a CPU reference; it reports 6.879e-04, and it is what
// caught the stride bug described below.
__global__ void gr_dots_kernel(const float* __restrict__ normed, const half* __restrict__ down,
                               int R, int H, int D, int rank, float* __restrict__ t) {
  constexpr int DTR = 8, DTI = 32, DBK = 128, DTP = DBK + 2;   // DTP pads the smem row stride
  extern __shared__ char smem_raw[];
  float* ns = (float*)smem_raw;                 // [DTR][DBK]
  half*  ds = (half*)(ns + DTR * DBK);         // [DTI][DTP]
  const int HD = H * D;
  const int tid = threadIdx.x;                  // 0..255, one output per thread
  const int lr = tid / DTI, li = tid % DTI;
  const int r0 = blockIdx.y * DTR, i0 = blockIdx.x * DTI;
  float acc = 0.f;
  for (int k0 = 0; k0 < HD; k0 += DBK) {
    // The stride must be the block size. A literal 256 silently leaves kk = 256..DBK-1 of every row
    // NEVER WRITTEN at any other block size, and the kernel then reads whatever the previous k0
    // chunk left in shared memory - a wrong-results bug no bounds checker sees, and one a
    // 256-thread launch cannot detect. That is how a tile sweep came to report 0.000e+00 error and
    // 373 tok/s for a 128-thread config that was in fact computing half of nothing.
    //
    // It is a constexpr rather than blockDim.x on purpose: the launch passes DTR*DTI, so the two
    // cannot disagree (static_assert below), and reading blockDim.x in the loop condition instead
    // measured ~7% slower on prefill by defeating constant folding.
    constexpr int NT = DTR * DTI;
    static_assert(NT % 32 == 0, "gr_dots: one warp per output column, so NT must be a multiple of 32");
    for (int e = tid; e < DTR * DBK; e += NT) {           // threadIdx -> consecutive k: coalesced
      const int rr = e / DBK, kk = e % DBK;
      // The row bound belongs HERE, not only on the store below. R is almost never a multiple of
      // DTR (970, 15302, ...), so the final row-tile stages rows R..DTR-1 that do not exist. The
      // store is guarded, so the values were discarded - but they were still READ, past the end of
      // `normed`. That is only harmless while the allocation happens to be large enough to absorb
      // it; enabling QSA shifts the bump allocator so the same 41-byte overread lands in unmapped
      // memory and the run dies with an illegal access 40 bytes past a 39,731,200-byte block
      // (= exactly 970 rows of HD=10240 fp32). Zeroed rather than left uninitialised so the
      // compute loop consumes defined data and the kernel stays deterministic.
      ns[rr * DBK + kk] = (r0 + rr < R) ? normed[(size_t)(r0 + rr) * HD + k0 + kk] : 0.0f;
    }
    for (int e = tid; e < DTI * DBK; e += NT) {
      const int ii = e / DBK, kk = e % DBK;
      ds[ii * DTP + kk] = down[(size_t)(i0 + ii) * HD + k0 + kk];
    }
    __syncthreads();
    const float* nrow = ns + lr * DBK;
    const half*  drow = ds + li * DTP;
    for (int kk = 0; kk < DBK; ++kk) acc = fmaf(nrow[kk], __half2float(drow[kk]), acc);
    __syncthreads();
  }
  if (r0 + lr < R && i0 + li < rank) t[(size_t)(r0 + lr) * rank + i0 + li] = siluf_(acc / (float)H);
}

// ---------------------------------------------------------------------------------------------
// gr_dots2: same tile, same accumulation order, and a shared-memory access per FMA instead of two.
//
// gr_dots_kernel above is 11 % of all prefill GPU time. Its inner loop is
//
//     for (kk = 0; kk < DBK; ++kk) acc = fmaf(nrow[kk], __half2float(drow[kk]), acc);
//
// i.e. one 32-bit and one 16-bit shared-memory load PER FMA. Ampere retires one warp-level LDS
// per cycle per SM (128 B of banks), so at DBK=128 the block spends 256 LDS instructions moving
// 128 FMAs, and the stage runs at half the LDS issue rate no matter how much arithmetic is left.
// 512 x 320 x 10240 = 1.68 G FMAs per call, and the measured 0.885 ms/call is within 25 % of the
// 0.65 ms that LDS issue rate predicts - this stage is not compute bound and not bandwidth bound,
// it is bound by loading its operands one at a time.
//
// Three changes, none of which touches the arithmetic:
//   * Each thread owns TWO output rows (lr and lr + DTR/2) instead of one, so each loaded `down`
//     element feeds 2 FMAs and each loaded `normed` element feeds a full float4 of k.
//   * `normed` chunks live in REGISTERS (float4 loads), and `down` is read as half2. The loop body
//     is now 8 float4 + 16 half2 LDS per 2 x 16 = 32 FMAs, i.e. 0.75 LDS per FMA against 2.0.
//   * DTR goes 8 -> 16 with the same 256 threads, which halves the `down` traffic: `down` is
//     re-read once per row-tile, so 32 row-tiles read 210 MB per call against 419 MB.
//
//+// acc is still a single running fmaf chain over k = 0, 1, ... HD-1 in order, so the result is
// BITWISE IDENTICAL to gr_dots_kernel - only the route the operands take to the FMA changed.
// ---------------------------------------------------------------------------------------------

// Tile: 32 rows x 32 cols, k chunk 64, 256 threads, NROWS outputs per thread. The two numbers that
// matter are (a) how many FMAs each shared-memory operand feeds, because Ampere's smem delivers
// 128 B/cycle/SM and that is a byte budget, not an instruction budget, and (b) how many barriers
// there are, because the staging load's latency is exposed across each one. `down` is the byte
// hog - 2 B per lane per FMA - so it is what NROWS amortises; `normed` is warp-uniform and costs
// 4 B per LDS however many lanes take it.
constexpr int kDots2Rows = 32, kDots2Cols = 32, kDots2K = 64, kDots2KP = kDots2K + 2;
constexpr int kDots2NRow = 4;                                    // outputs per thread
constexpr int kDots2Threads = kDots2Rows * kDots2Cols / kDots2NRow;

__global__ __launch_bounds__(kDots2Threads)
void gr_dots2_kernel(const float* __restrict__ normed, const half* __restrict__ down,
                     int R, int H, int D, int rank, float* __restrict__ t) {
  extern __shared__ char smem2_raw[];
  float* ns = (float*)smem2_raw;                    // [kDots2Rows][kDots2K]
  half*  ds = (half*)(ns + kDots2Rows * kDots2K);  // [kDots2Cols][kDots2KP]
  const int HD = H * D;
  const int tid = threadIdx.x;
  const int lr = tid / kDots2Cols, li = tid % kDots2Cols;   // lr < 8; rows lr + 8 * (0..NROW-1)
  const int r0 = blockIdx.y * kDots2Rows, i0 = blockIdx.x * kDots2Cols;
  constexpr int RSTEP = kDots2Rows / kDots2NRow;
  const float4* ns4 = (const float4*)ns;
  float acc[kDots2NRow];
  #pragma unroll
  for (int u = 0; u < kDots2NRow; ++u) acc[u] = 0.f;
  for (int k0 = 0; k0 < HD; k0 += kDots2K) {
    // Stage this k-chunk. threadIdx maps to consecutive k so the global reads coalesce; the row
    // bound lives on the READ, not only on the store below, for the reason spelled out in
    // gr_dots_kernel (the final row-tile stages rows that do not exist).
    for (int e = tid; e < kDots2Rows * kDots2K; e += kDots2Threads) {
      const int rr = e / kDots2K, kk = e % kDots2K;
      const int gr = r0 + rr;
      ns[rr * kDots2K + kk] = (gr < R) ? normed[(size_t)gr * HD + k0 + kk] : 0.0f;
    }
    for (int e = tid; e < kDots2Cols * kDots2K; e += kDots2Threads) {
      const int ii = e / kDots2K, kk = e % kDots2K;
      ds[ii * kDots2KP + kk] = down[(size_t)(i0 + ii) * HD + k0 + kk];
    }
    __syncthreads();
    // `down` row li, this chunk, in registers: loaded once and reused by every row this thread owns.
    half2 drow[kDots2K / 2];
    {
      const half2* dp = (const half2*)(ds + li * kDots2KP);
      #pragma unroll
      for (int q = 0; q < kDots2K / 2; ++q) drow[q] = dp[q];
    }
    // acc[u] is a single running fmaf chain over k = 0, 1, ... HD-1 in order, exactly as in
    // gr_dots_kernel, so the result is bitwise identical - only the route to the FMA changed.
    #pragma unroll
    for (int q = 0; q < kDots2K / 4; ++q) {
      float4 a[kDots2NRow];
      #pragma unroll
      for (int u = 0; u < kDots2NRow; ++u) a[u] = ns4[(lr + RSTEP * u) * (kDots2K / 4) + q];
      #pragma unroll
      for (int v = 0; v < 4; ++v) {
        const float2 d = __half22float2(drow[2 * q + (v >> 1)]);
        const float dv = (v & 1) ? d.y : d.x;
        #pragma unroll
        for (int u = 0; u < kDots2NRow; ++u) acc[u] = fmaf(((const float*)&a[u])[v], dv, acc[u]);
      }
    }
    __syncthreads();
  }
  if (i0 + li < rank) {
    #pragma unroll
    for (int u = 0; u < kDots2NRow; ++u) {
      const int gr = r0 + lr + RSTEP * u;
      if (gr < R) t[(size_t)gr * rank + i0 + li] = siluf_(acc[u] / (float)H);
    }
  }
}

// Decode path (R small). The tiled form above fixes prefill, but at R=1 only one of its DTR=8 row
// tiles is live, so 7/8 of every block's FMAs are discarded - an 8x regression on the metric that
// matters most. At R=1 the reduction form is already fine (320 blocks, coalesced, 97% warps), so
// both live and gr_mix picks on R. Threshold 8 is where the tiled form's first row tile is full.
__global__ void gr_dots_small_kernel(const float* __restrict__ normed, const half* __restrict__ down,
                                     int R, int H, int D, int rank, float* __restrict__ t) {
  const int r = blockIdx.y;
  const int i = blockIdx.x;
  if (i >= rank) return;
  const int HD = H * D;
  const float* nr = normed + (size_t)r * HD;
  const half* dr = down + (size_t)i * HD;
  float acc = 0.f;
  for (int j = threadIdx.x; j < HD; j += blockDim.x) acc = fmaf(nr[j], __half2float(dr[j]), acc);
  __shared__ float red[32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    const int nw = (int)(blockDim.x >> 5);
    acc = lane < nw ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) t[(size_t)r * rank + i] = siluf_(acc / (float)H);
  }
}

// One WARP per output element of the (R, H*D) gated stream.
//
// A thread-per-element form makes adjacent threads read 640 B apart (one `rank`-long row each),
// which is one sector per 64 B request - measured at ~87 GB/s, a tenth of the card's bandwidth.
// Warp-per-element reads each row with 32 consecutive lanes across 20 half2 loads, i.e. coalesced.
//
// With rank = 24 only 12 of 32 lanes load, then 5 shuffles reduce, so the obvious next move is to batch
// several elements per warp for memory-level parallelism. That was tried (8 elements/warp, same
// per-lane accumulation order and butterfly reduction, so it should have been bitwise identical) and
// it is WORSE: the stage reads 33.23 us against 21.10. Fewer, fatter warps means less parallelism, not
// more - at this size the launch already has 2048 blocks and the limiter is not load batching. Reverted.
// WEIGHT-STATIONARY. ncu at prefill (R=256): this read 957.2 MB from DRAM for a 6.55 MB `up`
// matrix - a 146x re-read - at 79.5% of peak memory throughput, so the kernel was never the
// problem; there were simply 146x too many bytes. The (j, r) grid gives every one of the R tokens
// its own sweep of the whole matrix and L2 does not capture the reuse across a 327,680-block grid.
//
// A block now owns a tile of TJ output elements and streams all R tokens through it, staging its
// TJ `up` rows into shared memory once. DRAM for the stage falls to ~`up` + normed + gated
// (~27 MB). The per-lane accumulation order and the butterfly are untouched, so the result is
// bitwise identical to the old form - the only change is that `up` is read from smem, not DRAM.
// NITER is ceil(rank/2 / 32), the number of times the inner loop runs, made a template parameter
// so the trip count is a compile-time constant. With rank = 320 that is 5, and leaving it to the
// runtime cost a compare, a branch and an index recompute on every one of those 5 iterations -
// about 15 instructions of the ~67 this loop issues per (warp, r), for 10 FMAs. Templating also
// lets ptxas batch the five `t` loads, which are the same 96 bytes for every warp in the block.
// The arithmetic is untouched: same terms, same order, same butterfly, so still bitwise identical.
// NITER = 0 is the generic form, for a rank too wide for the specialisations.
template <int NITER>
__global__ void gr_up_kernel(const float* __restrict__ normed, const float* __restrict__ t,
                             const half* __restrict__ up, int R, int H, int D, int rank,
                             float* __restrict__ gated) {
  extern __shared__ half ush[];                   // TJ * rank, one row per warp
  const int HD = H * D;
  const int TJ = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int j = blockIdx.x * TJ + warp;
  if (j >= HD) return;
  const int lane = threadIdx.x & 31;
  const int npair = rank >> 1;
  half2* us = (half2*)&ush[(size_t)warp * rank];
  const half2* ur = (const half2*)(up + (size_t)j * rank);
  // Left in fp16, at 4 B per lane. Staging the tile as fp32 instead - which would remove the two
  // cvt instructions per term that __half22float2 costs, 10 of the ~49 this loop issues per
  // (warp, r) - measured 951.7 ms against 936.9 ms for the fp16 tile: inside the run-to-run spread,
  // and not worth doubling the shared memory for. The loop is not cvt-bound. What it is bound by
  // is the dependent chain (10 serial FMAs, then 5 serial shuffles, per r) and the sigmoid's MUFU
  // work, and staging wider touches neither.
  for (int i = lane; i < npair; i += 32) us[i] = ur[i];   // stage once
  __syncwarp();
  const half* uraw = &ush[(size_t)warp * rank];
  float* gp = gated + (size_t)j;
  const float* np = normed + (size_t)j;
  for (int r = 0; r < R; ++r) {
    const float* tr = t + (size_t)r * rank;        // t is FP32 (it holds silu of the down dots)
    float acc = 0.f;
    if (NITER) {
      #pragma unroll
      for (int u = 0; u < NITER; ++u) {
        const int i = lane + 32 * u;
        if (i < npair) {
          const float2 a = ((const float2*)tr)[i];
          const float2 b = __half22float2(us[i]);
          acc = fmaf(a.x, b.x, acc);
          acc = fmaf(a.y, b.y, acc);
        }
      }
    } else {
      for (int i = lane; i < npair; i += 32) {
        const float2 a = ((const float2*)tr)[i];
        const float2 b = __half22float2(us[i]);
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
      }
    }
    if (rank & 1) {                                // odd rank: one scalar tail lane
      if (lane == 0) acc = fmaf(tr[rank - 1], __half2float(uraw[rank - 1]), acc);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) gp[(size_t)r * HD] = sigmoidf_(acc) * np[(size_t)r * HD];
  }
}

// mixed[r][d] = (1/H) * sum_h gated[r][h][d].
//
// It reads only 50 KB and is 27% of the hc mix, which looks like a block-count problem: the grid is
// (R, ceil(D/block)) = 10 blocks at decode. It is not. A grid-stride rewrite at 512 blocks measured
// 14.10 us against 14.13 - no change - and would launch FEWER blocks than this one does at prefill,
// where R is the chunk size. Reverted. Whatever this stage's 14 us is, it is not launch width.
__global__ void gr_fin_kernel(const float* __restrict__ gated, int R, int H, int D,
                              float* __restrict__ mixed) {
  const int r = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= D) return;
  const float* gr = gated + (size_t)r * H * D;
  float acc = 0.f;
  for (int h = 0; h < H; h++) acc += gr[(size_t)h * D + d];
  mixed[(size_t)r * D + d] = acc / (float)H;
}

__global__ void gr_post_kernel(const float* __restrict__ normed, const half* __restrict__ inject,
                               int R, int H, int D, float* __restrict__ post) {
  const int r = blockIdx.x;
  const int h = blockIdx.y;
  const int HD = H * D;
  const float* nr = normed + (size_t)r * HD;
  const half* ir = inject + (size_t)h * HD;
  float acc = 0.f;
  for (int j = threadIdx.x; j < HD; j += blockDim.x) acc = fmaf(nr[j], __half2float(ir[j]), acc);
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  __shared__ float red[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    const int nwarp = (blockDim.x + 31) >> 5;
    float v = (lane < nwarp) ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) post[(size_t)r * H + h] = 2.0f * sigmoidf_(v / (float)H);
  }
}

// `post` may be null, matching gr_mix's contract (`if (post && inject)`), which is what the MTP draft
// relies on: it keeps no post gate, because with a single token the stream identity is irrelevant and
// the reference chain re-derives the stack from the tap each step. A null post means "no gating", so
// the sublayer output is applied unscaled. This used to dereference the pointer unconditionally, which
// made every draft step fault with an illegal access to 0x0 - the draft path had never actually run.
__global__ void gr_apply_kernel(float* __restrict__ streams, const float* __restrict__ y,
                                const float* __restrict__ post, int R, int H, int D) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n = (size_t)R * H * D;
  if (i >= n) return;
  const int r = (int)(i / ((size_t)H * D));
  const int h = (int)((i / D) % H);
  const float g = post ? post[(size_t)r * H + h] : 1.0f;
  streams[i] = fmaf(g, y[(size_t)r * D + i % D], streams[i]);
}

// HELIOS_GR_UP=0 pins gr_up_kernel to its GENERIC form - the runtime-trip-count inner loop, which
// is byte for byte the pre-template kernel. A/B lever, and the switch test_aux uses to hold the
// two against each other with memcmp: the property that makes the template safe to ship is EXACT
// equality, not closeness, so that is the thing worth a test.
bool g_force_legacy_up = getenv("HELIOS_GR_UP") && atoi(getenv("HELIOS_GR_UP")) == 0;
bool gr_legacy_up_forced() { return g_force_legacy_up; }

// Same lever for the `dots` stage, and the same reason: the two kernels return the same t bit for
// bit, so the only thing worth testing is EXACT equality, not a tolerance.
bool g_force_legacy_dots = getenv("HELIOS_GR_DOTS") && atoi(getenv("HELIOS_GR_DOTS")) == 0;
bool gr_legacy_dots_forced() { return g_force_legacy_dots; }


}  // namespace

// Defined below; only the fp32 path's outputs are meaningful to dump, so it runs after the kernels.
static void dump_mixer_once(const float* streams, const half* norm_raw, const half* down,
                            const half* up, const half* inject, const float* mixed,
                            const float* post, int R, int H, int D, int rank, float eps, Stream s);

void gr_mix(const float* streams, const half* norm_raw, const half* down, const half* up,
            const half* inject, int R, int H, int D, int rank, float eps, float* mixed,
            float* post, Stream s) {
  if (R <= 0) return;
  const size_t HD = (size_t)H * D;
  // Grow-and-retain: the temporaries are (R*HD) and (R*rank) floats and every call site has the
  // same R, so they are allocated once, which keeps the kernel signature (and both call sites)
  // unchanged.
  //
  // PER DEVICE, which the two-card layer pipeline (HELIOS_PIPELINE=1) makes mandatory: the cache
  // used to be four function-level statics, on the stated assumption that "this is a per-process,
  // single-device engine". With layers split across two cards the second card's gr_mix would be
  // handed the FIRST card's pointers - an illegal access, reported some launches later at whatever
  // CUDA call noticed. Each device therefore owns its block and grows it independently; a grow on
  // one card must not free or move the other card's.
  struct Cache { float *normed = nullptr, *tbuf = nullptr, *gated = nullptr,
                 *sums = nullptr;                 // per-(row, stream) RMS sums, R*H floats
                 half *nrm16 = nullptr, *t16 = nullptr;   // TC path: normed as fp16 (R, H*D), and
                                                  // the rank latent (R, rank)
                 float *rmr = nullptr, *ph = nullptr;   // TC path: (R, H) and (R, H, H)
                 size_t cap_hd = 0, cap_r = 0; };
  static Cache cache[2];
  int dev = 0;
  HELIOS_CUDA_CHECK(cudaGetDevice(&dev));
  Cache& cz = cache[dev >= 0 && dev < 2 ? dev : 0];
  if ((size_t)R * HD > cz.cap_hd || (size_t)R * rank > cz.cap_r) {
    if (cz.normed) { cudaFree(cz.normed); cudaFree(cz.tbuf); cudaFree(cz.gated); cudaFree(cz.sums); }
    if (cz.t16) { cudaFree(cz.nrm16); cudaFree(cz.t16); cudaFree(cz.rmr); cudaFree(cz.ph); }
    cz.cap_hd = (size_t)R * HD;
    cz.cap_r = (size_t)R * rank;
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.normed, cz.cap_hd * sizeof(float)));
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.tbuf, cz.cap_r * sizeof(float)));
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.gated, cz.cap_hd * sizeof(float)));
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.sums, (size_t)R * H * sizeof(float)));
    // The TC path's temporaries live here rather than in gr_mix_tc for the same reason the others
    // do: one grow-and-retain per device, warmed up before any CUDA graph capture. nrm16 is the
    // dots GEMM's A operand (normed narrowed to fp16, 21 MB at R=1024) and t16/rmr/ph are small.
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.nrm16, cz.cap_hd * sizeof(half)));
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.t16, cz.cap_r * sizeof(half)));
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.rmr, (size_t)R * H * sizeof(float)));
    HELIOS_CUDA_CHECK(cudaMalloc(&cz.ph, (size_t)R * H * H * sizeof(float)));
  }
  float* const normed = cz.normed;
  float* const tbuf = cz.tbuf;
  float* const gated = cz.gated;
  float* const sums = cz.sums;

  // HELIOS_MIXER_TC=1. Both paths get the same workspace, so the flag can be flipped between calls
  // without a realloc; the dispatch is on R as well because the TC kernels tile 64 rows and the
  // fp32 decode kernels are built for R=1.
  if (gr_mixer_tc() && gr_mix_tc_applicable(R, H, D, rank, gr_mixer_tc_min_r())) {
    gr_mix_tc(streams, norm_raw, down, up, inject, R, H, D, rank, eps, mixed, post, cz.nrm16,
              cz.t16, cz.rmr, cz.ph, s);
    dump_mixer_once(streams, norm_raw, down, up, inject, mixed, post, R, H, D, rank, eps, s);
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
    return;
  }

  const int threads = 256;
  // Per-stage attribution: the five stages are memory-bound but launched with few blocks (gr_norm gets
  // R, gr_dots gets (rank/8, R) = 3 at decode), so each reads its weights at a small fraction of
  // device bandwidth. Fusing them removed four launches and changed nothing, which rules out launch
  // overhead and points at block count. Measure before rewriting: HELIOS_GMIXPROF=1.
  static double gs[5] = {0};
  static long gn = 0;
  const bool gp = getenv("HELIOS_GMIXPROF") != nullptr;
  static cudaEvent_t ge[6];
  static bool ge_ready = false;
  if (gp && !ge_ready) { for (int i = 0; i < 6; i++) cudaEventCreate(&ge[i]); ge_ready = true; }
  int ge_n = 0;
  auto gmk = [&]() { if (gp && ge_n < 6) cudaEventRecord(ge[ge_n++], s); };
  struct GOut { ~GOut() {
      if (getenv("HELIOS_GMIXPROF") && gn > 0) {
        fprintf(stderr, "[gmix] per-call us: norm=%.2f dots=%.2f up=%.2f fin=%.2f post=%.2f "
                        "apply=%.2f (calls=%ld)\n", gs[0]/gn*1000, gs[1]/gn*1000, gs[2]/gn*1000,
                gs[3]/gn*1000, gs[4]/gn*1000, gs[5]/gn*1000, gn);
      }
    } } gout;
  if (gp) { cudaStreamSynchronize(s); gmk(); }
  // One sync per CALL, not per stage: a cudaStreamSynchronize after each of five tiny kernels costs
  // ~15-20 us of latency each, which swamps every stage equally and made the per-stage numbers
  // useless for attribution (dots read 27.24 before and 27.49 after an 8x increase in its block count).
  // Record the events back to back and read the deltas once.
  auto gsum_all = [&]() {
    if (!gp || ge_n < 2) return;
    cudaStreamSynchronize(s);
    for (int i = 1; i < ge_n; i++) {
      float ms = 0;
      if (cudaEventElapsedTime(&ms, ge[i - 1], ge[i]) == cudaSuccess) gs[i - 1] += ms;
    }
    gn++;
    ge_n = 0;
  };
  // These two reductions are the only stages whose grid is fixed at (R, H) = 4 blocks at decode,
  // so at R=1 they occupy 4 of 82 SMs and each block walks its whole row in blockDim-sized strides
  // - 40 dependent memory rounds for gr_post's 10,240 elements. ncu put gr_post at 16.85 us of PURE
  // kernel time for 60 KB of traffic, so this is row-parallelism, not launch overhead. Widening the
  // block cuts the strided walk in quarters; the per-thread accumulation order and the butterfly are
  // untouched, so the result is bitwise identical.
  const int red_threads = 1024;
  {
    dim3 hg((unsigned)R, (unsigned)H);
    gr_hsum_kernel<<<hg, red_threads, 0, s>>>(streams, H, D, sums);
    const size_t st = (size_t)R * H * D;
    int sb = (int)((st + threads - 1) / threads);
    // The old `if (sb > 128) sb = 128` cap was inert at decode, where this computes to 40, and
    // only ever bound at prefill: R=256 wants 10,240 blocks for R*H*D and got 128 - 1.56 blocks/SM
    // on an 82-SM card, each thread striding 80 elements on a pure streaming kernel. It was tuned
    // against decode and silently crippled prefill. The grid-stride loop above already covers
    // any block count, so one element per thread is correct at both R.
    gr_scale_kernel<<<sb, threads, 0, s>>>(streams, norm_raw, sums, R, H, D, eps, normed);
  }
  gmk();
  gmk();

  {
    // Three `dots` stages, all producing the same t bit for bit. gr_dots2_kernel is the
    // register-tiled form (see its comment: the legacy one is bound by shared-memory ISSUE, not by
    // arithmetic, and does two LDS per FMA). It needs rank to be a multiple of its 32-column tile
    // so no block stages a ragged `down` column, and HD a multiple of its 16-deep k chunk.
    // HELIOS_GR_DOTS=0 pins the legacy kernel - the A/B lever, and the switch test_aux uses to
    // hold the two against each other with memcmp.
    constexpr int DTR = 8, DTI = 32, DBK = 128, DTP = DBK + 2;
    const bool dots2 = R >= 8 && rank % kDots2Cols == 0 && (H * D) % kDots2K == 0 &&
                       !gr_legacy_dots_forced();
    if (dots2) {
      dim3 g((unsigned)(rank / kDots2Cols), (unsigned)((R + kDots2Rows - 1) / kDots2Rows));
      const size_t sm2 = (size_t)kDots2Rows * kDots2K * sizeof(float) +
                         (size_t)kDots2Cols * kDots2KP * sizeof(half);
      gr_dots2_kernel<<<g, kDots2Threads, sm2, s>>>(normed, down, R, H, D, rank, tbuf);
    } else if (R >= 8) {
      dim3 dots_grid((unsigned)((rank + DTI - 1) / DTI), (unsigned)((R + DTR - 1) / DTR));
      const size_t sm = (size_t)DTR * DBK * sizeof(float) + (size_t)DTI * DTP * sizeof(half);
      gr_dots_kernel<<<dots_grid, DTR * DTI, sm, s>>>(normed, down, R, H, D, rank, tbuf);
    } else {
      // Left at `threads`, not red_threads: measured neutral-to-negative (46.53/47.19/46.98 against
      // 47.36/47.35/47.24). This kernel has 320 blocks, so its latency is already hidden - unlike the
      // two (R,H) reductions, which have 4 - and the event timer puts it at 3.10 us, not the 31 us an
      // ncu run suggested. ncu serialises kernels, so its per-kernel figures are for RANKING, never
      // for absolute cost; the event timer is the one that tracks end-to-end time.
      dim3 dots_grid((unsigned)rank, (unsigned)R);
      gr_dots_small_kernel<<<dots_grid, threads, 0, s>>>(normed, down, R, H, D, rank, tbuf);
    }
  }

constexpr int kUpMaxIter = 8;      // widest specialisation of gr_up_kernel's inner loop

  // One warp per output element, so a block covers (threads/32) of them - not `threads`. The grid
  // is 1-D: the token loop lives inside the kernel, which is what makes the form
  // weight-stationary. Each block stages its TJ rows of `up` (TJ * rank halfs) in shared memory.
  // Swept (HELIOS_UP_THREADS) because the structure is "one warp per output element", so block
  // width sets how many `up` rows a block stages at once. It is NOT the lever: 128/256/512 give
  // 47.51/47.43/47.43 tok/s (inside noise) and 1024 regresses to 45.71. Left at `threads` (256) and
  // kept sweepable so the negative result stays reproducible rather than folklore.
  const int up_threads = getenv("HELIOS_UP_THREADS") ? atoi(getenv("HELIOS_UP_THREADS")) : threads;
  const int warps_per_block = up_threads >> 5;
  const size_t up_smem = (size_t)warps_per_block * rank * sizeof(half);
  dim3 up_grid((unsigned)((HD + warps_per_block - 1) / warps_per_block));
  // The inner loop runs ceil(rank/2 / 32) times; dispatch on that so the trip count is a
  // compile-time constant and the loop unrolls (see gr_up_kernel). NITER = 0 is the generic
  // runtime-loop form, used above kUpMaxIter and whenever HELIOS_GR_UP=0 pins it for A/B.
  const int up_iter = (int)(((size_t)(rank >> 1) + 31) / 32);
  #define HELIOS_LAUNCH_UP(N)                                                            \
    gr_up_kernel<N><<<up_grid, up_threads, up_smem, s>>>(normed, tbuf, up, R, H, D, rank, gated)
  if (gr_legacy_up_forced() || up_iter > kUpMaxIter) {
    HELIOS_LAUNCH_UP(0);
  } else {
    switch (up_iter) {
      case 1: HELIOS_LAUNCH_UP(1); break;
      case 2: HELIOS_LAUNCH_UP(2); break;
      case 3: HELIOS_LAUNCH_UP(3); break;
      case 4: HELIOS_LAUNCH_UP(4); break;
      case 5: HELIOS_LAUNCH_UP(5); break;
      case 6: HELIOS_LAUNCH_UP(6); break;
      case 7: HELIOS_LAUNCH_UP(7); break;
      case 8: HELIOS_LAUNCH_UP(8); break;
      default: HELIOS_LAUNCH_UP(0); break;
    }
  }
  #undef HELIOS_LAUNCH_UP
  gmk();

  dim3 fin_grid((unsigned)R, (unsigned)((D + threads - 1) / threads));
  gr_fin_kernel<<<fin_grid, threads, 0, s>>>(gated, R, H, D, mixed);
  gmk();

  if (post && inject) {
    dim3 post_grid((unsigned)R, (unsigned)H);
    gr_post_kernel<<<post_grid, red_threads, 0, s>>>(normed, inject, R, H, D, post);
  }
  gmk();
  gsum_all();
  dump_mixer_once(streams, norm_raw, down, up, inject, mixed, post, R, H, D, rank, eps, s);
  HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------------------------
// HELIOS_DUMP_MIXER=<path>: write one real call's inputs and fp32-path outputs to a file, once.
//
// The tensor-core path is judged on ACCURACY, and accuracy against synthetic Gaussian streams
// says nothing: what matters is the dynamic range, the outliers and the near-cancellation that
// real activations have and noise does not. So the harness in test/gr_mix_tc.cpp replays a real
// call rather than a fabricated one, and this is where the real call comes from. It is written
// from inside gr_mix rather than from the runner so that it cannot drift from the operator's own
// argument order, and it refuses to run under stream capture (a host file write there would fault
// the capture) - the first call of a run is always the uncaptured warmup.
static void dump_mixer_once(const float* streams, const half* norm_raw, const half* down,
                            const half* up, const half* inject, const float* mixed,
                            const float* post, int R, int H, int D, int rank, float eps, Stream s)
{
  static int done = 0;
  if (done) return;
  const char* path = getenv("HELIOS_DUMP_MIXER");
  if (!path || R < 256) return;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) return;
  done = 1;
  const size_t HD = (size_t)H * D;
  std::vector<float> hs((size_t)R * HD), hm((size_t)R * D), hp((size_t)R * H);
  std::vector<half> hn(HD), hd((size_t)rank * HD), hu(HD * rank), hi((size_t)H * HD);
  const size_t b = (size_t)R * HD * sizeof(float);
  HELIOS_CUDA_CHECK(cudaMemcpy(hs.data(), streams, b, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(hn.data(), norm_raw, HD * 2, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(hd.data(), down, hd.size() * 2, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(hu.data(), up, hu.size() * 2, cudaMemcpyDeviceToHost));
  if (inject)
    HELIOS_CUDA_CHECK(cudaMemcpy(hi.data(), inject, hi.size() * 2, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(hm.data(), mixed, (size_t)R * D * sizeof(float),
                                cudaMemcpyDeviceToHost));
  if (post)
    HELIOS_CUDA_CHECK(cudaMemcpy(hp.data(), post, (size_t)R * H * sizeof(float),
                                  cudaMemcpyDeviceToHost));
  FILE* f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "[gr_dump] cannot open %s\n", path); return; }
  int32_t hdr[8] = {(int32_t)0x584d5447, R, H, D, rank, 0, 0, 0};
  memcpy(&hdr[5], &eps, 4);
  fwrite(hdr, 4, 8, f);
  fwrite(hs.data(), 4, hs.size(), f);
  fwrite(hn.data(), 2, hn.size(), f);
  fwrite(hd.data(), 2, hd.size(), f);
  fwrite(hu.data(), 2, hu.size(), f);
  fwrite(hi.data(), 2, hi.size(), f);
  fwrite(hm.data(), 4, hm.size(), f);
  fwrite(hp.data(), 4, hp.size(), f);
  fclose(f);
  fprintf(stderr, "[gr_dump] wrote %s: R=%d H=%d D=%d rank=%d (%.1f MB)\n", path, R, H, D, rank,
          (double)(b + hd.size() * 2 + hu.size() * 2) / 1048576.0);
}

void gr_force_legacy_up(bool on) { g_force_legacy_up = on; }

void gr_force_legacy_dots(bool on) { g_force_legacy_dots = on; }

void gr_apply(float* streams, const float* y, const float* post, int R, int H, int D, Stream s) {
  const size_t n = (size_t)R * H * D;
  const int threads = 256;
  gr_apply_kernel<<<(int)((n + threads - 1) / threads), threads, 0, s>>>(streams, y, post, R, H, D);
}

}} // namespace helios::aux
