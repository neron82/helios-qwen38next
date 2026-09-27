#include "engine/glue2.cuh"
#include "cuda/aux/gr_mix.cuh"
#include <cstdlib>
#include <cuda_fp16.h>

namespace helios { namespace glue {

// ---------------------------------------------------------------- fp16 GEMM (w row-major [N,K])
// Block computes a 16(M) x 16(N) output tile, smem-tiled over K in 16-wide chunks.
__global__ void gemm_nt_f16_k(const half* __restrict__ x, const half* __restrict__ w,
                              void* __restrict__ y, int M, int N, int K, bool y_fp32, bool add) {
  __shared__ half xs[16][17], ws[16][17];
  int tm = blockIdx.y * 16, tn = blockIdx.x * 16;
  int tx = threadIdx.x, ty = threadIdx.y;        // 16x16 threads
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    if (tm + ty < M && k0 + tx < K) xs[ty][tx] = x[(size_t)(tm + ty) * K + k0 + tx];
    else xs[ty][tx] = __float2half(0.f);
    // ws[n_local][k_local] = w[tn + n_local][k0 + k_local]
    if (tn + ty < N && k0 + tx < K) ws[ty][tx] = w[(size_t)(tn + ty) * K + k0 + tx];
    else ws[ty][tx] = __float2half(0.f);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 16; k++) acc += __half2float(xs[ty][k]) * __half2float(ws[tx][k]);
    __syncthreads();
  }
  int m = tm + ty, n = tn + tx;
  if (m >= M || n >= N) return;
  if (y_fp32) {
    float* yf = (float*)y;
    if (add) atomicAdd(&yf[(size_t)m * N + n], acc);
    else yf[(size_t)m * N + n] = acc;
  } else {
    ((half*)y)[(size_t)m * N + n] = __float2half_rn(acc);
  }
}

// ---------------------------------------------------------------- fp16 GEMM, tensor-core path
//
// The kernel above is 16x16 of output per 256 threads with one scalar half->float convert and FMA
// per MAC. On the PLE's two prefill shapes - key_proj (M=1024, N=10240, K=2560) and value_proj
// (M=1024, N=2560, K=2560) - it measures 23.67 ms and 5.95 ms per call, i.e. 2.27 TFLOPS. The block
// shape is the problem, not the instruction mix: 16x16 is a 1:1 MAC:load ratio with no reuse of
// either operand inside a block, and the tensor cores sit idle through all of it.
//
// This is the structure gr_mix_tc.cu already proves in-tree, specialised for the NT layout the
// checkpoint weights are stored in: A = x (M,K) row-major and B = w (N,K) row-major, so B is
// already the column-major (K,N) operand mma.row.col wants and neither side is ever transposed.
// 128x128 block tile, 8 warps in a 2x4 grid (64x32 per warp: 4 m-tiles x 4 n-tiles = 16
// mma.m16n8k16 and 64 fp32 accumulators per lane), k staged 32 deep in a double buffer.
// Measured on the same two shapes: 0.92 ms and 0.24 ms, 58.3 and 55.1 TFLOPS - 25.7x and 24.5x the
// scalar path, and 82% of the 71 TFLOPS fp16-tensor-with-fp32-accumulate dense peak of this card.
//
// NUMERICS: this is a different numerical path, not a faster spelling of the scalar one. Both
// accumulate in fp32, but the tensor core sums each 16-term contraction in its own order, so the
// fp32 result differs in the last bits and the fp16 store can land on the other side of a rounding
// boundary. It therefore follows the engine's tensor-core switch, HELIOS_MIXER_TC: on by default,
// and off exactly when the engine is pinned to its scalar reference oracle (HELIOS_GEMM_MMA
// overrides it in either direction).
namespace {

constexpr int kGemmBM = 128, kGemmBN = 128, kGemmBK = 32, kGemmSP = kGemmBK + 8;
constexpr int kGemmWarps = 8, kGemmWM = 2, kGemmWN = 4;
constexpr int kGemmMT = kGemmBM / kGemmWM / 16;   // m-tiles per warp (4)
constexpr int kGemmNT = kGemmBN / kGemmWN / 8;    // n-tiles per warp (4)
// 40 halfs of pitch = 80 B = 20 words, so row r starts on bank 20r and the eight ldmatrix row
// addresses of one fragment cover banks 0-3, 20-23, 8-11, 28-31, 16-19, 4-7, 24-27, 12-15 -
// all 32 exactly once, for both the .x4 (A) and .x2 (B) loads.
constexpr int kGemmSmem = 2 * (kGemmBM * kGemmSP + kGemmBN * kGemmSP) * (int)sizeof(half);
static_assert(kGemmMT == 4 && kGemmNT == 4 && kGemmSmem <= 48 * 1024,
              "the launch below is written for this block shape and needs no smem opt-in");

__device__ __forceinline__ void mma16816_g(float (&c)[4], const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned smem_addr_g(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

// One 16x16 A fragment out of a [row][k] tile with k contiguous: lanes 0-7 -> rows 0-7, 8-15 ->
// rows 8-15, 16-23 -> rows 0-7 at k+8, 24-31 -> rows 8-15 at k+8.
__device__ __forceinline__ void gemm_frag_a(uint32_t (&a)[4], const half* tile, int m0, int k0,
                                            int lane) {
  const int row = m0 + (lane & 7) + 8 * ((lane >> 3) & 1);
  const int k = k0 + 8 * (lane >> 4);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem_addr_g(tile + (size_t)row * kGemmSP + k)));
}

// One 16x8 B fragment, lanes 0-15 supplying the row addresses. The checkpoint holds B as
// (N,K) row-major, which is exactly the (K,N) column-major operand: lane i ends up with
// tile[n0 + i/4][k0 + 2*(i%4)] and [.. +8], which is what mma.row.col wants.
__device__ __forceinline__ void gemm_frag_b(uint32_t (&b)[2], const half* tile, int n0, int k0,
                                            int lane) {
  const int l2 = lane & 15;
  const int row = n0 + (l2 & 7);
  const int k = k0 + 8 * (l2 >> 3);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(smem_addr_g(tile + (size_t)row * kGemmSP + k)));
}

template <bool ADD>
__global__ __launch_bounds__(kGemmWarps * 32)
void gemm_nt_mma_k(const half* __restrict__ x, const half* __restrict__ w, void* __restrict__ y,
                   int M, int N, int K, bool y_fp32, int tiles_m) {
  extern __shared__ __align__(16) half smem[];
  half* As = smem;                                // [2][kGemmBM][kGemmSP]
  half* Bs = As + 2 * kGemmBM * kGemmSP;          // [2][kGemmBN][kGemmSP]

  // M is the fastest-varying axis of the 1-D grid, so the tiles_m consecutive blocks that share
  // one 128-row B tile are also the blocks the scheduler puts on the card together. B is 52 MB
  // at the key_proj shape against a 6 MB L2, and a row-major grid would re-stream all of it once
  // per M-tile row (419 MB); this way it is streamed once and the small operand (A, 5 MB) is what
  // L2 re-reads.
  const int m0 = (blockIdx.x % tiles_m) * kGemmBM;
  const int n0 = (blockIdx.x / tiles_m) * kGemmBN;
  if (m0 >= M || n0 >= N) return;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / kGemmWN, wn = warp % kGemmWN;
  // 32 halfs per row = 4 threads x uint4, so 256 threads cover 64 rows and two passes the tile.
  const int arow = tid >> 2, ak = (tid & 3) << 3;

  float acc[kGemmMT][kGemmNT][4];
#pragma unroll
  for (int i = 0; i < kGemmMT; ++i)
#pragma unroll
    for (int j = 0; j < kGemmNT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  uint4 areg[2], breg[2];
  auto load_stage = [&](int k0) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int row = arow + 64 * u;
      const bool kok = (k0 + ak) < K;
      const half* ap = x + (size_t)(m0 + row) * K + k0 + ak;
      const half* bp = w + (size_t)(n0 + row) * K + k0 + ak;
      areg[u] = (kok && m0 + row < M) ? *(const uint4*)ap : make_uint4(0, 0, 0, 0);
      breg[u] = (kok && n0 + row < N) ? *(const uint4*)bp : make_uint4(0, 0, 0, 0);
    }
  };
  auto store_stage = [&](int st) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      *(uint4*)(As + st * kGemmBM * kGemmSP + (arow + 64 * u) * kGemmSP + ak) = areg[u];
      *(uint4*)(Bs + st * kGemmBN * kGemmSP + (arow + 64 * u) * kGemmSP + ak) = breg[u];
    }
  };

  const int nstage = (K + kGemmBK - 1) / kGemmBK;
  load_stage(0);
  store_stage(0);
  __syncthreads();
  for (int s = 0; s < nstage; ++s) {
    if (s + 1 < nstage) {
      load_stage((s + 1) * kGemmBK);
      store_stage((s + 1) & 1);
    }
    __syncthreads();
    const half* at = As + (s & 1) * kGemmBM * kGemmSP;
    const half* bt = Bs + (s & 1) * kGemmBN * kGemmSP;
#pragma unroll
    for (int ks = 0; ks < kGemmBK; ks += 16) {
      uint32_t af[kGemmMT][4], bf[kGemmNT][2];
#pragma unroll
      for (int i = 0; i < kGemmMT; ++i) gemm_frag_a(af[i], at, wm * 64 + i * 16, ks, lane);
#pragma unroll
      for (int j = 0; j < kGemmNT; ++j) gemm_frag_b(bf[j], bt, wn * 32 + j * 8, ks, lane);
#pragma unroll
      for (int i = 0; i < kGemmMT; ++i)
#pragma unroll
        for (int j = 0; j < kGemmNT; ++j) mma16816_g(acc[i][j], af[i], bf[j]);
    }
    __syncthreads();
  }

  half* yh = (half*)y;
  float* yf = (float*)y;
#pragma unroll
  for (int i = 0; i < kGemmMT; ++i)
#pragma unroll
    for (int j = 0; j < kGemmNT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = m0 + wm * 64 + i * 16 + (lane >> 2) + 8 * (q >> 1);
        const int n = n0 + wn * 32 + j * 8 + 2 * (lane & 3) + (q & 1);
        if (m >= M || n >= N) continue;
        if (y_fp32) {
          if (ADD) atomicAdd(&yf[(size_t)m * N + n], acc[i][j][q]);
          else yf[(size_t)m * N + n] = acc[i][j][q];
        } else {
          // The scalar kernel never accumulated into an fp16 output, so this does not either.
          yh[(size_t)m * N + n] = __float2half_rn(acc[i][j][q]);
        }
      }
}

}  // namespace

// ---------------------------------------------------------------------------
// Single-row GEMV: y[n] = dot(x[0..K), w[n][0..K)). One warp per output with the lanes splitting
// K, so the weight row is read coalesced (32 lanes x half2 = 64 halves per transaction group).
// The tiled kernel above is 100x slower at M=1: it computes a 16x16 output tile of which 15 rows
// are discarded, with a scalar half->float conversion per MAC. At 1 token per step the engine
// calls this ~8 times per layer, and it measured as 18.2 of the KDA layer's 23.1ms per token.
__global__ void gemv_nt_f16_k(const half* __restrict__ x, const half* __restrict__ w,
                              void* __restrict__ y, int N, int K, bool y_fp32) {
  extern __shared__ half xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (wid >= N) return;
  const half* wr = w + (size_t)wid * K;
  float acc = 0.f;
  for (int k = lane * 2; k + 1 < K; k += 64) {
    float2 xf = __half22float2(*(const half2*)(xs + k));
    float2 wf = __half22float2(*(const half2*)(wr + k));
    acc += xf.x * wf.x + xf.y * wf.y;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    if (y_fp32) ((float*)y)[wid] = acc;
    else ((half*)y)[wid] = __float2half_rn(acc);
  }
}


// ---------------------------------------------------------------- tiled f16 NN GEMM
// C[M,N] = A[M,K] * B[K,N], all fp16 in / fp16 out with fp32 accumulation. 64x64 output tile per
// block, 256 threads, 4x4 outputs per thread, K staged in 16-wide smem tiles with half2 loads.
// The scalar kernels this replaces (o_absorb_t, the KDA low-rank projections, the indexer
// projections) each run at 0.6-1.5 TFLOPS because they issue one convert+FMA per MAC; measured
// 8192-row prefill shapes make them ~4s of a 28s batch. B must be [K,N] row-major (coalesced in N):
// callers hold [N,K] weights, so the engine builds transposed copies once at init.
#define GNN_TM 64
#define GNN_TN 64
#define GNN_TK 16
__global__ void gemm_nn_f16_k(const half* __restrict__ A, const half* __restrict__ B,
                              void* __restrict__ Cv, int M, int N, int K, int lda, int ldc,
                              bool c_fp32) {
  half* C = nullptr; float* Cf = nullptr;
  if (c_fp32) Cf = (float*)Cv; else C = (half*)Cv;
  __shared__ half as[GNN_TM][GNN_TK + 8];
  __shared__ half bs[GNN_TK][GNN_TN + 8];
  int m0 = blockIdx.y * GNN_TM, n0 = blockIdx.x * GNN_TN;
  // 4 rows x 4 cols per thread (256 threads). An 8x4 variant benched 4-5x faster but was
  // non-deterministic (the same kernel reported all-wrong, 0.8%-wrong and 99%-wrong across runs),
  // and it bought nothing in-engine, so the 4x4 form stays until the race is found.
  int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;      // 16x16 threads
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += GNN_TK) {
    // stage A: 64 rows x 16 k   (2624 halves / 256 threads)
    for (int i = threadIdx.x; i < GNN_TM * GNN_TK; i += 256) {
      int r = i >> 4, c = i & 15;
      int gm = m0 + r, gk = k0 + c;
      as[r][c] = (gm < M && gk < K) ? A[(size_t)gm * lda + gk] : __float2half(0.f);
    }
    // stage B: 16 k x 64 n
    for (int i = threadIdx.x; i < GNN_TK * GNN_TN; i += 256) {
      int kk = i >> 6, n = i & 63;
      int gk = k0 + kk, gn = n0 + n;
      bs[kk][n] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : __float2half(0.f);
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < GNN_TK; kk++) {
      // A fragment: 4 ROWS at column kk (an earlier revision walked along k here and paired the wrong
      // rows with the weights - caught by the greedy-output comparison).
      const half* brow = &bs[kk][tx * 4];
      float a[4], b[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) a[i] = __half2float(as[ty * 4 + i][kk]);
      #pragma unroll
      for (int j = 0; j < 4; j++) b[j] = __half2float(brow[j]);
      #pragma unroll
      for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int gm = m0 + ty * 4 + i;
    if (gm >= M) continue;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int gn = n0 + tx * 4 + j;
      if (gn >= N) continue;
      if (c_fp32) Cf[(size_t)gm * ldc + gn] = acc[i][j];
      else C[(size_t)gm * ldc + gn] = __float2half_rn(acc[i][j]);
    }
  }
}

// src [rows][cols] f16 -> dst [cols][rows] f16. One thread per element: the smem-tiled version I
// first wrote had four separate indexing bugs (inverted fragment pairing, swapped load bounds,
// swapped grid axes) and the bench self-test below caught each one. This runs once at init on ~230MB,
// so the strided side costs a couple of seconds and correctness-by-construction is worth more.
__global__ void transpose_f16_k(const half* __restrict__ src, half* __restrict__ dst,
                                long long n, int rows, int cols) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int r = (int)(i / cols), c = (int)(i % cols);
  dst[(size_t)c * rows + r] = src[i];
}

void transpose_f16_half(void* dst, const half* src, int rows, int cols, Stream s) {
  long long n = (long long)rows * cols;
  if (n <= 0) return;
  long long blocks = (n + 255) / 256;
  transpose_f16_k<<<(unsigned)blocks, 256, 0, s>>>(src, (half*)dst, n, rows, cols);
  cuda_check(cudaPeekAtLastError());
}

void gemm_nn_f16(void* C, const half* A, const half* B, int M, int N, int K, Stream s, int ldc,
                 bool c_fp32, int lda) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (ldc <= 0) ldc = N;
  if (lda <= 0) lda = K;
  dim3 grid((N + GNN_TN - 1) / GNN_TN, (M + GNN_TM - 1) / GNN_TM, 1);
  gemm_nn_f16_k<<<grid, 256, 0, s>>>(A, B, C, M, N, K, lda, ldc, c_fp32);
  cuda_check(cudaPeekAtLastError());
}

void gemv_nt_f16(void* y, const half* x, const half* w, int N, int K, bool y_fp32, Stream s) {
  int block = 256;
  int grid = (N + block / 32 - 1) / (block / 32);
  gemv_nt_f16_k<<<grid, block, (size_t)K * 2, s>>>(x, w, y, N, K, y_fp32);
  cuda_check(cudaPeekAtLastError());
}

// Is the tensor-core GEMM allowed? It follows the engine-wide tensor-core switch so that the one
// variable which pins helios to its scalar reference oracle (HELIOS_MIXER_TC=0) also pins this
// kernel, and HELIOS_GEMM_MMA=0/1 overrides it in either direction. See the note on
// gemm_nt_mma_k: the two paths agree to fp32 but are not bit-identical.
bool gemm_mma_enabled() {
  static const bool on = [] {
    if (const char* e = getenv("HELIOS_GEMM_MMA")) return atoi(e) != 0;
    return aux::gr_mixer_tc();
  }();
  return on;
}

void gemm_nt_f16(void* y, const half* x, const half* w, int M, int N, int K,
                 bool y_fp32, bool add, Stream s) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  // Route the decode case (one token) to the GEMV above; K must be even for the half2 path.
  static const bool rc = getenv("HELIOS_GEMM_RC") == nullptr || atoi(getenv("HELIOS_GEMM_RC")) != 0;
  if (rc && M == 1 && !add && (K % 2) == 0) { gemv_nt_f16(y, x, w, N, K, y_fp32, s); return; }
  // Below half an M-tile of rows the 128x128 block is mostly empty and the tensor cores idle on
  // the guard lanes; that is only the speculative-verify shapes (n of 1..8), where this whole call
  // is a few microseconds either way, so they stay on the scalar kernel.
  if (gemm_mma_enabled() && M >= kGemmBM / 2 && K >= 16) {
    const int tiles_m = (M + kGemmBM - 1) / kGemmBM, tiles_n = (N + kGemmBN - 1) / kGemmBN;
    const int grid = tiles_m * tiles_n;
    if (add && y_fp32)
      gemm_nt_mma_k<true><<<grid, kGemmWarps * 32, kGemmSmem, s>>>(x, w, y, M, N, K, true,
                                                                   tiles_m);
    else
      gemm_nt_mma_k<false><<<grid, kGemmWarps * 32, kGemmSmem, s>>>(x, w, y, M, N, K, y_fp32,
                                                                    tiles_m);
    cuda_check(cudaPeekAtLastError());
    return;
  }
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  gemm_nt_f16_k<<<grid, block, 0, s>>>(x, w, y, M, N, K, y_fp32, add);
  cuda_check(cudaPeekAtLastError());
}

// src [M,F] fp32 -> dst [F,M] bf16 through a shared 32x32 tile. The straight version mapped one
// thread to one (m,f), read coalesced along f, then wrote dst[f*M+m] with a stride of M*2 B: a
// 2-byte store per thread into its own 32 B sector, so the write side moved 16x the bytes it
// needed and the whole transpose measured 109 GB/s against a 936 GB/s peak. Staging the tile gives
// the store side the same coalesced pattern the load side already had.
__global__ void transpose_f32_bf16_k(const float* __restrict__ src, unsigned short* __restrict__ dst,
                                     int M, int F) {
  __shared__ float tile[32][33];
  const int m0 = blockIdx.y * 32, f0 = blockIdx.x * 32;
  const int tid = threadIdx.x;
  // tile[a][b] is src row m0+a, column f0+b. The load walks it with consecutive tid on
  // consecutive b (contiguous in src); the store takes consecutive tid to consecutive b again,
  // which is consecutive in dst because dst is [F, M]. The one column of padding keeps the 32
  // lanes of each pass on 32 distinct banks.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = tid + i * 256;
    const int a = e >> 5, b = e & 31;
    tile[a][b] = (m0 + a < M && f0 + b < F) ? src[(size_t)(m0 + a) * F + f0 + b] : 0.f;
  }
  __syncthreads();
  // Self-report what this block actually holds, for the rows it was meant to load. Read once and
  // compared against src on the host, this separates "the load indexed wrongly", "the guard dropped
  // writes" and "the store addressed wrongly" in a single run - which six readings of this kernel did
  // not manage, and a marker test should always have come before the seventh.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = tid + i * 256;
    const int a = e >> 5, b = e & 31;
    // dst row index is m0+b and its column f0+a; guarding on `a` for both would let the ragged
    // last tile write past M and race the next block's rows.
    if (m0 + b < M && f0 + a < F)
      dst[(size_t)(f0 + a) * M + m0 + b] = __bfloat16_as_ushort(__float2bfloat16(tile[b][a]));
  }
}

void transpose_f32_bf16(const float* src, void* dst, int M, int F, Stream s) {
  dim3 g((F + 31) / 32, (M + 31) / 32);
  transpose_f32_bf16_k<<<g, 256, 0, s>>>(src, (unsigned short*)dst, M, F);
  cuda_check(cudaPeekAtLastError());
}

__global__ void cast_f16_bf16_k(const half* __restrict__ s, unsigned short* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) d[i] = __bfloat16_as_ushort(__float2bfloat16(__half2float(s[i])));
}
__global__ void cast_f32_bf16_k(const float* __restrict__ s, unsigned short* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) d[i] = __bfloat16_as_ushort(__float2bfloat16(s[i]));
}
void cast_f16_bf16(const half* src, void* dst, size_t n, Stream s) {
  cast_f16_bf16_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, (unsigned short*)dst, n);
  cuda_check(cudaPeekAtLastError());
}
void cast_f32_bf16(const float* src, void* dst, size_t n, Stream s) {
  cast_f32_bf16_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, (unsigned short*)dst, n);
  cuda_check(cudaPeekAtLastError());
}

__global__ void add_f32_inplace_k(float* __restrict__ a, const float* __restrict__ b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void sigmoid_f32_k(float* __restrict__ x, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.0f / (1.0f + __expf(-x[i]));
}
void sigmoid_f32(float* x, size_t n, Stream s) {
  sigmoid_f32_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, n);
  cuda_check(cudaPeekAtLastError());
}

void add_f32_inplace(float* a, const float* b, size_t n, Stream s) {
  add_f32_inplace_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(a, b, n);
  cuda_check(cudaPeekAtLastError());
}

__global__ void scatter_row_k(half* __restrict__ dst, const half* __restrict__ src,
                              const int* __restrict__ row_idx, int width) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < width) dst[(size_t)(*row_idx) * width + c] = src[c];
}
void scatter_row(half* dst, const half* src, const int* row_idx, int width, Stream s) {
  scatter_row_k<<<(width + 255) / 256, 256, 0, s>>>(dst, src, row_idx, width);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------- MoE permutation
__global__ void moe_count_k(const int64_t* __restrict__ ids, int total, int E,
                            int64_t* __restrict__ count) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  long long c = 0;
  // ids[i] == -1 (an expert on the other card) matches no e in [0, E), so it is excluded already.
  for (int i = 0; i < total; i++) if (ids[i] == e) c++;
  count[e] = c;
}

__global__ void moe_offset_k(const int64_t* __restrict__ count, int E,
                             int64_t* __restrict__ offset, int64_t* __restrict__ expert_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int64_t acc = 0;
  for (int e = 0; e < E; e++) {
    offset[e] = acc;
    acc += count[e];
    expert_count[e] = count[e];
  }
  expert_count[E] = 0;   // ignored by the kernel
  offset[E] = acc;       // total, used as the atomic cursor base
}

// Longest-processing-time-first visiting order for the grouped MoE kernel: order[r] is the expert
// that should be processed r-th, so rank r runs on group r % num_groups. Ordering by
// (token_count DESC, expert_index ASC) puts the fat experts first, which is what keeps the launch's
// makespan near the average group's load instead of the unluckiest group's.
//
// The order is a permutation of ALL E experts, empty ones included: a zero count sorts last (ties
// by index), and the kernel skips zero-count and over-capacity experts without consuming a ticket,
// exactly as the old scan-order loop did. Every rank is written by exactly one thread, so the
// scatter is race-free.
//
// rank[e] = #{b : count[b] > count[e]} + #{b < e : count[b] == count[e]}, i.e. "how many experts
// outrank me" under that total order - one pass, no sorting network, no host round-trip. The
// (count, index) pair is a total order, so the ranks are a permutation of [0, E) and the result is a
// pure function of the counts, which are themselves deterministic. That is the whole determinism
// argument: nothing here reads a clock, an atomic ticket or a scheduling artefact.
//
// One block, one thread per expert, counts staged in dynamic smem (E int64 = 4 KB at 512 experts).
// The O(E^2) ranking is a few microseconds next to a launch that takes milliseconds. The block is
// rounded up to a power of two, so this form covers E <= 1024; a larger model needs a per-warp or
// multi-block variant, and would fail loudly at launch rather than silently mis-rank.
__global__ void moe_lpt_order_k(const int64_t* __restrict__ count, int E, int* __restrict__ order) {
  extern __shared__ int64_t s_cnt[];   // E entries
  const int e = threadIdx.x;
  if (e < E) s_cnt[e] = count[e];
  __syncthreads();                     // every thread has staged the table before anyone ranks
  if (e >= E) return;
  const int64_t c = s_cnt[e];
  int rank = 0;
  for (int b = 0; b < E; ++b) {
    const int64_t cb = s_cnt[b];
    rank += (cb > c) | ((cb == c) & (b < e));
  }
  order[rank] = e;
}

const int64_t* moe_permute_offsets(const int64_t* workspace, int E) { return workspace + E + 2; }

const int* moe_permute_lpt_order(const int64_t* workspace, int E) {
  return (const int*)(workspace + 2 * (E + 2));
}

// Deterministic stable partition by expert.
//
// The previous form claimed a slot per element with `atomicAdd(&cursor[e], 1)`, so the order tokens
// appear inside an expert's bucket was scheduling-dependent, and the grouped GEMM reduces over that
// order. Measured consequence: greedy decode diverged between identical runs, and the bisect put
// the FIRST disagreement at prefill layer 0's MoE output with a relative spread of 2.68e-10, growing
// monotonically to 1.5e-2 by mid-network and eventually flipping a token id. See RESULTS.md.
//
// Here the slot is a pure function of the input: pos(i) = offset[e_i] + #{j < i : e_j == e_i}.
// Within a warp, __ballot_sync + __popc(mask & lanemask_lt) gives each matching lane its exact
// in-i rank with no atomic at all; a thread-0 exclusive scan over the per-warp counts finishes the
// block. Integer atomics remain fine for the *counts* in moe_count_k - integer addition is exact and
// order-independent - it is only the per-element cursor that was unsound.
//
// One block per expert, scanning the whole input range, which is the same O(E * total) shape as the
// moe_count_k pass above, so this adds one more such pass rather than changing the complexity.
__global__ void moe_scatter_k(const int64_t* __restrict__ ids, const half* __restrict__ weights,
                              int total, int topk, int E, const int64_t* __restrict__ offset,
                              int64_t* __restrict__ token_sorted, half* __restrict__ weight_sorted) {
  const int e = blockIdx.x;
  if (e >= E) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;
  __shared__ int s_warp[32];      // per-warp exclusive prefix within the current wave
  __shared__ int s_wave;          // elements matched in the current wave
  __shared__ int s_base;          // this expert's elements written before the current wave
  if (threadIdx.x == 0) s_base = 0;
  __syncthreads();

  for (int base = 0; base < total; base += blockDim.x) {
    const int i = base + threadIdx.x;
    const bool m = (i < total) && ((int)ids[i] == e);
    const unsigned mask = __ballot_sync(0xffffffffu, m);
    const int wrank = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) s_warp[warp] = __popc(mask);
    __syncthreads();
    if (threadIdx.x == 0) {              // sequential over warps, so the order is fixed
      int excl = 0;
      for (int w = 0; w < nwarp; w++) { const int c = s_warp[w]; s_warp[w] = excl; excl += c; }
      s_wave = excl;
    }
    __syncthreads();
    if (m) {
      const int pos = (int)(offset[e] + s_base + s_warp[warp] + wrank);
      token_sorted[pos] = i / topk;
      weight_sorted[pos] = weights[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) s_base += s_wave;
    __syncthreads();
  }
}

void moe_permute(const int64_t* ids, const half* weights, int M, int topk, int E,
                 int64_t* expert_count, int64_t* token_sorted, half* weight_sorted,
                 int64_t* workspace, Stream s) {
  int64_t* cnt = workspace;
  int64_t* off = workspace + E + 2;
  // The third slice used to hold a copy of `off` used as the atomic cursor base. The scatter no
  // longer needs a cursor, so the slice now holds the LPT visiting order (int32) the grouped kernel
  // walks the experts in - see moe_permute_lpt_order.
  int* order = (int*) moe_permute_lpt_order(workspace, E);
  int total = M * topk;
  moe_count_k<<<(E + 127) / 128, 128, 0, s>>>(ids, total, E, cnt);
  moe_offset_k<<<1, 1, 0, s>>>(cnt, E, off, expert_count);
  moe_scatter_k<<<E, 256, 0, s>>>(ids, weights, total, topk, E, off, token_sorted, weight_sorted);
  // LPT order for the grouped kernel. One block, one thread per expert; E <= 512, so the
  // O(E^2) ranking is a few microseconds next to a launch that takes milliseconds. Emitted after
  // the scatter because it reads cnt, which moe_offset_k does not modify but which is the authority.
  {
    int block = 1; while (block < E) block <<= 1;
    moe_lpt_order_k<<<1, block, (size_t)E * sizeof(int64_t), s>>>(cnt, E, order);
  }
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------- KDA prepare
// qkv transpose+cast [S,F] fp32 -> [F,S] bf16, tiled through smem so both sides are coalesced.
// The elementwise form in kda_prepare_k writes with stride S (8192 floats = 32KB), i.e. one sector per
// element - it was 28ms per layer, 0.95s of a 8192-token prefill batch.
__global__ void kda_transpose_cast_k(const float* __restrict__ src, unsigned short* __restrict__ dst,
                                     int S, int F) {
  __shared__ float t[32][33];
  int s0 = blockIdx.y * 32, f0 = blockIdx.x * 32;
  int fs = f0 + threadIdx.x;                       // source column (fast axis of src)
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int ss = s0 + threadIdx.y + j;
    t[threadIdx.y + j][threadIdx.x] = (ss < S && fs < F) ? src[(size_t)ss * F + fs] : 0.f;
  }
  __syncthreads();
  // dst[f][s]: consecutive threads (tx) must write consecutive s, so s = s0+tx and the smem entry
  // needed is t[tx][ty+j] (s relative = tx, f relative = ty+j).
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int df = f0 + threadIdx.y + j;
    int ds = s0 + threadIdx.x;
    if (df < F && ds < S) dst[(size_t)df * S + ds] = __bfloat16_as_ushort(__float2bfloat16(t[threadIdx.x][threadIdx.y + j]));
  }
}

void kda_transpose_cast(void* dst, const float* src, int S, int F, Stream s) {
  dim3 grid((F + 31) / 32, (S + 31) / 32), block(32, 8);
  kda_transpose_cast_k<<<grid, block, 0, s>>>(src, (unsigned short*)dst, S, F);
  cuda_check(cudaPeekAtLastError());
}

// beta = sigmoid(b) and g = lb*sigmoid(exp(A_log[h])*(f + dt_bias[channel])), both elementwise and
// already coalesced (they are the remaining quarter of kda_prepare).
__global__ void kda_gate_k(const float* __restrict__ b, const float* __restrict__ f,
                           const float* __restrict__ dt_bias, const float* __restrict__ a_log,
                           float lb, unsigned short* __restrict__ beta, float* __restrict__ g,
                           int S, int H, int Dk) {
  int HD = H * Dk;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < S * HD; i += gridDim.x * blockDim.x) {
    int s = i / HD, rem = i % HD, hh = rem / Dk, dd = rem % Dk;
    float decay = __expf(a_log[hh]);
    g[i] = lb * (1.0f / (1.0f + __expf(-(decay * (f[i] + dt_bias[hh * Dk + dd])))));
    (void)s;
  }
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < S * H; j += gridDim.x * blockDim.x)
    beta[j] = __bfloat16_as_ushort(__float2bfloat16(1.0f / (1.0f + __expf(-b[j]))));
}

void kda_gate(const float* b, const float* f, const float* dt_bias, const float* a_log, float lb,
              void* beta_bf16, float* g, int S, int H, int Dk, Stream s) {
  int total = S * H * Dk;
  kda_gate_k<<<(total + 511) / 512, 512, 0, s>>>(b, f, dt_bias, a_log, lb, (unsigned short*)beta_bf16,
                                                g, S, H, Dk);
  cuda_check(cudaPeekAtLastError());
}

__global__ void kda_prepare_k(const float* __restrict__ qkv, const float* __restrict__ b,
                              const float* __restrict__ f, const float* __restrict__ dt_bias,
                              const float* __restrict__ a_log, float lb,
                              unsigned short* __restrict__ mqkv, unsigned short* __restrict__ beta,
                              float* __restrict__ g, int S, int F, int H, int Dk) {
  int total = S * F + S * H + S * H * Dk;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    if (i < S * F) {                       // qkv transpose+cast: [S,F] -> [F,S]
      int s = i / F, ff = i % F;
      mqkv[(size_t)ff * S + s] = __bfloat16_as_ushort(__float2bfloat16(qkv[i]));
    } else if (i < S * F + S * H) {        // beta
      int j = i - S * F;
      beta[j] = __bfloat16_as_ushort(__float2bfloat16(1.0f / (1.0f + __expf(-b[j]))));
    } else {                               // g
      int j = i - S * F - S * H;
      int h = j % H, d = j / H;            // j indexes [S, H, Dk] -> s = d
      (void)d;
      int s = j / (H * Dk), rem = j % (H * Dk);
      int hh = rem / Dk, dd = rem % Dk;
      float decay = __expf(a_log[hh]);
      g[j] = lb * (1.0f / (1.0f + __expf(-(decay * (f[j] + dt_bias[hh * Dk + dd])))));
      (void)h;
    }
  }
}

void kda_prepare(const float* qkv, const float* b, const float* f, const float* dt_bias,
                 const float* a_log, float lower_bound, void* mixed_qkv_bf16, void* beta_bf16,
                 float* g, int S, int F, int H, int Dk, Stream s) {
  int total = S * F + S * H + S * H * Dk;
  kda_prepare_k<<<(total + 511) / 512, 512, 0, s>>>(qkv, b, f, dt_bias, a_log, lower_bound,
                                                   (unsigned short*)mixed_qkv_bf16,
                                                   (unsigned short*)beta_bf16, g, S, F, H, Dk);
  cuda_check(cudaPeekAtLastError());
}

__global__ void swiglu_clamp_k(const float* __restrict__ gate, const float* __restrict__ up,
                               half* __restrict__ out, size_t n, float limit) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gv = fminf(gate[i], limit);
  float uv = fminf(fmaxf(up[i], -limit), limit);
  float silu = gv / (1.0f + __expf(-gv));
  out[i] = __float2half_rn(silu * uv);
}
void swiglu_clamp(const float* gate, const float* up, half* out, size_t n, float limit, Stream s) {
  swiglu_clamp_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(gate, up, out, n, limit);
  cuda_check(cudaPeekAtLastError());
}

__global__ void layernorm_f16_k(const half* __restrict__ x, const half* __restrict__ w,
                                const half* __restrict__ b, half* __restrict__ y, int rows, int dim,
                                float eps) {
  int r = blockIdx.x;
  const half* xr = x + (size_t)r * dim;
  float sum = 0, sq = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = __half2float(xr[i]);
    sum += v; sq += v * v;
  }
  __shared__ float ss[2];
  __shared__ float s1[32], s2[32];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, o); sq += __shfl_xor_sync(0xffffffffu, sq, o); }
  if (lane == 0) { s1[warp] = sum; s2[warp] = sq; }
  __syncthreads();
  if (warp == 0) {
    int nw = (blockDim.x + 31) / 32;
    float a = lane < nw ? s1[lane] : 0.f, c = lane < nw ? s2[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); c += __shfl_xor_sync(0xffffffffu, c, o); }
    if (lane == 0) { ss[0] = a / dim; ss[1] = c / dim; }
  }
  __syncthreads();
  float mean = ss[0], var = ss[1] - mean * mean;
  float inv = rsqrtf(var + eps);
  half* yr = y + (size_t)r * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = (__half2float(xr[i]) - mean) * inv * __half2float(w[i]) + __half2float(b[i]);
    yr[i] = __float2half_rn(v);
  }
}
void layernorm_f16(const half* x, const half* w, const half* b, half* y, int rows, int dim,
                   float eps, Stream s) {
  layernorm_f16_k<<<rows, 128, 0, s>>>(x, w, b, y, rows, dim, eps);
  cuda_check(cudaPeekAtLastError());
}

}}  // namespace helios::glue
