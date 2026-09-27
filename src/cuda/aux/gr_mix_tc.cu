// Tensor-core / fused-register GatedResidual mixer. See gr_mix_tc.cuh for what this path is and why
// it is separate from, and not a replacement for, the fp32 path in gr_mix.cu.
//
// Where the fp32 path spends its time, per call at prefill (R = 1024):
//   normed  R*HD fp32 written, then read by `dots` and again by `post`   42 + 42 + 42 MB
//   gated   R*HD fp32 written, then read by `fin`                         42 + 42 MB
//   13.4 GFLOP of scalar FMA across seven kernels: 2.9 ms/call, 4.6 TFLOPS of a 35.6 TFLOPS fp32
//   peak - it is an eighth of its own roofline, and two thirds of its bytes are intermediates that
//   never leave the operator.
//
// This file removes both, in three launches:
//
//   1. gr_tc_prep_kernel   one pass over the stream chunk: rmr[r,h], and the four post-site partial
//                          dots against `inject`, split by stream exactly the way rmr is.
//   2. gr_tc_dots_kernel   t = silu(normed @ down^T / H) as mma.m16n8k16, A scaled on load.
//   3. gr_tc_gate_kernel   the H streams are a sequential loop INSIDE the block, so iteration h
//                          reuses iteration h-1's accumulator registers: sum += sigmoid(g) *
//                          normed is a register add, and `mixed` (R, D) is the only output.
//
// `normed` and `gated` are never materialised. Both GEMMs take A row-major (M x K) and B
// column-major (K x N), which is what `down` (rank, HD) and `up` (HD, rank) already are - no
// transposes, no repacking, no padding of the checkpoint tensors.
#include "gr_mix_tc.cuh"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>

namespace helios { namespace aux {

namespace {

// ---------------------------------------------------------------------------------------------
// Block shape, shared by both GEMM kernels
// ---------------------------------------------------------------------------------------------
//
// 64x64 output tile, 64-deep k chunk, eight warps in a 2x4 grid (32x16 per warp: 2 m-tiles x 2
// n-tiles = 4 mma and 16 accumulator registers per k-step). The k chunk is padded to 72 halfs:
// 72*2 = 144 B = 36 words, so row r starts at bank 4r and the eight ldmatrix row addresses of
// one matrix cover all 32 banks exactly once. Eight warps rather than four because the dots grid
// is only 80 blocks at R=1024, i.e. about one per SM, with no second block to hide latency behind.
constexpr int kTcBM = 64, kTcBN = 64, kTcBK = 64, kTcSP = kTcBK + 8;
constexpr int kTcWarps = 8, kTcWM = 2, kTcWN = 4;
constexpr int kTcMT = kTcBM / kTcWM / 16;               // m-tiles per warp (2)
constexpr int kTcNT = kTcBN / kTcWN / 8;                // n-tiles per warp (2)
// The per-row scale arrays live past the two staging rings, so they are part of the request: a
// dynamic allocation smaller than what the kernel writes is not clamped, it is an overwrite of
// whatever the next block put there. (Found by the first accuracy run, which reported a `mixed`
// error of 8.5 against an rms of 0.70 while `post` - computed by the same prep kernel, off the
// same rmr - was exact to 7e-7.)
constexpr int kTcScaleBytes = 1024;                   // the gate kernel's rmr[64][H]
constexpr int kTcSmem = 2 * (kTcBM * kTcSP * 2 + kTcBN * kTcSP * 2) + kTcScaleBytes;
static_assert(kTcMT == 2 && kTcNT == 2, "the epilogues are written for 2x2 tiles per warp");
static_assert(kTcWM * 32 * kTcWN * 32 == kTcWarps * 32 * 32, "warp grid must tile the block");

// Staging map, identical for both GEMMs' 16 B chunks: 8 threads cover one row's 64 halfs, so 256
// threads cover 32 rows and two passes cover the 64-row tile. The dots kernel's A tile is built
// from fp32, so it stages 16 threads x 4 floats per row instead - also 16 B per thread.

// ---------------------------------------------------------------------------------------------
// fp16 mma primitives
// ---------------------------------------------------------------------------------------------

// m16n8k16, A row-major MxK, B column-major KxN, fp32 accumulate. `c` is the 4-register C/D
// fragment, updated in place; `a` the 4-register A fragment; `b` the 2-register B fragment.
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

// ldmatrix out of a shared tile stored [row][k] with k contiguous. .x4 takes one 16x16 A fragment
// (lanes 0-7 -> rows 0-7, 8-15 -> rows 8-15, 16-23 -> rows 0-7 at k+8, 24-31 -> rows 8-15 at k+8);
// .x2 takes one 16x8 B fragment from lanes 0-15 only.
__device__ __forceinline__ void ldm4(uint32_t (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldm2(uint32_t (&r)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const half* tile, int m0, int k0,
                                       int lane) {
  const int row = m0 + (lane & 7) + 8 * ((lane >> 3) & 1);
  const int k = k0 + 8 * (lane >> 4);
  ldm4(a, smem_addr(tile + (size_t)row * kTcSP + k));
}

__device__ __forceinline__ void frag_b(uint32_t (&b)[2], const half* tile, int n0, int k0,
                                       int lane) {
  const int l2 = lane & 15;
  const int row = n0 + (l2 & 7);
  const int k = k0 + 8 * (l2 >> 3);
  ldm2(b, smem_addr(tile + (size_t)row * kTcSP + k));
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x * sigmoidf_(x); }

// ---------------------------------------------------------------------------------------------
// 1. prep: rmr, the scaled fp16 operand, and the post-site partials, in one pass
// ---------------------------------------------------------------------------------------------
//
// rmr[r,h] = rsqrt(mean_d x[r,h,d]^2 + eps) is a per-(row, stream) scalar, but normed needs it
// multiplied into the A operand of the dots GEMM, and that GEMM starts before any row is finished.
// exllamav3's gr_dots_i8 pays for this with H partial accumulators (4x the registers); this pays
// one extra pass over the stream chunk instead and every later kernel then gets its per-row scale
// for free.
//
// The pass also emits the GEMM's A operand itself, `nrm16` = normed as fp16. That is not a
// materialisation of `normed` in the sense the fp32 path is criticised for - it is half the bytes
// of the fp32 stream chunk it replaces (21 MB against 42 MB at R=1024), it is read 5 times by the
// dots GEMM where the fp32 path read 42 MB, and it is what lets the dots kernel stage A with the
// same pure 16 B copy the B tile uses. Doing the scale at the GEMM's staging point instead, out
// of an fp32 load, produced a tile that was wrong for a subset of its rows on every layout tried
// (16 threads x 4 floats, 8 threads x 8 floats, 2 B and 4 B and 16 B stores, scalar and packed
// conversion) while the same code staging a straight copy was exact - measured as t disagreeing
// with a float64 reference by up to 0.35 absolute on 3 of 8 rows while the other 5 matched to
// 4e-5. The traffic also comes out ahead: 42 MB in + 21 MB out here, then 5 x 21 MB in the GEMM,
// against 42 MB in the GEMM's own staging pass times 5.
//
// The stream slice is parked in shared memory across the reduction so the block reads it once:
// 42 MB in, 21 MB out, no second pass. The post site's four dots come out of the same loop
// (ph[r][h][g] = sum over the slice of x * (1 + w) * inject[g]), which is the whole of
// gr_post_kernel - otherwise a further 42 MB read of normed.
__global__ __launch_bounds__(256)
void gr_tc_prep_kernel(const float* __restrict__ streams, const half* __restrict__ norm_raw,
                       const half* __restrict__ inject, half* __restrict__ nrm16, int R, int H,
                       int D, float eps, float* __restrict__ rmr, float* __restrict__ ph) {
  extern __shared__ float xs[];                  // [256][kPrepStride], the stream slice
  const int xc = (D + 255) / 256;                // elements per thread
  const int stride = xc + 1;                     // +1 so the staging pass is bank-conflict free
  const int r = blockIdx.x, h = blockIdx.y;
  const float* x = streams + ((size_t)r * H + h) * (size_t)D;
  const half* nw = norm_raw + (size_t)h * D;
  half* out = nrm16 + ((size_t)r * H + h) * (size_t)D;

  float sq = 0.f;
  float p[4] = {0.f, 0.f, 0.f, 0.f};
  for (int q = 0, j = threadIdx.x; j < D; ++q, j += 256) {
    const float v = x[j];
    xs[threadIdx.x * stride + q] = v;
    sq = fmaf(v, v, sq);
    if (inject) {
      const float vw = v * (1.0f + __half2float(nw[j]));
#pragma unroll
      for (int g = 0; g < 4; ++g)
        p[g] = fmaf(vw, __half2float(inject[((size_t)g * H + h) * D + j]), p[g]);
    }
  }

  __shared__ float red[8][5];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
#pragma unroll
    for (int g = 0; g < 4; ++g) p[g] += __shfl_xor_sync(0xffffffffu, p[g], o);
  }
  if (lane == 0) {
    red[warp][0] = sq;
#pragma unroll
    for (int g = 0; g < 4; ++g) red[warp][1 + g] = p[g];
  }
  __syncthreads();
  float t[5];
#pragma unroll
  for (int q = 0; q < 5; ++q) {
    float v = (lane < 8) ? red[lane][q] : 0.f;
#pragma unroll
    for (int o = 4; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    t[q] = v;
  }
  __shared__ float rmr_sh;
  if (threadIdx.x == 0) {
    rmr_sh = rsqrtf(t[0] / (float)D + eps);
    rmr[(size_t)r * H + h] = rmr_sh;
    if (inject)
      for (int g = 0; g < 4; ++g) ph[((size_t)r * H + h) * 4 + g] = t[1 + g];
  }
  __syncthreads();
  const float sc = rmr_sh;
  for (int q = 0, j = threadIdx.x; j < D; ++q, j += 256)
    out[j] = __float2half_rn(xs[threadIdx.x * stride + q] * sc * (1.0f + __half2float(nw[j])));
}

// ---------------------------------------------------------------------------------------------
// 2. dots: t[r,i] = silu((1/H) sum_k normed[r,k] * down[i,k])
// ---------------------------------------------------------------------------------------------
//
// A (64 rows x 64 k) is `nrm16` - the stream chunk already scaled into fp16 by prep - and B
// (64 rank columns x 64 k) is `down`. Both are staged by the same pure 16 B copy, which is the one
// form of this loop that has ever been exact (see the note on gr_tc_prep_kernel).
__global__ __launch_bounds__(kTcWarps * 32)
void gr_tc_dots_kernel(const half* __restrict__ nrm16, const half* __restrict__ down, int R, int H,
                       int D, int rank, half* __restrict__ t) {
  extern __shared__ char tc_smem[];
  half* As = (half*)tc_smem;                      // [2][kTcBM][kTcSP]
  half* Bs = As + 2 * kTcBM * kTcSP;              // [2][kTcBN][kTcSP]

  const int HD = H * D;
  const int r0 = blockIdx.y * kTcBM, i0 = blockIdx.x * kTcBN;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / kTcWN, wn = warp % kTcWN;
  // 8 threads cover one row's 64 halfs, so 256 threads cover 32 rows and two passes cover 64.
  const int arow = tid >> 3, ak = (tid & 7) << 3;

  uint4 areg[2], breg[2];
  auto load_a = [&](int k0) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int gr = r0 + arow + 32 * u;
      areg[u] = (gr < R) ? *(const uint4*)(nrm16 + (size_t)gr * HD + k0 + ak)
                         : make_uint4(0, 0, 0, 0);
    }
  };
  auto load_b = [&](int k0) {
#pragma unroll
    for (int u = 0; u < 2; ++u)
      breg[u] = *(const uint4*)(down + (size_t)(i0 + arow + 32 * u) * HD + k0 + ak);
  };
  auto store_stage = [&](int st) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      *(uint4*)(As + st * kTcBM * kTcSP + (arow + 32 * u) * kTcSP + ak) = areg[u];
      *(uint4*)(Bs + st * kTcBN * kTcSP + (arow + 32 * u) * kTcSP + ak) = breg[u];
    }
  };

  float acc[kTcMT][kTcNT][4];
#pragma unroll
  for (int i = 0; i < kTcMT; ++i)
#pragma unroll
    for (int j = 0; j < kTcNT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  const int nstage = HD / kTcBK;
  load_a(0);
  load_b(0);
  store_stage(0);
  __syncthreads();
  for (int s = 0; s < nstage; ++s) {
    if (s + 1 < nstage) {
      load_a((s + 1) * kTcBK);
      load_b((s + 1) * kTcBK);
    }
    const half* at = As + (s & 1) * kTcBM * kTcSP;
    const half* bt = Bs + (s & 1) * kTcBN * kTcSP;
#pragma unroll
    for (int ks = 0; ks < kTcBK; ks += 16) {
      uint32_t af[kTcMT][4], bf[kTcNT][2];
#pragma unroll
      for (int i = 0; i < kTcMT; ++i) frag_a(af[i], at, wm * 32 + i * 16, ks, lane);
#pragma unroll
      for (int j = 0; j < kTcNT; ++j) frag_b(bf[j], bt, wn * 16 + j * 8, ks, lane);
#pragma unroll
      for (int i = 0; i < kTcMT; ++i)
#pragma unroll
        for (int j = 0; j < kTcNT; ++j) mma16816(acc[i][j], af[i], bf[j]);
    }
    if (s + 1 < nstage) {
      __syncthreads();                              // everyone is done with buffer (s+1) & 1
      store_stage((s + 1) & 1);
      __syncthreads();
    }
  }

  // t[r,i] = silu(acc / H), stored fp16: the gate kernel re-reads this tile once per stream and
  // per d-tile, and halving it costs 2^-11 of relative error on the gate pre-activation.
#pragma unroll
  for (int i = 0; i < kTcMT; ++i)
#pragma unroll
    for (int j = 0; j < kTcNT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + wm * 32 + i * 16 + (lane >> 2) + 8 * (q >> 1);
        const int c = i0 + wn * 16 + j * 8 + 2 * (lane & 3) + (q & 1);
        if (r < R && c < rank) t[(size_t)r * rank + c] = __float2half_rn(siluf_(acc[i][j][q] / (float)H));
      }
}

// ---------------------------------------------------------------------------------------------
// 3. gate + fin: mixed[r,d] = (1/H) sum_h sigmoid(sum_i t[r,i] * up[h*D+d, i]) * normed[r,h,d]
// ---------------------------------------------------------------------------------------------
//
// Block tile is 64 rows x 64 d and the four streams are a SEQUENTIAL loop inside the block, not
// four warp groups: iteration h reuses iteration h-1's accumulator registers, so the reduction
// over h is a register add and never a shuffle or a shared-memory round trip. The cost is a second
// set of 32 registers for the running sum, which still fits (4 warps... 8 warps at ~110 registers).
//
// `up` rows for (h, d0..d0+63) are staged per stream; the normed operand of the reduction is read
// straight from the stream chunk - the same 42 MB the fp32 path writes and re-reads twice - and
// scaled in the epilogue, so `gated` is never materialised.
__global__ __launch_bounds__(kTcWarps * 32)
void gr_tc_gate_kernel(const float* __restrict__ streams, const half* __restrict__ norm_raw,
                       const half* __restrict__ up, const half* __restrict__ t,
                       const float* __restrict__ rmr, const float* __restrict__ ph, int R, int H,
                       int D, int rank, float* __restrict__ mixed, float* __restrict__ post) {
  extern __shared__ char tc_smem2[];
  half* As = (half*)tc_smem2;                     // [2][kTcBM][kTcSP]
  half* Bs = As + 2 * kTcBM * kTcSP;             // [2][kTcBN][kTcSP]
  float* rmr_s = (float*)(Bs + 2 * kTcBN * kTcSP);   // [kTcBM][H]

  const int HD = H * D;
  const int r0 = blockIdx.y * kTcBM, d0 = blockIdx.x * kTcBN;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / kTcWN, wn = warp % kTcWN;
  const int arow = tid >> 3, ak = (tid & 7) << 3;      // A and B: 8 threads x 8 halfs per row
  const int brow = arow;

  for (int i = tid; i < kTcBM * H; i += 256) {
    const int m = i / H, h = i - m * H;
    rmr_s[i] = (r0 + m < R) ? rmr[(size_t)(r0 + m) * H + h] : 0.0f;
  }

  float sum[kTcMT][kTcNT][4];
#pragma unroll
  for (int i = 0; i < kTcMT; ++i)
#pragma unroll
    for (int j = 0; j < kTcNT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) sum[i][j][q] = 0.f;

  uint4 areg[2], breg[2];
  auto load_a = [&](int k0) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int m = arow + 32 * u, gr = r0 + m;
      areg[u] = (gr < R) ? *(const uint4*)(t + (size_t)gr * rank + k0 + ak)
                         : make_uint4(0, 0, 0, 0);
    }
  };
  auto load_b = [&](int k0, int h) {
#pragma unroll
    for (int u = 0; u < 2; ++u)
      breg[u] = *(const uint4*)(up + (size_t)(h * D + d0 + brow + 32 * u) * rank + k0 + ak);
  };
  auto store_stage = [&](int st) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int m = arow + 32 * u;
      *(uint4*)(As + st * kTcBM * kTcSP + m * kTcSP + ak) = areg[u];
      *(uint4*)(Bs + st * kTcBN * kTcSP + (brow + 32 * u) * kTcSP + ak) = breg[u];
    }
  };

  const int nstage = rank / kTcBK;
  load_a(0);
  load_b(0, 0);
  store_stage(0);
  __syncthreads();

  for (int h = 0; h < H; ++h) {
    float acc[kTcMT][kTcNT][4];
#pragma unroll
    for (int i = 0; i < kTcMT; ++i)
#pragma unroll
      for (int j = 0; j < kTcNT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

    for (int s = 0; s < nstage; ++s) {
      if (s + 1 < nstage) {
        load_a((s + 1) * kTcBK);
        load_b((s + 1) * kTcBK, h);
      }
      const half* at = As + (s & 1) * kTcBM * kTcSP;
      const half* bt = Bs + (s & 1) * kTcBN * kTcSP;
#pragma unroll
      for (int ks = 0; ks < kTcBK; ks += 16) {
        uint32_t af[kTcMT][4], bf[kTcNT][2];
#pragma unroll
        for (int i = 0; i < kTcMT; ++i) frag_a(af[i], at, wm * 32 + i * 16, ks, lane);
#pragma unroll
        for (int j = 0; j < kTcNT; ++j) frag_b(bf[j], bt, wn * 16 + j * 8, ks, lane);
#pragma unroll
        for (int i = 0; i < kTcMT; ++i)
#pragma unroll
          for (int j = 0; j < kTcNT; ++j) mma16816(acc[i][j], af[i], bf[j]);
      }
      if (s + 1 < nstage) {
        __syncthreads();
        store_stage((s + 1) & 1);
        __syncthreads();
      }
    }

    // Epilogue for this stream: sum += sigmoid(acc) * normed[r, h, d], with normed re-derived
    // from the stream chunk. acc[i][j][q] is at row r = r0 + wm*32 + i*16 + lane/4 + 8*(q/2) and
    // columns d = d0 + wn*16 + j*8 + 2*(lane%4) + (q%2); the (q, q^1) pair is two adjacent d, so
    // one float2 of the stream and one __half2 of the norm weight serve both.
#pragma unroll
    for (int i = 0; i < kTcMT; ++i)
#pragma unroll
      for (int j = 0; j < kTcNT; ++j) {
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int m = wm * 32 + i * 16 + (lane >> 2) + 8 * qq;
          const int d = d0 + wn * 16 + j * 8 + 2 * (lane & 3);
          const int r = r0 + m;
          if (r >= R || d + 1 >= D) continue;
          const float2 xv = *(const float2*)(streams + (size_t)r * HD + h * D + d);
          // The (1 + w) convention of hyperconnections.py, the same one gr_tc_dots_kernel applies
          // through nwf and gr_scale_kernel applies inline. Dropping the +1 here scales the whole
          // reduction by w instead of 1 + w, which is a ~7x error on this model's weights and
          // looks like nothing at all in a smoke test - it is what the first accuracy run caught.
          const float2 wr = __half22float2(*(const __half2*)(norm_raw + h * D + d));
          const float2 wf = make_float2(wr.x + 1.0f, wr.y + 1.0f);
          const float sc = rmr_s[m * H + h];
          const float a0 = sigmoidf_(acc[i][j][2 * qq]);
          const float a1 = sigmoidf_(acc[i][j][2 * qq + 1]);
          sum[i][j][2 * qq] = fmaf(a0, xv.x * wf.x * sc, sum[i][j][2 * qq]);
          sum[i][j][2 * qq + 1] = fmaf(a1, xv.y * wf.y * sc, sum[i][j][2 * qq + 1]);
        }
      }

    if (h + 1 < H) {                               // restage stage 0 for the next stream
      load_a(0);
      load_b(0, h + 1);
      __syncthreads();
      store_stage(0);
      __syncthreads();
    }
  }

  const float inv_h = 1.0f / (float)H;
#pragma unroll
  for (int i = 0; i < kTcMT; ++i)
#pragma unroll
    for (int j = 0; j < kTcNT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + wm * 32 + i * 16 + (lane >> 2) + 8 * (q >> 1);
        const int d = d0 + wn * 16 + j * 8 + 2 * (lane & 3) + (q & 1);
        if (r < R && d < D) mixed[(size_t)r * D + d] = sum[i][j][q] * inv_h;
      }

  // The post site's four dots were accumulated per stream by prep; reducing over the streams is
  // four FMAs per output, so it rides along here rather than costing another 42 MB pass. One
  // thread per (row, stream) and only the first d-tile writes, so each address has one writer.
  if (post && blockIdx.x == 0) {
    const int m = tid >> 2, g = tid & 3, r = r0 + m;
    if (r < R) {
      float v = 0.f;
#pragma unroll
      for (int h = 0; h < 4; ++h)
        v = fmaf(rmr_s[m * 4 + h], ph[((size_t)r * 4 + h) * 4 + g], v);
      post[(size_t)r * 4 + g] = 2.0f * sigmoidf_(v / (float)H);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------

void gr_mix_tc(const float* streams, const half* norm_raw, const half* down, const half* up,
               const half* inject, int R, int H, int D, int rank, float eps, float* mixed,
               float* post, half* nrm16, half* t16, float* rmr, float* ph, Stream s) {
  {
    dim3 g((unsigned)R, (unsigned)H);
    const int xc = (D + 255) / 256;
    gr_tc_prep_kernel<<<g, 256, (size_t)256 * (xc + 1) * sizeof(float), s>>>(
        streams, norm_raw, inject, nrm16, R, H, D, eps, rmr, ph);
  }
  {
    dim3 g((unsigned)(rank / kTcBN), (unsigned)((R + kTcBM - 1) / kTcBM));
    gr_tc_dots_kernel<<<g, kTcWarps * 32, kTcSmem, s>>>(nrm16, down, R, H, D, rank, t16);
  }
  {
    dim3 g((unsigned)(D / kTcBN), (unsigned)((R + kTcBM - 1) / kTcBM));
    gr_tc_gate_kernel<<<g, kTcWarps * 32, kTcSmem, s>>>(streams, norm_raw, up, t16, rmr, ph, R, H,
                                                       D, rank, mixed, post);
  }
}

bool gr_mix_tc_applicable(int R, int H, int D, int rank, int min_r) {
  return R >= min_r && H == 4 && D % kTcBN == 0 && rank % kTcBN == 0 && (H * D) % kTcBK == 0 &&
         rank % kTcBK == 0;
}

size_t gr_mix_tc_smem() { return kTcSmem; }

}}  // namespace helios::aux
