// Diagnostic: does the ported EXL3 inner GEMM compute the same rows at 32 and 64 per tile as it
// does at 16?
//
// This exists because the MoE port is missing exllamav3's 32/64-row MoE tiles. Adding them measured
// +17% on prefill and produced degenerate output ("hort-hort-hort-..."), so this pins down whether
// the wide shapes are actually supported before anyone attempts the port proper.
//
// The comparison is kernel-to-kernel, not against a CPU reference, and deliberately so: the
// 128x128 Hadamard the kernel applies is a real transform, and an earlier version of this file
// approximated it as a scalar 1/sqrt(128) - which reported mt=16 (the production-correct shape) as
// broken. mt=16 is the reference for the rows it produces because it is what ships and is known
// good; a wide tile that computes the same rows must agree with it. Rows that only a wide tile
// produces are cross-checked against the other wide tile, and the comparison is done per 16-row
// block - see the note in probe() for why a whole-buffer comparison is not meaningful here.
//
// It calls exl3_gemm_kernel_inner directly rather than the public gemm(), because gemm() always
// decomposes M into 16-row slabs and therefore never instantiates a wide tile - which is exactly why
// the existing test suite could not see the problem.
//
// This file defines HELIOS_ALLOW_WIDE_ROW_TILES so the inner kernel's `TILESIZE_M == 16` guard
// relaxes. No production translation unit defines it.

#define HELIOS_ALLOW_WIDE_ROW_TILES 1

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "helios_shim.cuh"
#include "ptx.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_moe_common.cuh"
#include "exl3_gemm_inner.cuh"
#include "pack.cuh"
#include "reconstruct.cuh"

using namespace helios;
using namespace helios::exl3;

namespace
{

constexpr int BITS = 2;    // model is 2.05 bpw; 2 is the nearest buildable fixed bitrate

constexpr int LOCKS = 4096;

void* dev_alloc(size_t bytes)
{
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { printf("  FAIL: cudaMalloc(%zu)\n", bytes); exit(1); }
    return p;
}

// exl3_gemm_kernel_inner is `inline __device__`, so it is reached through a __global__ wrapper -
// exactly as exl3_moe_kernel does. The wrapper clears `locks` first: the group barriers leave them
// non-zero, and a second launch would read stale barriers.
int g_N = 256, g_K = 256;
constexpr int NT = 128;   // N tile: the reference forces N=128 whenever it picks a wide row tile

template<int mt>
// __launch_bounds__ mirrors exl3_moe_kernel.cuh, which launches the inner kernel the same way:
// with MOE_TILESIZE_K=32 the block is 512 threads, i.e. 128 registers/thread is the hard ceiling.
// Without the annotation ptxas sizes a 64-row-tile kernel (one fp16 + one fp32 accumulator set
// per 16-row block) at 152 registers and the launch dies with "too many resources requested".
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void rows_wrap(const half* A, const uint16_t* B, void* C, int m, int k, int n,
               int* locks, const half* post)
{
    for (int i = threadIdx.x; i < LOCKS; i += blockDim.x) locks[i] = 0;
    __syncthreads();
    // ALL threads must enter: this is a cooperative kernel and the group barriers need every
    // participant. Gating it on thread 0 (the obvious-looking "one thread calls it" shape) leaves
    // the barriers unreached and the output all zeros.
    exl3_gemm_kernel_inner
    <
            BITS, false, 2, mt, MOE_TILESIZE_K, 128, MOE_SH_STAGES, (mt >= 64) ? 2 : MOE_FRAG_STAGES,
        false
    >(A, B, C, m, k, n, locks, post);
}

template<int mt>
std::vector<half> run_shape(const half* d_A, const uint16_t* d_B, half* d_C, int* d_locks, int m)
{
    const int tbk = MOE_TILESIZE_K / 16, tbn = NT / 16;
    const int frags_n = 2 * tbn / (EXL3_GEMM_BASE_THREADS / 32);
    const size_t sh_a = (size_t)mt * MOE_TILESIZE_K * sizeof(half);
    const size_t sh_b = (size_t)tbk * tbn * 256 / 16 * BITS * sizeof(uint16_t);
    // must match the kernel: the split-K scratch scales with the row tile
    const size_t sh_c = (size_t)(4 * EXL3_GEMM_BASE_THREADS * frags_n * (mt / 16));
    const size_t smem = MOE_SH_STAGES * (2 * sh_a + 2 * sh_b) + 4 * sh_c;

    if (smem > (size_t)SMEM_MAX)
    {
        printf("  mt=%-3d SKIP (smem %zu > SMEM_MAX %d)\n", mt, smem, SMEM_MAX);
        return std::vector<half>((size_t)m * g_N, __float2half(0.0f));
    }

    HELIOS_CUDA_CHECK(cudaFuncSetAttribute((const void*)rows_wrap<mt>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    HELIOS_CUDA_CHECK(cudaMemset(d_C, 0, (size_t)m * g_N * sizeof(half)));
    dim3 grid(1, 1, 1), blk(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16);
    rows_wrap<mt><<<grid, blk, smem>>>(d_A, d_B, d_C, m, g_K, g_N, d_locks, nullptr);
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> out((size_t)m * g_N);
    HELIOS_CUDA_CHECK(cudaMemcpy(out.data(), d_C, out.size() * sizeof(half), cudaMemcpyDeviceToHost));
    return out;
}


// Worst relative error over a row range of the output, N values per row.
static double worst_rows(const std::vector<half>& got, const std::vector<half>& ref, int N,
                         int r0, int r1)
{
    double worst = 0.0;
    for (int r = r0; r < r1; r++)
        for (int n = 0; n < N; n++)
        {
            double a = __half2float(ref[(size_t)r * N + n]), b = __half2float(got[(size_t)r * N + n]);
            double d = fabs(a - b) / (fabs(a) + 1e-3);
            if (!(d == d)) d = 1e9;                 // NaN
            if (d > worst) worst = d;
        }
    return worst;
}

static bool all_zero(const std::vector<half>& v, int N, int r0, int r1)
{
    for (int r = r0; r < r1; r++)
        for (int n = 0; n < N; n++)
            if (__half2float(v[(size_t)r * N + n]) != 0.0f) return false;
    return true;
}

static const char* verdict(double w) { return w > 0.05 ? "DIVERGES" : "match"; }

static void probe(int K, int N, const char* label)
{
    g_K = K; g_N = N;
    printf("\n=== %s  (K=%d, N=%d, N tile=%d) ===\n", label, K, N, NT);

    const int M = 64;
    std::vector<half> A((size_t)M * K);
    std::mt19937 rng(1234);
    for (auto& v : A) v = __float2half(((float)(rng() % 2001) / 1000.0f - 1.0f) * 0.5f);
    std::vector<uint16_t> tref((size_t)(K / 16) * (N / 16) * 16 * BITS);
    for (auto& v : tref) v = (uint16_t)(rng() & 0xFFFF);

    half* d_A    = (half*)dev_alloc(A.size() * sizeof(half));
    uint16_t* d_T = (uint16_t*)dev_alloc(tref.size() * sizeof(uint16_t));
    half* d_Bhat = (half*)dev_alloc((size_t)K * N * sizeof(half));
    half* d_C    = (half*)dev_alloc((size_t)M * N * sizeof(half));
    int* d_locks = (int*)dev_alloc(LOCKS * sizeof(int));

    HELIOS_CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(half), cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_T, tref.data(), tref.size() * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice));
    reconstruct(d_Bhat, d_T, K / 16, N / 16, N, BITS, /*mcg=*/false, /*mul1=*/false);
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

    // A tile writes exactly rows 0..TILESIZE_M-1. The kernel has no slice_m loop - index_m() is 0
    // and slice_m is 0 - so size_m only widens the A-load row predicate and the epilogue store
    // guards. Two consequences, and the previous whole-buffer comparison got both backwards:
    //
    //   - The mt=16 tile is identically ZERO for rows >= 16, so comparing a wide tile's rows
    //     16..31 against it asserts "a wide tile must compute nothing there" - the exact
    //     opposite of what this test exists to catch.
    //   - size_m > TILESIZE_M is outside the kernel's contract. The A load spreads
    //     EXL3_GEMM_BASE_THREADS threads over sh0_a_stride_m / 8 int4 elements without clamping
    //     the row, so the surplus threads write past the A stage into the neighbouring stages.
    //     That is why the mt=16 reference used to change its own rows 0..15 with m.
    //
    // So each 16-row block is compared only between tiles that are in contract (TILESIZE_M >= m)
    // and actually write that block (TILESIZE_M > r0). A block with a single producing tile
    // cannot be cross-checked, so it is instead checked for not being identically zero: A is
    // random and the product dense, so an all-zero block means the accumulator was never
    // written, which is the signature this test was written for.
    for (int m : {16, 32, 64})
    {
        std::vector<half> r16, r32, r64;
        if (m <= 16) r16 = run_shape<16>(d_A, d_T, d_C, d_locks, m);
        if (m <= 32) r32 = run_shape<32>(d_A, d_T, d_C, d_locks, m);
        if (m <= 64) r64 = run_shape<64>(d_A, d_T, d_C, d_locks, m);

        for (int b = 0; b * 16 < m; ++b)
        {
            int r0 = b * 16, r1 = r0 + 16;
            const std::vector<half>* res[3];
            const char* lab[3];
            int n = 0;
            if (m <= 16 && 16 > r0) { res[n] = &r16; lab[n] = "mt=16"; ++n; }
            if (m <= 32 && 32 > r0) { res[n] = &r32; lab[n] = "mt=32"; ++n; }
            if (m <= 64 && 64 > r0) { res[n] = &r64; lab[n] = "mt=64"; ++n; }

            printf("  m=%-3d rows %2d-%2d :", m, r0, r1 - 1);
            if (n < 2) printf(" %s single-tile %s\n", lab[0], all_zero(*res[0], N, r0, r1) ?
                                                              "ALL-ZERO (BAD)" : "non-zero  ");
            for (int i = 1; i < n; i++)
                printf("  %s vs %s %s", lab[i - 1], lab[i], verdict(worst_rows(*res[i], *res[i - 1], N, r0, r1)));
            if (n >= 2) printf("\n");
        }
    }

    cudaFree(d_A); cudaFree(d_T); cudaFree(d_Bhat); cudaFree(d_C); cudaFree(d_locks);
}

}  // namespace

int main()
{
    printf("EXL3 inner GEMM row-tile diagnostic. Reference is mt=16 - the shape that ships.\n");
    printf("m sweeps 16/32/64; each tile runs at a row tile >= m, and every 16-row block is\n"
           "compared between the tiles that actually produce it.\n");
    probe(256,  128, "toy");
    probe(2560, 640, "MoE gate/up (hidden->intermediate)");
    probe(640,  2560, "MoE down (intermediate->hidden)");
    printf("\nA wide tile that DIVERGES computes the wrong thing. A match on every row block means\n"
           "the shape is supported end to end.\n");
    return 0;
}
