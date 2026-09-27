// Correctness of the prefill reconstruct path (exl3::linear with HELIOS_RECONSTRUCT_PREFILL=1)
// against the trellis kernel it replaces (exl3::gemm).
//
// The reconstruct path dequantizes the weight to fp16 and runs a dense fp16-accumulator tensor-core
// GEMM instead of the EXL3 trellis kernel. The two are not expected to agree bit for bit - that is
// the whole point of the reference's own choice, and it roughly doubles the relative error of the
// product - but they must compute the SAME MATHEMATICAL PRODUCT. The failure modes this pins down
// are the ones a speed-only measurement cannot see:
//
//   - a weight written transposed (the reconstruct emits (K, N) row-major, the GEMM reads B as
//     (K, N); a swap shows up only as a wrong answer, never as a slower one),
//   - the ORIGINAL-basis vs rotated-basis mix-up (reconstruct_had_slice folds BOTH Hadamards and
//     the suh/svh sign vectors into the weight; skipping either is still a plausible-looking
//     number),
//   - a wrong row stride / fp32-vs-fp16 output convention on a caller that asks for fp32,
//   - a K/N mix-up on a shape where K != N.
//
// The trellis kernel is the oracle because it is what ships and is verified elsewhere; the two
// paths are compared on the same random inputs. Tolerance is a relative L2 over the whole output
// plus a scale ratio, both of which are stable for random data, rather than a per-element bound
// that would be dominated by cancellation in the tail.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "helios_shim.cuh"
#include "exl3_gemm.cuh"

using namespace helios;
using namespace helios::exl3;

namespace
{

// Relative L2 the fp16 weight + fp16-accumulate path may differ by. Measured on the shapes below
// it lands at 1e-4..1e-3; 5e-3 is two orders of headroom and still far below any real regression
// (a transposed weight or a missing Hadamard is O(1)).
constexpr double kTol = 5e-3;
constexpr double kScaleTol = 2e-3;

void* dev_alloc(size_t bytes)
{
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { printf("  FAIL: cudaMalloc(%zu)\n", bytes); exit(1); }
    return p;
}

int g_fail = 0;

void probe(int M, int N, int K, int bits, bool mul1, const char* label)
{
    printf("\n=== %s  (M=%d N=%d K=%d bits=%d %s) ===\n", label, M, N, K, bits,
           mul1 ? "mul1" : "default codebook");

    std::mt19937 rng(0x51ed ^ (unsigned)(M * 131 + N * 17 + K));
    std::vector<half> A((size_t) M * K);
    for (auto& v : A) v = __float2half(((float)(rng() % 2001) / 1000.0f - 1.0f) * 0.5f);
    std::vector<uint16_t> T((size_t)(K / 16) * (N / 16) * 16 * bits);
    for (auto& v : T) v = (uint16_t)(rng() & 0xFFFF);
    // suh/svh are per-element sign+scale vectors; random +-1 keeps the reconstructed weight in a
    // sane range so the comparison is not dominated by a handful of huge entries.
    std::vector<half> SUH(K), SVH(N);
    for (auto& v : SUH) v = __float2half((rng() & 1) ? 1.0f : -1.0f);
    for (auto& v : SVH) v = __float2half((rng() & 1) ? 1.0f : -1.0f);

    half* d_A = (half*) dev_alloc(A.size() * sizeof(half));
    uint16_t* d_T = (uint16_t*) dev_alloc(T.size() * sizeof(uint16_t));
    half* d_SUH = (half*) dev_alloc(SUH.size() * sizeof(half));
    half* d_SVH = (half*) dev_alloc(SVH.size() * sizeof(half));
    float* d_ref = (float*) dev_alloc((size_t) M * N * sizeof(float));
    float* d_got = (float*) dev_alloc((size_t) M * N * sizeof(float));
    half* d_had = (half*) dev_alloc((size_t) M * K * sizeof(half));
    half* d_had2 = (half*) dev_alloc((size_t) M * K * sizeof(half));
    HELIOS_CUDA_CHECK(cudaMemset(d_ref, 0, (size_t) M * N * sizeof(float)));
    HELIOS_CUDA_CHECK(cudaMemset(d_got, 0, (size_t) M * N * sizeof(float)));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(half), cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_T, T.data(), T.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_SUH, SUH.data(), SUH.size() * sizeof(half), cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_SVH, SVH.data(), SVH.size() * sizeof(half), cudaMemcpyHostToDevice));

    GroupWords w{d_T, d_SUH, d_SVH, mul1 ? 1 : 0};

    // Oracle: the trellis kernel, which is what ships today.
    gemm(d_ref, d_A, w, M, N, K, bits, /*y_fp32=*/true, 0, d_had);
    // The path under test.
    linear(d_got, d_A, w, M, N, K, bits, /*y_fp32=*/true, 0, d_had2);
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ref((size_t) M * N), got((size_t) M * N);
    HELIOS_CUDA_CHECK(cudaMemcpy(ref.data(), d_ref, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaMemcpy(got.data(), d_got, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double num = 0.0, den = 0.0, gsum = 0.0, rsum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        double d = (double) got[i] - (double) ref[i];
        if (!(d == d) || !(ref[i] == ref[i])) { printf("  FAIL: NaN at %zu\n", i); g_fail++; goto done; }
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
        gsum += (double) got[i] * (double) got[i];
        rsum += (double) ref[i] * (double) ref[i];
    }
    {
        double rel = den > 0.0 ? sqrt(num / den) : 0.0;
        double scale = rsum > 0.0 ? sqrt(gsum / rsum) : 0.0;
        bool ok = rel < kTol && fabs(scale - 1.0) < kScaleTol;
        // rel == 0 would mean both paths returned the same bits, i.e. the reconstruct branch was
        // NOT taken and this shape tested nothing. With random weights the fp16 round trip always
        // perturbs the product, so require a nonzero difference as well as a small one.
        if (rel == 0.0) printf("  (reconstruct branch NOT taken for this shape - tested nothing)\n");
        ok = ok && rel > 0.0;
        printf("  relative L2 %.3e (limit %.1e, must be > 0)   rms ratio %.6f   %s\n", rel, kTol,
               scale, ok ? "match" : "DIVERGES");
        if (!ok) g_fail++;
    }
done:
    cudaFree(d_A); cudaFree(d_T); cudaFree(d_SUH); cudaFree(d_SVH);
    cudaFree(d_ref); cudaFree(d_got); cudaFree(d_had); cudaFree(d_had2);
}

}  // namespace

int main()
{
    // linear() reads this once, on its first call.
    setenv("HELIOS_RECONSTRUCT_PREFILL", "1", 1);
    // Without this the narrow shapes below fall through to the trellis kernel (upstream declines
    // them because its fallback is cuBLAS), and the test would compare the kernel with itself.
    setenv("HELIOS_HGEMM_SMALL_SHAPES", "1", 1);
    printf("Prefill reconstruct path vs the trellis GEMM (same random inputs, same shapes).\n");
    printf("M is above the 144-row threshold, so linear() must take the reconstruct branch; each\n");
    printf("case must also show a nonzero (but small) difference, i.e. the branch really ran.\n");
    probe(512,  2560,  2560, 2, false, "GDN qkv-ish");
    probe(512,   640,  2560, 2, false, "shared expert gate/up (N < K)");
    probe(512,  2560,   640, 2, false, "shared expert down (K < N)");
    probe(512,   512,  2560, 2, false, "attention k/v (N = 512)");
    probe(200,  1280,  2560, 3, true,  "mul1 codebook, M not a multiple of 128");
    probe(1024, 2560,  2560, 2, false, "prefill width");
    probe(1024, 12288, 2560, 2, false, "attention q (widest projection)");
    printf("\nA DIVERGES line means the reconstruct path computes a different product from the\n"
           "kernel it replaces - a transposed weight, a missing Hadamard, or a bad stride.\n");
    return g_fail ? 1 : 0;
}
