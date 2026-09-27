// Semantics test for exl3::mgemm's weighted multi-matrix mode - the mode the MoE decode path uses.
//
// The MoE's decode path passes one call per projection with `bszm_in = bszm_out = slots`,
// `indices = the topk experts`, and (for the down projection) `weights = the topk weights`, expecting
// the per-slot outputs to be weighted and summed into one row per token. That expectation is what this
// checks, against 10 separate well-tested `gemm` calls - no host dequantization needed, and `gemm`
// already has its own smoke test.
//
// Reading the kernel rather than guessing: slot j uses input row j, matrix B_list[indices[j]], weight
// [j], and writes output row j; the reduction then sums rows [t*stride, (t+1)*stride) into row t,
// skipping any slot whose index is < 0. Two consequences this test pins down:
//   * with min_index >= 0 and num_tokens == 1 the indices/weights are COMPACTED in place while the
//     input rows are not, so slots no longer line up - the filter must be avoided (-1, -1) and
//     out-of-range slots expressed as a negative index instead;
//   * num_tokens == 1 reduces every slot into row 0.
//
//   ./build/quant/test_mgemm_semantics
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "helios_shim.cuh"
#include "pack.cuh"
#include "exl3_gemm.cuh"

using namespace helios;
using namespace helios::exl3;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { printf("  FAIL: %s\n", what); failures++; }
}

constexpr int kSlots = 10;
constexpr int N = 640;        // expert intermediate width
constexpr int K = 2560;       // hidden
constexpr int kBits = 2;

struct Mat {
  uint16_t* pack = nullptr;
  half* suh = nullptr;
  half* svh = nullptr;
};

// The 10 matrices, laid out so that a single mgemm's indices select them in a scrambled order (the
// real router never picks experts in ascending order, and a test that uses 0,1,2.. would not catch an
// index/row misalignment).
const int kOrder[kSlots] = {7, 2, 9, 0, 4, 1, 8, 3, 6, 5};

void run(int num_tokens, int bszm_in, int bszm_out, int M, int skip_pattern, const char* label) {
  std::mt19937 rng(1234u + num_tokens * 7 + bszm_in);
  const int slots = bszm_in > 0 ? bszm_in : kSlots;

  Mat mats[kSlots];
  std::vector<std::vector<uint16_t>> m_idx(kSlots), m_suh(kSlots), m_svh(kSlots);
  const size_t words = (size_t)(K / 16) * (N / 16) * (256 * kBits / 16);

  for (int m = 0; m < kSlots; m++) {
    m_idx[m].resize((size_t)K * N);
    for (auto& v : m_idx[m]) v = (uint16_t)(rng() & ((1 << kBits) - 1));
    m_suh[m].resize(K);
    m_svh[m].resize(N);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : m_suh[m]) { __half h = __float2half_rn(u(rng)); __builtin_memcpy(&v, &h, 2); }
    for (auto& v : m_svh[m]) { __half h = __float2half_rn(u(rng)); __builtin_memcpy(&v, &h, 2); }
    uint16_t* d_idx = nullptr; void* d_pack = nullptr;
    cudaMalloc(&d_idx, sizeof(uint16_t) * m_idx[m].size());
    cudaMalloc(&d_pack, sizeof(uint16_t) * words);
    cudaMalloc((void**)&mats[m].suh, sizeof(half) * K);
    cudaMalloc((void**)&mats[m].svh, sizeof(half) * N);
    cudaMemcpy(d_idx, m_idx[m].data(), sizeof(uint16_t) * m_idx[m].size(), cudaMemcpyHostToDevice);
    cudaMemcpy(mats[m].suh, m_suh[m].data(), sizeof(half) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(mats[m].svh, m_svh[m].data(), sizeof(half) * N, cudaMemcpyHostToDevice);
    pack_trellis((uint16_t*)d_pack, d_idx, K / 16, N / 16, kBits);
    mats[m].pack = (uint16_t*)d_pack;
    cudaFree(d_idx);
  }

  // Device pointer tables, as the MoE builds them.
  std::vector<const uint16_t*> h_b(kSlots);
  std::vector<const half*> h_suh(kSlots), h_svh(kSlots);
  for (int j = 0; j < kSlots; j++) {
    h_b[j] = mats[kOrder[j]].pack;
    h_suh[j] = mats[kOrder[j]].suh;
    h_svh[j] = mats[kOrder[j]].svh;
  }
  const uint16_t** d_b = nullptr; const half** d_suh = nullptr; const half** d_svh = nullptr;
  cudaMalloc(&d_b, sizeof(void*) * kSlots);
  cudaMalloc(&d_suh, sizeof(void*) * kSlots);
  cudaMalloc(&d_svh, sizeof(void*) * kSlots);
  cudaMemcpy(d_b, h_b.data(), sizeof(void*) * kSlots, cudaMemcpyHostToDevice);
  cudaMemcpy(d_suh, h_suh.data(), sizeof(void*) * kSlots, cudaMemcpyHostToDevice);
  cudaMemcpy(d_svh, h_svh.data(), sizeof(void*) * kSlots, cudaMemcpyHostToDevice);

  // Inputs: one row per slot, distinct per slot so a misalignment cannot cancel out.
  const int rows = (bszm_in == 1) ? 1 : slots;
  std::vector<uint16_t> h_x((size_t)rows * M * K);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (auto& v : h_x) { __half h = __float2half_rn(u(rng)); __builtin_memcpy(&v, &h, 2); }
  half* d_x = nullptr;
  cudaMalloc((void**)&d_x, sizeof(half) * h_x.size());
  cudaMemcpy(d_x, h_x.data(), sizeof(half) * h_x.size(), cudaMemcpyHostToDevice);

  // Slot selections and weights, built first because the reference uses them too.
  std::vector<int64_t> h_ind((size_t)slots);
  std::vector<uint16_t> h_w((size_t)slots);
  // skip_pattern=0 reproduces the simple case; 1 reproduces what the ENGINE actually passes: some
  // slots belong to the other card and arrive as -1, and the router's weights are non-uniform.
  std::vector<double> wj((size_t)slots, 0.0);
  double wsum = 0;
  for (int j = 0; j < slots; j++) {
    const bool skip = (skip_pattern && (j % 3 == 1));
    h_ind[j] = skip ? -1 : (j % 5);
    const double w = skip ? 0.0 : (0.5 + 0.1 * (j % 4));
    wj[j] = w;
    wsum += w;
    __half hh = __float2half_rn((float)w);
    __builtin_memcpy(&h_w[j], &hh, 2);
  }
  for (int j = 0; j < slots; j++) wj[j] /= (wsum > 0 ? wsum : 1.0);   // normalize as the router does
  for (int j = 0; j < slots; j++) {
    __half hh = __float2half_rn((float)wj[j]);
    __builtin_memcpy(&h_w[j], &hh, 2);
  }

  // Reference: 10 separate gemm calls, one per slot, with the same row/expert pairing.
  std::vector<float> ref((size_t)slots * M * N, 0.0f);
  half* d_ahad = nullptr;
  cudaMalloc((void**)&d_ahad, sizeof(half) * (size_t)M * K);
  float* d_scratch = nullptr;
  cudaMalloc(&d_scratch, sizeof(float) * (size_t)slots * std::max(M, 1) * N);
  for (int j = 0; j < slots; j++) {
    // reference for slot j uses the matrix that slot j selects (h_ind[j]), not matrix j
    const int mj = h_ind[j] >= 0 ? (int)h_ind[j] : 0;
    GroupWords w;
    w.trellis = h_b[mj]; w.suh = h_suh[mj]; w.svh = h_svh[mj]; w.mul1 = 0;
    const half* xj = (const half*)h_x.data() + (size_t)(bszm_in == 1 ? 0 : j) * (size_t)M * K;
    cudaMemset(d_scratch, 0, sizeof(float) * (size_t)M * N);
    int sh = gemm(d_scratch, xj, w, M, N, K, kBits, true, 0, (half*)d_ahad, false, 0, 0);
    cudaError_t ge = cudaDeviceSynchronize();
    if (j == 0) printf("    [ref] gemm shape=%d err=%s\n", sh, cudaGetErrorString(ge));
    cudaMemcpy(ref.data() + (size_t)j * M * N, d_scratch, sizeof(float) * (size_t)M * N, cudaMemcpyDeviceToHost);
  }

  // mgemm: one call, indices = the identity order (j -> matrix j in the table), weights supplied so
  // the weighted reduction runs. num_tokens == 1 must collapse everything into row 0.
  int64_t* d_ind = nullptr; half* d_w = nullptr;
  cudaMalloc((void**)&d_ind, sizeof(int64_t) * slots);
  cudaMalloc((void**)&d_w, sizeof(half) * slots);
  cudaMemcpy(d_ind, h_ind.data(), sizeof(int64_t) * slots, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w.data(), sizeof(half) * slots, cudaMemcpyHostToDevice);

  float* d_y = nullptr;
  cudaMalloc(&d_y, sizeof(float) * (size_t)std::max(slots, num_tokens) * std::max(M,1) * N);
  cudaMemset(d_y, 0, sizeof(float) * (size_t)std::max(slots, num_tokens) * std::max(M,1) * N);
  // a_had must hold bszm * M * K halves (the kernel writes one transformed input slab PER SLOT).
  // Getting this wrong does not fault - it produces NaN - so size it from the same expression the
  // kernel uses rather than a guess.
  const size_t had_elems = (size_t)std::max(bszm_in, bszm_out) * (size_t)M * (size_t)K;
  half* d_had2 = nullptr;
  cudaMalloc((void**)&d_had2, sizeof(half) * had_elems);

  const int rc = mgemm(d_y, d_x, d_b, d_suh, d_svh, kSlots, /*M=*/M, N, K, kBits,
                       /*y_fp32=*/true, 0, (half*)d_had2, bszm_in, bszm_out, d_ind, d_w, slots,
                       /*a_had_elems=*/(int64_t)had_elems, /*min_index=*/-1, /*max_index=*/-1,
                       num_tokens, nullptr,
                       nullptr, /*mcg=*/false, /*mul1=*/false);
  cudaError_t err = cudaDeviceSynchronize();
  check(err == cudaSuccess, "mgemm returned no CUDA error");
  if (err != cudaSuccess) { printf("  cuda error: %s\n", cudaGetErrorString(err)); }
  (void)rc;

  std::vector<float> got((size_t)std::max(slots, num_tokens) * std::max(M,1) * N);
  cudaMemcpy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost);

  // Expected: token t's row is the weighted sum of ITS slots, [t*stride, (t+1)*stride). For
  // num_tokens == 1 that is every slot into row 0, which is what this asserted before; for
  // num_tokens == 2 it is two independent reductions, and it is that second case paired decode
  // depends on that nothing here ever exercised.
  const int stride = num_tokens > 0 ? slots / num_tokens : slots;
  double num = 0, den = 0, worst = 0;
  for (int t = 0; t < std::max(num_tokens, 1); t++) {
    for (int i = 0; i < M * N; i++) {
      double want = 0;
      for (int j = t * stride; j < (t + 1) * stride && j < slots; j++)
        if (h_ind[j] >= 0) want += wj[j] * ref[(size_t)j * M * N + i];
      const double g = got[(size_t)t * M * N + i];
      double d = g - want;
      num += d * d; den += want * want;
      double rel = std::fabs(d) / std::max(1e-3, std::fabs(want));
      if (rel > worst) worst = rel;
    }
  }
  const double rel_rms = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  printf("    [dbg] ref[0..2]=%.5f %.5f %.5f  got[0..2]=%.5f %.5f %.5f  den=%.3e\n",
         ref[0], ref[1], ref[2], got[0], got[1], got[2], den);
  printf("  %-46s rel RMS %.3e  worst %.3e\n", label, rel_rms, worst);
  check(rel_rms < 0.05, "weighted reduction matches 10 separate gemm calls");

  for (int m = 0; m < kSlots; m++) { cudaFree(mats[m].pack); cudaFree(mats[m].suh); cudaFree(mats[m].svh); }
  cudaFree(d_b); cudaFree(d_suh); cudaFree(d_svh); cudaFree(d_x);
  cudaFree(d_ahad); cudaFree(d_scratch); cudaFree(d_ind); cudaFree(d_w); cudaFree(d_y); cudaFree(d_had2);
}

}  // namespace

int main() {
  cudaSetDevice(0);
  printf("mgemm weighted multi-matrix semantics (slots=%d, N=%d, K=%d, %d-bit)\n", kSlots, N, K, kBits);
  // gate/up shape: M=1, distinct input row per slot
  run(1, kSlots, kSlots, /*M=*/1, 0, "gate/up shape: M=1,     bszm=slots");
  run(1, 1, kSlots, /*M=*/1, 0, "gate/up shape: M=1,     bszm_in=1 (broadcast)");
  run(1, kSlots, kSlots, /*M=*/1, 0, "down shape:    M=1,     bszm=slots");
  // The engine's real configuration: -1 for slots owned by the other card, non-uniform weights
  run(1, kSlots, kSlots, /*M=*/1, 1, "down shape: M=1, bszm=slots, SKIPS + non-uniform w");
  run(1, kSlots, kSlots, /*M=*/kSlots, 1, "down shape: M=slots, SKIPS + non-uniform w");
  // TWO tokens in one call. This is the configuration batched decode issues and the only one these
  // assertions never covered: everything above passes num_tokens=1, where the reduction is a single
  // sum into row 0. With num_tokens=2 the slots must split into two contiguous groups, each reduced
  // into its own row - and if that split is wrong, the two rows of a paired forward come out
  // different even when the two sequences are identical.
  const int kSlots2 = 2 * kSlots;
  run(2, kSlots2, kSlots2, /*M=*/1, 0, "2 tokens: num_tokens=2, bszm=2*slots");
  run(2, kSlots2, kSlots2, /*M=*/1, 1, "2 tokens: num_tokens=2, SKIPS + non-uniform w");
  if (failures) { printf("FAILURES: %d\n", failures); return 1; }
  printf("ALL OK\n");
  return 0;
}
