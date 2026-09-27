// Bandwidth benchmark for the ported EXL3 GEMM, on the shapes this checkpoint actually uses.
//
// The engine's MoE phase runs at ~54-60 GB/s effective, on both prefill (grouped kernel) and decode
// (mgemm), while the card has ~700-800 GB/s available. The same ~13x shortfall in two different
// kernels is suspicious, so this measures the raw kernel in isolation: if `gemm` reaches bandwidth
// here, the engine's call pattern is at fault; if it does not, the kernel is.
//
// Reports effective GB/s over the weight bytes actually read (n*k*bits/8), which is the only traffic
// a memory-bound quantized GEMM must pay: the activations are negligible and the output is n floats.
//
//   ./build/quant/bench_gemm [--quick]
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "helios_shim.cuh"
#include "pack.cuh"
#include "exl3_gemm.cuh"

using namespace helios;
using namespace helios::exl3;

namespace {

struct Shape {
  const char* name;
  int bits, N, K;          // C is [M, N], A is [M, K], B is [K, N]
};

// - expert gate/up/down: N=640, K=2560, 2-bit  (the MoE hot path)
// - q_proj:              N=12288, K=2560, 4-bit
// - o_proj:              N=2560, K=12288, 4-bit
// - lm_head:             N=248320, K=2560, 5-bit
const Shape kShapes[] = {
    {"expert   N=640   K=2560  b2", 2, 640, 2560},
    {"q_proj   N=12288 K=2560  b4", 4, 12288, 2560},
    {"o_proj   N=2560  K=12288 b4", 4, 2560, 12288},
    {"lm_head  N=248320 K=2560 b5", 5, 248320, 2560},
};

const int kMValues[] = {1, 2, 4, 10, 32, 128};

void run_shape(const Shape& sh, int M, int device) {
  const int bits = sh.bits, N = sh.N, K = sh.K;
  const int packed_rows = K / 16, packed_cols = N / 16;
  const size_t trellis_words = (size_t)packed_rows * packed_cols * (256 * bits / 16);

  std::mt19937 rng(0xC0FFEEu + bits * 97 + M);
  std::uniform_int_distribution<int> idx_dist(0, (1 << bits) - 1);
  std::vector<uint16_t> h_idx((size_t)K * N);
  for (auto& v : h_idx) v = (uint16_t)idx_dist(rng);
  std::vector<uint16_t> h_scale(K + N);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  for (auto& v : h_scale) {
    float f = unit(rng);
    __half h = __float2half_rn(f);
    __builtin_memcpy(&v, &h, 2);
  }
  std::vector<uint16_t> h_A((size_t)M * K);
  for (auto& v : h_A) {
    float f = unit(rng);
    __half h = __float2half_rn(f);
    __builtin_memcpy(&v, &h, 2);
  }

  void *d_idx = nullptr, *d_pack = nullptr;
  half *d_suh = nullptr, *d_svh = nullptr, *d_A = nullptr, *d_ahad = nullptr;
  float* d_C = nullptr;
  cudaMalloc(&d_idx, sizeof(uint16_t) * h_idx.size());
  cudaMalloc(&d_pack, sizeof(uint16_t) * trellis_words);
  cudaMalloc((void**)&d_suh, sizeof(half) * K);
  cudaMalloc((void**)&d_svh, sizeof(half) * N);
  cudaMalloc((void**)&d_A, sizeof(half) * (size_t)M * K);
  cudaMalloc((void**)&d_ahad, sizeof(half) * (size_t)M * K);
  cudaMalloc((void**)&d_C, sizeof(float) * (size_t)M * N);
  cudaMemcpy(d_idx, h_idx.data(), sizeof(uint16_t) * h_idx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_suh, h_scale.data(), sizeof(half) * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_svh, h_scale.data() + K, sizeof(half) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A.data(), sizeof(half) * (size_t)M * K, cudaMemcpyHostToDevice);
  pack_trellis((uint16_t*)d_pack, (const uint16_t*)d_idx, packed_rows, packed_cols, bits);

  GroupWords w;
  w.trellis = (uint16_t*)d_pack;
  w.suh = d_suh;
  w.svh = d_svh;
  w.mul1 = 0;

  const int iters = M <= 2 ? 200 : (M <= 10 ? 100 : 30);
  int shape = 0;
  for (int i = 0; i < 3; i++)   // warm-up (attribute set, module load, L2)
    shape = gemm(d_C, d_A, w, M, N, K, bits, true, 0, d_ahad, false, 0, 0);
  cudaDeviceSynchronize();

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0, 0);
  for (int i = 0; i < iters; i++)
    gemm(d_C, d_A, w, M, N, K, bits, true, 0, d_ahad, false, 0, 0);
  cudaEventRecord(e1, 0);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  ms /= iters;

  const double wbytes = (double)N * K * bits / 8.0;
  const double gbs = wbytes / (ms * 1e-3) / 1e9;
  const double gflop = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
  printf("  %-32s M=%-4d %7.3f ms  %7.1f GB/s  %6.2f TFLOP/s  shape=%d%s\n", sh.name, M, ms, gbs,
         gflop, shape, shape == 90 ? " (gemv)" : "");

  cudaFree(d_idx); cudaFree(d_pack); cudaFree(d_suh); cudaFree(d_svh);
  cudaFree(d_A); cudaFree(d_ahad); cudaFree(d_C);
  (void)device;
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  cudaSetDevice(0);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  // memoryClockRate was removed from cudaDeviceProp in CUDA 13, so report what is still available.
  printf("device: %s (%d SMs, bus %d-bit)\n", prop.name, prop.multiProcessorCount,
         prop.memoryBusWidth);
  for (const Shape& sh : kShapes) {
    for (int M : kMValues) {
      if (quick && M != 1 && M != 10) continue;
      run_shape(sh, M, 0);
    }
  }
  return 0;
}
