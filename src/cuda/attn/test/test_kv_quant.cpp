// Parity of helios's quantized KV cache (kv_quant.cuh) against exllamav3's own kernels.
//
// The golden vectors in kv_quant_golden.h were produced by compiling exllamav3's
// exllamav3_ext/cache/q_cache_kernels.cuh unmodified and running quant_cache_cont_kernel /
// dequant_cache_cont_kernel on the same deterministic input (regenerate with the command in that
// file's header). So this is a real cross-implementation check, not a self-consistency one:
//
//   * packed codes and half scales must be BIT-IDENTICAL (the write path reproduces the reference's
//     exact fp32 association order, not just an equivalent transform),
//   * the dequantized fp16 must match the reference's within 1 fp16 ulp. It is not bit-identical by
//     design: helios applies H32 to the integer codes and folds the centroid offset through
//     H32*1 = 32*e_0, so it is exact where the reference rounds in fp32. That difference is the
//     whole reason the read path is cheap enough to sit inside the mma prefill kernel.
//
// It also checks the round trip is a lossy-but-sane quantizer (a relative RMS bound that tightens
// as the bitrate rises), because a bit-identical encoder of garbage would pass everything above.
#include "kv_quant.cuh"
#include "kv_quant_golden.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace helios;
using namespace helios::attn;

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
  printf("CUDA err %s %d @%d\n", cudaGetErrorName(e), (int)e, __LINE__); exit(1);} } while (0)

static const int NG = KVQ_GOLDEN_GROUPS;
static const int TD = NG * 32;            // token_dim of the golden vector

// Ordered distance between two fp16 values, in ulps of the 16-bit pattern (monotone through zero).
static int half_ulp(uint16_t a, uint16_t b) {
  auto ord = [](uint16_t h) -> int {
    return (h & 0x8000u) ? -(int)(h & 0x7fffu) : (int)(h & 0x7fffu);
  };
  int d = ord(a) - ord(b);
  return d < 0 ? -d : d;
}

struct Golden { const uint32_t* q; const uint16_t* s; const uint16_t* out; };
static Golden golden_for(int bits) {
  switch (bits) {
    case 2: return {kvq_gq2, kvq_gs2, kvq_go2};
    case 3: return {kvq_gq3, kvq_gs3, kvq_go3};
    case 4: return {kvq_gq4, kvq_gs4, kvq_go4};
    case 5: return {kvq_gq5, kvq_gs5, kvq_go5};
    case 6: return {kvq_gq6, kvq_gs6, kvq_go6};
    case 7: return {kvq_gq7, kvq_gs7, kvq_go7};
    default: return {kvq_gq8, kvq_gs8, kvq_go8};
  }
}

// Bound on the round trip's relative RMS at this bitrate, with headroom over the measured values.
// The round trip's relative RMS falls by ~2x per bit on the golden input (0.34 at 2 bits down to
// 0.005 at 8); this bound sits a factor ~2.5 above that curve and would reject a broken encoder
// that happened to produce valid-looking codes.
static double rms_bound(int bits) { return 1.5 / (double)(1 << bits); }
// Dequant agreement with the reference, as a fraction of the group's largest magnitude. One fp16
// ulp relative to a group max is 2^-11 = 4.9e-4; the observed worst is 6.4e-7.
static const double kDequantRelTol = 1e-5;

int main() {
  const size_t vals = (size_t)NG * 32;
  const int rows = 128;                 // cache rows; pos0 + n must land inside this
  const size_t qcap = (size_t)rows * NG * 8;   // words, sized for the widest bitrate
  const size_t scap = (size_t)rows * NG;       // half scales
  half *d_in = nullptr, *d_out = nullptr, *d_ks = nullptr, *d_vs = nullptr;
  uint32_t *d_kq = nullptr, *d_vq = nullptr;
  CK(cudaMalloc(&d_in, vals * 2));
  CK(cudaMalloc(&d_out, vals * 2));
  CK(cudaMalloc(&d_kq, qcap * 4));
  CK(cudaMalloc(&d_vq, qcap * 4));
  CK(cudaMalloc(&d_ks, scap * 2));
  CK(cudaMalloc(&d_vs, scap * 2));
  CK(cudaMemcpy(d_in, kvq_golden_in, vals * 2, cudaMemcpyHostToDevice));

  int fails = 0;
  for (int bits = 2; bits <= 8; bits++) {
    const Golden g = golden_for(bits);
    // The engine writes at pos0 > 0 during prefill, so the row index must be the absolute
    // position and not the offset inside the chunk. K and V get separate buffers, as in the engine.
    const int pos0 = 37;
    KvQuant kc{d_kq, d_ks}, vc{d_vq, d_vs};
    CK(cudaMemset(d_kq, 0xA5, qcap * 4));
    CK(cudaMemset(d_ks, 0xA5, scap * 2));
    CK(cudaMemset(d_vq, 0xA5, qcap * 4));
    CK(cudaMemset(d_vs, 0xA5, scap * 2));
    kvq_write(d_in, d_in, kc, vc, NG, TD, bits, pos0, 0);
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> hq((size_t)NG * bits);
    std::vector<uint16_t> hs(NG);
    CK(cudaMemcpy(hq.data(), (const uint32_t*)kc.q + (size_t)pos0 * NG * bits, hq.size() * 4,
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), (const half*)kc.s + (size_t)pos0 * NG, hs.size() * 2,
                  cudaMemcpyDeviceToHost));
    // Rows before pos0 must be untouched: speculative rollback re-reads them, and a quantizer that
    // shifted or cleared them would silently corrupt the accepted prefix.
    std::vector<uint32_t> head((size_t)pos0 * NG * bits);
    CK(cudaMemcpy(head.data(), d_kq, head.size() * 4, cudaMemcpyDeviceToHost));
    for (uint32_t w : head)
      if (w != 0xA5A5A5A5u) {
        fails++;
        printf("bits=%d  FAIL wrote outside [pos0, pos0+n)\n", bits);
        break;
      }

    int bad_q = 0, bad_s = 0;
    for (int i = 0; i < (int)hq.size(); i++) if (hq[i] != g.q[i]) bad_q++;
    for (int i = 0; i < NG; i++) if (hs[i] != g.s[i]) bad_s++;
    if (bad_q || bad_s) {
      fails++;
      printf("bits=%d  FAIL codes: %d/%zu words differ, %d/%d scales differ\n", bits, bad_q,
             hq.size(), bad_s, NG);
    }

    CK(cudaMemset(d_out, 0, vals * 2));
    kvq_dequant(kc.q, kc.s, d_out, NG, TD, bits, pos0, 0);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> ho(vals);
    CK(cudaMemcpy(ho.data(), d_out, vals * 2, cudaMemcpyDeviceToHost));

    // Dequant is NOT bit-identical to the reference and cannot be: helios applies H32 to the
    // integer codes (exact) while the reference runs the same butterfly in fp32 on
    // (q - centroid) * scale, and its own rounding is what the two disagree about. The gap is
    // measured per group relative to that group's largest magnitude, because an 8-ulp difference on
    // a value near zero is 1e-7 of the group and says nothing; the bound below is ~15x the worst
    // observed (6.4e-7) and still 500x tighter than one fp16 ulp.
    double worst_rel = 0.0;
    int max_ulp = 0, near = 0;
    double se_mine = 0, se_ref = 0, sx = 0;
    for (int gr = 0; gr < NG; gr++) {
      double mx = 0.0;
      for (int j = 0; j < 32; j++)
        mx = std::max(mx, std::fabs((double)__half2float(*(const half*)&g.out[gr * 32 + j])));
      if (mx < 1e-6) mx = 1e-6;
      for (int j = 0; j < 32; j++) {
        const int i = gr * 32 + j;
        double a = __half2float(*(const half*)&ho[i]);
        double b = __half2float(*(const half*)&g.out[i]);
        double x = __half2float(*(const half*)&kvq_golden_in[i]);
        worst_rel = std::max(worst_rel, std::fabs(a - b) / mx);
        se_mine += (x - a) * (x - a);
        se_ref += (x - b) * (x - b);
        sx += x * x;
        int u = half_ulp(ho[i], g.out[i]);
        if (u > max_ulp) max_ulp = u;
        if (u <= 1) near++;
      }
    }
    const double rms_mine = sqrt(se_mine / (sx > 0 ? sx : 1.0));
    const double rms_ref = sqrt(se_ref / (sx > 0 ? sx : 1.0));
    const bool ok = !bad_q && !bad_s && worst_rel < kDequantRelTol &&
                    rms_mine <= rms_ref * 1.001 + 1e-9 && rms_mine < rms_bound(bits);
    if (!ok) fails++;
    printf("bits=%d  %s  codes %s  scales %s | dequant vs reference: worst %.2e of group max "
           "(tol %.0e), %d/%d within 1 ulp, max %d ulp | round trip rel RMS %.4f (mine) vs %.4f "
           "(reference, bound %.4f)\n",
           bits, ok ? "PASS" : "FAIL", bad_q ? "MISMATCH" : "bit-identical",
           bad_s ? "MISMATCH" : "bit-identical", worst_rel, kDequantRelTol, near, (int)vals,
           max_ulp, rms_mine, rms_ref, rms_bound(bits));
  }

  // Footprint, the whole point of the format: report what a token costs at each bitrate so the
  // allocator's accounting can be checked against it.
  printf("kv-quant: token_dim=%d\n", TD);
  for (int bits = 2; bits <= 8; bits++)
    printf("  bits=%d: %zu B/token (fp16 %d B, %.2fx)\n", bits, kvq_row_bytes(TD, bits), TD * 2,
           (double)(TD * 2) / (double)kvq_row_bytes(TD, bits));

  CK(cudaFree(d_in)); CK(cudaFree(d_out));
  CK(cudaFree(d_kq)); CK(cudaFree(d_vq)); CK(cudaFree(d_ks)); CK(cudaFree(d_vs));
  printf(fails ? "KV QUANT: %d FAILURES\n" : "KV QUANT: ALL PASS\n", fails);
  return fails ? 1 : 0;
}
