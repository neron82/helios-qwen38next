// Does cuda_recurrent_gated_delta_rule compute the right thing at bsz = 2, seqlen = 1?
//
// Batched decode is the only caller that asks the recurrence for two INDEPENDENT one-step states in
// a single launch. Every prefill chunk is bsz=1, seqlen=n, so this configuration had never been
// executed - and paired decode is currently wrong for reasons that narrow to it: in a paired forward
// with identical inputs in both slots, the two rows come out different, and every other n=2 path
// (the EXL3 projections, the mHC mixers) has been shown to agree with a multi-row prefill chunk.
//
// The test is the direct A/B, with no engine involved:
//   paired  - one call, bsz=2, seqlen=1, slots={0,1}, history_stride=1, over a [2][state] block
//   serial  - two calls, bsz=1, seqlen=1, each with its own base into that same block
// and compares BOTH the output rows and the resulting states. Either diverging is a failure: the
// output is what decode reads, and the state is what the next step would carry.
//
// history_stride=1 is the value the engine passes, because the per-slot state block is carved
// [layer_rank][slot]: two slots of the same layer are adjacent, so a slot difference is one state.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../gdn.cuh"

using namespace helios;
using namespace helios::aux;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

// Model shape, matching the checkpoint: 16 key heads, 48 value heads, 128 dims each.
static const int kNk = 16, kNv = 48, kHk = 128, kHv = 128;
static const size_t kGroup = (size_t)(kNv / kNk) * kNk * kHk * kHv;   // == state_size in the kernel
static const size_t kStateFloats = kGroup;                            // floats per sequence
static const int kQkvOut = (2 * kHk * kNk) + kHv * kNv;               // 10240, as on the real model

static float frand() { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

int main() {
  srand(1234);
  const size_t qkv_n = (size_t)kQkvOut;          // one token's qkv
  const size_t gn = (size_t)(kNv / kNk) * kNk; // one token's g and beta heads

  // Two independent tokens and two independent zero states.
  std::vector<__nv_bfloat16> h_qkv(2 * qkv_n);
  std::vector<float> h_g(2 * gn), h_beta(2 * gn);
  for (size_t i = 0; i < 2 * qkv_n; i++) h_qkv[i] = __float2bfloat16(frand() * 0.5f);
  for (size_t i = 0; i < 2 * gn; i++) { h_g[i] = frand() * 0.1f; h_beta[i] = frand() * 0.1f; }

  __nv_bfloat16 *d_qkv = nullptr; float *d_g = nullptr; __nv_bfloat16 *d_beta = nullptr;
  CK(cudaMalloc(&d_qkv, h_qkv.size() * 2));
  CK(cudaMalloc(&d_g, h_g.size() * 4));
  CK(cudaMalloc(&d_beta, h_beta.size() * 2));
  CK(cudaMemcpy(d_qkv, h_qkv.data(), h_qkv.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_g, h_g.data(), h_g.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * 2, cudaMemcpyHostToDevice));

  // State block carved the way the ENGINE carves it, not the way this test first did: a multi-layer
  // [layer_rank][slot] block with the base handed over at THIS layer's rank. The first version used a
  // bare [2][state] block whose base was the block start, so every offset the kernel computes was
  // relative to offset 0 - and the one structural difference from the real engine went untested.
  const int kLayers = 18, kSlot = 3;               // layers per card, slots per engine
  const size_t blk = (size_t)kLayers * kSlot * kStateFloats;
  const size_t rank = 5;                          // this layer's rank within its card
  const size_t base_off = (rank * kSlot + 0) * kStateFloats;
  float *d_state_paired = nullptr, *d_state_serial = nullptr;
  CK(cudaMalloc(&d_state_paired, blk * 4));
  CK(cudaMalloc(&d_state_serial, blk * 4));
  CK(cudaMemset(d_state_paired, 0, blk * 4));
  CK(cudaMemset(d_state_serial, 0, blk * 4));
  float* base_p = d_state_paired + base_off;
  float* base_s = d_state_serial + base_off;
  // Slot numbers are relative to row 0's slot, as decode_pair passes them.
  const int rel[2] = {0, 1};
  int* d_rel = nullptr;
  CK(cudaMalloc(&d_rel, 2 * sizeof(int)));
  CK(cudaMemcpy(d_rel, rel, 2 * sizeof(int), cudaMemcpyHostToDevice));

  __nv_bfloat16 *d_out_paired = nullptr, *d_out_serial = nullptr;
  CK(cudaMalloc(&d_out_paired, 2 * (size_t)kNv * kHv * 2));
  CK(cudaMalloc(&d_out_serial, 2 * (size_t)kNv * kHv * 2));
  CK(cudaMemset(d_out_paired, 0, 2 * (size_t)kNv * kHv * 2));
  CK(cudaMemset(d_out_serial, 0, 2 * (size_t)kNv * kHv * 2));

  int h_slots[2] = {0, 1};
  int *d_slots = nullptr;
  CK(cudaMalloc(&d_slots, 2 * sizeof(int)));
  CK(cudaMemcpy(d_slots, h_slots, 2 * sizeof(int), cudaMemcpyHostToDevice));

  // --- paired: one launch, bsz=2 ---
  cuda_recurrent_gated_delta_rule(
      d_qkv, d_g, d_beta, base_p, d_out_paired,
      /*bsz=*/2, /*seqlen=*/1, kNk, kNv, kHk, kHv,
      /*history_stride=*/1, d_rel, /*channelwise=*/false, /*history=*/false, 0);
  CK(cudaDeviceSynchronize());

  // --- serial: two launches, bsz=1, same block, each addressing its own slot ---
  for (int b = 0; b < 2; b++) {
    cuda_recurrent_gated_delta_rule(
        d_qkv + (size_t)b * qkv_n, d_g + (size_t)b * gn, d_beta + (size_t)b * gn,
        base_s + (size_t)b * kStateFloats, d_out_serial + (size_t)b * kNv * kHv,
        /*bsz=*/1, /*seqlen=*/1, kNk, kNv, kHk, kHv,
        /*history_stride=*/0, nullptr, /*channelwise=*/false, /*history=*/false, 0);
    CK(cudaDeviceSynchronize());
  }

  const size_t out_n = (size_t)kNv * kHv;
  std::vector<__nv_bfloat16> op(2 * out_n), os(2 * out_n);
  CK(cudaMemcpy(op.data(), d_out_paired, op.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(os.data(), d_out_serial, os.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<float> sp(2 * kStateFloats), ss(2 * kStateFloats);
  CK(cudaMemcpy(sp.data(), base_p, sp.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ss.data(), base_s, ss.size() * 4, cudaMemcpyDeviceToHost));

  double worst_out = 0, worst_st = 0;
  for (size_t i = 0; i < 2 * out_n; i++) {
    double a = __bfloat162float(op[i]), b = __bfloat162float(os[i]);
    double d = fabs(a - b) / (fabs(b) + 1e-6);
    if (d > worst_out) worst_out = d;
  }
  for (size_t i = 0; i < 2 * kStateFloats; i++) {
    double d = fabs((double)sp[i] - (double)ss[i]) / (fabs((double)ss[i]) + 1e-6);
    if (d > worst_st) worst_st = d;
  }

  printf("gdn bsz=2 vs 2x bsz=1: worst relative output diff %.3e, worst relative state diff %.3e\n",
         worst_out, worst_st);
  // The recurrence is fp32-accumulated but the inputs are bf16, so exact equality is not the bar;
  // a correct bsz=2 path lands in the same place as two bsz=1 calls to rounding.
  const double tol = 1e-5;
  if (worst_out > tol || worst_st > tol) {
    printf("FAIL: bsz=2 does not match two bsz=1 calls (tolerance %g)\n", tol);
    return 1;
  }
  // ---- the other half of the GDN layer: the depthwise conv ----
  //
  // test_gdn_bsz2 above covers the RECURRENCE only. The layer also calls cuda_causal_conv1d_update,
  // which in a batched decode is the first caller to pass bsz=2, seqlen=1 and a per-slot conv_state.
  // Identical input, identical state and a bit-identical recurrence still produced a divergent
  // per-row GDN output, so this is the remaining call that had never been executed at that shape.
  const int kDim = 10240, kK = 4, kState = 4;      // gdn_qkv_out, gdn_conv_k
  // x is [bsz, dim, seqlen], so with seqlen == 1 a batch item is `dim` halves. Sizing the
  // per-batch block as dim*K (which is what this first did) leaves the batch stride in the kernel,
  // dim*seqlen, disagreeing with the buffer, and BOTH rows then read the wrong x - which looks
  // exactly like a kernel bug and is not one. The per-slot report is what made that visible: slot
  // 0's state was bit-identical while its output was not, which can only be the input.
  const size_t xn = (size_t)kDim;                    // per-sequence input rows (seqlen == 1)
  std::vector<__nv_bfloat16> h_xc(2 * xn), h_wc((size_t)kDim * kK), h_bc(kDim);
  for (auto& v : h_xc) v = __float2bfloat16(frand() * 0.5f);
  for (auto& v : h_wc) v = __float2bfloat16(frand() * 0.1f);
  for (auto& v : h_bc)  v = __float2bfloat16(frand() * 0.1f);

  __nv_bfloat16 *d_xc = nullptr, *d_wc = nullptr, *d_bc = nullptr;
  CK(cudaMalloc(&d_xc, h_xc.size() * 2));
  CK(cudaMalloc(&d_wc, h_wc.size() * 2));
  CK(cudaMalloc(&d_bc, h_bc.size() * 2));
  CK(cudaMemcpy(d_xc, h_xc.data(), h_xc.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_wc, h_wc.data(), h_wc.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bc, h_bc.data(), h_bc.size() * 2, cudaMemcpyHostToDevice));

  // Non-zero initial state, so a slot writing into the wrong neighbour is visible.
  const size_t cs_n = (size_t)kDim * kState;       // one sequence's conv state, in halves
  std::vector<__nv_bfloat16> h_cs(2 * cs_n);
  for (auto& v : h_cs) v = __float2bfloat16(frand() * 0.2f);
  __nv_bfloat16 *d_cs_p = nullptr, *d_cs_s = nullptr;
  CK(cudaMalloc(&d_cs_p, h_cs.size() * 2));
  CK(cudaMalloc(&d_cs_s, h_cs.size() * 2));
  CK(cudaMemcpy(d_cs_p, h_cs.data(), h_cs.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cs_s, h_cs.data(), h_cs.size() * 2, cudaMemcpyHostToDevice));

  __nv_bfloat16 *d_outc_p = nullptr, *d_outc_s = nullptr;
  CK(cudaMalloc(&d_outc_p, 2 * xn * 2));
  CK(cudaMalloc(&d_outc_s, 2 * xn * 2));
  CK(cudaMemset(d_outc_p, 0, 2 * xn * 2));
  CK(cudaMemset(d_outc_s, 0, 2 * xn * 2));

  // paired: one launch, bsz=2, slots={0,1}, history_stride irrelevant here (the conv kernel's
  // per-slot stride is dim * state_size, which is one state - matching the [layer_rank][slot] carve).
  cuda_causal_conv1d_update(d_xc, d_cs_p, d_slots, d_wc, d_bc, d_outc_p,
                            /*bsz=*/2, kDim, /*seqlen=*/1, kState, kK,
                            /*activation=*/false, /*history=*/false, 0);
  CK(cudaDeviceSynchronize());
  // serial: two launches with the engine's exact single-sequence call, each on its own state base.
  for (int b = 0; b < 2; b++) {
    cuda_causal_conv1d_update(d_xc + (size_t)b * xn, d_cs_s + (size_t)b * cs_n, nullptr,
                              d_wc, d_bc, d_outc_s + (size_t)b * xn,
                              /*bsz=*/1, kDim, /*seqlen=*/1, kState, kK,
                              /*activation=*/false, /*history=*/false, 0);
    CK(cudaDeviceSynchronize());
  }

  std::vector<__nv_bfloat16> cop(2 * xn), cos_(2 * xn), ccp(2 * cs_n), ccs(2 * cs_n);
  CK(cudaMemcpy(cop.data(), d_outc_p, cop.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cos_.data(), d_outc_s, cos_.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ccp.data(), d_cs_p, ccp.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ccs.data(), d_cs_s, ccs.size() * 2, cudaMemcpyDeviceToHost));
  double worst_o = 0, worst_c = 0;
  for (size_t i = 0; i < 2 * xn; i++) {
    double d = fabs((double)__bfloat162float(cop[i]) - (double)__bfloat162float(cos_[i]));
    if (d > worst_o) worst_o = d;
  }
  for (size_t i = 0; i < 2 * cs_n; i++) {
    double d = fabs((double)__bfloat162float(ccp[i]) - (double)__bfloat162float(ccs[i]));
    if (d > worst_c) worst_c = d;
  }
  printf("conv1d bsz=2 vs 2x bsz=1: worst abs output diff %.6e, worst abs state diff %.6e\n",
         worst_o, worst_c);
  for (int b = 0; b < 2; b++) {
    double wo = 0, wc = 0;
    for (size_t i = 0; i < xn; i++) {
      double d = fabs((double)__bfloat162float(cop[(size_t)b * xn + i]) -
                      (double)__bfloat162float(cos_[(size_t)b * xn + i]));
      if (d > wo) wo = d;
    }
    for (size_t i = 0; i < cs_n; i++) {
      double d = fabs((double)__bfloat162float(ccp[(size_t)b * cs_n + i]) -
                      (double)__bfloat162float(ccs[(size_t)b * cs_n + i]));
      if (d > wc) wc = d;
    }
    printf("    slot %d: worst output diff %.6e, worst state diff %.6e\n", b, wo, wc);
  }
  if (worst_o > 1e-6 || worst_c > 1e-6) {
    printf("FAIL: conv1d bsz=2 does not match two bsz=1 calls\n");
    return 1;
  }

  printf("PASS\n");
  return 0;
}
