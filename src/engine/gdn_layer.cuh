#pragma once
// One Qwen3.8 GatedDeltaNet (linear attention) layer, single sequence, n tokens per call.
//
// Every call in the sequence was verified against the ported kernel sources rather than assumed:
//   projections      exl3::gemm (EXL3, Hadamard handled internally, fp32 out; src/engine/runner.cpp
//                    in the GLM engine uses the same pattern)
//   a/b              gdn_ba_gemv: y[rows,n] = x[rows,k] @ w_t[n,k].T, w_t = the checkpoint's
//                    [48,2560] orientation, fp32 output (gdn.cu:1134-1136)
//   packing          gdn_pack: flat [q|k|v] + [z] -> per-k-head q|k|v|z, and [b|a] per k-head
//   split/gate/beta  gated_delta_net_fused_op: mixed_qkv [Fout,S] bf16, z [S,Nv,Hv] bf16,
//                    beta = sigmoid(b)*scale, g = -exp(a_log) * softplus(a + dt_bias) (gdn.cu:46-178)
//   conv             cuda_causal_conv1d_update: x [bsz,dim,seqlen] bf16, state [dim,state_size] bf16,
//                    activation = swish, bias null (this checkpoint has no conv bias) (gdn.cu:944-960)
//   recurrence       cuda_recurrent_gated_delta_rule: head-wise decay (channelwise=false),
//                    slots=null/history=false for a single sequence, scale 1/sqrt(Hk)
//   output norm      gated_rms_norm with gate_act = sigmoid, dim 128, weight 128 (norm.cuh:71-90)
//   out_proj         exl3::gemm -> fp32 (the hyper-connection apply consumes a fp32 sublayer output)

#include "core/model.hpp"
#include "cuda/cuda_shim.hpp"

namespace helios {

// Per-layer scratch for a chunk of up to max_n tokens (single sequence).
struct GdnScratch {
  float *qkv_flat = nullptr, *z_flat = nullptr, *a_out = nullptr, *b_out = nullptr;
  float *qkvz = nullptr, *ba = nullptr, *g = nullptr;
  void *mixed_qkv = nullptr, *z16 = nullptr, *beta = nullptr, *conv_out = nullptr, *core_out = nullptr;
  half *normed = nullptr;
  void* a_had = nullptr;
  // Chunked gated-delta-rule workspace (see aux::cuda_chunked_gated_delta_rule). Sized off max_n and
  // only populated when that path is enabled; gdn_chunk_tokens is how many leading tokens it may
  // take, the rest falls to the serial kernel.
  void* gdn_chunk_ws = nullptr;
  size_t gdn_chunk_ws_bytes = 0;
  int gdn_chunk_tokens = 0;
  int max_n = 0;
};

bool gdn_scratch_init(GdnScratch& sc, int max_n, int device);
void gdn_scratch_free(GdnScratch& sc);

// x: collapsed hidden for this chunk, fp16 [n, hidden]. y: sublayer output fp32 [n, hidden].
// conv_state: bf16 [qkv_out, 4]; rec_state: fp32 [v_heads, v_dim, k_dim] - both owned by the cache.
// bsz / gdn_slots / conv_slots / slot_layers batch `bsz` sequences (one row each) through one
// forward; see the definition for how the two state arrays are addressed differently. Defaults keep
// the single-sequence call site byte-identical.
void gdn_layer(const GdnWeights& w, const Config& cfg, GdnScratch& sc, const half* x, float* y,
               void* conv_state, float* rec_state, int n, cudaStream_t s, int bsz = 1,
               const int* gdn_slots = nullptr, const int* conv_slots = nullptr,
               int slot_layers = 0);

}  // namespace helios
