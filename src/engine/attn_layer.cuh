#pragma once
// One Qwen3.8 full-attention layer (12 of 48), single sequence, n tokens per call.
//
// The op sequence mirrors exllamav3's `decode_flash_attn_nc` (modules/attn.py:715-800), with every
// call verified against the ported kernel headers:
//   q_proj 2560->12288 fp16, then deinterleave_qg -> q [n,24,256] and gate g [n,24,256]
//     (the reference needs half for this split: modules/attn.py:554-560)
//   k_proj / v_proj 2560->512 fp16
//   per-head RMSNorm over head_dim 256 with constant_bias 1.0 (Qwen's q_norm/k_norm)
//   partial NEOX rope over the leading 64 channels (qwen_rope.cuh)
//   append k/v to the layer's contiguous cache, then dense causal GQA (qwen_gqa.cuh)
//   o *= sigmoid(g)  (interleaved gate; aux mul_sigmoid_)
//   o_proj 6144->2560 fp32 (the hyper-connection apply consumes a fp32 sublayer output)
//
// Norm and rope are separate calls rather than the reference's fused kernel. That is equivalent:
// the RMS is taken over all 256 channels and rope's rotation preserves the squared sum of the 64
// rotated ones, so the scale is identical, and rope is linear - verified numerically by the rope
// and GQA parity tests.

#include "core/model.hpp"
#include <vector>
#include "../cuda/attn/qsa.cuh"
#include "cuda/cuda_shim.hpp"

namespace helios {

struct AttnScratch {
  void *q_flat = nullptr, *k_flat = nullptr, *v_flat = nullptr;   // fp16 projections
  void *q = nullptr, *k = nullptr, *g = nullptr;                  // fp16 split, per-head
  void *o = nullptr;                                             // fp16 attention output
  int* pos = nullptr;                                            // [max_n] absolute positions
  void* a_had = nullptr;
  float* kv_part = nullptr;   // split-KV decode partials [n, nq, 16, 2+hd]

  // Dequant STAGING for a quantized KV cache (HELIOS_KV_QUANT, kv_quant.cuh). Laid out exactly like
  // the fp16 cache - [ctx][n_kv_heads][head_dim] fp16, row == absolute position - so the attention
  // kernels are pointed at this and none of them change. One pair per device, reused by every layer:
  // attn_layer is called once per layer on a stream that runs layers in order, so no two layers'
  // attention over the same buffer are ever in flight. Null when the cache is fp16.
  void *cq_k = nullptr, *cq_v = nullptr;

  // QSA sparse attention. One QsaLayerState per full-attention layer (there are 12), each holding that
  // layer's pooled 4-token block keys. Allocated in one slab, LAST, after everything else.
  std::vector<attn::QsaLayerState> qsa;
  void* qsa_slab = nullptr;    // the pooled_k / block_pos / tail_raw storage
  void* qsa_scratch = nullptr;
  void *qsa_tok_idx = nullptr, *qsa_tok_block = nullptr, *qsa_tok_off = nullptr;
  void* qsa_part = nullptr;
  void* qsa_grp = nullptr;
  int qsa_max_ctx = 0, qsa_scratch_rows = 0;

  int max_n = 0;
};

bool attn_scratch_init(AttnScratch& sc, int max_n, int device, int n_qsa_layers, int max_ctx,
                       int n_sel_override, int qsa_scratch_rows_override, int hidden_for_qsa,
                       int kv_token_dim = 0);
void attn_scratch_free(AttnScratch& sc);

// x: collapsed hidden fp16 [n, hidden]; y: sublayer output fp32 [n, hidden].
// k_cache/v_cache: fp16 [Tmax, n_kv_heads, head_dim], owned by the cache; this call writes rows
// [pos0, pos0+n) and then attends over [0, pos0+n).
// layer_index: position among the model's full-attention layers (0..n_full-1), used to index the
// per-layer QSA pooled-key state. Defaults to 0 so the MTP call site (a single draft layer) still
// compiles; the trunk runner passes the real index.
void attn_layer(const AttnWeights& w, const Config& cfg, AttnScratch& sc, const half* x, float* y,
                void* k_cache, void* v_cache, int n, int pos0, cudaStream_t s, int layer_index = 0);

}  // namespace helios
