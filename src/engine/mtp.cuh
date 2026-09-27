#pragma once
// MTP draft head for Qwen3.8-Flash-Next: input combine over the trunk's PRE-collapse stream stack
// plus the next token's embedding, one full decoder block, then its own combine-less mixer.
//
// Upstream (qwen4_exp_mtp.py) states there is no reference implementation for this head and that
// the stream-tap handling is a semantic guess to be confirmed by acceptance rate; `stream_tap`
// therefore selects between the two candidate semantics and defaults to the shipped one.
#include "core/model.hpp"
#include "engine/attn_layer.cuh"
#include "engine/moe_layer.cuh"

#include <vector>

namespace helios {

struct MtpScratch {
  // Input combine
  float* stack_ = nullptr;     // [n, H, D] fp32 normed trunk tap
  half* normed_ = nullptr;     // [n * H, D] fp16 for fc_hidden
  half* hidden_ = nullptr;     // [n * H, D] fp16 fc_hidden output
  half* emb16_ = nullptr;      // [n, D] token embedding
  half* emb_norm_ = nullptr;   // [n, D] pre_fc_norm_embedding output
  half* emb_fc_ = nullptr;     // [n, D] fc_embedding output
  float* streams_ = nullptr;   // [n, H, D] fp32 draft stream stack
  float* tap_ = nullptr;       // [n, H, D] fp32 saved pre-mixer stack (feeds the next step)
  void* had_ = nullptr;        // fp16 Hadamard scratch for the two fc gemms
  int* ids_dev_ = nullptr;
  float* post_ = nullptr;     // [n * H] fp32 per-stream residual gate of each hc site
  int max_n = 0;
  bool ready = false;
};

bool mtp_scratch_init(MtpScratch& sc, const Config& cfg, int max_n, int device);

// One draft step. `trunk_streams` is the trunk's PRE-collapse stack [n, H, D] fp32 at the current
// position; `ids` are the n tokens being fed to the draft head (the first is the just-sampled
// token). Writes `draft_ids` (device int[n-1]) and the new tap stack, and applies the draft mixer
// to `logits` (device fp32 [vocab]).
//
// Decode-time use: run the trunk on one token, call this with the trunk's saved stack and that
// token's id to predict the following token, verify it against the trunk's own next output, and
// re-run with both ids on a hit (greedy chain). `n == 2` is the first step of that chain.
void mtp_draft_step(Model& m, MtpScratch& sc, AttnScratch& attn, MoeScratch& moe,
                    const float* trunk_streams, const int* ids, int n, int pos0, void* k_cache,
                    void* v_cache,
                    float* tap_out, int* draft_ids, float* logits, bool stream_tap,
                    cudaStream_t s,
                    // Prefill uses this to fill the draft head's KV cache and nothing else: the head
                    // gemm reads the whole 5-bit lm_head (397 MB) and its result is discarded there.
                    bool compute_head = true,
                    // HELIOS_PIPELINE: the embedding table lives on layer 0's card while this head
                    // runs on the LAST layer's card, so the caller gathers the n rows on card 0 and
                    // hands them over. Non-null => use it verbatim and skip the (wrong-device) gather.
                    const half* emb16_in = nullptr);

// Trunk-side per-stream RMS norm used by the MTP tap: out[h,d] = x[h,d] * rsqrt(mean_d x^2 + eps)
// * (w[h,d] + 1). Exposed because it is also the exact form the draft input needs.
void mtp_tap_norm(const float* stack, const half* norm_w, float* out, int n, int H, int D, float eps,
                  cudaStream_t s);

// Greedy argmax of each of the first `rows` rows of a (rows x vocab) fp32 logits buffer, written to
// `out` as `rows` ints. This replaces a per-row 992 KB device-to-host copy plus a host max_element
// over 248,077 floats, which the speculative path was paying every step; the comparison it exists
// for (does the trunk agree with the draft?) only needs one int per row back.
//
// Ties resolve to the LOWEST index, matching std::max_element's first-maximum behaviour, so the
// token stream is unchanged.
void row_argmax(const float* logits, int rows, int vocab, int* out, cudaStream_t s);

}  // namespace helios
