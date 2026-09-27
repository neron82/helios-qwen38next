#pragma once
// Standard softmax router for Qwen3.8-Flash-Next (new for this port).
//
// The helios port kept only the DS3 routing (sigmoid + pre-top-k bias); this model uses the std
// variant, so it is ported here from exllamav3_ext/routing.cu. Semantics, verified from that source:
//   * selection key   = the raw logit (+ bias if provided; this checkpoint has none)
//   * weights         = softmax over the SELECTED top-k logits only, i.e. exp(logit - max_logit)
//                       normalised by the sum over those k - not a global softmax
//   * optional per-expert scale (this checkpoint has none; config has no routed_scaling_factor)
//   * indices are int64, weights fp16, K per row
// Input is the precomputed logit row [bsz, num_experts] fp16.

#include "../cuda_shim.hpp"

namespace helios { namespace aux {

void routing_std_logits
(
    const half* scores,           // [bsz, num_experts] fp16
    int64_t* topk_indices,        // out [bsz, K]
    half* topk_weights,           // out [bsz, K]
    int bsz,
    int num_experts,
    int K,
    Stream s = 0
);

}} // namespace helios::aux
