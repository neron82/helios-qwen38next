#pragma once
// Partial NEOX RoPE for Qwen3.8-Flash-Next (new for this port - the GLM engine's MLA layers had no
// rotary, so there was no rope kernel to reuse).
//
// Verified against exllamav3 (util/rope.py):
//   rope_type "mrope" takes the *default* parameter path (rope.py:171-172), so:
//     rotary_dim      = int(head_dim * partial_rotary_factor) = int(256 * 0.25) = 64
//     inv_freq[i]     = 1 / theta^(2i / rotary_dim), i in [0, 32), theta = 1e7 (rope.py:191)
//     attn_factor     = 1.0 (no YaRN / longrope scaling on this checkpoint)
//     mrope_section   stored but NOT applied ("Ignored in HF impl., always True", rope.py:193)
//   rope_style defaults to RopeStyle.NEOX (rope.py:25) and qwen4_exp does not override it, so pairs
//   are the two halves of the rotary block: (i, i + rotary_dim/2).
// The rotation applies contiguously to the leading rotary_dim channels of each head, after the
// per-head q/k RMSNorm (which is a separate call - see qwen_attn.cuh).

#include "../cuda_shim.hpp"

namespace helios { namespace attn {

// q: [rows, n_q_heads, head_dim] fp16 in place; k: same with n_kv_heads, or null.
// pos: [rows] int32 absolute positions. head_dim is the full head width; rotary_dim the rotated part.
void rope_qk_partial_neox
(
    void* q,
    void* k,
    int rows,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim,
    const int* pos,
    float theta,
    Stream s = 0
);

}} // namespace helios::attn
