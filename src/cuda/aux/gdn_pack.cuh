#pragma once
// GDN input packing for Qwen3.8-Flash-Next (new for the Qwen port).
//
// exllamav3's GDN prologue (`gated_delta_net_fused_op`) reads its inputs in a per-k-head layout that
// the checkpoint's two separate projections do not produce. Verified against the kernel's own
// indexing (gdn.cu:46-160):
//   in_qkvz [B,S,Nk,Fseg], Fseg = 2*Hk + 2*Ng*Hv, per k-head: q | k | v (Ng*Hv) | z (Ng*Hv)
//   in_ba   [B,S,Nk,Fba],  Fba  = 2*Ng,           per k-head: b (Ng) | a (Ng)
// while the checkpoint gives flat [q (Nk*Hk) | k (Nk*Hk) | v (Nv*Hv)] and [z (Nv*Hv)], with v-head
// kh*Ng+g belonging to k-head kh. Repacking at runtime costs one gather of 16 KB per token and keeps
// the quantized weights untouched (a load-time permutation of exl3 rows would be the alternative).

#include "../cuda_shim.hpp"

namespace helios { namespace aux {

// qkv: [S, 2*Nk*Hk + Nv*Hv] fp32 (q then k then v), z: [S, Nv*Hv] fp32,
// a, b: [S, Nv] fp32. Outputs qkvz [S, Nk*Fseg] and ba [S, Nk*2*Ng], both fp32.
void gdn_pack
(
    const float* qkv,
    const float* z,
    const float* a,
    const float* b,
    float* qkvz,
    float* ba,
    int S, int Nk, int Ng, int Hk, int Hv,
    Stream s = 0
);

}} // namespace helios::aux
