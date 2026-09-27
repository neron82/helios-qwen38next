#pragma once
// GatedResidual (qwen4_exp hyper-connection site) - new for the Qwen port. The mHC kernels above
// are GLM's; this class is a different operator: no combine matrix, no Sinkhorn, no hc_eps.
//
// Implements hyperconnections.py::_mix_ref exactly, because exllamav3 designates that pure-fp32
// reference as the parity oracle (its own fused gr_mix folds the norm into fp16 weights and the
// half-GEMM path runs in fp16 - the two disagree in low-order bits, so a port must pick one, and
// the reference is the one their tests compare against).
//
//   rmr[r,h]    = rsqrt(mean_d(x[r,h,d]^2) + eps)                     per stream, over D
//   normed      = x * rmr * (norm_w_raw + 1)                          +1 constant bias
//   t[r,i]      = silu( (1/H) * sum_{h,d} normed[r,h,d] * down[i, h*D+d] )
//   g[r,h*D+d]  = sum_i t[r,i] * up[h*D+d, i]
//   mixed[r,d]  = (1/H) * sum_h sigmoid(g[r,h,d]) * normed[r,h,d]
//   post[r,h]   = 2 * sigmoid( (1/H) * sum_d normed[r,h,d] * inject[h, h*D+d] )   [site form only]
//
// apply_ (site form): x[r,h,d] <- post[r,h] * y[r,d] + x[r,h,d]  (post is per stream, shared across d)

#include "../cuda_shim.hpp"

namespace helios { namespace aux {

// mixed is (R, D) fp32 (the caller casts to fp16 for the sublayer input); post is (R, H) fp32 and
// may be null for the combine-less final mixer. inject may be null when post is null.
void gr_mix
(
    const float* streams,   // (R, H, D) fp32, contiguous
    const half* norm_raw,   // (H*D) checkpoint hc_norm.weight (used as 1 + w)
    const half* down,       // (rank, H*D)
    const half* up,         // (H*D, rank)
    const half* inject,     // (H, H*D) or null
    int R, int H, int D, int rank,
    float eps,
    float* mixed,           // out (R, D)
    float* post,            // out (R, H) or null
    Stream s = 0
);

// Site residual update, in place: x[r,h,d] = post[r,h] * y[r,d] + x[r,h,d].
void gr_apply
(
    float* streams,         // (R, H, D) fp32, updated in place
    const float* y,         // (R, D) sublayer output
    const float* post,      // (R, H)
    int R, int H, int D,
    Stream s = 0
);

// Test hook: pin gr_mix's `up` stage to the legacy warp-per-element kernel, or release it.
// The two are bitwise identical by construction; this is what lets test_aux prove it with memcmp
// instead of a tolerance. Same switch as HELIOS_GR_UP=0.
void gr_force_legacy_up(bool on);

// Test hook: pin gr_mix's `dots` stage to the legacy DBK=128 tiled kernel, or release it. Same
// switch as HELIOS_GR_DOTS=0.
void gr_force_legacy_dots(bool on);

// Test/bench hook: pin gr_mix to the tensor-core path (gr_mix_tc.cu) or release it back to the
// env. `min_r` is the row count below which the fp32 path is kept either way; 0 pins it for every
// R. This is the A/B lever, and what the accuracy harness uses to run both paths over the same
// real activations in one process. Same switch as HELIOS_MIXER_TC=1.
void gr_force_tc(bool on, int min_r = 0);

// The engine-wide "use the tensor-core paths" bit behind HELIOS_MIXER_TC (on unless the variable
// is 0), with no applicability test. Anything outside the mixer that wants to ride the same switch
// - so that one variable pins the whole engine to its scalar reference oracle, or turns the whole
// engine's tensor-core surface on - reads this rather than re-parsing the environment.
bool gr_mixer_tc();

// True when the tensor-core path would serve this call (rows, H, D, rank and the threshold).
bool gr_mixer_tc_active(int R, int H, int D, int rank);

}} // namespace helios::aux
