#pragma once
// HELIOS_MIXER_TC=1: tensor-core / fused-register GatedResidual mixer.
//
// A SEPARATE path from the fp32 one in gr_mix.cu, which stays exactly as it was and remains the
// numerical oracle every parity test compares against. The switch lives in gr_mix(); this header
// only exposes the implementation and the shapes it can take.
//
// What it changes, structurally:
//   * `normed` (R, H*D) is never written to memory. The stream chunk is re-read and re-scaled
//     into registers by each kernel that needs it, and the per-(row, stream) rmr is split out of
//     the reduction up front (gr_tc_prep_kernel) so the dots GEMM can scale its A operand on load.
//   * `gated` (R, H*D) is never written to memory. The gate GEMM's epilogue reduces the H streams
//     in registers against the same normed operand and writes `mixed` (R, D) directly.
//   * Both projections are mma.m16n8k16 with fp32 accumulate instead of scalar FMA.
//
// It is not bit-identical to the fp32 path and does not claim to be: the A operands (normed, and
// t) are narrowed to fp16, which the checkpoint's own down/up/inject/norm weights already are.
// Measured error against the fp32 oracle is in RESULTS.md.

#include "../cuda_shim.hpp"

namespace helios { namespace aux {

// True when the TC path can serve this call: it needs 4 streams, a rank and a D that tile into the
// 64x64 block shape, and enough rows to fill one row-tile (below that the fp32 path's decode
// kernels, which are latency-shaped rather than throughput-shaped, are faster - measured).
bool gr_mix_tc_applicable(int R, int H, int D, int rank, int min_r);

// Dynamic shared memory the two GEMM kernels ask for, so gr_mix() can check it against the device
// limit once instead of letting the launch fail.
size_t gr_mix_tc_smem();

// Workspace the caller (gr_mix) owns and grows with its own temporaries:
//   t16 (R, rank) fp16, rmr (R, H) fp32, ph (R, H, H) fp32.
void gr_mix_tc
(
    const float* streams,
    const half* norm_raw,
    const half* down,
    const half* up,
    const half* inject,     // may be null, matching gr_mix: no post site
    int R, int H, int D, int rank,
    float eps,
    float* mixed,           // out (R, D) fp32
    float* post,            // out (R, H) fp32 or null
    half* nrm16,            // work (R, H*D) fp16: the dots GEMM's A operand
    half* t16,              // work (R, rank) fp16: the gate GEMM's A operand
    float* rmr,             // work (R, H) fp32
    float* ph,              // work (R, H, H) fp32
    Stream s = 0
);

}} // namespace helios::aux
