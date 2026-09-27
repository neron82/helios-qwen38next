#pragma once
// Dense fp16 GEMM with an fp16-accumulator tensor-core MMA (ported from
// exllamav3_ext/hgemm_f16acc.cu).
//
// GeForce parts (this box: RTX 3090, sm_86) run mma.sync with fp32 accumulators at half the
// rate of the fp16-accumulator form. The kernel runs the MMA with fp16 accumulators and flushes
// the partial sums into fp32 registers every KSLICE = 32 elements of K, so the accumulation
// across K stays fp32 and the error is that of 32-term fp16 partials. That is what makes the
// reconstruct (prefill) formulation affordable on this card, and it is the kernel exllamav3 runs
// for its dense projections once the row count passes AUTO_RECONSTRUCT_THRESHOLD.
//
// Only the launcher changed in the port (kernel body and heuristics are byte-exact):
//   at::Tensor / at::cuda::getCurrentCUDAStream  -> raw device pointers + explicit Stream
//   TORCH_CHECK                                   -> HELIOS_ASSERT
//   at::cuda::getDeviceProperties                 -> cudaDeviceGetAttribute
//   strided-batched (hgemm_batched) entry point   -> dropped, not needed by the linear path
//   cuBLAS fallback (hgemm / hgemm_recon)         -> dropped: helios links no cuBLAS, so a
//                                                   caller that cannot use this kernel must fall
//                                                   back to the EXL3 trellis GEMM, not to cuBLAS

#include "helios_shim.cuh"

namespace helios
{
namespace exl3
{

// C[M, N] = A[M, K] @ B[K, N], all row-major and contiguous, A and B float16, C float16 or
// float32 (c_fp32). ldc is C's row stride in elements (N for a contiguous C).
//
// Returns false WITHOUT launching anything when the shape is not covered (K % 64, N % 128,
// 16-byte aligned A/B, vector-aligned C, sm_80+), or not worthwhile (too few blocks to fill the
// device and HELIOS_HGEMM_SMALL_SHAPES is unset), or when the device probe says the
// fp16-accumulator MMA does not pay. The caller is then expected to use the EXL3 trellis GEMM
// instead.
//
// Upstream declines the narrow shapes because its fallback is cuBLAS, which is still a
// tensor-core GEMM. helios links no cuBLAS, so its fallback is the trellis kernel; for a narrow
// projection (N = 512 or 640 at prefill width) that is not a near miss. HELIOS_HGEMM_SMALL_SHAPES=1
// therefore accepts any shape that passes the coverage test regardless of the block count.
bool hgemm_f16acc_try
(
    void*               c,
    const void*         a,
    const void*         b,
    int                 M,
    int                 N,
    int                 K,
    int                 ldc,
    bool                c_fp32,
    Stream              s = 0
);

// Per-device enable state as exllamav3 computes it: -1 unknown, 0 off, 1 on. Exposed so the
// engine can report which way the probe went for a given card. HELIOS_HGEMM_F16ACC=0/1
// overrides the probe, exactly as EXL3_HGEMM_F16ACC does upstream.
int hgemm_f16acc_status(int device);

} // namespace exl3
} // namespace helios
