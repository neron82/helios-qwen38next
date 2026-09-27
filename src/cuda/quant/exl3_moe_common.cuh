#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1
#define MOE_ACT_RELU2_NOGATE 2  // non-gated relu2 (NemotronH): gate GEMM and staging skipped

#define MOE_SMS_PER_EXPERT 8       // default/minimum group width, also sets max concurrency (buffer count)
#define MOE_MAX_SMS_PER_EXPERT 32  // widest expert group when few experts are active
#define MOE_TILESIZE_K 32
#define MOE_TILESIZE_M 16
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3

#ifndef EXL3_GEMM_BASE_THREADS
#define EXL3_GEMM_BASE_THREADS 256
#endif

#ifndef SMEM_MAX
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6
#endif

// output_state is a 2^40 FIXED-POINT int64 accumulator, not fp32. Several groups handle different
// experts of the same token and accumulate into that token's row concurrently, with no cross-group
// barrier; a float atomicAdd there is order-dependent, which made greedy decode irreproducible
// (seeded at ~3e-10 in the layer-0 MoE output, compounding with depth until a token flipped). See
// had_hf_r_128_d_inner. The host zeroes it, runs the kernel, then converts back to fp32 once.
// expert_offset is the exclusive prefix sum of expert_count in EXPERT-INDEX order (the same
// `offset` moe_offset_k already builds); lpt_order is the permutation of 0..num_experts-1 sorted
// by (token_count DESC, expert_index ASC), i.e. the LPT visiting order. Both are required because
// the kernel walks experts in LPT order, not index order. See exl3_moe_kernel.cuh.
#define EXL3_MOE_KERNEL_ARGS                    \
    const half* __restrict__ hidden_state,      \
    half* __restrict__ temp_state_g,            \
    half* __restrict__ temp_state_u,            \
    half* __restrict__ temp_intermediate_g,     \
    half* __restrict__ temp_intermediate_u,     \
    long long* __restrict__ output_state,      \
    const uint16_t** __restrict__ gate_trellis, \
    const half** __restrict__ gate_suh,         \
    const half** __restrict__ gate_svh,         \
    const uint16_t** __restrict__ up_trellis,   \
    const half** __restrict__ up_suh,           \
    const half** __restrict__ up_svh,           \
    const uint16_t** __restrict__ down_trellis, \
    const half** __restrict__ down_suh,         \
    const half** __restrict__ down_svh,         \
                                                \
    const int64_t* __restrict__ expert_count,   \
    const int64_t* __restrict__ expert_offset,  \
    const int* __restrict__ lpt_order,          \
    const int64_t* __restrict__ token_sorted,   \
    const half* __restrict__ weight_sorted,     \
                                                \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
    const int max_tokens_per_expert,            \
    const int concurrency,                      \
    const float act_limit,                      \
    const int act_function,                     \
    const int K_gate,                           \
    const int K_up,                             \
    const int K_down,                           \
                                                \
    int* __restrict__ locks
