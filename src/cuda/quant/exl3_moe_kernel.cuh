#pragma once

#include <cuda_bf16.h>

#include <stdio.h>
#include <stdlib.h>

#include "exl3_moe_common.cuh"
#include "helios_shim.cuh"
#include "helios_shim.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"
#include "ptx.cuh"

template<int t_bits, int MOE_TILESIZE_N, int cb, int M_TILE = MOE_TILESIZE_M>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int group_size = gridDim.x;  // SMs per expert, set at launch
    const int num_groups = gridDim.z;
    const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;  // blockDim.x
    const int group_threads = group_size * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    // Buffers for group
    temp_state_g += group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += group_idx * max_tokens_per_expert * intermediate_dim;

    // Barriers for group sync
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Expert scheduler state, self-resetting: [0] next ticket, [1] retired groups, [2 + g] ticket for group g
    int* sched = locks + MOE_SCHED_OFFSET;

    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    // Deterministic LPT expert scheduler.
    //
    // The host precomputes `lpt_order`, the permutation of 0..num_experts-1 sorted by
    // (token_count DESC, expert_index ASC), and rank r is handled by group r % num_groups in round
    // r / num_groups. The previous schedule round-robined over SCAN order, which hands the k-th
    // expert in index order to group k % num_groups: a group that draws several of the fat experts
    // runs long while another idles, and this launch is makespan-bound on the busiest group. With
    // prefill's counts (top-10 of 512 experts, ~19 tokens/expert on average but a 201-token worst
    // case) scan order is not close to balanced, so routing the heavy experts first is the fix.
    // LPT is the right shape rather than a measured-cost greedy: what makes this reproducible is
    // that the assignment is a pure function of the counts, and the counts are themselves
    // deterministic, so a count-based order is as reproducible as the index order was.
    //
    // Which expert a group handles, and in what order, is the only thing that has to stay fixed:
    // every group accumulates into the same output rows (a token's top-k experts can be spread
    // across groups), so a varying assignment also varies the order in which one token's expert
    // contributions are summed. The accumulation itself is exact integer addition into 2^40
    // fixed point (see had_hf_r_128_d_inner) and is therefore order-independent. This was tried
    // with a greedy `atomicAdd(&sched[0], 1)` ticket draw and identical runs diverged: relative
    // spread ~3e-10 seeded at prefill layer 0's MoE output, growing monotonically with depth to
    // ~1.5e-2 and eventually flipping a token id. The reference is byte-reproducible, and
    // determinism is not optional here - so the ticket stays a pure function of group_idx and the
    // round count, with no shared mutable state.
    int ticket = group_idx;
    int ticket_round = 0;

    // Loop over experts, in LPT order
    int active_rank = 0;
    for (int rank = 0; rank < num_experts; ++rank)
    {
        const int expert_idx = lpt_order[rank];
        // Token span for this expert. `expert_offset` is the exclusive prefix sum of the counts in
        // EXPERT-INDEX order (moe_offset_k), so it has to be indexed by expert_idx: this loop no
        // longer walks experts in index order, and the running `end` the scan-order loop used to
        // accumulate would gather each expert's tokens from the wrong span.
        const int64_t start = expert_offset[expert_idx];
        const int token_count = (int) expert_count[expert_idx];

        // Skip if no tokens or too many tokens for fused kernel (batch is handled by reconstruct path outside kernel)
        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;

        // Skip if expert is claimed by a different group. active_rank numbers the experts that
        // survive the two skips above, so it is this expert's rank in LPT order among the ACTIVE
        // experts - the ticket space the schedule above is defined over.
        if (active_rank++ != ticket) continue;

        // EXL3 weights for g, u, d
        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx];
        const half* exp_gate_suh = gate_suh[expert_idx];
        const half* exp_gate_svh = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis = up_trellis[expert_idx];
        const half* exp_up_suh = up_suh[expert_idx];
        const half* exp_up_svh = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx];
        const half* exp_down_suh = down_suh[expert_idx];
        const half* exp_down_svh = down_svh[expert_idx];

        // Gather + input hadamard for g, u. Non-gated mode skips the g staging (and the g GEMM
        // below); the activation synthesizes the gate lane from u
        const bool gated = act_function != MOE_ACT_RELU2_NOGATE;
        auto had_gather_gu_in = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + token_idx * hidden_dim + token_off * 128;
                if (gated)
                    had_hf_r_128_inner<true, false>
                    (
                        in_ptr,
                        temp_state_g + 128 * warp_idx,
                        exp_gate_suh + 128 * token_off,
                        0.088388347648f
                    );
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        };

        had_gather_gu_in();

        // g, u GEMM
        auto gemm_up = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    MIN(size_m, M_TILE),    \
                    hidden_dim,         \
                    intermediate_dim,   \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    M_TILE,              \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    ((M_TILE >= 64) ? 2 : MOE_FRAG_STAGES)
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, cb, SHAPE_ARGS, false>(ARGS);
                else switch(K)
                {
                    case 1: exl3_gemm_kernel_inner<1, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 2: exl3_gemm_kernel_inner<2, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 5: exl3_gemm_kernel_inner<5, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 6: exl3_gemm_kernel_inner<6, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 7: exl3_gemm_kernel_inner<7, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 8: exl3_gemm_kernel_inner<8, false, cb, SHAPE_ARGS, false>(ARGS); break;
                };
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr += M_TILE * hidden_dim;
                out_addr += M_TILE * intermediate_dim;
                size_m -= M_TILE;
            }
        };

        if (gated)
            gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate);
        gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for g, u + activation+gate + input hadamard for d
        auto had_guad = [&]()
        {
            const int warps_per_token = intermediate_dim / 128;
            const int total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner
                (
                    temp_intermediate_g + 128 * warp_idx,
                    temp_intermediate_u + 128 * warp_idx,
                    temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off,
                    exp_up_svh + 128 * token_off,
                    exp_down_suh + 128 * token_off,
                    0.088388347648f,
                    act_limit,
                    act_function
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        };

        had_guad();

        // d GEMM
        auto gemm_down = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    MIN(size_m, M_TILE),    \
                    intermediate_dim,   \
                    hidden_dim,         \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    M_TILE,              \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    ((M_TILE >= 64) ? 2 : MOE_FRAG_STAGES)
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, cb, SHAPE_ARGS, false>(ARGS);
                else switch(K)
                {
                    case 1: exl3_gemm_kernel_inner<1, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 2: exl3_gemm_kernel_inner<2, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 5: exl3_gemm_kernel_inner<5, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 6: exl3_gemm_kernel_inner<6, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 7: exl3_gemm_kernel_inner<7, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 8: exl3_gemm_kernel_inner<8, false, cb, SHAPE_ARGS, false>(ARGS); break;
                };
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr += M_TILE * intermediate_dim;
                out_addr += M_TILE * hidden_dim;
                size_m -= M_TILE;
            }
        };

        gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for d + scatter add
        auto had_d_out = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            const half* weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                half weight = weights[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                long long* out_ptr = output_state + token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_d_inner
                (
                    temp_state_g + 128 * warp_idx,
                    out_ptr,
                    exp_down_svh + 128 * token_off,
                    0.088388347648f * __half2float(weight)
                );
            }
        };

        had_d_out();

        // Advance to this group's next ticket, then cross the end-of-expert barrier, which is what
        // protects the temp buffers for reuse.
        //
        // Every block of a group computes the same value from the same round count and group_idx,
        // and the barrier keeps them in lockstep, so nothing has to be published. The previous form
        // did have to publish: atomicAdd had to be issued by a single thread, so the result lived in
        // one block's registers until it was written to sched[2 + group_idx] for the others. Dropping
        // the publish while dropping the atomic left the group's other blocks holding a stale ticket
        // forever, which hangs the launch rather than merely misordering it.
        ++ticket_round;
        ticket = num_groups * ticket_round + group_idx;
        group_barrier(group_idx, group_size, barrier_counters_sense);
    }

}
