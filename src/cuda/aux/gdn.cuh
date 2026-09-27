#pragma once
// Ported from exllamav3 exllamav3_ext/gdn.cuh: decode-side GDN / KDA entry points,
// raw-pointer signatures, explicit stream. The CUDA-graph variants and all mamba2_*
// paths were dropped.

#include "../cuda_shim.hpp"
#include <cstdint>
#include <vector>

namespace helios { namespace aux {

using bfloat16 = __nv_bfloat16;

void gated_delta_net_fused_op
(
    const float* mixed_qkvz,
    const float* mixed_ba,
    const bfloat16* dt_bias,
    const bfloat16* a_log,
    bfloat16* mixed_qkv,
    bfloat16* z,
    bfloat16* beta,
    float* g,
    int B, int S,
    size_t num_k_heads,
    size_t num_v_heads,
    size_t k_head_dim,
    size_t v_head_dim,
    const float beta_scale,
    Stream stream
);

void gated_delta_net_fused_op_2
(
    const float* b,
    const float* a,
    const bfloat16* dt_bias,
    const void* a_log,
    bool a_log_fp32,
    bfloat16* beta,
    float* g,
    int B, int S, int H,
    const float beta_scale,
    Stream stream
);

void cuda_recurrent_gated_delta_rule
(
    const bfloat16* mixed_qkv,
    const float* g,
    const bfloat16* beta,
    float* recurrent_state,
    bfloat16* core_attn_out,
    int bsz,
    int seqlen,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    int history_stride,
    const int* slots,
    bool channelwise,
    bool history,
    Stream stream
);

// Chunked (WY-representation) gated delta rule - the parallel-scan form of the same recurrence,
// for prefill-sized token counts. Sends the first (seqlen/64)*64 tokens through the three chunked
// stages and the remaining seqlen%64 tail through cuda_recurrent_gated_delta_rule, so every token
// is produced by exactly one path and the tail keeps the serial kernel bit for bit.
// `workspace` must be 256-byte aligned and hold at least
// gdn_chunked_workspace_bytes(num_v_heads, num_k_heads, seqlen) bytes; workspace_tokens caps how
// many leading tokens the chunked part may take, the rest going to the serial kernel.
// Returns false WITHOUT touching any output when the shape is unsupported or the workspace is too
// small, so the caller can fall back to the serial path wholesale. Once it starts, a CUDA failure
// aborts rather than returning, because the recurrent state would be left half-advanced.
bool cuda_chunked_gated_delta_rule
(
    const bfloat16* mixed_qkv,
    const float* g,
    const bfloat16* beta,
    float* recurrent_state,
    bfloat16* core_attn_out,
    int bsz,
    int seqlen,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    void* workspace,
    int workspace_tokens,
    Stream stream
);

// Workspace size in bytes for the chunked path (rounded down to a 64-token multiple of tokens).
size_t gdn_chunked_workspace_bytes(int num_v_heads, int num_k_heads, int tokens);

// The three chunked stages on their own: L2-normalize q/k, build W/U, run the inter-chunk state
// scan, then the per-chunk output. `tokens` must be a positive multiple of 64 that the workspace
// can hold. Used by cuda_chunked_gated_delta_rule; exposed so the parity test can drive the stages
// without the dispatch.
void gdn_chunked_stages
(
    const bfloat16* mixed_qkv,
    const float* g,
    const bfloat16* beta,
    float* recurrent_state,
    bfloat16* core_attn_out,
    int tokens,
    int num_k_heads,
    int num_v_heads,
    void* workspace,
    Stream stream
);

void cuda_causal_conv1d_update
(
    const bfloat16* x,
    bfloat16* conv_state,
    const int* slots,
    const bfloat16* weight,
    const bfloat16* bias,
    bfloat16* out,
    int bsz,
    int dim,
    int seqlen,
    int state_size,
    int K,
    bool activation,
    bool history,
    Stream stream
);

void gated_delta_net_fused_op_3
(
    const float* qkv,
    const float* ba,
    const bfloat16* dt_bias,
    const void* a_log,
    bool a_log_fp32,
    bfloat16* mixed_qkv,
    bfloat16* beta,
    float* g,
    int B, int S, int F, int H,
    const float beta_scale,
    Stream stream
);

void gdn_ba_gemv
(
    const half* x,
    const half* w_t,
    const half* bias,
    float* y,
    int rows,
    int k,
    int n,
    Stream stream
);

void gdn_lowrank_gemv_f
(
    const float* x,
    const half* w_t,
    float* y,
    int rows,
    int k,
    int n,
    Stream stream
);

void kda_gate_op
(
    const float* qkv,
    const float* b,
    const float* f,
    const bfloat16* dt_bias,
    const void* a_log,
    bool a_log_fp32,
    bfloat16* mixed_qkv,
    bfloat16* beta,
    float* g,
    int B, int S, int F, int H, int Dk,
    const float lower_bound,
    const float beta_scale,
    Stream stream
);

struct ConvRewindJob
{
    uintptr_t src;
    uintptr_t dst;
    int dim;
    int cdim;
    int stride;

    ConvRewindJob() = default;
    ConvRewindJob(uintptr_t _src, uintptr_t _dst, int _dim, int _cdim, int _stride) :
        src(_src), dst(_dst), dim(_dim), cdim(_cdim), stride(_stride) {}
};

struct StateRewindJob
{
    uintptr_t src;
    uintptr_t dst;
    int64_t num_elements;

    StateRewindJob() = default;
    StateRewindJob(uintptr_t _src, uintptr_t _dst, int64_t _num_elements) :
        src(_src), dst(_dst), num_elements(_num_elements) {}
};

void batched_conv_rewind(std::vector<ConvRewindJob> const& jobs, Stream stream);
void batched_state_rewind(std::vector<StateRewindJob> const& jobs, Stream stream);

}} // namespace helios::aux