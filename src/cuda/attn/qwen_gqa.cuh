#pragma once
// Dense causal GQA attention for Qwen3.8-Flash-Next (new for this port; the GLM kernels are MLA,
// absorbed-latent and NoPE, so nothing there applies).
//
// First correct version, deliberately not flash: one warp per (row, q head) with an online softmax
// over the KV cache. exllamav3 only switches to the sparse QSA path above a sequence threshold, so
// dense is the *reference* behaviour at short to moderate depth and a valid first target; a blocked
// or sparse variant is an optimization to make once parity is proven.
//
// Shapes: q [n, n_q_heads, head_dim] fp16, k/v [Tmax, n_kv_heads, head_dim] fp16 (rope already
// applied to q and k), out [n, n_q_heads, head_dim] fp16. Row r is causal at absolute position
// pos0 + r and attends to keys [0, pos0 + r] inclusive. GQA grouping is contiguous: q head h uses
// kv head h / (n_q_heads / n_kv_heads). sm_scale = 1/sqrt(head_dim).

#include "../cuda_shim.hpp"
namespace helios { namespace attn {

void gqa_dense_f16
(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int n,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int pos0,
    float scale,
    Stream s = 0
);

// Tensor-core (mma.m16n8k16) flash-attention prefill. Same contract as gqa_dense_f16 above - q/k/v
// fp16, rope already applied, bottom-right causal - but one block owns a 64-row tile of queries and
// the softmax is online in fp32 registers, so the KV cache is streamed once per 64 rows instead of
// once per row. Exported separately because gqa_dense_f16 only dispatches to it for prefill-sized
// launches (HELIOS_ATTN_MMA=1, n >= 32), and the parity test needs it at shapes the engine never
// runs. n <= 0 or head_dim != 256 is a no-op here, as in gqa_dense_f16.
void gqa_dense_f16_mma
(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int n,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int pos0,
    float scale,
    Stream s = 0
);
// Split-KV decode attention, used when n * n_q_heads alone cannot fill the GPU.
//
// gqa_dense_f16 launches one block per (row, q_head) and splits the key range across 16 warps
// *within* that block. At prefill (n = 256) that is 6144 blocks and fine. At decode (n = 1) it is
// 24 blocks on an 82-SM part: 71% of the GPU idle while the kernel streams the whole KV cache,
// which measured 11.1 ms per token at 15k context against a ~0.53 ms bandwidth bound.
//
// This one warp per (row, kv head, kv chunk), carrying all `n_q_heads / n_kv_heads` query heads
// that share the kv head, and then combines the per-chunk (m, l, o) triples with the same
// arithmetic and the same order as the shared-memory combine it replaces. The chunk count is
// chosen at launch from the SM count and the key range instead of being a constant, so the grid
// scales with context; see gqa_split_kv_kernel's header for the two defects this removes and for
// what it does and does not change numerically.
//
// partial layout: [n * n_q_heads][split][2 + 256] floats (m, l, then 8 values per lane x 32),
// with `split` the chunk count the launcher picks for the current key range.
//
// `row_pos` and `kv_slot_stride` are what make a BATCHED decode possible: the rows of one call do
// not have to be consecutive positions in one KV region. Pass row_pos = nullptr and
// kv_slot_stride = 0 for the single-sequence case, where row r is at pos0 + r over one shared
// cache, and that is bit-for-bit the behaviour this kernel always had. With them set, row r is at
// row_pos[r] over the cache starting kv_slot_stride*row elements in - which is two sequences
// decoded together, each reading its own slot partition, rather than two tokens of one sequence.
void gqa_dense_split_kv_decode
(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* partial,
    int n,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int pos0,
    float scale,
    Stream s = 0,
    const int* row_pos = nullptr,
    size_t kv_slot_stride = 0
);

// Bytes gqa_dense_split_kv_decode needs for its partial buffer for up to `n` rows, at any position
// up to `ctx_cap`. The chunk count grows with the key range, so the buffer has to be sized from the
// context capacity rather than from the current position; sizing it from a capacity is also what
// keeps it smaller than the old fixed-16 buffer, because the chunk count falls as `n` grows.
size_t gqa_split_kv_bytes(int n, int n_q_heads, int n_kv_heads, int head_dim, int ctx_cap);

}} // namespace helios::attn
