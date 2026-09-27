#pragma once
// QSA (Qwen sparse attention) indexer - source-verified against
// ~/projects/exllamav3/exllamav3/modules/qsa_indexer.py
//
// Semantics, transcribed from the reference module's docstring and code:
//
//   project:  index_qk_proj maps hidden -> (n_heads + kv_heads) * head_dim. The first n_heads*head_dim
//             are the query heads, the last head_dim is ONE raw key. q is RMS-normed (constant_bias 1)
//             then partial-roped at the query position. raw_k is cached UNNORMED and UNROPED.
//   pool:     every COMPLETE compress_ratio (4) block of raw keys is mean-pooled in fp32, cast to fp16,
//             RMS-normed (constant_bias 1), then partial-roped at the block's START position. A block is
//             therefore final the moment its last key arrives, which is what makes the pool cacheable.
//   score:    block_scores = relu(q . k) summed over the index heads, times 1/sqrt(head_dim).
//   select:   each query keeps the top (budget / compress_ratio) = 512 blocks, plus always the
//             incomplete tail block, ANDed with causality. Selection is at 4-token block granularity,
//             so it selects 2048 of every context's tokens once the context exceeds the budget.
//
// Unlike the MLA/GLM port, this checkpoint's indexer is 4 heads x 128 dim with a single raw key head,
// and the main attention is plain GQA - the sparse kernels here select blocks and hand a token mask to a
// GQA kernel, not an MLA decode.

#include "../cuda_shim.hpp"

namespace helios { namespace attn {

// Per-layer persistent state for the indexer. Sized once at context capacity.
struct QsaLayerState {
  void* pooled_k = nullptr;    // fp16 [max_blocks, head_dim], completed+pooled blocks only
  void* tail_raw = nullptr;    // fp16 [compress_ratio, head_dim], keys of the block still filling
  int*   block_pos = nullptr;  // int32 [max_blocks], start position of each pooled block
  int    n_blocks = 0;         // completed blocks
  int    n_tail = 0;           // keys accumulated in the current (incomplete) block
  int    max_blocks = 0;
};

// Scratch for one indexer invocation at a given batch size.
struct QsaScratch {
  void* had = nullptr;         // fp16 [n * hidden], Hadamard scratch for the qk projection gemm
  void* qk = nullptr;          // fp16 [n * (n_heads + kv_heads) * head_dim], raw projection output
  void* q = nullptr;           // fp16 [n, n_heads, head_dim], normed + roped
  void* raw_k = nullptr;       // fp16 [n, head_dim], the new keys' unnormed/unroped values
  void* sel = nullptr;         // int32 [n, n_sel] selected block ids, ascending, padded with -1
  void* score = nullptr;       // fp32 [n, n_blocks]
  int*   pos = nullptr;        // int32 [n] absolute positions
  int    n = 0;
  int    n_heads = 0, head_dim = 0;   // indexer shape, carried so kernels need not hardcode it
  int    n_sel = 0;
};

// Bytes QsaScratch needs for this shape. n_blocks is the *current* completed-block count.
size_t qsa_scratch_bytes(int n, int n_blocks, int n_heads, int head_dim, int n_sel,
                         int hidden = 0);

// Per-layer bytes at a given context capacity.
size_t qsa_layer_bytes(int max_ctx, int head_dim, int compress_ratio);

// Project x for the indexer: runs the fused qk projection, norm+rope on q, and leaves raw_k in
// scratch.raw_k. Does NOT touch the pooled state.
void qsa_project
(
    const void* x,                    // fp16 [n, hidden]
    int n,
    const void* qk_w, const void* qk_suh, const void* qk_svh, int qk_mul1, int qk_bits, int hidden,
    const void* q_ln_w, const void* k_ln_w,   // fp16 [head_dim]
    int n_heads, int head_dim, int rotary_dim, float rope_theta, float rms_eps,
    QsaScratch& sc, Stream s = 0
);

// Fold the new raw keys into the layer's pooled blocks, completing as many 4-key blocks as possible.
void qsa_pool_update
(
    QsaLayerState& st,
    const half* new_raw_k,            // [n, head_dim]
    int n,
    const void* k_ln_w,               // fp16 [head_dim]
    int head_dim, int rotary_dim, int compress_ratio, float rope_theta, float rms_eps,
    Stream s = 0
);

// Score and select for the current token. n == 1 (decode) is the supported path.
// Writes sc.sel as n_sel ascending block ids, padded with -1, and sc.score as the raw block scores.
void qsa_select
(
    const QsaLayerState& st,
    QsaScratch& sc,
    int n_heads, int head_dim,
    int n_sel, float scale,
    Stream s = 0
);

// Expand a query's selected blocks into the exact token indices it may attend to (on device - the
// selection never leaves the GPU, so no per-layer D2H copy or sync). Writes tok_idx / tok_block / tok_off
// and the count into n_out. Capacity for all rows must be n_rows * (n_sel * compress_ratio + compress_ratio).
void qsa_expand
(
    const QsaLayerState& st, const QsaScratch& sc,
    int n_rows, int n_sel, int compress_ratio, int pos, int cap,
    void* tok_idx, void* tok_block, void* tok_off,
    Stream s = 0
);

// Sparse GQA decode attention over an expanded token set.
//
// This is the dense split-KV kernel with the key range replaced by the gathered token indices: instead of
// walking keys [0, pos0+row], each chunk walks tok_idx[start..start+chunk). The per-lane dot, online
// softmax update and combine are the SAME expressions as gqa_dense_split_kernel, and each gathered token
// is visited exactly once, so this is numerically the same computation over a subset - not an
// approximation of it.
//
// tok_off[q * cap] holds the count for query q (written by qsa_expand).
void gqa_sparse_decode
(
    const void* q,          // fp16 [n, n_q_heads, head_dim]
    const void* k,          // fp16 [T, n_kv_heads, head_dim]
    const void* v,          // fp16 [T, n_kv_heads, head_dim]
    const int* tok_idx,     // int32 [n * cap] gathered token positions
    const int* tok_off,     // int32 [n * cap], slot 0 of each row = count
    int cap,
    void* out,              // fp16 [n, n_q_heads, head_dim]
    float* part,            // fp32 [n, n_q_heads, chunks, 2 + head_dim] scratch
    float* grp,             // fp32 [n, n_q_heads, 64, 2 + head_dim] combine scratch
    int n, int n_q_heads, int n_kv_heads, int head_dim,
    float scale,
    Stream s = 0
);

}} // namespace helios::attn
