#pragma once
// PLE (per-layer embedding) for Qwen3.8-Flash-Next: host-side n-gram hashing, the device port of
// exllamav3's ngram_dequant kernel, and the layer that injects the result into the four
// hyper-connection streams ahead of layer 1.
//
// The hashing runs on the HOST next to the token ids (the reference keeps the id history on the CPU
// so it never round-trips through the device), producing 16 ids per token. Table rows are then
// gathered and decoded on the device: each row is 1 + ROW_DIM*K/16 int16 words whose first word is
// an fp16 row scale and whose remainder is an exl3 trellis ring over the mul1 codebook.
//
// Layer sequence (ple.py forward_streams / ple.cu, all verified against the reference):
//   key   = norm_key(key_proj(emb))        grouped RMS, groups=4, bias 1.0, fp32
//   query = norm_query(streams)            grouped RMS, groups=4, bias 1.0, fp32
//   gate  = row-wise dot(query, key)       fp32, per stream
//   gated = sigmoid(sign(x)*sqrt(max(|x|,1e-6))) * value,  x = gate_scale * gate, scale = 1/sqrt(2560)
//   normed= norm_conv(gated)               grouped RMS -> fp16 [n, 10240]
//   conv  = silu(dilated depthwise causal conv(normed))   dilation 3, 4 taps, 9-column state
//   x    += gated + conv                   elementwise over all four streams, fp32
//
// The conv is new code: ple.cu calls torch's conv1d with dilation 3, so unlike the n-gram decode
// there is no CUDA kernel upstream to port.

#include "core/model.hpp"
#include "cuda/cuda_shim.hpp"

namespace helios {

// history[n_hist] holds the ids the table is hashed over (ctx carried ids followed by this chunk's
// ids). Writes (n_hist - n_ctx) * (ngram_size-1) * heads_per_ngram ids, row-major per token.
void ple_ngram_ids(const int64_t* history, int n_hist, int n_ctx, int64_t eos, int ngram_size,
                   int heads_per_ngram, const int64_t* multipliers, const int64_t* vocab_sizes,
                   const int64_t* offsets, int64_t* out_ids);

// Decode U table rows into fp16 [U, ROW_DIM]. heads[u] selects the bias row. K = 4, words = 41 here.
void ngram_dequant_f16(const void* packed, const int* heads, const half* bias, half* out, int U,
                       int K, int words, cudaStream_t s);

// gated[r,h,d] = sigmoid(sign(x)*sqrt(max(|x|,1e-6))) * value[r,d], x = gate_scale * gate[r,h].
// In place over `gated` (it holds the query/key dot on entry). New: no upstream kernel.
void ple_gate(const float* gate, const void* value, float* gated, float gate_scale, int rows,
              int heads, int dim, cudaStream_t s);

// Dilated depthwise causal conv with a 9-column state, then SiLU, fp16 in/out.
// conv_state is [channels, 9] fp16 and is updated to the last 9 columns of [state | x].
void ple_dilated_conv(const void* x, void* conv_state, const half* w, void* out, int n, int ch,
                      int dilation, cudaStream_t s);

struct PleScratch {
  int max_n = 0, ch = 0, dim = 0;
  void* key16 = nullptr;      // [n, ch] fp16 (key_proj output, pre-norm)
  void* value16 = nullptr;    // [n, dim] fp16 (value_proj output)
  void* key = nullptr;        // [n, 4, dim] fp32 (grouped-norm output)
  void* query = nullptr;      // [n, 4, dim] fp32
  void* gate = nullptr;       // [n, 4] fp32
  void* gated = nullptr;      // [n, 4, dim] fp32
  void* normed = nullptr;     // [n, ch] fp16
  void* conv_out = nullptr;   // [n, ch] fp16
  void* rows = nullptr;       // [max_rows, 41] int16 staged table rows (device)
  void* row_ids = nullptr;    // [max_rows] int32 absolute row ids
  void* decoded = nullptr;    // [max_rows, ROW_DIM] fp16
  void* head_ids = nullptr;   // [max_rows] int32 head index per row
  void* bias_dev = nullptr;   // [num_heads, ROW_DIM] fp16 copy of head_bias
  int* row_ids_host = nullptr;
  int* head_ids_host = nullptr;
  int64_t* hash_host = nullptr;
  // Shard-bucketed row gather. The n-gram table is 128 mmap'd shards, so a chunk's 82-byte rows
  // are grouped by shard on the host: the pinned staging buffer holds them contiguously per
  // shard, the per-shard spans become one H2D each, and row_stage/row_slot put them back into
  // caller row order on the device. That is ~125 transfers per 1024-token chunk instead of one
  // per row (16,384), which were pageable and sat at the very front of the stream.
  //
  // Pageable -> pinned is a real change in ownership: a pageable H2D is staged by the driver
  // before the call returns, so the source can be rewritten immediately, but a pinned one DMAs
  // out of the buffer LATER. Prefill runs a whole prompt's chunks back to back with no sync
  // between them (Runner::prefill), so the host is free to be several chunks ahead - and
  // overwriting a slot the copy has not read yet would corrupt the rows silently. Hence the ring:
  // a slot is only reused once its own event says the copies out of it have landed.
  static constexpr int kGatherSlots = 3;
  void* row_stage = nullptr;    // [max_rows, 41] int16 device staging (shard order)
  void* row_slot = nullptr;     // [max_rows] int32 device: staged slot -> caller row
  char* row_pin[kGatherSlots] = {};        // pinned [max_rows, 41] int16 gather staging
  int* row_slot_host[kGatherSlots] = {};   // pinned [max_rows] int32: staged slot -> caller row
  cudaEvent_t row_pin_ev[kGatherSlots] = {};  // copies out of this slot have landed
  int row_pin_cur = 0;
  int* row_shard_host = nullptr;// [max_rows] int32 shard id per caller row
  int* shard_cnt = nullptr;     // [ngram_shards] rows per shard, this call
  int* shard_start = nullptr;   // [ngram_shards] first staged slot of each shard
  int* shard_cur = nullptr;     // [ngram_shards] write cursor into the staging buffer
  // Speculative-rollback snapshots. The dilated conv is a running 9-column state, so a partial
  // accept has to restore it and re-run the accepted prefix - exactly what the GDN recurrence
  // needs and what this layer was silently missing. conv_state_snap is taken BEFORE the batch
  // advances it, conv_in_snap holds the pre-conv activations of the batch so the replay has the
  // same inputs the original forward did.
  void* conv_state_snap = nullptr;   // [ch, 9] fp16
  void* conv_in_snap = nullptr;      // [max_n, ch] fp16
  bool bias_ready = false;
};

bool ple_scratch_init(PleScratch& sc, const Config& cfg, int max_n, int device);
void ple_scratch_free(PleScratch& sc);

// streams: fp32 [n, 4, dim] in/out. embed: fp16 [n, hidden] (the token embedding rows for this
// chunk). history/n_hist/n_ctx describe the host id history; conv_state is the layer's [ch,9] fp16.
// `capture` snapshots the conv state and this chunk's conv input for a later partial-accept
// replay (Runner::ple_replay). It is false everywhere except a speculative verify batch.
void ple_layer(const PleWeights& w, const Config& cfg, PleScratch& sc, const half* embed,
               float* streams, const int64_t* history, int n_hist, int n_ctx, void* conv_state,
               int n, cudaStream_t s, bool capture = false);

}  // namespace helios
