#pragma once
// Model registry for Qwen3.8-Flash-Next-exl3 (arch `qwen4_exp`): config parsing, tensor placement,
// EXL3 group structs and the loader.
//
// Unlike the GLM engine, this model's experts total 31.31 GB, so they are **fully resident** - split
// *within each layer* across the two cards. Every layer therefore runs attention on GPU0 and its
// experts on both, with one activation round trip per layer, and there is no slot pool, no census,
// no PCIe expert streaming and no RAM arena for experts.
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/safetensors.hpp"

namespace helios {

struct Config {
  int n_layers = 48;
  int hidden = 2560;

  // ---- gated delta net (36 layers) ----
  int gdn_k_heads = 16, gdn_v_heads = 48, gdn_k_dim = 128, gdn_v_dim = 128;
  int gdn_conv_k = 4;
  int gdn_qkv_out = 10240;      // 16*128 (q) + 16*128 (k) + 48*128 (v)
  int gdn_z_out = 6144;         // 48*128, the output gate
  float gdn_decay_scale = 1.0f; // A_log / dt_bias are per v-head (48 values)

  // ---- full attention (12 layers) ----
  int attn_heads = 24, attn_kv_heads = 2, attn_head_dim = 256;
  int attn_q_out = 12288;       // 24*256 q interleaved with 24*256 gate
  int idx_heads = 4, idx_kv_heads = 1, idx_dim = 128;
  int idx_budget = 2048, idx_compress = 4, idx_qk_out = 640;  // 4*128 q + 1*128 k, fused
  float rope_theta = 10000000.0f;
  float partial_rotary = 0.25f;
  int mrope_section[3] = {11, 11, 10};

  // ---- MoE (all 48 layers) ----
  int n_expert = 512, topk = 10, moe_inter = 640, shared_inter = 640;
  bool router_bias = false;     // this checkpoint has none (GLM did)
  float routed_scale = 1.0f;

  // ---- hyper-connections (gated residual, low rank) ----
  int hc_mult = 4, hc_rank = 320;
  int hc_dim = 10240;           // hc_mult * hidden
  // ---- PLE (n-gram embedding) ----
  int ple_layer = 1;            // 0-based layer index the PLE precedes (config: ple_layer_ids [2], 1-based)
  int ple_dim = 2560, ngram_size = 3, heads_per_ngram = 8;
  int ngram_shards = 128;
  int64_t ngram_rows_per_shard = 2500012;
  int ngram_words_per_row = 41;
  int64_t ngram_eos = 248044; // eos token id, for the PLE's segment-aware hashing
  int ngram_heads = 16;      // (ngram_size - 1) * heads_per_ngram
  int ngram_row_dim = 160;   // ple_dim / ngram_heads
  int ngram_K = 4;           // trellis bits per element (41 i16 words = 656 bits ~ 160 at 4 bpw)

  float rms_eps = 1e-6f;
  int vocab = 248320;

  std::vector<bool> full_attn;  // per layer: false = GDN, true = full attention
  bool has_mtp = true;

  // Expert split: how many of the 512 experts live on GPU1; the rest stay on GPU0 with attention.
  // 352 = 11/16 of them, which puts ~21.1 GB on GPU1 (24 GB card, no KV there) and leaves GPU0 room
  // for the dense stack, the embed, the KV cache and this card's own 160 experts.
  int experts_gpu1 = 352;
  int experts_gpu0() const { return n_expert - experts_gpu1; }   // pipeline OFF only

  // ---- two-stage layer pipeline (HELIOS_PIPELINE, DEFAULT OFF) ----
  //
  // OFF, every accessor below returns exactly what the shipped engine uses, so this block cannot
  // change the default path: the dense stack, the embedding and the KV cache are on card 0, and the
  // routed experts are split WITHIN each layer (352 on card 1, 160 on card 0).
  //
  // ON, placement becomes per LAYER: layers [0, split) on card 0, [split, n_layers) on card 1, and
  // each card holds ALL 512 experts - but only for its own half of the layers, so the two expert
  // arenas are the same total bytes as today, just re-cut. This flag changes WHERE weights and
  // scratch live; it does NOT change the forward schedule. See Loader::load for the per-tensor
  // routing and Runner::init for the scratch/state side (which refuses to run a pipeline-mode
  // forward until the scheduler stage lands).
  bool pipeline = false;
  int split = 0;                        // first layer index owned by card 1
  int layer_dev(int l) const { return (!pipeline || l < split) ? 0 : 1; }
  // The draft head follows the LAST layer's card: it reads the trunk's final mixer output and has
  // its own experts, so it belongs to whichever card produces the end of the stack.
  int mtp_dev() const { return pipeline ? layer_dev(n_layers - 1) : 0; }
  // Expert slabs per layer resident on `card`, in that card's own (local) expert index space.
  int experts_per_card(int card) const {
    return pipeline ? n_expert : (card == 1 ? experts_gpu1 : n_expert - experts_gpu1);
  }
  // First GLOBAL expert id resident on `card`; a router id e has local index e - this.
  int expert_base(int card) const { return pipeline ? 0 : (card == 1 ? 0 : experts_gpu1); }
  // Layers whose expert slabs live on `card` (the MTP layer counts as a layer of the last card).
  int expert_layers(int card) const {
    if (!pipeline) return n_layers + (has_mtp ? 1 : 0);
    return card == 0 ? split : n_layers - split + (has_mtp ? 1 : 0);
  }
  // Index of a (layer | MTP) slot within `card`'s own arena. Compact, so each arena is dense.
  int expert_slot(int slot, int card) const { return (pipeline && card == 1) ? slot - split : slot; }
};

// One EXL3 quantized matrix: K = trellis.d2 / 16 bits per element.
struct Group {
  void* trellis = nullptr;        // device i16 [d0,d1,K/16]
  const half* suh = nullptr;      // device f16 [in]
  const half* svh = nullptr;      // device f16 [out]
  int mul1 = 0;                   // codebook multiplier word (host value)
  int K = 0, out = 0, in = 0;
  void set_dims(const std::vector<int64_t>& trellis_shape);
};

// Gated delta net: qkv / z / out are quantized, the rest is small and dense.
struct GdnWeights {
  Group qkv;                      // 2560 -> 10240
  Group z;                        // 2560 -> 6144
  Group out;                      // 6144 -> 2560
  const void* conv = nullptr;     // [10240,4] bf16 depthwise, kept as stored: the conv
                                  // kernel takes a bfloat16 weight
  const half* a_proj = nullptr;   // [48,2560] f16
  const half* b_proj = nullptr;   // [48,2560] f16
  const void* A_log = nullptr;    // [48] bf16, kept as stored (the GDN prologue takes bfloat16*)
  const void* dt_bias = nullptr;  // [48] bf16
  const void* norm = nullptr;     // [128] bf16, kept as stored: the reference loads it with
                                  // allow_bf16 and the kernel takes a bfloat16 weight
};

// QSA sparse full attention: GQA + fused indexer.
struct AttnWeights {
  Group q, k, v, o;               // q: 2560->12288, k/v: 2560->512, o: 6144->2560
  const half* q_norm = nullptr;   // [256] f16 (file bf16), applied with constant_bias 1.0
  const half* k_norm = nullptr;   // [256] f16
  Group idx_qk;                   // 2560 -> 640 (4 indexer heads x 128 q, 1 x 128 k)
  const half* idx_q_ln = nullptr; // [128] f16
  const half* idx_k_ln = nullptr; // [128] f16
};

// MoE: a router, a sigmoid-gated shared expert, and the layer's slice of the resident experts.
struct MoeWeights {
  const half* router = nullptr;      // [512,2560] f16
  Group shared[3];                   // gate, up, down (quantized 4-bit)
  const half* shared_gate = nullptr; // [1,2560] f16 (sigmoid gate on the shared output)
  int expert_base = 0;               // index of this layer's first expert in the expert arrays
};

// Gated-residual hyper-connection site: norm, low-rank mix, per-stream injection.
struct HcWeights {
  const half* norm = nullptr;     // [10240] f16
  const half* down = nullptr;     // [320,10240] f16
  const half* up = nullptr;       // [10240,320] f16
  const half* inject = nullptr;   // [4,10240] f16 (absent on the final mixer)
};

// PLE: hashed n-gram embeddings injected ahead of one early layer.
struct PleWeights {
  const half* conv = nullptr;        // [10240,4] f16 (file [10240,1,4])
  const half* key_proj = nullptr;    // [10240,2560] f16
  const half* value_proj = nullptr;  // [2560,2560] f16
  const half* norm_conv = nullptr;   // [10240] f16 (file bf16)
  const half* norm_key = nullptr;    // [10240] f16
  const half* norm_query = nullptr;  // [10240] f16
  // The 26.24 GB table is mapped from ngram_embedding.safetensors, not copied. That file holds
  // 128 trellis shards plus head_bias (F16 [16,160]) and three int64 aux arrays, which drive both
  // the host-side n-gram hashing and the per-head row decode.
  const char* table = nullptr;       // shard 0's rows (host, mmap'd)
  const char* shards[128] = {};      // per-shard row arrays; shard = row_id / rows_per_shard
  const half* head_bias = nullptr;   // [num_heads, row_dim] fp16, added after the trellis decode
  const int64_t* head_offsets = nullptr;      // [num_heads]
  const int64_t* head_vocab_sizes = nullptr;  // [num_heads]
  const int64_t* layer_multipliers = nullptr; // [ngram_size]
};

struct Layer {
  int index = 0;
  bool full = false;                 // full attention vs GDN
  int attn_ord = -1;                 // ordinal among full-attention layers (KV/indexer indexing)
  int gdn_ord = -1;                  // ordinal among GDN layers (state indexing)
  GdnWeights gdn;
  AttnWeights attn;
  MoeWeights moe;
  HcWeights hc_attn, hc_mlp;
  PleWeights ple;                    // only used when index == cfg.ple_layer
  bool has_ple = false;
};

struct MTPWeights {
  Group fc_embedding, fc_hidden;     // 2560 -> 2560 each
  Layer layer;                       // the single draft layer (full attention, own MoE)
  HcWeights mixer;                   // mtp.hyper_connection_mixer (external patch file)
  // The input combine's two norms. Both follow the (w + 1) convention: pre_fc_norm_hidden is the
  // per-stream trunk tap norm (RMSNorm over the hidden axis, independently per stream) and
  // pre_fc_norm_embedding is the token-embedding norm (constant_bias = 1.0 upstream).
  const half* pre_fc_norm_hidden = nullptr;      // [hc_mult * hidden]
  const half* pre_fc_norm_embedding = nullptr;   // [hidden]
};

struct Model {
  Config cfg;
  ShardSet shards;                   // the 5 model shards
  ShardSet extra;                    // ngram_embedding.safetensors + mtp patch (headers + mmap)

  // ---- dense stack. With the flag OFF all of it is on card 0 (the shipped engine). With
  // HELIOS_PIPELINE=1 every one of these is on the card that owns its LAYER (Config::layer_dev),
  // except the embedding (layer 0's card) and lm_head (the last layer's card). ----
  const void* embed = nullptr;       // bf16 [vocab,2560] (gather kernel converts)
  Group lm_head;                     // 4-bit [vocab,2560]
  std::vector<Layer> layers;
  MTPWeights mtp;
  HcWeights mixer;                   // model.language_model.hyper_connection_mixer
  // The global (combine-less) hyper-connection mixer is ONE weight set that every layer's collapse
  // uses, and a device pointer is not dereferenceable from the other card. It is therefore
  // DUPLICATED rather than routed: ~13 MB (norm + [320,10240] down + [10240,320] up) is a rounding
  // error next to 29 GB of experts, whereas routing it to the owning card's stream would cost a
  // copy on every forward and a second stream to order against. `mixer1` is null unless
  // HELIOS_PIPELINE is on; with the flag off, every layer reads `mixer` on card 0.
  HcWeights mixer1;                    // second copy of `mixer`, for card 1
  const HcWeights& mixer_for(int dev) const { return dev == 0 ? mixer : mixer1; }

  // ---- experts: fully resident, one contiguous arena per card ----
  // Pipeline OFF: every layer has cfg.experts_gpu0() slabs on GPU0 and cfg.experts_gpu1() on GPU1
  // (the within-layer split). Pipeline ON: each card has all n_expert slabs, but only for its own
  // cfg.expert_layers(card) layers - the same total bytes, re-cut along the layer axis. A slab is
  // addressed as experts{card} + (cfg.expert_slot(layer, card) * cfg.experts_per_card(card)
  // + e_local) * expert_stride; cfg.expert_base(card) maps a global router id to the local one.
  char* experts0 = nullptr;          // GPU0 arena (see above)
  char* experts1 = nullptr;          // GPU1 arena
  size_t expert_stride = 0;          // bytes per expert slab
  int expert_K = 2;                  // trellis bits of the routed experts (per-slab)
  size_t experts0_bytes = 0, experts1_bytes = 0;
  // Byte offsets of the 12 slab pieces inside an expert slab: gate trellis/suh/svh/mul1, then up,
  // then down. The runner needs them to build the nine per-expert pointer tables.
  size_t slab_off[12] = {};

  // Stats
  size_t gpu0_bytes = 0, gpu1_bytes = 0, ram_bytes = 0;

  bool load(const std::string& dir, bool ram_only, bool verbose);
  // Expert slab for (layer, expert) on its owning card, or nullptr if the card is the other one.
  const char* expert_slab(int layer, int expert, int* device) const;
};

// bf16 -> fp16 (and -> fp32 for the two decay tensors and the norms that feed exponentials)
void cvt_bf16_f16(const void* src, half* dst, size_t n);
void cvt_bf16_f32(const void* src, float* dst, size_t n);

}  // namespace helios
