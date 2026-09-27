// Loader vocabulary + config test for Qwen3.8-Flash-Next-exl3.
//
// Runs Model::load in RAM-only mode: no CUDA allocation, no GPU, no weights read - but the loader
// still resolves every tensor name it depends on (T() reports each missing one) and parses
// config.json. Runs in ~2 s and fails on the first name typo, which is the class of bug this port
// is most exposed to with 304k tensors.
#include "core/model.hpp"
#include "engine/ple.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace helios;

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : "/home/neron/models/Qwen3.8-Flash-Next-exl3";
  Model m;
  if (!m.load(dir, /*ram_only=*/true, /*verbose=*/true)) {
    fprintf(stderr, "MODEL LOAD (ram-only) FAILED\n");
    return 1;
  }
  const Config& c = m.cfg;
  printf("CONFIG layers=%d hidden=%d | gdn k=%d v=%d kd=%d vd=%d conv=%d\n", c.n_layers, c.hidden,
         c.gdn_k_heads, c.gdn_v_heads, c.gdn_k_dim, c.gdn_v_dim, c.gdn_conv_k);
  printf("CONFIG attn h=%d kv=%d d=%d q_out=%d | idx heads=%d d=%d budget=%d compress=%d\n",
         c.attn_heads, c.attn_kv_heads, c.attn_head_dim, c.attn_q_out, c.idx_heads, c.idx_dim,
         c.idx_budget, c.idx_compress);
  printf("CONFIG moe experts=%d topk=%d inter=%d shared=%d | hc mult=%d dim=%d rank=%d\n",
         c.n_expert, c.topk, c.moe_inter, c.shared_inter, c.hc_mult, c.hc_dim, c.hc_rank);
  printf("CONFIG ple layer=%d dim=%d heads=%d row_dim=%d | ngram size=%d per=%d shards=%d\n",
         c.ple_layer, c.ple_dim, c.ngram_heads, c.ngram_row_dim, c.ngram_size, c.heads_per_ngram,
         c.ngram_shards);
  printf("CONFIG rope theta=%.0f partial=%.2f | vocab=%d mtp=%d | experts gpu1=%d gpu0=%d\n",
         c.rope_theta, c.partial_rotary, c.vocab, (int)c.has_mtp, c.experts_gpu1,
         c.experts_gpu0());

  int bad = 0;
  auto chk = [&](bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "  CHECK FAILED: %s\n", what); bad++; }
  };
  chk(c.n_layers == 48, "48 layers");
  chk(c.hidden == 2560, "hidden 2560");
  chk(c.attn_heads == 24 && c.attn_kv_heads == 2 && c.attn_head_dim == 256, "GQA 24/2 x 256");
  chk(c.attn_q_out == 12288, "q_proj out 12288 (q interleaved with gate)");
  chk(c.gdn_k_heads == 16 && c.gdn_v_heads == 48, "GDN 16 k-heads / 48 v-heads");
  chk(c.gdn_qkv_out == 10240 && c.gdn_z_out == 6144, "GDN qkv 10240 / z 6144");
  chk(c.n_expert == 512 && c.topk == 10, "MoE 512 experts top-10");
  chk(c.moe_inter == 640 && c.shared_inter == 640, "MoE intermediate 640");
  chk(c.hc_mult == 4 && c.hc_rank == 320 && c.hc_dim == c.hc_mult * c.hidden, "hc 4 streams rank 320");
  chk(c.idx_heads == 4 && c.idx_dim == 128 && c.idx_qk_out == 640, "indexer fused q|k = 640");
  chk(c.ple_layer == 1, "PLE precedes layer index 1 (config ple_layer_ids [2])");
  chk(c.ngram_heads == 16 && c.ngram_row_dim == 160, "ngram 16 heads x 160 dims");
  chk(c.vocab == 248320, "vocab 248320");
  chk(c.has_mtp, "MTP present");
  chk(c.full_attn.size() == 48 && c.full_attn[3] && !c.full_attn[2] && c.full_attn[47],
      "layer_types: every 4th layer is full attention");
  int nfull = 0;
  for (bool f : c.full_attn) nfull += f ? 1 : 0;
  chk(nfull == 12, "12 full-attention layers / 36 GDN");
  chk(m.layers.size() == 48, "48 layers materialised");
  chk(m.layers[1].has_ple, "layer 1 carries the PLE");
  chk(m.layers[3].full && m.layers[3].attn_ord == 0 && m.layers[0].gdn_ord == 0,
      "layer ordinals assigned");
  // The PLE's n-gram hashing needs four aux tensors from ngram_embedding.safetensors. Validate the
  // shape/ordering assumptions the port is built on against the real file rather than trusting them.
  {
    const PleWeights& pl = m.layers[c.ple_layer].ple;
    chk(pl.head_bias != nullptr, "ngram head_bias present");
    chk(pl.head_offsets && pl.head_vocab_sizes && pl.layer_multipliers, "ngram aux arrays present");
    int asc = 0, pos_vocab = 0;
    for (int h = 0; h < c.ngram_heads; h++) {
      if (h && pl.head_offsets[h] > pl.head_offsets[h - 1]) asc++;
      if (pl.head_vocab_sizes[h] > 0) pos_vocab++;
    }
    chk(asc == c.ngram_heads - 1, "head_offsets strictly ascending (the row-to-head search relies on it)");
    chk(pos_vocab == c.ngram_heads, "head_vocab_sizes all positive");
    chk(pl.layer_multipliers[0] != 0, "layer_multipliers loaded");
    printf("PLE ngram: heads=%d row_dim=%d | offsets[0..3]=%lld,%lld,%lld,%lld vocab[0..3]=%lld,%lld,%lld,%lld mult=%lld,%lld,%lld\n",
           c.ngram_heads, c.ngram_row_dim, (long long)pl.head_offsets[0], (long long)pl.head_offsets[1],
           (long long)pl.head_offsets[2], (long long)pl.head_offsets[3],
           (long long)pl.head_vocab_sizes[0], (long long)pl.head_vocab_sizes[1],
           (long long)pl.head_vocab_sizes[2], (long long)pl.head_vocab_sizes[3],
           (long long)pl.layer_multipliers[0], (long long)pl.layer_multipliers[1],
           (long long)pl.layer_multipliers[2]);
    // table size sanity: 128 shards x rows_per_shard rows, matching the mapped file
    printf("PLE ngram: total rows = %lld (128 shards x %lld)\n",
           (long long)(128 * (c.ngram_rows_per_shard)), (long long)c.ngram_rows_per_shard);
  }

  // PLE n-gram hashing: every id must fall inside its head's row range, since the gather indexes the
  // flat table with it. Offsets/vocab handling or a signed modulo bug shows up immediately here.
  {
    const PleWeights& pl = m.layers[c.ple_layer].ple;
    const int nh = (c.ngram_size - 1) * c.heads_per_ngram;      // 16
    const int64_t eos = 248044;                                 // config text_config.eos_token_id
    std::vector<int64_t> hist = {eos, eos, 100, 200, 300, eos, 400, eos, 500, 600};
    const int n_ctx = c.ngram_size - 1;
    std::vector<int64_t> ids((size_t)(hist.size() - n_ctx) * nh, -1);
    ple_ngram_ids(hist.data(), (int)hist.size(), n_ctx, eos, c.ngram_size, c.heads_per_ngram,
                  pl.layer_multipliers, pl.head_vocab_sizes, pl.head_offsets, ids.data());
    int out_of_range = 0, unfilled = 0, distinct = 0;
    for (int t = 0; t < (int)(hist.size() - n_ctx); t++)
      for (int h = 0; h < nh; h++) {
        const int64_t id = ids[(size_t)t * nh + h];
        if (id < 0) { unfilled++; continue; }
        if (id < pl.head_offsets[h] || id >= pl.head_offsets[h] + pl.head_vocab_sizes[h])
          out_of_range++;
        distinct++;
      }
    // determinism: the same history must hash to the same ids
    std::vector<int64_t> ids2(ids.size(), -1);
    ple_ngram_ids(hist.data(), (int)hist.size(), n_ctx, eos, c.ngram_size, c.heads_per_ngram,
                  pl.layer_multipliers, pl.head_vocab_sizes, pl.head_offsets, ids2.data());
    const bool deterministic = (ids == ids2);
    printf("PLE hash: %d ids, %d out of head range, %d unfilled, deterministic=%d\n", distinct,
           out_of_range, unfilled, (int)deterministic);
    chk(out_of_range == 0, "every n-gram id inside its head's row range");
    chk(unfilled == 0, "every position produced ids");
    chk(deterministic, "hashing is deterministic");
    // EOS must actually break the n-gram: inserting one changes the following tokens' ids
    std::vector<int64_t> hist_seg = {eos, eos, 100, eos, 200, 300};
    std::vector<int64_t> ids_seg((size_t)(hist_seg.size() - n_ctx) * nh, -1);
    ple_ngram_ids(hist_seg.data(), (int)hist_seg.size(), n_ctx, eos, c.ngram_size, c.heads_per_ngram,
                  pl.layer_multipliers, pl.head_vocab_sizes, pl.head_offsets, ids_seg.data());
    const bool differs =
        !std::equal(ids_seg.begin() + nh, ids_seg.end(), ids.begin() + nh);
    chk(differs, "eos segmentation changes downstream n-grams");
    printf("PLE hash: eos segmentation alters ids = %d\n", (int)differs);
  }

  printf("MODEL LOAD TEST %s (%d checks failed)\n", bad ? "FAIL" : "OK", bad);
  return bad ? 1 : 0;
}
