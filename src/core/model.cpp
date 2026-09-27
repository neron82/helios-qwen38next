// Loader for Qwen3.8-Flash-Next-exl3 (arch `qwen4_exp`).
//
// Placement: GPU0 keeps the embedding, the attention/GDN stacks, the PLE, the shared experts and a
// slice of the routed experts; GPU1 keeps the rest of the routed experts and the MTP head. Experts
// are fully resident (31.3 GB total), so there is no host arena, no slot pool and no per-forward
// expert streaming - only one activation round trip per layer, which at ~10 KB/token is ~17 MB/s
// against a 2.85 GB/s link. The 26.24 GB n-gram table is mapped read-only and gathered per token.
//
// HELIOS_PIPELINE=1 re-cuts that placement along the LAYER axis instead of the expert axis: layer
// l's whole dense stack, its router and shared expert and its 512 routed experts go to the card
// that owns l (Config::layer_dev), the embedding goes to layer 0's card, lm_head and the MTP head to
// the last layer's card, and the one global hyper-connection mixer is duplicated onto both. The two
// expert arenas keep the same total bytes - the same slabs, cut between the layers instead of
// inside them. This changes WHERE weights live and nothing about the forward: Runner::init refuses
// to run a pipeline-mode forward until the scheduler stage that consumes this layout lands.
//
// The job runner (16 threads, pinned double-buffered large transfers, bf16/f32 conversions) is the
// proven one from the GLM engine; what changed is the tensor vocabulary and the destinations.
#include "core/model.hpp"
#include "core/device.hpp"
#include "cuda/cuda_shim.hpp"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>

namespace helios {

using nlohmann::json;

static std::string L(int l) { return "model.language_model.layers." + std::to_string(l); }

void cvt_bf16_f16(const void* src, half* dst, size_t n) {
  const uint16_t* s = (const uint16_t*)src;
  for (size_t i = 0; i < n; i++) {
    uint32_t bits = ((uint32_t)s[i]) << 16;
    dst[i] = __float2half_rn(*(float*)&bits);
  }
}
void cvt_bf16_f32(const void* src, float* dst, size_t n) {
  const uint16_t* s = (const uint16_t*)src;
  for (size_t i = 0; i < n; i++) {
    uint32_t bits = ((uint32_t)s[i]) << 16;
    dst[i] = *(float*)&bits;
  }
}
static void cvt_f32_f16(const void* src, half* dst, size_t n) {
  const float* s = (const float*)src;
  for (size_t i = 0; i < n; i++) dst[i] = __float2half_rn(s[i]);
}

void Group::set_dims(const std::vector<int64_t>& ts) { K = (int)(ts[2] / 16); }

// ---------- expert slab layout ----------
static size_t align64(size_t x) { return (x + 63) & ~(size_t)63; }
struct SlabLayout {
  size_t off[12];    // order: gate t,suh,svh,mul1, up..., down...
  size_t stride;
  size_t piece[12];
};
static SlabLayout make_slab_layout(int hidden, int moe_inter, int K) {
  size_t t = (size_t)hidden * moe_inter * K / 8;
  SlabLayout sl{};
  size_t o = 0;
  sl.piece[0] = t;                   sl.off[0] = o;  o += t;            // gate trellis
  sl.piece[1] = (size_t)hidden * 2;  sl.off[1] = o;  o += hidden * 2;   // gate suh (in = hidden)
  sl.piece[2] = (size_t)moe_inter * 2; sl.off[2] = o; o += moe_inter * 2; // gate svh (out = inter)
  sl.piece[3] = 4;                   sl.off[3] = o;  o = align64(o + 4); // gate mul1
  sl.piece[4] = t;                   sl.off[4] = o;  o += t;            // up
  sl.piece[5] = (size_t)hidden * 2;  sl.off[5] = o;  o += hidden * 2;
  sl.piece[6] = (size_t)moe_inter * 2; sl.off[6] = o; o += moe_inter * 2;
  sl.piece[7] = 4;                   sl.off[7] = o;  o = align64(o + 4);
  sl.piece[8] = t;                   sl.off[8] = o;  o += t;            // down
  sl.piece[9] = (size_t)moe_inter * 2; sl.off[9] = o; o += moe_inter * 2;
  sl.piece[10] = (size_t)hidden * 2; sl.off[10] = o; o += hidden * 2;
  sl.piece[11] = 4;                  sl.off[11] = o; o = align64(o + 4);
  sl.stride = o;
  return sl;
}

// ---------- loading jobs ----------
struct Job {
  const TensorInfo* ti = nullptr;    // source tensor
  const ShardSet* set = nullptr;     // which ShardSet ti belongs to (nullptr = the 5 model shards)
  void* dst = nullptr;
  size_t bytes = 0, elems = 0;
  int device = -1;                   // -1 = host destination (no CUDA)
  enum Kind : uint8_t { RAW, BF16_F16, BF16_F32, F32_F16, MUL1 } kind = RAW;
  std::string name;
};

// Which device the MTP draft head runs on. It must match the device the runner drives the draft
// from: `Runner::spec_step` uses gpu(0).stream(0), `mtp_scratch_init` allocates on card 0, the draft's
// KV rows are in kv_[0] and the draft logits are written to row 1 of the card-0 head buffer. Loading
// the weights anywhere else makes every draft kernel dereference another context's pointers - an
// illegal memory access, which is how this was found (compute-sanitizer flagged the weight operand
// of tap_norm_k). That is the default, and it is now expressed as Config::mtp_dev(), which returns
// card 0 with the pipeline off and the LAST layer's card with it on - the draft consumes the trunk's
// final mixer output, so it has to live wherever the end of the stack is produced.

struct Loader : Model {
  bool ram_only = false;
  std::string dir_;              // model directory (side files: ngram table, mtp patch)
  std::vector<Job> jobs;
  SlabLayout slay;
  int missing = 0;
  bool verbose_alloc = true;
  const char* ngram_shard_[512] = {nullptr};

  const TensorInfo* T(const std::string& name) {
    auto* t = shards.find(name);
    if (!t) { fprintf(stderr, "[model] MISSING tensor %s\n", name.c_str()); missing++; }
    return t;
  }
  void* A(size_t bytes, int dev, int align = 256) {
    if (ram_only) return nullptr;
    if (verbose_alloc && bytes > (16u << 20))
      printf("[model] alloc dev%d %.1f MB (gpu%d total %.2f GB)\n", dev, bytes / 1048576.0, dev,
             (dev == 0 ? gpu0_bytes : gpu1_bytes) / 1073741824.0);
    void* p = Engine::instance().gpu(dev).alloc(bytes, align);
    if (dev == 0) gpu0_bytes += bytes; else gpu1_bytes += bytes;
    return p;
  }
  void j(const TensorInfo* t, void* dst, int dev, Job::Kind k, size_t elems = 0) {
    if (!t || ram_only) return;
    Job jb; jb.ti = t; jb.dst = dst; jb.device = dev; jb.kind = k;
    jb.bytes = t->bytes; jb.elems = elems ? elems : t->elems; jb.name = t->name;
    jobs.push_back(jb);
  }

  // one EXL3 group: four tensors; mul1 lands in the host struct.
  void load_group(const std::string& base, Group& g, int dst_dev) {
    auto* tr = T(base + ".trellis");
    auto* su = T(base + ".suh");
    auto* sv = T(base + ".svh");
    auto* mu = T(base + ".mul1");
    if (!tr || !su || !sv || !mu) return;
    g.set_dims(tr->shape);
    g.out = (int)sv->elems; g.in = (int)su->elems;
    g.trellis = ram_only ? nullptr : A(tr->bytes, dst_dev);
    g.suh = (const half*)(ram_only ? nullptr : A(su->bytes, dst_dev));
    g.svh = (const half*)(ram_only ? nullptr : A(sv->bytes, dst_dev));
    j(tr, g.trellis, dst_dev, Job::RAW);
    j(su, (void*)g.suh, dst_dev, Job::RAW);
    j(sv, (void*)g.svh, dst_dev, Job::RAW);
    if (mu && !ram_only) { Job jb; jb.ti = mu; jb.dst = &g.mul1; jb.device = -1; jb.kind = Job::MUL1;
                           jb.bytes = mu->bytes; jb.name = mu->name; jobs.push_back(jb); }
  }

  // Small dense tensors. This checkpoint stores most of them bf16; `src_bf16` converts to f16, and
  // the ones that feed exponentials or norms go to f32 via `f32`.
  void load_small(const std::string& name, const half*& dst, int dev, bool src_bf16) {
    auto* t = T(name);
    if (!t) return;
    if (!ram_only) dst = (const half*)A(t->elems * 2, dev);
    j(t, (void*)dst, dev, src_bf16 ? Job::BF16_F16 : Job::RAW);
  }
  void load_f32(const std::string& name, const float*& dst, int dev, bool src_bf16) {
    auto* t = T(name);
    if (!t) return;
    if (!ram_only) dst = (const float*)A(t->elems * 4, dev);
    j(t, (void*)dst, dev, src_bf16 ? Job::BF16_F32 : Job::RAW);
  }
  // Keep the checkpoint's bf16 as-is on device: the GDN prologue takes bfloat16 pointers and does
  // its own upcast, so any conversion here would be a rounding the reference does not have.
  void load_bf16(const std::string& name, const void*& dst, int dev) {
    auto* t = T(name);
    if (!t) return;
    if (!ram_only) dst = A(t->bytes, dev);
    j(t, (void*)dst, dev, Job::RAW);
  }

  bool parse_config(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) { fprintf(stderr, "[model] no config.json\n"); return false; }
    json c; f >> c;
    json& tc = c.contains("text_config") ? c["text_config"] : c;
    cfg.n_layers = tc.value("num_hidden_layers", 48);
    cfg.hidden = tc.value("hidden_size", 2560);
    cfg.gdn_k_heads = tc.value("linear_num_key_heads", 16);
    cfg.gdn_v_heads = tc.value("linear_num_value_heads", 48);
    cfg.gdn_k_dim = tc.value("linear_key_head_dim", 128);
    cfg.gdn_v_dim = tc.value("linear_value_head_dim", 128);
    cfg.gdn_conv_k = tc.value("linear_conv_kernel_dim", 4);
    cfg.gdn_qkv_out = 2 * cfg.gdn_k_heads * cfg.gdn_k_dim + cfg.gdn_v_heads * cfg.gdn_v_dim;
    cfg.gdn_z_out = cfg.gdn_v_heads * cfg.gdn_v_dim;
    cfg.attn_heads = tc.value("num_attention_heads", 24);
    cfg.attn_kv_heads = tc.value("num_key_value_heads", 2);
    cfg.attn_head_dim = tc.value("head_dim", 256);
    cfg.attn_q_out = 2 * cfg.attn_heads * cfg.attn_head_dim;   // q interleaved with its gate
    cfg.idx_heads = tc.value("indexer_n_heads", 4);
    cfg.idx_kv_heads = tc.value("indexer_kv_heads", 1);
    cfg.idx_dim = tc.value("indexer_head_dim", 128);
    cfg.idx_budget = tc.value("indexer_budget", 2048);
    cfg.idx_compress = tc.value("indexer_compress_ratio", 4);
    cfg.idx_qk_out = (cfg.idx_heads + cfg.idx_kv_heads) * cfg.idx_dim;
    cfg.n_expert = tc.value("num_experts", 512);
    cfg.topk = tc.value("num_experts_per_tok", 10);
    cfg.moe_inter = tc.value("moe_intermediate_size", 640);
    cfg.shared_inter = tc.value("shared_expert_intermediate_size", 640);
    cfg.hc_mult = tc.value("hc_count", 4);
    cfg.hc_dim = cfg.hc_mult * cfg.hidden;
    cfg.hc_rank = tc.value("hc_lowrank", 320);
    cfg.ple_dim = tc.value("ple_embed_dim", 2560);
    cfg.ngram_size = tc.value("ngram_size", 3);
    cfg.heads_per_ngram = tc.value("heads_per_ngram", 8);
    cfg.rms_eps = tc.value("rms_norm_eps", 1e-6f);
    cfg.vocab = tc.value("vocab_size", 248320);
    if (tc.contains("rope_parameters")) {
      auto& rp = tc["rope_parameters"];
      cfg.rope_theta = rp.value("rope_theta", 10000000.0f);
      cfg.partial_rotary = rp.value("partial_rotary_factor", 0.25f);
      if (rp.contains("mrope_section") && rp["mrope_section"].is_array())
        for (int i = 0; i < 3 && i < (int)rp["mrope_section"].size(); i++)
          cfg.mrope_section[i] = rp["mrope_section"][i].get<int>();
    }
    // PLE runs ahead of the layer whose 1-based index is listed (config: ple_layer_ids [2]).
    if (tc.contains("ple_layer_ids") && tc["ple_layer_ids"].is_array() && !tc["ple_layer_ids"].empty())
      cfg.ple_layer = tc["ple_layer_ids"][0].get<int>() - 1;
    cfg.full_attn.assign(cfg.n_layers, false);
    if (tc.contains("layer_types"))
      for (int i = 0; i < cfg.n_layers && i < (int)tc["layer_types"].size(); i++)
        cfg.full_attn[i] = tc["layer_types"][i].get<std::string>() == "full_attention";
    // MTP is a property of the checkpoint, not of a config key. The GLM checkpoint ships its draft
    // head as mtp.* tensors; Qwen3.8 ships none. The config key defaults to 1 when absent, which made
    // the engine build - and then run - a phantom draft layer whose weights were never loaded, giving
    // an illegal memory access on the very first decode step of the DEFAULT configuration. Derive it
    // from what the checkpoint actually declares.
    {
      auto ts = shards.tensors();   // sorted by name
      auto it = std::lower_bound(ts.begin(), ts.end(), std::string_view("mtp."),
                                 [](const TensorInfo* t, std::string_view v) {
                                   return t->name < v;
                                 });
      const bool declares_mtp = it != ts.end() && (*it)->name.rfind("mtp.", 0) == 0;
      cfg.has_mtp = declares_mtp && tc.value("mtp_num_hidden_layers", 1) > 0;
      if (!declares_mtp) {
        if (tc.contains("mtp_num_hidden_layers"))
          fprintf(stderr, "[model] config declares mtp_num_hidden_layers but the checkpoint ships no "
                          "mtp.* tensors; draft head disabled\n");
        cfg.has_mtp = false;
      }
    }
    // ---- two-stage layer pipeline (opt-in; OFF reproduces the shipped engine exactly) ----
    //
    // HELIOS_PIPELINE=1 changes WHERE the weights live and nothing else: layer l's dense stack,
    // its shared expert and its 512 routed experts go to the card that owns l, the embedding goes
    // to layer 0's card, lm_head to the last layer's card, the MTP head to the last layer's card,
    // and the one global hyper-connection mixer is duplicated onto both. The expert arenas are
    // re-cut along the layer axis, so their total is unchanged (each card now holds 512 slabs for
    // half as many layers instead of 160/352 slabs for all of them).
    //
    // HELIOS_PIPELINE_SPLIT moves the boundary (default n_layers/2 = 24/24) so the split can be
    // swept; it is clamped to [1, n_layers-1] because a card with no layer has no reason to exist
    // and layer 0's card must be the one holding the embedding.
    // The 2-GPU layer pipeline is the DEFAULT (HELIOS_PIPELINE=0 falls back to the single-pass
    // within-layer split). Measured on the 1:1 grid vs exllamav3: with the pipeline ON helios is 50% of
    // the baseline prefill (49-52% across 4k/8k/16k) and decode is 5-26% FASTER than the same build
    // with the pipeline off, because each card runs a disjoint layer half (its own attention + MoE
    // experts) instead of the two cards serializing within every layer. It reorders the per-token
    // expert sum (a documented reassociation, not a defect), so the token stream differs from the
    // pipeline-off build in the last digits — see RESULTS.md. HELIOS_PIPELINE_SPLIT moves the boundary
    // (default n_layers/2 = 24/24; measured 18..28 is within ~1%, i.e. flat).
    {
        const char* pv = getenv("HELIOS_PIPELINE");
        cfg.pipeline = !(pv && atoi(pv) == 0);
        if (cfg.pipeline) {
            int sp = cfg.n_layers / 2;
            if (const char* sv = getenv("HELIOS_PIPELINE_SPLIT")) sp = atoi(sv);
            if (sp < 1) sp = 1;
            if (sp > cfg.n_layers - 1) sp = cfg.n_layers - 1;
            cfg.split = sp;
            if (getenv("HELIOS_EXPERTS_GPU1"))
                fprintf(stderr, "[model] HELIOS_EXPERTS_GPU1 is ignored with the pipeline on: each "
                                "card holds all %d experts, for its own layers only\n", cfg.n_expert);
        }
    }
    // 512 experts x ~1.19 MB = 639 MB per layer. Pipeline OFF: GPU1 takes the majority (it has room
    // and the fast link), GPU0 keeps the rest beside the dense stack and the KV cache. Pipeline ON:
    // unused - each card takes all 512 for its own layers, which is what experts_per_card() returns.
    if (const char* ev = getenv("HELIOS_EXPERTS_GPU1")) cfg.experts_gpu1 = atoi(ev);
    if (cfg.experts_gpu1 <= 0 || cfg.experts_gpu1 > cfg.n_expert) cfg.experts_gpu1 = 352;
    cfg.ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;   // 16
    cfg.ngram_row_dim = cfg.ple_dim / cfg.ngram_heads;              // 160
    return true;
  }

  // ---- per-module loaders (all names from notes/qwen_tensor_map.md) ----
  //
  // `dev` is the card that OWNS the layer. With HELIOS_PIPELINE off it is always 0 - the shipped
  // layout, byte for byte. With it on, layer l's dense weights live on cfg.layer_dev(l), so a
  // layer's kernels never dereference a pointer from the other card's context.

  int load_gdn(int l, Layer& ly) {
    const int dev = cfg.layer_dev(l);
    std::string p = L(l) + ".linear_attn";
    load_group(p + ".in_proj_qkv", ly.gdn.qkv, dev);
    load_group(p + ".in_proj_z", ly.gdn.z, dev);
    load_group(p + ".out_proj", ly.gdn.out, dev);
    // Stays bf16: cuda_causal_conv1d_update takes a bfloat16 weight, so converting the file's
    // bf16 to fp16 here (as an earlier revision did) made the kernel read half of every
    // weight's bytes as garbage and emit an all-zero conv output.
    load_bf16(p + ".conv1d.weight", ly.gdn.conv, dev);
    load_small(p + ".in_proj_a.weight", ly.gdn.a_proj, dev, false);
    load_small(p + ".in_proj_b.weight", ly.gdn.b_proj, dev, false);
    // A_log / dt_bias stay bf16: the GDN prologue takes bfloat16 pointers and upcasts internally,
    // so converting them here would introduce a rounding the reference does not have.
    load_bf16(p + ".A_log", ly.gdn.A_log, dev);
    load_bf16(p + ".dt_bias", ly.gdn.dt_bias, dev);
    load_bf16(p + ".norm.weight", ly.gdn.norm, dev);   // stays bf16, like A_log/dt_bias
    return dev;
  }

  int load_attn(int l, Layer& ly) {
    const int dev = cfg.layer_dev(l);
    std::string p = L(l) + ".self_attn";
    load_group(p + ".q_proj", ly.attn.q, dev);      // 2560 -> 12288 (q interleaved with gate)
    load_group(p + ".k_proj", ly.attn.k, dev);
    load_group(p + ".v_proj", ly.attn.v, dev);
    load_group(p + ".o_proj", ly.attn.o, dev);
    load_small(p + ".q_norm.weight", ly.attn.q_norm, dev, true);
    load_small(p + ".k_norm.weight", ly.attn.k_norm, dev, true);
    std::string ip = p + ".indexer";
    load_group(ip + ".index_qk_proj", ly.attn.idx_qk, dev);        // 2560 -> 640 fused q|k
    load_small(ip + ".q_layernorm.weight", ly.attn.idx_q_ln, dev, true);
    load_small(ip + ".k_layernorm.weight", ly.attn.idx_k_ln, dev, true);
    return dev;
  }

  // mtp.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up} are shipped in
  // mtp_hyper_connection_mixer_patch.safetensors, not in the 5 model shards. The draft head's
  // mixer is its OWN weight set (not the shared global one), so it rides with the draft on
  // cfg.mtp_dev().
  void load_mtp_mixer() {
    try { extra.load_file(dir_ + "/mtp_hyper_connection_mixer_patch.safetensors"); }
    catch (const std::exception& e) { fprintf(stderr, "[model] mtp patch: %s\n", e.what()); return; }
    const int mdev = cfg.mtp_dev();
    auto add = [&](const char* nm, const half*& dst) {
      auto* t = extra.find(std::string("mtp.hyper_connection_mixer.") + nm);
      if (!t) { fprintf(stderr, "[model] MISSING mtp mixer %s\n", nm); missing++; return; }
      if (!ram_only) dst = (const half*)A(t->elems * 2, mdev);
      Job jb;
      jb.ti = t; jb.set = &extra; jb.dst = (void*)dst; jb.device = mdev; jb.kind = Job::RAW;
      jb.bytes = t->bytes; jb.elems = t->elems; jb.name = t->name;
      if (!ram_only) jobs.push_back(jb);
    };
    add("hc_norm.weight", mtp.mixer.norm);
    add("input_mix_weight_down.weight", mtp.mixer.down);
    add("input_mix_weight_up.weight", mtp.mixer.up);
  }


  void load_moe_common(const std::string& p, Layer& ly, int dev) {
    auto* gw = T(p + ".gate.weight");
    if (gw) {
      if (!ram_only) ly.moe.router = (const half*)A(gw->bytes, dev);
      j(gw, (void*)ly.moe.router, dev, Job::RAW);
    }
    load_group(p + ".shared_expert.gate_proj", ly.moe.shared[0], dev);
    load_group(p + ".shared_expert.up_proj", ly.moe.shared[1], dev);
    load_group(p + ".shared_expert.down_proj", ly.moe.shared[2], dev);
    load_small(p + ".shared_expert_gate.weight", ly.moe.shared_gate, dev, false);
  }

  // Gated-residual hyper-connection site. The MTP head has the same sites (and its own mixer).
  void load_hc(const std::string& base, HcWeights& hc, int dev) {
    load_small(base + ".hc_norm.weight", hc.norm, dev, false);
    load_small(base + ".input_mix_weight_down.weight", hc.down, dev, false);
    load_small(base + ".input_mix_weight_up.weight", hc.up, dev, false);
    if (shards.find(base + ".block_inject_weight.weight"))
      load_small(base + ".block_inject_weight.weight", hc.inject, dev, false);
  }

  // Routed experts go straight to the card that computes them: no host arena, no slots. Each card
  // holds a contiguous (layer, local-expert) arena; the slab pieces are gate/up/down trellis+suh+svh
  // (mul1 stays a host word - the kernels take it from the group struct).
  // `prefix` is the full tensor prefix (the MTP layer's experts live under mtp.layers.0, not under
  // the trunk's layer namespace), while `slot` is the arena slot the slabs land in.
  //
  // Pipeline OFF: the split is WITHIN each layer - GPU1 holds experts [0, experts_gpu1), GPU0 the
  // tail - so every layer has slabs on both cards. Pipeline ON: the split is ALONG layers - the
  // card that owns this layer holds all `n_expert` slabs, at their global ids, and the other card
  // holds none of them. The arena total is the same either way; only the cut moves.
  void load_experts(const std::string& prefix, int slot) {
    std::string p = prefix + ".mlp.experts.";
    static const char* tn[3] = {"gate_proj", "up_proj", "down_proj"};
    static const char* pn[4] = {".trellis", ".suh", ".svh", ".mul1"};
    for (int e = 0; e < cfg.n_expert; e++) {
      int dev = cfg.pipeline ? cfg.layer_dev(slot) : (e < cfg.expert_base(0) ? 1 : 0);
      int per_layer = cfg.experts_per_card(dev);
      int local = e - cfg.expert_base(dev);
      char* arena = dev == 1 ? experts1 : experts0;
      if (local < 0 || local >= per_layer || ram_only || !arena) continue;
      size_t base_off = ((size_t)cfg.expert_slot(slot, dev) * per_layer + local) * slay.stride;
      for (int t = 0; t < 3; t++) {
        for (int q = 0; q < 3; q++) {          // trellis, suh, svh (mul1 is a host value)
          auto* ti = shards.find(p + std::to_string(e) + "." + tn[t] + pn[q]);
          if (!ti) { fprintf(stderr, "[model] MISSING %s%d.%s%s\n", p.c_str(), e, tn[t], pn[q]); missing++; continue; }
          Job jb;
          jb.ti = ti; jb.dst = arena + base_off + slay.off[t * 4 + q]; jb.device = dev;
          jb.bytes = ti->bytes; jb.name = ti->name; jb.kind = Job::RAW;
          jobs.push_back(jb);
        }
      }
      // the mul1 word is identical across experts of a layer; keep the first one for the layer
      if (e == 0) {
        auto* mu = shards.find(p + "0." + tn[0] + ".mul1");
        if (mu) { Job jb; jb.ti = mu; jb.dst = &ly_mul1_[slot]; jb.device = -1; jb.kind = Job::MUL1;
                  jb.bytes = mu->bytes; jb.name = mu->name; jobs.push_back(jb); }
      }
    }
  }

  int ly_mul1_[64] = {0};

  // The PLE runs ahead of one early layer (config: ple_layer_ids [2] -> layer 1), so it is a
  // per-LAYER weight set: it goes on the card that owns that layer, and so does its conv state.
  void load_ple() {
    int l = cfg.ple_layer;
    if (l < 0 || l >= cfg.n_layers) return;
    Layer& ly = layers[l];
    ly.has_ple = true;
    const int dev = cfg.layer_dev(l);
    std::string p = L(l) + ".ple";
    load_small(p + ".conv1d.weight", ly.ple.conv, dev, false);
    load_small(p + ".key_proj.weight", ly.ple.key_proj, dev, false);
    load_small(p + ".value_proj.weight", ly.ple.value_proj, dev, false);
    load_small(p + ".norm_conv.weight", ly.ple.norm_conv, dev, true);
    load_small(p + ".norm_key.weight", ly.ple.norm_key, dev, true);
    load_small(p + ".norm_query.weight", ly.ple.norm_query, dev, true);
  }

  // The draft head. It is ONE extra layer that consumes the trunk's final mixer output, so it
  // belongs to the card that owns the LAST layer (cfg.mtp_dev()): 0 with the pipeline off - the
  // shipped layout, where the runner drives the whole draft from card 0's stream and every draft
  // weight had to be there - and 1 with it on, where the last layer's card produces that output.
  void load_mtp() {
    if (!cfg.has_mtp) return;
    Layer& ly = mtp.layer;
    ly.index = cfg.n_layers;
    ly.full = true;
    // The draft head's weights live on the device the draft actually runs on. They were loaded to
    // card 1 while the runner drove the whole draft from card 0's stream, with the scratch, the KV
    // rows and the logits row all on card 0 - so every draft kernel dereferenced card-1 pointers from
    // card-0's context. compute-sanitizer named it exactly: an invalid __global__ read in
    // tap_norm_k on the weight operand. 56 non-expert tensors, 61.9 MB, and card 0 has ~2.3 GB free.
    const int M = cfg.mtp_dev();
    load_group("mtp.fc_embedding", mtp.fc_embedding, M);
    load_group("mtp.fc_hidden", mtp.fc_hidden, M);
    load_small("mtp.pre_fc_norm_hidden.weight", mtp.pre_fc_norm_hidden, M, false);
    load_small("mtp.pre_fc_norm_embedding.weight", mtp.pre_fc_norm_embedding, M, true);
    std::string p = "mtp.layers.0";
    load_group(p + ".self_attn.q_proj", ly.attn.q, M);
    load_group(p + ".self_attn.k_proj", ly.attn.k, M);
    load_group(p + ".self_attn.v_proj", ly.attn.v, M);
    load_group(p + ".self_attn.o_proj", ly.attn.o, M);
    load_small(p + ".self_attn.q_norm.weight", ly.attn.q_norm, M, true);
    load_small(p + ".self_attn.k_norm.weight", ly.attn.k_norm, M, true);
    load_group(p + ".self_attn.indexer.index_qk_proj", ly.attn.idx_qk, M);
    load_small(p + ".self_attn.indexer.q_layernorm.weight", ly.attn.idx_q_ln, M, true);
    load_small(p + ".self_attn.indexer.k_layernorm.weight", ly.attn.idx_k_ln, M, true);
    load_moe_common(p + ".mlp", ly, M);
    load_hc(p + ".attn_hyper_connection", ly.hc_attn, M);
    load_hc(p + ".mlp_hyper_connection", ly.hc_mlp, M);
    load_mtp_mixer();
    // Experts follow the slot's owning card like every other layer's: with the pipeline off that is
    // the within-layer 352/160 split, with it on the last card's own 512.
    load_experts("mtp.layers.0", cfg.n_layers);
  }

  // Map ngram_embedding.safetensors read-only. The per-head offsets, vocab sizes and layer
  // multipliers live in its three I64 tensors and must be read at runtime; the 128 trellis payloads
  // are gathered straight from host memory per token.
  bool map_ngram(const std::string& dir, PleWeights& ple) {
    std::string path = dir + "/ngram_embedding.safetensors";
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[model] cannot open %s\n", path.c_str()); return false; }
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
    void* map = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) { fprintf(stderr, "[model] mmap %s failed\n", path.c_str()); return false; }
    uint64_t hlen = 0;
    memcpy(&hlen, map, 8);
    json h = json::parse(std::string((const char*)map + 8, (size_t)hlen), nullptr, false);
    if (h.is_discarded()) { fprintf(stderr, "[model] bad ngram header\n"); return false; }
    const char* base = (const char*)map + 8 + hlen;
    int nsh = 0;
    for (auto it = h.begin(); it != h.end(); ++it) {
      const std::string& nm = it.key();
      if (nm == "__metadata__") continue;
      uint64_t off = it.value()["data_offsets"][0].get<uint64_t>();
      const char* ptr = base + off;
      const std::string dt = it.value()["dtype"].get<std::string>();
      if (dt == "I16") {
        size_t sp = nm.find("shard_");
        if (sp == std::string::npos) continue;
        int k = atoi(nm.c_str() + sp + 6);
        if (k >= 0 && k < cfg.ngram_shards) { ngram_shard_[k] = ptr; nsh++; }
      } else if (dt == "F16") {
        ple.head_bias = (const half*)ptr;                  // [num_heads, row_dim]
      } else if (dt == "I64") {
        // three arrays, told apart by name (head_offsets / head_vocab_sizes / layer_multipliers)
        if (nm.find("head_offsets") != std::string::npos) ple.head_offsets = (const int64_t*)ptr;
        else if (nm.find("vocab") != std::string::npos) ple.head_vocab_sizes = (const int64_t*)ptr;
        else if (nm.find("multiplier") != std::string::npos) ple.layer_multipliers = (const int64_t*)ptr;
      }
    }
    ple.table = ngram_shard_[0];
    for (int k = 0; k < cfg.ngram_shards && k < 128; k++) ple.shards[k] = ngram_shard_[k];
    ram_bytes += (size_t)st.st_size;
    printf("[model] ngram table mapped: %d/%d shards, %.2f GB (bias=%s offs=%s vocab=%s mult=%s)\n",
           nsh, cfg.ngram_shards, (double)st.st_size / 1073741824.0,
           ple.head_bias ? "ok" : "MISSING", ple.head_offsets ? "ok" : "MISSING",
           ple.head_vocab_sizes ? "ok" : "MISSING", ple.layer_multipliers ? "ok" : "MISSING");
    return nsh == cfg.ngram_shards;
  }

  // Compare device copies back against the shard bytes. Catches placement/transfer bugs before any
  // kernel runs - the cheap check that made the GLM port debuggable.
  void verify_gpu_tensors() {
    struct Item { std::string name; void* dev; size_t bytes; };
    std::vector<Item> items;
    auto addg = [&](const std::string& n, const Group& g) {
      if (g.trellis) items.push_back({n + ".trellis", g.trellis, (size_t)g.out * g.in * g.K / 8 * 2});
      if (g.suh) items.push_back({n + ".suh", (void*)g.suh, (size_t)g.in * 2});
      if (g.svh) items.push_back({n + ".svh", (void*)g.svh, (size_t)g.out * 2});
    };
    for (int l = 0; l < cfg.n_layers; l++) {
      Layer& ly = layers[l];
      std::string p = L(l);
      if (cfg.full_attn[l]) {
        addg(p + ".self_attn.q_proj", ly.attn.q);
        addg(p + ".self_attn.o_proj", ly.attn.o);
        addg(p + ".self_attn.indexer.index_qk_proj", ly.attn.idx_qk);
      } else {
        addg(p + ".linear_attn.in_proj_qkv", ly.gdn.qkv);
        addg(p + ".linear_attn.out_proj", ly.gdn.out);
      }
      addg(p + ".mlp.shared_expert.gate_proj", ly.moe.shared[0]);
    }
    addg("lm_head", lm_head);
    int bad = 0, n = 0;
    std::vector<char> buf, ref;
    for (auto& it : items) {
      const TensorInfo* ti = shards.find(it.name);
      if (!ti) continue;
      size_t bytes = std::min(it.bytes, ti->bytes);
      if (!bytes) continue;
      cudaPointerAttributes at{};
      int dev = -1;
      if (cudaPointerGetAttributes(&at, it.dev) == cudaSuccess && at.type == cudaMemoryTypeDevice)
        dev = at.device;
      if (dev < 0) { fprintf(stderr, "[verify-gpu] unresolved ptr for %s\n", it.name.c_str()); bad++; continue; }
      cudaSetDevice(dev);
      buf.resize(bytes); ref.resize(bytes);
      cudaMemcpy(buf.data(), it.dev, bytes, cudaMemcpyDeviceToHost);
      shards.read(*ti, ref.data());
      // WHERE the bytes differ is the whole diagnosis: a wrong tail means the last chunk of a
      // double-buffered transfer never landed, a wrong middle band means the pinned staging buffer
      // was overwritten while its H2D was still in flight, and a scattered pattern means the
      // destination itself was clobbered. "MISMATCH" alone says none of that.
      size_t first = (size_t)-1, nbad = 0;
      for (size_t i = 0; i < bytes; i++)
        if (buf[i] != ref[i]) { if (first == (size_t)-1) first = i; nbad++; }
      if (nbad) {
        fprintf(stderr, "[verify-gpu] MISMATCH %s (%zu bytes compared, %zu differ, first at "
                        "0x%zx = %.1f%% in)\n",
                it.name.c_str(), bytes, nbad, first, 100.0 * (double)first / (double)bytes);
        bad++;
      }
      n++;
    }
    printf("[verify-gpu] %d device tensors checked, %d mismatches\n", n, bad);
    cudaSetDevice(Engine::instance().gpu(0).phys_idx());
  }

  void run_jobs(bool verbose);

  bool load(const std::string& dir, bool ro, bool verbose) {
    ram_only = ro;
    dir_ = dir;
    try { shards.load_dir(dir); }
    catch (const std::exception& e) { fprintf(stderr, "[model] shard load failed: %s\n", e.what()); return false; }
    if (!parse_config(dir)) return false;

    // expert K bits from a sample trellis (routed experts are 2-bit, the shared ones 4-bit)
    auto* probe = shards.find(L(0) + ".mlp.experts.0.gate_proj.trellis");
    int K = probe ? (int)(probe->shape[2] / 16) : 2;
    slay = make_slab_layout(cfg.hidden, cfg.moe_inter, K);
    expert_stride = slay.stride;
    expert_K = K;
    for (int i = 0; i < 12; i++) slab_off[i] = slay.off[i];   // the runner builds pointer tables
    printf("[model] layers=%d (%d full attn, %d GDN) expert slab %.3f MB x %d x %d -> %.2f GB\n",
           cfg.n_layers, (int)std::count(cfg.full_attn.begin(), cfg.full_attn.end(), true),
           (int)std::count(cfg.full_attn.begin(), cfg.full_attn.end(), false),
           slay.stride / 1048576.0, cfg.n_expert, cfg.n_layers + (cfg.has_mtp ? 1 : 0),
           (double)slay.stride * cfg.n_expert * (cfg.n_layers + (cfg.has_mtp ? 1 : 0)) / 1073741824.0);

    if (!ram_only) {
      // Fully resident expert arenas, no host copy. Pipeline OFF: every layer has cfg.experts_gpu0()
      // slabs on card 0 and cfg.experts_gpu1() on card 1 (the within-layer split). Pipeline ON: each
      // card holds ALL cfg.n_expert slabs but only for its own cfg.expert_layers(card) layers, so
      // the two arenas sum to the same bytes as before - the cut moves from "within a layer" to
      // "between the layers".
      experts1_bytes = (size_t)cfg.experts_per_card(1) * cfg.expert_layers(1) * slay.stride;
      experts0_bytes = (size_t)cfg.experts_per_card(0) * cfg.expert_layers(0) * slay.stride;
      experts1 = (char*)A(experts1_bytes, 1, 4096);
      experts0 = (char*)A(experts0_bytes, 0, 4096);
      // The embedding feeds layer 0 (and the PLE, at layer 1), so it belongs to the FIRST stage's
      // card; lm_head consumes the LAST layer's collapsed output, so it belongs to that card. With
      // the pipeline off both are layer_dev(0) == layer_dev(n-1) == 0, i.e. the shipped layout.
      const int d_first = cfg.layer_dev(0);
      const int d_last = cfg.layer_dev(cfg.n_layers - 1);
      embed = A((size_t)cfg.vocab * cfg.hidden * 2, d_first);   // bf16 kept, converted in the gather
      { auto* te = T("model.language_model.embed_tokens.weight"); j(te, (void*)embed, d_first, Job::RAW); }
      load_group("lm_head", lm_head, d_last);
      // The global mixer is used by EVERY layer, so it cannot belong to one card: it is loaded
      // once per card when the pipeline is on (Model::mixer1) and looked up with mixer_for(dev).
      load_hc("model.language_model.hyper_connection_mixer", mixer, d_first);
      if (cfg.pipeline && d_last != d_first)
        load_hc("model.language_model.hyper_connection_mixer", mixer1, d_last);
    }

    layers.resize(cfg.n_layers);
    int n_full = 0, n_gdn = 0;
    for (int l = 0; l < cfg.n_layers; l++) {
      Layer& ly = layers[l];
      ly.index = l;
      ly.full = cfg.full_attn[l];
      ly.attn_ord = ly.full ? n_full++ : -1;
      ly.gdn_ord = ly.full ? -1 : n_gdn++;
      // Everything this layer touches - its projections, its per-layer hyper-connection sites, its
      // router and shared expert, and all 512 of its routed experts in pipeline mode - goes to the
      // card that owns it. Pipeline OFF it is all card 0 except the routed experts, which keep the
      // within-layer split (the far card receives hidden rows plus a compact assignment list,
      // ~10 KB per token).
      const int dev = cfg.layer_dev(l);
      if (ly.full) load_attn(l, ly); else load_gdn(l, ly);
      load_moe_common(L(l) + ".mlp", ly, dev);
      load_hc(L(l) + ".attn_hyper_connection", ly.hc_attn, dev);
      load_hc(L(l) + ".mlp_hyper_connection", ly.hc_mlp, dev);
      load_experts(L(l), l);
    }
    load_ple();
    load_mtp();
    if (cfg.ple_layer >= 0 && cfg.ple_layer < cfg.n_layers && layers[cfg.ple_layer].has_ple &&
        !map_ngram(dir, layers[cfg.ple_layer].ple))
      fprintf(stderr, "[model] WARNING: n-gram table incomplete - PLE will be wrong\n");

    if (verbose)
      printf("[model] jobs=%zu gpu0=%.2fGB gpu1=%.2fGB experts(%.2f+%.2fGB) ngram=%.2fGB\n",
             jobs.size(), (double)gpu0_bytes / (1 << 30), (double)gpu1_bytes / (1 << 30),
             (double)experts0_bytes / (1 << 30), (double)experts1_bytes / (1 << 30),
             (double)ram_bytes / (1 << 30));

    // Per-card placement report. With the pipeline on this is the whole point of the load, so it
    // is printed unconditionally (not under `verbose`): the operator has to be able to see which
    // card owns which layer range before anything runs.
    if (cfg.pipeline) {
      printf("[pipeline] split=%d  card0=layers[0,%d)  card1=layers[%d,%d)%s\n", cfg.split,
             cfg.split, cfg.split, cfg.n_layers, cfg.has_mtp ? " +MTP" : "");
      for (int d = 0; d < 2; d++) {
        const int lo = d == 0 ? 0 : cfg.split;
        const int hi = d == 0 ? cfg.split : cfg.n_layers;
        int nfa = 0, ngd = 0;
        for (int l = lo; l < hi; l++) (cfg.full_attn[l] ? nfa : ngd)++;
        const size_t eb = d == 0 ? experts0_bytes : experts1_bytes;
        printf("[pipeline]   card%d: %d layers (%d full-attn, %d GDN), %d experts/layer x %d "
               "layers = %.2f GB experts, total card bytes %.2f GB\n",
               d, hi - lo, nfa, ngd, cfg.experts_per_card(d), cfg.expert_layers(d),
               (double)eb / (1 << 30), (double)(d == 0 ? gpu0_bytes : gpu1_bytes) / (1 << 30));
      }
      printf("[pipeline]   embed->card%d  lm_head->card%d  MTP->card%d  mixer duplicated on both\n",
             cfg.layer_dev(0), cfg.layer_dev(cfg.n_layers - 1), cfg.mtp_dev());
    }
    run_jobs(verbose);
    if (!ram_only) verify_gpu_tensors();
    if (missing) fprintf(stderr, "[model] WARNING: %d tensors missing\n", missing);
    return missing == 0;
  }
};

// ---------- parallel job runner ----------
// 16 threads; large device copies go through pinned double-buffered staging (reads from the shards
// overlap the H2D), small ones through a scratch buffer. This is the GLM engine's runner unchanged.
struct Ctx {
  char* pin[2] = {nullptr, nullptr};
  // Events are PER DEVICE. A cudaEvent belongs to the context that created it, and recording a
  // card-0 event on card 1's stream is cudaErrorInvalidResourceHandle. One pair per card is what
  // makes a >16 MB copy to either card work; with the shipped placement the card-1 pair is simply
  // never used, because no large job is scheduled before a thread's first card-1 job.
  cudaEvent_t ev[N_GPU][2] = {};
  void* tmp = nullptr;
  size_t pin_cap = 64u << 20, tmp_cap = 64u << 20;
  bool used[2] = {false, false};
};

void Loader::run_jobs(bool verbose) {
  std::atomic<size_t> next{0};
  int nthreads = 16;
  auto t0 = std::chrono::steady_clock::now();
  auto worker = [&]() {
    Ctx cx;
    cx.tmp = malloc(cx.tmp_cap);
    if (!ram_only) {
      // Pinned staging is device-agnostic: allocate it once. Only the EVENTS are per device,
      // because an event belongs to the context that created it (see Ctx::ev).
      cudaSetDevice(Engine::instance().gpu(0).phys_idx());
      cudaMallocHost(&cx.pin[0], cx.pin_cap); cudaMallocHost(&cx.pin[1], cx.pin_cap);
      for (int d = 0; d < N_GPU; d++) {
        cudaSetDevice(Engine::instance().gpu(d).phys_idx());
        cudaEventCreateWithFlags(&cx.ev[d][0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&cx.ev[d][1], cudaEventDisableTiming);
      }
    }
    for (;;) {
      size_t i = next.fetch_add(1);
      if (i >= jobs.size()) break;
      const Job& jb = jobs[i];
      if (jb.device >= 0) cudaSetDevice(Engine::instance().gpu(jb.device).phys_idx());
      switch (jb.kind) {
        case Job::MUL1:
        case Job::RAW:
          if (jb.device < 0) { (jb.set ? *jb.set : shards).read(*jb.ti, jb.dst); break; }
          if (jb.bytes <= (16u << 20)) {
            (jb.set ? *jb.set : shards).read(*jb.ti, cx.tmp);
            cudaMemcpyAsync(jb.dst, cx.tmp, jb.bytes, cudaMemcpyHostToDevice,
                            Engine::instance().gpu(jb.device).stream(2));
            cudaStreamSynchronize(Engine::instance().gpu(jb.device).stream(2));
          } else {
            size_t done = 0, slot = 0;
            while (done < jb.bytes) {
              size_t chunk = std::min(cx.pin_cap, jb.bytes - done);
              if (cx.used[slot]) while (cudaEventQuery(cx.ev[jb.device][slot]) == cudaErrorNotReady) {}
              int fd = shards.shard_fd(jb.ti->shard);
              uint64_t pos = shards.data_start(jb.ti->shard) + jb.ti->offset + done;
              size_t rd = 0;
              while (rd < chunk) {
                ssize_t got = ::pread(fd, cx.pin[slot] + rd, chunk - rd, pos + rd);
                if (got <= 0) throw std::runtime_error("pread failed " + jb.name);
                rd += (size_t)got;
              }
              cudaStream_t st = Engine::instance().gpu(jb.device).stream(2 + slot);
              HELIOS_CUDA_CHECK(cudaMemcpyAsync((char*)jb.dst + done, cx.pin[slot], chunk,
                                                cudaMemcpyHostToDevice, st));
              HELIOS_CUDA_CHECK(cudaEventRecord(cx.ev[jb.device][slot], st));
              // Mark the buffer in use. Without this the wait above never fires, the next chunk
              // refills pin[slot] while this copy is still reading it, and the destination gets a
              // 64 MB band of the WRONG tensor's bytes - silently, intermittently, and only for the
              // two tensors big enough to take the double-buffered path (embed, lm_head).
              cx.used[slot] = true;
              done += chunk; slot ^= 1;
            }
          }
          break;
        case Job::BF16_F16: case Job::BF16_F32: case Job::F32_F16: {
          (jb.set ? *jb.set : shards).read(*jb.ti, cx.tmp);
          size_t out_bytes;
          if (jb.kind == Job::BF16_F16) { cvt_bf16_f16(cx.tmp, (half*)cx.tmp, jb.elems); out_bytes = jb.elems * 2; }
          else if (jb.kind == Job::BF16_F32) { cvt_bf16_f32(cx.tmp, (float*)cx.tmp, jb.elems); out_bytes = jb.elems * 4; }
          else { cvt_f32_f16(cx.tmp, (half*)cx.tmp, jb.elems); out_bytes = jb.elems * 2; }
          cudaMemcpyAsync(jb.dst, cx.tmp, out_bytes, cudaMemcpyHostToDevice,
                          Engine::instance().gpu(jb.device).stream(2));
          cudaStreamSynchronize(Engine::instance().gpu(jb.device).stream(2));
          break;
        }
      }
    }
    if (!ram_only) {
      // Drain BOTH cards before the pinned staging ring goes away. A large job's H2D is
      // asynchronous, and the current device here is whichever card the per-device event setup
      // last selected - so a bare cudaDeviceSynchronize() drains ONE card and frees pinned memory
      // the other card's copies are still reading from. That is a silent, run-to-run-varying
      // corruption of the weight arenas, not a crash: it produced fluent-looking garbage on roughly
      // half of all greedy runs before the loop below was made explicit.
      for (int d = 0; d < N_GPU; d++) {
        cudaSetDevice(Engine::instance().gpu(d).phys_idx());
        cudaDeviceSynchronize();
      }
      cudaFreeHost(cx.pin[0]); cudaFreeHost(cx.pin[1]);
      for (int d = 0; d < N_GPU; d++) {
        cudaSetDevice(Engine::instance().gpu(d).phys_idx());
        cudaEventDestroy(cx.ev[d][0]); cudaEventDestroy(cx.ev[d][1]);
      }
    }
    free(cx.tmp);
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
  for (auto& t : th) t.join();
  if (!ram_only) {
    cudaSetDevice(Engine::instance().gpu(0).phys_idx()); cudaDeviceSynchronize();
    cudaSetDevice(Engine::instance().gpu(1).phys_idx()); cudaDeviceSynchronize();
    cudaSetDevice(Engine::instance().gpu(0).phys_idx());
  }
  if (verbose) {
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    printf("[model] load time %.1f s (%.1f GB/s)\n", dt / 1000.0,
           (double)(ram_bytes + gpu0_bytes + gpu1_bytes) / (dt / 1000.0) / (1 << 30));
  }
}

bool Model::load(const std::string& dir, bool ram_only, bool verbose) {
  Loader ld;
  if (!ld.load(dir, ram_only, verbose)) return false;
  *static_cast<Model*>(this) = std::move(ld);
  return true;
}

}  // namespace helios
