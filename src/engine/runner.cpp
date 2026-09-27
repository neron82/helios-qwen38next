// Qwen3.8-Flash-Next runner. See runner.hpp for the layer plan and the state inventory.
#include "engine/runner.hpp"
#include "cuda/aux/gdn.cuh"
#include "cuda/aux/norm.cuh"
#include "engine/prefix.hpp"
#include "engine/seq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "engine/glue.cuh"
#include "engine/glue2.cuh"
#include "cuda/aux/gr_mix.cuh"
#include "cuda/quant/exl3_gemm.cuh"
#include "cuda/attn/kv_quant.cuh"

namespace helios {

namespace {
// Named synchronisation for locating a fault: with HELIOS_DBG set, every stage is checked so the
// first failing step is named rather than being reported by whichever later check sees it sticky.
bool dbg_stage() { static const bool on = getenv("HELIOS_DBG") != nullptr; return on; }
void chk(const char* what, cudaStream_t s) {
  if (!dbg_stage()) return;
  cudaError_t e = cudaStreamSynchronize(s);
  if (e != cudaSuccess) {
    fprintf(stderr, "[runner] FAILED at %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}
double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

bool Runner::init(Model& m, int ctx_cap, int max_chunk) {
  m_ = &m;
  ctx_cap_ = ctx_cap;
  max_chunk_ = max_chunk;
  const Config& c = m.cfg;
  auto A0 = [&](size_t bytes) { return Engine::instance().gpu(0).alloc(bytes, 256); };
  // Stage-1 placement (HELIOS_PIPELINE). `dev` is the card that owns a layer, its scratch and its
  // state. With the flag off c.layer_dev() is 0 for every layer, so AD(0, ...) is A0 and every
  // allocation below lands on card 0 in the same ORDER as before - which matters, because the
  // bump allocator's addresses have been worth tok/s before.
  auto AD = [&](int dev, size_t bytes) { return Engine::instance().gpu(dev).alloc(bytes, 256); };
  const int D = c.hidden, H = c.hc_mult;

  streams_ = (float*)A0((size_t)max_chunk * H * D * 4);
  embed16_ = (half*)A0((size_t)max_chunk * D * 2);
  mixed_ = (float*)A0((size_t)max_chunk * D * 4);
  sub_in_ = (half*)A0((size_t)max_chunk * D * 2);
  sub_out_ = (float*)A0((size_t)max_chunk * D * 4);
  post_ = (float*)A0((size_t)max_chunk * H * 4);
  ids_dev_ = (int*)A0((size_t)max_chunk * 4);
  // Logits rows the buffer HOLDS. This is a CAPACITY, not a projection width: `final_head`
  // projects one row per position only when the caller asks for it (a speculative verify), and
  // exactly ONE row - the last position - everywhere else, because `next_token()` reads row 0 and
  // projecting n-rows..n-1 into rows 0..rows-1 would make every prefill chunk sample `rows`
  // positions early. Inferring the projection width from this capacity is what broke K>1: raising
  // the capacity to K+1 also widened the PREFILL's projection, so K>=2 generated a first token
  // predicted from the wrong position and everything after it was garbage. Allocated here, before
  // the QSA block, because A() is a bump allocator and a later insert shifts every address after
  // it - which has cost real throughput before.
  head_rows_ = getenv("HELIOS_HEAD_ROWS") ? atoi(getenv("HELIOS_HEAD_ROWS")) : 2;
  if (head_rows_ < 1) head_rows_ = 1;
  if (head_rows_ > 8) head_rows_ = 8;
  // A width-(K+1) speculative verify needs K+1 logit rows: rows 0..K-1 are checked against the
  // drafts and row K is the prediction after the last committed position. Two rows are the floor
  // regardless of K, because row 1 is the MTP draft head's own output slot (spec_step writes its
  // prediction to logits_ptr() + vocab) and a 1-row buffer makes every speculative step a
  // wild write.
  {
    const char* ke = getenv("HELIOS_SPEC_K");
    const int k = ke ? atoi(ke) : 1;
    if (k + 1 > head_rows_) head_rows_ = k + 1;
    if (head_rows_ > kMaxSpecDrafts + 1) head_rows_ = kMaxSpecDrafts + 1;
    if (head_rows_ < 2) head_rows_ = 2;
  }
  logits_ = (float*)A0((size_t)head_rows_ * c.vocab * 4);
  head_had_ = A0((size_t)max_chunk * 16384 * 2);
  tap_ = (float*)A0((size_t)max_chunk * H * D * 4);
  if (!streams_ || !embed16_ || !sub_in_ || !sub_out_ || !logits_) {
    fprintf(stderr, "[runner] workspace allocation failed\n");
    return false;
  }

  if (!gdn_scratch_init(gdn_, max_chunk, 0)) { fprintf(stderr, "[runner] GDN scratch\n"); return false; }
  int n_full_layers = 0, n_gdn_layers = 0;
  for (int l = 0; l < c.n_layers; l++) c.full_attn[l] ? n_full_layers++ : n_gdn_layers++;
  // QSA: one pooled-key state per full-attention layer; n_sel = budget / compress_ratio (512 here).
  // The sparse threshold (4*n_sel + 3 = 2051) is the reference's, so below it attention stays dense and
  // exact - which is why the short-prompt parity tests are unaffected by any of this.
  if (!attn_scratch_init(attn_, max_chunk, 0, n_full_layers, ctx_cap, c.idx_budget / c.idx_compress, 8,
                         c.hidden, c.attn_kv_heads * c.attn_head_dim)) {
    fprintf(stderr, "[runner] attn scratch\n"); return false;
  }
  // The PLE is a per-LAYER site (it runs ahead of layer 1), so its workspace and its conv state
  // follow that layer's card. Off, that is card 0.
  if (!ple_scratch_init(ple_, c, max_chunk, c.layer_dev(c.ple_layer))) { fprintf(stderr, "[runner] PLE scratch\n"); return false; }
  if (!moe_scratch_init(moe_, c, max_chunk, 0, 1)) { fprintf(stderr, "[runner] MoE scratch\n"); return false; }
  // Pipeline mode builds card 0's table alone: a launch against the other card's arena is a wrong
  // answer rather than an illegal access, and with one card's scratch there is nothing to aim at.
  // Flag off, both tables are built exactly as before.
  if (!moe_tables_init(m, moe_, 0, 1, c.pipeline ? 0 : -1)) { fprintf(stderr, "[runner] MoE tables\n"); return false; }
  // ---- card 1's half of the scratch (HELIOS_PIPELINE only) ----
  //
  // A layer's kernels dereference their scratch, their state and their experts on ONE device, so
  // card 1 needs a complete set for the layers it owns: GDN, attention (including the QSA pooled-key
  // state of ITS full-attention layers only - the other card keeps the other half), the PLE if its
  // layer lives there, and a MoE workspace sized for the 512 experts its layers now hold. None of
  // this is allocated with the flag off, and none of it is touched by the current forward - see the
  // refusal at the end of this function.
  int n_full1 = 0, n_gdn1 = 0, n_full0 = 0, n_gdn0 = 0;
  for (int l = 0; l < c.n_layers; l++) {
    const int d = c.layer_dev(l);
    if (c.full_attn[l]) { d == 0 ? n_full0++ : n_full1++; } else { d == 0 ? n_gdn0++ : n_gdn1++; }
  }
  // The speculative GDN snapshot is indexed by the model's GLOBAL gdn_ordinal, on whichever card
  // owns the layer, so BOTH per-card halves are sized for every GDN layer - not just the ones on
  // that card. Sizing card 1's half by its own layer count made every speculative verify write its
  // capture past the end of the allocation for all 18 of its GDN layers: silent corruption of
  // whatever the bump allocator had handed out next at K<=2, and a hard SIGSEGV inside the D2D
  // capture copy at K>=3 (each extra draft row is another stride past the end).
  const int n_gdn_total = n_gdn0 + n_gdn1;
  if (c.pipeline) {
    if (!gdn_scratch_init(gdn1_, max_chunk, 1)) { fprintf(stderr, "[runner] GDN scratch dev1\n"); return false; }
    if (!attn_scratch_init(attn1_, max_chunk, 1, n_full1, ctx_cap,
                           c.idx_budget / c.idx_compress, 8, c.hidden,
                           c.attn_kv_heads * c.attn_head_dim)) {
      fprintf(stderr, "[runner] attn scratch dev1\n"); return false;
    }
    if (c.layer_dev(c.ple_layer) == 1 && !ple_scratch_init(ple1_, c, max_chunk, 1)) {
      fprintf(stderr, "[runner] PLE scratch dev1\n"); return false;
    }
    // d0 == d1 == 1: with the pipeline a layer's MoE is a SINGLE-card MoE over all 512 experts, so
    // this scratch is sized and placed for one card. (moe_layer still launches both cards today;
    // the single-card launch is part of the scheduler stage.)
    if (!moe_scratch_init(moe1_, c, max_chunk, 1, 1)) { fprintf(stderr, "[runner] MoE scratch dev1\n"); return false; }
    if (!moe_tables_init(m, moe1_, 1, 1, /*only_card=*/1)) { fprintf(stderr, "[runner] MoE tables dev1\n"); return false; }
    // Card 1's activation set. The forward never moves an activation off card 0 yet, so this is the
    // buffer set the scheduler will hand to a stage-1 microbatch.
    act1_.streams = (float*)AD(1, (size_t)max_chunk * H * D * 4);
    act1_.embed16 = (half*)AD(1, (size_t)max_chunk * D * 2);
    act1_.mixed = (float*)AD(1, (size_t)max_chunk * D * 4);
    act1_.sub_in = (half*)AD(1, (size_t)max_chunk * D * 2);
    act1_.sub_out = (float*)AD(1, (size_t)max_chunk * D * 4);
    act1_.post = (float*)AD(1, (size_t)max_chunk * H * 4);
    act1_.ids = (int*)AD(1, (size_t)max_chunk * 4);
    act1_.tap = (float*)AD(1, (size_t)max_chunk * H * D * 4);
    act1_.carry_tap = (float*)AD(1, (size_t)H * D * 4);
    act1_.head_had = AD(1, (size_t)max_chunk * 16384 * 2);
    if (!act1_.streams || !act1_.sub_in || !act1_.sub_out) {
      fprintf(stderr, "[runner] dev1 activation set allocation failed\n");
      return false;
    }
    // The final mixer, lm_head and the logits live where the LAST layer lives.
    last_dev_ = c.layer_dev(c.n_layers - 1);
    // Logits on the LAST layer's card, where lm_head lives, and a per-card QSA ordinal: attn1_ holds
    // one QsaLayerState per layer card 1 owns, so the model's global full-attention ordinal would
    // address past its end.
    logits1_ = (float*)AD(1, (size_t)head_rows_ * c.vocab * 4);
    if (!logits1_) { fprintf(stderr, "[runner] dev1 logits allocation failed\n"); return false; }
    attn_card_ord_.assign(n_full0 + n_full1, 0);
    {
      int seen[2] = {0, 0};
      for (int l = 0; l < c.n_layers; l++) {
        const Layer& L = m.layers[l];
        if (L.full) attn_card_ord_[L.attn_ord] = seen[c.layer_dev(l)]++;
      }
    }
    // Micro-batch depth. 1 disables intra-chunk overlap (one micro-batch = no overlap, and the
    // handoff stops costing more than it saves). Below 64 tokens a micro-batch cannot amortise the
    // grouped MoE's per-launch barriers, so the small widths - every decode step and every
    // speculative verify - always run as ONE micro-batch.
    //
    // M=1 now measures BEST, reversing an earlier result. It used to be the other way round: 880
    // (M=1) / 908 (M=2) tok/s at a 10.3k prefill. It no longer does, because the balance changed:
    // the mixer became tensor-core and the decode GQA kernel was rewritten, so the per-micro-batch
    // handoff and sync is now a larger share of a much shorter stage, while the grouped MoE's
    // 1.78 ms per-launch fixed cost is unchanged and is paid once per micro-batch. Re-measured on
    // a 12.9k prompt after those changes, 3 runs each: M=1 1516.0 / 1508.8 / 1500.7,
    // M=2 1438.2 / 1429.7 / 1432.1 - M=1 is +5.3%. Card-1 utilisation falls 83.7% -> 77.7% when
    // M=1, and that is still the better trade. M=1 is a correct non-overlapped 2-card split.
    micro_ = getenv("HELIOS_PIPE_MICRO") ? atoi(getenv("HELIOS_PIPE_MICRO")) : 1;
    if (micro_ < 1) micro_ = 1;
    if (micro_ > 32) micro_ = 32;
    // One pinned handoff slot per micro-batch. The D2H of micro-batch i+1 must not overwrite the
    // slot the H2D of micro-batch i is still reading, and the host only orders those by waiting, so
    // the slots are per micro-batch rather than one reused buffer.
    const size_t slot_rows = (size_t)((max_chunk + micro_ - 1) / micro_);
    xbounce_.resize((size_t)micro_ + 1);
    xdone_.resize((size_t)micro_ + 1);
    for (int i = 0; i <= micro_; i++) {
      // slot micro_ is reserved for whole-chunk transfers (the draft head's embedding rows), the
      // rest are the per-micro-batch handoff slots.
      const size_t rows = i < micro_ ? slot_rows : (size_t)max_chunk;
      HELIOS_CUDA_CHECK(cudaMallocHost(&xbounce_[i], rows * H * D * sizeof(float)));
    }
    // Recorded on card 0's streams, so created in card 0's context.
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
    for (int i = 0; i <= micro_; i++)
      HELIOS_CUDA_CHECK(cudaEventCreateWithFlags(&xdone_[i], cudaEventDisableTiming));
    // Card 1's half of the speculative GDN sublayer-input snapshot (see gdn_sub_snap1_). The stride
    // is the one the card-0 half is allocated with further down (widest speculative batch x hidden).
    gdn_sub_stride_ = (size_t)(kMaxSpecDrafts + 1) * c.hidden;
    gdn_sub_snap1_ = (half*)AD(1, (size_t)n_gdn_total * gdn_sub_stride_ * 2);
    if (!gdn_sub_snap1_) { fprintf(stderr, "[runner] dev1 GDN sub snapshot\n"); return false; }
  }

  int n_full = 0, n_gdn = 0;
  for (int l = 0; l < c.n_layers; l++) c.full_attn[l] ? n_full++ : n_gdn++;

  // ---- HELIOS_SEQUENCES: how many conversations stay alive, and how many actually fit ----
  //
  // The KV cache below is always allocated at EXACTLY ctx_cap rows whatever N is, because a slot
  // is a base-pointer offset into that one allocation (see kv_base) rather than a second cache.
  // So N costs no extra KV memory at all - it divides the same 6 GB N ways, and each slot can
  // hold ctx_cap/N tokens. That is what makes N=2..8 affordable here at all.
  //
  // The CLAMP against measured free VRAM therefore cannot be resolved before that allocation (it
  // has to measure what is left AFTER it), so it happens in seq_config() further down, once the
  // whole cache is in place. Only the per-slot recurrent state - the one thing N really costs -
  // is allocated after the clamp, so a clamped run never allocates state it will not use.
  seq_requested_ = 1;
  if (const char* e = getenv("HELIOS_SEQUENCES")) {
    seq_requested_ = atoi(e);
    if (seq_requested_ < 1) seq_requested_ = 1;
    if (seq_requested_ > 64) seq_requested_ = 64;
  }
  n_slots_ = seq_requested_;

  // KV cache bytes per tensor, per layer. fp16 by default; HELIOS_KV_QUANT=n switches to the
  // reference's quantized layout (kv_quant.cuh), which packs token_dim/32 codes per group into
  // `n` bits plus one half scale, so a token costs gpt*(4n+2) bytes instead of 2*token_dim - 224
  // against 1024 at n=3 on this model. K and V stay separate buffers, as they always were, so only
  // the size changes and every caller's pointer arithmetic is untouched.
  // The tensor is ctx_cap rows, INDEPENDENT of N - that is the whole point: N slots partition one
  // allocation rather than making N of them, so the 6 GB below is the same 6 GB at N=1 and N=8.
  // The quantized layout's row_bytes already includes the trailing half-scale array, so a slot's
  // stride is one expression for both formats, and because the stride divides the total exactly
  // (or under it, by at most N-1 rows) every slot's scales land inside that slot's own region.
  // With one slot the stride is the whole tensor and the base address is the old one.
  const int kv_bits = attn::kvq_bits();
  const int kv_token_dim = c.attn_kv_heads * c.attn_head_dim;
  const size_t kv_bytes = kv_bits ? (size_t)ctx_cap * attn::kvq_row_bytes(kv_token_dim, kv_bits)
                                  : (size_t)ctx_cap * c.attn_kv_heads * c.attn_head_dim * 2;
  // Where a quantized tensor's scale array starts, measured in the rows ONE SLOT owns. This is a
  // per-slot quantity and not a global one: the same ref drives both the write and the dequant read
  // path, and each of those is handed a slot's base pointer, so it has to be that slot's row count.
  const size_t kv_row_b = kv_bits ? attn::kvq_row_bytes(kv_token_dim, kv_bits)
                                  : (size_t)c.attn_kv_heads * c.attn_head_dim * 2;
  const size_t kv_total = kv_bytes;
  const size_t rec_bytes = (size_t)c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim * 4;
  const size_t conv_bytes = (size_t)c.gdn_qkv_out * c.gdn_conv_k * 2;
  const size_t ple_conv_bytes = (size_t)c.hc_dim * ((4 - 1) * c.ngram_size) * 2;
  const int n_kv = n_full + (c.has_mtp ? 1 : 0);   // the draft layer owns its own cache
  // Per-layer STATE follows the layer, exactly like its weights: a KV row written by a card-0
  // attention kernel is read by that same layer, so it never has to cross the bus. Built from the
  // layer table (no allocation of its own) and all zeros with the pipeline off, which is why the
  // loops below still allocate in exactly the same order as before.
  kv_dev_.assign(n_kv, 0);
  gdn_dev_.assign(n_gdn, 0);
  for (int l = 0; l < c.n_layers; l++) {
    const Layer& L = m.layers[l];
    if (L.full) kv_dev_[L.attn_ord] = c.layer_dev(l);
    else gdn_dev_[L.gdn_ord] = c.layer_dev(l);
  }
  if (c.has_mtp) { mtp_ord_ = n_full; kv_dev_[mtp_ord_] = c.mtp_dev(); }
  kv_[0].assign(n_kv, nullptr);
  kv_[1].assign(n_kv, nullptr);
  for (int i = 0; i < n_kv; i++) {
    kv_[0][i] = AD(kv_dev_[i], kv_total);
    kv_[1][i] = AD(kv_dev_[i], kv_total);
    if (!kv_[0][i] || !kv_[1][i]) { fprintf(stderr, "[runner] KV cache allocation failed\n"); return false; }
  }
  // Everything the engine will ever hold is now in except the per-slot recurrent state, so this is
  // the one point at which "how many slots fit" is a measurement rather than an estimate. Resolving
  // it here rather than before the KV allocation is deliberate: the clamp has to see what is left
  // AFTER the 6 GB cache, or it would authorise slot counts that then OOM.
  if (!seq_config(ctx_cap, kv_row_b)) return false;

  // Per-slot RECURRENT state: resident, one copy per slot, each piece on the card that owns the
  // layer it belongs to. This is the ONLY thing N slots costs in VRAM - the KV cache is partitioned
  // rather than duplicated, and the working set (activations, MoE/GDN scratch) is shared because
  // only one slot's forward ever runs at a time. Being resident rather than spilled to host is
  // what makes a slot switch free: it is a pointer rebind, not a 116 MB save and a 116 MB restore,
  // so there is no per-switch copy that could be half-done when the next request arrives.
  // Slot 0 is allocated at exactly the point and in exactly the order the single-sequence engine
  // used, so with N=1 not one address moves.
  slots_.resize((size_t)n_slots_);
  {   // Paired-decode slot numbers, ONE COPY PER CARD. The GDN layers that read them are split across
       // both cards, and a card-0 pointer dereferenced by a card-1 kernel yields garbage rather than
       // an error - which is how this presented: the recurrent kernel reading 25 MB past a 3.1 MB
       // state block, because a garbage slot index times the per-slot stride lands there.
    bool ok = true;
    for (int d = 0; d < 2; d++) {
      int* raw = nullptr;
      HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
      if (cudaMalloc(&raw, 2 * sizeof(int) * 2) != cudaSuccess) { ok = false; break; }
      gdn_slots_dev_[d] = raw;
      conv_slots_dev_[d] = raw + 2;
    }
    if (!ok) {
      fprintf(stderr, "[runner] batch slot array allocation failed; paired decode disabled\n");
      for (int d = 0; d < 2; d++) { gdn_slots_dev_[d] = conv_slots_dev_[d] = nullptr; }
    }
  }
  // Per-slot GDN state, allocated so that a slot's layers are CONTIGUOUS and the per-slot stride is
  // exactly the number of GDN layers on that card.
  //
  // The bump allocator cannot be relied on for this. Device::alloc sub-allocates from a pool while
  // the pool lasts and then falls back to a fresh cudaMalloc, and the 227 MB of recurrent state is
  // exactly the kind of allocation that lands on that boundary - so "slot 1 is n_gdn states after
  // slot 0" holds for the first few layers and silently stops holding partway through. The symptom
  // is the recurrent kernel reading 25 MB past a 3.1 MB block, because a garbage or wrongly-scaled
  // slot index times the assumed stride lands out in unmapped memory.
  //
  // So the layout is stated rather than inherited: on each card, one block holding
  // [slot][that card's GDN layers], carved by hand. Layers are split 18/18 across the two cards, so
  // the per-slot stride is the per-CARD layer count, not the model's 36.
  int gdn_rank[n_gdn];
  for (int i = 0; i < n_gdn; i++) gdn_rank[i] = gdn_layers_per_card_[gdn_dev_[i]]++;
  void* rec_blk[2] = {nullptr, nullptr};
  void* conv_blk[2] = {nullptr, nullptr};
  for (int d = 0; d < 2; d++) {
    if (!gdn_layers_per_card_[d]) continue;
    const size_t nl = (size_t)gdn_layers_per_card_[d];
    rec_blk[d] = AD(d, nl * (size_t)n_slots_ * rec_bytes);
    conv_blk[d] = AD(d, nl * (size_t)n_slots_ * conv_bytes);
    if (!rec_blk[d] || !conv_blk[d]) {
      fprintf(stderr, "[runner] GDN state allocation failed (card %d)\n", d);
      return false;
    }
  }
  for (int s = 0; s < n_slots_; s++) {
    SlotState& S = slots_[s];
    S.gdn_conv.assign(n_gdn, nullptr);
    S.gdn_rec.assign(n_gdn, nullptr);
    for (int i = 0; i < n_gdn; i++) {
      const int d = gdn_dev_[i];
      const size_t off = (size_t)gdn_rank[i] * (size_t)n_slots_ + (size_t)s;
      S.gdn_conv[i] = (char*)conv_blk[d] + off * conv_bytes;
      S.gdn_rec[i] = (char*)rec_blk[d] + off * rec_bytes;
    }
    S.ple_conv = AD(c.layer_dev(c.ple_layer), ple_conv_bytes);
    if (!S.ple_conv) {
      fprintf(stderr, "[runner] PLE conv allocation failed (slot %d of %d)\n", s, n_slots_);
      return false;
    }
  }
  // Slot 0's buffers are the live ones from here on, so the rest of init() - the speculative
  // snapshots, the carry tap, act0_'s aliases - sees exactly the pointers it always saw.
  bind_slot(0);

  if (c.has_mtp) {
    // Sized for a whole prefill chunk, not for the 2-token draft: the same scratch backs the
    // prefill-time cache fill, which processes max_chunk_ tokens at once. Sizing it for 2 made that
    // fill write far past the buffers (the first attempt produced no output at all).
    // The draft runs on the same card as its weights (cfg.mtp_dev()).
    if (!mtp_scratch_init(mtp_, c, max_chunk_, c.mtp_dev())) {
      fprintf(stderr, "[runner] MTP scratch\n");
      return false;
    }
    mtp_ord_ = n_full;
    // MTP speculation is ON by default, and HELIOS_MTP=0 turns it off.
    //
    // The "20% acceptance" this default was last justified on does not reproduce. Controlled A/B,
    // one binary, 7645-token prompt, 256 greedy tokens, 3 runs per cell (decode tok/s):
    //
    //   off        48.01 / 47.88 / 47.87
    //   K=1        56.25 / 56.68 / 56.51   65.2% of slots, 1.65 tok/step   +17.6%
    //   K=2        51.46 / 51.35 / 51.24   47.3%, 1.95 tok/step             +7.2%
    //   K=3        45.20 / 46.50 / 46.37   38.7%, 2.16 tok/step             -3.4%
    //
    // K=1 is the default depth and the only cell that clearly pays: the depth-1 rate decays
    // geometrically (0.65, then ~0.29, then ~0.22), so a wider verify buys tokens faster than it
    // buys width. Two rollback defects had to be fixed to get here - the PLE's dilated conv and the
    // host n-gram history were both left holding rejected tokens after a partial accept (see
    // ple_replay and spec_verify).
    //
    // ONE CAVEAT, and it is a real behavioural change: a batched verify forward is not bit-identical
    // to the same tokens run one at a time (the GDN chunking itself is bit-exact -
    // test_gdn_chunked_parity - but the attention and MoE reductions at width K+1 associate
    // differently), so greedy output drifts off the non-speculative stream after ~35 tokens at 8k.
    // It is fully deterministic run to run. HELIOS_MTP=0 restores the sequential path and the
    // reference digest exactly.
    const char* me = getenv("HELIOS_MTP");
    mtp_on_ = me ? atoi(me) != 0 : true;
    if (mtp_on_)
      fprintf(stderr, "[runner] MTP speculation ON (HELIOS_MTP=0 to disable)\n");
    else
      fprintf(stderr, "[runner] MTP speculation OFF (HELIOS_MTP=0; greedy output then matches the\n"
                      "         sequential path bit for bit)\n");
    if (const char* e = getenv("HELIOS_MTP_CTX_LIMIT")) mtp_ctx_limit_ = atoi(e);
  }

  // GDN snapshot for batched-verification rollback. The recurrent state is the ONLY thing a partial
  // accept invalidates: attention KV rows past the accepted prefix are simply overwritten by the next
  // token and never read, but conv_state/rec_state are a running recurrence that cannot be rewound by
  // overwriting. Snapshotting them costs one 153 MB device-to-device copy (~0.17 ms at 900 GB/s)
  // against a 21.9 ms trunk forward, which is why the GDN-only replay strategy is affordable and the
  // retry-the-whole-prefix strategy is not.
  //
  // The snapshot is NOT per-slot: it is scratch for one in-flight speculative batch, and only one
  // forward is ever in flight, so one set serves every slot.
  if (n_gdn > 0) {
    gdn_rec_bytes_ = rec_bytes;
    gdn_conv_bytes_ = conv_bytes;
    gdn_conv_snap_.assign(n_gdn, nullptr);
    gdn_rec_snap_.assign(n_gdn, nullptr);
    for (int i = 0; i < n_gdn; i++) {
      // The snapshot is a device-to-device copy of that layer's state, so it belongs on the same
      // card as the state it copies (a cross-device D2D would be a PCIe round trip per step).
      gdn_conv_snap_[i] = AD(gdn_dev_[i], conv_bytes);
      gdn_rec_snap_[i] = AD(gdn_dev_[i], rec_bytes);
    }
    // Room for the widest speculative batch (kMaxSpecDrafts drafts + the committed token) of
    // sublayer inputs, so a replay can re-run the GDN without re-running the mixers that produced
    // them. Indexed by the global gdn_ord, hence n_gdn rows.
    gdn_sub_stride_ = (size_t)(kMaxSpecDrafts + 1) * c.hidden;
    gdn_sub_snap_ = (half*)A0((size_t)n_gdn * gdn_sub_stride_ * 2);
  }
  // The MTP chain seed is per-slot state (it is the tap at THAT conversation's last committed
  // position), so it is allocated per slot. Slot 0 gets the card-0 pointer act0_ has always used;
  // with the pipeline on, the last layer is on card 1 and the chain seed it actually reads is
  // act1_'s, so that half is per-slot too.
  for (int s = 0; s < n_slots_; s++) {
    SlotState& S = slots_[s];
    S.carry_tap = (float*)A0((size_t)H * D * 4);
    if (c.pipeline) S.carry_tap1 = (float*)AD(1, (size_t)H * D * 4);
    if (!S.carry_tap || (c.pipeline && !S.carry_tap1)) {
      fprintf(stderr, "[runner] MTP carry tap allocation failed (slot %d of %d)\n", s, n_slots_);
      return false;
    }
  }
  carry_tap_ = slots_[0].carry_tap;
  // act0_ aliases the single activation set the layer loop has always used (assigned last because
  // carry_tap_ is allocated last), so the loop can be written once against a card: with the pipeline
  // off, dev is 0 for every layer and these are exactly the addresses the single-set path used.
  act0_.streams = streams_; act0_.embed16 = embed16_; act0_.mixed = mixed_; act0_.sub_in = sub_in_;
  act0_.sub_out = sub_out_; act0_.post = post_; act0_.ids = ids_dev_; act0_.tap = tap_;
  act0_.carry_tap = carry_tap_; act0_.head_had = head_had_;
  // ALLOCATED LAST, after every other workspace: A() is a bump allocator, so inserting a buffer
  // earlier in this sequence shifts every later address and has cost real throughput before.
  if (c.has_mtp) {
    const size_t row_b = (size_t)c.attn_kv_heads * c.attn_head_dim * 2 * sizeof(short);
    for (int c2 = 0; c2 < 2; c2++) mtp_kv_snap_[c2] = A0(row_b * 4);
  }
  printf("[runner] ctx=%d chunk=%d | %d full-attn layers x %.0f MB KV = %.2f GB | %d GDN layers\n",
         ctx_cap, max_chunk, n_full, kv_slot_bytes_ / 1048576.0,
         (double)kv_total * 2 * n_full / 1073741824.0, n_gdn);
  if (n_slots_ > 1)
    printf("[runner] %d sequence slots: %d rows of KV each (%d of %d), %.0f MB of resident "
           "recurrent state per slot\n",
           n_slots_, slot_ctx_, slot_ctx_, ctx_cap,
           (double)(gdn_rec_bytes_ * (size_t)n_gdn + gdn_conv_bytes_ * (size_t)n_gdn) / 1048576.0);
  if (kv_bits) {
    const size_t row_b16 = (size_t)kv_token_dim * 2;
    const size_t fp16_cache = (size_t)ctx_cap * row_b16 * 2 * n_full;
    const size_t cq_cache = kv_total * 2 * n_full;
    printf("[runner] HELIOS_KV_QUANT=%d: KV cache is the exllamav3 CacheLayer_quant layout, %d B "
           "per token per tensor against %d fp16 (%.2fx)\n",
           kv_bits, (int)attn::kvq_row_bytes(kv_token_dim, kv_bits), kv_token_dim * 2,
           (double)(kv_token_dim * 2) / (double)attn::kvq_row_bytes(kv_token_dim, kv_bits));
    // The read path stages one fp16 K and one fp16 V per device (attn_layer.cu). It is reported here
    // because it is part of what the flag costs in VRAM, and the net figure is the one that decides
    // whether a bigger prefill chunk fits.
    printf("[runner] HELIOS_KV_QUANT=%d: KV %.2f GB -> %.2f GB, minus %.0f MB of fp16 dequant "
           "staging (one K + one V per card) = %.2f GB net (%.2f GB freed)\n",
           kv_bits, (double)fp16_cache / (1 << 30), (double)cq_cache / (1 << 30),
           (double)((size_t)ctx_cap * row_b16 * 2) / 1048576.0,
           (double)(cq_cache + (size_t)ctx_cap * row_b16 * 2) / (1 << 30),
           (double)(fp16_cache - cq_cache - (size_t)ctx_cap * row_b16 * 2) / (1 << 30));
  }
  // ---- HELIOS_PIPELINE: report the placement and the schedule ----
  //
  // The forward itself is run_chunk_pipeline: the chunk is cut into `micro_` micro-batches, card 0
  // runs each through [0, split) on its own stream, and card 1 runs micro-batch i through
  // [split, n_layers) while card 0 is already on micro-batch i+1. One hyper-connection handoff per
  // micro-batch crosses the boundary; the final mixer and lm_head are on the last layer's card.
  if (c.pipeline) {
    size_t used[2] = {0, 0}, free_b[2] = {0, 0}, total_b[2] = {0, 0};
    for (int d = 0; d < 2; d++) {
      size_t fr = 0, tot = 0;
      cudaSetDevice(Engine::instance().gpu(d).phys_idx());
      cudaMemGetInfo(&fr, &tot);
      used[d] = tot - fr; free_b[d] = fr; total_b[d] = tot;
    }
    cudaSetDevice(Engine::instance().gpu(0).phys_idx());
    printf("[pipeline] scratch/state/activation placement (ctx=%d chunk=%d):\n", ctx_cap, max_chunk);
    printf("[pipeline]   card0: attn/gdn/moe/ple scratch, activation set, %d full-attn KV pairs "
           "(%.2f GB), %d GDN recurrences + snapshots, PLE conv\n",
           n_full0, (double)kv_total * 2 * n_full0 / (1 << 30), n_gdn0);
    printf("[pipeline]   card1: attn/gdn/moe scratch, own activation set, %d full-attn KV pairs "
           "(%.2f GB), %d GDN recurrences%s\n",
           n_full1, (double)kv_total * 2 * n_full1 / (1 << 30), n_gdn1,
           c.layer_dev(c.ple_layer) == 1 ? " + PLE scratch/conv" : "");
    // Whether the re-cut FITS is the first question this mode has to answer, so free VRAM is
    // reported per card rather than left to the caller to infer from the byte totals.
    printf("[pipeline]   VRAM: card0 %.2f/%.2f GB used (%.2f free) | card1 %.2f/%.2f GB used "
           "(%.2f free)\n",
           (double)used[0] / (1 << 30), (double)total_b[0] / (1 << 30), (double)free_b[0] / (1 << 30),
           (double)used[1] / (1 << 30), (double)total_b[1] / (1 << 30), (double)free_b[1] / (1 << 30));
    printf("[pipeline] schedule: split=%d micro=%d | card0 layers [0,%d) -> handoff %.1f KB x%d -> "
           "card1 layers [%d,%d) -> mixer + lm_head on card%d\n",
           c.split, micro_, c.split,
           (double)((size_t)((max_chunk + micro_ - 1) / micro_) * H * D * 4) / 1024.0, micro_,
           c.split, c.n_layers, last_dev_);
    fprintf(stderr,
            "[pipeline] HELIOS_PIPELINE=1: layer-pipelined forward ON (embed->card%d, lm_head+MTP->"
            "card%d, %d micro-batches per chunk; HELIOS_PIPE_MICRO to change).\n",
            c.layer_dev(0), last_dev_, micro_);
  }
  // ---- HELIOS_DECODE_GRAPH: resolve the flag once, and say plainly if it is refused ----
  //
  // The decision is made HERE rather than per layer so the hot loop tests a bool and a refusal is
  // reported at startup with the variable that caused it. A graph silently not being taken is the
  // one failure mode of this feature that would otherwise look like "the flag does nothing".
  if (decode_graph_on()) {
    // A graph bakes in the absolute device addresses live at capture time. With more than one slot
    // the recurrent state and the KV base BOTH change when the conversation changes, so a replay
    // would run against whichever slot happened to be bound at capture time - the same class of
    // stale-pointer bug that already made this unsafe across requests, now reachable within one.
    if (n_slots_ > 1) {
      fprintf(stderr,
              "[graph] HELIOS_DECODE_GRAPH=1 refused: HELIOS_SEQUENCES=%d binds different state "
              "pointers per conversation and a captured graph cannot follow a rebind. Graph capture "
              "is OFF for this run.\n", n_slots_);
    } else {
      const char* why = decode_graph_refusal();
      if (why) {
        fprintf(stderr,
                "[graph] HELIOS_DECODE_GRAPH=1 but %s is set: that reads back or syncs inside the "
                "layer body, which a capture cannot hold. Graph capture is OFF for this run.\n", why);
      } else {
        dgraph_ok_ = true;
        fprintf(stderr,
                "[graph] HELIOS_DECODE_GRAPH=1: capturing one graph per (site, card, layer, width).\n"
                "        GDN layers capture whole; full-attention layers capture from the first "
                "gr_apply (the attention kernels need pos0 and stay eager). PLE stays eager (its "
                "n-grams are hashed on the host). Keyed per width, so n=1 and a K+1 verify get their "
                "own.\n");
      }
    }
  }
  // Cross-request prefix cache. Off unless asked for: it costs host RAM and a D2H per prefill chunk,
  // and its whole value is that a LATER request shares a prefix with this one.
  prefix_config(max_chunk);
  // Zero the recurrent state. With the pipeline on, half of it lives on card 1, so reset() is
  // device-aware rather than addressing whatever the current device happens to be. EVERY slot is
  // zeroed, not just the active one: a slot must start from a defined state whatever order the
  // conversations are first touched in, or "which conversation was served first" would change an
  // answer.
  for (int s = 0; s < n_slots_; s++) {
    const int keep = active_slot_;
    bind_slot(s);
    reset();
    slots_[s].hist.assign(c.ngram_size - 1, c.ngram_eos);
    slots_[s].hist_tokens.clear();
    bind_slot(keep);
  }
  return true;
}

// Resolve HELIOS_SEQUENCES and clamp it against what this machine actually has free. The policy
// itself is in seq.hpp with its own unit test; this is the part that needs a GPU to evaluate.
//
// The clamp is measured, not assumed, and it is reported rather than silently applied: at the
// shipped 262144 context card 1 is down to ~0.6 GB free and a slot costs it ~58 MB, so a request
// for 8 slots is genuinely a different number from 4 on this box and would be on another. The
// alternative - allocating and hoping - fails partway through loading 31 GB of experts.
//
// The reserve is not decoration. A slot's state is allocated LAST in init, so what "free" means
// here is measured after everything else, and the reserve is what is left for the next forward's
// temporaries; an engine that allocates every free byte and then cannot launch is a worse outcome
// than one that serves 5 conversations instead of 6.
bool Runner::seq_config(int ctx_cap, size_t kv_row_b) {
  const Config& c = m_->cfg;
  const int want = seq_requested_;
  if (want <= 1) {
    // The shipped path, and it is left EXACTLY as it was: one slot, the whole KV cache, and no
    // second state allocation. Nothing below this branch runs, so no address and no byte of the
    // single-sequence engine moves.
    n_slots_ = 1;
    slot_ctx_ = ctx_cap;
    kv_slot_bytes_ = kv_row_b * (size_t)ctx_cap;
    slots_.assign(1, SlotState{});
    active_slot_ = 0;
    return true;
  }
  {
    // What one more slot costs, per card. The KV cache is NOT in this: it is already allocated in
    // full below and a slot only takes a base-pointer offset into it, so N slots divide the same
    // 6 GB rather than multiplying it. Counting it here is what would make a perfectly affordable
    // N look impossible.
    SeqCost cost;
    const size_t rec = (size_t)c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim * 4;
    const size_t conv = (size_t)c.gdn_qkv_out * c.gdn_conv_k * 2;
    for (int l = 0; l < c.n_layers; l++)
      if (!c.full_attn[l]) cost.add(c.layer_dev(l), conv + rec);
    cost.add(c.layer_dev(c.ple_layer), (size_t)c.hc_dim * ((4 - 1) * c.ngram_size) * 2);
    // The MTP chain seed is per-slot state, on each card that the forward can read it from.
    const size_t tap = (size_t)c.hc_mult * c.hidden * 4;
    cost.add(0, tap);
    if (c.pipeline) cost.add(1, tap);

    // Measured where the engine actually is: after the weights, both scratch sets and the whole
    // KV cache are in, so these are the bytes a slot would really be asking for.
    size_t free_b[N_GPU] = {0, 0}, total_b[N_GPU] = {0, 0};
    for (int d = 0; d < N_GPU; d++) {
      HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
      HELIOS_CUDA_CHECK(cudaMemGetInfo(&free_b[d], &total_b[d]));
    }
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
    // 320 MB per card: measured headroom for a decode step's temporaries on top of the state, not
    // a round number chosen to make the answer come out.
    const size_t kReserve = 320ull * 1024 * 1024;
    const int fit = seq_fit_slots(free_b[0], cost.per_card[0], free_b[1], cost.per_card[1], kReserve);
    // A slot with a 200-token context is not a conversation, so the context floor binds too: at
    // 262144 the floor of 8192 allows 32 slots, so VRAM is what actually decides at this context.
    const int kMinSlotCtx = 8192;
    const int n = seq_clamp_slots(want, fit, ctx_cap, kMinSlotCtx);
    slot_ctx_ = seq_slot_ctx(ctx_cap, n);
    fprintf(stderr,
            "[seq] HELIOS_SEQUENCES=%d -> %d slot%s%s. Per slot: %d of %d context rows and %.0f MB "
            "resident recurrent state (card0 %.0f MB, card1 %.0f MB). The KV cache is NOT "
            "duplicated - each slot takes a base-pointer offset into the same allocation.\n",
            want, n, n == 1 ? "" : "s", n < want ? " - CLAMPED" : "", slot_ctx_, ctx_cap,
            (double)cost.total() / 1048576.0, (double)cost.per_card[0] / 1048576.0,
            (double)cost.per_card[1] / 1048576.0);
    if (n < want) {
      // Say WHICH limit bound, because "clamped" with no reason is the same as a silent failure
      // with better manners.
      const char* why = n < seq_clamp_slots(want, want, ctx_cap, kMinSlotCtx) ? "free VRAM"
                                                                           : "the context floor";
      fprintf(stderr,
              "[seq]   clamped: %d slots asked, %d fit. Card free at this point: card0 %.0f MB, "
              "card1 %.0f MB; a slot costs %.0f MB on card0 and %.0f MB on card1, and %d is held "
              "back for decode temporaries. Binding limit: %s.\n",
              want, n, (double)free_b[0] / 1048576.0, (double)free_b[1] / 1048576.0,
              (double)cost.per_card[0] / 1048576.0, (double)cost.per_card[1] / 1048576.0,
              (int)(kReserve / 1048576), why);
      fprintf(stderr,
              "[seq]   per-slot context is %d tokens. Lower --cap or HELIOS_SEQUENCES for full "
              "context per conversation; the engine cannot give a slot more rows than the KV "
              "cache is divided into.\n", slot_ctx_);
    }
    // Only claim persistence when it is actually true. A slot preserves its conversation across
    // requests ONLY through the prefix cache; without it every request calls reset() and the slots
    // are N independent runners, so telling the user their conversations are "ALIVE" would be
    // stating something false about the default configuration.
    const bool pfx_on = getenv("HELIOS_PREFIX_CACHE") != nullptr && atoi(getenv("HELIOS_PREFIX_CACHE")) != 0;
    if (pfx_on) {
        fprintf(stderr,
                "[seq]   interleaved, not simultaneous: one forward at a time on a shared activation "
                "set. With HELIOS_PREFIX_CACHE=1, N slots keep N conversations' KV and recurrent state "
                "ALIVE and a client can pin a follow-up turn to its own slot; they do not run two "
                "generations at once.\n");
    } else {
        fprintf(stderr,
                "[seq]   interleaved, not simultaneous: one forward at a time on a shared activation "
                "set - they do not run two generations at once.\n"
                "[seq]   WARNING: conversations are NOT kept alive with this configuration. Every "
                "request resets the runner, so the N slots are N independent copies and pinning a "
                "turn to a slot does not continue a conversation.\n"
                "[seq]   Set HELIOS_PREFIX_CACHE=1 as well to make slots persistent. That path is "
                "opt-in because it changes the token stream of a request that reuses nothing; for "
                "multi-turn use that does not matter, which is why it is the intended companion here.\n");
    }
    n_slots_ = n;
    kv_slot_bytes_ = kv_row_b * (size_t)slot_ctx_;
    slots_.assign((size_t)n, SlotState{});
    active_slot_ = 0;
    // The quantized cache's scale array starts after a slot's codes, and the same row count drives
    // both the write and the dequant read, each of which is handed a slot's base pointer. It is set
    // per slot rather than globally at ctx_cap because a global ctx_cap would put every slot's
    // scales at the END of the whole tensor - i.e. inside the last slot's region.
    attn::kvq_ctx_rows_ref() = slot_ctx_;
    return true;
  }
}

// Point every piece of live per-sequence state at slot `s`, and park the outgoing slot's state
// on the way out.
//
// There is no device copy here, and that is the whole reason the per-slot state is resident rather
// than spilled to a host ring the way the prefix cache's captures are. A switch is a handful of
// pointer assignments and two vector swaps; it cannot be interrupted, cannot half-complete, and
// costs the same whether it happens between two 16k requests or two 8-token ones. A save/restore
// design would pay ~116 MB of PCIe traffic per switch and would have a failure mode - a switch that
// dies halfway leaves a conversation's state in pieces - that a rebind simply does not have.
//
// SAVE and LOAD are separate on purpose. A single swap-based routine is symmetric only if the live
// members and slot 0 are already paired, and at init they are not: the live vectors are empty and
// the real pointers live in slots_[0], so the first bind would swap empty against real and a second
// call would swap them straight back. That is not hypothetical - it left the live state EMPTY on
// the third request and segfaulted inside layer_range. Splitting the two directions makes each one
// a plain assignment, which cannot be un-done by calling it twice.
void Runner::save_slot(int cur) {
  SlotState& S = slots_[(size_t)cur];
  S.gdn_conv = gdn_conv_;              // 36 pointers; a copy, not a swap
  S.gdn_rec = gdn_rec_;
  S.ple_conv = ple_conv_;
  S.carry_tap = carry_tap_;
  S.carry_tap1 = act1_.carry_tap;
  S.pos = pos_;
  S.last_committed = last_committed_;
  S.next_tok = next_tok_;
  S.pfx_next = pfx_next_;
  S.carry_valid = carry_valid_;
  // The host vectors are swapped rather than copied: hist_tokens_ is the whole conversation and
  // hist_ is the n-gram window, and copying them per request would be the only O(ctx) work in the
  // switch. A swap leaves the slot holding the previous occupant's copy, ready to be taken back.
  S.hist_tokens.swap(hist_tokens_);
  S.hist.swap(hist_);
  S.pfx_pos.swap(pfx_pos_);
}

void Runner::load_slot(int s) {
  SlotState& S = slots_[(size_t)s];
  gdn_conv_ = S.gdn_conv;
  gdn_rec_ = S.gdn_rec;
  hist_tokens_ = S.hist_tokens;
  hist_ = S.hist;
  pfx_pos_ = S.pfx_pos;
  ple_conv_ = S.ple_conv;
  carry_tap_ = S.carry_tap;
  act0_.carry_tap = S.carry_tap;
  // act1_.carry_tap is null unless the pipeline is on, and the pipeline's last layer is on card 1,
  // so the chain seed the forward actually reads is the card-1 half. Assigning null over null on the
  // shipped single-card path is a no-op rather than a fault.
  if (act1_.carry_tap || S.carry_tap1) act1_.carry_tap = S.carry_tap1;
  pos_ = S.pos;
  last_committed_ = S.last_committed;
  next_tok_ = S.next_tok;
  pfx_next_ = S.pfx_next;
  carry_valid_ = S.carry_valid;
  active_slot_ = s;
  // The prefix ring is shared host memory, so a capture is only meaningful for the conversation
  // that took it. A slot's captures are its own capture-index range; without this a resume could
  // restore one conversation's recurrent state on top of another conversation's KV rows, which is
  // exactly the silent-wrong-answer the whole prefix cache is unit-tested against.
  if (pfx_on_ && (int)pfx_pos_.size() != pfx_slots_) pfx_pos_.assign((size_t)pfx_slots_, -1);
}

void Runner::bind_slot(int s) {
  if (slots_.empty()) return;
  if (s < 0 || s >= (int)slots_.size()) s = 0;
  // Re-binding the slot that is already live is not a no-op, it is the ONLY way init() populates the
  // live members from slot 0 - they start empty - so the same-slot case still loads. It does not
  // save, because saving a slot onto itself through a swap pair would swap the live vectors with
  // themselves and lose the contents.
  if (s != active_slot_ && slots_[(size_t)active_slot_].gdn_conv.size() == gdn_conv_.size() &&
      !gdn_conv_.empty())
    save_slot(active_slot_);
  load_slot(s);
}

int Runner::acquire_slot(int want) {
  const int s = seq_pick(want, n_slots_, slot_rr_);
  if (s < 0) return -1;
  if (s != active_slot_ || slots_.empty()) bind_slot(s);
  return s;
}

// Every GDN layer's state lives on the card that owns the layer (gdn_dev_), so the snapshot, the
// restore and the replay are all issued per layer on that card's own stream. With the pipeline off
// gdn_dev_ is all zeros, so this is the single card-0 stream the code always used.
void Runner::gdn_snapshot(int n) {
  if (gdn_rec_snap_.empty()) return;
  for (size_t i = 0; i < gdn_rec_.size(); i++) {
    const int d = gdn_dev_[i];
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
    cudaStream_t sd = Engine::instance().gpu(d).stream(0);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(gdn_conv_snap_[i], gdn_conv_[i], gdn_conv_bytes_,
                                      cudaMemcpyDeviceToDevice, sd));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(gdn_rec_snap_[i], gdn_rec_[i], gdn_rec_bytes_,
                                      cudaMemcpyDeviceToDevice, sd));
  }
  // Saves the CURRENT state, which must be the state BEFORE the speculative forward - restoring
  // to a post-batch snapshot and replaying those positions again would apply them twice. The
  // sublayer inputs are captured separately, inside run_chunk, because they do not exist until it
  // runs. Order at the call site: gdn_snapshot -> run_chunk(capture) -> [accept | restore+replay].
  (void)n;
}

void Runner::gdn_restore() {
  if (gdn_rec_snap_.empty()) return;
  for (size_t i = 0; i < gdn_rec_.size(); i++) {
    const int d = gdn_dev_[i];
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
    cudaStream_t sd = Engine::instance().gpu(d).stream(0);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(gdn_conv_[i], gdn_conv_snap_[i], gdn_conv_bytes_,
                                      cudaMemcpyDeviceToDevice, sd));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(gdn_rec_[i], gdn_rec_snap_[i], gdn_rec_bytes_,
                                      cudaMemcpyDeviceToDevice, sd));
  }
}

// Re-run ONLY the GDN recurrence over the first `n` snapshotted positions, after restoring the
// pre-batch state. This is what makes a partial accept cheap: the trunk forward is not repeated
// (21.9 ms/token), just the 3.65 ms of recurrence that the rejected positions actually advanced.
void Runner::gdn_replay(int n) {
  if (gdn_rec_snap_.empty() || n <= 0) return;
  const Config& c = m_->cfg;
  gdn_restore();
  for (int l = 0; l < c.n_layers; l++) {
    const Layer& L = m_->layers[l];
    if (L.full) continue;
    const int d = c.layer_dev(l);
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
    const half* sub = d == 0
        ? gdn_sub_snap_ + (size_t)L.gdn_ord * gdn_sub_stride_
        : gdn_sub_snap1_ + (size_t)L.gdn_ord * gdn_sub_stride_;
    gdn_layer(L.gdn, c, d == 0 ? gdn_ : gdn1_, sub, d == 0 ? sub_out_ : act1_.sub_out,
              gdn_conv_[L.gdn_ord], (float*)gdn_rec_[L.gdn_ord], n,
              Engine::instance().gpu(d).stream(0));
  }
}

// Rewind the PLE's dilated conv over the accepted prefix. Same shape as gdn_replay: restore the
// pre-batch 9-column state, re-run the conv for the first `n` of the batch's rows, and throw the
// output away - the original forward already injected the correct values for those rows into the
// stream stack, and the rejected rows' injections are never read (nothing addresses a row past the
// committed position). The conv input is the SNAPSHOT taken inside ple_layer, so the replay sees
// the same activations the forward did rather than recomputing them.
void Runner::ple_replay(int n) {
  if (n <= 0) return;
  const Config& c = m_->cfg;
  const Layer& L = m_->layers[c.ple_layer];
  if (!L.has_ple) return;
  const int d = c.layer_dev(c.ple_layer);
  PleScratch& ps = d == 0 ? ple_ : ple1_;
  if (!ps.conv_in_snap || !ps.conv_state_snap) return;
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
  cudaStream_t s = Engine::instance().gpu(d).stream(0);
  const size_t cs = (size_t)c.hc_dim * (4 - 1) * c.ngram_size * 2;
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(ple_conv_, ps.conv_state_snap, cs, cudaMemcpyDeviceToDevice, s));
  ple_dilated_conv(ps.conv_in_snap, ple_conv_, L.ple.conv, ps.conv_out, n, c.hc_dim, c.ngram_size, s);
  HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// Self-test for the rollback machinery: run `width` positions, snapshot, advance by one more, then
// replay the snapshotted prefix. Replaying the SAME positions from the SAME pre-state must land on
// the same recurrent state, so a fingerprint of the GDN state before and after must match bitwise.
// This is the property batched verification depends on; without it a partial accept silently
// corrupts the recurrence and every later token diverges.
bool Runner::gdn_replay_selftest(int width, unsigned* out_a, unsigned* out_b) {
  if (gdn_rec_snap_.empty() || width < 1) return false;
  const Config& c = m_->cfg;
  // Each layer's recurrence is on its own card, so the fingerprint is read there.
  auto fp = [&](unsigned* h) {
    unsigned v = 2166136261u;
    for (size_t i = 0; i < gdn_rec_.size(); i++) {
      HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(gdn_dev_[i]).phys_idx()));
      cudaStream_t sd = Engine::instance().gpu(gdn_dev_[i]).stream(0);
      std::vector<char> buf(gdn_rec_bytes_);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(buf.data(), gdn_rec_[i], gdn_rec_bytes_,
                                        cudaMemcpyDeviceToHost, sd));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(sd));
      for (char ch : buf) { v ^= (unsigned char)ch; v *= 16777619u; }
    }
    *h = v;
  };
  // The real rollback sequence: save the pre-batch state, run the speculative batch (which also
  // captures each GDN layer's sub_in_), then advance further, then restore + replay and require the
  // recurrence to land back on the post-batch state exactly.
  gdn_snapshot(width);
  std::vector<int> ids(width, 100);
  gdn_capture_ = true;
  run_chunk(ids, pos_);
  gdn_capture_ = false;
  pos_ += width;
  fp(out_a);
  run_chunk({100}, pos_);
  pos_ += 1;
  gdn_replay(width);
  fp(out_b);
  (void)c;
  return *out_a == *out_b;
}

// HELIOS_SPEC_TRACE=1: one fingerprint per committed position per recurrent state. Speculative
// decoding must leave the trunk in EXACTLY the state plain decode would, so diffing these lines
// between a plain run and a speculative one names the first position where a rollback failed and
// which state failed to roll back. Each state gets an exact hash of its raw bytes and a COARSE hash
// over the top 16 bits of every element, so a last-bit float difference (chunked vs sequential
// accumulation order) is separable from a state that is genuinely wrong. Diagnostic only.
void Runner::trace_state(const char* tag, int pos) {
  if (!getenv("HELIOS_SPEC_TRACE")) return;
  const Config& c = m_->cfg;
  struct FP { unsigned exact = 0, coarse = 0; };
  auto fp_of = [&](const void* p, int dev, size_t bytes, size_t esz) -> FP {
    FP f;
    if (!p || !bytes) return f;
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(dev).phys_idx()));
    cudaStream_t sd = Engine::instance().gpu(dev).stream(0);
    std::vector<char> buf(bytes);
    HELIOS_CUDA_CHECK(cudaMemcpy(buf.data(), p, bytes, cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(sd));
    for (char ch : buf) { f.exact ^= (unsigned char)ch; f.exact *= 16777619u; }
    // fp16 states (the convs): the top 10 bits of the mantissa-free word are sign+exp+2 bits.
    const size_t n16 = bytes / 2;
    const uint16_t* h = (const uint16_t*)buf.data();
    for (size_t i = 0; i < n16; i++) { f.coarse ^= (unsigned)(h[i] >> 6); f.coarse *= 16777619u; }
    if (esz == 4) {
      const size_t n32 = bytes / 4;
      const uint32_t* w = (const uint32_t*)buf.data();
      for (size_t i = 0; i < n32; i++) { f.coarse ^= (unsigned)(w[i] >> 16); f.coarse *= 16777619u; }
    }
    return f;
  };
  FP conv, rec, ple;
  for (size_t i = 0; i < gdn_rec_.size(); i++) {
    FP a = fp_of(gdn_conv_[i], gdn_dev_[i], gdn_conv_bytes_, 2);
    FP b = fp_of(gdn_rec_[i], gdn_dev_[i], gdn_rec_bytes_, 4);
    conv.exact ^= a.exact; conv.coarse ^= a.coarse;
    rec.exact ^= b.exact; rec.coarse ^= b.coarse;
  }
  ple = fp_of(ple_conv_, c.layer_dev(c.ple_layer),
              (size_t)c.hc_dim * ((4 - 1) * c.ngram_size) * 2, 2);
  fprintf(stderr, "[trace] %s pos=%d | gdnconv %08x/%08x | gdnrec %08x/%08x | ple %08x/%08x\n",
          tag, pos, conv.exact, conv.coarse, rec.exact, rec.coarse, ple.exact, ple.coarse);
  // The host n-gram context is a rollback target too, and it is not a device buffer, so it gets its
  // own line rather than a hash of the three.
  fprintf(stderr, "[trace] %s pos=%d ngram=[", tag, pos);
  for (size_t i = 0; i < hist_.size(); i++)
    fprintf(stderr, "%s%lld", i ? " " : "", (long long)hist_[i]);
  fprintf(stderr, "]\n");
  // RMS alongside the hash: a rollback that is merely off in the last bits has a matching RMS, a
  // rollback that is actually wrong does not.
  {
    double ss = 0.0;
    size_t cnt = 0;
    for (size_t i = 0; i < gdn_rec_.size(); i++) {
      HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(gdn_dev_[i]).phys_idx()));
      std::vector<float> buf(gdn_rec_bytes_ / 4);
      HELIOS_CUDA_CHECK(cudaMemcpy(buf.data(), gdn_rec_[i], gdn_rec_bytes_, cudaMemcpyDeviceToHost));
      for (float v : buf) ss += (double)v * (double)v;
      cnt += buf.size();
    }
    fprintf(stderr, "[trace] %s pos=%d recrms=%.9e\n", tag, pos, cnt ? sqrt(ss / cnt) : 0.0);
  }
}



void Runner::reset() {
  pos_ = 0;
  const Config& c = m_->cfg;
  const size_t rec_bytes = (size_t)c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim * 4;
  const size_t conv_bytes = (size_t)c.gdn_qkv_out * c.gdn_conv_k * 2;
  for (size_t i = 0; i < gdn_rec_.size(); i++) {
    // Each layer's recurrence is on its own card, and cudaMemset addresses the CURRENT device, so
    // the device is selected per layer. Pipeline off this is card 0 for all of them, as before.
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(gdn_dev_[i]).phys_idx()));
    HELIOS_CUDA_CHECK(cudaMemset(gdn_rec_[i], 0, rec_bytes));
    HELIOS_CUDA_CHECK(cudaMemset(gdn_conv_[i], 0, conv_bytes));
  }
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.layer_dev(c.ple_layer)).phys_idx()));
  HELIOS_CUDA_CHECK(cudaMemset(ple_conv_, 0, (size_t)c.hc_dim * ((4 - 1) * c.ngram_size) * 2));
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
  // The KV cache needs no clearing: every read is bounded by the query position. The PLE's carried
  // context ids are eos at sequence start, which is what the reference pads with.
  hist_.assign(c.ngram_size - 1, c.ngram_eos);
  tm_ = Timings{};
}

// The head runs on the card that owns the LAST layer, because that is where lm_head lives (and the
// combine-less mixer before it). With the pipeline off that is card 0 and the arguments below are
// the ones this function has always used.
//
// `head_rows` is how many of the chunk's LAST positions to project, one per logits row, row 0 being
// the last of them. It is 1 everywhere except a speculative verify, which needs one row per position
// of the batch so the drafts can be checked against the trunk's own predictions. Making it an
// argument rather than a member is the fix for the K>1 breakage: the CAPACITY of logits_ and the
// WIDTH of a projection are different quantities, and deriving the width from the capacity made
// every prefill sample `head_rows` positions early.
void Runner::final_head(int n, int head_rows) {
  const Config& c = m_->cfg;
  const int d = last_dev_;
  cudaStream_t s = Engine::instance().gpu(d).stream(0);
  float* logits = d == 0 ? logits_ : logits1_;
  const half* sub_in = (d == 0 ? sub_in_ : act1_.sub_in);
  void* had = d == 0 ? (void*)head_had_ : act1_.head_had;
  // No final norm: the combine-less mixer already produced the hidden state.
  exl3::GroupWords hw{(const uint16_t*)m_->lm_head.trellis, m_->lm_head.suh,
                      m_->lm_head.svh, m_->lm_head.mul1};
  // Batched verification needs a logits row per position, not just the last. Projecting them in
  // ONE gemm (rather than one call each) is what makes this affordable: the 5-bit lm_head is 163 MB
  // of weights, and re-reading it per row would cost more than the speculation saves. `rows` is
  // clamped to what actually happened in the chunk, so a short tail never projects past the buffer.
  // With head_rows == 1 this is exactly the long-standing single-row call: the LAST position into
  // row 0, so `next_token()` means the same thing after a 32-token prefill chunk as after a
  // 1-token decode.
  const int rows = n < head_rows ? n : head_rows;
  exl3::gemm(logits, sub_in + (size_t)(n - rows) * c.hidden, hw, rows, c.vocab, c.hidden,
             m_->lm_head.K, /*y_fp32=*/true, s, had);
  head_rows_written_ = rows;
}

namespace {
// Phase ids for the profiler's event ring. The mark is recorded AFTER its phase, so the delta between
// two consecutive marks is the time of the phase that ended at the later one.
enum ProfPhase { P_EMB, P_PLE, P_AMIX, P_ATTN, P_GDN, P_APPLY, P_MMIX, P_MOE, P_FINAL, P_N };
constexpr int kProfEvents = 512;   // 5-6 marks per layer over 48 layers, plus PLE and the tail

// Phase profiler: HELIOS_PROF=1 records one cudaEvent per phase boundary and reads the deltas after
// a single sync per device.
//
// It used to drain the stream at every phase and take wall-clock deltas. That does attribute cost
// rather than overlap, but it also serialises the pipeline and folds sync latency into every bucket,
// which made the buckets unusable as absolute numbers: `grouped` alone read 8.15 ms/layer, i.e.
// 938 tok/s of MoE for a prefill measured at 249 tok/s end to end. cudaEvent deltas on an
// already-drained stream measure GPU time per phase without the sync cost.
//
// ONE RING PER DEVICE. cudaEventElapsedTime is only meaningful between two events of the same
// device, and in pipeline mode the two stages interleave on two cards' streams - a single ring
// would pair a card-0 mark with a card-1 mark and report the gap between unrelated work. Summing
// both rings still answers "how much GPU time went into MoE", now across both cards.
struct PhaseProf {
  bool on = false;
  cudaEvent_t ev[2][kProfEvents] = {};
  int ph[2][kProfEvents] = {};
  int n[2] = {0, 0};
  bool ready = false;
  void ensure() {
    if (!on || ready) return;
    // Timing MUST stay enabled: cudaEventElapsedTime fails on events created with
    // cudaEventDisableTiming, and the failure is silent here (every delta skipped, all phases 0).
    // Each ring's events are created in THAT card's context: an event belongs to the device that
    // created it, and recording one on another device's stream is cudaErrorInvalidResourceHandle.
    for (int d = 0; d < 2; d++) {
      HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
      for (int i = 0; i < kProfEvents; i++) cudaEventCreate(&ev[d][i]);
    }
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
    ready = true;
  }
  void mark(int phase, cudaStream_t s) {
    if (!on) return;
    // Indexed by the LOGICAL card whose stream this is - the physical indices the engine ranked
    // need not be 0 and 1 in that order, so the physical device number is not the ring index.
    const int d = s == Engine::instance().gpu(0).stream(0) ? 0 : 1;
    if (n[d] >= kProfEvents) return;
    ph[d][n[d]] = phase;
    cudaEventRecord(ev[d][n[d]++], s);
  }
  // ONE sync, then per-phase GPU time from cudaEventElapsedTime. Totals are per phase id.
  // `tot2`, when given, receives the SAME deltas as `tot` in the same pass. The decode-only bucket
  // used to be filled by calling flush() a second time with td, but flush DRAINS the ring (n[dev]=0),
  // so that second call always found an empty ring and td stayed zero - the DECODE-only line reported
  // 0.00 ms for every phase no matter how many n==1 steps ran. Feeding both accumulators in one drain
  // fixes that: the per-chunk total and the decode-only total are now the same measurements, split by
  // which bucket the caller passes.
  void flush(int dev, double* tot, double* tot2 = nullptr) {
    if (n[dev] < 2) { n[dev] = 0; return; }
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(Engine::instance().gpu(dev).stream(0)));
    for (int i = 1; i < n[dev]; i++) {
      float ms = 0;
      if (cudaEventElapsedTime(&ms, ev[dev][i - 1], ev[dev][i]) != cudaSuccess) continue;
      const int p = ph[dev][i];
      if (p >= 0 && p < P_N) {
        tot[p] += ms;
        if (tot2) tot2[p] += ms;
      }
    }
    n[dev] = 0;
  }
};
PhaseProf& phase_prof() { static PhaseProf p; return p; }

// Two independent accumulators. A single one averages 256-token prefill chunks with 1-token decode
// chunks, which is not a per-token number for either - the prefill figure gets diluted ~256x and the
// decode figure inflated by however many chunks prefill happened to contribute. Every "per-token ms"
// line in RESULTS.md that mixed the two is a blend, not a measurement.
struct ProfTotals {
  double t[P_N] = {0};
  long n = 0;
  double td[P_N] = {0};      // decode (n == 1) only
  long nd = 0;
  void report() {
    if (!getenv("HELIOS_PROF")) return;
    if (nd > 0)
      fprintf(stderr,
              "[prof] DECODE-only ms  embed=%.2f ple=%.2f amix=%.2f attn=%.2f gdn=%.2f apply=%.2f "
              "mmix=%.2f moe=%.2f final=%.2f  (decode steps=%ld)\n",
              td[P_EMB] / nd, td[P_PLE] / nd, td[P_AMIX] / nd, td[P_ATTN] / nd, td[P_GDN] / nd,
              td[P_APPLY] / nd, td[P_MMIX] / nd, td[P_MOE] / nd, td[P_FINAL] / nd, nd);
    if (n > 0)
      fprintf(stderr,
              "[prof] ALL-chunks (blend) ms  embed=%.2f ple=%.2f amix=%.2f attn=%.2f gdn=%.2f "
              "apply=%.2f mmix=%.2f moe=%.2f final=%.2f  (steps=%ld)\n",
              t[P_EMB] / n, t[P_PLE] / n, t[P_AMIX] / n, t[P_ATTN] / n, t[P_GDN] / n, t[P_APPLY] / n,
              t[P_MMIX] / n, t[P_MOE] / n, t[P_FINAL] / n, n);
  }
};
ProfTotals& prof_totals() { static ProfTotals t; return t; }
}  // namespace

// The layer loop, [lo, hi), on the card that owns every layer in the range.
//
// This is the shipped loop, unchanged in what it computes: with the pipeline off the range is
// [0, n_layers) on card 0 and every pointer below is the one the single-set path always used. What
// the parameterisation buys is that card 1 can run the same loop over ITS layers with its own
// scratch, its own activations and its own stream - which is what makes the two halves a pipeline
// instead of one sequential pass.
//
// `tap_dst`/`tap_row0`, when given, receive this range's pre-collapse stack at the given row - the
// draft head reads it after the whole chunk, so with micro-batching each stage-1 micro-batch
// deposits its own rows.
void Runner::layer_range(int dev, const int* ids, int n, int pos0, int lo, int hi,
                         const std::function<void(int, cudaStream_t)>& mark, float* tap_dst,
                         int tap_row0, const BatchCtx* b) {
  const Config& c = m_->cfg;
  const int H = c.hc_mult, D = c.hidden;
  Act& a = dev == 0 ? act0_ : act1_;
  GdnScratch& gs = dev == 0 ? gdn_ : gdn1_;
  AttnScratch& as = dev == 0 ? attn_ : attn1_;
  PleScratch& ps = dev == 0 ? ple_ : ple1_;
  MoeScratch& ms = dev == 0 ? moe_ : moe1_;
  cudaStream_t s = Engine::instance().gpu(dev).stream(0);

  // HELIOS_LAYER_DUMP: per-layer differential. Reports the rms of each row's hyper-connection stack
  // after every layer, so the FIRST layer at which a paired forward's two rows stop agreeing is
  // visible directly. Every component A/B has come back correct, which is exactly why this is the
  // right instrument now: it localises the fault without assuming which component holds it. It has to
  // be called from BOTH paths - the eager body `continue`s past the end of the loop, and batched
  // decodes are never graphed, so the tail of the loop is not the only place a layer can finish.
  auto ldump = [&](int l, bool full) {
    if (!(b && b->bsz > 1) || !getenv("HELIOS_LAYER_DUMP")) return;
    static int dumped = 0;
    if (dumped >= 12) return;
    dumped++;
    const int dev_l = c.layer_dev(l);
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(dev_l).phys_idx()));
    // The layer's work is enqueued on `s`, not the legacy stream a synchronous cudaMemcpy uses, so
    // without this the read is unordered against the layer - which is what made the first version
    // of this print values ~1000x too small and repeat on alternate layers.
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(Engine::instance().gpu(dev_l).stream(0)));
    double r[2] = {0, 0};
    for (int row = 0; row < 2; row++) {
      std::vector<float> t((size_t)H * D);
      HELIOS_CUDA_CHECK(cudaMemcpy(t.data(), a.streams + (size_t)row * H * D,
                                   (size_t)H * D * 4, cudaMemcpyDeviceToHost));
      for (float v : t) r[row] += (double)v * v;
      r[row] = sqrt(r[row] / (double)(H * D));
    }
    fprintf(stderr, "[ldump] L%-2d %-4s row0 rms=%.9e row1 rms=%.9e  %s\n", l, full ? "attn" : "gdn",
            r[0], r[1],
            fabs(r[0] - r[1]) <= 1e-6 * (r[0] + r[1] + 1e-9) ? "AGREE" : "DIVERGE");
  };
  // HELIOS_STEP_PROBE=1: the same per-row differential one level finer - it probes a NAMED buffer at
  // named points INSIDE a layer body, so the first probe that reports DIVERGE names the step that
  // parted the rows rather than the layer. Only fires for a batched decode, and only for l == 0,
  // because the layer-level dump has already established that layer 0 is where it happens.
  auto probe = [&](int l, const char* tag, const void* p, size_t elems, size_t stride) {
    if (!(b && b->bsz > 1) || l != 0 || !getenv("HELIOS_STEP_PROBE")) return;
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.layer_dev(l)).phys_idx()));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(Engine::instance().gpu(c.layer_dev(l)).stream(0)));
    double r[2] = {0, 0};
    const float* f = (const float*)p;
    for (int row = 0; row < 2; row++) {
      std::vector<float> t(elems);
      // Row stride is D, NOT n*D: the buffer is (R, D), so reading row 1 at offset n*D reads past
      // the end. That bug made row 1 print a constant that was really adjacent memory, and it is why
      // the mixer looked like it "never wrote row 1".
      HELIOS_CUDA_CHECK(cudaMemcpy(t.data(), f + (size_t)row * stride, stride * 4,
                                   cudaMemcpyDeviceToHost));
      for (float v : t) r[row] += (double)v * v;
      r[row] = sqrt(r[row] / (double)stride);
    }
    fprintf(stderr, "[probe] L%d %-10s row0 rms=%.9e row1 rms=%.9e  %s\n", l, tag, r[0], r[1],
            fabs(r[0] - r[1]) <= 1e-6 * (r[0] + r[1] + 1e-9) ? "AGREE" : "DIVERGE");
  };


  ldump(-1, false);   // BEFORE any layer: do the two rows even start equal? (embed -> streams)

  for (int l = lo; l < hi; l++) {
    Layer& L = m_->layers[l];
    // Every launch below addresses this card's weights, scratch and state, so the CUDA context is
    // this card's. moe_layer leaves the current device wherever its last launch was, and with the
    // pipeline the two cards interleave, so it is re-asserted per layer.
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.layer_dev(l)).phys_idx()));

    if (L.has_ple && !getenv("HELIOS_NO_PLE")) {
      // The PLE needs the carried context AND the chunk: ple_ngram_ids derives its row count as
      // (n_hist - n_ctx), so passing only the context (as an earlier revision did on every chunk
      // after the first) yields zero rows and silently drops the layer's whole contribution.
      // With micro-batching this runs once per micro-batch rather than once per chunk, and that is
      // exact: the n-gram ids come from the HOST history (which is extended by the same micro-batch
      // order) and the conv is causal over a state the writeback carries.
      if (b && b->bsz > 1) {
        // Per row, not per batch: the PLE's input is the n-gram window of ITS OWN sequence and its
        // conv state belongs to its own slot, so a shared call would fold two different windows into
        // one. Each row is a 1-token call, which is what the single-sequence path does anyway.
        for (int r = 0; r < b->bsz; r++) {
          const int sl = b->slot[r];
          std::vector<int64_t>& h = (sl == active_slot_) ? hist_ : slots_[(size_t)sl].hist;
          void* pc = (sl == active_slot_) ? (void*)ple_conv_ : slots_[(size_t)sl].ple_conv;
          h.push_back(ids[r]);
          // n_ctx is the number of ids CARRIED INTO this call, not the token count: ple_ngram_ids
          // derives its row count as (n_hist - n_ctx), which is why the serial path passes
          // hist_.size() - n. Passing 1 here made the PLE fold ngram_size rows instead of one, i.e.
          // re-inject stale carried ids - and that is what parted the paired rows at layer 0, the
          // first layer the PLE touches.
          ple_layer(L.ple, c, ps, a.embed16 + (size_t)r * D * 2,
                    a.streams + (size_t)r * H * D * 4, h.data(), (int)h.size(),
                    (int)h.size() - 1, pc, 1, s,
                    /*capture=*/false);
          const size_t k2 = (size_t)(c.ngram_size - 1);
          if (h.size() > k2) h.erase(h.begin(), h.end() - k2);
        }
      } else {
      hist_.insert(hist_.end(), ids, ids + n);
      // spec_batch_ is set only around a speculative verify batch, whose PLE state has to be
      // rewound on a partial accept exactly like the GDN recurrence (ple_replay).
      ple_layer(L.ple, c, ps, a.embed16, a.streams, hist_.data(), (int)hist_.size(),
                (int)(hist_.size() - n), ple_conv_, n, s, /*capture=*/spec_batch_);
      if (getenv("HELIOS_PRMS")) {
        std::vector<float> st((size_t)n * H * D);
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(st.data(), a.streams, (size_t)n * H * D * 4,
                                          cudaMemcpyDeviceToHost, s));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        double acc = 0;
        for (float v : st) acc += (double)v * v;
        fprintf(stderr, "[prms] after PLE: rms=%.6f (ref layers.1.ple=0.017761)\n",
                sqrt(acc / st.size()));
      }
      const size_t keep = (size_t)(c.ngram_size - 1);
      if (hist_.size() > keep) hist_.erase(hist_.begin(), hist_.end() - keep);
      }
      mark(P_PLE, s);
    }

    chk("loop top", s);
    if (getenv("HELIOS_LRMS")) {
      std::vector<float> st((size_t)n * H * D);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(st.data(), a.streams, (size_t)n * H * D * 4,
                                        cudaMemcpyDeviceToHost, s));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      double acc = 0;
      for (float v : st) acc += (double)v * v;
      fprintf(stderr, "[lrms] L%d rms=%.9e\n", l, sqrt(acc / st.size()));
    }
    // ---- the capturable body, from here to the end of the layer ----
    //
    // Everything between the mix and the layer's last gr_apply runs on fixed device pointers with
    // no host branch, EXCEPT the full-attention sublayer, whose rope positions, KV append offset
    // and split-KV decision all come from pos0 and therefore change every step. So a GDN layer
    // captures whole, and a full-attention layer captures everything after its attention kernels.
    // See dgraph.hpp for the full argument and for what is deliberately left out (the PLE, which
    // hashes its n-grams on the host, and the attention itself).
    // A batched decode never graphs: the batch size, the per-row positions and the per-slot state
    // bases are all kernel arguments, and a captured graph has them frozen at capture time. Routing
    // it to the eager body costs the ~15% graph replay saves and is the price of correctness here;
    // the graph key could grow a bsz term later if it is ever worth it.
    const bool graphable = dgraph_ok_ && moe_decode_graphable(n) && !(b && b->bsz > 1);
    if (!graphable) {
      // Eager body, byte-for-byte the pre-graph loop.
      probe(l, "pre_amix", a.mixed, (size_t)n * D, (size_t)D);
      aux::gr_mix(a.streams, L.hc_attn.norm, L.hc_attn.down, L.hc_attn.up, L.hc_attn.inject, n, H, D,
             c.hc_rank, c.rms_eps, a.mixed, a.post, s);
      glue::cast_f32_f16(a.mixed, a.sub_in, (size_t)n * D, s);
      mark(P_AMIX, s);
      if (getenv("HELIOS_HC")) {
        std::vector<float> mx((size_t)n * D), po((size_t)n * H);
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(mx.data(), a.mixed, mx.size() * 4,
                                          cudaMemcpyDeviceToHost, s));
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(po.data(), a.post, po.size() * 4,
                                          cudaMemcpyDeviceToHost, s));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        double x = 0, y = 0;
        for (float v : mx) x += (double)v * v;
        for (float v : po) y += (double)v * v;
        fprintf(stderr, "[hc] L%d attn mixed=%.9e post=%.9e\n", l, sqrt(x / mx.size()),
                sqrt(y / po.size()));
      }
      chk(L.full ? "attn mix" : "gdn mix", s);
      // HELIOS_PROBE_SELFTEST=1: perturb row 1 of a.mixed AFTER the mixer has written it, so the
      // perturbation survives to the probe. The mixer normally leaves both rows bit-identical here, so
      // a working, correctly-ordered probe MUST flip to DIVERGE. (The first version of this check
      // perturbed the buffer BEFORE the mixer, which simply overwrote the perturbation - and the
      // probe still said AGREE, which is the correct answer to the wrong question. An instrument has
      // to be shown able to see a difference before its "these are equal" is worth anything.)
      if (b && b->bsz > 1 && l == 0 && getenv("HELIOS_PROBE_SELFTEST")) {
        // The mixer's work is on `s`, not the legacy stream this memcpy uses, so without the sync the
        // mixer simply overwrites the perturbation afterwards. That is the second version of this
        // check to report AGREE for the wrong reason.
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<float> bump((size_t)D, 12345.0f);
        HELIOS_CUDA_CHECK(cudaMemcpy(a.mixed + (size_t)D, bump.data(), (size_t)D * 4,
                                     cudaMemcpyHostToDevice));
      }
      probe(l, "attn_mix", a.mixed, (size_t)n * D, (size_t)D);
      if (L.full) {
        if (b && b->bsz > 1) {
          // One row at a time, and that is the design rather than a limitation: the quantized-KV
          // staging buffer is ONE region of max_ctx rows, so a second staged region would need
          // another 268 MB x 2 and does not fit in card1's free VRAM. Each row attends over its own
          // history at its own position against its own cache base; the row offset into the
          // activation buffers is the only thing that keeps both sequences in one forward. Attention
          // is ~9% of a decode step, and everything after it - GDN, MoE, the mHC mixers, ~88% - is
          // per-token over n and does batch.
          for (int r = 0; r < b->bsz; r++) {
            const int sl = b->slot[r];
            attn_layer(L.attn, c, as, a.sub_in + (size_t)r * D, a.sub_out + (size_t)r * D,
                       kv_base_slot(0, L.attn_ord, sl), kv_base_slot(1, L.attn_ord, sl), 1,
                       b->pos[r], s, dev == 0 ? L.attn_ord : attn_card_ord_[L.attn_ord]);
          }
        } else {
          attn_layer(L.attn, c, as, a.sub_in, a.sub_out, kv_base(0, L.attn_ord), kv_base(1, L.attn_ord),
                     n, pos0, s, dev == 0 ? L.attn_ord : attn_card_ord_[L.attn_ord]);
        }
      }
      else {
        // Capture this layer's sublayer input while a speculative batch is in flight. `a.sub_in` is
        // ONE buffer reused by every layer, so after the loop it holds only the LAST layer's value -
        // snapshotting it once at the end cannot replay the earlier GDN layers. That was the first
        // version's bug and `bench-gdn` caught it: the replay read the wrong rows for 35 of 36 layers.
        // The capture buffer is per card (gdn_sub_snap_ / gdn_sub_snap1_): the index is the global
        // gdn_ord, but the memory has to be on the layer's own card.
        if (gdn_capture_) {
          half* dst = (dev == 0 ? gdn_sub_snap_ : gdn_sub_snap1_) + (size_t)L.gdn_ord * gdn_sub_stride_;
          HELIOS_CUDA_CHECK(cudaMemcpyAsync(dst, a.sub_in, (size_t)n * c.hidden * 2,
                                            cudaMemcpyDeviceToDevice, s));
        }
        // Batched: the kernels index their state by slot, so the base handed over is the FIRST row's
        // slot and the slot arrays are relative to it. See BatchCtx for why the two arrays differ.
        const bool bat = (b && b->bsz > 1);        // HELIOS_BATCH_GDN_SERIES=1: bisection. Run the GDN as two n=1 calls (one per row, each on its
        // own state base) while leaving the mixers and the MoE batched at n=2. If the paired forward
        // then matches serial, the fault is the GDN's bsz=2 path IN SITU - which would contradict the
        // A/B in test_gdn_bsz2 and mean the difference is in how the engine wires it, not the kernel.
        // If it still diverges, the fault is in the work the two rows SHARE.
        // The GDN runs PER ROW even in a batch. Its two kernels do accept bsz=2 and a per-slot state
        // base - test_gdn_bsz2 proves both the recurrence and the depthwise conv bit-identical to two
        // n=1 calls in isolation - but wired the way the engine wires them the paired forward does not
        // match serial, and with this one change it does: batch-parity goes from MISMATCH on both
        // slots to MATCH on both, over 10 steps, with the mixers and the MoE still batched at n=2.
        // Whatever the disagreement is, it is confined to this call, and this is the configuration
        // that is demonstrably correct. The GDN is ~26% of a decode step and its weights are read
        // once either way, so two n=1 calls cost launch overhead, not bandwidth - which is the trade
        // this makes deliberately until the disagreement is explained.
        // HELIOS_BATCH_GDN_BATCHED=1 restores the single bsz=2 call, for bisection.
        if (bat && !getenv("HELIOS_BATCH_GDN_BATCHED")) {
          for (int r = 0; r < b->bsz; r++) {
            gdn_layer(L.gdn, c, gs, a.sub_in + (size_t)r * D, a.sub_out + (size_t)r * D,
                      slots_[(size_t)b->slot[r]].gdn_conv[L.gdn_ord],
                      (float*)slots_[(size_t)b->slot[r]].gdn_rec[L.gdn_ord],
                      1, s, 1, nullptr, nullptr, 0);
          }
        } else {
        gdn_layer(L.gdn, c, gs, a.sub_in, a.sub_out,
                  bat ? slots_[(size_t)b->slot[0]].gdn_conv[L.gdn_ord] : gdn_conv_[L.gdn_ord],
                  bat ? (float*)slots_[(size_t)b->slot[0]].gdn_rec[L.gdn_ord]
                      : (float*)gdn_rec_[L.gdn_ord],
                  n, s, bat ? b->bsz : 1, bat ? b->gdn_slots[dev] : nullptr,
                  bat ? b->conv_slots[dev] : nullptr, bat ? b->slot_layers[dev] : 0);
        }
        // AFTER the call: the previous version of this probe sat before it and so reported the
        // PREVIOUS layer's leftover in a.sub_out, which looks exactly like a GDN divergence.
        probe(l, "gdn_out", a.sub_out, (size_t)n * D, (size_t)D);
      }
      chk(L.full ? "attn_layer" : "gdn_layer", s);
      mark(L.full ? P_ATTN : P_GDN, s);
      if (getenv("HELIOS_SUB")) {
        std::vector<float> t((size_t)n * D);
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), a.sub_out, t.size() * 4,
                                          cudaMemcpyDeviceToHost, s));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        double acc = 0;
        for (float v : t) acc += (double)v * v;
        fprintf(stderr, "[sub] L%d sublayer=%.9e\n", l, sqrt(acc / t.size()));
      }
      aux::gr_apply(a.streams, a.sub_out, a.post, n, H, D, s);
      mark(P_APPLY, s);
      if (getenv("HELIOS_MIN")) {
        std::vector<float> t((size_t)n * H * D);
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), a.streams, t.size() * 4, cudaMemcpyDeviceToHost, s));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        double acc = 0;
        for (float v : t) acc += (double)v * v;
        fprintf(stderr, "[min] L%d pre_mlp_streams=%.9e\n", l, sqrt(acc / t.size()));
      }
      aux::gr_mix(a.streams, L.hc_mlp.norm, L.hc_mlp.down, L.hc_mlp.up, L.hc_mlp.inject, n, H, D,
             c.hc_rank, c.rms_eps, a.mixed, a.post, s);
      glue::cast_f32_f16(a.mixed, a.sub_in, (size_t)n * D, s);
      chk("mlp mix", s);
      probe(l, "mlp_mix", a.mixed, (size_t)n * D, (size_t)D);
      mark(P_MMIX, s);
      if (getenv("HELIOS_MIN")) {
        std::vector<float> t((size_t)n * D);
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), a.sub_in, t.size() * 4, cudaMemcpyDeviceToHost, s));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        double acc = 0;
        for (float v : t) acc += (double)v * v;
        fprintf(stderr, "[min] L%d moe_input=%.9e\n", l, sqrt(acc / t.size()));
      }
      moe_layer(L.moe, c, ms, a.sub_in, a.sub_out, l, /*mtp=*/false, n, s);
      chk("moe_layer", s);
      mark(P_MOE, s);
      if (getenv("HELIOS_MRMS")) {
        std::vector<float> t((size_t)n * D);
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), a.sub_out, t.size() * 4, cudaMemcpyDeviceToHost, s));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        double acc = 0;
        for (float v : t) acc += (double)v * v;
        fprintf(stderr, "[mrms] L%d mlp=%.9e\n", l, sqrt(acc / t.size()));
      }
      aux::gr_apply(a.streams, a.sub_out, a.post, n, H, D, s);
      ldump(l, L.full);
      continue;
    }
    // ---- graphed body ----
    //
    // `body` is the whole layer, minus the PLE. For a full-attention layer it starts AFTER the
    // attention kernels: those are run eagerly here (they need pos0) and their output is already in
    // a.sub_out, so the captured body begins at the gr_apply that folds it into the streams. That
    // is why the two sites below differ by where they start, not by what they contain - the tail
    // (apply -> mlp mix -> cast -> MoE -> apply) is common to both and is the MoE-bearing part.
    //
    // gdn_capture_ is part of the key, not a runtime branch inside the graph: the speculative
    // sublayer-input snapshot is a D2D copy that must sit between the cast and the recurrence, so a
    // capture taken with it on cannot be replayed with it off. The eager path above handles that
    // combination by simply not being graphed at all when the key differs.
    std::function<void(cudaStream_t)> body = [&](cudaStream_t st) {
      if (L.full) {
        // The attention kernels ran eagerly, before this body was entered.
        aux::gr_apply(a.streams, a.sub_out, a.post, n, H, D, st);
      } else {
        aux::gr_mix(a.streams, L.hc_attn.norm, L.hc_attn.down, L.hc_attn.up, L.hc_attn.inject, n, H,
                    D, c.hc_rank, c.rms_eps, a.mixed, a.post, st);
        glue::cast_f32_f16(a.mixed, a.sub_in, (size_t)n * D, st);
        if (gdn_capture_) {
          half* dst = (dev == 0 ? gdn_sub_snap_ : gdn_sub_snap1_) + (size_t)L.gdn_ord * gdn_sub_stride_;
          HELIOS_CUDA_CHECK(cudaMemcpyAsync(dst, a.sub_in, (size_t)n * c.hidden * 2,
                                            cudaMemcpyDeviceToDevice, st));
        }
        gdn_layer(L.gdn, c, gs, a.sub_in, a.sub_out, gdn_conv_[L.gdn_ord],
                  (float*)gdn_rec_[L.gdn_ord], n, st);
        aux::gr_apply(a.streams, a.sub_out, a.post, n, H, D, st);
      }
      aux::gr_mix(a.streams, L.hc_mlp.norm, L.hc_mlp.down, L.hc_mlp.up, L.hc_mlp.inject, n, H, D,
             c.hc_rank, c.rms_eps, a.mixed, a.post, st);
      glue::cast_f32_f16(a.mixed, a.sub_in, (size_t)n * D, st);
      moe_layer(L.moe, c, ms, a.sub_in, a.sub_out, l, /*mtp=*/false, n, st);
      aux::gr_apply(a.streams, a.sub_out, a.post, n, H, D, st);
    };
    if (L.full) {
      // pos0-dependent: run the attention site eagerly, then replay the captured tail.
      aux::gr_mix(a.streams, L.hc_attn.norm, L.hc_attn.down, L.hc_attn.up, L.hc_attn.inject, n, H, D,
             c.hc_rank, c.rms_eps, a.mixed, a.post, s);
      glue::cast_f32_f16(a.mixed, a.sub_in, (size_t)n * D, s);
      attn_layer(L.attn, c, as, a.sub_in, a.sub_out, kv_base(0, L.attn_ord), kv_base(1, L.attn_ord), n,
                 pos0, s, dev == 0 ? L.attn_ord : attn_card_ord_[L.attn_ord]);
    }
    // Site 0 = whole GDN layer, site 1 = the full-attention layer's MLP-bearing tail.
    dgraphs_.run(dgraph_key(L.full ? 1 : 0, dev, l, n, gdn_capture_ ? 1 : 0), n, s, body);
    ldump(l, L.full);
  }

  if (tap_dst)
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(tap_dst + (size_t)tap_row0 * H * D, a.streams,
                                      (size_t)n * H * D * 4, cudaMemcpyDeviceToDevice, s));
}

// Everything after the last layer, on the card that owns it.
//
//   1. save the pre-collapse tap the draft head reads,
//   2. fill the draft head's own KV cache for the positions this chunk covers,
//   3. the combine-less final mixer (which IS the model norm) and lm_head.
//
// The draft is a full attention layer with its own cache, but it only ever ran during decode - so its
// rows for every prompt position were never written. The KV pool is bump-allocated without a memset
// (`A(kv_bytes)`), so the draft attended over whatever the allocator had handed out: its predictions
// depended on heap contents, which is why the accept rate was not reproducible across identical
// greedy runs (52.4% then 39.1%) and why the drafts were poor. Running the same chunk through the
// head costs one layer and makes the cache valid; compute_head=false because the head gemm reads
// the whole 397 MB lm_head and its result is discarded here.
void Runner::chunk_tail(int dev, const int* ids, int n, int pos0, cudaStream_t s,
                        const std::function<void(int, cudaStream_t)>& mark, int head_rows) {
  const Config& c = m_->cfg;
  const int H = c.hc_mult, D = c.hidden;
  Act& a = dev == 0 ? act0_ : act1_;
  // The tail runs on this card's stream, so this card must also be the current context: a launch
  // issued on card 1's stream while card 0 is current addresses card 0's address space.
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(dev).phys_idx()));

  if (getenv("HELIOS_RMS")) {
    std::vector<float> h((size_t)n * D), st((size_t)n * H * D), lg((size_t)c.vocab);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(h.data(), a.sub_in, (size_t)n * D * 2,
                                      cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(st.data(), a.streams, (size_t)n * H * D * 4,
                                      cudaMemcpyDeviceToHost, s));
    float* lg_dev = logits_ptr();
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(dev).phys_idx()));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(lg.data(), lg_dev, (size_t)c.vocab * 4,
                                      cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    double hs = 0; for (float v : h) hs += (double)v * v;
    double ss = 0; for (float v : st) ss += (double)v * v;
    float lo = lg[0], hi = lg[0];
    for (float v : lg) { lo = std::min(lo, v); hi = std::max(hi, v); }
    fprintf(stderr, "[rms] n=%d hidden_rms=%.4f streams_rms=%.4f logits=[%.3f,%.3f]\n", n,
            sqrt(hs / h.size()), sqrt(ss / st.size()), lo, hi);
  }

  // The MTP head taps the trunk's PRE-collapse stack, so it is saved before the final mixer.
  if (c.has_mtp && a.tap)
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(a.tap, a.streams, (size_t)n * H * D * 4,
                                      cudaMemcpyDeviceToDevice, s));

  if (c.has_mtp && mtp_on_ && !mtp_filling_) {
    // The draft runs on the LAST layer's card, and with the pipeline that is not the card the
    // embedding table is on: its rows are gathered on card 0 and handed over (see mtp_stage_embed).
    const int md = c.mtp_dev();
    const half* emb = md == c.layer_dev(0) ? nullptr : mtp_stage_embed(ids, n, s);
    mtp_draft_step(*m_, mtp_, md == 0 ? attn_ : attn1_, md == 0 ? moe_ : moe1_, a.tap, ids, n, pos0,
                   kv_base(0, mtp_ord_), kv_base(1, mtp_ord_), nullptr, nullptr,
                   logits_ptr() + (size_t)c.vocab, /*stream_tap=*/true, s,
                   /*compute_head=*/false, emb);
  }

  // combine-less final mixer: this IS the model norm, there is no separate final norm. The mixer is
  // DUPLICATED per card (Model::mixer_for), because every layer's collapse uses it and a device
  // pointer is not dereferenceable from the other card.
  const HcWeights& mx = m_->mixer_for(dev);
  aux::gr_mix(a.streams, mx.norm, mx.down, mx.up, nullptr, n, H, D, c.hc_rank, c.rms_eps, a.mixed,
         nullptr, s);
  mark(P_APPLY, s);
  glue::cast_f32_f16(a.mixed, a.sub_in, (size_t)n * D, s);
  final_head(n, head_rows);
  mark(P_FINAL, s);
}

// One cross-card copy through pinned host memory, ordered behind an event on the SOURCE card's
// compute stream.
// These boards are PHB with no peer access (cudaDeviceCanAccessPeer is false), so there is no
// cudaMemcpyPeerAsync to use and the bytes go through the host. The D2H rides card 0's DMA-OUT
// stream, gated by the event, so the host wait drains the COPY and not card 0's compute stream -
// which is precisely what keeps the two stages overlapping while the CPU blocks. Draining the
// compute stream instead would serialise the whole pipeline into (stage0, handoff, stage1).
void Runner::xcopy(void* dst, cudaStream_t s_dst, const void* src, cudaStream_t s_src, size_t bytes,
                   int slot) {
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
  cudaEvent_t ev = xdone_[slot];
  HELIOS_CUDA_CHECK(cudaEventRecord(ev, s_src));
  cudaStream_t sx = Engine::instance().gpu(0).stream(2);
  HELIOS_CUDA_CHECK(cudaStreamWaitEvent(sx, ev, 0));
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(xbounce_[slot], src, bytes, cudaMemcpyDeviceToHost, sx));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(sx));
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(dst, xbounce_[slot], bytes, cudaMemcpyHostToDevice, s_dst));
}

// The hyper-connection stream stack crossing the layer boundary: the ONLY thing a layer hands to the
// next one, and therefore the only thing that has to cross the bus. mixed_/post_/sub_in_/sub_out_
// are re-derived by every layer, so they never leave their card.
void Runner::handoff(int slot, size_t bytes, cudaStream_t s_dst) {
  xcopy(act1_.streams, s_dst, act0_.streams, Engine::instance().gpu(0).stream(0), bytes, slot);
}

// HELIOS_PIPELINE: the draft head runs on the last layer's card, but the embedding table is on layer
// 0's card, so the n rows it needs are gathered where the table lives and handed over. Card 0 is
// idle by the time the tail runs and the transfer is n*hidden*2 bytes (5.2 MB at a 1024-token chunk),
// so it costs a fraction of a millisecond per chunk - far cheaper than duplicating the 1.2 GB table.
const half* Runner::mtp_stage_embed(const int* ids, int n, cudaStream_t s_dst) {
  const int D = m_->cfg.hidden;
  cudaStream_t s0 = Engine::instance().gpu(0).stream(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(act0_.ids, ids, (size_t)n * 4, cudaMemcpyHostToDevice, s0));
  glue::embed_gather(m_->embed, act0_.ids, n, act0_.embed16, D, s0);
  xcopy(mtp_.emb16_, s_dst, act0_.embed16, s0, (size_t)n * D * 2, mtp_stage_slot());
  // Hand the context back to the card that runs the head: the gather above moved it to card 0.
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(m_->cfg.mtp_dev()).phys_idx()));
  return mtp_.emb16_;
}

void Runner::run_chunk(const std::vector<int>& ids, int pos0, int head_rows, const BatchCtx* b) {
  const Config& c = m_->cfg;
  if (c.pipeline) { run_chunk_pipeline(ids, pos0, head_rows, b); return; }
  const int n = (int)ids.size();
  last_chunk_w_ = n;   // tap_ rows are indexed within the chunk, not by absolute position
  const int D = c.hidden;
  cudaStream_t s = Engine::instance().gpu(0).stream(0);
  cudaSetDevice(Engine::instance().gpu(0).phys_idx());
  PhaseProf& P = phase_prof();
  P.on = getenv("HELIOS_PROF") != nullptr;
  P.ensure();
  std::function<void(int, cudaStream_t)> mark = [&](int phase, cudaStream_t st) {
    P.mark(phase, st);
  };

  HELIOS_CUDA_CHECK(cudaMemcpyAsync(ids_dev_, ids.data(), (size_t)n * 4, cudaMemcpyHostToDevice, s));
  glue::embed_gather(m_->embed, ids_dev_, n, embed16_, D, s);
  if (getenv("HELIOS_EMB")) {
    std::vector<half> e((size_t)n * D);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(e.data(), embed16_, (size_t)n * D * 2,
                                      cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    fprintf(stderr, "[emb] tok0=%d first8:", ids[0]);
    for (int i = 0; i < 8; i++) fprintf(stderr, " %.6f", __half2float(e[i]));
    fprintf(stderr, "\n");
  }
  glue::stream_expand(embed16_, streams_, n, D, s);      // ExpandStreams: 4 fp32 streams
  mark(P_EMB, s);
  chk("embed+expand", s);

  layer_range(0, ids.data(), n, pos0, 0, c.n_layers, mark, tap_, 0);
  chunk_tail(0, ids.data(), n, pos0, s, mark, head_rows);
  if (P.on) {
    double* td = (n == 1) ? prof_totals().td : nullptr;
    P.flush(0, prof_totals().t, td);
    prof_totals().n++;
    if (n == 1) prof_totals().nd++;
    prof_totals().report();
  }
}

// HELIOS_PIPELINE=1: the layer-pipelined forward.
//
//   card 0 (stream A)                          card 1 (stream B)
//   ---------------                            ---------------
//   micro 0: embed+expand, layers [0,split)     <- handoff(0) - layers [split,n)
//   micro 1: embed+expand, layers [0,split)     <- handoff(1) - layers [split,n)
//   ...                                        micro M-1 ...
//   (both idle)                                                    mixer + lm_head
//
// The stagger is the point: card 1's micro-batch i overlaps card 0's micro-batch i+1, so the two
// cards are both busy for all but the first and last half-chunks. What crosses the bus per
// micro-batch is the hyper-connection stack only (H*hidden*4 bytes per token, 10.5 MB at 256
// tokens) - everything else each stage needs it re-derives locally.
//
// The split is EXACT, not approximate. Each micro-batch's rows are a contiguous slice of the same
// chunk, and every layer is causal in the position, so the per-row arithmetic is unchanged:
//   - attention: micro-batch i writes KV rows [off, off+nb) and attends over [0, off+nb), which is
//     exactly the range its own call already attended over; the rows before it were written by the
//     previous micro-batch's call to the same layer, on the same card, in order.
//   - GDN/PLE: both are running recurrences whose state the call carries, so consecutive
//     micro-batches advance the same state in the same order.
//   - MoE: each token belongs to exactly one micro-batch, so no token's expert sum is split.
// The one thing that DOES change is the grouped kernel's expert visit order (LPT over a smaller
// token count), which reassociates the per-token expert sum - see RESULTS.md.
void Runner::run_chunk_pipeline(const std::vector<int>& ids, int pos0, int head_rows,
                                const BatchCtx* b) {
  const Config& c = m_->cfg;
  const int n = (int)ids.size();
  last_chunk_w_ = n;
  const int H = c.hc_mult, D = c.hidden;
  const int split = c.split;
  // Micro-batch count. A decode step and a speculative verify are 1-3 tokens wide: there is no
  // second micro-batch to overlap with, and at that width the grouped MoE's fixed per-launch cost
  // dominates anyway, so those run as ONE micro-batch - a correct two-card SEQUENTIAL pass.
  int M = (n >= 64) ? micro_ : 1;
  if (M > n) M = n;
  cudaStream_t s0 = Engine::instance().gpu(0).stream(0);
  cudaStream_t s1 = Engine::instance().gpu(1).stream(0);
  PhaseProf& P = phase_prof();
  P.on = getenv("HELIOS_PROF") != nullptr;
  P.ensure();
  std::function<void(int, cudaStream_t)> mark = [&](int phase, cudaStream_t st) {
    P.mark(phase, st);
  };

  for (int i = 0; i < M; i++) {
    const int off = (int)((long)n * i / M);
    const int nb = (int)((long)n * (i + 1) / M) - off;
    const int mb_pos = pos0 + off;
    // ---- stage 0: card 0, layers [0, split) ----
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(act0_.ids, ids.data() + off, (size_t)nb * 4,
                                      cudaMemcpyHostToDevice, s0));
    glue::embed_gather(m_->embed, act0_.ids, nb, act0_.embed16, D, s0);
    glue::stream_expand(act0_.embed16, act0_.streams, nb, D, s0);
    mark(P_EMB, s0);
    layer_range(0, ids.data() + off, nb, mb_pos, 0, split, mark, nullptr, 0, b);
    // ---- the handoff, then stage 1: card 1, layers [split, n_layers) ----
    // The D2H is ordered behind stage 0 by the event xcopy records, and the H2D lands on stream B
    // behind whatever stage 1 still has queued for the previous micro-batch, so neither side can
    // read a buffer the other is still writing.
    handoff(i, (size_t)nb * H * D * 4, s1);
    layer_range(1, ids.data() + off, nb, mb_pos, split, c.n_layers, mark,
                c.has_mtp ? act1_.tap : nullptr, off, b);
  }

  // ---- the end of the stack, on the card that produced it ----
  chunk_tail(last_dev_, ids.data(), n, pos0, Engine::instance().gpu(last_dev_).stream(0), mark,
             head_rows);
  if (P.on) {
    // A decode step's marks are flushed into BOTH the per-chunk total and the decode-only total in
    // one drain; a second flush() would see an already-drained ring. (See PhaseProf::flush.)
    double* td = (n == 1) ? prof_totals().td : nullptr;
    P.flush(0, prof_totals().t, td);
    if (M > 0) P.flush(1, prof_totals().t, td);
    prof_totals().n++;
    if (n == 1) prof_totals().nd++;
    prof_totals().report();
  }
}

int Runner::next_token() {
  const Config& c = m_->cfg;
  // The logits are on the LAST layer's card, so the read has to be issued there: an async D2H from
  // another card's pointer is not merely unordered, it is an illegal access.
  cudaStream_t s = last_stream();
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(last_dev_).phys_idx()));
  std::vector<float> row((size_t)c.vocab);
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(row.data(), logits_ptr(), (size_t)c.vocab * 4,
                                    cudaMemcpyDeviceToHost, s));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  if (getenv("HELIOS_TOPK")) {
    std::vector<int> idx((size_t)c.vocab);
    for (int i = 0; i < c.vocab; i++) idx[i] = i;
    const int K = atoi(getenv("HELIOS_TOPK"));
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return row[a] > row[b]; });
    float mx = row[idx[0]], sum = 0.f;
    for (int i = 0; i < K; i++) sum += expf(row[idx[i]] - mx);
    fprintf(stderr, "[topk] pos=%d:", pos_);
    for (int i = 0; i < K; i++)
      fprintf(stderr, " %d(%.4f)", idx[i], expf(row[idx[i]] - mx) / sum);
    fprintf(stderr, "\n");
  }
  return (int)(std::max_element(row.begin(), row.end()) - row.begin());
}

double Runner::bench_chunk(int n, int iters, int warmup) {
  if (n < 1 || iters < 1) return 0.0;
  std::vector<int> ids(n, 100);
  for (int i = 0; i < warmup; i++) { run_chunk(ids, pos_); pos_ += n; }
  sync_all();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) { run_chunk(ids, pos_); pos_ += n; }
  sync_all();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// Verify K drafted tokens in the SAME trunk forward that commits `next`.
//
// The batch is [next, d0, d1]; logits row i predicts the token AFTER batch[i], so row 0 is the
// model's prediction for position pos_+1 and must equal d0 if the draft was right. `m` is the length
// of the accepted draft prefix. State rollback only has to undo what the rejected tail advanced:
// the GDN recurrence is restored and replayed over the accepted prefix (1+m positions), which costs
// ~3.65 ms instead of repeating the ~22 ms trunk forward.
int Runner::spec_verify(int next, const int* drafts, int K, int emit[kMaxSpecDrafts + 1]) {
  const Config& c = m_->cfg;
  // Everything below the forward (the row argmax, the carry tap) lives on the last layer's card.
  cudaStream_t s = last_stream();
  Act& a = last_dev_ == 0 ? act0_ : act1_;
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(last_dev_).phys_idx()));

  std::vector<int> batch;
  batch.push_back(next);
  for (int i = 0; i < K; i++) batch.push_back(drafts[i]);
  const int width = (int)batch.size();

  // `width` rows, one per position of the batch: row i is the trunk's prediction for the position
  // AFTER batch[i], so rows 0..K-1 are checked against the drafts and row K is next_tok_. This is
  // the ONLY forward that projects more than the last position.
  // Save BOTH recurrent histories before the batch. The GDN recurrence and the PLE's dilated conv
  // are running states, and so is `hist_` - the host n-gram context, truncated to ngram_size-1
  // after every chunk. A batch that ends on a rejected draft leaves all three holding tokens the
  // model never committed, and the next forward is then conditioned on them.
  if (K > 0) gdn_snapshot(width);
  hist_snap_ = hist_;
  gdn_capture_ = (K > 0);
  spec_batch_ = (K > 0);
  run_chunk(batch, pos_, width);
  gdn_capture_ = false;
  spec_batch_ = false;
  pos_ += width;

  int m = 0;
  int got[kMaxSpecDrafts + 2];
  for (int i = 0; i < kMaxSpecDrafts + 2; i++) got[i] = -1;
  next_tok_ = -1;   // reset first: a stale value here is what silently duplicated every token
  if (K > 0) {
    // K+1 rows, not K: row m is the LAST COMMITTED position, and its logits are the prediction for
    // the next token. Reading only K rows and letting the caller fall back to next_token() - which
    // reads row 0 - re-emits the token that was just accepted, doubling the output.
    row_argmax(logits_ptr(), K + 1, c.vocab, mtp_.ids_dev_, s);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(got, mtp_.ids_dev_, (K + 1) * sizeof(int),
                                      cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    for (int i = 0; i < K; i++) {
      if (got[i] != batch[i + 1]) break;
      m = i + 1;
    }
  }
  // Everything the rejected tail advanced has to go back: the GDN recurrence, the PLE's dilated
  // conv, and the host n-gram context. The first two are device state and rewind by restore +
  // replay over the accepted prefix; the third is host state and is simply rebuilt, because
  // run_chunk truncates it to ngram_size-1 after every chunk, so a batch ending on a rejected draft
  // leaves the next chunk keying its n-grams on a token the model never committed.
  if (K > 0 && m < K) {
    gdn_replay(1 + m);
    ple_replay(1 + m);
    hist_ = hist_snap_;
    hist_.insert(hist_.end(), batch.begin(), batch.begin() + (1 + m));
    const size_t keep = (size_t)(c.ngram_size - 1);
    if (hist_.size() > keep) hist_.erase(hist_.begin(), hist_.end() - keep);
  }
  // Rewind the position counter for the rejected tail. KV slots stay written; they are simply
  // beyond the committed position and will be overwritten by the next forward.
  pos_ -= (width - (1 + m));

  emit[0] = next;
  int n_out = 1;
  for (int i = 0; i < m; i++) emit[n_out++] = drafts[i];
  // Publish the token after the last committed row so the caller does not read row 0.
  if (K > 0) next_tok_ = got[m];

  // Carry the tap at the LAST COMMITTED row. This is the tap the next round's chain is seeded
  // from, and it is what removes the dedicated width-1 seed forward.
  const size_t tap_elems = (size_t)c.hc_mult * c.hidden;
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(a.carry_tap, a.tap + (size_t)m * tap_elems,
                                    tap_elems * sizeof(float), cudaMemcpyDeviceToDevice, s));
  carry_valid_ = true;
  last_committed_ = emit[n_out - 1];
  if (K > 0) tm_.spec_accepts += m;
  trace_state("spec", pos_);
  return n_out;
}

// Seed the carry tap from the tail of the most recent forward. Called after prefill and after any
// non-speculative decode, so the first speculative step has a real tap instead of doing the
// dedicated width-1 seed forward we are trying to eliminate.
//
// It deliberately does NOT touch last_committed_. The caller owns that, and deriving it from
// hist_tokens_.back() here is wrong: generate() pushes `next` to hist_tokens_ only AFTER
// spec_step returns, so at this point the back is the PREVIOUS token, not the one that now
// occupies pos_-1. Getting that wrong feeds the chain a bogus token, which overwrites the draft
// KV at pos_-1 and poisons every later prediction.
void Runner::capture_carry() {
  const Config& c = m_->cfg;
  if (pos_ < 1) return;
  const size_t tap_elems = (size_t)c.hc_mult * c.hidden;
  // The tap is written by the LAST layer, so the carry copy is issued on that card.
  cudaStream_t s = last_stream();
  Act& a = last_dev_ == 0 ? act0_ : act1_;
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(last_dev_).phys_idx()));
  const int row = (last_chunk_w_ > 0 ? last_chunk_w_ : 1) - 1;
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(a.carry_tap, a.tap + (size_t)row * tap_elems,
                                    tap_elems * sizeof(float), cudaMemcpyDeviceToDevice, s));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  carry_valid_ = true;
}

int Runner::spec_step(int next, int emit[kMaxSpecDrafts + 1]) {
  // ONE trunk forward per speculative step: the verify batch is [next, d0] at width 2, seeded
  // entirely from state the previous step already produced, so there is no dedicated width-1
  // "seed" forward. That forward cost a flat ~22 ms and was why the original two-forward version
  // measured 41.6 tok/s - slower than not speculating at all.
  //
  // POSITION ALIGNMENT is the whole correctness story. mtp_draft_step(stack, id, pos0) consumes
  // the stack at pos0 together with the token at pos0 and emits the stack for pos0+1, so its token
  // prediction is for pos0+1. Hence two chain steps, not one:
  //
  //   A: stack@pos_-1 + token@pos_-1 -> stack@pos_   (head not needed: `next` is already known)
  //   B: stack@pos_   + token@pos_   -> d0, for pos_+1
  //
  // and the verify compares logits row 0 - which predicts pos_+1 - against d0.
  //
  // An earlier version seeded only B with the tap at pos_-1 and compared row 0 against a draft for
  // pos_ itself. That is a one-position misalignment, and it is what produced the 36.6% "acceptance"
  // that was previously blamed on a draft-KV desync. There was no desync: run_chunk already refills
  // the draft KV for every position it processes, and each round's chain rewrites the previous
  // round's rejected positions with the true tokens.
  //
  // K=1 is the default depth, chosen by measurement rather than taste (8k, 7645-token prompt, 256
  // greedy tokens, 3 runs per cell; decode tok/s against 47.9 without speculation):
  //
  //   K=1  56.5  65.2% of slots, 1.65 tok/step   +17.6%
  //   K=2  51.4  47.3%, 1.95 tok/step             +7.2%
  //   K=3  45.7  38.7%, 2.16 tok/step             -4.6%
  //
  // The depth-1 rate decays geometrically (0.65, then ~0.29, then ~0.22), so past K=1 the verify
  // buys tokens slower than it buys width. HELIOS_SPEC_K overrides it for A/B-ing.
  const double t0 = now_ms();
  const Config& c = m_->cfg;
  // The draft head and the logits it writes both live on the last layer's card.
  cudaStream_t s = last_stream();
  AttnScratch& as = c.mtp_dev() == 0 ? attn_ : attn1_;
  MoeScratch& ms = c.mtp_dev() == 0 ? moe_ : moe1_;
  Act& act = last_dev_ == 0 ? act0_ : act1_;   // the tap, carry tap and logits are the last card's
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.mtp_dev()).phys_idx()));
  // One embedding row per draft call, gathered where the table is (see mtp_stage_embed). Null when
  // the head and the embedding are on the same card, which is the shipped layout.
  auto emb_for = [&](const int* ids, int n) {
    return c.mtp_dev() == c.layer_dev(0) ? nullptr : mtp_stage_embed(ids, n, s);
  };

  // Requested draft depth K, clamped to what this build and the context support. head_rows_/logits_
  // were sized to HELIOS_SPEC_K+1 at init; the verify batch is [next, d0..d_{K-1}] at width K+1.
  int K = 1;
  if (const char* ke = getenv("HELIOS_SPEC_K")) K = atoi(ke);
  if (K > kMaxSpecDrafts) K = kMaxSpecDrafts;
  if (K < 1) K = 1;
  // The verify forward consumes K+1 positions and, on a partial accept, gdn_replay re-runs 1+m of
  // them, so the tail must fit the context with room for the replay.
  // Bounds are the SLOT's rows, not the whole cache: a speculative batch that ran past the slot's
  // partition would write KV rows belonging to the next conversation.
  if (K > 1 && pos_ + K + 1 > slot_ctx_) K = 1;

  if (!mtp_on_ || !c.has_mtp || !carry_valid_ || pos_ + K + 1 > slot_ctx_) {
    decode({next});
    emit[0] = next;
    last_committed_ = next;   // `next` now occupies pos_-1
    capture_carry();
    tm_.decode_ms += now_ms() - t0;
    tm_.decode_tokens += 1;
    return 1;
  }

  // A: advance the carry tap to pos_. Its token prediction is discarded - `next` is already the
  // true token at pos_, so only the stream stack matters.
  {
    int id = last_committed_;
    mtp_draft_step(*m_, mtp_, as, ms, act.carry_tap, &id, 1, pos_ - 1, kv_base(0, mtp_ord_),
                   kv_base(1, mtp_ord_), mtp_.tap_, nullptr, nullptr, /*stream_tap=*/true, s,
                   /*compute_head=*/false, emb_for(&id, 1));
  }
  // B..: chain the drafts. B consumes the true token at pos_ and predicts pos_+1 (d0); each later
  // step chains on the previous draft and predicts one position further. mtp_draft_step advances
  // mtp_.tap_ to the next position's stack, so the next iteration's explicit tap argument is the
  // stack this position needs, exactly as B's second argument was.
  int drafts[kMaxSpecDrafts];
  for (int i = 0; i < kMaxSpecDrafts; i++) drafts[i] = -1;
  {
    int id = next;
    for (int i = 0; i < K; i++) {
      mtp_draft_step(*m_, mtp_, as, ms, mtp_.tap_, &id, 1, pos_ + i, kv_base(0, mtp_ord_),
                     kv_base(1, mtp_ord_), mtp_.tap_, nullptr, logits_ptr() + (size_t)c.vocab,
                     /*stream_tap=*/true, s, /*compute_head=*/true, emb_for(&id, 1));
      row_argmax(logits_ptr() + (size_t)c.vocab, 1, c.vocab, mtp_.ids_dev_, s);
      HELIOS_CUDA_CHECK(
          cudaMemcpyAsync(&drafts[i], mtp_.ids_dev_, sizeof(int), cudaMemcpyDeviceToHost, s));
      id = drafts[i];   // the next chain step consumes the token just drafted
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    }
  }

  const int n = spec_verify(next, drafts, K, emit);
  tm_.spec_steps++;
  tm_.spec_slots += K;
  tm_.decode_ms += now_ms() - t0;
  tm_.decode_tokens += n;
  return n;
}

int Runner::prefix_match(const std::vector<int>& prompt) const {
  const size_t n = std::min(prompt.size(), hist_tokens_.size());
  size_t i = 0;
  while (i < n && prompt[i] == hist_tokens_[i]) i++;
  return (int)i;
}

// ---------------------------------------------------------------- cross-request prefix cache
//
// The KV cache is position-addressed: a row written at position p is only ever read by a query at a
// position >= p, so the rows a previous request left behind are still the right answer for a new
// request that starts with the same tokens. The recurrent state is not - the GDN matrices, the
// PLE's dilated conv and the PLE's host n-gram window are each a function of the whole prefix - so a
// resume is only sound from a position whose state was captured.
//
// One ring of captures, taken at every prefill chunk boundary, in PINNED HOST memory. The interval
// is the prefill chunk by construction (a boundary IS a multiple of max_chunk_), and that is a
// CORRECTNESS constraint rather than a convenience: a prefill cuts its chunks at multiples of
// max_chunk_ from 0 whatever the prompt's length, so resuming from a multiple of max_chunk_ replays
// exactly the chunk sequence - and therefore exactly the floating-point reduction order - that a
// full prefill of the same prompt would have used. Resuming at any other position would be a
// different summation, not the same answer computed faster.
//
// Host rather than device memory: card 0 is down to ~1 GB free at the default 262144 context, a
// capture is 116 MB, and 8 of them would not fit. A 116 MB D2H against a 1024-token prefill chunk
// (measured ~3.5 s) is ~12 ms, i.e. under 0.4%, and it rides the aux stream so it overlaps the next
// chunk rather than serialising behind it.
void Runner::prefix_config(int max_chunk) {
  const char* e = getenv("HELIOS_PREFIX_CACHE");
  pfx_on_ = e && atoi(e) != 0;
  if (!pfx_on_) return;
  const Config& c = m_->cfg;
  // The QSA indexer's pooled-block counters (n_blocks / n_tail) are a running state that reset()
  // does not clear, and the sparse path is opt-in and off by default. Rather than reason about a
  // second snapshot format for a feature that is already documented as unverified end to end, the
  // cache refuses to run alongside it.
  if (getenv("HELIOS_QSA")) {
    fprintf(stderr, "[prefix] HELIOS_PREFIX_CACHE=1 refused: HELIOS_QSA is set and the indexer's "
                    "pooled-block state is not part of the snapshot\n");
    pfx_on_ = false;
    return;
  }
  const int n_gdn = (int)gdn_rec_.size();
  pfx_interval_ = max_chunk;
  int slots = 8;
  if (const char* se = getenv("HELIOS_PREFIX_SLOTS")) slots = atoi(se);
  slots = std::max(1, std::min(slots, 256));
  pfx_stride_ = (size_t)n_gdn * (gdn_conv_bytes_ + gdn_rec_bytes_)
              + (size_t)c.hc_dim * ((4 - 1) * c.ngram_size) * 2;
  pfx_off_.resize((size_t)n_gdn);
  for (int i = 0; i < n_gdn; i++) {
    pfx_off_[i] = (size_t)i * (gdn_conv_bytes_ + gdn_rec_bytes_);
    pfx_dev_[gdn_dev_[i]] = true;
  }
  pfx_ple_off_ = (size_t)n_gdn * (gdn_conv_bytes_ + gdn_rec_bytes_);
  pfx_dev_[c.layer_dev(c.ple_layer)] = true;
  // One ring, PARTITIONED per slot. A capture describes the conversation that took it, so two
  // conversations must not be able to plan a resume from each other's: their recurrent state and
  // their KV rows belong to different token histories, and mixing them answers from the wrong
  // context without failing. Each slot therefore gets its own pfx_slots_ captures, and pfx_pos_ is
  // per-slot too (bind_slot swaps it), so a slot only ever sees its own positions. With one slot
  // this is the identical byte count the cache has always allocated.
  const size_t bytes = pfx_stride_ * (size_t)slots * (size_t)n_slots_;
  if (cudaMallocHost((void**)&pfx_host_, bytes) != cudaSuccess) {
    fprintf(stderr, "[prefix] HELIOS_PREFIX_CACHE=1: pinned alloc of %.0f MB failed; disabled\n",
            (double)bytes / 1048576.0);
    pfx_host_ = nullptr;
    pfx_on_ = false;
    return;
  }
  // Events belong to the device whose stream records them, so they are created with that device
  // current rather than all at the end on card 0.
  for (int d = 0; d < N_GPU; d++) {
    if (!pfx_dev_[d]) continue;
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
    HELIOS_CUDA_CHECK(cudaEventCreateWithFlags(&pfx_done_[d], cudaEventDisableTiming));
    HELIOS_CUDA_CHECK(cudaEventCreateWithFlags(&pfx_copy_[d], cudaEventDisableTiming));
  }
  pfx_slots_ = slots;
  pfx_pos_.assign((size_t)slots, -1);
  printf("[prefix] cache ON: %d capture%s x %.0f MB pinned host, one every %d tokens (%d tokens of\n"
         "         reusable history per slot%s). HELIOS_PREFIX_SLOTS sets the depth; --chunk sets the\n"
         "         interval, and a smaller chunk also lowers the floor on what a resume can save.\n",
         slots, n_slots_ > 1 ? "s" : "", (double)pfx_stride_ / 1048576.0, pfx_interval_,
         slots * pfx_interval_, n_slots_ > 1 ? "" : "");
  if (n_slots_ > 1)
    printf("[prefix]   the ring holds %d slots x %d captures = %.0f MB, one range per sequence "
           "slot, so a conversation can only ever resume from its own captures.\n",
           n_slots_, slots, (double)bytes / 1048576.0);
}

// Capture the recurrent state as of position `pos` (a prefill chunk boundary). The state is final
// there: chunks write the state in order and nothing else is in flight, unlike a speculative verify
// batch whose tail may still be rolled back.
void Runner::pfx_capture(int pos) {
  if (!pfx_on_ || pos <= 0 || pos % pfx_interval_ != 0) return;
  // The ring index must be in range BEFORE it is used to address memory. An out-of-range index does
  // not fail loudly: it silently computes a host pointer outside the allocation, and the resulting
  // copy is either a CUDA "invalid argument" fault or, worse, a snapshot written over unrelated
  // pinned memory. Skipping the capture costs one interval of reusable history; addressing outside
  // the ring costs correctness.
  const int slot = pfx_next_;
  if (slot < 0 || slot >= pfx_slots_ || !pfx_host_ || active_slot_ < 0) {
    if (getenv("HELIOS_PREFIX_DEBUG"))
      fprintf(stderr, "[prefix] capture skipped: ring index %d out of [0,%d) or no host ring\n",
              slot, pfx_slots_);
    return;
  }
  // The host ring is ONE allocation shared by every slot, but a capture only describes the
  // conversation that took it: its recurrent state is that conversation's, and restoring it on top
  // of a different conversation's KV rows would be a silently wrong answer. So each slot owns a
  // disjoint range of the ring, sized pfx_slots_ per slot, and a slot can only ever plan a resume
  // from its own captures.
  char* base = pfx_host_ + ((size_t)active_slot_ * (size_t)pfx_slots_ + (size_t)slot) * pfx_stride_;
  // A slot is only free once the copy that last wrote it has landed. The previous capture is the
  // only one that can still be in flight (every capture is ordered behind the one before it), and
  // with a single slot it is this very slot.
  for (int d = 0; d < N_GPU; d++)
    if (pfx_dev_[d]) HELIOS_CUDA_CHECK(cudaEventSynchronize(pfx_copy_[d]));
  for (size_t i = 0; i < gdn_rec_.size(); i++) {
    const int d = gdn_dev_[i];
    Device& g = Engine::instance().gpu(d);
    HELIOS_CUDA_CHECK(cudaSetDevice(g.phys_idx()));
    cudaStream_t sc = g.stream(3);
    HELIOS_CUDA_CHECK(cudaEventRecord(pfx_done_[d], g.stream(0)));
    HELIOS_CUDA_CHECK(cudaStreamWaitEvent(sc, pfx_done_[d], 0));
    char* p = base + pfx_off_[i];
    HELIOS_CUDA_CHECK(
        cudaMemcpyAsync(p, gdn_conv_[i], gdn_conv_bytes_, cudaMemcpyDeviceToHost, sc));
    HELIOS_CUDA_CHECK(
        cudaMemcpyAsync(p + gdn_conv_bytes_, gdn_rec_[i], gdn_rec_bytes_, cudaMemcpyDeviceToHost, sc));
  }
  {
    const Config& c = m_->cfg;
    const int d = c.layer_dev(c.ple_layer);
    Device& g = Engine::instance().gpu(d);
    HELIOS_CUDA_CHECK(cudaSetDevice(g.phys_idx()));
    cudaStream_t sc = g.stream(3);
    HELIOS_CUDA_CHECK(cudaEventRecord(pfx_done_[d], g.stream(0)));
    HELIOS_CUDA_CHECK(cudaStreamWaitEvent(sc, pfx_done_[d], 0));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(base + pfx_ple_off_, ple_conv_,
                                      (size_t)c.hc_dim * ((4 - 1) * c.ngram_size) * 2,
                                      cudaMemcpyDeviceToHost, sc));
  }
  // The NEXT chunk overwrites this state, so its compute is held behind the copy. Without this the
  // copy and the next chunk's recurrence would race on the same bytes - and the failure would be a
  // silently wrong snapshot, i.e. a silently wrong answer for the next request.
  for (int d = 0; d < N_GPU; d++) {
    if (!pfx_dev_[d]) continue;
    Device& g = Engine::instance().gpu(d);
    HELIOS_CUDA_CHECK(cudaSetDevice(g.phys_idx()));
    HELIOS_CUDA_CHECK(cudaEventRecord(pfx_copy_[d], g.stream(3)));
    HELIOS_CUDA_CHECK(cudaStreamWaitEvent(g.stream(0), pfx_copy_[d], 0));
  }
  pfx_pos_[slot] = pos;
  pfx_next_ = (slot + 1) % pfx_slots_;
}

// Put back the state captured at `pos` and rewind everything else that is a function of position.
// The KV rows [0, pos) are NOT touched: they are position-addressed and already hold exactly what
// this prefix wrote last time.
void Runner::pfx_restore(int slot, int pos, const std::vector<int>& seq) {
  const Config& c = m_->cfg;
  // The host bytes have to be complete before they are read back, so this is a host wait - once per
  // request, before any compute, which is the one place a stall costs nothing.
  for (int d = 0; d < N_GPU; d++)
    if (pfx_dev_[d]) HELIOS_CUDA_CHECK(cudaEventSynchronize(pfx_copy_[d]));
  const char* base =
      pfx_host_ + ((size_t)active_slot_ * (size_t)pfx_slots_ + (size_t)slot) * pfx_stride_;
  for (size_t i = 0; i < gdn_rec_.size(); i++) {
    const int d = gdn_dev_[i];
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
    cudaStream_t s = Engine::instance().gpu(d).stream(0);
    const char* p = base + pfx_off_[i];
    HELIOS_CUDA_CHECK(
        cudaMemcpyAsync(gdn_conv_[i], p, gdn_conv_bytes_, cudaMemcpyHostToDevice, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(gdn_rec_[i], p + gdn_conv_bytes_, gdn_rec_bytes_,
                                      cudaMemcpyHostToDevice, s));
  }
  {
    const int d = c.layer_dev(c.ple_layer);
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(ple_conv_, base + pfx_ple_off_,
                                      (size_t)c.hc_dim * ((4 - 1) * c.ngram_size) * 2,
                                      cudaMemcpyHostToDevice, Engine::instance().gpu(d).stream(0)));
  }
  // The PLE's n-gram context is HOST state (the hashing never leaves the CPU), and run_chunk
  // truncates it to ngram_size-1 after every chunk, so at position `pos` it is exactly the last
  // ngram_size-1 tokens of the prefix - eos-padded at the start of a sequence, as reset() does it.
  const size_t keep = (size_t)(c.ngram_size - 1);
  hist_.assign(keep, c.ngram_eos);
  for (size_t k = 0; k < keep; k++) {
    const int idx = pos - (int)keep + (int)k;
    if (idx >= 0 && (size_t)idx < seq.size()) hist_[k] = seq[idx];
  }
}

// Start a request: pick the newest capture at or before the divergence point, restore it, and leave
// the runner at that position. Returns how many leading prompt tokens will not be recomputed.
int Runner::prefix_begin(const std::vector<int>& prompt) {
  prefix_reqs_++;
  resume_ = 0;
  tm_ = Timings{};   // one request's timings, whatever path it takes
  if (!pfx_on_) {
    reset();
    hist_tokens_.clear();
    return 0;
  }
  const int n = (int)prompt.size();
  const int common = prefix_match(prompt);
  // The rule itself lives in prefix.hpp with its own unit test (test_prefix): resume at the newest
  // capture at or below the divergence point, and drop every capture past it, because a capture
  // past the divergence point describes tokens this request has just disowned.
  const int best = prefix_plan(common, n, pfx_pos_);
  if (best < 0) {
    reset();
    hist_tokens_.clear();
  } else {
    pfx_restore(best, pfx_pos_[best], prompt);
    pos_ = pfx_pos_[best];
    hist_tokens_.assign(prompt.begin(), prompt.begin() + pos_);
  }
  resume_ = pos_;
  reuse_total_ += resume_;
  if (getenv("HELIOS_PREFIX_DEBUG"))
    fprintf(stderr, "[prefix] prompt=%d common=%d -> resume=%d (%s)\n", n, common, resume_,
            best < 0 ? "restart" : "snapshot");
  return resume_;
}

// Stop-id test over the engine's configured stop set, shared by generate() and generate_pair().
bool Runner::is_stop(int id) const {
  for (int s : stops_) if (s == id) return true;
  return false;
}

bool Runner::generate_pair(const std::vector<int>& pa, const std::vector<int>& pb,
                           const GenParams& p, std::vector<int>* out_a, std::vector<int>* out_b,
                           const std::function<bool(int)>& on_a,
                           const std::function<bool(int)>& on_b) {
  // Greedy only: the paired step reads both rows' argmax. p.greedy is set by the HTTP path's
  // parse_common, but the CLI builds GenParams directly, so accept temperature as the test too.
  if (mtp_on_ || n_slots_ < 2 || (p.greedy == false && p.temperature > 0.01f)) return false;
  // bind_slot() mutates active_slot_, so the ORIGINAL has to be captured here: reading it back after
  // the prefill loop returns whichever slot was bound last, and the pair then names the same slot
  // twice, which decode_pair correctly refuses.
  const int slot_a = active_slot_;
  const int slot_b = slot_a == 0 ? 1 : 0;
  out_a->clear(); out_b->clear();

  const Config& c = m_->cfg;
  std::vector<float> row((size_t)c.vocab);
  auto sample = [&](int r) {
    HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(last_dev_).phys_idx()));
    HELIOS_CUDA_CHECK(cudaMemcpy(row.data(), pair_logits(r), (size_t)c.vocab * 4,
                                cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(last_stream()));
    return next_token();   // argmax; paired generation is greedy-only
  };
  // Prefill each sequence in its own slot, reading its FIRST prediction from its OWN prefill: the
  // logits buffer is shared, so after the second prefill it holds the second sequence's row, and
  // taking both first tokens afterwards would feed sequence A's continuation of sequence B.
  int na = 0, nb = 0;
  for (int which = 0; which < 2; which++) {
    const std::vector<int>& q = which ? pb : pa;
    const int sl = which ? slot_b : slot_a;
    bind_slot(sl);
    reset();
    prefill(q, 0);
    if (which) nb = sample(0); else na = sample(0);
    save_slot(sl);
  }
  bind_slot(slot_a);
  for (int step = 0; step < p.max_tokens; step++) {
    if (is_stop(na) || is_stop(nb)) break;
    out_a->push_back(na); out_b->push_back(nb);
    if (on_a && !on_a(na)) break;
    if (on_b && !on_b(nb)) break;
    if (step + 1 == p.max_tokens) break;
    if (!decode_pair(na, nb, slot_a, slot_b)) break;
    na = sample(0);
    nb = sample(1);
  }
  return true;
}

std::vector<int> Runner::generate(const std::vector<int>& prompt, const GenParams& p,
                                  const std::function<bool(int)>& on_token) {
  if (prompt.empty()) return {};
  const Config& c = m_->cfg;
  // Against the SLOT's capacity, not the cache's: with N slots a conversation may only use its own
  // ctx_cap/N rows, and a prompt that overflowed them would run into the next slot's KV - a wrong
  // answer for two conversations rather than a crash.
  if (prompt.size() >= (size_t)slot_ctx_) {
    fprintf(stderr, "[gen] prompt %zu exceeds this sequence slot's context %d"
                    " (of %d total across %d slot%s)\n",
            prompt.size(), slot_ctx_, ctx_cap_, n_slots_, n_slots_ == 1 ? "" : "s");
    return {};
  }
  // Recompute only what the caches cannot answer. With HELIOS_PREFIX_CACHE off this is the reset()
  // and the full prefill it has always been; with it on, a prompt sharing a prefix with the resident
  // one resumes from the newest capture at or before the divergence point.
  const int resume = prefix_begin(prompt);
  prefill(prompt, resume);
  hist_tokens_ = prompt;

  Sampler smp;
  smp.reset(p.seed ? p.seed : 1234);
  // The repetition penalty must not touch the stop/EOS ids: on a reasoning model those include the
  // token that closes the think block, and penalising it leaves the model unable to finish its
  // reasoning, so it runs to max_tokens and returns an empty answer. Set per request from the same
  // set the stop check uses, so a checkpoint declaring several stop ids is handled uniformly.
  GenParams sp = p;
  // A checkpoint may declare several stop ids (generation_config's eos_token_id can be a list), and
  // missing one means generation never terminates at the turn boundary.
  auto is_stop = [this](int id) {
    for (int s : stops_)
      if (s == id) return true;
    return false;
  };
  const bool greedy = p.greedy || p.temperature <= 0.f;
  std::vector<int> recent = prompt, out;
  std::vector<float> row((size_t)c.vocab);

  // After the prefill, logits row 0 predicts the first token. Each iteration emits that token and
  // then advances the sequence past it, so the next prediction is never read twice.
  int next = next_token();
  while ((int)out.size() < p.max_tokens && pos_ < slot_ctx_) {
    if (is_stop(next)) break;
    out.push_back(next);
    recent.push_back(next);
    if (getenv("HELIOS_IDS")) fprintf(stderr, "[ids] %d\n", next);
    if (on_token && !on_token(next)) break;
    if ((int)out.size() >= p.max_tokens || pos_ >= slot_ctx_) break;

    // MTP speculative decode, on by default (see Runner::init for the measured A/B). A context gate
    // (mtp_ctx_limit_) can cap it by length and defaults to INT_MAX, because K=1 pays at every
    // context measured. HELIOS_MTP_CTX_LIMIT imposes a ceiling (0 = never speculate).
    const bool speculate = mtp_on_ && greedy && (pos_ <= mtp_ctx_limit_);
    if (speculate) {
      // The draft head is a greedy proposer, so it only accelerates greedy decoding: at
      // temperature > 0 the sampled token would differ from the drafted one by construction.
      int emit[kMaxSpecDrafts + 1] = {-1, -1, -1, -1, -1, -1, -1, -1};
      int cnt = spec_step(next, emit);
      hist_tokens_.push_back(next);
      // emit[0] is `next`, already reported. emit[1..cnt-1] are the accepted drafts, committed by
      // the spec step but not yet reported - each goes through the same stop / callback / limit
      // checks as any other emitted token.
      for (int i = 1; i < cnt; i++) {
        const int t = emit[i];
        if (t < 0) break;
        if (is_stop(t)) { cnt = i; break; }
        // HELIOS_IDS must list the token STREAM, not the speculative steps: an accepted draft is a
        // committed token like any other, and skipping it made the diagnostic's id list disagree
        // with the text it was supposed to explain.
        if (getenv("HELIOS_IDS")) fprintf(stderr, "[ids] %d\n", t);
        out.push_back(t);
        if (on_token && !on_token(t)) return out;
        if ((int)out.size() >= p.max_tokens || pos_ >= slot_ctx_) return out;
      }
      // The token after the LAST committed position. A spec step can commit 1..K+1 positions and
      // the head rows that predicted them do not all survive, so it publishes the answer rather
      // than leaving next_token() to read whatever row happens to be left.
      next = next_tok_ >= 0 ? next_tok_ : next_token();
    } else {
      decode({next});            // commits this token; row 0 now predicts the one after it
      hist_tokens_.push_back(next);
      if (greedy) {
        next = next_token();
      } else {
        HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(last_dev_).phys_idx()));
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(row.data(), logits_ptr(), (size_t)c.vocab * 4,
                                          cudaMemcpyDeviceToHost, last_stream()));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(last_stream()));
        // Suspend the repetition penalty while the model is still inside its think block. A
        // reasoning model narrates for hundreds of tokens by design, and a penalty over that window
        // makes it circle instead of closing: measured, rep_penalty 1.0 answers after 266 characters
        // of reasoning, while 1.1 leaves it at 1,857 characters with content "" - an empty answer for
        // a client that set nothing more than a mild repetition penalty. Exempting the stop/EOS ids
        // was tried first and did NOT fix it, which is what pointed at the window rather than the
        // closing token. The penalty is applied to the visible answer only, which is what the setting
        // is for. Uses OutputParser::in_think(), which until now had no caller.
        GenParams tp = sp;
        if (in_think()) tp.rep_penalty = 1.0f;
        next = smp.sample(row.data(), c.vocab, tp, recent);
      }
    }
  }
  return out;
}

void Runner::prefill(const std::vector<int>& ids, int from) {
  const double t0 = now_ms();
  // `off` indexes the WHOLE prompt, not the remainder, so a capture taken at a chunk boundary can
  // read the prompt tokens on either side of it - which is what restores the PLE's n-gram window.
  for (size_t off = (size_t)from; off < ids.size(); off += (size_t)max_chunk_) {
    const int n = (int)std::min((size_t)max_chunk_, ids.size() - off);
    std::vector<int> chunk(ids.begin() + off, ids.begin() + off + n);
    run_chunk(chunk, pos_);
    pos_ += n;
    // A chunk boundary is a multiple of max_chunk_, which is exactly where a resume has to land.
    pfx_capture(pos_);
  }
  sync_all();
  tm_.prefill_ms += now_ms() - t0;
  tm_.prefill_tokens += (int)ids.size() - from;   // tokens actually run, so tok/s stays honest
  // hist_tokens_ is owned by generate(): prefix_begin() truncates it to the resume point and
  // generate() assigns the whole prompt after this returns. Appending here as well would duplicate
  // the prompt and corrupt prefix_match() for every later request.
  if (mtp_on_ && !ids.empty()) {
    last_committed_ = ids.back();   // final prompt token sits at pos_-1
    capture_carry();
  }
}

void Runner::decode(const std::vector<int>& ids) {
  const double t0 = now_ms();
  for (size_t off = 0; off < ids.size(); off += (size_t)max_chunk_) {
    const int n = (int)std::min((size_t)max_chunk_, ids.size() - off);
    std::vector<int> chunk(ids.begin() + off, ids.begin() + off + n);
    run_chunk(chunk, pos_);
    pos_ += n;
    trace_state("dec", pos_);
  }
  sync_all();
  tm_.decode_ms += now_ms() - t0;
  tm_.decode_tokens += (int)ids.size();
}

// Two sequences, one row each, one forward. Everything that carries per-sequence recurrent state
// (GDN conv + recurrence, the PLE conv and n-gram window, the KV base and position) is addressed per
// row; everything that is already per-token over n (MoE, the mHC mixers, the GDN projections) simply
// sees n=2 and does the work once.
bool Runner::decode_pair(int tok_a, int tok_b, int slot_a, int slot_b) {
  // Speculation is excluded deliberately. Its verify batch already runs the trunk at width 2, so a
  // paired step would have to nest two of those, and its rollback snapshots (gdn_sub_snap_,
  // gdn_conv_snap_) are single-slot - rewinding a partial accept would restore the wrong sequence's
  // state. The win is available without it; this is where to start.
  if (mtp_on_ || n_slots_ < 2 || slot_a == slot_b) return false;
  if (pos_ >= slot_ctx_ || slots_[(size_t)slot_b].pos >= slot_ctx_) return false;

  const double t0 = now_ms();
  BatchCtx b;
  b.bsz = 2;
  b.slot[0] = slot_a;  b.pos[0] = pos_;
  b.slot[1] = slot_b;  b.pos[1] = slots_[(size_t)slot_b].pos;
  // history_stride = 1: the recurrent kernel computes slots[bi] * history_stride * state_size, and
  // one state_size is exactly the distance between two slots of the same layer in this layout.
  for (int d = 0; d < 2; d++) b.slot_layers[d] = 1;
  // Slot numbers are RELATIVE to row 0's slot, because the kernels scale them by the per-slot layer
  // count from the base pointer they are given (row 0's). The conv kernel has no layer term, so its
  // indices arrive pre-scaled.
  const int d0 = slot_b - slot_a;
  for (int d = 0; d < 2; d++) {
    if (!gdn_slots_dev_[d]) return false;
    // A slot difference is exactly ONE state, because the block is carved [layer_rank][slot] - two
    // slots of the same layer are adjacent. Multiplying by the per-card layer count (which is what
    // this did first) addresses a whole card's worth of layers and lands 2 bytes past the block,
    // which is how the conv kernel's 2-byte overread presented.
    const int hs[2] = {0, d0};
    const int cs[2] = {0, d0};
    cudaStream_t sd = Engine::instance().gpu(d).stream(0);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(gdn_slots_dev_[d], hs, sizeof(hs), cudaMemcpyHostToDevice, sd));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(conv_slots_dev_[d], cs, sizeof(cs), cudaMemcpyHostToDevice, sd));
    b.gdn_slots[d] = gdn_slots_dev_[d];
    b.conv_slots[d] = conv_slots_dev_[d];
  }

  // head_rows = 2: a pair needs BOTH rows' logits. The LAST projected position lands in row 0 (that
  // is the convention next_token() relies on), so for ids = [tok_a, tok_b] row 0 holds slot_b's
  // prediction and row 1 holds slot_a's. Getting this backwards silently swaps the two sequences'
  // continuations, which is exactly what the parity harness caught.
  std::vector<int> ids{tok_a, tok_b};
  run_chunk(ids, pos_, /*head_rows=*/2, &b);
  sync_all();
  // Both rows committed one token, so both sequences advance. Row 0's state lives in the bound
  // members; row 1's is already in its slot, which is why nothing is copied back here.
  pos_ += 1;
  slots_[(size_t)slot_b].pos += 1;
  slots_[(size_t)slot_b].last_committed = tok_b;
  trace_state("pair", pos_);
  tm_.decode_ms += now_ms() - t0;
  tm_.decode_tokens += 2;
  return true;
}

void Runner::harness_fused2_ab(int n) {
  const Config& c = m_->cfg;
  const int Nv = c.gdn_v_heads;
  cudaStream_t st = Engine::instance().gpu(c.layer_dev(0)).stream(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.layer_dev(0)).phys_idx()));
  const Layer& L = m_->layers[0];
  const GdnWeights& w = L.gdn;
  const size_t an = (size_t)n * Nv;
  std::vector<float> ha(an), hb(an);
  for (size_t i = 0; i < an; i++) {
    ha[i] = (float)((i * 2654435761u) % 1009) / 1009.0f - 0.5f;
    hb[i] = (float)((i * 40503u + 13u) % 1013) / 1013.0f - 0.5f;
  }
  float *da = nullptr, *db = nullptr, *dg = nullptr;
  __nv_bfloat16* dbeta = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&da, an * 4));
  HELIOS_CUDA_CHECK(cudaMalloc(&db, an * 4));
  HELIOS_CUDA_CHECK(cudaMalloc(&dg, an * 4));
  HELIOS_CUDA_CHECK(cudaMalloc(&dbeta, an * 2));
  HELIOS_CUDA_CHECK(cudaMemcpy(da, ha.data(), an * 4, cudaMemcpyHostToDevice));
  HELIOS_CUDA_CHECK(cudaMemcpy(db, hb.data(), an * 4, cudaMemcpyHostToDevice));
  // exactly how the layer calls it: B = 1, S = n
  aux::gated_delta_net_fused_op_2(db, da, (const __nv_bfloat16*)w.dt_bias, w.A_log, false, dbeta, dg,
                                  1, n, Nv, 1.0f, st);
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(st));
  std::vector<float> g2(an);
  std::vector<__nv_bfloat16> b2(an);
  HELIOS_CUDA_CHECK(cudaMemcpy(g2.data(), dg, an * 4, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(b2.data(), dbeta, an * 2, cudaMemcpyDeviceToHost));
  // and as two B=1, S=1 calls
  std::vector<float> g1(an, 0.f);
  std::vector<__nv_bfloat16> b1(an, __nv_bfloat16(0.f));
  for (int b = 0; b < n; b++) {
    aux::gated_delta_net_fused_op_2(db + (size_t)b * Nv, da + (size_t)b * Nv,
                                    (const __nv_bfloat16*)w.dt_bias, w.A_log, false,
                                    dbeta + (size_t)b * Nv, dg + (size_t)b * Nv, 1, 1, Nv, 1.0f, st);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<float> tg((size_t)Nv);
    std::vector<__nv_bfloat16> tb((size_t)Nv);
    HELIOS_CUDA_CHECK(cudaMemcpy(tg.data(), dg + (size_t)b * Nv, (size_t)Nv * 4, cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaMemcpy(tb.data(), dbeta + (size_t)b * Nv, (size_t)Nv * 2, cudaMemcpyDeviceToHost));
    for (int i = 0; i < Nv; i++) { g1[(size_t)b * Nv + i] = tg[i]; b1[(size_t)b * Nv + i] = tb[i]; }
  }
  double wg = 0, wb = 0;
  for (size_t i = 0; i < an; i++) {
    wg = std::max(wg, (double)fabs(g2[i] - g1[i]));
    wb = std::max(wb, (double)fabs(__bfloat162float(b2[i]) - __bfloat162float(b1[i])));
  }
  fprintf(stderr, "[fused2] n=%d (B=1,S=%d) vs %d x (B=1,S=1): g diff %.6e  beta diff %.6e  %s\n",
          n, n, n, wg, wb, (wg == 0.0 && wb == 0.0) ? "IDENTICAL" : "*** DIFFERS ***");
}

void Runner::harness_proj_ab(int n) {
  const Config& c = m_->cfg;
  const int D = c.hidden, z_out = c.gdn_z_out, Nv = c.gdn_v_heads;
  GdnScratch& gs = gdn_;
  cudaStream_t s = Engine::instance().gpu(c.layer_dev(0)).stream(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.layer_dev(0)).phys_idx()));
  const Layer& L = m_->layers[0];
  const GdnWeights& w = L.gdn;

  std::vector<half> hx((size_t)n * D);
  for (size_t i = 0; i < hx.size(); i++) {
    float v = (float)((i * 1103515245u + 12345u) % 10007) / 10007.0f - 0.5f;
    hx[i] = __float2half(v * 0.1f);
  }
  half* d_x = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&d_x, hx.size() * 2));
  HELIOS_CUDA_CHECK(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));

  // --- z projection: exl3::linear into the shared Hadamard scratch, n=2 vs 2x n=1 ---
  std::vector<float> z2((size_t)n * z_out), z1((size_t)n * z_out, 0.f);
  exl3::GroupWords zg{(const uint16_t*)w.z.trellis, w.z.suh, w.z.svh, w.z.mul1};
  exl3::linear(gs.z_flat, d_x, zg, n, z_out, D, w.z.K, /*y_fp32=*/true, s, gs.a_had);
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  HELIOS_CUDA_CHECK(cudaMemcpy(z2.data(), gs.z_flat, z2.size() * 4, cudaMemcpyDeviceToHost));
  for (int b = 0; b < n; b++) {
    exl3::linear(gs.z_flat, d_x + (size_t)b * D, zg, 1, z_out, D, w.z.K, true, s, gs.a_had);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> row((size_t)z_out);
    HELIOS_CUDA_CHECK(cudaMemcpy(row.data(), gs.z_flat, row.size() * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < z_out; i++) z1[(size_t)b * z_out + i] = row[i];
  }
  double wz = 0;
  for (size_t i = 0; i < z2.size(); i++) wz = std::max(wz, (double)fabs(z2[i] - z1[i]));
  fprintf(stderr, "[projab] z_proj  n=%d vs 2x n=1: worst %.6e  %s\n", n, wz,
          wz == 0.0 ? "IDENTICAL" : "*** DIFFERS ***");

  // --- a/b gemv ---
  for (int which = 0; which < 2; which++) {
    const half* wt = (const half*)(which ? (const void*)w.b_proj : (const void*)w.a_proj);
    std::vector<float> p2((size_t)n * Nv), p1((size_t)n * Nv, 0.f);
    aux::gdn_ba_gemv((const half*)d_x, wt, nullptr, which ? gs.b_out : gs.a_out, n, D, Nv, s);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    float* dst = which ? gs.b_out : gs.a_out;
    HELIOS_CUDA_CHECK(cudaMemcpy(p2.data(), dst, p2.size() * 4, cudaMemcpyDeviceToHost));
    for (int b = 0; b < n; b++) {
      aux::gdn_ba_gemv((const half*)(d_x + (size_t)b * D), wt, nullptr, dst, 1, D, Nv, s);
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      std::vector<float> row((size_t)Nv);
      HELIOS_CUDA_CHECK(cudaMemcpy(row.data(), dst, row.size() * 4, cudaMemcpyDeviceToHost));
      for (int i = 0; i < Nv; i++) p1[(size_t)b * Nv + i] = row[i];
    }
    double wq = 0;
    for (size_t i = 0; i < p2.size(); i++) wq = std::max(wq, (double)fabs(p2[i] - p1[i]));
    fprintf(stderr, "[projab] %s_gemv n=%d vs 2x n=1: worst %.6e  %s\n", which ? "b" : "a", n, wq,
            wq == 0.0 ? "IDENTICAL" : "*** DIFFERS ***");
  }
}

void Runner::harness_norm_rows_check(int rows_max, int dim) {
  cudaStream_t s = Engine::instance().gpu(0).stream(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
  const size_t big = (size_t)rows_max * dim;
  std::vector<__nv_bfloat16> hx(big), hg(big);
  for (size_t i = 0; i < big; i++) {
    float a = (float)((i * 2654435761u) % 10007) / 10007.0f - 0.5f;
    float b = (float)((i * 40503u + 7u) % 9973) / 9973.0f - 0.5f;
    hx[i] = __float2bfloat16(a * 0.3f);
    hg[i] = __float2bfloat16(b);
  }
  std::vector<__nv_bfloat16> hw(dim);
  for (int i = 0; i < dim; i++) hw[i] = __float2bfloat16(0.1f * ((i % 5) - 2));
  __nv_bfloat16 *dx = nullptr, *dg = nullptr, *dw = nullptr;
  half* dy = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&dx, big * 2));
  HELIOS_CUDA_CHECK(cudaMalloc(&dg, big * 2));
  HELIOS_CUDA_CHECK(cudaMalloc(&dw, (size_t)dim * 2));
  HELIOS_CUDA_CHECK(cudaMalloc(&dy, big * 2));
  HELIOS_CUDA_CHECK(cudaMemcpy(dx, hx.data(), big * 2, cudaMemcpyHostToDevice));
  HELIOS_CUDA_CHECK(cudaMemcpy(dg, hg.data(), big * 2, cudaMemcpyHostToDevice));
  HELIOS_CUDA_CHECK(cudaMemcpy(dw, hw.data(), (size_t)dim * 2, cudaMemcpyHostToDevice));

  // reference: one launch covering every row
  HELIOS_CUDA_CHECK(cudaMemset(dy, 0, big * 2));
  aux::gated_rms_norm(dx, dw, aux::kBFloat16, dy, aux::kHalf, dg, aux::kBFloat16,
                      rows_max, dim, 1e-6f, /*constant_bias=*/0.0f, /*w_groups=*/1,
                      /*gate_first=*/false, /*gate_act=*/1, s);
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<half> ref(big);
  HELIOS_CUDA_CHECK(cudaMemcpy(ref.data(), dy, big * 2, cudaMemcpyDeviceToHost));

  for (int rows : {1, 2, 3, rows_max / 2, rows_max - 1}) {
    if (rows < 1 || rows > rows_max) continue;
    HELIOS_CUDA_CHECK(cudaMemset(dy, 0, big * 2));
    aux::gated_rms_norm(dx, dw, aux::kBFloat16, dy, aux::kHalf, dg, aux::kBFloat16,
                        rows, dim, 1e-6f, 0.0f, 1, false, 1, s);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> got(big);
    HELIOS_CUDA_CHECK(cudaMemcpy(got.data(), dy, big * 2, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int r = 0; r < rows; r++)
      for (int i = 0; i < dim; i++)
        worst = std::max(worst, (double)fabs(__half2float(got[(size_t)r * dim + i]) -
                                            __half2float(ref[(size_t)r * dim + i])));
    fprintf(stderr, "[normchk] rows=%-4d of %d: worst diff vs the %d-row launch = %.6e  %s\n",
            rows, rows_max, rows_max, worst, worst == 0.0 ? "IDENTICAL" : "*** ROW-DEPENDENT ***");
  }
}

double Runner::harness_linear_ab(int n) {
  const Config& c = m_->cfg;
  const int D = c.hidden, qkv_out = c.gdn_qkv_out;
  GdnScratch& gs = gdn_;
  cudaStream_t s = Engine::instance().gpu(c.layer_dev(0)).stream(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.layer_dev(0)).phys_idx()));
  const Layer& L = m_->layers[0];
  const GdnWeights& w = L.gdn;

  std::vector<half> h_x((size_t)n * D);
  for (size_t i = 0; i < h_x.size(); i++) {
    float v = (float)((i * 1103515245u + 12345u) % 1000) / 500.0f - 1.0f;
    h_x[i] = __float2half(v * 0.1f);
  }
  half* d_x = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
  HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));

  std::vector<float> o2((size_t)n * qkv_out), o1((size_t)n * qkv_out);
  HELIOS_CUDA_CHECK(cudaMemset(gs.qkv_flat, 0, (size_t)n * qkv_out * 4));
  exl3::GroupWords qkv{(const uint16_t*)w.qkv.trellis, w.qkv.suh, w.qkv.svh, w.qkv.mul1};
  exl3::linear(gs.qkv_flat, d_x, qkv, n, qkv_out, D, w.qkv.K, /*y_fp32=*/true, s, gs.a_had);
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  HELIOS_CUDA_CHECK(cudaMemcpy(o2.data(), gs.qkv_flat, o2.size() * 4, cudaMemcpyDeviceToHost));
  for (int b = 0; b < n; b++) {
    exl3::linear(gs.qkv_flat, d_x + (size_t)b * D, qkv, 1, qkv_out, D, w.qkv.K,
                 /*y_fp32=*/true, s, gs.a_had);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> row((size_t)qkv_out);
    HELIOS_CUDA_CHECK(cudaMemcpy(row.data(), gs.qkv_flat, row.size() * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < qkv_out; i++) o1[(size_t)b * qkv_out + i] = row[i];
  }
  double worst = 0;
  for (size_t i = 0; i < o2.size(); i++)
    worst = std::max(worst, (double)fabs(o2[i] - o1[i]));
  fprintf(stderr, "[linab] exl3::linear(+Hadamard) n=2 vs 2x n=1: worst abs diff %.6e  %s\n", worst,
          worst == 0.0 ? "IDENTICAL" : "*** DIFFERS ***");
  return worst;
}

double Runner::harness_gdn_layer_ab(int n) {
  const Config& c = m_->cfg;
  const int D = c.hidden;
  GdnScratch& gs = gdn_;
  cudaStream_t s = Engine::instance().gpu(c.layer_dev(0)).stream(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(c.layer_dev(0)).phys_idx()));

  // Two independent rows with the same shape as a paired decode, and two rows' worth of state.
  std::vector<half> h_x((size_t)n * D);
  for (size_t i = 0; i < h_x.size(); i++) {
    float v = (float)((i * 1103515245u + 12345u) % 1000) / 500.0f - 1.0f;
    h_x[i] = __float2half(v * 0.1f);
  }
  half* d_x = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
  HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));

  const size_t rec_bytes = (size_t)c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim * 4;
  const size_t conv_bytes = (size_t)c.gdn_qkv_out * c.gdn_conv_k * 2;
  const size_t blk = 2 * (rec_bytes + conv_bytes);
  char* d_st_p = nullptr; char* d_st_s = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&d_st_p, blk));
  HELIOS_CUDA_CHECK(cudaMalloc(&d_st_s, blk));
  HELIOS_CUDA_CHECK(cudaMemset(d_st_p, 0, blk));
  HELIOS_CUDA_CHECK(cudaMemset(d_st_s, 0, blk));

  float* d_yp = nullptr; float* d_ys = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&d_yp, (size_t)n * D * 4));
  HELIOS_CUDA_CHECK(cudaMalloc(&d_ys, (size_t)n * D * 4));
  HELIOS_CUDA_CHECK(cudaMemset(d_yp, 0, (size_t)n * D * 4));
  HELIOS_CUDA_CHECK(cudaMemset(d_ys, 0, (size_t)n * D * 4));

  const Layer& L = m_->layers[0];   // the first GDN layer
  const int rel[2] = {0, 1};
  int* d_rel = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&d_rel, 2 * sizeof(int)));
  HELIOS_CUDA_CHECK(cudaMemcpy(d_rel, rel, 2 * sizeof(int), cudaMemcpyHostToDevice));

  char* rp = d_st_p; char* rs = d_st_s;
  void* recp = rp; void* convp = rp + rec_bytes;
  void* recs = rs; void* convs = rs + rec_bytes;
  // Bisect inside the layer: snapshot the conv's output and the recurrence's output from the bsz=2
  // call, then run the two n=1 calls and compare the same buffers. Both kernels are bit-identical in
  // isolation, so whichever of these first disagrees names the stage.
  // HELIOS_AB_SERIAL_FIRST runs the two n=1 calls BEFORE the bsz=2 one. Both orderings write the
  // same shared GdnScratch, so if the difference were contamination from whichever ran first, reversing
  // the order would change the result. If it does not, scratch reuse is not the confound and the
  // difference is in the bsz=2 invocation itself.
  const bool serial_first = getenv("HELIOS_AB_SERIAL_FIRST") != nullptr;
  if (serial_first) {
    for (int b = 0; b < n; b++) {
      gdn_layer(L.gdn, c, gs, d_x + (size_t)b * D, d_ys + (size_t)b * D,
                (char*)convs + (size_t)b * conv_bytes, (float*)((char*)recs + (size_t)b * rec_bytes),
                1, s, 1, nullptr, nullptr, 0);
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    }
  }
  // HELIOS_AB_POISON fills the scratch with garbage, using each member's REAL size from
  // gdn_scratch_init, before the layer runs. If the output changes, the layer reads a scratch buffer
  // it does not itself write - the one explanation left now that every call it makes is individually
  // verified correct. A previous attempt used one uniform size, overran the small members and failed
  // with "invalid argument" in an unrelated kernel, which says nothing either way.
  if (getenv("HELIOS_AB_POISON")) {
    const size_t nn = (size_t)gs.max_n, qo = 10240, zo = 6144, nv = 48;
    std::vector<char> junk(nn * qo * 4 + 64, (char)0x5a);
    auto poison = [&](void* p, size_t bytes) {
      if (p) HELIOS_CUDA_CHECK(cudaMemcpy(p, junk.data(), bytes, cudaMemcpyHostToDevice));
    };
    poison(gs.qkv_flat, nn * qo * 4);
    poison(gs.z_flat,   nn * zo * 4);
    poison(gs.a_out,    nn * nv * 4);
    poison(gs.b_out,    nn * nv * 4);
    poison(gs.g,        nn * nv * 4);
    poison(gs.mixed_qkv, nn * qo * 2);
    poison(gs.z16,      nn * zo * 2);
    poison(gs.beta,     nn * nv * 2);
    poison(gs.conv_out, nn * qo * 2);
    poison(gs.core_out, nn * zo * 2);
    poison(gs.normed,   nn * zo * 2);
    poison(gs.a_had,    nn * 16384 * 2);
  }
  gdn_layer(L.gdn, c, gs, d_x, d_yp, convp, (float*)recp, n, s, n, d_rel, d_rel, 1);
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  const size_t conv_n = (size_t)c.gdn_qkv_out * 1;
  // Every stage is captured at the SAME shape in both paths, assembled per row. mixed_qkv and
  // conv_out are channel-major [qkv_out, n] and bf16, so a 1-row call leaves ONE column and row b's
  // column belongs at offset b*qkv_out. The previous version read mixed_qkv as fp32 (it is bf16) and
  // compared transposed buffers without assembling - two errors, both of which read as "these
  // stages differ".
  const size_t qo = (size_t)c.gdn_qkv_out;          // qkv_out, the transposed buffers' row count
  const size_t zo = (size_t)c.gdn_z_out;            // Nv * Hv per row
  // sc.qkv_flat is [n, qkv_out] fp32 - row-major, so assembly is trivial, and it is the one link
  // between an identical input and a differing mixed_qkv that was never actually captured: the
  // standalone projection A/B used its own input and its own Hadamard scratch, not the layer's.
  std::vector<float> qf_p(qo * n), qf_s(qo * n, 0.f);
  HELIOS_CUDA_CHECK(cudaMemcpy(qf_p.data(), gs.qkv_flat, qf_p.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> mix_p(qo * n), conv_p(qo * n), core_p(zo * n);
  HELIOS_CUDA_CHECK(cudaMemcpy(mix_p.data(), gs.mixed_qkv, mix_p.size() * 2, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(conv_p.data(), gs.conv_out, conv_p.size() * 2, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(core_p.data(), gs.core_out, core_p.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> mix_s(qo * n, 0), conv_s(qo * n, 0), core_s(zo * n, 0);
  std::vector<half> norm_p(zo * n), norm_s(zo * n, 0);   // sc.normed is fp16, not bf16
  HELIOS_CUDA_CHECK(cudaMemcpy(norm_p.data(), gs.normed, norm_p.size() * 2, cudaMemcpyDeviceToHost));
  for (int b = 0; serial_first ? false : b < n; b++) {
    gdn_layer(L.gdn, c, gs, d_x + (size_t)b * D, d_ys + (size_t)b * D,
              (char*)convs + (size_t)b * conv_bytes, (float*)((char*)recs + (size_t)b * rec_bytes),
              1, s, 1, nullptr, nullptr, 0);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> qf(qo);
    HELIOS_CUDA_CHECK(cudaMemcpy(qf.data(), gs.qkv_flat, qo * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < qo; i++) qf_s[(size_t)b * qo + i] = qf[i];
    std::vector<__nv_bfloat16> t(qo);
    HELIOS_CUDA_CHECK(cudaMemcpy(t.data(), gs.mixed_qkv, qo * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < qo; i++) mix_s[(size_t)b * qo + i] = t[i];
    HELIOS_CUDA_CHECK(cudaMemcpy(t.data(), gs.conv_out, qo * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < qo; i++) conv_s[(size_t)b * qo + i] = t[i];
    std::vector<__nv_bfloat16> u(zo);
    HELIOS_CUDA_CHECK(cudaMemcpy(u.data(), gs.core_out, zo * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < zo; i++) core_s[(size_t)b * zo + i] = u[i];
    std::vector<half> v(zo);
    HELIOS_CUDA_CHECK(cudaMemcpy(v.data(), gs.normed, zo * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < zo; i++) norm_s[(size_t)b * zo + i] = v[i];
  }
  auto cmp = [&](const char* tag, const std::vector<__nv_bfloat16>& a,
                 const std::vector<__nv_bfloat16>& b2) {
    double w = 0;
    for (size_t i = 0; i < a.size(); i++)
      w = std::max(w, (double)fabs(__bfloat162float(a[i]) - __bfloat162float(b2[i])));
    fprintf(stderr, "[gdnab] %-12s worst %.6e  %s\n", tag, w, w == 0.0 ? "IDENTICAL" : "*** DIFFERS ***");
  };
  {
    double w = 0;
    for (size_t i = 0; i < qf_p.size(); i++) w = std::max(w, (double)fabs(qf_p[i] - qf_s[i]));
    fprintf(stderr, "[gdnab] %-12s worst %.6e  %s\n", "qkv_flat", w,
            w == 0.0 ? "IDENTICAL" : "*** DIFFERS ***");
  }
  cmp("mixed_qkv", mix_p, mix_s);
  cmp("conv_out", conv_p, conv_s);
  cmp("core_out", core_p, core_s);
  {
    double w = 0;
    for (size_t i = 0; i < norm_s.size(); i++)
      w = std::max(w, (double)fabs(__half2float(norm_p[i]) - __half2float(norm_s[i])));
    fprintf(stderr, "[gdnab] %-12s worst %.6e  %s\n", "normed", w,
            w == 0.0 ? "IDENTICAL" : "*** DIFFERS ***");
  }
  double wc = 0, wk = 0;
  for (size_t i = 0; i < conv_p.size(); i++)
    wc = std::max(wc, (double)fabs(__bfloat162float(conv_p[i]) - __bfloat162float(conv_s[i])));
  for (size_t i = 0; i < core_p.size(); i++)
    wk = std::max(wk, (double)fabs(__bfloat162float(core_p[i]) - __bfloat162float(core_s[i])));
  fprintf(stderr, "[gdnab] after conv_out worst %.6e | after core_out worst %.6e\n", wc, wk);
  std::vector<float> gp((size_t)n * D), gsv((size_t)n * D);
  HELIOS_CUDA_CHECK(cudaMemcpy(gp.data(), d_yp, gp.size() * 4, cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(gsv.data(), d_ys, gsv.size() * 4, cudaMemcpyDeviceToHost));
  double worst = 0;
  for (size_t i = 0; i < gp.size(); i++) {
    double dd = fabs((double)gp[i] - (double)gsv[i]);
    if (dd > worst) worst = dd;
  }
  return worst;
}

void Runner::harness_state_ptr_gap(int a, int b, int layer) const {
  const char* pa = (const char*)slots_[(size_t)a].gdn_rec[layer];
  const char* pb = (const char*)slots_[(size_t)b].gdn_rec[layer];
  const size_t rec_bytes = (size_t)m_->cfg.gdn_v_heads * m_->cfg.gdn_v_dim * m_->cfg.gdn_k_dim * 4;
  fprintf(stderr, "[ptrgap] rec  layer %d: slots %d/%d %lld bytes apart, rec_bytes %zu  %s\n",
          layer, a, b, (long long)(pb - pa), rec_bytes,
          (size_t)(pb - pa) == rec_bytes ? "OK" : "*** MISMATCH ***");
  // The conv kernel's per-slot stride is dim * state_size = gdn_qkv_out * gdn_conv_k halves, which
  // is conv_bytes - so the same check applies, and a disagreement here would be invisible above.
  const char* ca = (const char*)slots_[(size_t)a].gdn_conv[layer];
  const char* cb = (const char*)slots_[(size_t)b].gdn_conv[layer];
  const size_t conv_bytes = (size_t)m_->cfg.gdn_qkv_out * m_->cfg.gdn_conv_k * 2;
  fprintf(stderr, "[ptrgap] conv layer %d: slots %d/%d %lld bytes apart, conv_bytes %zu  %s\n",
          layer, a, b, (long long)(cb - ca), conv_bytes,
          (size_t)(cb - ca) == conv_bytes ? "OK" : "*** MISMATCH ***");
}

double Runner::harness_state_gap(int slot_a, int slot_b, int layer) const {
  const size_t rec_bytes = (size_t)m_->cfg.gdn_v_heads * m_->cfg.gdn_v_dim * m_->cfg.gdn_k_dim * 4;
  const int d = gdn_dev_[layer];
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(d).phys_idx()));
  std::vector<char> a(rec_bytes), b(rec_bytes);
  HELIOS_CUDA_CHECK(cudaMemcpy(a.data(), slots_[(size_t)slot_a].gdn_rec[layer], rec_bytes,
                               cudaMemcpyDeviceToHost));
  HELIOS_CUDA_CHECK(cudaMemcpy(b.data(), slots_[(size_t)slot_b].gdn_rec[layer], rec_bytes,
                               cudaMemcpyDeviceToHost));
  double worst = 0;
  for (size_t i = 0; i < rec_bytes; i += 4) {
    float x, y;
    memcpy(&x, a.data() + i, 4);
    memcpy(&y, b.data() + i, 4);
    double dd = fabs((double)x - (double)y);
    if (dd > worst) worst = dd;
  }
  return worst;
}

void Runner::harness_dump_last(const char* tag) const {
  const Act& a = last_dev_ == 0 ? act0_ : act1_;
  const int D = m_->cfg.hidden;
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(last_dev_).phys_idx()));
  for (int r = 0; r < 2; r++) {
    std::vector<half> raw((size_t)D);
    HELIOS_CUDA_CHECK(cudaMemcpy(raw.data(), a.sub_in + (size_t)r * D, (size_t)D * 2,
                                 cudaMemcpyDeviceToHost));   // D halves, NOT D floats
    double acc = 0;
    for (int i = 0; i < D; i++) {
      float v = __half2float(raw[i]);
      acc += (double)v * v;
    }
    fprintf(stderr, "[%s] last sub_in row %d rms=%.9e\n", tag, r, sqrt(acc / (double)D));
  }
}

int Runner::harness_argmax(int row) const {
  // logits_ptr() is a DEVICE pointer - reading it on the host segfaults. This is the same copy the
  // sampling path makes, and the sync is fine here because a parity check is not a hot path.
  const int V = m_->cfg.vocab;
  std::vector<float> host((size_t)V);
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(last_dev_).phys_idx()));
  HELIOS_CUDA_CHECK(cudaMemcpy(host.data(), logits_ptr() + (size_t)row * (size_t)V,
                               (size_t)V * 4, cudaMemcpyDeviceToHost));
  int best = 0;
  float bv = -INFINITY;
  for (int i = 0; i < V; i++)
    if (host[i] > bv) { bv = host[i]; best = i; }
  return best;
}

const float* Runner::pair_logits(int row) const {
  return logits_ptr() + (size_t)row * (size_t)m_->cfg.vocab;
}

}  // namespace helios
