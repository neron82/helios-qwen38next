#pragma once
// Qwen3.8 MoE layer: 512 experts top-10, shared expert with a sigmoid gate, no router bias.
//
// The routed experts are split *within every layer* across the two cards (GPU1 holds experts
// [0, experts_gpu1), GPU0 the rest), so one layer's MoE is two grouped launches whose fp32 outputs
// are summed. That split is forced by arithmetic, not preference: 48 x 512 x 1.19 MB = 29.2 GB of
// experts do not fit on one 24 GB card. With HELIOS_PIPELINE=1 the cut moves to the layer axis
// instead: each card holds all 512 experts but only for the layers it owns, and a layer's MoE
// becomes a single-card launch. Both cuts are the same total bytes; see Config::experts_per_card.
//
// The nine per-expert pointer tables are static (the arenas never move), so they are built once at
// init and indexed per layer - rebuilding them per forward would mean ~3.5 MB of H2O traffic on
// every token.
//
// Router semantics (ported from exllamav3_ext/routing.cu and verified by test): selection on the raw
// logit, weights = softmax over the *selected* top-k only, no bias, no routed scale, fp16 weights.
// The grouped kernel is called with mul1 = true (this checkpoint's codebook), act_limit = 0.0f and
// SiLU, matching what exllamav3 passes for this architecture (its BlockSparseMLP default act_limit
// is 0.0 for qwen4_exp - GLM's 10.0 was config-specific).

#include "core/model.hpp"
#include "cuda/cuda_shim.hpp"
#include <vector>
#include "cuda/quant/exl3_moe.cuh"

namespace helios {

struct MoeScratch {
  int max_n = 0;
  int concurrency = 8;
  int max_tokens_per_expert = 16;

  void* x16 = nullptr;                 // fp16 copy of the layer input on each card's device
  void* x16_1 = nullptr;
  half* scores = nullptr;              // [n, n_expert] router logits (card 0)
  // The router runs on card 0, but each card's permute reads these, and with no peer access a
  // pointer from the other card is not dereferenceable. Hence one copy per card: the router fills
  // card 0's, which is then staged to card 1 for its permute.
  int64_t* topk_idx[2] = {nullptr, nullptr};   // [n, topk]
  half* topk_w[2] = {nullptr, nullptr};        // [n, topk], one per card (see topk_idx)
  int64_t* remap[2] = {nullptr, nullptr};      // [n, topk] ids in one card's local expert space
  int64_t* counts[2] = {nullptr, nullptr};
  int64_t* sorted[2] = {nullptr, nullptr};
  half* wsorted[2] = {nullptr, nullptr};
  int64_t* perm_ws[2] = {nullptr, nullptr};
  float* part[2] = {nullptr, nullptr}; // per-card fp32 partial [n, hidden]
  // Pinned bounce for the two per-layer cross-card transfers. These GeForce boards are PHB with no
  // peer access (cudaDeviceCanAccessPeer is false), so the 10 KB input rows and the 20 KB partial
  // travel through host memory instead. At n = 1 the round trip is a few microseconds.
  float* bounce_f = nullptr;           // [max_n, hidden] fp32
  half* bounce_h = nullptr;            // [max_n, hidden] fp16
  // Orders reuse of the bounce buffers: a layer issues several cross-card copies and they share one
  // pinned pair, so each D2H must wait until the previous H2D has finished reading it.
  cudaEvent_t bounce_ev = nullptr;
  void* tsg[2] = {nullptr, nullptr}, *tsu[2] = {nullptr, nullptr};
  void* tig[2] = {nullptr, nullptr}, *tiu[2] = {nullptr, nullptr};
  // Hadamard scratch for the shared expert's three dense gemms (the grouped path carries its own).
  // Required by exl3::gemm, which always applies the input transform; needs room for m * k fp16.
  void* had = nullptr;                 // max_n * hidden * 2 bytes (card 0)
  void* had1 = nullptr;                // same, on card 1 (the decode path runs mgemm on both)
  // Scratch for the mgemm decode path, indexed by card.
  void* had_or(int c) const { return c == 0 ? had : had1; }
  int expert_K = 2;                    // trellis bits of the routed experts
  // Down-projection output of the mgemm decode path: one row per (token, top_k) SLOT, which
  // is topk times wider than part[c] (one row per token).
  float* dec_out = nullptr;            // [MOE_DECODE_MAX_N * topk, hidden] fp32, card 0
  float* dec_out1 = nullptr;           // same, card 1
  // mgemm takes x as [bszm_in, M, K]; the decode path uses M=1 and one gathered row per
  // (token, top_k) slot, exactly as the reference builds its run_bszN input.
  void* xgath = nullptr;               // [MOE_DECODE_MAX_N * topk, hidden] fp16, card 0
  void* xgath1 = nullptr;              // same, card 1
  // The shared expert runs entirely on card 0 (it is card 0's weight), so its fp32 output needs a
  // card-0 buffer: writing it into part[1] would be a cross-device write from a card-0 stream.
  float* shared_out = nullptr;         // [max_n, hidden] fp32
  float* part_sum = nullptr;           // [max_n, hidden] fp32: card 0's + card 1's partials

  // ---- n == 1 per-expert GEMV decode path (HELIOS_MOE_GEMV=1, default OFF) ----
  //
  // A host-side expert loop needs the router's selected ids and weights ON THE HOST to pick a
  // trellis pointer per launch, hence the pinned pair (one D2H per layer, then a stream sync).
  int64_t* gv_hidx = nullptr;         // pinned [topk]: this token's expert ids
  half* gv_hw = nullptr;              // pinned [topk]: their routing weights
  // Pinned per-card staging for the id-sorted weights. The reduce kernel reads them from the
  // device, so they have to cross the bus; one pinned buffer PER CARD so a card's next write can
  // never land on a copy that same card's stream has not issued yet (its next call syncs on that
  // stream first, the same ordering the two-card xcard_copy already relies on).
  half* gv_hw_stage[2] = {nullptr, nullptr};
  // Gate and up rows for all topk slots, laid out [topk, moe_inter] so the activation is ONE
  // silu_mul over topk * moe_inter instead of topk calls.
  void* gvg[2] = {nullptr, nullptr}, *gvu[2] = {nullptr, nullptr};
  half* gvw[2] = {nullptr, nullptr};  // [topk] fp16, the id-sorted weights the accumulate reads
  // Host mirror of the nine device pointer tables (moe_tables_init already builds a host vector per
  // field; the GEMV loop indexes it directly instead of dereferencing the device copy). ~1.8 MB
  // per card for 49 x 512 x 9 pointers.
  std::vector<const void*> table_host[2][9];
  // Layer-indexed device pointer tables, [n_layers + 1][E_local] per field per card.
  void* table_base[2][9] = {};
};

// Builds the static pointer tables from the model's resident expert arenas. Call once after load.
// `only_card` >= 0 builds just that card's table (pipeline mode: the scratch lives on one card and
// the layer's MoE is a single-card launch); -1 builds both, for the shipped two-card split.
bool moe_tables_init(Model& m, MoeScratch& sc, int device0, int device1, int only_card = -1);

// The mgemm decode path is used for short steps. It is the verified configuration: it agrees with
// the grouped path to 6 digits on the MoE output and removes the cooperative kernel's ~0.75 ms
// per-launch barrier cost from every decode step.
//
// Runtime-tunable via HELIOS_MOE_DECODE_MAX_N because the speculation work needs to know how the
// two paths amortise as n grows, and that is only answerable by measurement. The ported mgemm
// launcher faults at n = 6, so the value is clamped to 5 there. Default 2 keeps the verified
// decode configuration; prefill therefore uses the grouped path.
extern int g_moe_decode_max_n;

// True when a call at width n takes the device-side mgemm decode path, and can therefore sit
// inside a CUDA graph capture. False for the per-expert GEMV path (it reads the router's expert
// ids back to the host to pick a weight pointer per launch) and for the grouped path (it D2H-reads
// the permute counts). See the definition in moe_layer.cu and the scope argument in dgraph.hpp.
bool moe_decode_graphable(int n);

bool moe_scratch_init(MoeScratch& sc, const Config& cfg, int max_n, int device0, int device1);
void moe_scratch_free(MoeScratch& sc);

// x: hidden for this layer, fp16 [n, hidden]. y: fp32 [n, hidden] accumulator (zero-initialised
// inside). layer: tensor-namespace layer index; mtp selects the MTP head's own expert set.
//
// HELIOS_PIPELINE=1: `sc` must be the scratch built for the card that OWNS the layer (moe_ or
// moe1_), and the call becomes a single-card launch over that card's 512 experts - the same
// kernel, the same accumulation order inside the grouped kernel, minus the cross-card partial sum.
// moe_layer derives the owning card from cfg (cfg.layer_dev(layer), or cfg.mtp_dev() for the draft
// layer) and launches on the stream it is given, so the caller owns the device/stream binding.
void moe_layer(const MoeWeights& w, const Config& cfg, MoeScratch& sc, const half* x, float* y,
               int layer, bool mtp, int n, cudaStream_t s);

}  // namespace helios
