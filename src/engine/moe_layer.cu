// Qwen3.8 MoE layer. See moe_layer.cuh for the design and why the expert split is forced.
#include "engine/moe_layer.cuh"

#include "core/device.hpp"
#include <chrono>
#include "engine/glue.cuh"
#include "engine/glue2.cuh"
#include "cuda/aux/routing.cuh"
#include "cuda/aux/routing_std.cuh"
#include "cuda/aux/activation.cuh"
#include "cuda/quant/exl3_gemm.cuh"

#include <algorithm>
#include <vector>

namespace helios {

// Runtime gate for the mgemm decode path, sized from before any scratch exists. Read once at
// static-init so it cannot change under an already-allocated scratch.
//
// 0 disables mgemm entirely, which is the point: every decode measurement so far has been taken on
// the mgemm path, yet the REFERENCE does not use it at M=1 - it routes through exl3::gemm's QTIP
// GEMV. So "mgemm vs grouped at n=1" has never actually been measured, and that is the single
// largest unexplained term in the 45.4 vs 72.1 tok/s gap. 0 is a legal value precisely so it can
// be measured; 1..5 select mgemm, clamped at 5 because the ported launcher faults at n = 6.
// The DEFAULT tracks the widest production forward, which is the speculative verify width K+1.
// Hard-coding 2 was a live 43% cliff: a K=2 step is width 3, fell off mgemm onto the grouped kernel
// (measured 38.16 vs 54.66 tok/s), and nothing in the output said so - the tokens were correct,
// just slow. Deriving the default from the same HELIOS_SPEC_K the runner reads means the gate can
// no longer be narrower than the engine's own widest forward, and an explicit env value still wins
// for anyone deliberately measuring the grouped path.
static int read_decode_gate() {
  const char* e = getenv("HELIOS_MOE_DECODE_MAX_N");
  if (e) {
    int v = atoi(e);
    if (v < 0) v = 0;
    if (v > 5) v = 5;
    return v;
  }
  const char* k = getenv("HELIOS_SPEC_K");
  const int spec_k = k ? atoi(k) : 1;
  int v = spec_k + 1;          // a K-draft verify is a width-(K+1) forward
  if (v < 2) v = 2;            // never below the pre-existing default
  if (v > 5) v = 5;
  return v;
}
int g_moe_decode_max_n = read_decode_gate();

namespace {


// Remap router ids into one card's local expert space, or -1 if the expert lives on the other card.
// Needed because moe_permute produces *cumulative* counts, so a global permutation cannot simply be
// masked down to one card's experts.
__global__ void remap_ids_kernel(const int64_t* __restrict__ ids, int64_t* __restrict__ out,
                                 long n, int base, int count) {
  const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t e = ids[i];
  out[i] = (e >= base && e < base + count) ? e - base : -1;
}

}  // namespace

bool moe_tables_init(Model& m, MoeScratch& sc, int d0, int d1, int only_card) {
  const Config& c = m.cfg;
  // + MTP, which has its own expert set - but only when the checkpoint actually ships one, or
  // the table would address a layer the arena no longer reserves.
  const int n_layers = c.n_layers + (c.has_mtp ? 1 : 0);
  const int ncards = 2;
  sc.expert_K = m.expert_K;
  // Pipeline OFF: 160/352 slabs per layer, the within-layer split. Pipeline ON: all cfg.n_expert on
  // each card, for that card's own layers only.
  const int per[2] = {c.experts_per_card(0), c.experts_per_card(1)};
  const char* arena[2] = {m.experts0, m.experts1};
  // slab pieces: gate trellis/suh/svh = 0,1,2 ; up = 4,5,6 ; down = 8,9,10
  const int piece[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};

  // `only_card` (pipeline mode) builds ONE card's table, for a scratch that lives entirely on that
  // card: a layer's MoE is then a single-card launch over the 512 experts its own card holds, and
  // the other card's table would be a table of pointers into memory the host cannot dereference
  // from this device. Unset, both tables are built and the launch stays two-card.
  for (int card = 0; card < ncards; card++) {
    if (only_card >= 0 && card != only_card) continue;
    if (!arena[card] || per[card] <= 0) continue;
    const size_t bytes = (size_t)n_layers * per[card] * sizeof(void*);
    for (int f = 0; f < 9; f++) {
      void* dev = Engine::instance().gpu(card == 0 ? d0 : d1).alloc(bytes, 256);
      if (!dev) return false;
      std::vector<const void*> host((size_t)n_layers * per[card]);
      for (int l = 0; l < n_layers; l++) {
        // Pipeline ON: a card only holds the experts of ITS OWN layers, so a row for a layer it
        // does not own has no arena behind it. mgemm dereferences a whole row [per] wide, so those
        // rows point at this card's row 0 - a live arena slot. A launch against a card that does
        // not own the layer is then a wrong answer rather than an illegal access, and the
        // scheduler stage is what prevents it.
        const int src = (c.pipeline && c.layer_dev(l) != card) ? 0 : l;
        for (int e = 0; e < per[card]; e++)
          host[(size_t)l * per[card] + e] =
              arena[card] + ((size_t)c.expert_slot(src, card) * per[card] + e) * m.expert_stride +
              m.slab_off[piece[f]];
      }
      // The per-expert GEMV decode path indexes the trellis/suh/svh pointers from the HOST, once
      // per active expert per layer, so it needs them addressable without a D2H read per launch.
      // The vectors are already built here; keeping them costs ~1.8 MB of host memory per card and
      // is only ever populated (the default paths read table_base and never look at this).
      sc.table_host[card][f] = host;
      cudaSetDevice(Engine::instance().gpu(card == 0 ? d0 : d1).phys_idx());
      HELIOS_CUDA_CHECK(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice));
      sc.table_base[card][f] = dev;
    }
  }
  cudaSetDevice(Engine::instance().gpu(d0).phys_idx());
  return true;
}

bool moe_scratch_init(MoeScratch& sc, const Config& cfg, int max_n, int d0, int d1) {
  sc.max_n = max_n;
  const int E = cfg.n_expert, topk = cfg.topk, hid = cfg.hidden, inter = cfg.moe_inter;
  auto A = [&](int dev, size_t bytes) { return Engine::instance().gpu(dev).alloc(bytes, 256); };
  // The device maximum is num_sms / MOE_SMS_PER_EXPERT (10 on this 82-SM card); more groups
  // means each group walks fewer experts, and the kernel pays ~7 group barriers per expert.
  const int maxc = exl3::moe_max_concurrency(Engine::instance().gpu(d1).phys_idx());
  sc.concurrency = getenv("HELIOS_MOE_CONC") ? atoi(getenv("HELIOS_MOE_CONC")) : maxc;
  if (sc.concurrency > maxc) sc.concurrency = maxc;
  if (sc.concurrency < 1) sc.concurrency = 1;
  // Capacity, not a loop bound: the grouped kernel skips any expert whose token count
  // exceeds this (`if (token_count > max_tokens_per_expert) continue`). It must
  // therefore cover the worst case, which is all n tokens landing on one expert, or
  // prefill would silently drop those experts' contributions.
  sc.max_tokens_per_expert = max_n;

  sc.x16 = A(d0, (size_t)max_n * hid * 2);
  sc.x16_1 = A(d1, (size_t)max_n * hid * 2);
  sc.scores = (half*)A(d0, (size_t)max_n * E * 2);
  sc.topk_idx[0] = (int64_t*)A(d0, (size_t)max_n * topk * 8);
  sc.topk_idx[1] = (int64_t*)A(d1, (size_t)max_n * topk * 8);
  sc.topk_w[0] = (half*)A(d0, (size_t)max_n * topk * 2);
  sc.topk_w[1] = (half*)A(d1, (size_t)max_n * topk * 2);
  sc.remap[0] = (int64_t*)A(d0, (size_t)max_n * topk * 8);
  sc.remap[1] = (int64_t*)A(d1, (size_t)max_n * topk * 8);
  const int per[2] = {cfg.experts_per_card(0), cfg.experts_per_card(1)};
  for (int c = 0; c < 2; c++) {
    sc.counts[c] = (int64_t*)A(c, (size_t)(per[c] + 1) * 8);
    sc.sorted[c] = (int64_t*)A(c, (size_t)max_n * topk * 8);
    sc.wsorted[c] = (half*)A(c, (size_t)max_n * topk * 2);
    sc.perm_ws[c] = (int64_t*)A(c, (size_t)3 * (per[c] + 2) * 8);
    // The mgemm path memsets and writes one row per (token, top_k) SLOT into part[c], while the
    // grouped path writes one row per token. Sizing at max_n rows alone is safe only because the
    // default max_n (1024) happens to exceed top_k; set --chunk below top_k and it overran the
    // allocation. max_n dominates in every real configuration, so this costs nothing there.
    const size_t part_rows = (size_t)std::max(max_n, g_moe_decode_max_n * cfg.topk);
    sc.part[c] = (float*)A(c, part_rows * hid * 4);
    const size_t tsz = (size_t)sc.concurrency * sc.max_tokens_per_expert * hid * 2;
    const size_t isz = (size_t)sc.concurrency * sc.max_tokens_per_expert * inter * 2;
    sc.tsg[c] = A(c, tsz);
    sc.tsu[c] = A(c, tsz);
    sc.tig[c] = A(c, isz);
    sc.tiu[c] = A(c, isz);
  }
  sc.had = A(d0, (size_t)max_n * hid * 2);
  sc.had1 = A(d1, (size_t)max_n * hid * 2);
  sc.dec_out = (float*)A(d0, (size_t)g_moe_decode_max_n * cfg.topk * hid * 4);
  sc.dec_out1 = (float*)A(d1, (size_t)g_moe_decode_max_n * cfg.topk * hid * 4);
  sc.xgath = A(d0, (size_t)g_moe_decode_max_n * cfg.topk * hid * 2);
  sc.xgath1 = A(d1, (size_t)g_moe_decode_max_n * cfg.topk * hid * 2);
  sc.shared_out = (float*)A(d0, (size_t)max_n * hid * 4);
  sc.part_sum = (float*)A(d0, (size_t)max_n * hid * 4);
  HELIOS_CUDA_CHECK(cudaMallocHost(&sc.bounce_f, (size_t)max_n * hid * 4));
  HELIOS_CUDA_CHECK(cudaMallocHost(&sc.bounce_h, (size_t)max_n * hid * 2));
  HELIOS_CUDA_CHECK(cudaEventCreateWithFlags(&sc.bounce_ev, cudaEventDisableTiming));
  // n == 1 per-expert GEMV decode path (HELIOS_MOE_GEMV=1). Gate/up rows for all top_k slots are
  // laid out [top_k, moe_inter] so the activation is one silu_mul over top_k * moe_inter, and the
  // down results land in dec_out (already [decode_max_n * top_k, hidden] fp32) one row per slot,
  // which moe_gemv_reduce_k then folds in ascending slot order.
  //
  // Allocated unconditionally and sized for one token, so the gate can be flipped at any point in
  // a run without a re-init. had_or(c) is reused as the GEMV a_had: at n=1 it needs M*K = hidden
  // (gate/up) or moe_inter (down) halves, and it holds max_n * hidden, so it is always large
  // enough. had must not alias x or y, and it does not (it is a dedicated scratch).
  for (int c = 0; c < 2; c++) {
    sc.gvg[c] = A(c, (size_t)cfg.topk * inter * 2);
    sc.gvu[c] = A(c, (size_t)cfg.topk * inter * 2);
    sc.gvw[c] = (half*)A(c, (size_t)cfg.topk * 2);
  }
  for (int c = 0; c < 2; c++)
    HELIOS_CUDA_CHECK(cudaMallocHost(&sc.gv_hw_stage[c], (size_t)cfg.topk * 2));
  HELIOS_CUDA_CHECK(cudaMallocHost(&sc.gv_hidx, (size_t)cfg.topk * 8));
  HELIOS_CUDA_CHECK(cudaMallocHost(&sc.gv_hw, (size_t)cfg.topk * 2));
  return sc.scores && sc.topk_idx[0] && sc.topk_idx[1] && sc.counts[0] && sc.part[0] &&
         sc.part[1] && sc.bounce_f;
}

void moe_scratch_free(MoeScratch& sc) {
  if (sc.bounce_f) cudaFreeHost(sc.bounce_f);
  if (sc.bounce_h) cudaFreeHost(sc.bounce_h);
  if (sc.gv_hidx) cudaFreeHost(sc.gv_hidx);
  if (sc.gv_hw) cudaFreeHost(sc.gv_hw);
  for (int c = 0; c < 2; c++) cudaFreeHost(sc.gv_hw_stage[c]);
  sc = MoeScratch{};
}

// Device-to-device copy across the two cards. Peer access is unavailable on these PHB GeForce
// boards, so the fallback stages through pinned host memory. Both paths are asynchronous and
// serialised on the destination stream, so callers keep their existing ordering assumptions.
static void xcard_copy(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes,
                       void* bounce, cudaEvent_t ev, cudaStream_t s_dst) {
  Engine& e = Engine::instance();
  if (e.gpu(0).stats().p2p_to_other && e.gpu(1).stats().p2p_to_other) {
    HELIOS_CUDA_CHECK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, s_dst));
    return;
  }
  cudaStream_t s_src = e.gpu(src_dev == e.gpu(1).phys_idx() ? 1 : 0).stream(0);
  // The bounce buffer is shared by every copy the layer issues, so each copy must be complete
  // before the next one starts overwriting it. A per-device event cannot order these (the two
  // streams belong to different device contexts, which makes cudaStreamWaitEvent reject the handle
  // as an invalid resource), so the destination is drained here instead. That serialises the copies
  // - a few microseconds each, against a layer that is otherwise ~200us of work - and it is the
  // reason no expert assignment is lost.
  (void)ev;
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(bounce, src, bytes, cudaMemcpyDeviceToHost, s_src));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s_src));
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(dst, bounce, bytes, cudaMemcpyHostToDevice, s_dst));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s_dst));
}

namespace {
// Decode uses mgemm: one mgemm per projection per card over the raw (token, top_k) assignment slots,
// with no sort/dedup and no cross-block barriers. The grouped cooperative kernel has a ~0.75 ms fixed
// cost per launch, almost entirely barrier latency, and over 48 layers that is ~36 ms of a ~53 ms
// decode step - which is why this path is worth having.
//
// Verified numerically identical to the grouped path at the FIRST DECODE STEP, where both see the same
// input (comparing later steps is meaningless once the paths diverge):
//
//   layer 0, first decode step     grouped     mgemm
//     card 0                       0.010724    0.010723
//     card 1                       0.006499    0.006499
//
// and 38.3 tok/s against the grouped path's 28.4 on the same prompt, deterministic over repeated runs.
//
// The long-standing ~5% residual was a MISSING ARGUMENT, not arithmetic: the down call stopped at
// `c_ptrs` (24 positionals), so `mcg` and `mul1` took their defaults - and mul1 defaults to false,
// while the gate and up calls pass true. The down tables are expert tables of the same format, so that
// one projection was dequantising with the wrong scale.
//
// Two traps cost real time here, both worth remembering:
//   - The FIRST generated token is correct even on a broken path, because it comes from the last
//     prefill chunk and prefill always uses the grouped kernel. Only later tokens diverge, which reads
//     like a sampling or length problem rather than a wrong MoE.
//   - exl3::gemm is NOT a valid oracle for expert weights: it has no mul1 argument at all, so it
//     silently encodes the attention convention. Comparing mgemm against it "proves" agreement with
//     the wrong convention - which is exactly how a wrong fix got applied and then reverted here.
//     moe_grouped is the only valid reference for experts.
//
// HELIOS_MOE_GROUPED=1 forces the grouped path back on, for comparison.
bool moe_decode_mgemm() {
  static const bool on = getenv("HELIOS_MOE_GROUPED") == nullptr;
  return on;
}

// Fold the per-slot down-projection rows into one row per token. The mgemm down pass already
// applied the routing weight to each slot, so this is a plain sum over the token's top_k slots.
__global__ void moe_slot_reduce_k(const float* __restrict__ slot, float* __restrict__ y, int n,
                                  int topk, int hid) {
  const int t = blockIdx.x;
  const int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= hid) return;
  float acc = 0.f;
  for (int k = 0; k < topk; k++) acc += slot[((size_t)t * topk + k) * hid + i];
  y[(size_t)t * hid + i] = acc;
}

// Gather one hidden row per (token, top_k) slot so mgemm's bszm_in rows have distinct inputs.
__global__ void moe_slot_gather_k(const half* __restrict__ x, half* __restrict__ xg, int slots,
                                  int topk, int hid) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)slots * hid) return;
  const int slot = (int)(i / hid), d = (int)(i % hid);
  xg[i] = x[(size_t)(slot / topk) * hid + d];
}

// ---- n == 1 per-expert GEMV decode path -------------------------------------------------
//
// The reference's bsz-1 tier (BC_BlockSparseMLP::run_bszN) issues ONE exl3_mgemm per projection
// over all top_k slots, inside a CUDA graph. helios has no graph capture, so its mgemm decode path
// pays a fresh cooperative launch per projection per layer. The alternative modelled here is what
// the name says literally: a HOST-side loop over the token's active experts, each one a small
// forced exl3::gemv at M=1 (gate, up, silu, down), weight-scaled into the output row. There are no
// cooperative barriers anywhere in it, and the per-launch grid is tiny (a 640-wide GEMV is 20
// blocks), which is exactly the regime the QTIP kernel was written for.
//
// The cost is dispatch: top_k x 3 launches per layer, ~48 layers, every one of them a separate
// cudaLaunchCooperativeKernel plus a host-side pointer lookup. Whether that is cheaper than 3 mgemm
// launches per layer is an empirical question, so this path is OFF unless HELIOS_MOE_GEMV=1 and
// every measurement of it is reported (see RESULTS.md).
bool moe_decode_gemv() {
  static const bool on = getenv("HELIOS_MOE_GEMV") != nullptr;
  return on;
}


// out_row[i] = sum_k w[k] * rows[k * hid + i], with rows = the per-slot fp32 down-projection
// results. A separate kernel rather than an accumulate-into-y: the down GEMV writes its own row
// (y "need not be zeroed" is the same contract mgemm relies on), and one reduction afterwards
// keeps the accumulation order fixed at ascending slot index, so the result is bit-reproducible
// run to run - no atomics, no arrival order.
__global__ void moe_gemv_reduce_k(const float* __restrict__ rows, const half* __restrict__ w,
                                  float* __restrict__ y, int topk, int hid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hid) return;
  float acc = 0.f;
  for (int k = 0; k < topk; k++) acc += __half2float(w[k]) * rows[(size_t)k * hid + i];
  y[(size_t)blockIdx.y * hid + i] = acc;
}

// Named synchronisation for locating a fault: with HELIOS_DBG set, every stage is checked
// individually so the first failing launch is reported by name rather than by whichever later
// cudaPeekAtLastError happens to observe the sticky error.
// Cached diagnostic gates. These are read on every layer of every prefill chunk; getenv is a linear
// scan of environ, and the flags cannot change under a running engine, so each is read once.
namespace moe_diag {
  static const bool moec  = getenv("HELIOS_MOEC") != nullptr;
  static const bool moex  = getenv("HELIOS_MOEX") != nullptr;
  static const bool cnt   = getenv("HELIOS_CNT")  != nullptr;
}

bool dbg_stage() { static const bool on = getenv("HELIOS_DBG") != nullptr; return on; }
void chk_stage(const char* what, cudaStream_t s) {
  if (!dbg_stage()) return;
  cudaError_t e = cudaStreamSynchronize(s);
  if (e != cudaSuccess) {
    fprintf(stderr, "[moe] FAILED at %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}
}  // namespace

// Can this MoE call sit inside a CUDA graph capture? See dgraph.hpp for the scope argument.
//
// The mgemm decode path qualifies and needs nothing added to make it so: routing is already
// device-side. routing_std_logits writes the per-slot expert ids and weights into DEVICE memory,
// remap_ids_kernel maps them into this card's local expert space on device, and mgemm dereferences
// the device index array (negative entries skip the slot) and the device weight array. A capture
// therefore bakes in no routing decision - the only thing that varies per step is the CONTENT of
// those arrays, which is what a replay is for.
//
// The two paths that are NOT capturable, and why:
//   * The per-expert GEMV path (moe_decode_gemv, default off) D2H-reads the router's ids and
//     weights back to the host and sorts them there, because each per-expert launch needs that
//     expert's trellis/suh/svh pointers as arguments. That is a host branch on the routing, which
//     is precisely what a capture cannot contain - and there is no "capture once per distinct
//     expert set" that works, because the set changes every token.
//   * The grouped path is only reached at n > g_moe_decode_max_n (prefill widths), where the
//     permute/bincount also round-trips counts and sorted ids to the host.
// Both are excluded here rather than left to fail a capture at run time.
bool moe_decode_graphable(int n) {
  return !moe_decode_gemv() && moe_decode_mgemm() && n <= g_moe_decode_max_n;
}

void moe_layer(const MoeWeights& w, const Config& cfg, MoeScratch& sc, const half* x, float* y,
               int layer, bool mtp, int n, cudaStream_t s) {
  const int E = cfg.n_expert, topk = cfg.topk, hid = cfg.hidden, inter = cfg.moe_inter;
  // HELIOS_PIPELINE=1: this layer's MoE is a SINGLE-card launch. The card that owns the layer holds
  // all cfg.n_expert of that layer's experts, so there is nothing to combine: no per-card partial
  // sum, no xcard_copy of the hidden rows, and the output lands on the card that produced it. The
  // scratch handed in is the one built for that card (moe_tables_init was called with only_card =
  // own, so the other card's table is absent), and every per-card buffer below is indexed by `own`.
  // `own` is the LAYER's card for a trunk layer and cfg.mtp_dev() for the draft layer, which is the
  // same rule the loader used to place the experts.
  const int own = cfg.pipeline ? (mtp ? cfg.mtp_dev() : cfg.layer_dev(layer)) : -1;
  // How many experts each card holds for this layer, and which GLOBAL router ids those are. Off:
  // card 0 holds the tail [experts_gpu1, 512) and card 1 the head. On: the owning card holds all
  // 512 at their global ids, so the remap is the identity and no slot is dropped (-1).
  const int per[2] = {cfg.experts_per_card(0), cfg.experts_per_card(1)};
  const int base[2] = {cfg.expert_base(0), cfg.expert_base(1)};
  // -1 = unknown, which forces the kernel's narrow default group width (MOE_SMS_PER_EXPERT = 8).
  // A positive value widens them: num_groups = MIN(concurrency, num_active) and
  // group_size = MIN(num_sms/num_groups, MOE_MAX_SMS_PER_EXPERT). It is a grid-shape hint only - a
  // wrong value cannot change results, just speed - so it can be tuned before a real count is wired.
  // Cached: getenv is a linear scan of environ, and this runs on every layer of every chunk.
  // The flag cannot change under a running engine, so read it once.
  static const int k_num_active = getenv("HELIOS_MOE_ACTIVE") ? atoi(getenv("HELIOS_MOE_ACTIVE")) : -1;
  const int num_active = k_num_active;

  static double mp[7] = {0}; static long mn = 0;
  static const bool k_mprof = getenv("HELIOS_MPROF") != nullptr;
  const bool mprof = k_mprof;
  auto MNOW = []() { return std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now().time_since_epoch()).count(); };
  struct MOut { ~MOut() {
      if (k_mprof && mn > 0)   // k_mprof, not mprof: a local class cannot capture an automatic
        fprintf(stderr, "[mprof] per-layer ms rgemv=%.3f topk=%.3f permute=%.3f xfer=%.3f "
                        "grouped=%.3f finish=%.3f shared=%.3f (calls=%ld)\n",
                mp[0]/mn, mp[6]/mn, mp[1]/mn, mp[2]/mn, mp[3]/mn, mp[4]/mn, mp[5]/mn, mn);
    } } mout;
  // Drain before starting the clock. The first lap() syncs this stream, so without this it would
  // absorb whatever the previous stage left pending here - the attention/GDN work, ~0.21 ms/layer -
  // and report it as router time. That artifact made the router look like the largest MoE stage and
  // led to an optimisation aimed at a cost that was not there.
  double mt = 0;
  if (mprof) {
    cudaStreamSynchronize(s);
    mt = MNOW();
  }
  auto lap = [&](int i, cudaStream_t st) { if (mprof) { cudaStreamSynchronize(st);
      mp[i] += MNOW() - mt; mt = MNOW(); } };

  chk_stage("entry", s);
  // 1) router: gate is already [E, hidden], the orientation routing_gemv wants. No bias.
  aux::routing_gemv_batch(x, w.router, sc.scores, n, hid, E, s);
  lap(0, s);
  // Pipeline mode writes the owning card's index directly: there is no second card to stage to, and
  // the scratch's [own] arrays are the ones on that card (they are allocated per card index, not
  // per device argument). Flag off, `own` is -1 and this is index 0 exactly as before.
  const int rc = own >= 0 ? own : 0;
  aux::routing_std_logits(sc.scores, sc.topk_idx[rc], sc.topk_w[rc], n, E, topk, s);
  chk_stage("router", s);
  lap(6, s);

  // 2) one grouped launch per card over its own experts, into its own fp32 partial. Every call
  // below addresses a specific card, so the current device is set per card rather than left at
  // whatever the last allocation happened to select.
  cudaStream_t s1 = Engine::instance().gpu(0).stream(0);
  cudaStream_t s2 = Engine::instance().gpu(1).stream(0);
  const int dev0 = Engine::instance().gpu(0).phys_idx();
  const int dev1 = Engine::instance().gpu(1).phys_idx();
  // Single-card mode: every launch runs on the caller's stream, which the scheduler has already
  // bound to the owning card. The card's own compute stream would be the same object in the
  // shipped two-card path; going through the argument keeps the stage's stream the only one the
  // scheduler has to order.
  cudaStream_t stc[2] = {s1, s2};
  if (own >= 0) stc[own] = s;
  const int ncards = own >= 0 ? 1 : 2;
  // ---- n == 1 per-expert GEMV decode path (HELIOS_MOE_GEMV=1, default OFF) ----------------
  //
  // One forced exl3::gemv per active expert per projection: gate, up, silu_mul (one call over all
  // top_k rows), down, then a fixed-order weighted reduction into the card's fp32 partial. No
  // cooperative barrier, no sort/dedup, no permute - the top_k slots are simply walked.
  //
  // The expert list has to be on the HOST, because each launch needs that expert's trellis/suh/svh
  // pointers as arguments. That costs one D2H plus a stream sync per layer, which is why the gate
  // exists: the reference does not pay it because it keeps the slot list on the device and lets
  // one mgemm kernel resolve the pointers. Both facts are measured in RESULTS.md.
  // topk <= 64 bounds the three fixed host-side slot arrays below. A checkpoint above it would
  // silently smash the stack, so it falls through to the default path instead - a slower answer,
  // never a wrong one. (This checkpoint is topk = 10.)
  if (moe_decode_gemv() && n == 1 && topk <= 64) {
    // The router wrote to [rc]'s arrays on `s`. Read them back on the card that owns them.
    HELIOS_CUDA_CHECK(cudaSetDevice(own >= 0 ? (own == 0 ? dev0 : dev1) : dev0));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.gv_hidx, sc.topk_idx[rc], (size_t)topk * 8,
                                      cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.gv_hw, sc.topk_w[rc], (size_t)topk * 2,
                                      cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    // Fixed expert order: ascending global router id. Every downstream step (launch order, the
    // slot each result occupies, the reduction's summation order) is then a function of the
    // router output alone, never of pointer values or arrival order, so the path is bit-identical
    // run to run. Insertion sort: topk is 10 and this runs once per layer.
    int sel[64];
    half wt[64];
    for (int k = 0; k < topk; k++) { sel[k] = (int)sc.gv_hidx[k]; wt[k] = sc.gv_hw[k]; }
    for (int a = 1; a < topk; a++) {
      const int si = sel[a];
      const half wi = wt[a];
      int b = a - 1;
      while (b >= 0 && sel[b] > si) { sel[b + 1] = sel[b]; wt[b + 1] = wt[b]; b--; }
      sel[b + 1] = si; wt[b + 1] = wi;
    }
    // Pipeline mode: the activations are already on the owning card, and the scratch's [own]
    // arrays are that card's. Two-card mode: card 1 still needs the hidden row (10 KB at n=1).
    if (own < 0) {
      xcard_copy(sc.x16_1, dev1, x, dev0, (size_t)hid * 2, sc.bounce_h, sc.bounce_ev, s2);
    }
    for (int ci = 0; ci < ncards; ci++) {
      const int c = own >= 0 ? own : ci;
      if (per[c] <= 0) continue;
      HELIOS_CUDA_CHECK(cudaSetDevice(c == 0 ? dev0 : dev1));
      cudaStream_t st = stc[c];
      // this card's slice of the sorted slot list, and the weights that go with it
      int loc[64];
      int m = 0;
      for (int k = 0; k < topk; k++) {
        const int e = sel[k] - base[c];
        if (e >= 0 && e < per[c]) loc[m++] = e;
      }
      // A card with no selected expert contributes an exact zero, and the combine below sums both
      // partials - so the row has to be zeroed, not left holding the previous layer's numbers.
      // (The mgemm path gets this from the memset it does before its own launch.)
      if (m == 0) {
        HELIOS_CUDA_CHECK(cudaMemsetAsync(sc.part[c], 0, (size_t)hid * 4, st));
        continue;
      }
      // The reduce kernel reads the weights from device memory, so they go over as an async H2D
      // from this card's own pinned staging buffer, enqueued ahead of the kernel on the same
      // stream. Writing them straight into the device pointer from the host is a segfault - a
      // cudaMalloc'd range is not host-accessible.
      for (int k = 0; k < m; k++) sc.gv_hw_stage[c][k] = wt[k];
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.gvw[c], sc.gv_hw_stage[c], (size_t)m * 2,
                                        cudaMemcpyHostToDevice, st));
      const size_t off = (size_t)(mtp ? cfg.n_layers : layer) * per[c];
      const half* xin = (own >= 0 || c == 0) ? (const half*)x : (const half*)sc.x16_1;
      half* g = (half*)sc.gvg[c];
      half* u = (half*)sc.gvu[c];
      float* drows = (c == 0) ? sc.dec_out : sc.dec_out1;
      for (int k = 0; k < m; k++) {
        const size_t row = off + loc[k];
        const exl3::GroupWords gw{(const uint16_t*)sc.table_host[c][0][row],
                                  (const half*)sc.table_host[c][1][row],
                                  (const half*)sc.table_host[c][2][row], /*mul1=*/1};
        const exl3::GroupWords uw{(const uint16_t*)sc.table_host[c][3][row],
                                  (const half*)sc.table_host[c][4][row],
                                  (const half*)sc.table_host[c][5][row], /*mul1=*/1};
        const exl3::GroupWords dw{(const uint16_t*)sc.table_host[c][6][row],
                                  (const half*)sc.table_host[c][7][row],
                                  (const half*)sc.table_host[c][8][row], /*mul1=*/1};
        // a_had is the same scratch for all three: at M=1 the transform target is K halves, and
        // the three calls are stream-ordered so they never overlap. It must not alias x or y.
        void* a_had = sc.had_or(c);
        exl3::gemv(g + (size_t)k * inter, xin, gw, 1, inter, hid, sc.expert_K,
                   /*y_fp32=*/false, st, a_had);
        exl3::gemv(u + (size_t)k * inter, xin, uw, 1, inter, hid, sc.expert_K,
                   /*y_fp32=*/false, st, a_had);
      }
      // One activation over all m rows: silu(gate) * up, in place, act_limit 0 as everywhere else.
      // It must sit BETWEEN the gate/up GEMVs and the down GEMVs, since the down projection's
      // input is the activated row.
      aux::silu_mul(g, u, g, /*float_input=*/false, /*act_limit=*/0.0f, (size_t)m * inter, st);
      for (int k = 0; k < m; k++) {
        const size_t row = off + loc[k];
        const exl3::GroupWords dw{(const uint16_t*)sc.table_host[c][6][row],
                                  (const half*)sc.table_host[c][7][row],
                                  (const half*)sc.table_host[c][8][row], /*mul1=*/1};
        exl3::gemv(drows + (size_t)k * hid, g + (size_t)k * inter, dw, 1, hid, inter, sc.expert_K,
                   /*y_fp32=*/true, st, sc.had_or(c));
      }
      // Fixed-order weighted reduction: ascending slot index, so no atomics and no arrival order.
      // Rows beyond m are never read.
      moe_gemv_reduce_k<<<dim3((hid + 255) / 256, 1), 256, 0, st>>>(drows, sc.gvw[c], sc.part[c], m,
                                                                  hid);
    }
    chk_stage("gemv", own >= 0 ? stc[own] : s1);
    lap(3, own >= 0 ? stc[own] : s1);
    // Combine, exactly as the mgemm decode path does: one partial in pipeline mode, two summed on
    // card 0 otherwise.
    const int ec = own >= 0 ? own : 0;
    cudaStream_t so = own >= 0 ? stc[own] : s1;
    HELIOS_CUDA_CHECK(cudaSetDevice(ec == 0 ? dev0 : dev1));
    if (own >= 0) {
      glue::copy_f32(y, sc.part[own], (size_t)hid, so);
    } else {
      xcard_copy(sc.part_sum, dev0, sc.part[1], dev1, (size_t)hid * 4, sc.bounce_f, sc.bounce_ev,
                 s1);
      glue::add_f32_inplace(sc.part_sum, sc.part[0], (size_t)hid, s1);
      glue::copy_f32(y, sc.part_sum, (size_t)hid, s1);
    }
    lap(4, so);
    if (moe_diag::moec) {
      std::vector<float> t((size_t)hid);
      const float* routed = own >= 0 ? sc.part[own] : sc.part_sum;
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), routed, (size_t)hid * 4, cudaMemcpyDeviceToHost,
                                        so));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(so));
      double a = 0;
      for (float v : t) a += (double)v * v;
      fprintf(stderr, "[moec] L%d routed=%.9e (gemv)\n", layer, sqrt(a / t.size()));
    }
    if (moe_diag::moex) {
      // Elementwise fingerprint of the routed row, not its rms: rms is invariant under a sign flip
      // or a reordering of equal-magnitude entries, so two paths can agree on it to 9 digits and
      // still differ elementwise. These do not.
      std::vector<float> t((size_t)hid);
      const float* routed = own >= 0 ? sc.part[own] : sc.part_sum;
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), routed, (size_t)hid * 4, cudaMemcpyDeviceToHost,
                                        so));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(so));
      double sum = 0, asum = 0;
      for (int i = 0; i < hid; i++) { sum += (double)t[i] * (i + 1); asum += fabs((double)t[i]); }
      fprintf(stderr, "[moex] L%d cksum=%.12e asum=%.12e v0=%.9e v1=%.9e vlast=%.9e\n", layer, sum,
              asum, t[0], t[1], t[hid - 1]);
    }
    // Shared expert: identical dense path in both decode paths (gate/up -> silu -> down -> sigmoid
    // gate add), on the same card-local scratch the mgemm decode path uses.
    if (w.shared[0].trellis && w.shared_gate) {
      const exl3::GroupWords sg{(const uint16_t*)w.shared[0].trellis, w.shared[0].suh,
                                w.shared[0].svh, w.shared[0].mul1};
      const exl3::GroupWords su{(const uint16_t*)w.shared[1].trellis, w.shared[1].suh,
                                w.shared[1].svh, w.shared[1].mul1};
      const exl3::GroupWords sd{(const uint16_t*)w.shared[2].trellis, w.shared[2].suh,
                                w.shared[2].svh, w.shared[2].mul1};
      half* sh_g = (half*)sc.tsg[ec];
      half* sh_u = (half*)sc.tsu[ec];
      exl3::linear(sh_g, x, sg, 1, inter, hid, w.shared[0].K, false, so, sc.had_or(ec));
      exl3::linear(sh_u, x, su, 1, inter, hid, w.shared[1].K, false, so, sc.had_or(ec));
      aux::silu_mul(sh_g, sh_u, sh_g, false, 0.0f, (size_t)inter, so);
      exl3::linear(sc.shared_out, sh_g, sd, 1, hid, inter, w.shared[2].K, true, so, sc.had_or(ec));
      aux::add_sigmoid_gate_proj(sc.shared_out, x, y, w.shared_gate, 1, hid, so);
    }
    lap(5, so);
    if (mprof) mn++;
    return;
  }

  // ---- decode path: n small. One mgemm per projection per card over the raw (token, top_k)
  // assignment slots, with no sort/dedup and no cross-block barriers.
  //
  // This mirrors what the reference does for bsz 1..MAX_BSZN: "a single cooperative mgemm call per
  // projection across all bsz*top_k assignment slots (no sort/dedup -- overlap between tokens this
  // small is rare and not worth the argsort/bincount host-sync cost that the fused/exl3_moe path
  // pays)". The grouped cooperative kernel has a ~0.75 ms fixed cost per launch (measured: 0.77 ms
  // at n=1 vs 3.86 ms at n=200, i.e. almost entirely barrier latency), which over 48 layers is
  // ~36 ms of a ~53 ms decode step. mgemm has no barriers, and its range filter drops and rebases
  // the other card's experts, so the remap kernel is unnecessary here too.
  // NOTE: opt-in while the mgemm argument set is still being reconciled against the
  // reference's exl3_mgemm call (its positional list differs from the ported raw-pointer
  // form). The grouped path below is the verified one.
  if (moe_decode_mgemm() && n <= g_moe_decode_max_n) {
    const int slots = n * topk;
    // Both cards filter and weight their own slots against their own copy of the router output, but
    // routing_std_logits writes only card 0's arrays. Without this copy card 1 reads whatever the
    // scratch happened to hold - zeros, so every one of its slots selected expert 0 with weight 0 and
    // contributed nothing, silently dropping all per[1] experts (17 of 20 slots on this checkpoint).
    // The grouped path has always done this copy; the mgemm path's absence of it was the whole reason
    // its output diverged. Symptom to remember: the first generated token is right (it comes from the
    // grouped prefill), every token after it is not.
    // Pipeline mode: the router already wrote THIS card's index arrays and the activations are
    // already on this card, so there is nothing to stage and the whole block is skipped.
    if (own < 0) {
      // The destination stream must belong to the destination device (card 1), which is what the
      // grouped path does below: an async H2D into card 1's memory enqueued on card 0's stream is not
      // ordered against card 1's kernels.
      xcard_copy(sc.topk_idx[1], dev1, sc.topk_idx[0], dev0, (size_t)slots * 8, sc.bounce_h,
                 sc.bounce_ev, s2);
      xcard_copy(sc.topk_w[1], dev1, sc.topk_w[0], dev0, (size_t)slots * 2, sc.bounce_h,
                 sc.bounce_ev, s2);
      // ...and card 1 needs the activations as well: moe_slot_gather_k reads xin = sc.x16_1 on that
      // card. The grouped path copies it too; without it card 1 gathered from a buffer nothing had
      // written. Two missing copies, not one - the index copy alone left the output unchanged.
      xcard_copy(sc.x16_1, dev1, x, dev0, (size_t)n * hid * 2, sc.bounce_h, sc.bounce_ev, s2);
    }
    cudaStream_t st[2] = {s1, s2};
    const int devs[2] = {dev0, dev1};
    for (int ci = 0; ci < ncards; ci++) {
      const int c = own >= 0 ? own : ci;
      if (per[c] <= 0) continue;
      HELIOS_CUDA_CHECK(cudaSetDevice(devs[c]));
      st[c] = stc[c];
      const uint16_t** g_t = (const uint16_t**)sc.table_base[c][0] + (size_t)layer * per[c];
      const half** g_suh = (const half**)sc.table_base[c][1] + (size_t)layer * per[c];
      const half** g_svh = (const half**)sc.table_base[c][2] + (size_t)layer * per[c];
      const uint16_t** u_t = (const uint16_t**)sc.table_base[c][3] + (size_t)layer * per[c];
      const half** u_suh = (const half**)sc.table_base[c][4] + (size_t)layer * per[c];
      const half** u_svh = (const half**)sc.table_base[c][5] + (size_t)layer * per[c];
      const uint16_t** d_t = (const uint16_t**)sc.table_base[c][6] + (size_t)layer * per[c];
      const half** d_suh = (const half**)sc.table_base[c][7] + (size_t)layer * per[c];
      const half** d_svh = (const half**)sc.table_base[c][8] + (size_t)layer * per[c];
      half* gbuf = (half*)sc.tsg[c];          // [slots, inter]
      half* ubuf = (half*)sc.tsu[c];          // [slots, inter]
      half* xg = (half*)((c == 0) ? sc.xgath : sc.xgath1);
      // Single-card mode reads the caller's own activations; sc.x16_1 is only the staging copy the
      // two-card path needs for the far card.
      const half* xin = (own >= 0 || c == 0) ? (const half*)x : (const half*)sc.x16_1;
      moe_slot_gather_k<<<(unsigned)((slots * hid + 255) / 256), 256, 0, st[c]>>>(xin, xg, slots,
                                                                                 topk, hid);
      const half* xslot = xg;
      // Map the global router ids into this card's local expert space, with -1 for "other card".
      // mgemm's range filter must then stay OFF (min/max = -1 below): with min_index >= 0 and
      // num_tokens == 1 it compacts the retained indices and weights IN PLACE without permuting the
      // input rows, so slot j would pair its own activation with another slot's expert and weight.
      // A negative index already means "skip this slot", which is exactly the semantics needed here.
      remap_ids_kernel<<<(int)(((long)slots + 127) / 128), 128, 0, st[c]>>>(
          sc.topk_idx[c], sc.remap[c], (long)slots, base[c], per[c]);
      const int64_t* idx = sc.remap[c];
      // gate and up: per-slot outputs (weights null, so nothing is reduced)
      // M = 1 (one input row per slot), bszm_in = bszm_out = slots
      // mul1 MUST be true - the expert tables carry exl3's global per-tensor scale, which the grouped
      // kernel applies and which exl3::gemm has no argument for.
      //
      // I set this to false earlier and it was wrong. The probe was "does mgemm match exl3::gemm on
      // one matrix": gemm has no mul1, so mul1=false trivially matched it (cos 1.0000) while mul1=true
      // did not (cos -0.011). But gemm is the ATTENTION convention and the experts are not - the right
      // oracle is moe_grouped, which passes mul1=true. Comparing against gemm measured agreement with
      // the wrong convention. It also made the path worse, not better: the per-card values went from
      // 88-106% of the grouped ones to 54-70% at the first decode step, which is the opposite of what
      // "fixing the residual" should look like. Lesson: for expert weights, only moe_grouped is a
      // valid reference.
      // Tile shape for the mgemm's selector. Left at auto (0).
      //
      // Forcing shapes looked like a free win and is not: shape 4 "measured" 48.32 tok/s against auto's
      // 45.71, but it produces garbage - "We!!!!!!!!!!!!!!!!!!!" instead of the reference continuation -
      // and a broken kernel is fast because it stops doing the work. Shape 3 (47.92) was never checked
      // for correctness either. A speed sweep that does not verify output is worthless, and I ran one
      // here despite having criticised exactly that earlier in this log.
      static const int k_fshape = getenv("HELIOS_MOE_SHAPE") ? atoi(getenv("HELIOS_MOE_SHAPE")) : 0;
      const int fshape = k_fshape;
      // Optional explicit SM count rather than mgemm's auto path. Left at auto (0).
      //
      // Forcing it looked like +3.4% (47.26/47.19/46.98/47.35 for sms 8/16/24/40 against auto's
      // 45.60-45.85). It is not: a later auto run measured 47.27, inside the same band. Run-to-run
      // variance here is ~3.5%, because the exllamav3 server holds both GPUs and contention varies. A
      // 3% effect cannot be resolved with this methodology, so the setting is left alone. Any future
      // change at this scale needs many more repetitions or a quiet machine, not two runs.
      static const int k_fsms = getenv("HELIOS_MOE_SMS") ? atoi(getenv("HELIOS_MOE_SMS")) : 0;
      const int fsms = k_fsms;
      // Per-expert exl3::gemv was tried here as a way to dodge mgemm's M=1 inefficiency (it never
      // routes to the QTIP GEMV kernel that gemm has). It is numerically correct but SLOWER: 39.94 /
      // 40.24 tok/s against mgemm's 45.92 / 45.91. At 10 slots x 3 projections x 45 layers the dispatch
      // is 1350 launches per token, ~4 us each, which costs more than the bandwidth it recovers. So the
      // GEMV kernel is not the lever - either mgemm's own inner loop, or batching the dispatch (a graph),
      // would have to be.
      exl3::mgemm(gbuf, xslot, g_t, g_suh, g_svh, per[c], 1, inter, hid, sc.expert_K,
                  /*y_fp32=*/false, st[c], sc.had_or(c), /*bszm_in=*/slots, /*bszm_out=*/slots,
                  idx, nullptr, slots, 0, /*min_index=*/-1, /*max_index=*/-1, 1, nullptr, nullptr,
                  /*mcg=*/false, /*mul1=*/true, fshape, fsms);
      exl3::mgemm(ubuf, xslot, u_t, u_suh, u_svh, per[c], 1, inter, hid, sc.expert_K, false,
                  st[c], sc.had_or(c), slots, slots, idx, nullptr, slots, 0, -1, -1, 1, nullptr,
                  nullptr, /*mcg=*/false, /*mul1=*/true, fshape, fsms);
      aux::silu_mul(gbuf, ubuf, gbuf, /*float_input=*/false, 0.0f, (size_t)slots * inter, st[c]);
      // down: weighted reduction into one row per token
      // Slot j must consume row j of gbuf (its own activated intermediate). The kernel takes slot j's
      // input from `A + j * size_m * size_k`, so with M=1 and bszm_in=slots that address is
      // gbuf + j*inter - row j, which is what the per-slot layout needs.
      //
      // The two wrong forms both look plausible and neither faults. With M=slots, bszm_in=1 (what this
      // used to pass) the stride becomes slots*inter and every slot transforms the same block, so each
      // expert is applied to the wrong activations; with M=slots, bszm_in=slots the stride is right but
      // it runs past the end of gbuf. The output side is unchanged: num_tokens groups the slot rows
      // back into one row per token.
      // Slots owned by the other card carry idx = -1 and mgemm skips them, so their output rows are
      // never written - and the reduce below sums all top_k slots. fp32 output "need not be zeroed"
      // only holds when every slot is written; here the stale rows were being added, which is why the
      // path was non-deterministic run to run.
      // mul1 must be given EXPLICITLY here. This call used to stop at c_ptrs (24 positionals), so
      // mcg and mul1 both took their defaults - and mul1 defaults to false, while the gate and up
      // calls above pass true. The down tables are expert tables of the same format, so they need the
      // same convention; this one projection was silently dequantising with the wrong scale.
      HELIOS_CUDA_CHECK(cudaMemsetAsync(sc.part[c], 0, (size_t)slots * hid * 4, st[c]));
      exl3::mgemm(sc.part[c], gbuf, d_t, d_suh, d_svh, per[c], /*M=*/1, hid, inter, sc.expert_K,
                  /*y_fp32=*/true, st[c], sc.had_or(c), /*bszm_in=*/slots, /*bszm_out=*/slots, idx,
                  sc.topk_w[c], slots, 0, /*min_index=*/-1, /*max_index=*/-1, /*num_tokens=*/n,
                  nullptr, nullptr, /*mcg=*/false, /*mul1=*/true, fshape, fsms);
    }
    HELIOS_CUDA_CHECK(cudaSetDevice(own >= 0 ? devs[own] : dev0));
    cudaStream_t so = own >= 0 ? stc[own] : s1;   // the stream the result is produced on
    lap(3, so);
    lap(2, so);
    // The down pass wrote [slots, hid] per-slot partials into part[c]; summing the two cards' rows and
    // then folding the top-k groups into one row per token must happen on one card, so both stages run
    // on card 0. This read dec_out/dec_out1 - buffers this path never writes - so the experts' results
    // were dropped and two stale buffers summed into the output instead. That is why the MoE output
    // differed from the grouped path at n=1 (L0 0.025104 vs 0.028675) while both agreed at prefill
    // sizes, where the grouped path serves both.
    if (moe_diag::moec) {
      // The two cards' partials were written on their OWN streams, so each read must be ordered
      // against the stream that produced it. A plain cudaMemcpy here reads whatever is there at the
      // time - which produced identical values for every layer and still looked plausible.
      auto fr = [&](const float* q, size_t cnt, int c) {
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(c == 0 ? s1 : s2));
        std::vector<float> t(cnt);
        HELIOS_CUDA_CHECK(cudaMemcpy(t.data(), q, cnt * 4, cudaMemcpyDeviceToHost));
        double a = 0;
        for (float v : t) a += (double)v * v;
        fprintf(stderr, "[card] L%d card%d=%.9e (mgemm)\n", layer, c, sqrt(a / cnt));
      };
      fr(sc.part[0], (size_t)n * hid, 0);
      fr(sc.part[1], (size_t)n * hid, 1);
    }
    // n*hid, matching the grouped path: the down pass reduces into row 0 of each card's part, so only
    // the first n rows are meaningful. The slots*hid this used to copy was a leftover from when part
    // held raw per-slot partials, and it dragged rows 1..slots-1 of stale data across the bus.
    //
    // Pipeline mode: there is no second partial. The down pass already reduced the top-k into one
    // row per token on the owning card, so `y` is a plain copy of it - no bus, no extra add.
    if (own >= 0) {
      glue::copy_f32(y, sc.part[own], (size_t)n * hid, so);
    } else {
      xcard_copy(sc.part_sum, dev0, sc.part[1], dev1, (size_t)n * hid * 4, sc.bounce_f,
                 sc.bounce_ev, s1);
      glue::add_f32_inplace(sc.part_sum, sc.part[0], (size_t)n * hid, s1);
      // part[c] already holds one row per token: mgemm's weighted-reduction mode splits the slots
      // into num_tokens contiguous groups and reduces each group into its own output row, which is
      // the same per-token layout the grouped kernel produces. Folding the top-k again here
      // double-counted, so this is a plain copy of the [n, hid] sum.
      glue::copy_f32(y, sc.part_sum, (size_t)n * hid, s1);
    }
    lap(4, so);
    // Same diagnostic the grouped path carries: the routed contribution BEFORE the shared expert is
    // added, which is the only way to compare the two paths' expert arithmetic directly (the full
    // output mixes in the shared expert and hides the difference).
    if (moe_diag::moec) {
      std::vector<float> t((size_t)n * hid);
      const float* routed = own >= 0 ? sc.part[own] : sc.part_sum;   // where the sum actually is
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), routed, (size_t)n * hid * 4,
                                        cudaMemcpyDeviceToHost, so));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(so));
      double a = 0;
      for (float v : t) a += (double)v * v;
      fprintf(stderr, "[moec] L%d routed=%.9e (mgemm)\n", layer, sqrt(a / t.size()));
      // Elementwise fingerprint, so this path can be compared against the GEMV decode path without
      // trusting rms (see the note on the GEMV side).
      double sum = 0, asum = 0;
      for (int i = 0; i < n * hid; i++) { sum += (double)t[i] * (i + 1); asum += fabs((double)t[i]); }
      fprintf(stderr, "[moex] L%d cksum=%.12e asum=%.12e v0=%.9e v1=%.9e vlast=%.9e\n", layer, sum,
              asum, t[0], t[1], t[n * hid - 1]);
    }
    if (w.shared[0].trellis && w.shared_gate) {
      exl3::GroupWords sg{(const uint16_t*)w.shared[0].trellis, w.shared[0].suh, w.shared[0].svh,
                          w.shared[0].mul1};
      exl3::GroupWords su{(const uint16_t*)w.shared[1].trellis, w.shared[1].suh, w.shared[1].svh,
                          w.shared[1].mul1};
      exl3::GroupWords sd{(const uint16_t*)w.shared[2].trellis, w.shared[2].suh, w.shared[2].svh,
                          w.shared[2].mul1};
      // The shared expert's dense gemms need a card-local workspace: in pipeline mode that is the
      // owning card's slice (the [0] slice belongs to card 0 even in the card-1 scratch), and its
      // Hadamard scratch likewise. Flag off this is index 0 / sc.had exactly as before.
      const int ec = own >= 0 ? own : 0;
      half* sh_g = (half*)sc.tsg[ec];
      half* sh_u = (half*)sc.tsu[ec];
      exl3::linear(sh_g, x, sg, n, inter, hid, w.shared[0].K, false, so, sc.had_or(ec));
      exl3::linear(sh_u, x, su, n, inter, hid, w.shared[1].K, false, so, sc.had_or(ec));
      aux::silu_mul(sh_g, sh_u, sh_g, false, 0.0f, (size_t)n * inter, so);
      exl3::linear(sc.shared_out, sh_g, sd, n, hid, inter, w.shared[2].K, true, so, sc.had_or(ec));
      aux::add_sigmoid_gate_proj(sc.shared_out, x, y, w.shared_gate, (size_t)n, hid, so);
    }
    lap(5, so);
    if (mprof) mn++;
    return;
  }


  // Pipeline mode: the owning card runs the whole thing - one zeroed partial, no staging, and the
  // grouped kernel reads the caller's own activations instead of the far card's copy.
  if (own >= 0) {
    HELIOS_CUDA_CHECK(cudaSetDevice(own == 0 ? dev0 : dev1));
    HELIOS_CUDA_CHECK(cudaMemsetAsync(sc.part[own], 0, (size_t)n * hid * 4, stc[own]));
  } else {
    HELIOS_CUDA_CHECK(cudaSetDevice(dev0));
    HELIOS_CUDA_CHECK(cudaMemsetAsync(sc.part[0], 0, (size_t)n * hid * 4, s1));
    HELIOS_CUDA_CHECK(cudaSetDevice(dev1));
    HELIOS_CUDA_CHECK(cudaMemsetAsync(sc.part[1], 0, (size_t)n * hid * 4, s2));
    HELIOS_CUDA_CHECK(cudaSetDevice(dev0));
    // the far card needs the hidden rows (10 KB per token)
    xcard_copy(sc.x16_1, dev1, x, dev0, (size_t)n * hid * 2, sc.bounce_h, sc.bounce_ev, s2);
    // card 1's permute needs the router's expert selection and weights, which live on card 0
    xcard_copy(sc.topk_idx[1], dev1, sc.topk_idx[0], dev0, (size_t)n * topk * 8, sc.bounce_h,
               sc.bounce_ev, s2);
    xcard_copy(sc.topk_w[1], dev1, sc.topk_w[0], dev0, (size_t)n * topk * 2, sc.bounce_h,
               sc.bounce_ev, s2);
  }

  for (int ci = 0; ci < ncards; ci++) {
    const int c = own >= 0 ? own : ci;
    if (per[c] <= 0) continue;
    cudaStream_t ssc = stc[c];
    // Every launch in this iteration addresses card c's memory on card c's stream, so the CUDA
    // context must be that card's too.
    HELIOS_CUDA_CHECK(cudaSetDevice(c == 0 ? dev0 : dev1));
    const long total = (long)n * topk;
    remap_ids_kernel<<<(int)((total + 127) / 128), 128, 0, ssc>>>(
        sc.topk_idx[c], sc.remap[c], total, base[c], per[c]);
    glue::moe_permute(sc.remap[c], sc.topk_w[c], n, topk, per[c], sc.counts[c], sc.sorted[c],
                      sc.wsorted[c], sc.perm_ws[c], ssc);
    chk_stage(c == 0 ? "permute0" : "permute1", ssc);
    lap(1, ssc);
    {
      // what the grouped kernel will actually be asked to do, for the first layer only
      static bool once = false;
      if (!once && dbg_stage()) {
        once = true;
        std::vector<int64_t> cnt((size_t)per[c] + 1), srt((size_t)n * topk);
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(cnt.data(), sc.counts[c], ((size_t)per[c] + 1) * 8,
                                          cudaMemcpyDeviceToHost, ssc));
        HELIOS_CUDA_CHECK(cudaMemcpyAsync(srt.data(), sc.sorted[c], (size_t)n * topk * 8,
                                          cudaMemcpyDeviceToHost, ssc));
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(ssc));
        int64_t mx = 0, tot = 0;
        for (int e = 0; e < per[c]; e++) { mx = std::max(mx, cnt[e]); tot += cnt[e]; }
        fprintf(stderr, "[moe] card%d n=%d topk=%d per=%d total=%lld max_per_expert=%lld\n",
                c, n, topk, per[c], (long long)tot, (long long)mx);
        fprintf(stderr, "[moe] card%d first sorted ids:", c);
        for (int i = 0; i < std::min(8, n * topk); i++) fprintf(stderr, " %lld", (long long)srt[i]);
        fprintf(stderr, "\n");
      }
    }
    exl3::ExpertTables tb;
    const size_t stride = (size_t)(cfg.n_layers + 1) * per[c];
    const size_t off = (size_t)(mtp ? cfg.n_layers : layer) * per[c];
    const uint16_t** g0 = (const uint16_t**)sc.table_base[c][0];
    const uint16_t** u0 = (const uint16_t**)sc.table_base[c][3];
    const uint16_t** d0t = (const uint16_t**)sc.table_base[c][6];
    tb.gate_trellis = g0 + off; tb.gate_suh = (const half**)sc.table_base[c][1] + off;
    tb.gate_svh = (const half**)sc.table_base[c][2] + off;
    tb.up_trellis = u0 + off;   tb.up_suh = (const half**)sc.table_base[c][4] + off;
    tb.up_svh = (const half**)sc.table_base[c][5] + off;
    tb.down_trellis = d0t + off; tb.down_suh = (const half**)sc.table_base[c][7] + off;
    tb.down_svh = (const half**)sc.table_base[c][8] + off;
    (void)stride;
    exl3::moe_grouped((own >= 0 || c == 0) ? (const void*)x : (const void*)sc.x16_1, sc.part[c],
                      sc.counts[c],
                      // workspace: the kernel walks experts in LPT order, so it indexes the token
                      // spans by expert id instead of accumulating a running end along its loop
                      glue::moe_permute_offsets(sc.perm_ws[c], per[c]),
                      glue::moe_permute_lpt_order(sc.perm_ws[c], per[c]),
                      sc.sorted[c], sc.wsorted[c], sc.tsg[c], sc.tsu[c], sc.tig[c], sc.tiu[c], tb, n,
                      hid, inter, per[c], topk, sc.max_tokens_per_expert, sc.concurrency, 2, 2, 2,
                      /*mcg=*/false, /*mul1=*/true, /*act_limit=*/0.0f, MOE_ACT_SILU,
                      // num_active only sizes the grid: num_groups = MIN(concurrency, num_active) and
                      // group_size = MIN(num_sms/num_groups, MOE_MAX_SMS_PER_EXPERT), so it never
                      // changes which experts are processed - fewer groups still pull every ticket.
                      // Passing a smaller value therefore WIDENS the per-expert groups, spreading one
                      // expert's 3.6 MB across more SMs instead of 8. -1 (unknown) forces the narrow
                      // default. Overridable to measure the effect before wiring a real count.
                      num_active, ssc);
    chk_stage(c == 0 ? "grouped0" : "grouped1", ssc);
    lap(3, ssc);
    if (moe_diag::cnt) {
      std::vector<int64_t> cnt((size_t)per[c] + 1);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(cnt.data(), sc.counts[c], cnt.size() * 8,
                                        cudaMemcpyDeviceToHost, ssc));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(ssc));
      long long tot = 0;
      for (int e = 0; e < per[c]; e++) tot += cnt[e];
      fprintf(stderr, "[cnt] L%d card%d assigned=%lld (expected share of n*topk=%d)\n", layer, c, tot,
              n * topk);
    }
  }

  // 3) combine the partials into y. y is an output, not an accumulator: it is written (not added
  // to), so a caller that reuses its buffer between layers needs no clearing of its own. The
  // two-card sum must land in its own buffer - staging part[1] straight into part[0] would discard
  // the experts card 0 just computed. Pipeline mode has ONE partial, already on the owning card, so
  // `y` is a device-local copy of it: no second card, no bus, no extra rounding step.
  const int ec = own >= 0 ? own : 0;
  cudaStream_t so = stc[ec];
  HELIOS_CUDA_CHECK(cudaSetDevice(ec == 0 ? dev0 : dev1));
  lap(2, so);
  if (own < 0 && moe_diag::moec) {   // two partials only; single-card has nothing to compare
    auto fr = [&](const float* q, size_t cnt, int c) {
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(c == 0 ? s1 : s2));
      std::vector<float> t(cnt);
      HELIOS_CUDA_CHECK(cudaMemcpy(t.data(), q, cnt * 4, cudaMemcpyDeviceToHost));
      double a = 0;
      for (float v : t) a += (double)v * v;
      fprintf(stderr, "[card] L%d card%d=%.9e (grouped)\n", layer, c, sqrt(a / cnt));
    };
    fr(sc.part[0], (size_t)n * hid, 0);
    fr(sc.part[1], (size_t)n * hid, 1);
  }
  if (own >= 0) {
    glue::copy_f32(y, sc.part[own], (size_t)n * hid, so);
  } else {
    xcard_copy(sc.part_sum, dev0, sc.part[1], dev1, (size_t)n * hid * 4, sc.bounce_f,
               sc.bounce_ev, s1);
    glue::add_f32_inplace(sc.part_sum, sc.part[0], (size_t)n * hid, s1);
    glue::copy_f32(y, sc.part_sum, (size_t)n * hid, s1);
  }
  lap(4, so);

  if (moe_diag::moec) {
    auto fr = [&](const void* q, size_t cnt) {
      std::vector<float> t(cnt);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), q, cnt * 4, cudaMemcpyDeviceToHost, so));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(so));
      double a = 0;
      for (float v : t) a += (double)v * v;
      return sqrt(a / cnt);
    };
    fprintf(stderr,
            "[moec] L%d routed=%.9e shared_ptr=%d gate_ptr=%d expert0=%d\n", layer,
            fr(own >= 0 ? (const void*)sc.part[own] : (const void*)sc.part_sum, (size_t)n * hid),
            (int)(w.shared[0].trellis != nullptr),
            (int)(w.shared_gate != nullptr), (int)(w.router != nullptr));
  }

  // shared expert: gate/up -> silu -> down (fp32 out), then the sigmoid gate adds it to y
  if (w.shared[0].trellis && w.shared_gate) {
    exl3::GroupWords sg{(const uint16_t*)w.shared[0].trellis, w.shared[0].suh, w.shared[0].svh,
                        w.shared[0].mul1};
    exl3::GroupWords su{(const uint16_t*)w.shared[1].trellis, w.shared[1].suh, w.shared[1].svh,
                        w.shared[1].mul1};
    exl3::GroupWords sd{(const uint16_t*)w.shared[2].trellis, w.shared[2].suh, w.shared[2].svh,
                        w.shared[2].mul1};
    half* sh_g = (half*)sc.tsg[ec];
    half* sh_u = (half*)sc.tsu[ec];
    exl3::linear(sh_g, x, sg, n, inter, hid, w.shared[0].K, /*y_fp32=*/false, so, sc.had_or(ec));
    exl3::linear(sh_u, x, su, n, inter, hid, w.shared[1].K, false, so, sc.had_or(ec));
    aux::silu_mul(sh_g, sh_u, sh_g, /*float_input=*/false, /*act_limit=*/0.0f,
                  (size_t)n * inter, so);   // in place; 0 = no clamp, as the reference
    exl3::linear(sc.shared_out, sh_g, sd, n, hid, inter, w.shared[2].K, true, so, sc.had_or(ec));
    aux::add_sigmoid_gate_proj(sc.shared_out, x, y, w.shared_gate, (size_t)n, hid, so);
  }
  lap(5, so);
  if (mprof) mn++;
}

}  // namespace helios
