#pragma once
// Per-layer CUDA-graph capture/replay for the decode forward. Default OFF: HELIOS_DECODE_GRAPH=1.
//
// WHY THIS SCOPE. The decode step is ~38 ms of which the MoE is ~19 ms and the GDN recurrence
// ~8.4 ms (HELIOS_PROF, 64-token greedy). Both are latency-bound on LAUNCHES, not arithmetic: the
// MoE issues 3 cooperative mgemm launches + 3 small kernels per layer and the GDN 2 gemms + 6
// elementwise/reduction kernels, so a step is ~700 launches, each ~4-6 us of dispatch on a kernel
// that is itself 50-200 us. exllamav3 gets the same arithmetic in half the wall clock by capturing
// the block once and replaying it, which collapses the per-launch CPU cost.
//
// WHAT IS AND IS NOT CAPTUREABLE, and why the boundary is where it is:
//
//   * MoE (mgemm decode path, pipeline on) - CAPTURABLE. Routing is already device-side: the
//     router writes topk_idx/topk_w into device memory, remap_ids_kernel maps them into the card's
//     local expert space, and mgemm dereferences the device index/weight arrays. There is no host
//     branch on which experts were chosen, so a capture bakes in no routing decision.
//     (The per-expert GEMV path at moe_layer.cu:413 DOES read the router output back to the host
//     and is therefore excluded outright - see moe_decode_graphable() below.)
//   * GDN sublayer - CAPTURABLE. conv_state/rec_state are fixed device pointers, and nothing in
//     gdn_layer depends on the position.
//   * gr_mix / gr_apply / the fp32->fp16 cast - CAPTURABLE. Fixed buffers, fixed shapes.
//
//   * Full-attention sublayer - NOT CAPTURABLE. pos0 is a host value baked into the rope position
//     buffer, the KV append offset and the split-KV decision, and it changes every step. Capturing
//     it would freeze the position. QSA adds a second pos0-dependent branch on top. So for a
//     full-attention layer the attn site stays eager and only the MLP site is graphed.
//   * PLE - NOT CAPTURABLE. Its n-gram ids are hashed on the HOST out of hist_, which grows and
//     changes every chunk.
//
// So a GDN layer is captured WHOLE (mix -> sublayer -> apply -> mix -> MoE -> apply) and a
// full-attention layer is captured from its first gr_apply through the end of the MLP site, which
// is everything except the position-dependent attention kernels themselves.
//
// The capture is keyed by (site, device, layer, width, gdn_capture). Every pointer in these blocks
// is a fixed arena address or a per-layer slot, so one capture per key is valid for every later
// step; only the CONTENTS of those buffers change, which is exactly what a graph replay is for.
// Keying on width is required (n=1 decode and n=2 spec verify are different launches) and keying on
// gdn_capture is required because the speculative sublayer-input snapshot is a host-conditional
// D2D copy that has to sit INSIDE the graph, between the cast and the recurrence.
//
// A capture is preceded by an uncaptured warmup of the same body: gr_mix grows its temporaries with
// cudaMalloc on first use at a new width, DevCtx allocates the per-device lock/workspace buffers
// lazily, and ensure_kernel_attr calls cudaFuncSetAttribute. All three are illegal or unsafe inside
// a capture, and the warmup also produces the correct output for the first step, so the first
// step is not wasted work.
#include <cuda_runtime.h>

#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "cuda/cuda_shim.hpp"


namespace helios {

// HELIOS_DECODE_GRAPH=1. Read once: the flag must not change under an already-captured graph.
// Graph capture is OFF by default. It was measured on by a first pass and reported as +3-5%, but a
// controlled A/B (3 runs/cell, same prompt, graph flag the only variable) does not reproduce that:
// at 8k it is 22.5 tok/s with the graph against 30.0 without (-25%), and on a second 8k prompt
// 37.5 vs 38.9 (-3.5%). Only a 4k point ever looked positive, and by less than the run-to-run
// spread there. Output is byte-identical either way, so the default is the faster path; the graph
// stays available behind HELIOS_DECODE_GRAPH=1 for narrower contexts and further tuning.
// Read once: the flag must not change under an already-captured graph.
inline bool decode_graph_on() {
  // OFF by default. It IS a real decode win - measured +5.5% at 8k, +5.4% at 14k, +3.9% at 48k,
  // with draft acceptance identical to 0.1% and the sequential digest unchanged - but it is NOT SAFE
  // across requests, and a crash is worth far more than 5%.
  //
  // Graphs bake in the absolute device addresses that were live at capture time. Capture happens on
  // the first decode a process performs, and a later request in the same process can reach a buffer
  // the captured kernels do not describe, at which point the replay writes through a stale pointer:
  //   helios serve ... ; 3 short requests ; then an 8k request
  //   -> CUDA error an illegal memory access was encountered at runner.cpp:1286
  // and the process dies, taking the connection with it. The 8k request alone is fine, and so is ONE
  // short request followed by it, so this never appears in a benchmark: grid_bench spawns a fresh
  // process per cell, and a single-request measurement cannot see a second request at all.
  //
  // Isolated by elimination: HELIOS_MTP=0 and HELIOS_PREFIX_CACHE=0 both still crash; only
  // HELIOS_DECODE_GRAPH=0 avoids it. Fixing it properly means making every graph-captured address
  // invariant across requests (or re-capturing when a request changes any of them), which is the work
  // that makes this feature safe rather than fast. Until then it stays opt-in.
  static const bool on = getenv("HELIOS_DECODE_GRAPH") != nullptr && atoi(getenv("HELIOS_DECODE_GRAPH")) != 0;
  return on;
}

// True when the layer body this build would capture contains something a capture cannot hold.
// Checked once, before any capture, and reported by name so a refused graph is never silent.
//
//  - HELIOS_PROF / HELIOS_MPROF / HELIOS_DBG: the phase profiler and the named-sync helper both
//    cudaStreamSynchronize inside the block.
//  - HELIOS_MOEC / HELIOS_MOEX / HELIOS_MRMS / HELIOS_MIN / HELIOS_HC / HELIOS_SUB / HELIOS_LRMS /
//    HELIOS_PRMS: per-layer diagnostic D2H readbacks inside the block.
//  - HELIOS_GRMS / HELIOS_GDNSTATE: the same inside gdn_layer.
//  - HELIOS_QSA: the sparse-attention path, which is itself pos0-dependent (and, being inside the
//    attention sublayer, is already outside every capture - refused here so the reason is named).
inline const char* decode_graph_refusal() {
  static const char* names[] = {"HELIOS_PROF", "HELIOS_MPROF",   "HELIOS_DBG",   "HELIOS_MOEC",
                                "HELIOS_MOEX", "HELIOS_MRMS",    "HELIOS_MIN",   "HELIOS_HC",
                                "HELIOS_SUB",  "HELIOS_LRMS",    "HELIOS_PRMS",  "HELIOS_GRMS",
                                "HELIOS_GDNSTATE", "HELIOS_QSA",  "HELIOS_GMIXPROF"};
  static const char* why = []() -> const char* {
    for (const char* nm : names)
      if (getenv(nm)) return nm;
    return nullptr;
  }();
  return why;
}

// The capture cache. One instance per Runner; keys are built by the call sites.
class LayerGraphs {
 public:
  ~LayerGraphs() {
    for (auto& kv : execs_) cudaGraphExecDestroy(kv.second);
  }

  // Runs `body` on `s`, captured and replayed as one graph per `key`.
  //
  // First call for a key: run the body ONCE, uncaptured, then capture it and instantiate. The
  // capture executes NOTHING, so the first step's result is the warmup run's and the body has run
  // exactly once - which matters, because the body is not idempotent: gdn_layer ADVANCES conv_state
  // and rec_state, so running it twice in one step would apply the recurrence twice and the output
  // would diverge from the ungraphed engine in exactly the way a wrong answer looks (coherent
  // prose, wrong tokens, no error anywhere). Launching the freshly instantiated exec here instead
  // is what a graph API example does and it is what produced that divergence.
  //
  // Later calls: one cudaGraphLaunch. Any capture or instantiate failure falls back to the eager
  // body for that key PERMANENTLY and says so - a graph that cannot be built is a slower decode,
  // never a wrong one.
  void run(int key, int width, cudaStream_t s, const std::function<void(cudaStream_t)>& body) {
    if (!on_) { body(s); return; }
    bodies_++;
    note_vram();   // free VRAM before this key's graph exists
    auto it = execs_.find(key);
    if (it != execs_.end()) {
      if (it->second) {
        if (cudaGraphLaunch(it->second, s) != cudaSuccess) {
          fprintf(stderr, "[graph] launch failed for key %d (%s); falling back to eager\n", key,
                  cudaGetErrorString(cudaGetLastError()));
          it->second = nullptr;   // keep the entry, demoted to eager
          body(s);
          eager_++;
          return;
        }
        launched_++;
      } else {
        body(s);
        eager_++;
      }
      return;
    }
    // --- first contact: warm up outside any capture ---
    body(s);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    // A capture is only legal on a stream that nothing else is using concurrently. Decode is
    // single-threaded and its stages are issued in order, so the layer's own compute stream is
    // idle here. ThreadLocal (not Global) so an unrelated API call on another thread - the server's
    // reader threads, for instance - cannot abort this capture.
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      fprintf(stderr, "[graph] BeginCapture failed (%s); this key stays eager\n",
              cudaGetErrorString(cudaGetLastError()));
      execs_[key] = nullptr;
      eager_++;
      return;
    }
    body(s);
    cudaGraph_t g = nullptr;
    cudaError_t e = cudaStreamEndCapture(s, &g);
    if (e != cudaSuccess || !g) {
      fprintf(stderr, "[graph] EndCapture failed for key %d (%s); this key stays eager\n", key,
              cudaGetErrorString(e ? e : cudaGetLastError()));
      if (g) cudaGraphDestroy(g);
      execs_[key] = nullptr;
      eager_++;
      return;
    }
    cudaGraphExec_t ge = nullptr;
    e = cudaGraphInstantiate(&ge, g, 0ull);
    size_t nodes = 0;
    cudaGraphGetNodes(g, nullptr, &nodes);
    cudaGraphDestroy(g);
    if (e != cudaSuccess || !ge) {
      fprintf(stderr, "[graph] Instantiate failed for key %d (%s); this key stays eager\n", key,
              cudaGetErrorString(e ? e : cudaGetLastError()));
      execs_[key] = nullptr;
      eager_++;
      return;
    }
    execs_[key] = ge;
    captured_++;
    nodes_ += nodes;
    if (width >= 0 && width < 32) by_width_[width]++;
    note_vram();   // ...and after it does
    // Deliberately NOT launching here. The warmup above already produced this step's output, and a
    // launch would apply the body's stateful work (the GDN recurrence) a SECOND time. The graph is
    // used from the next call on.
  }

  // Printed from Runner's destructor: what was captured, how much of the work went through a replay
  // rather than a fresh launch sequence, and what the graphs cost in VRAM. A graph feature that
  // reports nothing is indistinguishable from a flag that does nothing.
  void report() const {
    if (!on_ || captured_ == 0) return;
    // The width histogram matters: a graph is keyed per width, so "48 graphs" is 48 keys, and the
    // histogram is what says whether they are 48 layers at one width or 24 layers at two.
    std::string widths;
    for (int w = 0; w < 32; w++)
      if (by_width_[w]) widths += " n=" + std::to_string(w) + ":" + std::to_string(by_width_[w]);
    fprintf(stderr,
            "[graph] captured %zu graphs, %zu nodes | replays %ld | eager fallbacks %ld | by width:%s\n",
            captured_, nodes_, launched_, eager_, widths.c_str());
    // MB figures are cast to double explicitly: a size_t passed to a varargs %f is read as a double
    // BIT PATTERN, which prints as a 300-digit number. That was the first version of this line.
    for (int d = 0; d < 2; d++)
      if (seen_[d])
        fprintf(stderr, "[graph]   card%d free VRAM %.0f MB before its first capture -> %.0f MB "
                        "after its last (graphs cost %.2f MB)\n",
                d, (double)first_[d] / 1048576.0, (double)last_[d] / 1048576.0,
                ((double)first_[d] - (double)last_[d]) / 1048576.0);
  }

 private:
  // Free VRAM, PER DEVICE. cudaMemGetInfo reports whichever device is current and the two cards
  // alternate between layers, so a single pair of samples reads card 0's free memory at one layer
  // and card 1's at the next - which reads as a ~900 MB "graph cost" when the graphs cost almost
  // nothing. That was the first version of this counter and it produced exactly that wrong number.
  void note_vram() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 2) return;
    size_t fr = 0, tot = 0;
    if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) return;
    if (!seen_[dev]) { first_[dev] = fr; seen_[dev] = true; }
    last_[dev] = fr;
  }

  bool on_ = decode_graph_on();
  std::map<int, cudaGraphExec_t> execs_;
  size_t captured_ = 0, nodes_ = 0;
  long launched_ = 0, eager_ = 0, bodies_ = 0;
  // Captures per decode width, so the report can show that a K+1 verify and a plain n=1 decode
  // really did get separate graphs. They must: the launch shapes differ with the width, so one
  // graph cannot serve both, and silently sharing one is a wrong answer, not a slow one.
  int by_width_[32] = {};
  size_t first_[2] = {0, 0}, last_[2] = {0, 0};
  bool seen_[2] = {false, false};
};


// Key layout, low bit first: width (5 bits, 0-4), layer (16 bits, 5-20), gdn_capture (bit 21),
// device (bits 22-23), site (bits 24-31). The widths are disjoint, so two different
// (layer, width, capture, card, site) tuples can never collide into one key - a collision here
// would mean replaying one layer's graph for another, which is a wrong answer rather than a slow
// one, so the layout is spelled out rather than left implicit in the shifts.
constexpr int dgraph_key(int site, int dev, int layer, int n, int gdn_cap) {
  return (site << 24) | (dev << 22) | ((gdn_cap & 1) << 21) | (layer << 5) | (n & 31);
}

}  // namespace helios
