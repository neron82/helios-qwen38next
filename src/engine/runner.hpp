#pragma once
#include <atomic>
// Qwen3.8-Flash-Next runner: single-sequence forward over 48 layers.
//
// Layer plan (qwen4_exp): embed -> ExpandStreams (broadcast the hidden into 4 fp32 streams) -> PLE
// ahead of layer 1 -> per layer [attn site: mix, GDN or GQA sublayer, apply] then [mlp site: mix,
// MoE, apply] -> combine-less mixer (which IS the model norm; there is no final norm) -> lm_head.
//
// State that is not position-addressed, and therefore has to be carried between chunks:
//   KV cache       12 full-attention layers, fp16 [ctx, 2, 256] for K and V
//   GDN           36 layers: conv state [10240, 4] and the recurrent matrix [48,128,128] fp32
//   PLE           one conv state [10240, 9] **and a host id history** (the n-gram hashing runs on
//                 the CPU, so the ids never round-trip through the device)
#include "core/model.hpp"
#include "core/device.hpp"
#include "engine/gdn_layer.cuh"
#include "engine/attn_layer.cuh"
#include "engine/moe_layer.cuh"
#include "engine/ple.cuh"
#include "engine/mtp.cuh"
#include "engine/sampler.hpp"
#include "engine/dgraph.hpp"

#include <functional>
#include <vector>

namespace helios {

class Runner {
public:
  // Widest speculative verify this build supports: up to kMaxSpecDrafts drafted tokens, committed
  // together with the verified token, so a step emits at most kMaxSpecDrafts+1 tokens and the
  // logits buffer must hold kMaxSpecDrafts+1 rows. head_rows_ is sized to HELIOS_SPEC_K+1 at init.
  static constexpr int kMaxSpecDrafts = 7;
  bool init(Model& m, int ctx_cap, int max_chunk = 2048);
  void reset();
  // Prompt ingestion: processes ids[from..end) starting at the current position, filling every
  // state. `from` is the cross-request prefix-cache resume point (0 = the whole prompt).
  void prefill(const std::vector<int>& ids, int from = 0);
  // Continuation: 1..k tokens appended at the current position.
  void decode(const std::vector<int>& ids);
  // Decode ONE token for each of two slots through a SINGLE forward. This is the whole point of
  // batching: decode re-reads every weight per token, so two sequences in one forward cost far less
  // than two forwards - the MTP width-2 step prices a second token at 1.45x, not 2x. Returns false
  // (and changes nothing) when the pair cannot be batched: speculation on, fewer than two slots, or
  // the two sequences at the same position, where a plain decode already does the work.
  bool decode_pair(int tok_a, int tok_b, int slot_a, int slot_b);
  // Row `row` of the pair's logits (row 0 = slot_a, row 1 = slot_b), so a caller can sample each
  // sequence from its own row. Valid only straight after a decode_pair that returned true.
  const float* pair_logits(int row) const;
  // ---- stepping API, for the batch-parity harness ----
  //
  // generate() owns prefill, sampling, stop strings and speculation, so it cannot be interleaved with
  // a second sequence. These expose exactly the deterministic core it runs, which is all a parity
  // check needs: reset, prefill, commit a KNOWN token, read the resulting argmax.
  void harness_reset() { reset(); }
  void harness_prefill(const std::vector<int>& ids) { prefill(ids, 0); }
  // Commit `tok` at the current position. The caller supplies the token so both the serial and the
  // paired run are driven by the SAME input, which is what makes the comparison exact.
  void harness_commit(int tok) { decode({tok}); hist_tokens_.push_back(tok); last_committed_ = tok; }
  // Argmax over row `row` of the last forward's logits.
  int harness_argmax(int row) const;
  // rms of the LAST layer's sublayer input, per row - the trunk's output BEFORE the final mixer and
  // lm_head. sub_in is fp16, so this copies D HALVES (D*2 bytes) and converts; reading D*4 bytes
  // and treating it as float is what made the first version of this print 5.2e10, which is not an
  // activation value but two rows of half bit patterns reinterpreted.
  void harness_dump_last(const char* tag) const;
  void harness_state_ptr_gap_pub(int a, int b, int l) const { harness_state_ptr_gap(a, b, l); }
  // Run ONE GDN layer over a 2-row input twice - once with bsz=2, once as two n=1 calls - and report
  // the worst per-element difference. This is the only unit neither kernel-level A/B covers: both of
  // those drive the kernels directly, skipping the projections, the norm and the output projection
  // that gdn_layer wraps around them.
  double harness_gdn_layer_ab(int n);
  // The qkv projection - the stage the layer bisect pointed at - run over a 2-row input once at
  // n=2 and once as two n=1 calls, comparing the outputs. This is exl3::linear with the Hadamard
  // transform, which is the one call in the GDN layer that no A/B has covered.
  double harness_linear_ab(int n);
  // gated_rms_norm, checked WITHOUT knowing its formula: feed a fixed [rows_max, dim] input, run it
  // once at rows_max and once at a smaller row count over the same leading data, and ask whether row
  // r's output changed. A per-row kernel must give identical answers for the same row regardless of
  // how many rows the launch covers; if it does not, the launch's row count leaks into the result.
  void harness_norm_rows_check(int rows_max, int dim);
  // The two projections in gdn_layer that no A/B covers: the z projection (exl3::linear into the
  // SHARED sc.a_had, the same Hadamard scratch the qkv and out projections use) and the a/b gemv.
  // Both are run at n=2 and as two n=1 calls through the same scratch, which is the wrapper's own
  // pattern rather than one kernel in isolation.
  void harness_proj_ab(int n);
  // gated_delta_net_fused_op_2 - the step that produces sc.g and sc.beta, called with B hardcoded to
  // 1 and n in the S slot, so a two-row batch arrives as one sequence of length 2. The only op in
  // gdn_layer that no A/B has covered, and the only place a batched forward collapses the
  // batch/sequence distinction.
  void harness_fused2_ab(int n);
  // max |state| difference between two slots' GDN recurrent state for the given layer. Prefilling
  // IDENTICAL prompts into both slots must leave identical state; if it does not, the batched decode
  // divergence is a slot-isolation fault in prefill rather than anything about pairing.
  double harness_state_gap(int slot_a, int slot_b, int layer) const;
  // Greedy speculative step. `next` is the verified token to consume at the current position; the
  // MTP draft head proposes the following token and the trunk verifies it. Commits the longest
  // agreeing prefix (1 or 2 tokens), writes them to `emit`, and returns the count.
  //
  // Contract: on return, logits row 0 always predicts the token after the last committed position,
  // so `next_token()` is the input for the following call whether the draft was accepted or not.
  int spec_step(int next, int emit[kMaxSpecDrafts + 1]);
  int spec_verify(int next, const int* drafts, int K, int emit[kMaxSpecDrafts + 1]);
  // Prints the decode-graph summary (captures, replays, VRAM cost) before the CUDA context the
  // graphs live in goes away, and releases the prefix ring - ~0.9 GB of PINNED host memory, which
  // the OS does not reclaim on its own. A no-op unless HELIOS_DECODE_GRAPH captured anything or
  // HELIOS_PREFIX_CACHE=1 allocated the ring.
  ~Runner() {
    dgraphs_.report();
    for (int d = 0; d < N_GPU; d++) {
      if (pfx_done_[d]) cudaEventDestroy(pfx_done_[d]);
      if (pfx_copy_[d]) cudaEventDestroy(pfx_copy_[d]);
    }
    if (pfx_host_) cudaFreeHost(pfx_host_);
  }
  // Greedy argmax of logits row 0 (the token after the last committed position).
  int next_token();
  bool mtp_enabled() const { return mtp_on_; }
  void set_mtp(bool on) { mtp_on_ = on; }

  // Full generation loop over an already-tokenized prompt. `on_token` may return false to stop
  // (that is how a disconnected client cancels). Stops at EOS, at max_tokens, or when the context
  // is full. Uses the MTP draft head when enabled and the request is greedy.
  // Two sequences, one forward per step. This is decode batching made reachable: generate() owns
  // prefill, sampling, stop strings and speculation, so it cannot be interleaved with a second
  // sequence - which is why this is a separate entry point rather than a flag on generate(). Returns
  // false (having changed nothing) when the pair cannot be batched, so a caller can fall back to two
  // ordinary generate() calls. Speculation is excluded: its verify batch already runs the trunk at
  // width 2, and its rollback snapshots are single-slot.
  bool is_stop(int id) const;
  bool generate_pair(const std::vector<int>& pa, const std::vector<int>& pb, const GenParams& p,
                     std::vector<int>* out_a, std::vector<int>* out_b,
                     const std::function<bool(int)>& on_a = nullptr,
                     const std::function<bool(int)>& on_b = nullptr);
  std::vector<int> generate(const std::vector<int>& prompt, const GenParams& p,
                            const std::function<bool(int)>& on_token = nullptr);

  // ---- cross-request prefix cache (HELIOS_PREFIX_CACHE=1, default OFF) ----
  //
  // The KV cache is position-addressed, so rows [0, n) stay valid for any later request whose first
  // n tokens are the same ones. The recurrent state is not: the GDN matrices, the PLE's dilated
  // conv and the PLE's host n-gram window are functions of the whole prefix, so a resume is only
  // sound from a position whose state has been captured. The ring below captures it at every prefill
  // chunk boundary, and a resume restores the newest capture at or before the divergence point, so
  // the tokens between the two are simply recomputed.
  //
  // Captures land on chunk boundaries ON PURPOSE. A prefill of any length cuts its chunks at
  // multiples of max_chunk_ from 0, so a resume at a multiple of max_chunk_ reproduces exactly the
  // chunk sequence - and therefore exactly the reduction order - a full prefill of the same prompt
  // would have used. Resuming anywhere else would be a different summation order, not the same
  // answer in more time.
  const std::vector<int>& history() const { return hist_tokens_; }
  // Longest common prefix of `prompt` against the resident history.
  int prefix_match(const std::vector<int>& prompt) const;
  // Position the last request resumed at, and the tokens resumed requests have saved so far.
  int prefix_resume() const { return resume_; }
  long long prefix_reuse_total() const { return reuse_total_; }
  long long prefix_requests() const { return prefix_reqs_; }
  int prefix_snapshot_count() const { return pfx_slots_; }
  int prefix_snapshot_interval() const { return pfx_interval_; }
  // Decide the resume point for `prompt`, restore the state it needs, and leave the runner at that
  // position. Returns the number of leading prompt tokens that will NOT be recomputed. Called by
  // generate(); with the cache off it is exactly the reset() the loop used to do.
  int prefix_begin(const std::vector<int>& prompt);

  // ---- HELIOS_SEQUENCES (default 1 = the shipped single-sequence engine) ----
  //
  // "Concurrent" here means INTERLEAVED, not simultaneous, and the distinction is the whole
  // design. The engine has ONE activation set, one logits buffer and one device-side recurrent
  // state per layer, so two forwards cannot be in flight at once; making them simultaneous is a
  // redesign of every kernel's scratch plumbing, not a flag. What N slots DO buy is N
  // conversations kept ALIVE: each slot owns a disjoint stride of the already-allocated KV cache
  // and its own copy of the small persistent recurrent state, so a server can hold several open
  // threads and run them one request at a time without evicting each other's state.
  //
  // A switch costs no device traffic at all. The per-slot state is RESIDENT, so binding a slot is
  // a pointer rebind, not a save/restore copy - which is why the state is budgeted at 116 MB a
  // sequence rather than spilled to host. What a slot does NOT survive is a request that resets
  // it: HELIOS_PREFIX_CACHE is what keeps a slot's position and token history across requests, so
  // N > 1 only persists conversations when the prefix cache is on too. That is stated at startup.
  int sequences() const { return n_slots_; }
  // Rows one slot may use. The KV cache is one flat buffer of ctx_cap_ rows, partitioned.
  int slot_context_cap() const { return slot_ctx_; }
  // Choose and bind the slot for one request: `want` < 0 round-robins, otherwise the client PINS
  // the slot and a follow-up turn must name the same one. Returns -1 if `want` is out of range -
  // folding it would answer a turn from the wrong conversation's state.
  int acquire_slot(int want);
  // The slot currently bound. Read by the server to echo back which conversation ran.
  int active_slot() const { return active_slot_; }

  // The generation loop stops at EOS; the id comes from the tokenizer, which the runner does not own.
  void set_eos(int id) { stops_.assign(1, id); }
  // Stop on any of these; checkpoints that end turns with a token other than <|endoftext|> need the
  // whole set or generation runs past the turn boundary.
  void set_stops(const std::vector<int>& ids) { stops_ = ids; }

  int pos() const { return pos_; }
  // What ONE conversation may use. With N slots this is ctx_cap/N, and it is what a client has to
  // be told: advertising the whole cache would invite a prompt that overruns the slot's partition.
  int context_cap() const { return slot_ctx_ ? slot_ctx_ : ctx_cap_; }
  // The whole cache, i.e. what all slots share. Reported alongside so the arithmetic is visible.
  int total_context_cap() const { return ctx_cap_; }
  const float* logits_dev() const { return logits_; }   // fp32 [vocab] of the last position

  struct Timings {
    double prefill_ms = 0, decode_ms = 0;
    int prefill_tokens = 0, decode_tokens = 0;
    // Speculative decoding counters. The accept rate is the number that decides whether MTP pays:
    // each step pays a full draft forward, so it only wins if the drafted token is usually right.
    long spec_steps = 0, spec_accepts = 0, spec_slots = 0;   // slots = drafts PROPOSED, not steps
    // Fraction of proposed draft slots that were accepted. Dividing accepts by STEPS instead makes
    // the number exceed 100% as soon as K > 1, because a step can accept up to K drafts.
    double accept_rate() const {
      return spec_slots > 0 ? (double)spec_accepts / (double)spec_slots : 0.0;
    }
  };
  const Timings& timings() const { return tm_; }

  // Average wall time of one run_chunk at width n. This is the quantity batched speculative
  // verification is designed around, and it is NOT reachable through --chunk: that also sets
  // max_n, and the mgemm path needs max_n >= n*topk, so shrinking the chunk to probe a width
  // under-allocates instead of measuring. Here max_n stays at the runner's real value and only n
  // moves. Position advances so attention and the caches see realistic state.
  double bench_chunk(int n, int iters, int warmup);
  // Self-test for the batched-verification rollback: snapshot -> advance -> replay must return the
  // GDN recurrence to a bit-identical state. If this fails, a partial accept silently diverges.
  bool gdn_replay_selftest(int width, unsigned* out_a, unsigned* out_b);
  // HELIOS_SPEC_TRACE=1: fingerprint every recurrent state at each committed position, so a
  // speculative run can be diffed against a plain-decode run and the first diverging position
  // attributed to a specific state. Diagnostic only; the fingerprints go to stderr.
  void trace_state(const char* tag, int pos);

  // Whether the model's output is still inside its think block. The parser that knows this lives in
  // the server and mirrors the state in through the token callback; the sampler reads it to suspend
  // the repetition penalty while the model is reasoning, because a penalty over a reasoning model's
  // narration window leaves it circling instead of closing its think block.
  bool in_think() const { return in_think_.load(std::memory_order_relaxed); }
  void set_in_think(bool v) { in_think_.store(v, std::memory_order_relaxed); }
  // Slot plumbing for the batch-parity harness (the engine binds slots internally).
  void bind_slot_pub(int s) { bind_slot(s); }
  void save_slot_pub(int s) { save_slot(s); }
  double harness_state_gap_pub(int a, int b, int l) const { return harness_state_gap(a, b, l); }
  // Byte gap between two slots' recurrent state for one layer, and the size the kernel assumes
  // (history_stride * state_size). The GDN kernel reaches slot b's state as
  // base + slots[b] * history_stride * state_size, so this gap IS the addressing the kernel assumes.
  void harness_state_ptr_gap(int a, int b, int layer) const;
private:
  // One batched decode: `bsz` sequences, one row each, through a single forward. The point is that
  // decode is bound by re-reading every weight per token, so N sequences in one forward cost far
  // less than N forwards - the MTP width-2 step prices a second token at 1.45x, not 2x.
  //
  // Only the parts that carry per-sequence recurrent state need this. Attention is NOT batched: its
  // quantized-KV staging buffer is 268 MB x 2 per card and a second staged region does not fit in
  // card1's 550 MB of free VRAM, so each row attends separately against its own cache base and
  // position, sharing one staging region. Attention is ~9% of a decode step; GDN, MoE and the mHC
  // mixers are ~88% and all already work per-token over n.
  struct BatchCtx {
    int bsz = 1;
    int pos[2] = {0, 0};               // absolute position of each row
    int slot[2] = {0, 0};              // slot each row belongs to
    // PER DEVICE, not one copy: half the GDN layers run on card 1, and a card-0 pointer read by a
    // card-1 kernel is not a slot number, it is garbage - the sanitizer shows the recurrent kernel
    // reading 25 MB past a 3.1 MB state block, which is exactly a garbage slot scaled by the
    // per-slot stride. Indexed by the layer's device.
    const int* gdn_slots[2] = {nullptr, nullptr};
    const int* conv_slots[2] = {nullptr, nullptr};
    // Per-CARD GDN layer count. The layers are split 18/18 across the two cards, so the
    // stride from one slot's state to the next is that card's own layer count, not the
    // model's 36 - the state block is carved [slot][layers on this card].
    int slot_layers[2] = {0, 0};
  };
  // `head_rows` = how many logits rows this forward projects, one per position, the last of them
  // landing in row 0. 1 for every ordinary chunk (prefill, decode): `next_token()` reads row 0, so
  // projecting more would sample early. A speculative verify passes K+1, one row per position of
  // its batch.
  void run_chunk(const std::vector<int>& ids, int pos0, int head_rows = 1,
                  const BatchCtx* b = nullptr);

  // HELIOS_PIPELINE=1 only. Micro-batches the chunk and runs the two layer ranges on their own
  // cards, staggered so card 0's micro-batch i+1 overlaps card 1's micro-batch i. n <= micro
  // threshold (every decode and speculative verify) degenerates to a correct, non-overlapped
  // two-card sequential pass: there is no second micro-batch to overlap with.
  // `b` is null on every shipped call and means bsz == 1, which is the original single-sequence
  // path with no branch taken anywhere below.
  void run_chunk_pipeline(const std::vector<int>& ids, int pos0, int head_rows = 1,
                          const BatchCtx* b = nullptr);
  // One forward's layer loop, restricted to [lo, hi), on the stage that owns those layers. `mark`
  // is the phase profiler's recorder (the shipped path passes the same one it always did).
  void layer_range(int dev, const int* ids, int n, int pos0, int lo, int hi,
                   const std::function<void(int, cudaStream_t)>& mark, float* tap_dst, int tap_row0,
                   const BatchCtx* b = nullptr);
  // Everything after the last layer: save the tap the draft head reads, fill the draft's own KV for
  // the positions this chunk covered, then the combine-less final mixer and lm_head. All of it on
  // the card that owns the LAST layer (card 0 with the pipeline off, which is the shipped layout).
  void chunk_tail(int dev, const int* ids, int n, int pos0, cudaStream_t s,
                  const std::function<void(int, cudaStream_t)>& mark, int head_rows = 1);
  // The hyper-connection stream stack crossing the layer boundary, card 0 -> card 1. These boards are
  // PHB with no peer access, so it stages through a pinned host slot. The D2H runs on card 0's
  // DMA-out stream behind the stage-0 completion event, so the host wait drains the HANDOFF only -
  // never card 0's compute stream - and the two stages keep overlapping while the CPU waits.
  void handoff(int slot, size_t bytes, cudaStream_t s_dst);
  // One cross-card copy through pinned host memory (see the definition for why the D2H rides
  // card 0's DMA-out stream). `slot` selects the pinned staging buffer, so two in-flight
  // micro-batches can never share one.
  void xcopy(void* dst, cudaStream_t s_dst, const void* src, cudaStream_t s_src, size_t bytes,
             int slot);
  // The draft head's embedding rows, gathered on the card that owns the embedding table and handed
  // to the card that runs the head. Returns the destination (mtp_.emb16_) for mtp_draft_step.
  const half* mtp_stage_embed(const int* ids, int n, cudaStream_t s_dst);
  int mtp_stage_slot() const { return (int)xbounce_.size() - 1; }   // reserved, whole-chunk sized
  // Logits live on the last layer's card, and so does everything that reads or writes them.
  float* logits_ptr() const { return last_dev_ == 0 ? logits_ : logits1_; }
  cudaStream_t last_stream() const { return Engine::instance().gpu(last_dev_).stream(0); }
  // Drain every stream a forward may have used. With the pipeline off that is card 0 only, which is
  // what every caller synced before.
  void sync_all() {
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(Engine::instance().gpu(0).stream(0)));
    if (last_dev_ != 0)
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(Engine::instance().gpu(1).stream(0)));
  }
  void final_head(int n, int head_rows);

  Model* m_ = nullptr;
  int ctx_cap_ = 0, max_chunk_ = 0, pos_ = 0;
  std::vector<int> stops_;
  std::vector<int> hist_tokens_;   // the token sequence the caches hold
  long long prefix_reqs_ = 0;

  // GPU0 workspaces
  float* streams_ = nullptr;    // [max_chunk, 4, hidden] fp32 hyper-connection streams
  half* embed16_ = nullptr;     // [max_chunk, hidden]
  float* mixed_ = nullptr;      // [max_chunk, hidden] fp32 collapse output
  half* sub_in_ = nullptr;      // [max_chunk, hidden] fp16 sublayer input
  float* sub_out_ = nullptr;    // [max_chunk, hidden] fp32 sublayer output (the mix apply)
  float* post_ = nullptr;       // [max_chunk, 4] fp32 hyper-connection post gate
  int* ids_dev_ = nullptr;
  float* logits_ = nullptr;     // [head_rows_, vocab] fp32
  // CAPACITY of logits_, in rows - NOT the width any one forward projects (that is the `head_rows`
  // argument, see run_chunk). Two rows is the floor whatever K is: row 1 is the MTP draft head's
  // own output slot, which spec_step writes directly.
  int head_rows_ = 2;
  // How many rows the last final_head() actually wrote.
  int head_rows_written_ = 1;
  int next_tok_ = -1;          // token following the last position a spec step committed
  // Tap at the last COMMITTED position, carried from one verify forward to the next. This is what
  // removes the dedicated width-1 "seed" forward: exllamav3 drafts from the tap the previous verify
  // already produced, so there is ONE trunk forward per speculative step, not two.
  float* carry_tap_ = nullptr;
  bool carry_valid_ = false;
  // Row count of the most recent run_chunk, so capture_carry can index tap_ correctly. tap_ is
  // indexed by row WITHIN the chunk, not by absolute position - a multi-chunk prefill would read
  // the wrong row entirely.
  int last_chunk_w_ = 0;
  // Token occupying the last committed position, i.e. the one carry_tap_ belongs to. The draft head
  // combines a trunk tap with the embedding of the token AT THE SAME position, so the two must be
  // paired; feeding it `next` against the previous position's tap silently poisons every draft.
  int last_committed_ = -1;
  void capture_carry();
  // Batched-verification state rollback. The GDN recurrence is the only trunk state a partial accept
  // invalidates; these snapshot it and can replay just the accepted prefix without repeating the
  // trunk forward.
  std::vector<void*> gdn_conv_snap_, gdn_rec_snap_;
  half* gdn_sub_snap_ = nullptr;   // per-GDN-layer captured sub_in_, for replay
  size_t gdn_sub_stride_ = 0;
  bool gdn_capture_ = false;       // set while a speculative batch is in flight
  size_t gdn_rec_bytes_ = 0, gdn_conv_bytes_ = 0;
  void gdn_snapshot(int n);
  void gdn_restore();
  // Self-test: snapshot -> advance -> replay must return the GDN recurrence to the same state.
  void gdn_replay(int n);
  // The PLE's dilated conv is a running 9-column state, exactly like the GDN recurrence, and a
  // partial accept invalidates it the same way. It was NOT rewound: the next forward's conv (and
  // its injection into all four streams) was conditioned on rejected tokens. ple_replay restores
  // the pre-batch state and re-runs the conv over the accepted prefix only; the conv OUTPUT is
  // discarded because the original forward already injected the right values for those rows.
  void ple_replay(int n);
  // True only while a speculative verify batch is in flight, so the PLE snapshots its conv state
  // and input (see ple_layer's `capture`).
  bool spec_batch_ = false;
  // Host n-gram history saved across a verify batch. It is truncated to ngram_size-1 after every
  // chunk, so a batch that ends on a REJECTED draft leaves the next chunk's n-grams keyed on a
  // token the model never committed.
  std::vector<int64_t> hist_snap_;
  void* head_had_ = nullptr;

  GdnScratch gdn_;
  AttnScratch attn_;
  PleScratch ple_;
  MoeScratch moe_;
  MtpScratch mtp_;
  // ---- per-card scratch, state and activations (HELIOS_PIPELINE) ----
  //
  // The shipped forward is one sequential loop on card 0, so above there is a SINGLE set: one GDN
  // workspace, one attention workspace, one activation set, serving all 48 layers in turn. With the
  // pipeline on, a layer's kernels must dereference only their OWN card's pointers, so card 1 gets
  // its own set and the scheduler (run_chunk_pipeline) picks the set by layer.
  //
  // act0_ is an ALIAS of the single set above, not a second allocation: with the flag off every
  // layer still runs on card 0 and the layer loop still touches the same addresses it always did.
  struct Act {
    float* streams = nullptr;    // [max_chunk, H, D] fp32 hyper-connection streams
    half* embed16 = nullptr;     // [max_chunk, D] fp16 token embeddings
    float* mixed = nullptr;      // [max_chunk, D] fp32 collapse output
    half* sub_in = nullptr;      // [max_chunk, D] fp16 sublayer input
    float* sub_out = nullptr;    // [max_chunk, D] fp32 sublayer output (the mix apply)
    float* post = nullptr;       // [max_chunk, H] fp32 hyper-connection post gate
    int* ids = nullptr;          // [max_chunk] this chunk's token ids
    float* tap = nullptr;        // [max_chunk, H, D] pre-collapse stack (MTP draft source)
    float* carry_tap = nullptr;  // [H, D] tap at the last committed position
    void* head_had = nullptr;    // lm_head gemm Hadamard scratch
  };
  Act act0_;                      // card 0's activations (alias of the set above)
  Act act1_;                      // card 1's activations (pipeline only)
  GdnScratch gdn1_;               // card 1's GDN workspace
  AttnScratch attn1_;             // card 1's attention workspace + its layers' QSA state
  PleScratch ple1_;               // card 1's PLE workspace (the PLE runs on its layer's card)
  MoeScratch moe1_;               // card 1's MoE workspace, sized for its own 512 experts
  // Which card owns each full-attention layer's KV rows and each GDN layer's recurrence, indexed by
  // attn_ord / gdn_ord. All zeros with the pipeline off, i.e. the shipped placement: card 0 holds
  // all of it, card 1 holds no KV and no recurrence at all.
  std::vector<int> kv_dev_, gdn_dev_;
  // Each card's QSA pooled-key state is indexed by an ordinal WITHIN that card, not by the model's
  // global full-attention ordinal: attn1_ holds one QsaLayerState per layer card 1 owns, so the
  // global index would address past its end. All zeros with the pipeline off.
  std::vector<int> attn_card_ord_;
  // Logits live on the LAST layer's card (that is where lm_head is), so pipeline mode needs its own
  // buffer there. Null with the pipeline off, where card 0's logits_ is the only one.
  float* logits1_ = nullptr;
  int last_dev_ = 0;               // card owning the final mixer, lm_head and logits
  // Micro-batches per chunk in pipeline mode (HELIOS_PIPE_MICRO, default 4) - and therefore the
  // number of pinned handoff slots, one per in-flight micro-batch, so a copy is never staged into
  // the slot the previous micro-batch's H2D is still reading.
  int micro_ = 1;
  std::vector<char*> xbounce_;
  std::vector<cudaEvent_t> xdone_;
  // Card 1's half of the speculative sublayer-input snapshot (gdn_sub_snap_ is card 0's): a GDN
  // layer must capture onto ITS OWN card, and the capture index is the global gdn_ord.
  half* gdn_sub_snap1_ = nullptr;
  // ---- CUDA-graph decode path (HELIOS_DECODE_GRAPH=1, default OFF) ----
  //
  // One graph per (site, card, layer, width, gdn_capture); see dgraph.hpp. dgraph_ok_ is the
  // resolved decision made once in init(): the flag is on AND nothing that would sync or read back
  // inside a captured block is set. Resolving it once means the hot loop tests a bool, and a
  // refused graph is reported by name at startup rather than failing a capture 40 layers in.
  LayerGraphs dgraphs_;
  bool dgraph_ok_ = false;
  bool mtp_on_ = false;
  // Context length (in tokens) above which speculative MTP decoding is disabled. This was ORIGINALLY
  // 3072, on a measurement that later proved wrong: a controlled re-sweep (best-of-2, 256-token
  // greedy) shows speculation pays at EVERY context measured, up to the longest tested:
  //   ~3k:  78.2 (spec) vs 53.5 (no spec)   +46%
  //   ~5.5k: 72.4 vs 41.3                     +75%
  //   ~8.8k: 63.1 vs 45.0                     +40%
  //   ~14k:  51.3 vs 38.7                     +33%
  //   ~28.6k: 41.2 vs 28.9                    +42%
  // So the gate was costing 33-75% of decode throughput. The default is now INT_MAX: always
  // speculate. HELIOS_MTP_CTX_LIMIT can still impose a ceiling (0 = never speculate) if a future
  // measurement at very long context finds a crossover, but as shipped speculation is always on.
  int mtp_ctx_limit_ = 0x7fffffff;
  int mtp_ord_ = -1;               // the draft layer's KV ordinal (own cache, after the trunk's)
  // Guards the prefill-time draft cache fill: mtp_draft_step does not call run_chunk, but keeping the
  // flag makes the reentrancy explicit rather than relying on that.
  bool mtp_filling_ = false;

  float* tap_ = nullptr;           // [max_chunk, H, D] trunk pre-collapse stack
  void* tap_kv_[2] = {nullptr, nullptr};
  // Draft-head KV rows saved across the HELIOS_CHAINDBG diagnostic. Chaining writes rows the next
  // spec step will read, so without this the measurement degrades the thing it measures (observed:
  // depth-1 acceptance 87.5% clean vs 66.7% with the diagnostic live). 4 rows is enough for a
  // depth-3 chain starting at pos_.
  void* mtp_kv_snap_[2] = {nullptr, nullptr};

  // Position-addressed state
  std::vector<void*> kv_[2];        // [12] fp16 [ctx_cap, nkv, hd] K and V, GPU0
  // Recurrent state
  std::vector<void*> gdn_conv_;     // [36] bf16 [gdn_qkv_out, 4]
  std::vector<void*> gdn_rec_;      // [36] fp32 [n_v_heads, v_dim, k_dim]
  void* ple_conv_ = nullptr;        // fp16 [hc_dim, 9]
  std::vector<int64_t> hist_;       // host id history (ngram_size-1 carried ids + the live tokens)

  // ---- multi-slot sequences ----
  //
  // Only the per-slot RECURRENT state is duplicated; the KV cache is partitioned rather than copied
  // and the working set is shared, because one forward runs at a time. Resident rather than spilled,
  // so a slot switch is a pointer rebind, not a ~111 MB save and restore.
  struct SlotState {
    std::vector<void*> gdn_conv, gdn_rec;   // [n_gdn] this slot's per-layer recurrent state
    void* ple_conv = nullptr;                // the PLE's dilated conv state
    float* carry_tap = nullptr;              // card 0 half of the speculative tap carry
    float* carry_tap1 = nullptr;             // card 1 half
    int pos = 0;
    int last_committed = -1;
    int next_tok = -1;
    int pfx_next = 0;                        // next snapshot RING index for this slot. This must
                                            // match pfx_next_: load_slot copies it straight into
                                            // pfx_next_ before the prefill that captures, so a -1
                                            // here silently disables every capture.
    bool carry_valid = false;
    std::vector<int> hist_tokens;            // the token sequence this slot's caches hold
    std::vector<int64_t> hist;               // the n-gram window
    std::vector<int> pfx_pos;                // snapshot positions, per snapshot slot
  };
  std::vector<SlotState> slots_;
  int n_slots_ = 1;
  int active_slot_ = 0;                      // the slot the currently bound state belongs to
  int slot_ctx_ = 0;                         // per-slot context capacity
  int seq_requested_ = 1;                    // what the operator asked for, before the VRAM clamp
  int slot_rr_ = 0;                          // round-robin cursor for acquire_slot
  size_t kv_slot_bytes_ = 0;                 // this slot's slice of each KV cache
  // Device-side slot numbers for a paired decode, so the kernels can index their per-sequence state
  // without a host round-trip. TWO ints per device, because the layers that consume them are split
  // across both cards. The recurrent kernel scales them by the per-slot layer count; the conv
  // kernel's indices arrive pre-scaled.
  int* gdn_slots_dev_[2] = {nullptr, nullptr};
  int* conv_slots_dev_[2] = {nullptr, nullptr};
  int gdn_layers_per_card_[2] = {0, 0};
  // "How many slots fit" is a measurement, not an estimate: resolved after the KV cache and both
  // scratch sets exist, so the clamp sees what is LEFT rather than what it hoped for. Resolving it
  // earlier would authorise a slot count that then OOMs on the 6 GB cache.
  bool seq_config(int ctx_cap, size_t kv_row_b);
  void bind_slot(int s);
  // Splitting save from load makes each a plain assignment, which cannot be un-done by calling it
  // twice. A swap-based version left the live state EMPTY on a third request and segfaulted inside
  // layer_range; the two directions are separate on purpose.
  void save_slot(int cur);
  void load_slot(int s);
  // This slot's slice of a KV cache, and an ARBITRARY slot's - a batched decode's rows belong to
  // two different sequences, so the per-row base cannot come from active_slot_.
  void* kv_base(int dev, int ord) const {
    return (char*)kv_[dev][ord] + (size_t)active_slot_ * kv_slot_bytes_;
  }
  void* kv_base_slot(int dev, int ord, int slot) const {
    return (char*)kv_[dev][ord] + (size_t)slot * kv_slot_bytes_;
  }

  // ---- cross-request prefix cache ----
  //
  // A snapshot of the recurrent state every `pfx_interval_` positions, so a later request sharing a
  // prefix adopts the nearest earlier snapshot instead of recomputing the prompt.
  bool pfx_on_ = false;
  int pfx_slots_ = 0;                        // snapshots retained
  int pfx_interval_ = 0;                     // positions between snapshots
  std::vector<size_t> pfx_off_;              // per-GDN-layer byte offset inside a snapshot
  size_t pfx_ple_off_ = 0;                   // the PLE conv slice, after every GDN layer
  size_t pfx_stride_ = 0;                    // bytes per slot, per snapshot
  bool pfx_dev_[2] = {false, false};         // which cards hold state worth snapshotting
  char* pfx_host_ = nullptr;                 // pinned staging for the device -> host copy
  char* pfx_dev_snap_ = nullptr;             // device-side snapshot ring
  int pfx_next_ = 0;                         // next snapshot position for the bound slot; a RING
                                            // index, so it must be in [0, pfx_slots_) before any
                                            // capture reads it. It used to default to -1, which made
                                            // the first capture compute a host pointer ~1.85 GB
                                            // before the ring and fault the copy.
  cudaEvent_t pfx_copy_[2] = {nullptr, nullptr};
  cudaEvent_t pfx_done_[2] = {nullptr, nullptr};
  std::vector<int> pfx_pos_;                 // snapshot position per snapshot slot, per sequence
  int resume_ = 0;                           // tokens the last request reused
  long long reuse_total_ = 0;                // cumulative tokens reused
  void prefix_config(int max_chunk);
  void pfx_capture(int pos);
  void pfx_restore(int slot, int pos, const std::vector<int>& seq);
  // Whether the model's output is still inside its think block. The parser that knows this lives in
  // the server and mirrors the state here through the token callback; the sampler reads it to
  // suspend the repetition penalty while the model is reasoning.
  std::atomic<bool> in_think_{false};

  Timings tm_;
};

}  // namespace helios
