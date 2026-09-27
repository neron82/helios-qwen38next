// Qwen3.8 full-attention layer. See attn_layer.cuh for the source-verified contract of each call.
#include "engine/attn_layer.cuh"

#include "core/device.hpp"
#include "cuda/quant/exl3_gemm.cuh"
#include "cuda/aux/activation.cuh"
#include "cuda/aux/norm.cuh"
#include "cuda/attn/qwen_rope.cuh"
#include "cuda/attn/qwen_gqa.cuh"
#include "cuda/attn/kv_quant.cuh"

namespace helios {


namespace {
// Per-stage timing inside the attention layer. The layer's total was ~13x the sum of its parts as
// measured by bench_gemm, so the parts have to be timed where they actually run rather than inferred.
struct StageTimer {
  cudaEvent_t ev[9];
  int n = 0;
  bool on = false;
  double tot[8] = {0};
  long calls = 0;
  StageTimer() {
    on = getenv("HELIOS_ATTNDBG") != nullptr;
    if (on) for (int i = 0; i < 9; i++) cudaEventCreate(&ev[i]);
  }
  ~StageTimer() {
    if (on && calls > 0) {
      // 8 marks give 7 deltas: the "before ogemm" mark closes the attention kernel and the final
      // mark closes the o gemm (which absorbs the output gating). Labelling these off by one made
      // kvcopy look like the 17.8 ms stage when it was really the attention kernel, and only the
      // skip-the-copy test caught it.
      static const char* nm[7] = {"qkv_gemm", "deint", "norm", "rope", "kvcopy", "attn", "ogemm"};
      fprintf(stderr, "[attndbg] per-layer ms:");
      for (int i = 0; i < 7; i++) fprintf(stderr, " %s=%.3f", nm[i], tot[i] / calls);
      fprintf(stderr, "  (calls=%ld)\n", calls);
    }
  }
  void mark(cudaStream_t s) { if (on && n < 9) cudaEventRecord(ev[n++], s); }
  void flush(cudaStream_t s) {
    if (!on) return;
    cudaStreamSynchronize(s);
    for (int i = 1; i < n; i++) {
      float ms = 0;
      if (cudaEventElapsedTime(&ms, ev[i - 1], ev[i]) == cudaSuccess) tot[i - 1] += ms;
    }
    calls++;
    n = 0;
  }
};
StageTimer& stage_timer() { static StageTimer t; return t; }
}  // namespace

// Above this many (row, head) pairs the plain kernel already has enough blocks to fill the
// GPU, so split-KV would only add a combine pass. 24 heads * 32 rows = 768 blocks.
static constexpr int kGqaSplitKvMaxRows = 32;
// Below this many keys per row the plain kernel is already fast enough that the combine launch is
// pure overhead: measured at a 6-token prompt the attention phase is 1.19 ms either way, so splitting
// buys nothing and costs a second launch. It pays from about 2k keys (15k context: 11.10 -> 9.08 ms).
static constexpr int kGqaSplitKvMinKeys = 2048;

bool attn_scratch_init(AttnScratch& sc, int max_n, int device, int n_qsa_layers, int max_ctx,
                       int n_sel_override, int qsa_scratch_rows_override, int hidden_for_qsa,
                       int kv_token_dim) {
  sc.max_n = max_n;
  const int q_out = 12288, kv_out = 512;
  auto A = [&](size_t bytes) { return Engine::instance().gpu(device).alloc(bytes, 256); };
  sc.q_flat = A((size_t)max_n * q_out * 2);
  sc.k_flat = A((size_t)max_n * kv_out * 2);
  sc.v_flat = A((size_t)max_n * kv_out * 2);
  sc.q = A((size_t)max_n * q_out / 2 * 2);      // 24 heads x 256
  sc.k = A((size_t)max_n * kv_out * 2);
  sc.g = A((size_t)max_n * q_out / 2 * 2);
  sc.o = A((size_t)max_n * q_out / 2 * 2);
  sc.pos = (int*)A((size_t)max_n * 4);
  sc.a_had = A((size_t)max_n * 16384 * 2);
  // ALLOCATED LAST ON PURPOSE. A() is a bump allocator, so this block shifts every allocation after
  // it. Allocated in the natural position it cost ~2.3 tok/s at short context (41.5 vs 43.8 mean
  // over 3 runs each) purely from the resulting address changes; last, it shifts nothing.
  //
  // Sized from max_ctx, not from the current position: the split-KV chunk count grows with the key
  // range, so the buffer has to bound the widest grid the engine can ever launch, not the one it
  // happens to launch now. That is also why it is not 8x the old size - the chunk count falls as n
  // grows, and n * chunks(n) is nearly flat, so 4.8 MB replaces the old 12.7 MB.
  sc.kv_part = (float*)A(attn::gqa_split_kv_bytes(kGqaSplitKvMaxRows, 24, 2, 256, max_ctx));

  // QSA sparse-attention state, one per full-attention layer. `n_qsa_layers` is passed in by the
  // runner (the layer count is a model property, not an attention-shard property).
  if (n_qsa_layers > 0) {
    const int hd = 128, cr = 4, nh = 4;
    const int n_sel = n_sel_override > 0 ? n_sel_override : 512;
    sc.qsa.resize((size_t)n_qsa_layers);
    size_t total = 0;
    for (int l = 0; l < n_qsa_layers; l++) total += attn::qsa_layer_bytes(max_ctx, hd, cr);
    sc.qsa_slab = A(total);
    char* base = (char*)sc.qsa_slab;
    // The three pieces must tile INSIDE one qsa_layer_bytes stride, not be added on top of it.
    // qsa_layer_bytes already counts pooled_k + block_pos + tail_raw, so advancing base past all
    // three again made the loop walk 9,232 bytes further per layer than `total` reserved - 110,784
    // bytes past the end of the slab over 12 layers, which is a silent heap overrun of the bump
    // allocator. It did not fault where it happened: the corruption surfaced much later as a
    // SIGSEGV inside cuMemcpyDtoDAsync, which is why this looked like a driver fault.
    const size_t nb = (size_t)(max_ctx / cr) + 2;
    const size_t pooled_b = nb * hd * 2;
    const size_t blockpos_b = nb * 4;
    const size_t tail_b = (size_t)cr * hd * 2;
    for (int l = 0; l < n_qsa_layers; l++) {
      attn::QsaLayerState& st = sc.qsa[l];
      st.pooled_k = base;
      st.block_pos = (int*)(base + pooled_b);
      st.tail_raw = base + pooled_b + blockpos_b;
      base += pooled_b + blockpos_b + tail_b;
      st.n_blocks = 0; st.n_tail = 0; st.max_blocks = (int)nb;
    }
    sc.qsa_max_ctx = max_ctx;
    // Gather/attention scratch for one decode row: tok_idx, tok_block, tok_off, and the chunk partials.
    const int cap = n_sel * cr + cr;
    // `part` holds one [2 + head_dim] partial per (head, chunk), and the sparse kernel chunks the
    // WHOLE selection - not a capped subset. The old code clamped nchunks to 64 here, but
    // cap = 512*4 + 4 = 2052 needs (2052+15)/16 = 129 chunks, so the buffer was sized for half of
    // what the kernel writes. That is the second QSA overrun: 129 sanitizer errors once `use_qsa`
    // goes live above the 2051 threshold. Sizing it for the real count costs 1.6 MB more.
    const int nchunks = (cap + 15) / 16;
    sc.qsa_tok_idx = A((size_t)cap * 4);
    sc.qsa_tok_block = A((size_t)cap * 4);
    sc.qsa_tok_off = A((size_t)cap * 4);
    // part is consumed by gqa_sparse_decode, which runs on the MAIN attention shapes - head_dim 256,
    // not the indexer's 128. Sizing it with hd=128 under-allocated it by 1.5 MB (24*nchunks*130 vs
    // 24*nchunks*258 floats) and the next layer's cuda_check reported the resulting corruption.
    sc.qsa_part = A((size_t)24 * nchunks * (2 + 256) * 4);
    // combine stage A reduces to at most 64 groups per (row, head) - that cap is the kernel's, not
    // ours, so this one stays.
    sc.qsa_grp = A((size_t)24 * 64 * (2 + 256) * 4);
    // Scratch: enough for the widest row count the sparse path will see (decode: 1; MTP: a few).
    sc.qsa_scratch_rows = qsa_scratch_rows_override > 0 ? qsa_scratch_rows_override : 8;
    size_t sbytes = 0;
    for (int r = 1; r <= sc.qsa_scratch_rows; r++)
      sbytes += attn::qsa_scratch_bytes(r, max_ctx / cr, nh, hd, n_sel, hidden_for_qsa);
    sc.qsa_scratch = A(sbytes);
    // A() is a bump allocator: it can return null, and the decode path dereferences these. Checked
    // here so a short allocation surfaces as an init failure rather than a SIGSEGV mid-prefill.
    if (!sc.qsa_slab || !sc.qsa_scratch || !sc.qsa_tok_idx || !sc.qsa_tok_block ||
        !sc.qsa_tok_off || !sc.qsa_part || !sc.qsa_grp) {
      fprintf(stderr, "[attn] QSA scratch allocation failed\n");
      return false;
    }
  }

  // Dequant staging for a quantized KV cache: one fp16 K and one fp16 V over the whole context, in
  // the same layout the fp16 cache has, so the attention kernels can be pointed straight at it. The
  // attention walk is causal over [0, pos0+n), so the staging covers the entire range of every call -
  // there is no smaller bound to dequantize.
  //
  // ALLOCATED LAST, after the QSA block, for the same reason kv_part is: A() is a bump allocator and
  // shifting a later address has cost real throughput before. With fp16 (the default) this block
  // does not run at all, so the shipped allocation sequence is unchanged byte for byte.
  if (attn::kvq_bits() && kv_token_dim > 0) {
    const size_t rows = (size_t)max_ctx;
    sc.cq_k = A(rows * (size_t)kv_token_dim * 2);
    sc.cq_v = A(rows * (size_t)kv_token_dim * 2);
    if (!sc.cq_k || !sc.cq_v) {
      fprintf(stderr, "[attn] quantized-KV dequant staging allocation failed (%.0f MB x2)\n",
              (double)(rows * (size_t)kv_token_dim * 2) / 1048576.0);
      return false;
    }
  }

  return sc.q_flat && sc.q && sc.o && sc.pos && sc.a_had && sc.kv_part;
}

void attn_scratch_free(AttnScratch& sc) { sc = AttnScratch{}; }

void attn_layer(const AttnWeights& w, const Config& cfg, AttnScratch& sc, const half* x, float* y,
                void* k_cache, void* v_cache, int n, int pos0, cudaStream_t s, int layer_index) {
  const int hd = cfg.attn_head_dim, nq = cfg.attn_heads, nkv = cfg.attn_kv_heads;
  const int q_out = cfg.attn_q_out, kv_out = nkv * hd;
  const int rot = (int)((float)hd * cfg.partial_rotary);

  // 1) q (with its interleaved gate), k, v projections - fp16 outputs, as the gate split requires
  StageTimer& ST = stage_timer();
  ST.mark(s);
  exl3::GroupWords qw{(const uint16_t*)w.q.trellis, w.q.suh, w.q.svh, w.q.mul1};
  exl3::linear(sc.q_flat, x, qw, n, q_out, cfg.hidden, w.q.K, /*y_fp32=*/false, s, sc.a_had);
  exl3::GroupWords kw{(const uint16_t*)w.k.trellis, w.k.suh, w.k.svh, w.k.mul1};
  exl3::linear(sc.k_flat, x, kw, n, kv_out, cfg.hidden, w.k.K, false, s, sc.a_had);
  exl3::GroupWords vw{(const uint16_t*)w.v.trellis, w.v.suh, w.v.svh, w.v.mul1};
  exl3::linear(sc.v_flat, x, vw, n, kv_out, cfg.hidden, w.v.K, false, s, sc.a_had);

  ST.mark(s);
  // 2) split the per-head-interleaved q projection into q and the gate
  aux::deinterleave_qg((const half*)sc.q_flat, (half*)sc.q, (half*)sc.g, hd,
                       (size_t)n * nq * hd, s);

  ST.mark(s);
  // 3) per-head q/k RMSNorm (constant_bias 1.0 in this model)
  aux::rms_norm(sc.q, aux::kHalf, w.q_norm, aux::kHalf, sc.q, aux::kHalf, n * nq, hd, cfg.rms_eps,
                1.0f, 1.0f, false, 1, s);
  aux::rms_norm(sc.k_flat, aux::kHalf, w.k_norm, aux::kHalf, sc.k, aux::kHalf, n * nkv, hd,
                cfg.rms_eps, 1.0f, 1.0f, false, 1, s);

  // 4) absolute positions for this chunk, then partial NEOX rope
  {
    std::vector<int> h_pos(n);
    for (int i = 0; i < n; i++) h_pos[i] = pos0 + i;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(sc.pos, h_pos.data(), (size_t)n * 4, cudaMemcpyHostToDevice, s));
  }
  ST.mark(s);
  attn::rope_qk_partial_neox(sc.q, sc.k, n, nq, nkv, hd, rot, sc.pos, cfg.rope_theta, s);

  // 5) append to the cache and attend densely (keys [0, pos0+n), row r causal at pos0+r)
  //
  // With HELIOS_KV_QUANT the cache is the reference's quantized layout (kv_quant.cuh) rather than
  // fp16, so the append is a quantize-on-write kernel instead of two device-to-device copies, and the
  // READ is a dequant into an fp16 staging buffer (sc.cq_k / sc.cq_v) that the attention kernels are
  // then pointed at. The staging buffer has the fp16 cache's own layout, so every attention kernel
  // below - the mma prefill, the scalar prefill, the split-KV decode - runs unmodified on values
  // that are fp16 in both cases. That is the whole read path: nothing inside the attention kernels
  // knows a quantizer exists, which is why this did not have to touch the tuned kernels' lane
  // mapping.
  //
  // Rows are indexed by absolute position, so a chunk writes at pos0 and the same bytes the kernels
  // read are the ones just written - which is what makes speculative rollback (rewrite the tail) work
  // unchanged. The staged range is [0, pos0+n): the whole prefix, because attention is causal over
  // all of it, so there is no smaller bound to dequantize.
  ST.mark(s);
  const int kv_bits = attn::kvq_bits();
  const int token_dim = nkv * hd;
  const int ctx_rows = attn::kvq_ctx_rows();
  const void* k_read = k_cache;
  const void* v_read = v_cache;
  if (kv_bits) {
    attn::KvQuant kq{k_cache, (char*)k_cache + attn::kvq_scale_offset(ctx_rows, token_dim, kv_bits)};
    attn::KvQuant vq{v_cache, (char*)v_cache + attn::kvq_scale_offset(ctx_rows, token_dim, kv_bits)};
    attn::kvq_write(sc.k, sc.v_flat, kq, vq, n, token_dim, kv_bits, pos0, s);
    if (!sc.cq_k || !sc.cq_v) {
      fprintf(stderr, "[attn] quantized KV set but the dequant staging buffer is null\n");
      return;
    }
    attn::kvq_dequant(kq.q, kq.s, sc.cq_k, pos0 + n, token_dim, kv_bits, 0, s);
    attn::kvq_dequant(vq.q, vq.s, sc.cq_v, pos0 + n, token_dim, kv_bits, 0, s);
    k_read = sc.cq_k;
    v_read = sc.cq_v;
  } else {
    const size_t row_bytes = (size_t)nkv * hd * 2;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync((char*)k_cache + (size_t)pos0 * row_bytes, sc.k,
                                      (size_t)n * row_bytes, cudaMemcpyDeviceToDevice, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync((char*)v_cache + (size_t)pos0 * row_bytes, sc.v_flat,
                                      (size_t)n * row_bytes, cudaMemcpyDeviceToDevice, s));
  }
  ST.mark(s);
  // QSA sparse attention. Above the reference's threshold (4 * block_topk + 3 = 2051 here) the indexer
  // picks 4-token blocks and attention reads only those; below it the reference is exactly dense, and so
  // are we, which is what keeps short-prompt parity intact.
  const int idx_hd = 128, idx_nh = 4, idx_cr = 4;
  const int n_sel = cfg.idx_budget / idx_cr;
  const int qsa_threshold = idx_cr * n_sel + 3;
  // Decode only for now: the scratch below is laid out for one row and qsa_expand handles a
  // single query. A prefill chunk (n up to 256) entering this branch overruns it - that is the
  // SIGSEGV it produced. Prefill keeps dense attention until its selection path exists.
  // DEFAULT OFF. The components are unit-tested (test_qsa_parity) but the engine wiring is not yet
  // correct end to end: with it on, decode at 15k context emits nothing while HELIOS_NO_QSA gives the
  // reference continuation at 33.4 tok/s. That is a wiring bug, not a kernel bug - the parity test
  // drives gqa_sparse_decode directly and never goes through the indexer, the pooled state or the
  // engine's buffer plumbing. Opt in with HELIOS_QSA=1 to bisect it.
  // The POOLED state must be built for every chunk, prefill included - otherwise the first decode
  // token finds n_blocks == 0 and there is nothing to select from. n_blocks=0 at decode is exactly
  // that symptom. Pooling is per-token and row-count independent, so it is safe for any n.
  // HELIOS_QSA used to be excluded when the KV cache is quantized, because the sparse path gathers
  // the cache as fp16 indexed by token id and a quantized cache is not that. With the dequant
  // staging buffer above it IS served correctly: the staged range [0, pos0+n) is every row the
  // gather can name, in the fp16 cache's own layout. That gate is lifted so the wiring bug this
  // branch actually has (it emitted nothing at 15k) can be reproduced and fixed at the kv_bits the
  // engine ships, rather than only in a configuration it does not.
  const bool qsa_pool_on = getenv("HELIOS_QSA") && w.idx_qk.trellis != nullptr
                          && sc.qsa_slab && layer_index >= 0 && layer_index < (int)sc.qsa.size();
  // Selection, expand and sparse attend are decode-only: the scratch is laid out for one row.
  const bool use_qsa = n == 1 && qsa_pool_on && (pos0 + n) > qsa_threshold;

  static const int qsa_stop = getenv("HELIOS_QSA_STOP") ? atoi(getenv("HELIOS_QSA_STOP")) : 99;
  if (qsa_pool_on) {
    attn::QsaLayerState& qs = sc.qsa[layer_index];
    // Indexer query + raw key for these tokens, then fold the raw keys into completed 4-token blocks.
    // Carve the ONE-ROW scratch and process the chunk a row at a time. Sizing the scratch for 256 rows
    // would be ~700 MB; a serial loop over rows costs nothing because these are 1-token projections
    // and the row loop is over the chunk, not over history.
    attn::QsaScratch qs_s{};                 // one row's worth; reused by select below
    const size_t off_had = 0;
    // qk holds the WHOLE fused projection, (n_heads + 1) * head_dim per row - the +1 is the raw key.
    const size_t off_qk  = off_had + (size_t)cfg.hidden * 2;
    const size_t qk_bytes = (size_t)(idx_nh + 1) * idx_hd * 2;
    const size_t off_q   = off_qk + qk_bytes;
    const size_t off_rk  = off_q + (size_t)idx_nh * idx_hd * 2;
    const size_t off_sel = off_rk + (size_t)idx_hd * 2;
    const size_t off_sc  = off_sel + (size_t)n_sel * 4;
    char* qb = (char*)sc.qsa_scratch;
    qs_s.had = qb + off_had; qs_s.qk = qb + off_qk; qs_s.q = qb + off_q; qs_s.raw_k = qb + off_rk;
    qs_s.sel = qb + off_sel; qs_s.score = qb + off_sc;
    for (int r0 = 0; r0 < n; r0++) {
    qs_s.pos = sc.pos + r0;                       // this row's absolute position
    qs_s.n = 1;
    int one_pos = pos0 + r0;
    cuda_check(cudaMemcpyAsync(qs_s.pos, &one_pos, sizeof(int), cudaMemcpyHostToDevice, s));
    attn::qsa_project(x + (size_t)r0 * cfg.hidden, 1, w.idx_qk.trellis, w.idx_qk.suh, w.idx_qk.svh,
                      w.idx_qk.mul1, w.idx_qk.K, cfg.hidden, w.idx_q_ln, w.idx_k_ln, idx_nh, idx_hd,
                      idx_hd / 4, cfg.rope_theta, cfg.rms_eps, qs_s, s);
    attn::qsa_pool_update(qs, (const half*)qs_s.raw_k, 1, w.idx_k_ln, idx_hd, idx_hd / 4, idx_cr,
                          cfg.rope_theta, cfg.rms_eps, s);
    }   // row loop
    if (use_qsa) {
      // The loop left qs_s holding the LAST row. For decode (n == 1) that is the row we select for;
      // if the last prefill chunk ran, re-project so the query is the current token's.
      qs_s.n = 1;
      cuda_check(cudaMemcpyAsync(qs_s.pos, sc.pos, sizeof(int), cudaMemcpyDeviceToDevice, s));
      attn::qsa_project(x + (size_t)(n - 1) * cfg.hidden, 1, w.idx_qk.trellis, w.idx_qk.suh,
                        w.idx_qk.svh, w.idx_qk.mul1, w.idx_qk.K, cfg.hidden, w.idx_q_ln, w.idx_k_ln,
                        idx_nh, idx_hd, idx_hd / 4, cfg.rope_theta, cfg.rms_eps, qs_s, s);
    }
    if (use_qsa && qsa_stop >= 2) {
      attn::qsa_select(qs, qs_s, idx_nh, idx_hd, n_sel, 1.0f / sqrtf((float)idx_hd), s);
    }

    // Expand the selected blocks to token indices and attend over just those. Decode-only for now
    // (n == 1); the prefill path stays dense until its selection is implemented.
    if (use_qsa && qsa_stop >= 3) {
      const int cap = n_sel * idx_cr + idx_cr;
      int* tok_idx = (int*)sc.qsa_tok_idx;
      int* tok_off = (int*)sc.qsa_tok_off;
      int* tok_block = (int*)sc.qsa_tok_block;
      float* part = (float*)sc.qsa_part;
      attn::qsa_expand(qs, qs_s, 1, n_sel, idx_cr, pos0, cap, tok_idx, (int*)tok_block, tok_off, s);
      if (getenv("QSA_STEP_DBG")) {
        cudaError_t e = cudaStreamSynchronize(s);
        int cnt = -1;
        cudaMemcpy(&cnt, tok_off, 4, cudaMemcpyDeviceToHost);
        fprintf(stderr, "[qsa] after expand: %s (cap=%d count=%d first=%d)\n", cudaGetErrorString(e),
                cap, cnt, cnt > 0 ? tok_idx[0] : -1);
      }
      float* grp = (float*)sc.qsa_grp;
      if (getenv("QSA_STEP_DBG")) {
        cudaGetLastError();
        fprintf(stderr, "[qsa] layer=%d before sparse: cap=%d nchunks=%d nq=%d nkv=%d hd=%d part=%p grp=%p "
                        "tok_idx=%p tok_off=%p sel=%p n_blocks=%d\n", cap, (cap + 15) / 16, nq, nkv, hd,
                (void*)part, (void*)grp, (void*)tok_idx, (void*)tok_off, qs_s.sel, qs.n_blocks);
        fprintf(stderr, "[qsa] layer=%d\n", layer_index);
      }
      attn::gqa_sparse_decode(sc.q, k_cache, v_cache, tok_idx, tok_off, cap, sc.o, part, grp, 1, nq,
                              nkv, hd, 1.0f / sqrtf((float)hd), s);
      if (getenv("QSA_STEP_DBG")) {
        cudaError_t e = cudaStreamSynchronize(s);
        fprintf(stderr, "[qsa] after sparse: %s\n", cudaGetErrorString(e));
      }
      ST.mark(s);
      goto qsa_done;
    }
  }
  qsa_done:;

  // Decode has only nq blocks in the plain kernel (one per row-head), which leaves most of an 82-SM GPU
  // idle while the whole KV cache streams past. Split the key range across blocks instead and combine
  // afterwards; the arithmetic per chunk, and the order of the combine, are unchanged, so this is
  // bitwise identical and only the scheduling differs.
  // k_read/v_read are k_cache/v_cache verbatim on the fp16 path and the dequantized staging buffer on
  // the quantized one: same layout, same kernels, same arithmetic, different source of the fp16.
  if (n <= kGqaSplitKvMaxRows && pos0 + n >= kGqaSplitKvMinKeys)
    attn::gqa_dense_split_kv_decode(sc.q, k_read, v_read, sc.o, sc.kv_part, n, nq, nkv, hd, pos0,
                                    1.0f / sqrtf((float)hd), s);
  else
    attn::gqa_dense_f16(sc.q, k_read, v_read, sc.o, n, nq, nkv, hd, pos0, 1.0f / sqrtf((float)hd), s);

  ST.mark(s);
  // 6) gate the attention output and project
  aux::mul_sigmoid_((half*)sc.o, (const half*)sc.g, (size_t)n * nq * hd, s);
  exl3::GroupWords ow{(const uint16_t*)w.o.trellis, w.o.suh, w.o.svh, w.o.mul1};
  exl3::linear(y, sc.o, ow, n, cfg.hidden, nq * hd, w.o.K, /*y_fp32=*/true, s, sc.a_had);
  ST.mark(s);
  ST.flush(s);
}

}  // namespace helios
