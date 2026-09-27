// Qwen3.8 GatedDeltaNet layer, single sequence. See gdn_layer.cuh for the source-verified contract
// of every call below.
#include "engine/gdn_layer.cuh"
#include "core/device.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cuda/quant/exl3_gemm.cuh"
#include "cuda/aux/gdn.cuh"
#include "cuda/aux/norm.cuh"
#include "engine/glue2.cuh"

namespace helios {

using bfloat16 = __nv_bfloat16;

// The chunked (WY) delta-rule path is ON by default: it is the same recurrence as the serial
// kernel, computed as three parallel stages instead of 1024 serial steps, and at the model's
// geometry it holds the serial recurrence to 3.8e-3 relative L2 on core_out and 2.4e-3 on the
// recurrent state, with max|delta| at 1 bf16 ulp of the reference peak (test_gdn_chunked_parity).
// Decode (n = 1) and short calls never reach it: cuda_chunked_gated_delta_rule declines below two
// chunks and the serial call below is the fallback. HELIOS_GDN_CHUNKED=0 forces the serial path
// everywhere, which is the escape hatch if a future port change regresses the parity numbers.
static bool gdn_chunked_enabled()
{
    static const bool on = !getenv("HELIOS_GDN_CHUNKED") || atoi(getenv("HELIOS_GDN_CHUNKED"));
    return on;
}

bool gdn_scratch_init(GdnScratch& sc, int max_n, int device) {
  sc.max_n = max_n;
  const int Nk = 16, Ng = 3, Hk = 128, Hv = 128;
  const int qkv_out = 2 * Nk * Hk + Nk * Ng * Hv;   // 10240
  const int z_out = Nk * Ng * Hv;                   // 6144
  const int Nv = Nk * Ng;                           // 48
  const size_t n = (size_t)max_n;
  auto A = [&](size_t bytes) { return Engine::instance().gpu(device).alloc(bytes, 256); };
  sc.qkv_flat = (float*)A(n * qkv_out * 4);
  sc.z_flat = (float*)A(n * z_out * 4);
  sc.a_out = (float*)A(n * Nv * 4);
  sc.b_out = (float*)A(n * Nv * 4);
  sc.qkvz = (float*)A(n * (qkv_out + z_out) * 4);
  sc.ba = (float*)A(n * 2 * Nv * 4);
  sc.g = (float*)A(n * Nv * 4);
  sc.mixed_qkv = A(n * qkv_out * 2);
  sc.z16 = A(n * z_out * 2);
  sc.beta = A(n * Nv * 2);
  sc.conv_out = A(n * qkv_out * 2);
  sc.core_out = A(n * z_out * 2);
  sc.normed = (half*)A(n * z_out * 2);
  sc.a_had = A((size_t)max_n * 16384 * 2);
  // Chunked delta-rule workspace, allocated ONLY when that path is enabled (it is ~68 MB at the
  // engine's default 1024-token prefill chunk and every scratch arena shares one GPU pool, so an
  // unconditional reservation would move every later allocation for no benefit while the path is
  // off). 1024 tokens is the engine's prefill chunk; the cap keeps a larger --chunk from
  // ballooning the arena, at the cost of routing the overflow tokens to the serial kernel.
  sc.gdn_chunk_tokens = 0;
  sc.gdn_chunk_ws_bytes = 0;
  if (gdn_chunked_enabled())
  {
      sc.gdn_chunk_tokens = max_n < 1024 ? max_n : 1024;
      sc.gdn_chunk_ws_bytes = aux::gdn_chunked_workspace_bytes(Nv, Nk, sc.gdn_chunk_tokens);
      sc.gdn_chunk_ws = A(sc.gdn_chunk_ws_bytes);
  }
  return sc.qkv_flat && sc.normed && sc.a_had && (sc.gdn_chunk_ws || !sc.gdn_chunk_ws_bytes);
}

void gdn_scratch_free(GdnScratch& sc) { sc = GdnScratch{}; }

// bsz / gdn_slots / conv_slots turn this into a BATCHED layer: one forward carrying `bsz` sequences,
// one row each. The two kernels already index their state by slot, so nothing here needs a new
// kernel - what batching needs is the right state BASE and the right slot numbers.
//
//   rec_state  the GDN kernel computes `slots[bi] * history_stride * state_size`, so passing
//              history_stride = n_gdn and slots = [0, 1, ...] reaches slot b's layer-i state inside
//              a slot-major [n_slots][n_gdn] block. `state_size` is already exactly rec_bytes.
//   conv_state the conv kernel computes `slots[bi] * dim * state_size`, which has no layer term,
//              so the slot indices are handed to it PRE-SCALED by n_gdn (conv_slots[b] = b*n_gdn).
//              Same block, different addressing - that is why there are two arrays.
//
// bsz == 1 with null slot arrays is the original single-sequence path exactly: the kernels then use
// bi as the slot and the caller passes that slot's own state pointer, as before.
void gdn_layer(const GdnWeights& w, const Config& cfg, GdnScratch& sc, const half* x, float* y,
               void* conv_state, float* rec_state, int n, cudaStream_t s, int bsz,
               const int* gdn_slots, const int* conv_slots, int slot_layers) {
  const int Nk = cfg.gdn_k_heads, Ng = cfg.gdn_v_heads / cfg.gdn_k_heads;
  const int Hk = cfg.gdn_k_dim, Hv = cfg.gdn_v_dim;
  const int Nv = cfg.gdn_v_heads;
  const int qkv_out = cfg.gdn_qkv_out, z_out = cfg.gdn_z_out;
  // The kernels take bsz and seqlen SEPARATELY and address batch item bi at `bi * seqlen`, so a
  // batch of B one-token rows is bsz=B, seqlen=1 - not bsz=1, seqlen=B. Passing the total row count
  // as seqlen made batch item 1 read at row 2 of a 2-row buffer, which is how paired decode diverged
  // from serial on the very first step.
  const int sl = bsz > 1 ? n / bsz : n;

  // 1) projections (EXL3; the Hadamard transform is internal, output fp32 for the fused prologue)
  exl3::GroupWords qkv{(const uint16_t*)w.qkv.trellis, w.qkv.suh, w.qkv.svh, w.qkv.mul1};
  exl3::linear(sc.qkv_flat, x, qkv, n, qkv_out, cfg.hidden, w.qkv.K, /*y_fp32=*/true, s, sc.a_had);
  exl3::GroupWords zg{(const uint16_t*)w.z.trellis, w.z.suh, w.z.svh, w.z.mul1};
  exl3::linear(sc.z_flat, x, zg, n, z_out, cfg.hidden, w.z.K, true, s, sc.a_had);

  if (getenv("HELIOS_GRMS")) {
    std::vector<float> t((size_t)n * qkv_out), z((size_t)n * z_out);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), sc.qkv_flat, t.size() * 4, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(z.data(), sc.z_flat, z.size() * 4, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    auto rms = [](const std::vector<float>& v) { double a = 0; for (float x : v) a += (double)x * x;
                                                 return sqrt(a / v.size()); };
    fprintf(stderr, "[grms] in_proj_qkv=%.6f in_proj_z=%.6f\n", rms(t), rms(z));
  }

  // 2) a and b: fp16 weights in the checkpoint's [48,2560] orientation, fp32 output (nullable bias
  //    is null here - this checkpoint has no bias on these projections)
  aux::gdn_ba_gemv(x, w.a_proj, nullptr, sc.a_out, n, cfg.hidden, Nv, s);
  aux::gdn_ba_gemv(x, w.b_proj, nullptr, sc.b_out, n, cfg.hidden, Nv, s);

  // 3) beta and log decay from the SPLIT projections.
  //    This checkpoint uses split in_proj_qkv / in_proj_z / in_proj_b / in_proj_a (not the fused
  //    qkvz/ba pair), and the reference is explicit that the fused packed layout must not be
  //    applied to split tensors: "applying it to split qkv tensors causes incorrect head ordering
  //    and broken generations". So qkv is used by a plain transpose (below) and the prologue is
  //    gated_delta_net_fused_op_2, which reads b and a in the [B,S,H] layout the split projections
  //    produce - exactly as gated_delta_net.py does on this branch.
  aux::gated_delta_net_fused_op_2(sc.b_out, sc.a_out, (const bfloat16*)w.dt_bias, w.A_log,
                                  /*a_log_fp32=*/false, (bfloat16*)sc.beta, sc.g, /*B=*/1, n, Nv,
                                  /*beta_scale=*/1.0f, s);

  // 4) mixed_qkv = transpose(qkv projection) to channel-major bf16 [fdim_qkv, S]; z to [S, Nv, Hv]
  //    bf16. The conv reads x as (bsz, dim, seqlen) and writes out as (bsz, seqlen, dim), which is
  //    the layout the delta-rule kernel then reads - so this transpose is the reference's
  //    `qkv.transpose(1, 2).to(torch.bfloat16)` and the conv's own output needs no second one.
  glue::transpose_f32_bf16(sc.qkv_flat, sc.mixed_qkv, n, qkv_out, s);
  glue::cast_f32_bf16(sc.z_flat, sc.z16, (size_t)n * z_out, s);

  if (getenv("HELIOS_GRMS")) {
    std::vector<float> a((size_t)n * Nv), b((size_t)n * Nv);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(a.data(), sc.a_out, a.size() * 4, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(b.data(), sc.b_out, b.size() * 4, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    auto rms = [](const std::vector<float>& v) { double acc = 0; for (float x : v) acc += (double)x * x;
                                                 return sqrt(acc / v.size()); };
    fprintf(stderr, "[grms] in_proj_a=%.6f in_proj_b=%.6f\n", rms(a), rms(b));
  }

  // 5) causal depthwise conv over q|k|v, with the layer's conv state; swish activation
  aux::cuda_causal_conv1d_update((const bfloat16*)sc.mixed_qkv, (bfloat16*)conv_state, conv_slots,
                                 (const bfloat16*)w.conv, nullptr, (bfloat16*)sc.conv_out, bsz,
                                 qkv_out, sl, cfg.gdn_conv_k, cfg.gdn_conv_k, /*activation=*/true,
                                 /*history=*/false, s);

  // 6) delta-rule recurrence: head-wise decay, single sequence (no slots, no paged history).
  //    Prefill-sized chunks may take the chunked (WY) path - the same recurrence, parallel over
  //    chunks and heads instead of 1024 serial steps - and the serial kernel keeps the T%64 tail
  //    and every small call (decode). It is gated off by default (see gdn_chunked_enabled above),
  //    and cuda_chunked_gated_delta_rule returns false without touching any output when it
  //    declines, so the serial call below runs unless the chunked path actually handled the chunk -
  //    it is the fallback, never a second pass.
  if (!gdn_chunked_enabled() ||
      !aux::cuda_chunked_gated_delta_rule((const bfloat16*)sc.conv_out, sc.g, (const bfloat16*)sc.beta, rec_state,
                                          (bfloat16*)sc.core_out, 1, n, Nk, Nv, Hk, Hv, sc.gdn_chunk_ws,
                                          sc.gdn_chunk_tokens, s))
  {
      aux::cuda_recurrent_gated_delta_rule((const bfloat16*)sc.conv_out, sc.g, (const bfloat16*)sc.beta,
                                           rec_state, (bfloat16*)sc.core_out, bsz, sl, Nk, Nv, Hk, Hv,
                                           /*history_stride=*/slot_layers, gdn_slots,
                                           /*channelwise=*/false, /*history=*/false, s);
  }

  // 7) gated RMS norm over each v-head, sigmoid gate from z, weight [128] shared across heads
  // w.norm stays bf16 as stored (the reference loads it with allow_bf16 and the kernel requires the
  // weight dtype to match the bf16 input).
  aux::gated_rms_norm(sc.core_out, w.norm, aux::kBFloat16, sc.normed, aux::kHalf,
                      sc.z16, aux::kBFloat16, n * Nv, Hv, cfg.rms_eps, 0.0f, 1, false, 1, s);

  if (getenv("HELIOS_GRMS")) {
    // NOTE the dtypes: core_out, z16, beta and conv_out are all bfloat16 (the kernels write bf16);
    // only `normed` is fp16. Reading a bf16 buffer as half reports plausible-looking garbage.
    auto bf_rms = [&](const void* p, size_t cnt) {
      std::vector<unsigned short> t(cnt);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), p, cnt * 2, cudaMemcpyDeviceToHost, s));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      double a = 0;
      for (size_t i = 0; i < cnt; i++) {
        const float f = __bfloat162float(__ushort_as_bfloat16(t[i]));
        a += (double)f * f;
      }
      return sqrt(a / cnt);
    };
    auto half_rms = [&](const void* p, size_t cnt) {
      std::vector<half> t(cnt);
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(t.data(), p, cnt * 2, cudaMemcpyDeviceToHost, s));
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      double a = 0;
      for (size_t i = 0; i < cnt; i++) { const float f = __half2float(t[i]); a += (double)f * f; }
      return sqrt(a / cnt);
    };
    std::vector<unsigned short> wt(128);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(wt.data(), w.norm, 256, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    double aw = 0;
    for (int i = 0; i < 128; i++) {
      const float f = __bfloat162float(__ushort_as_bfloat16(wt[i]));
      aw += (double)f * f;
    }
    std::vector<unsigned short> bt((size_t)n * Nv);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(bt.data(), sc.beta, bt.size() * 2, cudaMemcpyDeviceToHost, s));
    std::vector<float> gv((size_t)n * Nv);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(gv.data(), sc.g, gv.size() * 4, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,
            "[grms] core_out=%.9e z16=%.9e norm_w=%.6f conv_out=%.9e normed=%.9e "
            "beta0=%.4f g0=%.4f\n",
            bf_rms(sc.core_out, (size_t)n * z_out), bf_rms(sc.z16, (size_t)n * z_out),
            sqrt(aw / 128), bf_rms(sc.conv_out, (size_t)n * qkv_out),
            half_rms(sc.normed, (size_t)n * z_out),
            __bfloat162float(__ushort_as_bfloat16(bt[0])), gv[0]);
  }
  if (getenv("HELIOS_GRMS")) {
    std::vector<half> nm((size_t)n * z_out);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(nm.data(), sc.normed, nm.size() * 2, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    double acc = 0;
    for (half v : nm) { const float f = __half2float(v); acc += (double)f * f; }
    fprintf(stderr, "[grms] gated_norm=%.6f\n", sqrt(acc / nm.size()));
  }

  // 8) output projection -> fp32 (the hyper-connection apply needs a fp32 sublayer output)
  exl3::GroupWords og{(const uint16_t*)w.out.trellis, w.out.suh, w.out.svh, w.out.mul1};
  exl3::linear(y, sc.normed, og, n, cfg.hidden, z_out, w.out.K, true, s, sc.a_had);

  // Recurrent-state fingerprint. The decode path is not reproducible run to run while the
  // reference is, and the seed appears at token 1 - i.e. state written by this call and read back
  // by the next one. This prints an order-sensitive checksum of both persistent buffers so two
  // runs can be compared directly instead of inferring the cause from generated text.
  if (getenv("HELIOS_GDNSTATE")) {
    static int layer_no = 0;
    // conv_state is bf16 [qkv_out, k-1]; rec_state is fp32 [Nv, Hv, Hk]
    const size_t conv_n = (size_t)qkv_out * (cfg.gdn_conv_k - 1);
    std::vector<unsigned short> cs(conv_n);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(cs.data(), conv_state, conv_n * 2, cudaMemcpyDeviceToHost, s));
    std::vector<float> rs((size_t)Nv * Hv * Hk);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(rs.data(), rec_state, rs.size() * 4, cudaMemcpyDeviceToHost, s));
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    // FNV-1a over the raw bit patterns: any bit-level difference changes it, and unlike a
    // float sum it does not hide small perturbations in a large reduction.
    uint64_t hc = 1469598103934665603ull, hr = hc;
    for (unsigned short v : cs) { hc ^= v; hc *= 1099511628211ull; }
    for (float f : rs) { unsigned int b; memcpy(&b, &f, 4); hr ^= b; hr *= 1099511628211ull; }
    fprintf(stderr, "[gdnstate] L%d n=%d conv=%016llx rec=%016llx\n", layer_no, n,
            (unsigned long long)hc, (unsigned long long)hr);
    layer_no++;
  }
}

}  // namespace helios
