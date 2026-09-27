// Parity test for the QSA indexer's pooling and block selection.
//
// Scope: the two pieces with real algorithm in them - incremental 4-key mean pooling (with the
// k_layernorm and block-start rope the reference applies) and the top-k block selection. The fused
// index_qk_proj is EXL3-quantised and is covered by the engine's tensor verification instead; testing
// it here would duplicate that.
//
// The reference for both is computed on the host directly from the reference module's description:
//   pooled[j] = rope_at(j*4)( norm(mean_{t in block j} raw_k[t]) )
//   score[j]  = sum_h relu(q_h . pooled[j]) / sqrt(128)
//   select    = top n_sel block ids by score
#include "qsa.cuh"
#include "qwen_rope.cuh"
#include "../aux/norm.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using namespace helios;
using namespace helios::attn;

static int CK(cudaError_t e) { if (e != cudaSuccess) { printf("CUDA %s\n", cudaGetErrorString(e)); return 0; } return 1; }

int main() {
  const int hd = 128, nh = 4, cr = 4, n_sel = 8, nblocks = 40, ntok = nblocks * cr + 3;
  const float eps = 1e-6f, theta = 1e7f, rotary = 64;
  const float scale = 1.0f / sqrtf((float)hd);
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> U(-1.f, 1.f);

  // Inputs
  std::vector<float> rawf(ntok * hd);
  for (auto& v : rawf) v = U(rng);
  std::vector<half> rawh(ntok * hd);
  for (size_t i = 0; i < rawh.size(); i++) rawh[i] = __float2half_rn(rawf[i]);
  std::vector<float> klnf(hd);
  for (auto& v : klnf) v = U(rng);
  std::vector<half> kln(hd);
  for (size_t i = 0; i < kln.size(); i++) kln[i] = __float2half_rn(klnf[i]);

  // Device buffers
  half *draw, *dkln, *dpool, *dpos_h, *tail;
  int *dpos, *dsel; float* dscore;
  size_t pool_sz = (size_t)(nblocks + 2) * hd;
  CK(cudaMalloc(&draw, rawh.size() * 2));
  CK(cudaMalloc(&dkln, kln.size() * 2));
  CK(cudaMalloc(&dpool, pool_sz * 2));
  CK(cudaMalloc(&dpos_h, (size_t)cr * hd * 2));
  tail = dpos_h;                                  // tail_raw is its own small buffer
  CK(cudaMalloc(&dpos, ((size_t)(nblocks + 2)) * 4));
  CK(cudaMalloc(&dsel, (size_t)n_sel * 4));
  CK(cudaMalloc(&dscore, (size_t)nblocks * 4));
  CK(cudaMemcpy(draw, rawh.data(), rawh.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dkln, kln.data(), kln.size() * 2, cudaMemcpyHostToDevice));

  // Feed keys one at a time, exactly as decode does.
  QsaLayerState st{};
  st.pooled_k = dpool; st.tail_raw = tail; st.block_pos = dpos;
  st.n_blocks = 0; st.n_tail = 0; st.max_blocks = nblocks + 2;
  for (int i = 0; i < ntok; i++)
    qsa_pool_update(st, draw + (size_t)i * hd, 1, dkln, hd, rotary, cr, theta, eps, 0);
  CK(cudaDeviceSynchronize());

  int fails = 0;

  // ---- host reference for the pooled blocks ----
  std::vector<float> ref_pool(nblocks * hd);
  for (int b = 0; b < nblocks; b++) {
    std::vector<float> mean(hd, 0.f);
    for (int j = 0; j < cr; j++)
      for (int d = 0; d < hd; d++) mean[d] += rawf[(size_t)(b * cr + j) * hd + d];
    for (auto& v : mean) v /= cr;
    // RMSNorm, constant_bias 1
    float ss = 0.f;
    for (auto v : mean) ss += v * v;
    const float r = 1.f / sqrtf(ss / hd + eps);
    for (int d = 0; d < hd; d++) mean[d] = mean[d] * r * (klnf[d] + 1.0f);
    // partial NEOX rope at position b*cr, over the leading `rotary` channels
    const int pos = b * cr;
    for (int d = 0; d < rotary / 2; d++) {
      const float a = pos * powf(1.f / theta, 2.f * d / rotary);
      const float cs = cosf(a), sn = sinf(a);
      const float x1 = mean[d], x2 = mean[d + rotary / 2];
      mean[d] = x1 * cs - x2 * sn;
      mean[d + rotary / 2] = x1 * sn + x2 * cs;
    }
    for (int d = 0; d < hd; d++) ref_pool[(size_t)b * hd + d] = mean[d];
  }

  std::vector<half> got_pool((size_t)nblocks * hd);
  CK(cudaMemcpy(got_pool.data(), dpool, got_pool.size() * 2, cudaMemcpyDeviceToHost));
  double emax = 0, mag = 0;
  for (size_t i = 0; i < ref_pool.size(); i++) {
    emax = std::max(emax, std::fabs((double)__half2float(got_pool[i]) - ref_pool[i]));
    mag += std::fabs(ref_pool[i]);
  }
  mag /= ref_pool.size();
  const bool pool_ok = emax < 2e-2;
  printf("[qsa_pool] max_err %.3e on |x|~%.3f (%d blocks, tail=%d)  %s\n", emax, mag, st.n_blocks,
         st.n_tail, pool_ok ? "PASS" : "FAIL");
  fails += pool_ok ? 0 : 1;

  // ---- selection ----
  // One query vector (already normalised+roped upstream; here just random, the selection math is
  // what is under test).
  std::vector<float> qf(nh * hd);
  for (auto& v : qf) v = U(rng);
  half* dq;
  CK(cudaMalloc(&dq, qf.size() * 2));
  { std::vector<half> qh(nh * hd); for (size_t i = 0; i < qh.size(); i++) qh[i] = __float2half_rn(qf[i]);
    CK(cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice)); }

  int i0 = 0;
  QsaScratch sc{};
  sc.n = 1; sc.n_heads = nh; sc.head_dim = hd; sc.n_sel = n_sel;
  sc.q = dq; sc.sel = dsel; sc.score = dscore; sc.raw_k = draw; sc.qk = draw;
  sc.pos = (int*)malloc(4); CK(cudaMemcpy(sc.pos, &i0, 4, cudaMemcpyHostToDevice));
  qsa_select(st, sc, nh, hd, n_sel, scale, 0);
  CK(cudaDeviceSynchronize());

  // host reference
  std::vector<std::pair<float, int>> ranked;
  for (int b = 0; b < nblocks; b++) {
    float acc = 0.f;
    for (int h = 0; h < nh; h++) {
      float d = 0.f;
      for (int t = 0; t < hd; t++) d += qf[(size_t)h * hd + t] * ref_pool[(size_t)b * hd + t];
      acc += fmaxf(d, 0.f);
    }
    ranked.push_back({acc * scale, b});
  }
  std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second; });
  std::vector<int> want;
  for (int i = 0; i < n_sel; i++) want.push_back(ranked[i].second);
  std::sort(want.begin(), want.end());

  std::vector<int> got(n_sel);
  CK(cudaMemcpy(got.data(), dsel, got.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<int> gotset;
  for (int v : got) if (v >= 0) gotset.push_back(v);
  std::sort(gotset.begin(), gotset.end());
  const bool sel_ok = (gotset == want);
  printf("[qsa_sel] got %zu blocks, want %d  %s\n", gotset.size(), n_sel, sel_ok ? "PASS" : "FAIL");
  if (!sel_ok) {
    printf("  want:"); for (int v : want) printf(" %d", v); printf("\n  got :");
    for (int v : gotset) printf(" %d", v); printf("\n");
  }
  fails += sel_ok ? 0 : 1;

  // ---- sparse decode attention: must equal dense over the SAME token subset ----
  {
    const int nq = 4, nkv = 1, hdd = 256, T = 37, row0 = 0, pos = T - 1;
    const int cap = T + 8;
    std::vector<half> qh((size_t)1 * nq * hdd), kh((size_t)T * nkv * hdd), vh((size_t)T * nkv * hdd);
    for (auto& v : qh) v = __float2half_rn(U(rng));
    for (auto& v : kh) v = __float2half_rn(U(rng));
    for (auto& v : vh) v = __float2half_rn(U(rng));
    half *dq2, *dk2, *dv2, *dout2; int* didx; int* doff; float *dpart, *dgrp;
    CK(cudaMalloc(&dq2, qh.size() * 2)); CK(cudaMalloc(&dk2, kh.size() * 2));
    CK(cudaMalloc(&dv2, vh.size() * 2)); CK(cudaMalloc(&dout2, qh.size() * 2));
    CK(cudaMalloc(&didx, cap * 4)); CK(cudaMalloc(&doff, cap * 4));
    const int nch = (cap + 15) / 16;
    CK(cudaMalloc(&dpart, (size_t)1 * nq * nch * (2 + hdd) * 4));
    CK(cudaMalloc(&dgrp, (size_t)1 * nq * 64 * (2 + hdd) * 4));
    CK(cudaMemcpy(dq2, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk2, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv2, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice));
    // gather every token, ascending (the whole causal range)
    std::vector<int> idx(T);
    for (int i = 0; i < T; i++) idx[i] = i;
    CK(cudaMemcpy(didx, idx.data(), T * 4, cudaMemcpyHostToDevice));
    std::vector<int> off(cap, 0); off[0] = T;
    CK(cudaMemcpy(doff, off.data(), cap * 4, cudaMemcpyHostToDevice));

    const float sc = 1.0f / sqrtf((float)hdd);
    // host reference over the same subset
    std::vector<float> ref(qh.size(), 0.f);
    for (int h = 0; h < nq; h++) {
      std::vector<double> acc(hdd, 0.0); double mx = -1e30, sum = 0.0;
      std::vector<double> sc_(T);
      for (int t = 0; t < T; t++) {
        double d = 0.0;
        for (int dd = 0; dd < hdd; dd++)
          d += (double)__half2float(qh[(size_t)h * hdd + dd]) * __half2float(kh[(size_t)t * hdd + dd]);
        sc_[t] = d * sc;
        if (sc_[t] > mx) mx = sc_[t];
      }
      for (int t = 0; t < T; t++) {
        const double e = exp(sc_[t] - mx); sum += e;
        for (int dd = 0; dd < hdd; dd++) acc[dd] += e * __half2float(vh[(size_t)t * hdd + dd]);
      }
      for (int dd = 0; dd < hdd; dd++) ref[(size_t)h * hdd + dd] = (float)(acc[dd] / sum);
    }
    gqa_sparse_decode(dq2, dk2, dv2, didx, doff, cap, dout2, dpart, dgrp, 1, nq, nkv, hdd, sc, 0);
    CK(cudaDeviceSynchronize());
    std::vector<half> got(qh.size());
    CK(cudaMemcpy(got.data(), dout2, got.size() * 2, cudaMemcpyDeviceToHost));
    double e2 = 0, m2 = 0;
    for (size_t i = 0; i < ref.size(); i++) {
      e2 = std::max(e2, std::fabs((double)__half2float(got[i]) - ref[i]));
      m2 += std::fabs(ref[i]);
    }
    m2 /= ref.size();
    const bool sp_ok = e2 < 3e-3;
    printf("[qsa_sparse] max_err %.3e on |o|~%.3f (%d of %d tokens)  %s\n", e2, m2, T, T,
           sp_ok ? "PASS" : "FAIL");
    fails += sp_ok ? 0 : 1;

    // and a strict SUBSET must equal a reference over just that subset
    const int keep = 11;
    std::vector<int> sidx(keep);
    for (int i = 0; i < keep; i++) sidx[i] = i * 3;      // a sparse, non-contiguous subset
    CK(cudaMemcpy(didx, sidx.data(), keep * 4, cudaMemcpyHostToDevice));
    std::vector<int> off2(cap, 0); off2[0] = keep;
    CK(cudaMemcpy(doff, off2.data(), cap * 4, cudaMemcpyHostToDevice));
    std::vector<float> ref2(qh.size(), 0.f);
    for (int h = 0; h < nq; h++) {
      std::vector<double> acc(hdd, 0.0); double mx = -1e30, sum = 0.0;
      std::vector<double> sc2(keep);
      for (int i = 0; i < keep; i++) {
        double d = 0.0;
        for (int dd = 0; dd < hdd; dd++)
          d += (double)__half2float(qh[(size_t)h * hdd + dd]) * __half2float(kh[(size_t)sidx[i] * hdd + dd]);
        sc2[i] = d * sc;
        if (sc2[i] > mx) mx = sc2[i];
      }
      for (int i = 0; i < keep; i++) {
        const double e = exp(sc2[i] - mx); sum += e;
        for (int dd = 0; dd < hdd; dd++) acc[dd] += e * __half2float(vh[(size_t)sidx[i] * hdd + dd]);
      }
      for (int dd = 0; dd < hdd; dd++) ref2[(size_t)h * hdd + dd] = (float)(acc[dd] / sum);
    }
    gqa_sparse_decode(dq2, dk2, dv2, didx, doff, cap, dout2, dpart, dgrp, 1, nq, nkv, hdd, sc, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout2, got.size() * 2, cudaMemcpyDeviceToHost));
    double e3 = 0, m3 = 0;
    for (size_t i = 0; i < ref2.size(); i++) {
      e3 = std::max(e3, std::fabs((double)__half2float(got[i]) - ref2[i]));
      m3 += std::fabs(ref2[i]);
    }
    m3 /= ref2.size();
    const bool sp2_ok = e3 < 3e-3;
    printf("[qsa_sparse_subset] max_err %.3e on |o|~%.3f (%d of %d tokens)  %s\n", e3, m3, keep, T,
           sp2_ok ? "PASS" : "FAIL");
    fails += sp2_ok ? 0 : 1;
  }

  printf(fails ? "QSA PARITY: %d FAIL\n" : "QSA PARITY: ALL PASS\n", fails);
  return fails;
}
