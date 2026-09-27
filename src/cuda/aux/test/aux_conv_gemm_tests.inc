// Parity for the two kernels this file's build now carries from the engine layer: the tiled
// causal conv1d in gdn.cu (bit-exact against a CPU reference over the shape corners the engine
// actually hits) and the tensor-core PLE projection GEMM in engine/glue2.cu (rms relative error
// against an fp32 CPU reference, on both the tensor-core and the scalar path, so a regression in
// either is caught).
//
// Included from test_aux.cpp; not a translation unit of its own.

namespace {

// CPU reference for cuda_causal_conv1d_update. The input sequence is the concatenation
// seq = [conv_state[slot,d,0:K], x[b,d,0:seqlen]], and
//     out[b,s,d] = act(bias[d] + sum_k w[d,k] * seq[s+1+k]).
// Without history the state writeback is state[slot,d,k] = seq[seqlen+k]; with history it is the
// last min(state_size, K+seqlen) entries of seq, written at the tail of the state buffer.
//
// The activation is `_sigmoid_fast_exp` = 1 / (1 + __expf(-x)), and __expf is the approximate ex2
// path, which no host expf reproduces to the bit. The fmaf chain, the window bookkeeping, the
// conv_state writeback and the bf16 store are all exactly reproducible, so those are asserted
// bit-exact; the gated OUTPUT is asserted within one bf16 ulp. Each case runs the kernel twice on
// the same inputs, once gated and once not, so a fault in the window or the state shows up in the
// ungated arm instead of hiding behind the activation's tolerance.
void conv1d_ref(const __nv_bfloat16* x, const __nv_bfloat16* state0, const __nv_bfloat16* w,
                const __nv_bfloat16* bias, __nv_bfloat16* out, __nv_bfloat16* state1, int bsz,
                int dim, int seqlen, int state_size, int K, bool activation, bool history) {
    const int total = K + seqlen;
    const int write = state_size < total ? state_size : total;
    for (int b = 0; b < bsz; ++b) {
        for (int d = 0; d < dim; ++d) {
            auto seq = [&](int t) -> float {
                return t < K ? __bfloat162float(state0[((size_t)b * dim + d) * state_size + t])
                             : __bfloat162float(x[((size_t)b * dim + d) * seqlen + (t - K)]);
            };
            for (int s = 0; s < seqlen; ++s) {
                float acc = bias ? __bfloat162float(bias[d]) : 0.0f;
                for (int k = 0; k < K; ++k) acc = std::fmaf(__bfloat162float(w[(size_t)d * K + k]),
                                                             seq(s + 1 + k), acc);
                if (activation) acc *= 1.0f / (1.0f + std::exp2(-acc * 1.4426950408889634f));
                out[((size_t)b * seqlen + s) * dim + d] = __float2bfloat16_rn(acc);
            }
            __nv_bfloat16* st = state1 + (size_t)b * dim * state_size + (size_t)d * state_size;
            for (int k = 0; k < K; ++k) st[k] = __float2bfloat16_rn(seq(seqlen + k));
            if (history)
                for (int j = 0; j < write; ++j)
                    st[state_size - write + j] = __float2bfloat16_rn(seq(total - write + j));
        }
    }
}

// Width of one bf16 ulp at v (bf16 keeps 8 significand bits, so a ulp is 2^-7 of the binade).
double bf16_ulp(double v) {
    const double a = std::fabs(v);
    return a > 0.0 ? std::ldexp(1.0, std::ilogb(a) - 7) : 0.0;
}

void test_conv1d_case(int bsz, int dim, int seqlen, int state_size, int K, bool activation,
                      bool history, bool with_bias, std::mt19937& rng) {
    const size_t nx = (size_t)bsz * dim * seqlen;
    const size_t nst = (size_t)bsz * dim * state_size;   // conv_state is [num_slots, dim, ...]
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<__nv_bfloat16> hx(nx), hw((size_t)dim * K), hb(dim), hst(nst);
    for (size_t i = 0; i < nx; ++i) hx[i] = __float2bfloat16(dist(rng));
    for (size_t i = 0; i < (size_t)dim * K; ++i) hw[i] = __float2bfloat16(dist(rng));
    for (int i = 0; i < dim; ++i) hb[i] = __float2bfloat16(dist(rng));
    for (size_t i = 0; i < nst; ++i) hst[i] = __float2bfloat16(dist(rng));

    __nv_bfloat16 *dx, *dw, *db, *dout, *dstate;
    HELIOS_CUDA_CHECK(cudaMalloc(&dx, nx * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&dw, (size_t)dim * K * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&db, dim * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&dout, nx * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&dstate, nst * 2));
    HELIOS_CUDA_CHECK(cudaMemcpy(dx, hx.data(), nx * 2, cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(dw, hw.data(), (size_t)dim * K * 2, cudaMemcpyHostToDevice));
    if (with_bias) HELIOS_CUDA_CHECK(cudaMemcpy(db, hb.data(), dim * 2, cudaMemcpyHostToDevice));

    // Two arms on the same inputs: `act` is the one under test, `!act` is the bit-exact control.
    std::vector<__nv_bfloat16> want_out(nx), want_state(nst), got_out(nx), got_state(nst);
    for (int arm = 0; arm < 2; ++arm) {
        const bool act = arm == 0 ? activation : !activation;
        HELIOS_CUDA_CHECK(cudaMemcpy(dstate, hst.data(), nst * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemset(dout, 0, nx * 2));
        cuda_causal_conv1d_update(dx, dstate, nullptr, dw, with_bias ? db : nullptr, dout, bsz, dim,
                                  seqlen, state_size, K, act, history, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
        HELIOS_CUDA_CHECK(cudaMemcpy(got_out.data(), dout, nx * 2, cudaMemcpyDeviceToHost));
        HELIOS_CUDA_CHECK(cudaMemcpy(got_state.data(), dstate, nst * 2, cudaMemcpyDeviceToHost));
        conv1d_ref(hx.data(), hst.data(), hw.data(), with_bias ? hb.data() : nullptr,
                   want_out.data(), want_state.data(), bsz, dim, seqlen, state_size, K, act,
                   history);

        // Only the entries the kernel is meant to touch are compared: state[0:K] without history,
        // the last min(state_size, K+seqlen) entries with it, per sequence.
        const int total_in = K + seqlen;
        const int wsize = state_size < total_in ? state_size : total_in;
        for (int b = 0; b < bsz; ++b)
            for (int d = 0; d < dim; ++d)
                for (int k = history ? state_size - wsize : 0; k < (history ? state_size : K); ++k) {
                    const size_t i = ((size_t)b * dim + d) * state_size + k;
                    if (__bfloat162float(got_state[i]) == __bfloat162float(want_state[i])) continue;
                    printf("[conv1d] state mismatch bsz=%d dim=%d sl=%d K=%d act=%d hist=%d "
                           "bias=%d at b=%d d=%d k=%d got %.9g want %.9g\n", bsz, dim, seqlen, K,
                           (int)act, (int)history, (int)with_bias, b, d, k,
                           __bfloat162float(got_state[i]), __bfloat162float(want_state[i]));
                    fail("causal_conv1d_update conv_state is not bit-exact against the reference");
                }
        for (size_t i = 0; i < nx; ++i) {
            const double g = __bfloat162float(got_out[i]), w = __bfloat162float(want_out[i]);
            if (g == w) continue;
            if (act && std::fabs(g - w) <= bf16_ulp(w)) continue;
            printf("[conv1d] out mismatch bsz=%d dim=%d sl=%d K=%d act=%d hist=%d bias=%d at "
                   "b=%zu s=%zu d=%zu got %.9g want %.9g\n", bsz, dim, seqlen, K, (int)act,
                   (int)history, (int)with_bias, i / ((size_t)seqlen * dim), (i / dim) % seqlen,
                   i % dim, g, w);
            fail("causal_conv1d_update output disagrees with the CPU reference");
        }
    }

    HELIOS_CUDA_CHECK(cudaFree(dx)); HELIOS_CUDA_CHECK(cudaFree(dw));
    HELIOS_CUDA_CHECK(cudaFree(db)); HELIOS_CUDA_CHECK(cudaFree(dout));
    HELIOS_CUDA_CHECK(cudaFree(dstate));
}

void test_conv1d() {
    std::mt19937 rng(7);
    // dim and seqlen corners: a whole number of 64-wide channel tiles, a partial one, a seqlen
    // shorter than a tile, and the decode shape (seqlen = 1). K spans 1..8, and the history
    // (rewindable state) variant is exercised alongside the ordinary one.
    const int dims[] = {64, 130};
    const int lens[] = {1, 5, 64, 130, 256};
    for (int dim : dims)
        for (int sl : lens)
            for (int K = 1; K <= 8; ++K)
                for (bool act : {false, true})
                    for (bool hist : {false, true})
                        test_conv1d_case(1, dim, sl, 8, K, act, hist, (K & 1) == 0, rng);
    // A two-sequence call exercises the batch axis of the grid and the per-slot state.
    test_conv1d_case(2, 96, 70, 8, 4, true, false, false, rng);
    success("causal_conv1d_update (tiled) matches the CPU reference");
}

void test_gemm_nt_f16_case(int M, int N, int K, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half> hx((size_t)M * K), hw((size_t)N * K);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half(dist(rng));
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = __float2half(dist(rng));

    double sr = 0.0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double a = 0.0;
            for (int k = 0; k < K; ++k)
                a += (double)__half2float(hx[(size_t)m * K + k]) *
                     (double)__half2float(hw[(size_t)n * K + k]);
            sr += a * a;
        }

    half *dx, *dw;
    void* dy;
    HELIOS_CUDA_CHECK(cudaMalloc(&dx, hx.size() * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&dw, hw.size() * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&dy, (size_t)M * N * 2));
    HELIOS_CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    glue::gemm_nt_f16(dy, dx, dw, M, N, K, false, false, 0);
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> hy((size_t)M * N);
    HELIOS_CUDA_CHECK(cudaMemcpy(hy.data(), dy, hy.size() * 2, cudaMemcpyDeviceToHost));

    double se = 0.0, maxrel = 0.0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double want = 0.0;
            for (int k = 0; k < K; ++k)
                want += (double)__half2float(hx[(size_t)m * K + k]) *
                        (double)__half2float(hw[(size_t)n * K + k]);
            const double got = __half2float(hy[(size_t)m * N + n]);
            se += (got - want) * (got - want);
            maxrel = std::max(maxrel, std::fabs(got - want) / std::max(1e-3, std::fabs(want)));
        }
    const double rms = std::sqrt(se / sr);
    // fp16 operands, fp32 accumulate, fp16 store: the store alone is worth 2^-11 relative, and a
    // K-long fp32 sum in a different order adds ~1e-6 on top.
    if (!(rms < 1e-3)) fail("gemm_nt_f16 rms against the CPU reference");
    if (!(maxrel < 5e-3)) fail("gemm_nt_f16 max relative error");
    printf("[gemm_nt_f16] M=%d N=%d K=%d: max rel %.3e, rms rel %.3e\n", M, N, K, maxrel, rms);
    HELIOS_CUDA_CHECK(cudaFree(dx)); HELIOS_CUDA_CHECK(cudaFree(dw));
    HELIOS_CUDA_CHECK(cudaFree(dy));
}

void test_gemm_nt_f16() {
    std::mt19937 rng(11);
    // M spans the scalar kernel's regime (below one 128-row tile), exactly one tile, and a partial
    // last tile; N and K are non-multiples of the 128/16 block shape on purpose.
    for (int M : {1, 7, 63, 64, 128, 200, 256}) test_gemm_nt_f16_case(M, 300, 96, rng);
    test_gemm_nt_f16_case(64, 64, 64, rng);
    success("gemm_nt_f16 (tensor-core and scalar paths) matches the CPU reference");
}

// The GDN's channel-major mixed_qkv comes out of this one, so the tiled transpose is required to
// be bit-identical to a per-element copy, not merely close. M and F are deliberately not
// multiples of the 32x32 tile, so the ragged last tile is covered too.
void test_transpose_f32_bf16() {
    std::mt19937 rng(23);
    for (auto shape : {std::pair<int, int>{1024, 300}, {130, 300}, {33, 65}, {1, 1}}) {
        const int M = shape.first, F = shape.second;
        const size_t n = (size_t)M * F;
        std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
        std::vector<float> hs(n);
        for (size_t i = 0; i < n; ++i) hs[i] = dist(rng);
        float* d; void* o;
        HELIOS_CUDA_CHECK(cudaMalloc(&d, n * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&o, n * 2));
        HELIOS_CUDA_CHECK(cudaMemcpy(d, hs.data(), n * 4, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemset(o, 0xa5, n * 2));
        glue::transpose_f32_bf16(d, o, M, F, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned short> ho(n);
        HELIOS_CUDA_CHECK(cudaMemcpy(ho.data(), o, n * 2, cudaMemcpyDeviceToHost));
        for (int m = 0; m < M; ++m)
            for (int f = 0; f < F; ++f)
                if (ho[(size_t)f * M + m] != __bfloat16_as_ushort(__float2bfloat16(hs[(size_t)m * F + f])))
                    fail("transpose_f32_bf16 is not bit-exact against a per-element copy");
        HELIOS_CUDA_CHECK(cudaFree(d));
        HELIOS_CUDA_CHECK(cudaFree(o));
    }
    success("transpose_f32_bf16 (tiled) is bit-exact against a per-element copy");
}


}  // namespace
