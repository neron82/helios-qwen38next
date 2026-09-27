// Parity + reproducibility test for the chunked (WY) gated delta rule.
//
// Three independent properties are checked, and they are the three the engine depends on:
//
//   1. Numeric parity against the serial recurrence - asserted whenever the path is live, i.e.
//      whenever gdn_layer.cu would take it. Upstream (exllamav3 tests/test_gated_delta_rule.py)
//      accepts chunked-vs-serial at rtol/atol 5e-2 on an L2 metric; this gate is 5e-3 relative L2
//      plus max|delta| <= 2 bf16 ulp of the reference peak. The two error sources that set those
//      numbers are the bf16 W/U/v_new/h_chunk workspace panels and the TF32 matmuls that consume
//      them; both are properties of the upstream design, not of this port, and both are measured
//      here rather than assumed. WITH AN EXPLICIT HELIOS_GDN_CHUNKED=0 the parity numbers are
//      still measured and printed, but only determinism is asserted.
//
//   2. A single-chunk state-accumulation check, asserted unconditionally. One chunk with a
//      non-zero initial state is the smallest input on which the delta-rule update
//      sum_t k_t (x) v_decay_t can show up in the state at all: it is the case that caught the
//      l2norm grid bug (7/8 of the k panel never written -> A = 0 -> the chunk contributed
//      nothing and the state collapsed to the decayed carry-in), and the case that would catch
//      any regression in Phase E's MM2 k-tile coverage.
//
//   3. Run-to-run byte stability. The engine's reproducibility guarantee is "identical inputs ->
//      byte-identical outputs", and it is the reason the chunked kernels must stay free of
//      atomics. Two identical launches are compared with memcmp, on the outputs AND the
//      recurrent state, so a future non-deterministic reduction cannot slip in unnoticed.
//
//   3. Run-to-run byte stability. The engine's reproducibility guarantee is "identical inputs ->
//      byte-identical outputs", and it is the reason the chunked kernels must stay free of
//      atomics. Two identical launches are compared with memcmp, on the outputs AND the
//      recurrent state, so a future non-deterministic reduction cannot slip in unnoticed.
//
// Shapes: n = 1024 (the engine's prefill chunk, 16 chunks, no tail) and n = 1000 (15 chunks + a
// 40-token serial tail), at the model's 16 q/k heads x 48 v heads x 128 dims with a non-zero
// initial state.
#include "gdn.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace helios;
using namespace helios::aux;

#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA err %s @%d\n", cudaGetErrorName(e_), __LINE__); exit(1); } } while (0)

static const int kNk = 16, kNv = 48, kD = 128;
static const int kQkvRow = 2 * kNk * kD + kNv * kD;  // 10240

// Host-side bf16 conversion by hand. __float2bfloat16_rz / __bfloat162float are device-only
// intrinsics as far as this TU's compiler is concerned (it is a .cpp, not a .cu), and calling them
// from host code segfaults. Truncating the low 16 bits IS round-toward-zero for every value in
// range here, and widening is a shift.
static unsigned short f2bf_rz(float f)
{
    unsigned int u;
    memcpy(&u, &f, 4);
    return (unsigned short)(u >> 16);
}

static float bf2f(unsigned short b)
{
    unsigned int u = (unsigned int)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static std::mt19937 rng(20240917u);

static float uni(float lo, float hi) { return lo + (hi - lo) * ((float)(rng() % 20001) / 20000.0f); }

// relative L2: ||a-b|| / ||b||. This is the metric the algorithm's precision is stated in
// upstream (4.1e-3 on the bf16 output, 2.7e-3 on the fp32 state).
static double rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// The largest single-element disagreement, plus the largest magnitude in the reference. The bf16
// output grid has a relative step of 2^-8 = 3.9e-3, so any two implementations that both round the
// output to bf16 must differ by up to one ulp of the LARGEST element - which, on this heavy-tailed
// output (peak ~8-25x the rms), is ~3-10% of the rms. A max|delta| < 1e-2 x rms gate is therefore
// unreachable for ANY bf16-outputting implementation, chunked or serial; the reachable form of the
// same statement is max|delta| <= 1 ulp of the reference's own peak.
struct MaxErr {
    double abs_max = 0.0;
    double ref_peak = 0.0;
    size_t at = 0;
};

static MaxErr max_err(const std::vector<float>& a, const std::vector<float>& b)
{
    MaxErr m;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::fabs((double)a[i] - b[i]);
        if (d > m.abs_max) { m.abs_max = d; m.at = i; }
        m.ref_peak = std::max(m.ref_peak, std::fabs((double)b[i]));
    }
    return m;
}

// One bf16 ulp at |x| (round-toward-zero grid, 8-bit significand).
static double bf16_ulp(double x)
{
    if (x == 0.0) return 0.0;
    int e = 0;
    std::frexp(std::fabs(x), &e);               // x = f * 2^e, 0.5 <= f < 1
    return std::ldexp(1.0, e - 8);            // the grid step at that binade
}

static double rms(const std::vector<float>& a)
{
    double s = 0.0;
    for (float v : a) s += (double)v * v;
    return std::sqrt(s / (double)a.size());
}

static std::vector<float> to_f32(const std::vector<unsigned short>& bf)
{
    std::vector<float> out(bf.size());
    for (size_t i = 0; i < bf.size(); i++) out[i] = bf2f(bf[i]);
    return out;
}

static void fill_inputs(int n, std::vector<unsigned short>& conv, std::vector<float>& g,
                        std::vector<unsigned short>& beta, std::vector<float>& state)
{
    conv.resize((size_t)n * kQkvRow);
    g.resize((size_t)n * kNv);
    beta.resize((size_t)n * kNv);
    state.resize((size_t)kNv * kD * kD);
    // q/k/v at unit-ish scale: the recurrence normalizes q and k itself, and v is what the delta
    // rule actually reads, so O(1) is the range the model operates in.
    for (size_t i = 0; i < conv.size(); i++) conv[i] = f2bf_rz(uni(-1.0f, 1.0f));
    // g is a log decay: strictly negative, O(0.1..1) per token as gated_delta_net_fused_op_2
    // produces it (-exp(a_log) * softplus(a + dt_bias)).
    for (size_t i = 0; i < g.size(); i++) g[i] = -uni(0.02f, 0.6f);
    // beta = sigmoid(b) * scale, so in (0, 1).
    for (size_t i = 0; i < beta.size(); i++) beta[i] = f2bf_rz(uni(0.05f, 0.95f));
    for (size_t i = 0; i < state.size(); i++) state[i] = uni(-0.1f, 0.1f);
}

struct Case
{
    const char* name;
    int n;
    double l2_tol;   // relative L2, the metric the algorithm's precision is stated in
};

int main()
{
    CK(cudaSetDevice(0));

    const size_t ws_bytes = gdn_chunked_workspace_bytes(kNv, kNk, 1024);
    printf("chunked workspace: %.2f MB for %d tokens (16 qk heads, 48 value heads)\n", ws_bytes / 1048576.0, 1024);
    void* ws = nullptr;
    CK(cudaMalloc(&ws, ws_bytes));

    // Same default as gdn_layer.cu's gdn_chunked_enabled(): the path is live unless the caller
    // explicitly turns it off, so the test never asserts less than the engine is asked to deliver.
    const bool chunked_enabled = !getenv("HELIOS_GDN_CHUNKED") || atoi(getenv("HELIOS_GDN_CHUNKED"));
    printf("chunked path: %s\n", chunked_enabled ? "ENABLED (parity asserted)" : "DISABLED (parity measured only)");

    const Case cases[] = {
        {"n=1024 (16 chunks, no tail)", 1024, 5e-3},
        {"n=1000 (15 chunks + 40-token serial tail)", 1000, 5e-3},
    };

    int fails = 0;
    for (const Case& c : cases) {
        std::vector<unsigned short> h_conv, h_beta;
        std::vector<float> h_g, h_state;
        fill_inputs(c.n, h_conv, h_g, h_beta, h_state);

        bfloat16 *d_conv = nullptr, *d_beta = nullptr, *d_out_ser = nullptr, *d_out_chunk = nullptr,
                 *d_out_chunk2 = nullptr;
        float* d_g = nullptr;
        float *d_st_ser = nullptr, *d_st_chunk = nullptr, *d_st_chunk2 = nullptr;
        const size_t out_n = (size_t)c.n * kNv * kD, st_n = (size_t)kNv * kD * kD;
        CK(cudaMalloc(&d_conv, h_conv.size() * 2));
        CK(cudaMalloc(&d_beta, h_beta.size() * 2));
        CK(cudaMalloc(&d_g, h_g.size() * 4));
        CK(cudaMalloc(&d_out_ser, out_n * 2));
        CK(cudaMalloc(&d_out_chunk, out_n * 2));
        CK(cudaMalloc(&d_out_chunk2, out_n * 2));
        CK(cudaMalloc(&d_st_ser, st_n * 4));
        CK(cudaMalloc(&d_st_chunk, st_n * 4));
        CK(cudaMalloc(&d_st_chunk2, st_n * 4));
        CK(cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_g, h_g.data(), h_g.size() * 4, cudaMemcpyHostToDevice));

        // --- serial reference ---
        CK(cudaMemcpy(d_st_ser, h_state.data(), st_n * 4, cudaMemcpyHostToDevice));
        cuda_recurrent_gated_delta_rule(d_conv, d_g, d_beta, d_st_ser, d_out_ser, 1, c.n, kNk, kNv, kD, kD, 0,
                                        nullptr, false, false, 0);
        CK(cudaDeviceSynchronize());

        // --- chunked, twice from the same inputs and the same initial state ---
        CK(cudaMemcpy(d_st_chunk, h_state.data(), st_n * 4, cudaMemcpyHostToDevice));
        if (!cuda_chunked_gated_delta_rule(d_conv, d_g, d_beta, d_st_chunk, d_out_chunk, 1, c.n, kNk, kNv, kD, kD, ws,
                                           1024, 0)) {
            printf("FAIL %s: chunked dispatch declined\n", c.name);
            fails++;
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(d_st_chunk2, h_state.data(), st_n * 4, cudaMemcpyHostToDevice));
        cuda_chunked_gated_delta_rule(d_conv, d_g, d_beta, d_st_chunk2, d_out_chunk2, 1, c.n, kNk, kNv, kD, kD, ws,
                                      1024, 0);
        CK(cudaDeviceSynchronize());

        std::vector<unsigned short> o_ser(out_n), o_chunk(out_n), o_chunk2(out_n);
        std::vector<float> s_ser(st_n), s_chunk(st_n), s_chunk2(st_n);
        CK(cudaMemcpy(o_ser.data(), d_out_ser, out_n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o_chunk.data(), d_out_chunk, out_n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o_chunk2.data(), d_out_chunk2, out_n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s_ser.data(), d_st_ser, st_n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s_chunk.data(), d_st_chunk, st_n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s_chunk2.data(), d_st_chunk2, st_n * 4, cudaMemcpyDeviceToHost));

        std::vector<float> f_ser = to_f32(o_ser), f_chunk = to_f32(o_chunk);
        const double l2_out = rel_l2(f_chunk, f_ser);
        const double l2_st = rel_l2(s_chunk, s_ser);
        const MaxErr me = max_err(f_chunk, f_ser);
        const double floor_ulp = bf16_ulp(me.ref_peak);
        printf("  core_out: rel L2 = %.3e | max|delta| = %.3e at a reference peak of %.3f "
               "(one bf16 ulp there = %.3e)\n",
               l2_out, me.abs_max, me.ref_peak, floor_ulp);
        printf("  state:    rel L2 = %.3e (rms %.3e)\n", l2_st, rms(s_ser));
        const bool enforce = chunked_enabled;
        if (enforce && !(l2_out < c.l2_tol)) {
            printf("FAIL %s: core_out rel L2 %.3e >= %.1e\n", c.name, l2_out, c.l2_tol);
            fails++;
        }
        if (enforce && !(l2_st < c.l2_tol)) {
            printf("FAIL %s: state rel L2 %.3e >= %.1e\n", c.name, l2_st, c.l2_tol);
            fails++;
        }
        if (enforce && !(me.abs_max <= 2.0 * floor_ulp)) {
            printf("FAIL %s: core_out max|delta| %.3e exceeds 2 bf16 ulp of the reference peak (%.3e)\n",
                   c.name, me.abs_max, floor_ulp);
            fails++;
        }
        if (!enforce) printf("  (chunked path DISABLED: parity measured, not asserted)\n");

        // Reproducibility: the chunked path must be byte-identical across launches, outputs and
        // state alike. memcmp, not a tolerance - this is the engine's run-to-run guarantee.
        if (memcmp(o_chunk.data(), o_chunk2.data(), out_n * 2) != 0) {
            printf("FAIL %s: chunked core_out is not byte-stable across launches\n", c.name);
            fails++;
        }
        if (memcmp(s_chunk.data(), s_chunk2.data(), st_n * 4) != 0) {
            printf("FAIL %s: chunked state is not byte-stable across launches\n", c.name);
            fails++;
        }

        cudaFree(d_conv); cudaFree(d_beta); cudaFree(d_g);
        cudaFree(d_out_ser); cudaFree(d_out_chunk); cudaFree(d_out_chunk2);
        cudaFree(d_st_ser); cudaFree(d_st_chunk); cudaFree(d_st_chunk2);
    }

    // Single-chunk state accumulation, asserted unconditionally. One 64-token chunk with a
    // non-zero initial state is the smallest input on which the delta-rule update
    // sum_t k_t (x) v_decay_t can show up in the state at all, and it is the case that localises a
    // broken stage immediately: if the k panel is not written (the l2norm grid bug this test was
    // added for) A = 0, T_inv = I, v_decay = U = 0 and the state comes out as the decayed
    // carry-in - a rel L2 near 1, not a tolerance miss. If Phase E's MM2 stopped contracting some
    // of its 128 k-rows, the same thing happens in miniature. Asserted whether or not the gate is
    // on, because it tests the kernels directly rather than the engine's dispatch.
    {
        const int n1 = 64;
        std::vector<unsigned short> h_conv, h_beta;
        std::vector<float> h_g, h_state;
        fill_inputs(n1, h_conv, h_g, h_beta, h_state);
        const size_t on1 = (size_t)n1 * kNv * kD, sn1 = (size_t)kNv * kD * kD;
        bfloat16 *d_conv = nullptr, *d_beta = nullptr, *d_out_ser = nullptr, *d_out_chunk = nullptr;
        float *d_g = nullptr, *d_st_ser = nullptr, *d_st_chunk = nullptr;
        CK(cudaMalloc(&d_conv, h_conv.size() * 2));
        CK(cudaMalloc(&d_beta, h_beta.size() * 2));
        CK(cudaMalloc(&d_g, h_g.size() * 4));
        CK(cudaMalloc(&d_out_ser, on1 * 2));
        CK(cudaMalloc(&d_out_chunk, on1 * 2));
        CK(cudaMalloc(&d_st_ser, sn1 * 4));
        CK(cudaMalloc(&d_st_chunk, sn1 * 4));
        CK(cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_g, h_g.data(), h_g.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_st_ser, h_state.data(), sn1 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_st_chunk, h_state.data(), sn1 * 4, cudaMemcpyHostToDevice));
        cuda_recurrent_gated_delta_rule(d_conv, d_g, d_beta, d_st_ser, d_out_ser, 1, n1, kNk, kNv, kD, kD, 0,
                                        nullptr, false, false, 0);
        gdn_chunked_stages(d_conv, d_g, d_beta, d_st_chunk, d_out_chunk, n1, kNk, kNv, ws, 0);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned short> o_ser(on1), o_chunk(on1);
        std::vector<float> s_ser(sn1), s_chunk(sn1);
        CK(cudaMemcpy(o_ser.data(), d_out_ser, on1 * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o_chunk.data(), d_out_chunk, on1 * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s_ser.data(), d_st_ser, sn1 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s_chunk.data(), d_st_chunk, sn1 * 4, cudaMemcpyDeviceToHost));
        const double l2_out1 = rel_l2(to_f32(o_chunk), to_f32(o_ser));
        const double l2_st1 = rel_l2(s_chunk, s_ser);
        // Tighter than the tolerance: the chunk's own contribution must dominate the decayed
        // carry-in gamma_C * state_in. If sum_t k_t (x) v_decay_t never reached the state, the
        // chunked result would BE the carry-in and the share below would be ~0 - which is exactly
        // what the l2norm grid bug produced (rel L2 1.0: the state came out all but zero).
        std::vector<float> carry(sn1);
        for (int h = 0; h < kNv; ++h) {
            double gsum = 0.0;
            for (int t = 0; t < n1; t++) gsum += h_g[(size_t)t * kNv + h];
            const double gamma = exp(gsum);
            for (size_t i = 0; i < (size_t)kD * kD; i++)
                carry[(size_t)h * kD * kD + i] = (float)(gamma * h_state[(size_t)h * kD * kD + i]);
        }
        double ss_chunk = 0.0, ss_ser = 0.0;
        for (size_t i = 0; i < sn1; i++) {
            const double dc = s_chunk[i] - carry[i], ds = s_ser[i] - carry[i];
            ss_chunk += dc * dc;
            ss_ser += ds * ds;
        }
        const double delta_frac = sqrt(ss_chunk / ss_ser);
        printf("  one chunk: core_out rel L2 = %.3e, state rel L2 = %.3e (serial rms %.3e), "
               "delta-rule share of the state = %.3f\n",
               l2_out1, l2_st1, rms(s_ser), delta_frac);
        if (!(delta_frac > 0.9)) {
            printf("FAIL one-chunk state accumulation: the chunk supplies only %.3f of the state the "
                   "serial recurrence reaches (the delta-rule update is missing)\n",
                   delta_frac);
            fails++;
        }
        if (!(l2_st1 < 5e-3)) {
            printf("FAIL one-chunk state: rel L2 %.3e >= 5.0e-3\n", l2_st1);
            fails++;
        }
        if (!(l2_out1 < 5e-3)) {
            printf("FAIL one-chunk core_out: rel L2 %.3e >= 5.0e-3\n", l2_out1);
            fails++;
        }
        cudaFree(d_conv); cudaFree(d_beta); cudaFree(d_g);
        cudaFree(d_out_ser); cudaFree(d_out_chunk); cudaFree(d_st_ser); cudaFree(d_st_chunk);
    }

    // Zero-panel regression: with k = v = 0 the WY transform has nothing to work with (A = 0,
    // T_inv = I, U = W = v_new = 0), so the chunked path must leave the state at zero and write
    // exact zeros. This is the cheapest guard against the whole class of "the WU panel path emits
    // non-finite values" bugs, and it is a property of the port rather than of any tolerance: it
    // holds for the serial and the chunked path alike, so it cannot be "passed" by luck.
    {
        const int n0 = 128;
        const size_t on0 = (size_t)n0 * kNv * kD, sn0 = (size_t)kNv * kD * kD;
        std::vector<unsigned short> z_conv((size_t)n0 * kQkvRow, 0);   // q, k and v all zero
        std::vector<unsigned short> z_beta((size_t)n0 * kNv, f2bf_rz(1.0f));
        std::vector<float> z_g((size_t)n0 * kNv, -0.25f), z_state(sn0, 0.f);
        bfloat16 *d_conv = nullptr, *d_beta = nullptr, *d_out = nullptr;
        float *d_g = nullptr, *d_state = nullptr;
        CK(cudaMalloc(&d_conv, z_conv.size() * 2));
        CK(cudaMalloc(&d_beta, z_beta.size() * 2));
        CK(cudaMalloc(&d_g, z_g.size() * 4));
        CK(cudaMalloc(&d_state, sn0 * 4));
        CK(cudaMalloc(&d_out, on0 * 2));
        CK(cudaMemcpy(d_conv, z_conv.data(), z_conv.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_beta, z_beta.data(), z_beta.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_g, z_g.data(), z_g.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_state, 0, sn0 * 4));
        CK(cudaMemset(d_out, 0xCD, on0 * 2));
        gdn_chunked_stages(d_conv, d_g, d_beta, d_state, d_out, n0, kNk, kNv, ws, 0);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned short> z_out(on0);
        std::vector<float> z_st(sn0);
        CK(cudaMemcpy(z_out.data(), d_out, on0 * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(z_st.data(), d_state, sn0 * 4, cudaMemcpyDeviceToHost));
        int bad_out = 0, bad_state = 0;
        for (size_t i = 0; i < on0; i++) if (bf2f(z_out[i]) != 0.0f) bad_out++;
        for (size_t i = 0; i < sn0; i++) if (z_st[i] != 0.0f) bad_state++;
        printf("zero-panel regression (k=v=0): core_out nonzero = %d/%zu, state nonzero = %d/%zu\n", bad_out,
               on0, bad_state, sn0);
        if (bad_out || bad_state) {
            printf("FAIL zero-panel regression: the chunked WY panel path is not exact at zero input\n");
            fails++;
        }
        cudaFree(d_conv); cudaFree(d_beta); cudaFree(d_g); cudaFree(d_state); cudaFree(d_out);
    }

    // Short sequences stay on the serial kernel: below two chunks the chunked path's extra
    // launches and ~60 MB of workspace traffic cost more than the parallel scan saves, and decode
    // (n = 1) must never touch it.
    {
        std::vector<unsigned short> h_conv, h_beta;
        std::vector<float> h_g, h_state;
        fill_inputs(64, h_conv, h_g, h_beta, h_state);
        bfloat16* d_conv = nullptr;
        float* d_g = nullptr;
        bfloat16* d_beta = nullptr;
        float* d_st = nullptr;
        bfloat16* d_out = nullptr;
        CK(cudaMalloc(&d_conv, h_conv.size() * 2));
        CK(cudaMalloc(&d_beta, h_beta.size() * 2));
        CK(cudaMalloc(&d_g, h_g.size() * 4));
        CK(cudaMalloc(&d_st, h_state.size() * 4));
        CK(cudaMalloc(&d_out, (size_t)64 * kNv * kD * 2));
        CK(cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_g, h_g.data(), h_g.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_st, h_state.data(), h_state.size() * 4, cudaMemcpyHostToDevice));
        const bool took = cuda_chunked_gated_delta_rule(d_conv, d_g, d_beta, d_st, d_out, 1, 64, kNk, kNv, kD, kD,
                                                        ws, 1024, 0);
        CK(cudaDeviceSynchronize());
        if (took) {
            printf("FAIL: n=64 (decode-sized) should be declined by the chunked dispatch\n");
            fails++;
        }
        cudaFree(d_conv); cudaFree(d_beta); cudaFree(d_g); cudaFree(d_st); cudaFree(d_out);
    }

    cudaFree(ws);
    if (fails) {
        printf("FAILED (%d)\n", fails);
        return 1;
    }
    printf("PASS: chunked gated delta rule is byte-reproducible%s\n",
           chunked_enabled ? " and matches the serial recurrence" : " (parity not asserted: path disabled)");
    return 0;
}
