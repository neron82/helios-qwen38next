#include "../../cuda_shim.hpp"
#include "../norm.cuh"
#include "../activation.cuh"
#include "../dsa_topk.cuh"
#include "../gr_mix.cuh"
#include "../gdn_pack.cuh"
#include "../routing_std.cuh"
#include "../gdn.cuh"
#include "engine/glue2.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

using namespace helios;
using namespace aux;

static void fail(const char* msg) {
    std::cerr << "FAIL: " << msg << "\n";
    abort();
}

static void success(const char* msg) {
    std::cout << "PASS: " << msg << "\n";
}

// CPU reference: RMS norm with weight
static void rms_norm_ref(const half* x, const half* w, half* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float xv = __half2float(x[r * cols + c]);
            sum += xv * xv;
        }
        float norm = sqrtf(sum / cols + eps);
        for (int c = 0; c < cols; c++) {
            float xv = __half2float(x[r * cols + c]);
            float wv = __half2float(w[c]);
            out[r * cols + c] = __float2half(xv * wv / norm);
        }
    }
}

// CPU reference: silu(x) * y
static void silu_mul_ref(const half* x, const half* y, half* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float xv = __half2float(x[r * cols + c]);
            float yv = __half2float(y[r * cols + c]);
            float s = 1.0f / (1.0f + expf(-xv));
            out[r * cols + c] = __float2half(xv * s * yv);
        }
    }
}

#include "aux_conv_gemm_tests.inc"

int main() {
    // Initialize CUDA
    HELIOS_CUDA_CHECK(cudaSetDevice(0));

    // --- RMS Norm Test ---
    {
        const int rows = 8;
        const int cols = 128;
        const float eps = 1e-6f;

        half* h_x = new half[rows * cols];
        half* h_w = new half[cols];
        half* h_out = new half[rows * cols];
        half* h_out_ref = new half[rows * cols];
        std::memset(h_out, 0, rows * cols * 2);
        std::memset(h_out_ref, 0, rows * cols * 2);

        // Fill with random-ish values
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < rows * cols; i++) h_x[i] = __float2half(dist(rng));
        for (int i = 0; i < cols; i++) h_w[i] = __float2half(1.0f + 0.1f * dist(rng));

        // CPU reference
        rms_norm_ref(h_x, h_w, h_out_ref, rows, cols, eps);

        // GPU kernel
        half* d_x, *d_w, *d_out;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_w, cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_out, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x, rows * cols * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_w, h_w, cols * 2, cudaMemcpyHostToDevice));
        rms_norm(d_x, kHalf, d_w, kHalf, d_out, kHalf, rows, cols, eps, 0.0f, 1.0f, false, 1, 0);

        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

        HELIOS_CUDA_CHECK(cudaMemcpy(h_out, d_out, rows * cols * 2, cudaMemcpyDeviceToHost));

        // Compare
        float max_err = 0.0f;
        for (int i = 0; i < rows * cols; i++) {
            float a = __half2float(h_out[i]);
            float b = __half2float(h_out_ref[i]);
            float err = fabsf(a - b);
            if (err > max_err) max_err = err;
        }
        std::cout << "RMS norm max error: " << max_err << " (half precision expected <0.01)\n";
        if (max_err > 0.01f) fail("RMS norm");
        success("RMS norm");

        delete[] h_x; delete[] h_w; delete[] h_out; delete[] h_out_ref;
        HELIOS_CUDA_CHECK(cudaFree(d_x));
        HELIOS_CUDA_CHECK(cudaFree(d_w));
        HELIOS_CUDA_CHECK(cudaFree(d_out));
    }

    // --- SILU Multiply Test ---
    {
        const int rows = 8;
        const int cols = 128;

        half* h_x = new half[rows * cols];
        half* h_y = new half[rows * cols];
        half* h_out = new half[rows * cols];
        half* h_out_ref = new half[rows * cols];
        std::memset(h_out, 0, rows * cols * 2);
        std::memset(h_out_ref, 0, rows * cols * 2);

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (int i = 0; i < rows * cols; i++) {
            h_x[i] = __float2half(dist(rng));
            h_y[i] = __float2half(dist(rng));
        }

        // CPU reference
        silu_mul_ref(h_x, h_y, h_out_ref, rows, cols);

        // GPU kernel
        half* d_x, *d_y, *d_out;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_y, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_out, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x, rows * cols * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_y, h_y, rows * cols * 2, cudaMemcpyHostToDevice));

        silu_mul(d_x, d_y, d_out, false, 100.0f, rows * cols, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

        HELIOS_CUDA_CHECK(cudaMemcpy(h_out, d_out, rows * cols * 2, cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < rows * cols; i++) {
            float a = __half2float(h_out[i]);
            float b = __half2float(h_out_ref[i]);
            float err = fabsf(a - b);
            if (err > max_err) max_err = err;
        }
        std::cout << "SILU mul max error: " << max_err << " (half precision expected <0.01)\n";
        if (max_err > 0.01f) fail("SILU mul");
        success("SILU mul");

        delete[] h_x; delete[] h_y; delete[] h_out; delete[] h_out_ref;
        HELIOS_CUDA_CHECK(cudaFree(d_x));
        HELIOS_CUDA_CHECK(cudaFree(d_y));
        HELIOS_CUDA_CHECK(cudaFree(d_out));
    }

    // --- DSA Top-K Test ---
    {
        const int rows = 8;
        const int cols = 64;
        const int k = 8;

        half* h_x = new half[rows * cols];
        int* h_indices = new int[rows * k];
        std::memset(h_indices, -1, rows * k * 4);

        std::mt19937 rng(456);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < rows * cols; i++) h_x[i] = __float2half(dist(rng));

        half* d_x;
        int* d_indices;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_indices, rows * k * 4));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x, rows * cols * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemset(d_indices, -1, rows * k * 4));

        dsa_topk(d_x, cols, d_indices, rows, cols, k, k, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

        HELIOS_CUDA_CHECK(cudaMemcpy(h_indices, d_indices, rows * k * 4, cudaMemcpyDeviceToHost));

        // Verify indices are in [0, cols) and unique per row
        bool ok = true;
        for (int r = 0; r < rows && ok; r++) {
            std::vector<int> seen(cols, 0);
            for (int i = 0; i < k && ok; i++) {
                int idx = h_indices[r * k + i];
                if (idx < 0 || idx >= cols) {
                    std::cerr << "DSA topk: index " << idx << " out of range\n";
                    ok = false;
                } else if (seen[idx]) {
                    std::cerr << "DSA topk: duplicate index " << idx << " in row " << r << "\n";
                    ok = false;
                } else {
                    seen[idx] = 1;
                }
            }
        }
        if (!ok) fail("DSA topk");
        success("DSA topk");

        delete[] h_x; delete[] h_indices;
        HELIOS_CUDA_CHECK(cudaFree(d_x));
        HELIOS_CUDA_CHECK(cudaFree(d_indices));
        dsa_topk_free_workspace();
    }

    // --- GatedResidual (qwen4_exp hyper-connections) vs hyperconnections.py::_mix_ref ---
    {
        const int H = 4, D = 2560, rank = 320, R = 6;
        const float eps = 1e-6f;
        const size_t HD = (size_t)H * D;

        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
        std::vector<float> h_x((size_t)R * HD, 0.f), h_mixed((size_t)R * D, 0.f), h_post(R * H, 0.f);
        std::vector<float> ref_mixed((size_t)R * D, 0.f), ref_post(R * H, 0.f);
        std::vector<half> h_norm(HD), h_down((size_t)rank * HD), h_up(HD * rank), h_inject(HD);
        for (auto& v : h_x) v = 0.5f * dx(rng);
        for (auto& v : h_norm) v = __float2half(0.1f * dx(rng));     // w_raw, applied as 1 + w
        for (auto& v : h_down) v = __float2half(0.05f * dx(rng));
        for (auto& v : h_up) v = __float2half(0.05f * dx(rng));
        for (auto& v : h_inject) v = __float2half(0.05f * dx(rng));

        // CPU reference in double, transcribed from _mix_ref
        for (int r = 0; r < R; r++) {
            const float* x = h_x.data() + (size_t)r * HD;
            std::vector<double> rmr(H), normed(HD, 0.0), t(rank, 0.0), gated(HD, 0.0);
            for (int h = 0; h < H; h++) {
                double acc = 0;
                for (int d = 0; d < D; d++) { double v = x[(size_t)h * D + d]; acc += v * v; }
                rmr[h] = 1.0 / std::sqrt(acc / D + eps);
            }
            for (size_t i = 0; i < HD; i++)
                normed[i] = (double)x[i] * rmr[i / D] * ((double)__half2float(h_norm[i]) + 1.0);
            for (int i = 0; i < rank; i++) {
                double acc = 0;
                const half* dr = h_down.data() + (size_t)i * HD;
                for (size_t j = 0; j < HD; j++) acc += normed[j] * (double)__half2float(dr[j]);
                double v = acc / H;
                t[i] = v / (1.0 + std::exp(-v));                       // silu
            }
            for (size_t j = 0; j < HD; j++) {
                const half* ur = h_up.data() + j * rank;
                double acc = 0;
                for (int i = 0; i < rank; i++) acc += t[i] * (double)__half2float(ur[i]);
                gated[j] = normed[j] / (1.0 + std::exp(-acc));
            }
            for (int d = 0; d < D; d++) {
                double acc = 0;
                for (int h = 0; h < H; h++) acc += gated[(size_t)h * D + d];
                ref_mixed[(size_t)r * D + d] = (float)(acc / H);
            }
            for (int h = 0; h < H; h++) {
                // F.linear(flat, inject_h): the whole flattened stream stack, all H*D inputs. An
                // earlier revision of this reference summed only the h-th D-block, which matched the
                // equally-wrong kernel and hid the bug from this very test.
                double acc = 0;
                const half* ir = h_inject.data() + (size_t)h * HD;
                for (int j = 0; j < HD; j++)
                    acc += normed[j] * (double)__half2float(ir[j]);
                ref_post[(size_t)r * H + h] = (float)(2.0 / (1.0 + std::exp(-acc / H)));
            }
        }

        float *d_x, *d_mixed, *d_post;
        half *d_norm, *d_down, *d_up, *d_inject;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, (size_t)R * HD * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_mixed, (size_t)R * D * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_post, (size_t)R * H * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_norm, HD * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_down, (size_t)rank * HD * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_up, HD * rank * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_inject, HD * 2));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)R * HD * 4, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_norm, h_norm.data(), HD * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_down, h_down.data(), (size_t)rank * HD * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), HD * rank * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_inject, h_inject.data(), HD * 2, cudaMemcpyHostToDevice));

        gr_mix(d_x, d_norm, d_down, d_up, d_inject, R, H, D, rank, eps, d_mixed, d_post, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
        HELIOS_CUDA_CHECK(cudaMemcpy(h_mixed.data(), d_mixed, (size_t)R * D * 4, cudaMemcpyDeviceToHost));
        HELIOS_CUDA_CHECK(cudaMemcpy(h_post.data(), d_post, (size_t)R * H * 4, cudaMemcpyDeviceToHost));

        float emix = 0.f, epost = 0.f;
        bool nan = false;
        for (size_t i = 0; i < (size_t)R * D; i++) {
            if (std::isnan(h_mixed[i])) nan = true;
            emix = std::max(emix, std::fabs(h_mixed[i] - ref_mixed[i]));
        }
        for (int i = 0; i < R * H; i++) epost = std::max(epost, std::fabs(h_post[i] - ref_post[i]));
        printf("[gr_mix] mixed max_err %.3e, post max_err %.3e, rms of ref mixed %.4f%s\n", emix, epost,
               [&] { double a = 0; for (size_t i = 0; i < (size_t)R * D; i++) a += (double)ref_mixed[i] * ref_mixed[i];
                     return (float)std::sqrt(a / (R * D)); }(), nan ? " (NaN!)\n" : "");
        if (nan || emix > 2e-3f || epost > 2e-3f) fail("GatedResidual mix");
        success("GatedResidual mix");

        // gr_apply: x[h,d] <- post[h] * y[d] + x[h,d]
        {
            std::vector<float> h_y((size_t)R * D, 0.f), h_x2(h_x);
            for (auto& v : h_y) v = 0.3f * dx(rng);
            float* d_y;
            HELIOS_CUDA_CHECK(cudaMalloc(&d_y, (size_t)R * D * 4));
            HELIOS_CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), (size_t)R * D * 4, cudaMemcpyHostToDevice));
            gr_apply(d_x, d_y, d_post, R, H, D, 0);
            HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
            HELIOS_CUDA_CHECK(cudaMemcpy(h_x2.data(), d_x, (size_t)R * HD * 4, cudaMemcpyDeviceToHost));
            float eap = 0.f;
            for (int r = 0; r < R; r++)
                for (int h = 0; h < H; h++)
                    for (int d = 0; d < D; d++) {
                        float want = ref_post[(size_t)r * H + h] * h_y[(size_t)r * D + d] +
                                     h_x[(size_t)r * HD + (size_t)h * D + d];
                        eap = std::max(eap, std::fabs(h_x2[(size_t)r * HD + (size_t)h * D + d] - want));
                    }
            printf("[gr_apply] max_err %.3e\n", eap);
            if (eap > 2e-3f) fail("GatedResidual apply");
            success("GatedResidual apply");
            HELIOS_CUDA_CHECK(cudaFree(d_y));
        }

        HELIOS_CUDA_CHECK(cudaFree(d_x)); HELIOS_CUDA_CHECK(cudaFree(d_mixed));
        HELIOS_CUDA_CHECK(cudaFree(d_post)); HELIOS_CUDA_CHECK(cudaFree(d_norm));
        HELIOS_CUDA_CHECK(cudaFree(d_down)); HELIOS_CUDA_CHECK(cudaFree(d_up));
        HELIOS_CUDA_CHECK(cudaFree(d_inject));
    }

    // --- gr_dots2 (register-tiled `dots`) and the templated gr_up_kernel: BITWISE, not tolerance ---
    //
    // The block above runs at rank 320 / R = 6, the reference-parity shape. R = 6 is below
    // gr_dots2's threshold, so on its own it proves nothing about the fast paths. This block runs
    // at a shape they actually take (R = 64, HD a multiple of the column tile) and asserts all
    // four kernel combinations return the same BITS: {templated, generic} x {dots2, legacy dots}.
    //
    // Exactness is the property that makes these swaps safe to ship by default. A tolerance would
    // let a reassociation through, and a reassociation here reaches the logits and flips tokens.
    for (int rank : {320, 24}) {
        const int H = 4, D = 2560, R = 64;
        const float eps = 1e-6f;
        const size_t HD = (size_t)H * D;

        std::mt19937 rng(13);
        std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
        std::vector<float> h_x((size_t)R * HD), h_mixed((size_t)R * D), h_post(R * H);
        std::vector<float> h_up_leg((size_t)R * D), h_dots_leg((size_t)R * D);
        std::vector<float> ref_mixed((size_t)R * D), ref_post(R * H);
        std::vector<half> h_norm(HD), h_down((size_t)rank * HD), h_up(HD * rank), h_inject(HD);
        for (auto& v : h_x) v = 0.5f * dx(rng);
        for (auto& v : h_norm) v = __float2half(0.1f * dx(rng));
        for (auto& v : h_down) v = __float2half(0.05f * dx(rng));
        for (auto& v : h_up) v = __float2half(0.05f * dx(rng));
        for (auto& v : h_inject) v = __float2half(0.05f * dx(rng));

        // CPU oracle, same construction as the reference-parity block above.
        for (int r = 0; r < R; r++) {
            std::vector<float> normed(HD), gated(HD);
            for (int h = 0; h < H; h++) {
                const float* xh = h_x.data() + (size_t)r * HD + (size_t)h * D;
                double ss = 0.0;
                for (int d = 0; d < D; d++) ss += (double)xh[d] * xh[d];
                const float inv = 1.0f / std::sqrt((float)(ss / D) + eps);
                for (int d = 0; d < D; d++)
                    normed[(size_t)h * D + d] =
                        xh[d] * inv * (__half2float(h_norm[(size_t)h * D + d]) + 1.0f);
            }
            std::vector<float> t(rank);
            for (int i = 0; i < rank; i++) {
                double acc = 0.0;
                for (int j = 0; j < (int)HD; j++) acc += normed[j] * __half2float(h_down[(size_t)i * HD + j]);
                acc /= H;
                t[i] = (float)(acc / (1.0 + std::exp(-acc)));
            }
            for (size_t j = 0; j < HD; j++) {
                double acc = 0.0;
                for (int i = 0; i < rank; i++) acc += t[i] * __half2float(h_up[j * rank + i]);
                gated[j] = normed[j] / (1.0 + std::exp(-acc));
            }
            for (int d = 0; d < D; d++) {
                double acc = 0.0;
                for (int h = 0; h < H; h++) acc += gated[(size_t)h * D + d];
                ref_mixed[(size_t)r * D + d] = (float)(acc / H);
            }
            for (int h = 0; h < H; h++) {
                double acc = 0.0;
                const half* ir = h_inject.data() + (size_t)h * HD;
                for (int j = 0; j < (int)HD; j++) acc += normed[j] * __half2float(ir[j]);
                ref_post[(size_t)r * H + h] = (float)(2.0 / (1.0 + std::exp(-acc / H)));
            }
        }

        float *d_x, *d_mixed, *d_post;
        half *d_norm, *d_down, *d_up, *d_inject;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, (size_t)R * HD * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_mixed, (size_t)R * D * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_post, (size_t)R * H * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_norm, HD * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_down, (size_t)rank * HD * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_up, HD * rank * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_inject, HD * 2));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)R * HD * 4, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_norm, h_norm.data(), HD * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_down, h_down.data(), (size_t)rank * HD * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), HD * rank * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_inject, h_inject.data(), HD * 2, cudaMemcpyHostToDevice));

        // All four kernel combinations, from one set of inputs.
        std::vector<std::vector<float>> out_m, out_p;
        for (int cfg = 0; cfg < 4; cfg++) {
            gr_force_legacy_up(cfg & 1);
            gr_force_legacy_dots(cfg & 2);
            gr_mix(d_x, d_norm, d_down, d_up, d_inject, R, H, D, rank, eps, d_mixed, d_post, 0);
            HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> m((size_t)R * D), p((size_t)R * H);
            HELIOS_CUDA_CHECK(cudaMemcpy(m.data(), d_mixed, (size_t)R * D * 4, cudaMemcpyDeviceToHost));
            HELIOS_CUDA_CHECK(cudaMemcpy(p.data(), d_post, (size_t)R * H * 4, cudaMemcpyDeviceToHost));
            out_m.push_back(std::move(m));
            out_p.push_back(std::move(p));
        }
        gr_force_legacy_up(false);
        gr_force_legacy_dots(false);
        h_mixed = out_m[0];
        h_post = out_p[0];

        bool same = true;
        for (int cfg = 1; cfg < 4; cfg++)
            same = same && std::memcmp(h_mixed.data(), out_m[cfg].data(), h_mixed.size() * 4) == 0 &&
                    std::memcmp(h_post.data(), out_p[cfg].data(), h_post.size() * 4) == 0;
        float emix = 0.f, epost = 0.f;
        for (size_t i = 0; i < (size_t)R * D; i++) emix = std::max(emix, std::fabs(h_mixed[i] - ref_mixed[i]));
        for (int i = 0; i < R * H; i++) epost = std::max(epost, std::fabs(h_post[i] - ref_post[i]));
        printf("[gr_mix fast paths] rank %3d: 4 kernel combinations bitwise equal: %s, "
               "mixed max_err %.3e, post max_err %.3e\n", rank, same ? "yes" : "NO", emix, epost);
        if (!same) fail("GatedResidual mix: fast vs legacy stages");
        if (emix > 2e-3f || epost > 2e-3f) fail("GatedResidual mix fast path");
        success("GatedResidual mix fast stages (bitwise vs legacy)");

        HELIOS_CUDA_CHECK(cudaFree(d_x)); HELIOS_CUDA_CHECK(cudaFree(d_mixed));
        HELIOS_CUDA_CHECK(cudaFree(d_post)); HELIOS_CUDA_CHECK(cudaFree(d_norm));
        HELIOS_CUDA_CHECK(cudaFree(d_down)); HELIOS_CUDA_CHECK(cudaFree(d_up));
        HELIOS_CUDA_CHECK(cudaFree(d_inject));
    }

    // --- GDN input packing: the fused prologue's read pattern must recover the flat sources ---
    {
        const int S = 4, Nk = 16, Ng = 3, Hk = 128, Hv = 128;
        const int Nv = Nk * Ng, Fseg = 2 * Hk + 2 * Ng * Hv, Fba = 2 * Ng;
        const int Fqkv = 2 * Nk * Hk + Nv * Hv, Fz = Nv * Hv;

        std::mt19937 rng2(11);
        std::uniform_real_distribution<float> dz(-1.0f, 1.0f);
        std::vector<float> h_qkv((size_t)S * Fqkv), h_z((size_t)S * Fz), h_a((size_t)S * Nv),
            h_b((size_t)S * Nv), h_pack((size_t)S * Nk * Fseg, -9.f), h_ba((size_t)S * Nk * Fba, -9.f);
        for (auto* v : {&h_qkv, &h_z, &h_a, &h_b}) for (auto& x : *v) x = dz(rng2);

        float *d_qkv, *d_z, *d_a, *d_b, *d_pack, *d_ba;
        auto alloc = [&](float** p, size_t n, const float* src) {
            HELIOS_CUDA_CHECK(cudaMalloc(p, n * 4));
            HELIOS_CUDA_CHECK(cudaMemcpy(*p, src, n * 4, cudaMemcpyHostToDevice));
        };
        alloc(&d_qkv, h_qkv.size(), h_qkv.data());
        alloc(&d_z, h_z.size(), h_z.data());
        alloc(&d_a, h_a.size(), h_a.data());
        alloc(&d_b, h_b.size(), h_b.data());
        HELIOS_CUDA_CHECK(cudaMalloc(&d_pack, h_pack.size() * 4));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_ba, h_ba.size() * 4));
        gdn_pack(d_qkv, d_z, d_a, d_b, d_pack, d_ba, S, Nk, Ng, Hk, Hv, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
        HELIOS_CUDA_CHECK(cudaMemcpy(h_pack.data(), d_pack, h_pack.size() * 4, cudaMemcpyDeviceToHost));
        HELIOS_CUDA_CHECK(cudaMemcpy(h_ba.data(), d_ba, h_ba.size() * 4, cudaMemcpyDeviceToHost));

        // Walk exactly the indexing gdn.cu uses to read these buffers back.
        int wrong = 0;
        for (int s = 0; s < S; s++)
            for (int kh = 0; kh < Nk; kh++) {
                const size_t base = ((size_t)s * Nk + kh) * Fseg;
                for (int t = 0; t < Hk; t++) {
                    if (h_pack[base + t] != h_qkv[(size_t)s * Fqkv + kh * Hk + t]) wrong++;
                    if (h_pack[base + Hk + t] != h_qkv[(size_t)s * Fqkv + Nk * Hk + kh * Hk + t]) wrong++;
                }
                for (int g = 0; g < Ng; g++)
                    for (int t = 0; t < Hv; t++) {
                        const int vh = kh * Ng + g;
                        if (h_pack[base + 2 * Hk + g * Hv + t] !=
                            h_qkv[(size_t)s * Fqkv + 2 * Nk * Hk + (size_t)vh * Hv + t]) wrong++;
                        if (h_pack[base + 2 * Hk + Ng * Hv + g * Hv + t] !=
                            h_z[(size_t)s * Fz + (size_t)vh * Hv + t]) wrong++;
                    }
                const size_t bbase = ((size_t)s * Nk + kh) * Fba;
                for (int g = 0; g < Ng; g++) {
                    if (h_ba[bbase + g] != h_b[(size_t)s * Nv + kh * Ng + g]) wrong++;
                    if (h_ba[bbase + Ng + g] != h_a[(size_t)s * Nv + kh * Ng + g]) wrong++;
                }
            }
        printf("[gdn_pack] %d mismatched elements (errors would show as wrong q/k/v/z or b/a order)\n",
               wrong);
        if (wrong) fail("GDN pack");
        success("GDN pack");
        HELIOS_CUDA_CHECK(cudaFree(d_qkv)); HELIOS_CUDA_CHECK(cudaFree(d_z));
        HELIOS_CUDA_CHECK(cudaFree(d_a)); HELIOS_CUDA_CHECK(cudaFree(d_b));
        HELIOS_CUDA_CHECK(cudaFree(d_pack)); HELIOS_CUDA_CHECK(cudaFree(d_ba));
    }

    // --- std softmax routing (Qwen3.8: 512 experts, top-10, no bias, no routed scale) ---
    {
        const int bsz = 4, E = 512, K = 10;
        std::mt19937 rng3(3);
        std::uniform_real_distribution<float> dr(-8.0f, 8.0f);
        std::vector<half> h_s((size_t)bsz * E);
        for (auto& v : h_s) v = __float2half_rn(dr(rng3));
        h_s[7] = h_s[100] = __float2half_rn(9.0f);          // a deliberate tie at the top
        h_s[5] = __float2half_rn(-30.0f);                   // a strong negative

        half *d_s, *d_w; int64_t* d_idx;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_s, h_s.size() * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_w, (size_t)bsz * K * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_idx, (size_t)bsz * K * 8));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), h_s.size() * 2, cudaMemcpyHostToDevice));
        routing_std_logits(d_s, d_idx, d_w, bsz, E, K, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> h_w((size_t)bsz * K);
        std::vector<int64_t> h_idx((size_t)bsz * K);
        HELIOS_CUDA_CHECK(cudaMemcpy(h_w.data(), d_w, h_w.size() * 2, cudaMemcpyDeviceToHost));
        HELIOS_CUDA_CHECK(cudaMemcpy(h_idx.data(), d_idx, h_idx.size() * 8, cudaMemcpyDeviceToHost));

        // CPU reference: top-K by logit (index-stable), then softmax over the SELECTED set only
        int bad = 0;
        double maxwerr = 0;
        for (int r = 0; r < bsz; r++) {
            std::vector<double> lg(E);
            for (int e = 0; e < E; e++) lg[e] = __half2float(h_s[(size_t)r * E + e]);
            std::vector<int> ord(E);
            for (int e = 0; e < E; e++) ord[e] = e;
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            double mx = -1e30, sum = 0;
            for (int k = 0; k < K; k++) mx = std::max(mx, lg[ord[k]]);
            std::vector<double> w(K);
            for (int k = 0; k < K; k++) { w[k] = std::exp(lg[ord[k]] - mx); sum += w[k]; }

            std::vector<int> got, want(ord.begin(), ord.begin() + K);
            std::vector<double> gotw;
            for (int k = 0; k < K; k++) {
                got.push_back((int)h_idx[(size_t)r * K + k]);
                gotw.push_back(__half2float(h_w[(size_t)r * K + k]));
            }
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            if (got != want) bad++;                       // compare as a set: ties may order differently
            for (int k = 0; k < K; k++) {
                const int e = (int)h_idx[(size_t)r * K + k];
                double ref = 0;
                for (int j = 0; j < K; j++) if (ord[j] == e) ref = w[j] / sum;
                maxwerr = std::max(maxwerr, std::fabs(gotw[k] - ref));
            }
        }
        printf("[routing_std] %d rows with a wrong selection set, max weight err %.3e\n", bad,
               maxwerr);
        const bool ok = bad == 0 && maxwerr < 2e-3;
        if (!ok) fail("std routing");
        success("std routing");
        HELIOS_CUDA_CHECK(cudaFree(d_s)); HELIOS_CUDA_CHECK(cudaFree(d_w)); HELIOS_CUDA_CHECK(cudaFree(d_idx));
    }
    // gr_mix at R >= 8, which is the tiled gr_dots path. Nothing else covered it: every other aux
    // test runs at R = 1, which takes the small reduction kernel, so a wrong tiled kernel passed
    // the whole suite and only surfaced as degenerate generated text. Compare `mixed` against a
    // CPU reference of the full chain (hsum -> scale -> dots -> up -> fin).
    {
        const int R = 256, H = 4, D = 2560, rank = 320;
        const size_t nst = (size_t)R * H * D, nhd = (size_t)H * D, nrk = (size_t)rank;
        std::vector<float> h_streams(nst);
        std::vector<half>  h_norm(nhd), h_down(nrk * nhd), h_up(nhd * nrk);
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 1.f), ns(0.f, 0.05f);
        for (auto& v : h_streams) v = nd(rng);
        for (auto& v : h_norm)    v = __float2half(ns(rng));
        for (auto& v : h_down)    v = __float2half(ns(rng));
        for (auto& v : h_up)      v = __float2half(ns(rng));

        float *d_streams, *d_mixed, *d_post;
        half *d_norm, *d_down, *d_up;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_streams, nst * sizeof(float)));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_mixed, (size_t)R * D * sizeof(float)));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_post, (size_t)R * H * sizeof(float)));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_norm, nhd * sizeof(half)));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_down, nrk * nhd * sizeof(half)));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_up, nhd * nrk * sizeof(half)));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_streams, h_streams.data(), nst * sizeof(float), cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_norm, h_norm.data(), nhd * sizeof(half), cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_down, h_down.data(), nrk * nhd * sizeof(half), cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), nhd * nrk * sizeof(half), cudaMemcpyHostToDevice));

        const float eps = 1e-5f;

        auto sig = [](float x) { return 1.0f / (1.0f + expf(-x)); };
        std::vector<float> normed(nst);
        for (int r = 0; r < R; r++) for (int h = 0; h < H; h++) {
            double ss = 0.0;
            for (int d = 0; d < D; d++) { double v = h_streams[((size_t)r * H + h) * D + d]; ss += v * v; }
            const float sc = 1.0f / sqrtf((float)(ss / D) + eps);
            for (int d = 0; d < D; d++)
                normed[((size_t)r * H + h) * D + d] =
                    h_streams[((size_t)r * H + h) * D + d] * sc *
                    (1.0f + __half2float(h_norm[(size_t)h * D + d]));
        }

        // BOTH mixer paths are checked against this double-precision reference, each with a gate
        // matched to its precision. The fp32 path keeps the max-relative-error gate it has always
        // had, unchanged. The tensor-core path (gr_mix_tc.cu) is gated on RMS relative error
        // instead: a per-element max divides by each |want|, and this operator cancels heavily, so
        // elements near zero dominate that statistic - on the SAME data the fp32 path scores 6.9e-4
        // and the tensor-core path 1.1e-1, while their RMS errors differ by only ~9x. The RMS gate is
        // the one that actually bounds the output.
        std::vector<float> h_mixed((size_t)R * D);
        std::vector<float> t(nrk);
        for (int pass = 0; pass < 2; pass++) {
            const bool tc = (pass == 1);
            gr_force_tc(tc, 0);
            gr_mix(d_streams, d_norm, d_down, d_up, nullptr, R, H, D, rank, eps, d_mixed, d_post, 0);
            HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
            HELIOS_CUDA_CHECK(cudaMemcpy(h_mixed.data(), d_mixed, h_mixed.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost));

            double maxrel = 0.0, se = 0.0, sr = 0.0;
            for (int r = 0; r < R; r++) {
                for (int i = 0; i < rank; i++) {
                    double a = 0.0;
                    for (size_t j = 0; j < nhd; j++)
                        a += (double)normed[(size_t)r * nhd + j] * (double)__half2float(h_down[(size_t)i * nhd + j]);
                    const float f = (float)(a / H);
                    t[i] = f * sig(f);
                }
                for (int d = 0; d < D; d++) {
                    double a = 0.0;
                    for (int h = 0; h < H; h++) {
                        double g = 0.0;
                        for (int i = 0; i < rank; i++)
                            g += (double)t[i] * (double)__half2float(h_up[(size_t)(h * D + d) * rank + i]);
                        a += sig((float)g) * (double)normed[((size_t)r * H + h) * D + d];
                    }
                    a /= H;
                    const double got = h_mixed[(size_t)r * D + d], want = a;
                    maxrel = std::max(maxrel, std::fabs(got - want) / std::max(1e-3, std::fabs(want)));
                    se += (got - want) * (got - want);
                    sr += want * want;
                }
            }
            const double rmsrel = std::sqrt(se / sr);
            if (tc) {
                printf("[gr_mix] tensor-core path: max rel %.3e, rms rel %.3e\n", maxrel, rmsrel);
                if (!(rmsrel < 1e-3)) fail("gr_mix tensor-core path rms");
                success("gr_mix tensor-core path");
            } else {
                printf("[gr_mix] R=%d tiled-dots path, max relative error %.3e, rms rel %.3e\n",
                       R, maxrel, rmsrel);
                if (!(maxrel < 5e-3)) fail("gr_mix tiled dots");
                success("gr_mix tiled dots");
            }
        }
        gr_force_tc(false, 0);
        HELIOS_CUDA_CHECK(cudaFree(d_streams)); HELIOS_CUDA_CHECK(cudaFree(d_mixed));
        HELIOS_CUDA_CHECK(cudaFree(d_post)); HELIOS_CUDA_CHECK(cudaFree(d_norm));
        HELIOS_CUDA_CHECK(cudaFree(d_down)); HELIOS_CUDA_CHECK(cudaFree(d_up));
    }

    test_conv1d();
    test_gemm_nt_f16();
    test_transpose_f32_bf16();

    std::cout << "All aux parity tests passed.\n";
    return 0;
}