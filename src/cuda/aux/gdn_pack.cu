// GDN input packing for Qwen3.8-Flash-Next. See gdn_pack.cuh for the layouts and why this is done at
// runtime rather than by permuting the quantized weights at load.
#include "gdn_pack.cuh"

namespace helios { namespace aux {

namespace {

// One block per token; the per-token payload is 16384 + 96 floats, so a block of 256 strides it.
__global__ void gdn_pack_kernel(const float* __restrict__ qkv, const float* __restrict__ z,
                                const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ qkvz, float* __restrict__ ba,
                                int Nk, int Ng, int Hk, int Hv) {
  const int s = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int Fseg = 2 * Hk + 2 * Ng * Hv;
  const int Fba = 2 * Ng;

  const float* qkv_s = qkv + (size_t)s * (2 * Nk * Hk + Nk * Ng * Hv);
  const float* z_s = z + (size_t)s * (Nk * Ng * Hv);
  float* out = qkvz + (size_t)s * Nk * Fseg;

  // q and k: k-head kh takes q[kh*Hk ..] and k[Nk*Hk + kh*Hk ..]
  for (int i = tid; i < Nk * Hk; i += nthreads) {
    const int kh = i / Hk, t = i % Hk;
    out[kh * Fseg + t] = qkv_s[i];                            // q
    out[kh * Fseg + Hk + t] = qkv_s[Nk * Hk + i];             // k
  }
  // v and z: v-head kh*Ng+g, both Hv wide
  for (int i = tid; i < Nk * Ng * Hv; i += nthreads) {
    const int vh = i / Hv, t = i % Hv;
    const int kh = vh / Ng, g = vh % Ng;
    out[kh * Fseg + 2 * Hk + g * Hv + t] = qkv_s[2 * Nk * Hk + i];        // v
    out[kh * Fseg + 2 * Hk + Ng * Hv + g * Hv + t] = z_s[i];              // z
  }
  // b then a per k-head
  if (tid < Nk * Ng) {
    const int kh = tid / Ng, g = tid % Ng;
    ba[(size_t)s * Nk * Fba + kh * Fba + g] = b[(size_t)s * Nk * Ng + tid];
    ba[(size_t)s * Nk * Fba + kh * Fba + Ng + g] = a[(size_t)s * Nk * Ng + tid];
  }
}

}  // namespace

void gdn_pack(const float* qkv, const float* z, const float* a, const float* b, float* qkvz,
              float* ba, int S, int Nk, int Ng, int Hk, int Hv, Stream s) {
  if (S <= 0) return;
  gdn_pack_kernel<<<S, 256, 0, s>>>(qkv, z, a, b, qkvz, ba, Nk, Ng, Hk, Hv);
}

}} // namespace helios::aux
