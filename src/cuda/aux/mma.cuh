#pragma once
// Tensor-core / vector-memory primitives for the chunked gated-delta-rule kernels.
//
// Ported from ninfer-3090 src/ops/common/{memory,math,warp,mma}.cuh. helios' existing
// src/cuda/quant/ptx.cuh covers the fp16 m16n8k16 mma, cp.async and ldmatrix.x4, but it lives in
// a different directory under no namespace, and the chunked path additionally needs the bf16
// m16n8k16 mma, the tf32 m16n8k8 mma (in both its float and raw-bit forms), ldmatrix.x2 / x2.trans,
// ex2.approx and the load/store_vec family. Re-declaring just that set here keeps one primitive
// per name, in one namespace, instead of forking helios' existing global-scope helpers.
//
// Everything here is either an inline PTX wrapper or a trivially inlined template. There is no
// per-element dynamic work and no address arithmetic beyond what the PTX operands need, so these
// are free at runtime.
//
// Determinism: none of these use atomics or warp-vote-dependent ordering. The shuffle reductions
// in warp.cuh use a fixed __shfl_down tree, so a given input yields bit-identical output run to
// run (the property the engine's reproducibility guarantee rests on).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace helios { namespace ptx {

inline constexpr int kWarpSize = 32;
inline constexpr unsigned kFullWarpMask = 0xffffffffu;

// log2(e): the chunked kernels fold g (a natural-log decay) into exp2, so every g value is
// pre-scaled by this exactly once instead of paying for expf.
inline constexpr float kLog2E = 1.4426950408889634f;

enum class Cache { ca, cg };

// ---------------------------------------------------------------------------
// Vector memory (ops/common/memory.cuh)
// ---------------------------------------------------------------------------

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr)
{
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr)
{
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value)
{
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src)
{
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

// Wait until at most Groups async groups are still in flight. Groups is a compile-time immediate,
// so the wait is a single instruction with no runtime bookkeeping.
template <int Groups>
__device__ __forceinline__ void cp_wait()
{
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

// ---------------------------------------------------------------------------
// LDSM (ops/common/mma.cuh)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// ---------------------------------------------------------------------------
// MMA (ops/common/mma.cuh)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// TF32 mma taking raw bit patterns. The state_passing kernel uses it to feed operands straight out
// of ldmatrix, where a bf16x2 pair widened to two fp32 values and then reinterpreted as tf32 bits
// costs nothing but a shift/mask (bf16 -> tf32 is exact: both share the 8-bit exponent).
__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                              unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0, float a1,
                                         float a2, float a3, float b0, float b1)
{
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

// ---------------------------------------------------------------------------
// Math (ops/common/math.cuh)
// ---------------------------------------------------------------------------

// ex2.approx.f32: the hardware exp2 with ~1 ulp. The serial kernel uses __expf (also approximate),
// so the two paths carry the same relative error and neither is the bottleneck on parity.
__device__ __forceinline__ float exp2_approx(float x)
{
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float2 bf16x2_to_float2(__nv_bfloat162 value) { return __bfloat1622float2(value); }

// Round-toward-zero bf16x2 pack. helios writes every bf16 output with __float2bfloat16_rz (gdn.cu
// for core_out, norm.cuh for the norm weights), so the chunked path truncates too: it keeps the
// two implementations' rounding boundaries identical instead of differing by up to 1 bf16 ulp
// (4e-3 relative) purely on the rounding mode. CUDA has no packed _rz intrinsic, hence the pair.
__device__ __forceinline__ __nv_bfloat162 pack_bf16x2_rz(float lo, float hi)
{
    return __halves2bfloat162(__float2bfloat16_rz(lo), __float2bfloat16_rz(hi));
}

// Round-to-nearest-even bf16x2 pack, for the chunked path's INTERNAL workspace panels (q_norm /
// k_norm, W, U, v_new, h_chunk). Those are consumed by later stages, never compared against the
// serial kernel value by value, so the rounding mode is free - and round-to-zero is the wrong
// choice for them: it biases every stored value low by half an ulp (2^-9 = 2e-3 relative,
// one-signed), which the cancellation in T_inv @ (beta*V) and in the state accumulation turned
// into a systematic -0.6% state / -1% output error against the serial recurrence. core_attn_out
// keeps pack_bf16x2_rz: that one IS compared value-by-value with the serial kernel, which stores
// it with __float2bfloat16_rz.
__device__ __forceinline__ __nv_bfloat162 pack_bf16x2_rn(float lo, float hi)
{
    return __halves2bfloat162(__float2bfloat16(lo), __float2bfloat16(hi));
}

// ---------------------------------------------------------------------------
// Warp reductions (ops/common/warp.cuh)
// ---------------------------------------------------------------------------

// Fixed __shfl_down tree: the summation order depends only on the lane index, never on scheduling,
// so the result is bit-identical across launches.
template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_reduce_sum(T x, unsigned mask = kFullWarpMask)
{
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) { x += __shfl_down_sync(mask, x, offset, Width); }
    return x;
}

// ---------------------------------------------------------------------------
// BF16 vector packs (ops/common/bf16_vector.cuh)
// ---------------------------------------------------------------------------

template <int Pairs>
struct alignas(Pairs* static_cast<int>(sizeof(__nv_bfloat162))) Bf16PairPack
{
    static_assert(Pairs == 1 || Pairs == 2 || Pairs == 4);
    __nv_bfloat162 pair[Pairs];
};

using Bf16x4Pack = Bf16PairPack<2>;
using Bf16x8Pack = Bf16PairPack<4>;

static_assert(sizeof(Bf16x4Pack) == 8);
static_assert(sizeof(Bf16x8Pack) == 16);

}} // namespace helios::ptx
