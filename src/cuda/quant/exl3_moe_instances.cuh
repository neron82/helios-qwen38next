#pragma once
// Getter declarations for the MoE kernel instantiations built as separate translation units
// (comp_units/exl3_moe_inst_*.cu). Pruned versus exllamav3: only the runtime-bitrate instances
// (K = 0, n128/n256) and the fixed-bitrate instances K = 3 and K = 4 are built; K = 1, 2, 5, 6,
// 7, 8 are not, and the launcher falls back to the K = 0 variant for those (see README_PORT.md).

#include <cuda_runtime_api.h>
#include "exl3_moe_kernel.cuh"

// Kernel function pointer type: the launchers take the address of a __global__ kernel
// The 6th parameter is the 2^40 fixed-point accumulator (long long*), not float* - see
// had_hf_r_128_d_inner and EXL3_MOE_KERNEL_ARGS.
// The parameter list mirrors EXL3_MOE_KERNEL_ARGS; the two device arrays after expert_count are the
// prefix-sum offsets and the LPT order (see exl3_moe_kernel.cuh).
typedef void (*fp_exl3_moe_kernel)(const half*, half*, half*, half*, half*, long long*, const uint16_t**, const half**, const half**, const uint16_t**, const half**, const half**, const uint16_t**, const half**, const half**, const int64_t*, const int64_t*, const int*, const int64_t*, const half*, int, int, int, int, int, int, float, int, int, int, int, int*);

#define DECL_GETTER(K_, n_, cb_) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K_##_n##n_##_cb##cb_();

// Runtime bitrate (Kg/Ku/Kd chosen inside the kernel)
DECL_GETTER(0, 128, 1)
DECL_GETTER(0, 256, 1)
DECL_GETTER(0, 128, 2)
DECL_GETTER(0, 256, 2)

// Fixed bitrate, Kg = Ku = Kd
DECL_GETTER(3, 128, 1)
DECL_GETTER(3, 256, 1)
DECL_GETTER(3, 128, 2)
DECL_GETTER(3, 256, 2)
DECL_GETTER(4, 128, 1)
DECL_GETTER(4, 256, 1)
DECL_GETTER(4, 128, 2)
DECL_GETTER(4, 256, 2)

// Wide row tiles: 32 and 64 rows per GEMM tile, as in exllamav3. N = 128 and mul1 only.
#define DECL_GETTER_M(K_, MT_) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K_##_n128_cb2_m##MT_();
DECL_GETTER_M(0, 32)
DECL_GETTER_M(0, 64)
DECL_GETTER_M(3, 32)
DECL_GETTER_M(3, 64)
DECL_GETTER_M(4, 32)
DECL_GETTER_M(4, 64)
#undef DECL_GETTER_M

#undef DECL_GETTER