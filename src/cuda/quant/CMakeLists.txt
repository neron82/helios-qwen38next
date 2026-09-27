# Standalone build for the ported EXL3 quant core (helios).
# Deliberately self-contained: the parent helios CMakeLists integrates this later via
# add_subdirectory, so nothing here reaches outside this directory.
cmake_minimum_required(VERSION 3.24)

# CUDA 13 toolkit, explicit so the build does not depend on PATH ordering.
if(EXISTS "/usr/local/cuda-13.0/bin/nvcc")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
endif()

project(exl3_quant_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# sm_86 only: the kernels carry __CUDA_ARCH__ == 860 special-cases and the target is RTX 3090.
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Core: quant/dequant + pack/reconstruct/hadamard + the matmul/MoE layer.
# The comp_units/ instantiations are split per (bit width, codebook) exactly as in exllamav3;
# pruned versus the original: bit widths 1 and 7, gemm shape 1, the int8 GEMV path, the
# cooperative-kernel autotuner, and MoE fixed-bitrate instances other than K = 3 / K = 4.
add_library(exl3_quant_core STATIC
    pack.cu
    reconstruct.cu
    hadamard.cu
    util.cu
    exl3_kernel_map.cu
    exl3_devctx.cu
    exl3_gemm.cu
    hgemm_f16acc.cu
    exl3_gemv.cu
    exl3_moe.cu
    comp_units/exl3_comp_unit_2_cb0.cu
    comp_units/exl3_comp_unit_2_cb1.cu
    comp_units/exl3_comp_unit_2_cb2.cu
    comp_units/exl3_comp_unit_3_cb0.cu
    comp_units/exl3_comp_unit_3_cb1.cu
    comp_units/exl3_comp_unit_3_cb2.cu
    comp_units/exl3_comp_unit_4_cb0.cu
    comp_units/exl3_comp_unit_4_cb1.cu
    comp_units/exl3_comp_unit_4_cb2.cu
    comp_units/exl3_moe_inst_k0_cb2_m32.cu
    comp_units/exl3_moe_inst_k0_cb2_m64.cu
    comp_units/exl3_moe_inst_k3_cb2_m32.cu
    comp_units/exl3_moe_inst_k3_cb2_m64.cu
    comp_units/exl3_moe_inst_k4_cb2_m32.cu
    comp_units/exl3_moe_inst_k4_cb2_m64.cu
    comp_units/exl3_comp_unit_5_cb0.cu
    comp_units/exl3_comp_unit_5_cb1.cu
    comp_units/exl3_comp_unit_5_cb2.cu
    comp_units/exl3_comp_unit_6_cb0.cu
    comp_units/exl3_comp_unit_6_cb1.cu
    comp_units/exl3_comp_unit_6_cb2.cu
    comp_units/exl3_comp_unit_8_cb0.cu
    comp_units/exl3_comp_unit_8_cb1.cu
    comp_units/exl3_comp_unit_8_cb2.cu
    comp_units/exl3_moe_inst_k0_n128_cb1.cu
    comp_units/exl3_moe_inst_k0_n256_cb1.cu
    comp_units/exl3_moe_inst_k0_n128_cb2.cu
    comp_units/exl3_moe_inst_k0_n256_cb2.cu
    comp_units/exl3_moe_inst_k3_cb1.cu
    comp_units/exl3_moe_inst_k3_cb2.cu
    comp_units/exl3_moe_inst_k4_cb1.cu
    comp_units/exl3_moe_inst_k4_cb2.cu
)

target_include_directories(exl3_quant_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Mirrors exllamav3's own nvcc flags (-O3 --use_fast_math -lineinfo). No torch include paths.
target_compile_options(exl3_quant_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(exl3_quant_core PUBLIC CUDA::cudart)


add_executable(test_reconstruct test/test_reconstruct.cpp)
# Compiled by nvcc (LANGUAGE CUDA) so the fp16 helper unions/macros in the shim are available
# and the launcher headers resolve identically to the library TUs.
set_source_files_properties(test/test_reconstruct.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(test_reconstruct PROPERTIES CUDA_STANDARD 17)
target_link_libraries(test_reconstruct PRIVATE exl3_quant_core)


add_executable(test_gemm_smoke test/test_gemm_smoke.cpp)
set_source_files_properties(test/test_gemm_smoke.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(test_gemm_smoke PROPERTIES CUDA_STANDARD 17)
target_link_libraries(test_gemm_smoke PRIVATE exl3_quant_core)


# Row-tile diagnostic: instantiates the inner GEMM at 16/32/64 rows per tile. It is a diagnostic,
# not a gate - it exits 0 whatever it finds, because its job is to REPORT which row tiles are
# correct. It only exists to unblock porting exllamav3's 32/64-row MoE tiles.
add_executable(test_gemm_rows test/test_gemm_rows.cpp)
set_source_files_properties(test/test_gemm_rows.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(test_gemm_rows PROPERTIES CUDA_STANDARD 17)
target_link_libraries(test_gemm_rows PRIVATE exl3_quant_core)
add_executable(bench_gemm test/bench_gemm.cpp)
set_source_files_properties(test/bench_gemm.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(bench_gemm PROPERTIES CUDA_STANDARD 17)
target_link_libraries(bench_gemm PRIVATE exl3_quant_core)
add_executable(test_mgemm_semantics test/test_mgemm_semantics.cpp)
set_source_files_properties(test/test_mgemm_semantics.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(test_mgemm_semantics PROPERTIES CUDA_STANDARD 17)
target_link_libraries(test_mgemm_semantics PRIVATE exl3_quant_core)


# Prefill reconstruct path (exl3::linear with HELIOS_RECONSTRUCT_PREFILL=1) against the trellis
# kernel it replaces. A gate, not a diagnostic: the two must compute the same product.
add_executable(test_recon_linear test/test_recon_linear.cpp)
set_source_files_properties(test/test_recon_linear.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(test_recon_linear PROPERTIES CUDA_STANDARD 17)
target_link_libraries(test_recon_linear PRIVATE exl3_quant_core)
