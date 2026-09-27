cmake_minimum_required(VERSION 3.22)
project(aux_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_INCLUDE_DIR "/usr/local/cuda-13.0/targets/x86_64-linux/include")
set(CUDA_LIB_DIR "/usr/local/cuda-13.0/targets/x86_64-linux/lib")

add_compile_options(-O3 -Wall)
include_directories(${CUDA_INCLUDE_DIR})

add_library(aux STATIC
    activation.cu
    add.cu
    dsa_topk.cu
    gdn.cu
    gdn_chunked.cu
    gdn_pack.cu
    gr_mix.cu
    gr_mix_tc.cu
    hc_mix.cu
    norm.cu
    routing.cu
    routing_std.cu
)

target_include_directories(aux PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(aux PRIVATE
    -L${CUDA_LIB_DIR}
    -lcudart
)

# Parity tests. glue2.cu comes along because test_aux also covers the engine-layer GEMM it now
# carries the tensor-core path of (see aux_conv_gemm_tests.inc).
add_executable(test_aux test/test_aux.cpp ../../engine/glue2.cu)
target_link_libraries(test_aux PRIVATE aux -L${CUDA_LIB_DIR} -lcudart)
target_include_directories(test_aux PRIVATE ${CUDA_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Chunked gated delta rule: parity against the serial recurrence + run-to-run byte stability.
add_executable(test_gdn_chunked_parity test/test_gdn_chunked_parity.cpp)
target_link_libraries(test_gdn_chunked_parity PRIVATE aux -L${CUDA_LIB_DIR} -lcudart)
target_include_directories(test_gdn_chunked_parity PRIVATE ${CUDA_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Tensor-core mixer: accuracy against the fp32 oracle on a real dumped call, plus both paths'
# throughput. Needs a dump (HELIOS_DUMP_MIXER) from a real run, so it exits with a message rather
# than failing when one is not present.
# Batched decode: is the recurrence correct at bsz=2, seqlen=1? The only configuration paired
# decode needs and the only one no prefill chunk ever produces (they are all bsz=1, seqlen=n).
add_executable(test_gdn_bsz2 test/test_gdn_bsz2.cpp)
target_link_libraries(test_gdn_bsz2 PRIVATE aux -L${CUDA_LIB_DIR} -lcudart)
target_include_directories(test_gdn_bsz2 PRIVATE ${CUDA_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_gr_mix_tc test/test_gr_mix_tc.cpp)
target_link_libraries(test_gr_mix_tc PRIVATE aux -L${CUDA_LIB_DIR} -lcudart)
target_include_directories(test_gr_mix_tc PRIVATE ${CUDA_INCLUDE_DIR})
