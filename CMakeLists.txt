cmake_minimum_required(VERSION 3.24)
project(helios CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)   # model.hpp pulls in std::span
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=x86-64-v3 -Wall")
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")

# Kernel libraries. These live in standalone projects (each with its own project() and its own
# test executables, usable on their own), but they are added here as subprojects so that a clean
# checkout builds with one cmake invocation instead of three. Linking targets rather than archive
# paths also lets CMake order the dependencies itself.
add_subdirectory(src/cuda/quant ${CMAKE_BINARY_DIR}/quant)
add_subdirectory(src/cuda/aux   ${CMAKE_BINARY_DIR}/aux)
add_subdirectory(src/cuda/attn  ${CMAKE_BINARY_DIR}/attn)

add_executable(helios
  src/main.cpp
  src/core/safetensors.cpp
  src/core/device.cpp
  src/core/model.cpp
  src/engine/sampler.cpp
  src/engine/glue.cu
  src/engine/glue2.cu
  src/engine/attn_layer.cu
  src/engine/gdn_layer.cu
  src/engine/moe_layer.cu
  src/engine/ple.cu
  src/engine/mtp.cu
  src/engine/runner.cpp
  src/engine/server.cpp
  src/core/chat.cpp
  src/tokenizer/tokenizer.cpp
)
target_include_directories(helios PRIVATE src third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/attn
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/aux
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/quant)
target_link_libraries(helios PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads
  exl3_quant_core aux attn_core)

add_executable(helios_utf8_test test/test_utf8.cpp)
target_include_directories(helios_utf8_test PRIVATE src)

add_executable(helios_chat_test src/core/test_chat.cpp src/core/chat.cpp src/tokenizer/tokenizer.cpp)
add_executable(helios_chat_qwen_test src/core/test_chat_qwen.cpp src/core/chat.cpp)

# OutputParser tool-call parsing. The chat tests above cover the RENDERER only; without a parser test
# the renderer and the parser could disagree about the tool-call dialect and nothing would notice -
# which is exactly what happened: the renderer emitted Qwen3's <function=NAME>/<parameter=K> form
# while the parser understood only the older <arg_key>/<arg_value> one, so every tool call came back
# with the whole markup blob as its name and "{}" as its arguments.
add_executable(test_chat_parse test/test_chat_parse.cpp src/core/chat.cpp)
target_include_directories(test_chat_parse PRIVATE src third_party)
target_link_libraries(test_chat_parse PRIVATE Threads::Threads)
target_include_directories(helios_chat_qwen_test PRIVATE src third_party)
target_link_libraries(helios_chat_qwen_test PRIVATE Threads::Threads)
target_include_directories(helios_chat_test PRIVATE src third_party)
target_link_libraries(helios_chat_test PRIVATE Threads::Threads)

add_executable(helios_tokenizer_test src/tokenizer/test_tokenizer.cpp src/tokenizer/tokenizer.cpp)
target_include_directories(helios_tokenizer_test PRIVATE src third_party)
target_link_libraries(helios_tokenizer_test PRIVATE Threads::Threads)

# Prefix-cache resume policy: pure logic, no GPU and no weights.
add_executable(helios_prefix_test test/test_prefix.cpp)
target_include_directories(helios_prefix_test PRIVATE src)

# Sequence-slot policy: the VRAM fit clamp, the per-slot context division and the routing decision.
# Pure logic, no GPU and no weights, so it runs anywhere - and it pins arithmetic whose failure
# modes are all silent: an over-generous clamp OOMs late, a wrong context budget writes one
# conversation's KV rows over another's, and a misrouted slot answers from another thread's state
# while looking entirely healthy.
add_executable(helios_seq_test test/test_seq.cpp)
target_include_directories(helios_seq_test PRIVATE src)

# Loader vocabulary + config test: RAM-only, so it needs no GPU and no weights - it resolves every
# tensor name the loader depends on against the real checkpoint and checks the derived config.
add_executable(helios_model_test test/test_model_load.cpp src/core/model.cpp src/core/safetensors.cpp
  src/core/device.cpp src/engine/ple.cu src/engine/glue2.cu)
target_include_directories(helios_model_test PRIVATE src third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/aux)
target_link_libraries(helios_model_test PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads aux)

# ---------------------------------------------------------------------------
# One command for the whole suite:  ctest --test-dir build --output-on-failure
#
# Every test target is registered here rather than scattered across the subdirectory CMakeLists, so a
# new test cannot be added without appearing in the run. Three were already registered in
# src/cuda/quant; the rest - including test_gemm_rows and test_gr_mix_tc, which had never been run by
# hand at all - were not, which is the mechanism by which a test silently rots.
# ---------------------------------------------------------------------------
enable_testing()
foreach(t
    test_gemm_smoke test_mgemm_semantics test_reconstruct test_gemm_rows test_recon_linear
    test_aux test_gr_mix_tc test_gdn_chunked_parity test_gdn_bsz2
    test_attn_parity test_qsa_parity test_kv_quant
    helios_prefix_test helios_model_test helios_chat_test helios_chat_qwen_test helios_tokenizer_test
    test_chat_parse helios_seq_test)
  if(TARGET ${t})
    add_test(NAME ${t} COMMAND ${t})
  endif()
endforeach()

# The tokenizer test reads test/tokenizer_vectors.json by a repo-root-relative path (it also accepts
# the path as argv[2]). ctest runs each test with cwd = the build dir, so without this it failed with
# "no vectors at test/tokenizer_vectors.json" - the test was never wrong, only ever run from the one
# directory where the relative path happened to resolve.
set_tests_properties(helios_tokenizer_test PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
