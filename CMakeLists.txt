cmake_minimum_required(VERSION 3.25)
project(dgpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Target GB10's architecture-specific instructions, including the
# e2m1x2 -> f16x2 conversion used by the FP4 kernels. These binaries
# require compute capability 12.1.
set(CMAKE_CUDA_ARCHITECTURES 121a)
# SSH login shells may omit CUDA from PATH. Respect explicit compiler choices,
# including toolchains, before looking for the conventional DGX OS installation.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
  if(CUDAToolkit_ROOT OR DEFINED ENV{CUDAToolkit_ROOT})
    set(DGPP_CUDA_ROOT "${CUDAToolkit_ROOT}")
    if(NOT DGPP_CUDA_ROOT)
      set(DGPP_CUDA_ROOT "$ENV{CUDAToolkit_ROOT}")
    endif()
    find_program(DGPP_NVCC nvcc HINTS "${DGPP_CUDA_ROOT}/bin" NO_DEFAULT_PATH NO_CACHE)
    if(NOT DGPP_NVCC)
      message(FATAL_ERROR "No nvcc under CUDAToolkit_ROOT=${DGPP_CUDA_ROOT}. Select the CUDA 13 toolkit root, or set CUDACXX=/path/to/cuda/bin/nvcc and retry with cmake --fresh --preset release.")
    endif()
  else()
    find_program(DGPP_NVCC nvcc HINTS "$ENV{CUDA_PATH}/bin" PATHS /usr/local/cuda/bin NO_CACHE)
  endif()
  if(DGPP_NVCC)
    set(CMAKE_CUDA_COMPILER "${DGPP_NVCC}" CACHE FILEPATH "CUDA compiler")
  endif()
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CUDA compiler not found. Install the CUDA 13 toolkit, or select an existing installation with CUDACXX=/path/to/cuda/bin/nvcc cmake --fresh --preset release. See docs/getting-started.md.")
endif()
enable_language(CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DGPP_WERROR "Treat compiler warnings as errors" OFF)
option(DGPP_BUILD_BENCHMARKS "Build platform, kernel and transport microbenchmarks" ON)
set(DGPP_SANITIZE "" CACHE STRING "Host sanitizer (address or undefined)")
set_property(CACHE DGPP_SANITIZE PROPERTY STRINGS "" address undefined)

if(DGPP_SANITIZE)
  if(NOT DGPP_SANITIZE MATCHES "^(address|undefined)$")
    message(FATAL_ERROR "DGPP_SANITIZE must be address, undefined, or empty")
  endif()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${DGPP_SANITIZE}>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=${DGPP_SANITIZE}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>)
  # CMake may select CUDA as a target's link language while still invoking the
  # host C++ driver for the final link.  Linker options therefore need the host
  # spelling for both C++-only and mixed CUDA/C++ targets.
  add_link_options(-fsanitize=${DGPP_SANITIZE})
endif()

option(DGPP_ENABLE_IBV "Build serving and RDMA targets (required even for the single-node server)" ON)

add_library(dgpp_common INTERFACE)
target_include_directories(dgpp_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(CUDAToolkit 13 REQUIRED COMPONENTS cudart cublasLt)
find_package(Threads REQUIRED)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

include(cmake/pcre2.cmake)

if(DGPP_ENABLE_IBV)
  find_path(IBVERBS_INCLUDE_DIR NAMES infiniband/verbs.h)
  find_library(IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT IBVERBS_INCLUDE_DIR OR NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "libibverbs development files are required for dgpp-serve. Install libibverbs-dev, or use -DDGPP_ENABLE_IBV=OFF for a build WITHOUT serving/RDMA targets.")
  endif()
endif()

if(NOT DGPP_ENABLE_IBV)
  message(STATUS "DGPP_ENABLE_IBV=OFF: dgpp-serve, releases and RDMA targets will not be built")
endif()

if(DGPP_ENABLE_IBV)
  target_compile_definitions(dgpp_common INTERFACE DGPP_HAVE_IBV=1)
endif()

function(dgpp_set_common t)
  target_link_libraries(${t} PRIVATE dgpp_common)
  target_compile_options(${t} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-ffp-contract=off>)
  if(DGPP_WERROR)
    target_compile_options(${t} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
  endif()
endfunction()

find_program(DGPP_CLANG_FORMAT
  NAMES clang-format clang-format-19 clang-format-18 clang-format-17)
if(DGPP_CLANG_FORMAT)
  file(GLOB_RECURSE DGPP_FORMAT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu)
  add_custom_target(format
    COMMAND ${DGPP_CLANG_FORMAT} -i ${DGPP_FORMAT_SOURCES}
    COMMENT "Formatting C++ and CUDA sources")
  add_custom_target(format-check
    COMMAND ${DGPP_CLANG_FORMAT} --dry-run --Werror ${DGPP_FORMAT_SOURCES}
    COMMENT "Checking C++ and CUDA formatting")
else()
  message(STATUS "clang-format not found; format targets are unavailable")
endif()

# ---- libraries ------------------------------------------------------------

add_library(dgpp_core STATIC
  src/common/log.cpp
  src/common/bf16_residency.cpp
  src/common/process_memory.cpp
  src/core/trace.cpp)
dgpp_set_common(dgpp_core)

# M5 control plane: TCP roster/startup/health (host-only; no CUDA).
add_library(dgpp_net STATIC
  src/net/tcp.cpp
  src/net/roster.cpp)
target_link_libraries(dgpp_net PUBLIC dgpp_core Threads::Threads)
dgpp_set_common(dgpp_net)

add_library(dgpp_loaders STATIC
  src/loaders/minijson.cpp
  src/loaders/shardspec.cpp
  src/loaders/hf_cache.cpp
  src/loaders/architecture.cpp
  src/loaders/resident_image.cpp)
target_link_libraries(dgpp_loaders PUBLIC dgpp_core)
dgpp_set_common(dgpp_loaders)

# The host-side rope tables (kernels/rope_scaling.hpp): the YaRN builder the
# DeepSeek-V4.1 CSA2 layers and the Qwen QSA rope share. Host-only, so the
# unit gate links it without dragging the CUDA kernels in.
add_library(dgpp_rope_scaling STATIC src/kernels/rope_scaling.cpp)
target_include_directories(dgpp_rope_scaling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_rope_scaling PUBLIC dgpp_common)
dgpp_set_common(dgpp_rope_scaling)

# Device ranges whose memory can be returned while the address stays reserved
# (the CUDA virtual-memory API: the driver library, beside the runtime). The
# weight loaders' side grants — bf12-only residency — are built on it.
add_library(dgpp_device_ranges STATIC src/loaders/releasable_range.cpp)
target_link_libraries(dgpp_device_ranges PUBLIC CUDA::cudart CUDA::cuda_driver)
dgpp_set_common(dgpp_device_ranges)

add_library(dgpp_kernels STATIC
  src/kernels/gemm_cublaslt.cpp
  src/kernels/kernels.cu
  src/kernels/doll_attention.cu
  src/kernels/kda.cu
  src/kernels/dsa.cu
  src/kernels/fp8_dequant.cu
  src/kernels/fp8_gemm.cu
  src/kernels/fp8w_gemm.cu
  src/kernels/fp4w_gemm.cu
  src/kernels/fp8_per_tensor.cu
  src/kernels/fp4_gemv.cu
  src/kernels/packq_gemv.cu
  src/kernels/packq_a8_gemv.cu
  src/kernels/packq_gemm.cu
  src/kernels/packq_head.cu
  src/kernels/head_screen.cu
  src/kernels/draft_head.cpp
  src/kernels/hadamard32.cu
  src/kernels/scale_gemm.cu
  src/kernels/bf16_gemv.cu
  src/kernels/bf12_gemv.cu
  src/kernels/bf12_companions.cpp
  src/kernels/mma_gemv.cu
  src/kernels/exl3_gemv.cu
  src/kernels/dense_mma_bf16w.cu
  src/kernels/dense_fp4_decode.cu
  src/kernels/l2_prefetch.cu
  src/kernels/segment_copy.cu
  src/kernels/glm_mhc.cu
  src/kernels/glm_moe.cu
  src/kernels/moe_w4a4.cu
  src/kernels/gdn_chunk.cu
  src/kernels/glm_vision.cu
  src/kernels/qwen_vision.cu
  src/kernels/glm_norm.cu
  src/kernels/pick.cu
  src/kernels/sample_pick.cu
  src/kernels/glm_spec.cu
  src/kernels/qwen_norm.cu
  src/kernels/qwen_gr.cu
  src/kernels/qwen_moe.cu
  src/kernels/qwen_ple.cu
  src/kernels/dsv41_engram.cu
  src/kernels/dsv41_dspark.cu
  src/kernels/csa2.cu
  src/kernels/dsv4_attn.cu
  src/kernels/qwen_mtp.cu
  src/kernels/dflash2.cu
  src/kernels/qsa.cu
  src/kernels/qsa_warp.cu
  src/kernels/qsa_fp8.cu
  src/kernels/qsa_prefill.cu
  src/kernels/full_attn.cu
  src/kernels/glm4_attn.cu
  src/kernels/sink_attn.cu
  src/kernels/mimo_mtp_history.cu
  src/kernels/sink_attn_prefill.cu
  src/kernels/add_rmsnorm.cu
  src/kernels/dory_norm.cu
  src/kernels/dory_attn.cu
  src/kernels/dory_head.cu)
target_link_libraries(dgpp_kernels PUBLIC CUDA::cudart CUDA::cublasLt CUDA::cuda_driver dgpp_core dgpp_rope_scaling)
dgpp_set_common(dgpp_kernels)
# The device sampler mirrors the host oracle bit for bit: no contraction of
# its multiplies and adds (the host builds with -ffp-contract=off).
set_source_files_properties(src/kernels/sample_pick.cu PROPERTIES
  COMPILE_OPTIONS "--fmad=false")
# The same for the Dory normalizations: the residual combine
# (na + lr * (nb - na)) rounds like the host reference's, which is built with
# -ffp-contract=off, so the device must not contract it into an fma either.
# The explicit fma() chains of the row sums are unaffected by -fmad.
set_source_files_properties(src/kernels/dory_norm.cu PROPERTIES
  COMPILE_OPTIONS "--fmad=false")

add_library(dgpp_models_doll STATIC
  src/models/gpt_doll.cu)
target_link_libraries(dgpp_models_doll PUBLIC dgpp_kernels dgpp_core)
dgpp_set_common(dgpp_models_doll)

add_library(dgpp_models_kda STATIC
  src/models/kda_state.cpp
  src/models/kda_layer.cu
  src/models/kda_reference.cpp
  src/models/kda_dump.cpp)
target_link_libraries(dgpp_models_kda PUBLIC dgpp_kernels dgpp_loaders CUDA::cudart)
dgpp_set_common(dgpp_models_kda)

add_library(dgpp_models_dsa STATIC
  src/models/dsa_reference.cpp
  src/models/dsa_dump.cpp
  src/models/dsa_state.cpp
  src/models/dsa_layer.cu)
target_link_libraries(dgpp_models_dsa PUBLIC dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_dsa)

# Host-only GLM model assembly (config parsing, expected-tensor binding); no
# CUDA dependency so the loader/binding layer builds and tests anywhere.
# The model-agnostic layers (the layout split of 2026-09-06): the text
# stack (tokenizer, chat template, tool grammar and parser), the scheduler
# with its prefix cache index, and the sampler (header-only, src/sample/).
add_library(dgpp_text STATIC
  src/text/tokenizer.cpp
  src/text/unicode_normalize.cpp
  src/text/chat_template.cpp
  src/text/dsv41_prompt.cpp
  src/text/dsv4_prompt.cpp
  src/text/tool_parser.cpp
  src/text/tool_grammar.cpp
  src/text/string_constraint.cpp
  src/text/json_grammar.cpp)
target_link_libraries(dgpp_text PUBLIC dgpp_loaders dgpp_pcre2)
dgpp_set_common(dgpp_text)

add_library(dgpp_sched STATIC
  src/sched/disk_cache.cpp
  src/sched/prefix_cache.cpp
  src/sched/scheduler.cpp)
target_link_libraries(dgpp_sched PUBLIC dgpp_text dgpp_core)
dgpp_set_common(dgpp_sched)

add_library(dgpp_models_glm STATIC
  src/models/glm/config.cpp
  src/models/glm/binding.cpp
  src/models/glm/mhc_reference.cpp
  src/models/glm/moe_reference.cpp
  src/models/glm/trace.cpp)
target_link_libraries(dgpp_models_glm PUBLIC dgpp_loaders dgpp_text dgpp_sched)
dgpp_set_common(dgpp_models_glm)

# Streaming resident loader (M4): managed-memory layer residency over the
# binding table, with the block-scale dequant bridge for the M3 interfaces.
# Chunk 6 adds the full diagnostic forward (glm_forward) and the reference
# dump reader (glm_dump) on top; they drive the parity-tested M2/M3 layer
# modules, hence the kda/dsa library links.
# GLM-4.7 (2026-09-09, docs/glm47_plan.md): the host-side model description.
add_library(dgpp_models_glm4 STATIC
  src/models/glm4/config.cpp
  src/models/glm4/binding.cpp
  src/models/glm4/attn_reference.cpp)
target_include_directories(dgpp_models_glm4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_glm4 PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_glm4)

# The GLM-4.7 resident loader (loaders/resident_stream.hpp + the family's builders).
add_library(dgpp_models_glm4_loader STATIC
  src/models/glm4/loader.cpp)
target_include_directories(dgpp_models_glm4_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_glm4_loader PUBLIC dgpp_models_glm4 dgpp_loaders CUDA::cudart dgpp_device_ranges)
dgpp_set_common(dgpp_models_glm4_loader)

# The GLM-4.7 layer objects and model (the paged K/V pool first; the layers
# and the model follow).
add_library(dgpp_models_glm4_layers STATIC
  src/models/glm4/kv_pool.cpp
  src/models/glm4/layers.cpp
  src/models/glm4/forward.cpp)
target_include_directories(dgpp_models_glm4_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_glm4_layers
  PUBLIC dgpp_models_glm4 dgpp_models_glm4_loader dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_glm4_layers)

# MiMo-V2.6-Flash (2026-09-22, docs/mimo_v26_flash_plan.md): the host-side
# model description.
add_library(dgpp_models_mimo STATIC
  src/models/mimo/config.cpp
  src/models/mimo/binding.cpp
  src/models/mimo/attn_reference.cpp)
target_include_directories(dgpp_models_mimo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_mimo PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_mimo)

# Dory1.0-4B (2026-10-16, docs/dory_plan.md): the host-side model description.
add_library(dgpp_models_dory STATIC
  src/models/dory/binding.cpp
  src/models/dory/config.cpp
  src/models/dory/reference.cpp)
target_include_directories(dgpp_models_dory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dory PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_dory)

# The Dory1.0-4B resident loader (loaders/resident_stream.hpp + the family's
# builders): one rank, packed NVFP4 kept as shipped.
add_library(dgpp_models_dory_loader STATIC
  src/models/dory/loader.cpp)
target_include_directories(dgpp_models_dory_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dory_loader PUBLIC dgpp_models_dory dgpp_loaders CUDA::cudart dgpp_device_ranges)
dgpp_set_common(dgpp_models_dory_loader)

# The Dory layer objects and the looped model (the paged K/V pool, the
# attention and MLP blocks, the session-core walk).
add_library(dgpp_models_dory_layers STATIC
  src/models/dory/kv_pool.cpp
  src/models/dory/layers.cpp
  src/models/dory/forward.cpp)
target_include_directories(dgpp_models_dory_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dory_layers
  PUBLIC dgpp_models_dory dgpp_models_dory_loader dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_dory_layers)
# Flint (2026-10-22, docs/flint_plan.md): the host-side config, binding
# table and attention oracle.
add_library(dgpp_models_flint STATIC
  src/models/flint/config.cpp
  src/models/flint/binding.cpp
  src/models/flint/attn_reference.cpp)
target_include_directories(dgpp_models_flint PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_flint PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_flint)

# The MiMo resident loader (loaders/resident_stream.hpp + the family's builders).
add_library(dgpp_models_mimo_loader STATIC
  src/models/mimo/loader.cpp)
target_include_directories(dgpp_models_mimo_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_mimo_loader PUBLIC dgpp_models_mimo dgpp_loaders CUDA::cudart dgpp_device_ranges)
dgpp_set_common(dgpp_models_mimo_loader)

# The Flint loader (docs/flint_plan.md §2): the shared resident stream over
# the family's bindings and TP geometry.
add_library(dgpp_models_flint_loader STATIC
  src/models/flint/loader.cpp)
target_include_directories(dgpp_models_flint_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_flint_loader PUBLIC dgpp_models_flint dgpp_loaders CUDA::cudart dgpp_device_ranges)
dgpp_set_common(dgpp_models_flint_loader)

# The MiMo layer objects and model (the paged K/V pool first; the layers
# and the model follow).
# The paged K/V pool the MiMo and Flint families share (bf16 or fp8 rows
# per (token, kv head) at the family's head widths).
add_library(dgpp_models_mimo_kv STATIC
  src/models/mimo/kv_pool.cpp)
target_include_directories(dgpp_models_mimo_kv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_mimo_kv PUBLIC dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_mimo_kv)

add_library(dgpp_models_mimo_layers STATIC
  src/models/mimo/layers.cpp
  src/models/mimo/forward.cpp)
target_include_directories(dgpp_models_mimo_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_mimo_layers
  PUBLIC dgpp_models_mimo dgpp_models_mimo_loader dgpp_models_mimo_kv dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_mimo_layers)

# The Flint layers and model (docs/flint_plan.md §3): the sink-attention
# kernels at qk 128, the BF16 dense MLP, QwenMoeLayer's routed chain and
# shared-expert tail, the MiMo paged pool at 128 / 128.
add_library(dgpp_models_flint_layers STATIC
  src/models/flint/layers.cpp
  src/models/flint/forward.cpp)
target_include_directories(dgpp_models_flint_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_flint_layers
  PUBLIC dgpp_models_flint dgpp_models_flint_loader dgpp_models_mimo_kv dgpp_models_qwen_moe dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_flint_layers)

# The full GLM-5.3 model on the session core (docs/glm53_plan.md G5).
add_library(dgpp_models_glm_dsa_layers STATIC
  src/models/glm_dsa/model.cpp)
target_include_directories(dgpp_models_glm_dsa_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_glm_dsa_layers
  PUBLIC dgpp_models_glm_dsa dgpp_models_glm_dsa_loader dgpp_models_dsa dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_glm_dsa_layers)

# Full GLM-5.3 (2026-09-12, docs/glm53_plan.md): the host-side model description.
add_library(dgpp_models_glm_dsa STATIC
  src/models/glm_dsa/config.cpp
  src/models/glm_dsa/binding.cpp)
target_include_directories(dgpp_models_glm_dsa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_glm_dsa PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_glm_dsa)

# The full GLM-5.3 resident loader (G3): the shared weight builder at the
# family's TP geometry, the packed-int slices, the draft's requant.
add_library(dgpp_models_glm_dsa_loader STATIC
  src/models/glm_dsa/loader.cpp)
target_include_directories(dgpp_models_glm_dsa_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_glm_dsa_loader PUBLIC dgpp_models_glm_dsa dgpp_loaders CUDA::cudart dgpp_device_ranges)
dgpp_set_common(dgpp_models_glm_dsa_loader)

# DeepSeek-V4.1-Flash (2026-09-13, docs/deepseek_v41_flash_plan.md): the
# host-side model description (config, binding).
add_library(dgpp_models_dsv41 STATIC
  src/models/dsv41/config.cpp
  src/models/dsv41/binding.cpp
  src/models/dsv41/engram_tables.cpp)
target_include_directories(dgpp_models_dsv41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dsv41 PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_dsv41)

# The DeepSeek-V4.1-Flash loader (docs/deepseek_v41_flash_plan.md G3): the
# ResidentLayerStream family, the fp8 / MXFP4 slice builders, the Engram
# table mappings.
add_library(dgpp_models_dsv41_loader STATIC
  src/models/dsv41/loader.cpp)
target_include_directories(dgpp_models_dsv41_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dsv41_loader PUBLIC dgpp_models_dsv41 dgpp_loaders CUDA::cudart dgpp_device_ranges)
dgpp_set_common(dgpp_models_dsv41_loader)

# The CSA2 attention layer and its caches (plan G4).
add_library(dgpp_models_csa2 STATIC
  src/models/dsv41/csa2_state.cpp
  src/models/dsv41/csa2_layer.cu)
target_include_directories(dgpp_models_csa2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_csa2 PUBLIC dgpp_kernels dgpp_common CUDA::cudart)
dgpp_set_common(dgpp_models_csa2)

# The DeepSeek-V4.1 layer objects beyond CSA2 (plan G4/G5): the Engram
# layer over the mmap'ed tables (and, later, the model).
add_library(dgpp_models_dsv41_layers STATIC
  src/models/dsv41/engram_layer.cpp
  src/models/dsv41/model.cpp)
target_include_directories(dgpp_models_dsv41_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dsv41_layers PUBLIC dgpp_models_dsv41_loader dgpp_models_csa2 dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_dsv41_layers)

# The step engine (MOONSHOT-1.md, engine.decode_engine = step): the stream
# plan with its totals and checks (src/step), and each family's plan
# generator over its loader's layout pass. Host code, no device.
add_library(dgpp_step STATIC
  src/step/plan.cpp)
target_include_directories(dgpp_step PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
dgpp_set_common(dgpp_step)

add_library(dgpp_step_dsv41 STATIC
  src/step/dsv41/plan.cpp
  src/step/dsv41/arena.cpp)
target_include_directories(dgpp_step_dsv41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_step_dsv41 PUBLIC dgpp_step dgpp_models_dsv41_loader dgpp_models_csa2)
dgpp_set_common(dgpp_step_dsv41)

# The streamer: the plan lowered to device tables, the persistent kernel
# (chain block, workers, lead, governor, deadlines) and its host side.
add_library(dgpp_step_stream STATIC
  src/step/device_plan.cpp
  src/step/streamer.cpp
  src/kernels/step/stream.cu)
target_include_directories(dgpp_step_stream PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_step_stream PUBLIC dgpp_step CUDA::cudart)
dgpp_set_common(dgpp_step_stream)

# DeepSeek-V4-Flash (the 0731 release, 2026-10-01): the host-side model
# description (config, binding).
add_library(dgpp_models_dsv4 STATIC
  src/models/dsv4/config.cpp
  src/models/dsv4/binding.cpp)
target_include_directories(dgpp_models_dsv4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dsv4 PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_dsv4)

# The DeepSeek-V4-Flash loader: the ResidentLayerStream family, the fp8
# (128-grid) / MXFP4 slice builders, the hash-routing tables.
add_library(dgpp_models_dsv4_loader STATIC
  src/models/dsv4/loader.cpp)
target_include_directories(dgpp_models_dsv4_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dsv4_loader PUBLIC dgpp_models_dsv4 dgpp_loaders CUDA::cudart dgpp_device_ranges)
dgpp_set_common(dgpp_models_dsv4_loader)

# The DeepSeek-V4-Flash attention layer and its caches, then the model.
add_library(dgpp_models_dsv4_attn STATIC
  src/models/dsv4/state.cpp
  src/models/dsv4/attn_layer.cu)
target_include_directories(dgpp_models_dsv4_attn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dsv4_attn PUBLIC dgpp_models_dsv4_loader dgpp_kernels dgpp_common CUDA::cudart)
dgpp_set_common(dgpp_models_dsv4_attn)

add_library(dgpp_models_dsv4_layers STATIC
  src/models/dsv4/model.cpp)
target_include_directories(dgpp_models_dsv4_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dsv4_layers PUBLIC dgpp_models_dsv4_loader dgpp_models_dsv4_attn dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_dsv4_layers)

# The DFlash2 block drafter (family-agnostic): the drafter model's config,
# the standalone weight load, the shared execution (models/dflash2/
# drafter.hpp) and the pool's draft planes. The target families host one
# drafter over their own pool planes, tap sites and head dispatch.
add_library(dgpp_models_dflash2 STATIC
  src/models/dflash2/dflash2.cpp
  src/models/dflash2/drafter.cpp)
target_include_directories(dgpp_models_dflash2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_dflash2 PUBLIC dgpp_kernels dgpp_loaders dgpp_common CUDA::cudart)
dgpp_set_common(dgpp_models_dflash2)

# Qwen3.8-Flash-Next: the host-side model description.
add_library(dgpp_models_qwen STATIC
  src/models/qwen/config.cpp
  src/models/qwen/binding.cpp
  src/models/qwen/config35.cpp
  src/models/qwen/binding35.cpp
  src/models/qwen/gdn_reference.cpp
  src/models/qwen/norm_reference.cpp
  src/models/qwen/gr_reference.cpp
  src/models/qwen/ple_reference.cpp
  src/models/qwen/qsa_reference.cpp)
target_include_directories(dgpp_models_qwen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_qwen PUBLIC dgpp_loaders dgpp_common dgpp_rope_scaling)
dgpp_set_common(dgpp_models_qwen)

# The Qwen resident loader (Q2): the shared weight builder at the family's
# TP geometry, plus the n-gram table's own residency.
add_library(dgpp_models_qwen_loader STATIC
  src/models/qwen/vision.cpp
  src/models/qwen/loader.cpp
  src/models/qwen/loader35.cpp
  src/models/qwen/dflash.cpp)
target_include_directories(dgpp_models_qwen_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_qwen_loader PUBLIC dgpp_models_qwen dgpp_loaders dgpp_kernels CUDA::cudart dgpp_device_ranges dgpp_models_dflash2)
dgpp_set_common(dgpp_models_qwen_loader)

# The Qwen layers (Q3): the MoE on the shared MoE layer, with its oracle.
# The Qwen MoE layer (the routed chain + the BF16 / fp8 shared-expert
# tail), on its own so the Flint family links it without the Qwen stack.
add_library(dgpp_models_qwen_moe STATIC
  src/models/qwen/moe_layer.cpp
  src/models/qwen/moe_reference.cpp)
target_include_directories(dgpp_models_qwen_moe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_qwen_moe PUBLIC dgpp_models_qwen dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_qwen_moe)

add_library(dgpp_models_qwen_layers STATIC
  src/models/qwen/layers.cpp
  src/models/qwen/kv_pool.cpp
  src/models/qwen/forward.cpp
  src/models/qwen/model35.cpp
  src/models/qwen/model35_kernels.cu)
target_include_directories(dgpp_models_qwen_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_qwen_layers
  PUBLIC dgpp_models_qwen dgpp_models_qwen_loader dgpp_models_qwen_moe dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_qwen_layers)

# The Qwen forward (Q4): fixture -> smoke; the python reference chain follows.
add_executable(qwen_forward_test tests/cuda/qwen_forward_test.cpp)
target_include_directories(qwen_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_forward_test PRIVATE dgpp_models_qwen_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_forward_test)
set(QWEN_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen_forward_fixture)
add_test(NAME qwen_forward_fixture COMMAND qwen_forward_test --write-fixture ${QWEN_FORWARD_FIXTURE})
add_test(NAME qwen_plan_check COMMAND qwen_forward_test --plan-check)
add_test(NAME qwen_w4a4_plan_default COMMAND ${CMAKE_COMMAND} -E env
  --unset=DGPP_MOE_W4A4 DGPP_MOE_W4A4_MIN_ROWS=256
  $<TARGET_FILE:qwen_forward_test> --w4a4-plan-check)
foreach(_mode IN ITEMS 0 1)
  add_test(NAME qwen_w4a4_plan_mode_${_mode} COMMAND ${CMAKE_COMMAND} -E env
    DGPP_MOE_W4A4=${_mode} DGPP_MOE_W4A4_MIN_ROWS=256
    $<TARGET_FILE:qwen_forward_test> --w4a4-plan-check)
endforeach()
foreach(_mode IN ITEMS empty invalid)
  if(_mode STREQUAL "empty")
    set(_w4a4_value "")
  else()
    set(_w4a4_value "invalid")
  endif()
  add_test(NAME qwen_w4a4_plan_mode_${_mode} COMMAND ${CMAKE_COMMAND} -E env
    DGPP_MOE_W4A4=${_w4a4_value} DGPP_MOE_W4A4_MIN_ROWS=256
    $<TARGET_FILE:qwen_forward_test> --w4a4-plan-check)
endforeach()
add_test(NAME qwen_forward_smoke COMMAND qwen_forward_test --smoke ${QWEN_FORWARD_FIXTURE})
set_tests_properties(qwen_forward_smoke PROPERTIES DEPENDS qwen_forward_fixture)
add_test(NAME qwen_qsa_prefill
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/qwen_qsa_prefill_test.py
          $<TARGET_FILE:qwen_forward_test>)
set_tests_properties(qwen_qsa_prefill PROPERTIES RUN_SERIAL TRUE TIMEOUT 360)
# The decode graph's programmatic edges need every kernel to wait first (engine/graph_pdl.hpp).
add_test(NAME pdl_wait_audit COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/pdl_wait_audit.py ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(pdl_wait_audit PROPERTIES LABELS "host")
# QSA always sees BF16 activations, across both expert formats and dense modes.
foreach(_dense_fp8 IN ITEMS 0 1)
  add_test(NAME qwen_qsa_prefill_nvfp4_dense_${_dense_fp8}
    COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_NVFP4=1 DGPP_TEST_DENSE_FP8=${_dense_fp8}
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/qwen_qsa_prefill_test.py
            $<TARGET_FILE:qwen_forward_test>)
  set_tests_properties(qwen_qsa_prefill_nvfp4_dense_${_dense_fp8}
    PROPERTIES RUN_SERIAL TRUE TIMEOUT 360 LABELS "gpu")
endforeach()
# The same model under the engine's YaRN knob (engine.rope_scaling, factor 2
# over the fixture's 4096 native positions): the table, the mscale and the
# lifted ceiling through a whole forward, and logits that differ from the
# plain rope's.
add_test(NAME qwen_forward_smoke_yarn
  COMMAND qwen_forward_test --smoke ${QWEN_FORWARD_FIXTURE} --rope-scaling 2.0:4096)
set_tests_properties(qwen_forward_smoke_yarn PROPERTIES DEPENDS qwen_forward_fixture)
# The YaRN ceiling actually crossed (the 2026-09-18 review): a synthetic
# native limit of 8 under the factor-2 knob (the extended 16) — both
# forward modes over the final valid position, the session's refusal of
# work past the extended limit, and a pool below the ceiling.
add_test(NAME qwen_forward_cross_limit
  COMMAND qwen_forward_test --cross-limit ${QWEN_FORWARD_FIXTURE})
set_tests_properties(qwen_forward_cross_limit PROPERTIES DEPENDS qwen_forward_fixture)
add_test(NAME qwen_forward_generate
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen_pure_ref.qwndump)
set_tests_properties(qwen_forward_generate PROPERTIES DEPENDS qwen_forward_fixture)
add_test(NAME qwen_forward_test
  COMMAND qwen_forward_test --checkpoint-dir ${QWEN_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen_pure_ref.qwndump)
set_tests_properties(qwen_forward_test PROPERTIES DEPENDS qwen_forward_generate)
# The AutoRound hybrid's twin (docs/qwen38_autoround_int4_plan.md S2): GPTQ
# int4 g128 experts, the int8 g128 head and the shipped fp8 dense stack
# through the same fixture chain — the Python reference dequantizes the
# GPTQ triples exactly (no per-weight rounding, the packed core's policy).
set(QWEN_GPTQ_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen_gptq_fixture)
add_test(NAME qwen_gptq_fixture COMMAND qwen_forward_test --write-fixture ${QWEN_GPTQ_FIXTURE} --gptq)
add_test(NAME qwen_gptq_smoke COMMAND qwen_forward_test --smoke ${QWEN_GPTQ_FIXTURE})
set_tests_properties(qwen_gptq_smoke PROPERTIES DEPENDS qwen_gptq_fixture)
add_test(NAME qwen_gptq_generate
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN_GPTQ_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen_gptq_ref.qwndump)
set_tests_properties(qwen_gptq_generate PROPERTIES DEPENDS qwen_gptq_fixture)
add_test(NAME qwen_gptq_forward_test
  COMMAND qwen_forward_test --checkpoint-dir ${QWEN_GPTQ_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen_gptq_ref.qwndump)
set_tests_properties(qwen_gptq_forward_test PROPERTIES DEPENDS qwen_gptq_generate)

# The QAD export's twin (issue #113, local-inference-lab/Qwen3.8-Flash-Next-
# NVFP4): the MXFP8 dense stack on the 1 x 32 grid, the NVFP4 n-gram table,
# F32 norms and weight-only NVFP4 draft experts through the same fixture
# chain — the Python reference dequantizes every form exactly (power-of-two
# scales, the e2m1 x e4m3 x global chain in modelopt's order).
set(QWEN_QAD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen_qad_fixture)
add_test(NAME qwen_qad_fixture COMMAND qwen_forward_test --write-fixture ${QWEN_QAD_FIXTURE} --qad)
add_test(NAME qwen_qad_smoke COMMAND qwen_forward_test --smoke ${QWEN_QAD_FIXTURE})
set_tests_properties(qwen_qad_smoke PROPERTIES DEPENDS qwen_qad_fixture)
add_test(NAME qwen_qad_generate
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN_QAD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen_qad_ref.qwndump)
set_tests_properties(qwen_qad_generate PROPERTIES DEPENDS qwen_qad_fixture)
add_test(NAME qwen_qad_forward_test
  COMMAND qwen_forward_test --checkpoint-dir ${QWEN_QAD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen_qad_ref.qwndump)
set_tests_properties(qwen_qad_forward_test PROPERTIES DEPENDS qwen_qad_generate)

# The CSF export is lossless: compare its nested-shard fixture against the
# independent Python reference generated from the identical plain QAD bytes.
set(QWEN_FP4LOCAL_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen_fp4local_fixture)
add_test(NAME qwen_fp4local_fixture COMMAND qwen_forward_test --write-fixture ${QWEN_FP4LOCAL_FIXTURE} --fp4local)
add_test(NAME qwen_fp4local_smoke COMMAND qwen_forward_test --smoke ${QWEN_FP4LOCAL_FIXTURE})
set_tests_properties(qwen_fp4local_smoke PROPERTIES DEPENDS qwen_fp4local_fixture)
add_test(NAME qwen_fp4local_forward_test
  COMMAND qwen_forward_test --checkpoint-dir ${QWEN_FP4LOCAL_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen_qad_ref.qwndump)
set_tests_properties(qwen_fp4local_forward_test PROPERTIES DEPENDS "qwen_fp4local_fixture;qwen_qad_generate")

add_test(NAME qwen_fp4local_forward_tile
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_PREFILL_TILE=1 $<TARGET_FILE:qwen_forward_test>
          --checkpoint-dir ${QWEN_FP4LOCAL_FIXTURE} --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen_qad_ref.qwndump)
set_tests_properties(qwen_fp4local_forward_tile PROPERTIES DEPENDS "qwen_fp4local_fixture;qwen_qad_generate")

# The Qwen session surface (Q6 stage A): incremental decode vs the re-forward.
add_executable(qwen_decode_test tests/cuda/qwen_decode_test.cpp)
target_include_directories(qwen_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_decode_test PRIVATE dgpp_models_qwen_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_decode_test)
add_test(NAME qwen_decode_test COMMAND qwen_decode_test --fixture ${QWEN_FORWARD_FIXTURE})
set_tests_properties(qwen_decode_test PROPERTIES DEPENDS qwen_forward_fixture)
add_test(NAME qwen_fp4local_decode_test COMMAND qwen_decode_test --fixture ${QWEN_FP4LOCAL_FIXTURE})
add_test(NAME qwen_fp4local_decode_rows_invariance COMMAND qwen_decode_test --fixture ${QWEN_FP4LOCAL_FIXTURE} --rows-invariance)
set_tests_properties(qwen_fp4local_decode_test qwen_fp4local_decode_rows_invariance
  PROPERTIES DEPENDS qwen_fp4local_fixture RUN_SERIAL TRUE)

add_test(NAME qwen_fp4local_decode_tile
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_PREFILL_TILE=1 $<TARGET_FILE:qwen_decode_test>
          --fixture ${QWEN_FP4LOCAL_FIXTURE})
add_test(NAME qwen_fp4local_decode_rows_invariance_tile
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_PREFILL_TILE=1 $<TARGET_FILE:qwen_decode_test>
          --fixture ${QWEN_FP4LOCAL_FIXTURE} --rows-invariance)
set_tests_properties(qwen_fp4local_decode_tile qwen_fp4local_decode_rows_invariance_tile
  PROPERTIES DEPENDS qwen_fp4local_fixture RUN_SERIAL TRUE)

# Qwen3.8-Flash-Next's decode step is one chain across row counts under the
# shipped FP8-dense recipe (2026-10-05: the fp8 head and the dense sites on
# the streaming form from one row, the QSA projections through the dense
# lowering): a 1..15-row verify bitwise the sixteen-row verify's rows.
add_test(NAME qwen_decode_rows_invariance COMMAND qwen_decode_test --fixture ${QWEN_FORWARD_FIXTURE} --rows-invariance)
set_tests_properties(qwen_decode_rows_invariance PROPERTIES DEPENDS qwen_forward_fixture)
add_test(NAME qwen_decode_fp8_head
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=4
          $<TARGET_FILE:qwen_decode_test> --fixture ${QWEN_FORWARD_FIXTURE} --fp8-head)
add_test(NAME qwen_decode_fp8_head_row_independent
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256
          $<TARGET_FILE:qwen_decode_test> --fixture ${QWEN_FORWARD_FIXTURE} --fp8-head)
set_tests_properties(qwen_decode_fp8_head qwen_decode_fp8_head_row_independent
  PROPERTIES DEPENDS qwen_forward_fixture RUN_SERIAL TRUE)
# The draft head's screen (engine.draft_head screen; kernels/head_screen.hpp)
# on the fixture's bf16 and fp8 heads with a two-block cap of its four: the
# eager speculator's transcript stays the plain greedy one, and every
# draft-logit parity gate holds with the masked rows.
add_test(NAME qwen_decode_draft_screen COMMAND qwen_decode_test --fixture ${QWEN_FORWARD_FIXTURE} --draft-screen 2)
add_test(NAME qwen_decode_draft_screen_fp8_head
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=4
          $<TARGET_FILE:qwen_decode_test> --fixture ${QWEN_FORWARD_FIXTURE} --fp8-head --draft-screen 2)
set_tests_properties(qwen_decode_draft_screen qwen_decode_draft_screen_fp8_head
  PROPERTIES DEPENDS qwen_forward_fixture RUN_SERIAL TRUE)
add_test(NAME qwen_gptq_decode_test COMMAND qwen_decode_test --fixture ${QWEN_GPTQ_FIXTURE})
add_test(NAME qwen_qad_decode_test COMMAND qwen_decode_test --fixture ${QWEN_QAD_FIXTURE})
set_tests_properties(qwen_qad_decode_test PROPERTIES DEPENDS qwen_qad_fixture)
add_test(NAME qwen_gptq_prefill_head COMMAND qwen_decode_test --fixture ${QWEN_GPTQ_FIXTURE} --prefill-head)
set_tests_properties(qwen_gptq_decode_test qwen_gptq_prefill_head PROPERTIES DEPENDS qwen_gptq_fixture)
add_test(NAME qwen_prefill_head
  COMMAND qwen_decode_test --fixture ${QWEN_FORWARD_FIXTURE} --prefill-head)
add_test(NAME qwen_prefill_head_all_rows
  COMMAND ${CMAKE_COMMAND} -E env DGPP_PREFILL_HEAD_ALL_ROWS=1
          $<TARGET_FILE:qwen_decode_test> --fixture ${QWEN_FORWARD_FIXTURE} --prefill-head)
set_tests_properties(qwen_prefill_head qwen_prefill_head_all_rows
  PROPERTIES DEPENDS qwen_forward_fixture LABELS "gpu" RUN_SERIAL TRUE)

add_executable(qwen_image_prefill_test tests/cuda/qwen_image_prefill_test.cpp)
target_include_directories(qwen_image_prefill_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_image_prefill_test PRIVATE dgpp_models_qwen_layers CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_image_prefill_test)
add_test(NAME qwen_image_prefill_test
  COMMAND qwen_image_prefill_test ${CMAKE_CURRENT_BINARY_DIR}/qwen_image_prefill_fixture)
set_tests_properties(qwen_image_prefill_test PROPERTIES LABELS "gpu" RUN_SERIAL TRUE SKIP_RETURN_CODE 2)

add_executable(qwen35_image_prefill_test tests/cuda/qwen35_image_prefill_test.cpp)
target_include_directories(qwen35_image_prefill_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen35_image_prefill_test PRIVATE dgpp_models_qwen_layers CUDA::cudart Threads::Threads)
dgpp_set_common(qwen35_image_prefill_test)
add_test(NAME qwen35_image_prefill_test
  COMMAND qwen35_image_prefill_test ${CMAKE_CURRENT_BINARY_DIR}/qwen35_image_prefill_fixture)
add_test(NAME qwen35_mixed_image_prefill_test
  COMMAND qwen35_image_prefill_test ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_image_prefill_fixture --mixed)
set_tests_properties(qwen35_image_prefill_test qwen35_mixed_image_prefill_test
  PROPERTIES LABELS "gpu" RUN_SERIAL TRUE SKIP_RETURN_CODE 2)

# The MoE layer both families run: GLM's sigmoid+bias router
# with the FP8 shared expert in the chain, Qwen's softmax router without
# it (models/qwen/moe_layer.hpp composes the BF16 shared expert on top).
add_library(dgpp_models_moe STATIC
  src/models/glm/moe_layer.cpp)
target_include_directories(dgpp_models_moe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_moe PUBLIC dgpp_models_glm dgpp_kernels)
dgpp_set_common(dgpp_models_moe)

add_library(dgpp_models_glm_loader STATIC
  src/models/glm/vision.cpp
  src/models/glm/loader.cpp
  src/models/glm/dump.cpp
  src/models/glm/forward.cpp
  src/models/glm/decode.cpp
  src/models/glm/mtp.cpp
  src/models/glm/tp.cpp
  src/models/glm/tp_parity.cpp)
target_link_libraries(dgpp_models_glm_loader
  PUBLIC dgpp_models_glm dgpp_models_moe dgpp_kernels dgpp_models_kda dgpp_models_dsa dgpp_device_ranges)
dgpp_set_common(dgpp_models_glm_loader)

add_executable(glm_bind_check apps/glm_bind_check.cpp)
target_link_libraries(glm_bind_check PRIVATE dgpp_models_glm dgpp_loaders)
dgpp_set_common(glm_bind_check)

add_executable(glm_dsa_bind_check apps/glm_dsa_bind_check.cpp)
target_link_libraries(glm_dsa_bind_check PRIVATE dgpp_models_glm_dsa dgpp_loaders)
dgpp_set_common(glm_dsa_bind_check)
add_executable(dsv41_bind_check apps/dsv41_bind_check.cpp)
target_link_libraries(dsv41_bind_check PRIVATE dgpp_models_dsv41 dgpp_loaders)
dgpp_set_common(dsv41_bind_check)
add_executable(qwen35_bind_check apps/qwen35_bind_check.cpp)
target_link_libraries(qwen35_bind_check PRIVATE dgpp_models_qwen_loader dgpp_loaders)
dgpp_set_common(qwen35_bind_check)
add_executable(dsv4_forward_check apps/dsv4_forward_check.cpp)
target_include_directories(dsv4_forward_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv4_forward_check PRIVATE dgpp_models_dsv4_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv4_forward_check)
add_executable(dsv4_bind_check apps/dsv4_bind_check.cpp)
target_link_libraries(dsv4_bind_check PRIVATE dgpp_models_dsv4 dgpp_loaders)
dgpp_set_common(dsv4_bind_check)

# M5 d4 at real scale: the shard-parity gate (loader-vs-bind bitwise +
# replicated digest + byte reconcile) as a deployment command — the same
# driver the CI glm_tp_test pins at the fixture, against the real
# checkpoint via --model (HF hub cache). No bus dependency.
add_executable(glm_shard_parity apps/glm_shard_parity.cpp)
target_link_libraries(glm_shard_parity PRIVATE dgpp_models_glm_loader)
dgpp_set_common(glm_shard_parity)

# Deployment evidence: stream real layers, reconcile bytes vs formula.
add_executable(glm_stream_check apps/glm_stream_check.cpp)
target_link_libraries(glm_stream_check PRIVATE dgpp_models_glm_loader)
dgpp_set_common(glm_stream_check)

  # The small-collective stall repro (engine hunt instrument, not a CI gate
  # until the fix lands — see the file comment).
  add_executable(bus_small_repro apps/bus_small_repro.cu)
  target_link_libraries(bus_small_repro PRIVATE dgpp_bus CUDA::cudart
                                                Threads::Threads)
  dgpp_set_common(bus_small_repro)

  # The in-process two-rank graph scheduling model that root-caused the
  # batched-MTP loopback stall (docs/batched_mtp_graph_stall.md): kernels-only
  # graphs never deadlock; one memset node per collective does at
  # CUDA_DEVICE_MAX_CONNECTIONS=1. Diagnostic, not a test.
  add_executable(graph_queue_repro apps/graph_queue_repro.cu)
  target_link_libraries(graph_queue_repro PRIVATE CUDA::cudart Threads::Threads)
  dgpp_set_common(graph_queue_repro)

# M4 deliverables 1+5 deployment run: the curated reference suite over the
# real checkpoint (dumps from tools/glm_reference_dump.py gen-torch) and
# DGPPTC1 route-trace capture for the traffic model.
add_executable(glm_forward_check apps/glm_forward_check.cpp)
target_link_libraries(glm_forward_check PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_forward_check)

add_executable(dgppctl apps/dgppctl.cpp src/common/log.cpp)
target_link_libraries(dgppctl PRIVATE CUDA::cudart)
dgpp_set_common(dgppctl)

# M5 deliverable 1 deployment driver: roster/startup/health over TCP.
add_executable(roster_check apps/roster_check.cpp)
target_link_libraries(roster_check PRIVATE dgpp_net)
dgpp_set_common(roster_check)

# M5 deliverable 2: the CollectiveBus data plane (verbs + CUDA). Gated on
# ibverbs availability; the control plane (dgpp_net) never requires it.
if(DGPP_ENABLE_IBV)
  add_library(dgpp_bus STATIC
    src/net/bus_types.cpp
    src/net/verbs.cpp
    src/net/collective_bus.cpp
    src/net/bus_kernel.cu
    src/net/nic_visibility.cu)
  target_link_libraries(dgpp_bus
    PUBLIC dgpp_net CUDA::cudart ${IBVERBS_LIBRARY})
  target_include_directories(dgpp_bus SYSTEM PUBLIC ${IBVERBS_INCLUDE_DIR})
  dgpp_set_common(dgpp_bus)

  add_executable(bus_check apps/bus_check.cpp)
  target_link_libraries(bus_check PRIVATE dgpp_bus)
  dgpp_set_common(bus_check)

  # M5 deliverable 5: the transport regression command (NIC→GPU visibility
  # on every node pair after driver/firmware changes).
  add_executable(nic_regress apps/nic_regress.cpp)
  target_link_libraries(nic_regress PRIVATE dgpp_bus CUDA::cudart Threads::Threads)
  dgpp_set_common(nic_regress)
  add_test(NAME nic_regress_selftest COMMAND nic_regress selftest)
  set_tests_properties(nic_regress_selftest PROPERTIES SKIP_RETURN_CODE 2)

  # M5 deliverable 3: TP block-boundary parity over loopback buses
  # (real verbs QPs, one process). Needs the model stack AND the bus.
  # The fixture side is g++-only (nvcc cannot compile minijson).
  add_executable(glm_tp_test tests/cuda/glm_tp_test.cpp
                        tests/cuda/glm_tp_fixture.cpp)
  target_include_directories(glm_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(glm_tp_test PRIVATE dgpp_models_glm_loader dgpp_bus
                                         CUDA::cudart Threads::Threads)
  dgpp_set_common(glm_tp_test)
  add_test(NAME glm_tp_test COMMAND glm_tp_test)
  add_test(NAME glm_tp_test_draft_screen
    COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_DRAFT_SCREEN=2 $<TARGET_FILE:glm_tp_test>)
  set_tests_properties(glm_tp_test_draft_screen PROPERTIES RUN_SERIAL TRUE LABELS "gpu" SKIP_RETURN_CODE 2)
  # One process hosts every rank's streams (main, prefetch side streams, the
  # bus engines' streams). The decode graph is kernels-only by contract
  # (glm_check_decode_graph at every capture site): a memset/memcpy node rides
  # the process-shared copy-engine queue, where a peer rank's queued
  # dependency wait blocked this rank's node behind it while the peer's
  # collective spun waiting on ours (docs/batched_mtp_graph_stall.md, fixed
  # 2026-09-03). With that contract the loopback world runs with prefetch on
  # at any connection count (gated at 1 and 32); 32 keeps the streams from
  # sharing queues for the eager collectives' sake.
  set_tests_properties(glm_tp_test PROPERTIES SKIP_RETURN_CODE 2
                       ENVIRONMENT "CUDA_DEVICE_MAX_CONNECTIONS=32")
  add_test(NAME glm_mtp_sampling_no_pipeline COMMAND glm_tp_test)
  set_tests_properties(glm_mtp_sampling_no_pipeline PROPERTIES
      SKIP_RETURN_CODE 2 LABELS "gpu;rdma" RUN_SERIAL TRUE
      ENVIRONMENT "CUDA_DEVICE_MAX_CONNECTIONS=32;DGPP_PIPELINE=0;DGPP_TEST_FILTER=glm_tp_serving_mtp_graph_sampling_matches_eager_speculator")

  # M5 deliverable 3 fabric runner: TP forward on real nodes.
  add_executable(glm_tp_check apps/glm_tp_check.cpp)
  target_link_libraries(glm_tp_check PRIVATE dgpp_models_glm_loader dgpp_bus
                                          CUDA::cudart Threads::Threads)
  dgpp_set_common(glm_tp_check)

  # M6 Stage 1d: greedy generation end-to-end at real dims (w1 local
  # reference + fabric TP with the distributed pick).
  add_executable(glm_gen_check apps/glm_gen_check.cpp)
  target_link_libraries(glm_gen_check PRIVATE dgpp_models_glm_loader dgpp_bus
                                            CUDA::cudart Threads::Threads)
  dgpp_set_common(glm_gen_check)

  # M6 Stage 4a: the OpenAI-compatible serving app (world 1 — the local
  # reference engine; Stage 4b adds the fabric journal + resident TP).
  # The server binary: target dgpp_serve_app, installed as dgpp-serve (the
  # layout split of 2026-09-06; dgpp_serve is the library it links).
  add_executable(dgpp_serve_app apps/dgpp_serve.cpp)
  target_link_libraries(dgpp_serve_app PRIVATE dgpp_models_glm_loader dgpp_models_qwen_layers dgpp_models_glm4_layers dgpp_models_glm_dsa_layers
                                             dgpp_models_dsv41_layers dgpp_models_dsv4_layers dgpp_models_mimo_layers
                                             dgpp_models_dory_layers
                                             dgpp_models_flint_layers
                                             dgpp_loaders dgpp_bus
                                             dgpp_serve CUDA::cudart
                                             Threads::Threads)
  dgpp_set_common(dgpp_serve_app)
  set_target_properties(dgpp_serve_app PROPERTIES OUTPUT_NAME dgpp-serve)
  # The version stamp: cmake/version.cmake rewrites the header
  # at every build only when the tree's sha or dirtiness changed.
  set(DGPP_VERSION_HEADER ${CMAKE_BINARY_DIR}/generated/dgpp_version.hpp)
  add_custom_target(dgpp_version
    BYPRODUCTS ${DGPP_VERSION_HEADER}
    COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DOUT=${DGPP_VERSION_HEADER} -DBASE_VERSION=${PROJECT_VERSION}
            -P ${CMAKE_SOURCE_DIR}/cmake/version.cmake
    COMMENT "stamping dgpp_version.hpp")
  add_dependencies(dgpp_serve_app dgpp_version)
  target_include_directories(dgpp_serve_app PRIVATE ${CMAKE_BINARY_DIR}/generated)
  # The install layout (the versioned artifact, scripts/release.sh): the
  # binary finds the CUDA runtime it was built against beside itself, so
  # a node needs only the driver, rdma-core, libnl and libstdc++.
  set_target_properties(dgpp_serve_app PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  install(TARGETS dgpp_serve_app RUNTIME DESTINATION bin)
  foreach(_lib cudart cublasLt)
    get_target_property(_loc CUDA::${_lib} IMPORTED_LOCATION)
    if(NOT _loc)
      get_target_property(_loc CUDA::${_lib} LOCATION)
    endif()
    get_filename_component(_real "${_loc}" REALPATH)
    install(FILES "${_real}" DESTINATION lib
            RENAME lib${_lib}.so.${CUDAToolkit_VERSION_MAJOR})
  endforeach()
  install(PROGRAMS scripts/dgpp-cluster scripts/serve_api_check.py scripts/vision_api_check.py DESTINATION scripts)
  install(FILES scripts/site_env.py scripts/cluster_process.py scripts/cluster_doctor.py
    scripts/serve_client.py scripts/download_model.py scripts/cache_sync.py
    scripts/discover_roce.py scripts/data_paths.py DESTINATION scripts)
  install(FILES .env.example DESTINATION .)
  file(GLOB DGPP_DEPLOYMENT_TEMPLATES CONFIGURE_DEPENDS "deploy/*.example.json")
  install(FILES ${DGPP_DEPLOYMENT_TEMPLATES} DESTINATION deploy)
  install(FILES deploy/README.md DESTINATION deploy)
  install(FILES docs/release-install.md DESTINATION . RENAME README.md)
  install(FILES requirements-download.txt LICENSE DESTINATION .)
  install(FILES docs/dependencies.md docs/networking.md docs/vision.md DESTINATION docs)
  install(FILES docs/nvme-cache.md docs/configuration.md docs/prefix-cache.md DESTINATION docs)
  install(FILES docs/licenses/PCRE2.md docs/licenses/stb_image.md docs/licenses/simplewebp.md
                docs/licenses/pytorch.md DESTINATION docs/licenses)
  install(FILES deploy/nginx.example.conf DESTINATION deploy)
endif()

enable_testing()
add_executable(unit_tests
  tests/unit/disk_cache_test.cpp
  tests/unit/image_inputs_test.cpp
  tests/unit/file_inputs_test.cpp
  tests/unit/log_test.cpp
  tests/unit/process_memory_test.cpp
  tests/unit/minijson_test.cpp
  tests/unit/chat_template_test.cpp
  tests/unit/chat_template_limits_test.cpp
  tests/unit/prometheus_test.cpp
  tests/unit/utf8_test.cpp
  tests/unit/arena_test.cpp
  tests/unit/safetensors_test.cpp
  tests/unit/resident_image_test.cpp
  tests/unit/fp8_test.cpp
  tests/unit/fp4_test.cpp
  tests/unit/latent_format_test.cpp
  tests/unit/shardspec_test.cpp
  tests/unit/hf_cache_test.cpp
  tests/unit/kda_geometry_test.cpp
  tests/unit/dsa_geometry_test.cpp
  tests/unit/glm_config_test.cpp
  tests/unit/glm_binding_test.cpp
  tests/unit/glm_trace_test.cpp
  tests/unit/glm_route_audit_test.cpp
  tests/unit/sampler_test.cpp
  tests/unit/tool_parser_test.cpp
  tests/unit/tool_grammar_test.cpp
  tests/unit/json_grammar_test.cpp
  tests/unit/net_test.cpp
  tests/unit/cluster_config_test.cpp
  tests/unit/qwen_config_test.cpp
  tests/unit/qwen_csf_test.cpp
  tests/unit/qwen_image_stage_test.cpp
  tests/unit/qwen_vision_dequant_test.cpp
  tests/unit/qwen_binding_test.cpp
  tests/unit/qwen35_config_test.cpp
  tests/unit/qwen35_binding_test.cpp
  tests/unit/qwen_dflash2_config_test.cpp
  tests/unit/qwen_dflash_config_test.cpp
  tests/unit/dflash2_speculator_test.cpp
  tests/unit/qwen_rope_scaling_test.cpp
  tests/unit/glm4_config_test.cpp
  tests/unit/glm4_binding_test.cpp
  tests/unit/glm_dsa_config_test.cpp
  tests/unit/glm_dsa_binding_test.cpp
  tests/unit/dsv41_config_test.cpp
  tests/unit/dsv41_binding_test.cpp
  tests/unit/dsv4_config_test.cpp
  tests/unit/dsv4_binding_test.cpp
  tests/unit/mimo_config_test.cpp
  tests/unit/mimo_binding_test.cpp
  tests/unit/dory_config_test.cpp
  tests/unit/dory_binding_test.cpp
  tests/unit/dory_reference_test.cpp
  tests/unit/flint_config_test.cpp
  tests/unit/flint_binding_test.cpp
  tests/unit/nvfp4_quant_test.cpp
  tests/unit/verify_schedule_test.cpp
  tests/unit/lookup_draft_test.cpp
  src/core/trace.cpp)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(unit_tests PRIVATE CUDA::cudart dgpp_loaders
                                        dgpp_models_glm dgpp_models_qwen dgpp_models_qwen_loader
                                        dgpp_models_glm4
                                        dgpp_models_glm_dsa dgpp_models_dsv41 dgpp_models_dsv4 dgpp_models_mimo
                                        dgpp_models_dory
                                        dgpp_models_glm_dsa dgpp_models_dsv41 dgpp_models_dsv4 dgpp_models_mimo dgpp_models_flint
                                        dgpp_rope_scaling
                                        dgpp_net dgpp_serve Threads::Threads)
target_compile_definitions(unit_tests PRIVATE DGPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
dgpp_set_common(unit_tests)
add_test(NAME unit_tests COMMAND unit_tests)
target_sources(unit_tests PRIVATE tests/unit/site_options_test.cpp)

# The vLLM YaRN oracle, offline (2026-09-19): tests/python/qwen_yarn_oracle.py
# --emit records the recipe's rope goldens — the YaRN table, the attention
# mscale, the bf16 cos/sin cache rows and rotated Q/K at several positions,
# for the recipe and four non-default parameter combinations — under the
# digest of the container that produced them, and this gate compares the
# engine's own table builder, mscale and rotation against that file. No vLLM,
# no container, no GPU; refresh the fixture in the container when vLLM's yarn
# code or the recipe changes (tests/unit/qwen_yarn_fixture_test.cpp).
add_executable(qwen_yarn_fixture_test tests/unit/qwen_yarn_fixture_test.cpp)
target_include_directories(qwen_yarn_fixture_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qwen_yarn_fixture_test PRIVATE dgpp_loaders dgpp_models_qwen
                                        dgpp_rope_scaling Threads::Threads)
target_compile_definitions(qwen_yarn_fixture_test PRIVATE DGPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
dgpp_set_common(qwen_yarn_fixture_test)
add_test(NAME qwen_yarn_fixture_test
  COMMAND qwen_yarn_fixture_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen_yarn_oracle.json)

# The BPE work bound and phase semantics need no checkpoint or GPU.
add_executable(bpe_test tests/host/bpe_test.cpp)
target_include_directories(bpe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
dgpp_set_common(bpe_test)
add_test(NAME bpe_test COMMAND bpe_test)

add_executable(word_cache_test tests/host/word_cache_test.cpp)
target_include_directories(word_cache_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(word_cache_test PRIVATE Threads::Threads)
dgpp_set_common(word_cache_test)
add_test(NAME word_cache_test COMMAND word_cache_test)

add_executable(tokenizer_patterns_test tests/host/tokenizer_patterns_test.cpp)
target_link_libraries(tokenizer_patterns_test PRIVATE dgpp_text)
dgpp_set_common(tokenizer_patterns_test)
add_test(NAME tokenizer_patterns_test
  COMMAND tokenizer_patterns_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)

# M6 Stage 3: the tokenizer differential gate (host). Skips with rc 2 when
# the model cache is absent (same discipline as the real-checkpoint gates).
add_executable(tokenizer_test tests/host/tokenizer_test.cpp)
target_include_directories(tokenizer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(tokenizer_test PRIVATE dgpp_models_glm Threads::Threads)
dgpp_set_common(tokenizer_test)
add_test(NAME tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm_tokenizer_goldens.jsonl)
set_tests_properties(tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME qwen_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen_tokenizer_goldens.jsonl)
add_test(NAME dsv41_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dsv41_tokenizer_goldens.jsonl)
set_tests_properties(dsv41_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME dsv4_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dsv4_tokenizer_goldens.jsonl)
set_tests_properties(dsv4_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)

# The DeepSeek-V4.1 prompt renderer's differential gate (no Jinja: the
# checkpoint's encoding.py is the golden source).
add_executable(dsv41_prompt_test tests/host/dsv41_prompt_test.cpp)
target_include_directories(dsv41_prompt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dsv41_prompt_test PRIVATE dgpp_text dgpp_models_glm Threads::Threads)
dgpp_set_common(dsv41_prompt_test)
add_test(NAME dsv41_prompt_test
  COMMAND dsv41_prompt_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dsv41_prompt_goldens.jsonl)
set_tests_properties(dsv41_prompt_test PROPERTIES SKIP_RETURN_CODE 2)
# The EXL3 trellis host codec (models/exl3_matrix.hpp) against the numpy
# reference's pins on the LibertAIDAI DeepSeek-V4.1-Flash-1017 pack
# (docs/deepseek_v41_flash_exl3_plan.md §2.4). CUDA-free.
add_executable(exl3_codec_test tests/host/exl3_codec_test.cpp)
target_include_directories(exl3_codec_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(exl3_codec_test PRIVATE dgpp_common)
dgpp_set_common(exl3_codec_test)
add_test(NAME exl3_codec_test COMMAND exl3_codec_test)
# The DeepSeek-V4 (DeepSeek-V4-Flash-0731) prompt renderer's differential
# gate: the checkpoint's encoding_dsv4.py is the golden source; the same
# corpus walks its DSML spelling through the tool grammar and parser.
add_executable(dsv4_prompt_test tests/host/dsv4_prompt_test.cpp)
target_include_directories(dsv4_prompt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dsv4_prompt_test PRIVATE dgpp_text dgpp_models_glm Threads::Threads)
dgpp_set_common(dsv4_prompt_test)
add_test(NAME dsv4_prompt_test
  COMMAND dsv4_prompt_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dsv4_prompt_goldens.jsonl)
set_tests_properties(dsv4_prompt_test PROPERTIES SKIP_RETURN_CODE 2)
set_tests_properties(qwen_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm4_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm4_tokenizer_goldens.jsonl)
set_tests_properties(glm4_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm_dsa_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm_dsa_tokenizer_goldens.jsonl)
set_tests_properties(glm_dsa_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME mimo_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mimo_tokenizer_goldens.jsonl)
set_tests_properties(mimo_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME flint_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/flint_tokenizer_goldens.jsonl)
set_tests_properties(flint_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)

# M6 Stage 3b: the chat-template differential gate (host). Skips with rc 2
# when the model cache is absent, same as above.
add_executable(chat_template_test tests/host/chat_template_test.cpp)
target_include_directories(chat_template_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(chat_template_test PRIVATE dgpp_models_glm Threads::Threads)
dgpp_set_common(chat_template_test)
add_test(NAME chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm_chat_template_goldens.jsonl)
set_tests_properties(chat_template_test PROPERTIES SKIP_RETURN_CODE 2)

# The M6 tool-grammar regression harness (the templates' host machine over
# the real Dory checkpoint, grammar spec included): the probe set behind
# the serve-side list_mask hard failure of 2026-10-26.
add_executable(dory_grammar_repro tests/host/dory_grammar_repro.cpp)
target_include_directories(dory_grammar_repro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dory_grammar_repro PRIVATE dgpp_models_glm Threads::Threads)
dgpp_set_common(dory_grammar_repro)
add_test(NAME dory_grammar_repro COMMAND dory_grammar_repro)
set_tests_properties(dory_grammar_repro PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME qwen_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen_chat_template_goldens.jsonl)
set_tests_properties(qwen_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm4_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm4_chat_template_goldens.jsonl)
set_tests_properties(glm4_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm_dsa_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm_dsa_chat_template_goldens.jsonl)
set_tests_properties(glm_dsa_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME mimo_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mimo_chat_template_goldens.jsonl)
set_tests_properties(mimo_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME flint_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/flint_chat_template_goldens.jsonl)
set_tests_properties(flint_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)

# M6 Stage 4: the serving capstone — HTTP/SSE over epoll + the OpenAI
# generation service (host-only; the engine adapters live in the apps,
# dgpp_serve never links CUDA — CI runs the service gate with a fake
# engine and frontend, no model cache). Stage 4b adds the admission
# journal (fabric_serve.cpp over the host-only dgpp_net TCP primitives).
add_library(dgpp_serve STATIC
  src/serve/image_inputs.cpp
  src/serve/vision_frontend.cpp
  src/serve/glm_vision_frontend.cpp
  src/serve/qwen_vision_frontend.cpp
  src/serve/file_inputs.cpp
  src/serve/cluster_config.cpp
  src/serve/nvme_cache_check.cpp
  src/serve/http_server.cpp
  src/serve/generation_service.cpp
  src/serve/fabric_serve.cpp
  src/serve/serve_stats.cpp
  src/serve/prometheus.cpp
  src/serve/rank_metrics.cpp)
target_link_libraries(dgpp_serve
  PUBLIC dgpp_core dgpp_models_glm dgpp_loaders dgpp_net Threads::Threads)
dgpp_set_common(dgpp_serve)

add_executable(http_server_test tests/host/http_server_test.cpp)
target_link_libraries(http_server_test PRIVATE dgpp_serve Threads::Threads)
dgpp_set_common(http_server_test)
add_test(NAME http_server_test COMMAND http_server_test)

add_executable(nvme_cache_test tests/host/nvme_cache_test.cpp)
target_link_libraries(nvme_cache_test PRIVATE dgpp_serve)
dgpp_set_common(nvme_cache_test)
add_test(NAME nvme_cache_test COMMAND nvme_cache_test)

# M6 Stage 4 service gate: the REAL HTTP/SSE service over a fake engine
# + fake frontend — the OpenAI surface, host-only, always runs.
add_executable(serve_test tests/host/serve_test.cpp)
target_link_libraries(serve_test PRIVATE dgpp_serve Threads::Threads)
dgpp_set_common(serve_test)
add_test(NAME serve_test COMMAND serve_test)
add_test(NAME serve_test_wake COMMAND serve_test)
set_tests_properties(serve_test_wake PROPERTIES ENVIRONMENT "DGPP_TEST_STREAM_WAKE=1")

# M6 Stage 4b gate: the admission journal (the metronome), both loops,
# and the §11 identity — real localhost TCP + real HTTP, engine faked
# (host-only, always runs).
add_executable(fabric_serve_test tests/host/fabric_serve_test.cpp)
target_link_libraries(fabric_serve_test PRIVATE dgpp_serve
                                                   Threads::Threads)
dgpp_set_common(fabric_serve_test)
add_test(NAME fabric_serve_test COMMAND fabric_serve_test)
add_test(NAME fabric_serve_test_wake COMMAND fabric_serve_test)
set_tests_properties(fabric_serve_test_wake PROPERTIES ENVIRONMENT "DGPP_TEST_STREAM_WAKE=1")

# The step engine's plan gate (MOONSHOT-1.md section 5, phase 2): the
# DeepSeek-V4.1 plan from the config alone tiles every layer's resident
# span and matches the hand count from the checkpoint's own headers
# (host-only, no model, always runs).
# The streamer's rate gate (section 5, phase 2): the DeepSeek-V4.1 plan
# under the arena, streamed from a device arena of its layers' sizes with
# the design chain's latencies interleaved, at 95 percent of the ceiling;
# the deadline and poison exits. Needs the GPU and about 4 GB free.
add_executable(step_stream_test tests/cuda/step_stream_test.cpp tests/cuda/step_reference_kernels.cu)
target_include_directories(step_stream_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(step_stream_test PRIVATE dgpp_step_stream dgpp_step_dsv41 CUDA::cudart Threads::Threads)
dgpp_set_common(step_stream_test)
add_test(NAME step_stream_test COMMAND step_stream_test)

# The MXFP4 expert consumer's gate (phase 3): exact to two bf16 ulps
# against an fp32 reference over tile-major bytes, bitwise across launches
# and group sizes, the plain grids' rate reported. GPU, about 1 GB free.
add_executable(step_consumer_test tests/cuda/step_consumer_test.cu)
target_include_directories(step_consumer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(step_consumer_test PRIVATE CUDA::cudart)
dgpp_set_common(step_consumer_test)
add_test(NAME step_consumer_test COMMAND step_consumer_test)

add_executable(step_plan_test tests/host/step_plan_test.cpp)
target_include_directories(step_plan_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit)
target_link_libraries(step_plan_test PRIVATE dgpp_step_dsv41)
dgpp_set_common(step_plan_test)
add_test(NAME step_plan_test COMMAND step_plan_test)

# M6 Stage 2b: the scheduler policy gate (host, no model, no skip path —
# the FakeEngine replaces the engine, so the policy is pinned everywhere
# CI runs).
add_executable(scheduler_test tests/host/scheduler_test.cpp)
target_include_directories(scheduler_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(scheduler_test PRIVATE dgpp_models_glm Threads::Threads)
dgpp_set_common(scheduler_test)
add_test(NAME scheduler_test COMMAND scheduler_test)

add_executable(shutdown_watchdog_test tests/host/shutdown_watchdog_test.cpp)
target_link_libraries(shutdown_watchdog_test PRIVATE Threads::Threads)
dgpp_set_common(shutdown_watchdog_test)
add_test(NAME shutdown_watchdog_test COMMAND shutdown_watchdog_test)
set_tests_properties(shutdown_watchdog_test PROPERTIES TIMEOUT 15)

add_executable(engine_watchdog_test tests/host/engine_watchdog_test.cpp)
target_link_libraries(engine_watchdog_test PRIVATE Threads::Threads)
dgpp_set_common(engine_watchdog_test)
add_test(NAME engine_watchdog_test COMMAND engine_watchdog_test)
set_tests_properties(engine_watchdog_test PROPERTIES TIMEOUT 20)

add_executable(cuda_wait_test tests/cuda/cuda_wait_test.cpp)
target_link_libraries(cuda_wait_test PRIVATE CUDA::cudart Threads::Threads)
dgpp_set_common(cuda_wait_test)
add_test(NAME cuda_wait_test COMMAND cuda_wait_test)
set_tests_properties(cuda_wait_test PROPERTIES TIMEOUT 15 SKIP_RETURN_CODE 2)

add_executable(flag_protocol_test tests/cuda/flag_protocol_test.cu)
target_link_libraries(flag_protocol_test PRIVATE CUDA::cudart)
dgpp_set_common(flag_protocol_test)

add_test(NAME unit COMMAND unit_tests)
add_test(NAME checkpoint_audit_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/checkpoint_audit_test.py)
add_test(NAME chat_template_generator_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/chat_template_generator_test.py)
set_tests_properties(chat_template_generator_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME route_trace_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/route_trace_test.py)
add_test(NAME sampling_profile_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/sampling_profile_test.py)
add_test(NAME site_env_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/site_env_test.py)
add_test(NAME script_reports_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/script_reports_test.py)
add_test(NAME portability_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/portability_test.py)
add_test(NAME timed_load_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/timed_load_test.py)
add_test(NAME serve_campaign_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/serve_campaign_test.py)
add_test(NAME serve_admission_bench_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/serve_admission_bench_test.py)
add_test(NAME qwen_yarn_release_check_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/qwen_yarn_release_check_test.py)
add_test(NAME qwen_head_compare_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/qwen_head_compare_test.py)
add_test(NAME spark_cross_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/spark_cross_test.py)
add_test(NAME setup_tools_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/setup_tools_test.py)
add_test(NAME bootstrap_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/bootstrap_test.py)
add_test(NAME launcher_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/launcher_test.py)
add_test(NAME cluster_process_swap_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/cluster_process_swap_test.py -v
    ProcessSwapTest.test_invalid_policy_is_rejected
    ProcessSwapTest.test_version_probe_rejects_invalid_policy
    ProcessSwapTest.test_preflight_forwards_swap_policy_to_peer_version_probes
    ProcessSwapTest.test_preflight_respects_resolved_node_swap_policy)
add_test(NAME cluster_process_swap_systemd_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/cluster_process_swap_test.py -v
    ProcessSwapTest.test_scope_has_zero_swap_and_recorded_group_can_stop_it
    ProcessSwapTest.test_scope_failure_does_not_run_the_command_without_protection
    ProcessSwapTest.test_version_probe_uses_zero_swap_scope
    ProcessSwapTest.test_version_probe_does_not_fall_back_when_scope_fails)
if(TARGET dgpp_serve_app)
  add_test(NAME serve_startup_test
    COMMAND ${CMAKE_COMMAND} -E env "DGPP_SERVE_TEST_BINARY=$<TARGET_FILE:dgpp_serve_app>"
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/serve_startup_test.py)
endif()

# KDA operator tests (CUDA); skip without a device as above. The dump-parity
# runner is a host-only TU (minijson + nvcc don't mix).
add_executable(kda_test
  tests/cuda/kda_test.cu
  tests/cuda/kda_dump_parity.cpp)
target_include_directories(kda_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kda_test PRIVATE dgpp_models_kda CUDA::cudart Threads::Threads)
dgpp_set_common(kda_test)
add_test(NAME kda_test COMMAND kda_test)
set_tests_properties(kda_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(dsa_test
  tests/cuda/dsa_test.cu
  tests/cuda/dsa_dump_parity.cpp)
target_include_directories(dsa_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dsa_test PRIVATE dgpp_models_dsa dgpp_models_kda CUDA::cudart Threads::Threads)
dgpp_set_common(dsa_test)
add_test(NAME dsa_test COMMAND dsa_test)
set_tests_properties(dsa_test PROPERTIES SKIP_RETURN_CODE 2)

# M4 streaming loader tests (CUDA): synthetic mini-checkpoint on disk,
# byte-exact layout merges, dequant bridge vs host oracle, formula parity.
# Qwen3.8-Flash-Next loader (Q2): the fixture, the slices at worlds 1/2/4,
# the formulas, the n-gram table, the image round trip.
# The Gated DeltaNet recurrence (Q3): the KDA kernel's scalar-gate mode.
add_executable(qwen_gr_test tests/cuda/qwen_gr_test.cu)
target_include_directories(qwen_gr_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_gr_test PRIVATE dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_gr_test)
add_test(NAME qwen_gr_test COMMAND qwen_gr_test)

add_executable(qwen_norm_test tests/cuda/qwen_norm_test.cu)
target_include_directories(qwen_norm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_norm_test PRIVATE dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_norm_test)
add_test(NAME qwen_norm_test COMMAND qwen_norm_test)

add_executable(dflash2_kernels_test tests/cuda/dflash2_kernels_test.cu)
target_include_directories(dflash2_kernels_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dflash2_kernels_test PRIVATE dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(dflash2_kernels_test)
add_test(NAME dflash2_kernels_test COMMAND dflash2_kernels_test)

add_executable(gdn_test tests/cuda/gdn_test.cu)
target_include_directories(gdn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(gdn_test PRIVATE dgpp_models_qwen dgpp_models_kda CUDA::cudart Threads::Threads)
dgpp_set_common(gdn_test)
add_test(NAME gdn_test COMMAND gdn_test)

add_executable(qsa_test tests/cuda/qsa_test.cu)
target_include_directories(qsa_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qsa_test PRIVATE dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qsa_test)
add_test(NAME qsa_test COMMAND qsa_test)

add_executable(qwen_fp8_kv_attn_test tests/cuda/qwen_fp8_kv_attn_test.cu)
target_include_directories(qwen_fp8_kv_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_fp8_kv_attn_test PRIVATE dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_fp8_kv_attn_test)
add_test(NAME qwen_fp8_kv_attn_test COMMAND qwen_fp8_kv_attn_test)

add_executable(qsa_fp8_native_test tests/cuda/qsa_fp8_native_test.cu)
target_include_directories(qsa_fp8_native_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qsa_fp8_native_test PRIVATE dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qsa_fp8_native_test)
add_test(NAME qsa_fp8_native_test COMMAND qsa_fp8_native_test)

add_executable(qwen_ple_test tests/cuda/qwen_ple_test.cu)
target_include_directories(qwen_ple_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_ple_test PRIVATE dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_ple_test)
add_test(NAME qwen_ple_test COMMAND qwen_ple_test)

# DeepSeek-V4.1-Flash Engram kernels (2026-09-13, docs/deepseek_v41_flash_plan.md G2).
add_executable(csa2_test tests/cuda/csa2_test.cu)
target_include_directories(csa2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(csa2_test PRIVATE dgpp_models_dsa dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(csa2_test)
add_test(NAME csa2_test COMMAND csa2_test)

add_executable(csa2_layer_test tests/cuda/csa2_layer_test.cu)
target_include_directories(csa2_layer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(csa2_layer_test PRIVATE dgpp_models_csa2 dgpp_kernels CUDA::cudart CUDA::cublasLt Threads::Threads)
dgpp_set_common(csa2_layer_test)
add_test(NAME csa2_layer_test COMMAND csa2_layer_test)

# The dsv41 suites run twice: on the fp8 / MXFP4 release fixture and, as
# the `_exl3` variants, on the EXL3 pack fixture (DGPP_DSV41_FIXTURE=exl3,
# tests/cuda/dsv41_fixture.hpp) from their own working directory.
set(DSV41_EXL3_WORKDIR ${CMAKE_CURRENT_BINARY_DIR}/dsv41_exl3)
file(MAKE_DIRECTORY ${DSV41_EXL3_WORKDIR})
set(DSV41_EXL3_ENV ${CMAKE_COMMAND} -E env DGPP_DSV41_FIXTURE=exl3)

add_executable(dsv41_engram_layer_test tests/cuda/dsv41_engram_layer_test.cpp)
target_include_directories(dsv41_engram_layer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_engram_layer_test PRIVATE dgpp_models_dsv41_layers CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_engram_layer_test)
add_test(NAME dsv41_engram_layer_test COMMAND dsv41_engram_layer_test)
add_test(NAME dsv41_engram_layer_test_exl3 COMMAND ${DSV41_EXL3_ENV} $<TARGET_FILE:dsv41_engram_layer_test>
         WORKING_DIRECTORY ${DSV41_EXL3_WORKDIR})

add_executable(dsv41_model_test tests/cuda/dsv41_model_test.cpp)
target_include_directories(dsv41_model_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_model_test PRIVATE dgpp_models_dsv41_layers dgpp_step_dsv41 dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_model_test)
add_test(NAME dsv41_model_test COMMAND dsv41_model_test)
add_test(NAME dsv41_model_test_exl3 COMMAND ${DSV41_EXL3_ENV} $<TARGET_FILE:dsv41_model_test>
         WORKING_DIRECTORY ${DSV41_EXL3_WORKDIR})

# The forward against the python reference dump (numpy: the repo's .venv
# when present, else the system python).
add_executable(dsv41_forward_test tests/cuda/dsv41_forward_test.cpp)
target_include_directories(dsv41_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_forward_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_forward_test)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3)
  set(DSV41_PYTHON ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3)
else()
  set(DSV41_PYTHON ${Python3_EXECUTABLE})
endif()
# The forward chain on one fixture: S the test-name suffix ("" the release
# fixture, "_exl3" the EXL3 pack), FIXTURE_ENV the fixture switch's
# VAR=value (empty for the release).
function(dsv41_forward_chain S FIXTURE_ENV)
  set(FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/dsv41_forward_fixture${S})
  if(FIXTURE_ENV STREQUAL "")
    set(ENV "")
  else()
    set(ENV ${CMAKE_COMMAND} -E env ${FIXTURE_ENV})
  endif()
  add_test(NAME dsv41_forward_fixture${S} COMMAND ${ENV} $<TARGET_FILE:dsv41_forward_test> --write-fixture ${FIXTURE})
  add_test(NAME dsv41_forward_smoke${S} COMMAND ${ENV} $<TARGET_FILE:dsv41_forward_test> --smoke ${FIXTURE})
  set_tests_properties(dsv41_forward_smoke${S} PROPERTIES DEPENDS dsv41_forward_fixture${S})
  add_test(NAME dsv41_forward_generate${S}
    COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
            --checkpoint-dir ${FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_pure_ref${S}.dump --tokens 40)
  set_tests_properties(dsv41_forward_generate${S} PROPERTIES DEPENDS dsv41_forward_fixture${S})
  # End to end (the drift of eight layers with fp4 caches: relaxed budgets),
  # writing the engine's layer states; then the teacher-forced dump (every
  # reference layer fed the engine's input to it) under the strict budgets.
  add_test(NAME dsv41_forward_test_e2e${S}
    COMMAND ${CMAKE_COMMAND} -E env ${FIXTURE_ENV} DGPP_DSV41_DUMP_ENGINE_STATES=${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states${S}.bin
            $<TARGET_FILE:dsv41_forward_test> --checkpoint-dir ${FIXTURE}
            --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_pure_ref${S}.dump --relaxed)
  set_tests_properties(dsv41_forward_test_e2e${S} PROPERTIES DEPENDS dsv41_forward_generate${S})
  add_test(NAME dsv41_forward_generate_teacher${S}
    COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
            --checkpoint-dir ${FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_ref${S}.dump --tokens 40
            --teacher ${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states${S}.bin)
  set_tests_properties(dsv41_forward_generate_teacher${S} PROPERTIES DEPENDS dsv41_forward_test_e2e${S})
  add_test(NAME dsv41_forward_test${S}
    COMMAND ${ENV} $<TARGET_FILE:dsv41_forward_test> --checkpoint-dir ${FIXTURE} --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_ref${S}.dump)
  set_tests_properties(dsv41_forward_test${S} PROPERTIES DEPENDS dsv41_forward_generate_teacher${S})
  # The bounded prefill (plan §1.8, the production mode): the same chain
  # against the reference's --prefill bounded walk (the decoder over the
  # last window rows), end to end then teacher-forced.
  add_test(NAME dsv41_forward_generate_bounded${S}
    COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
            --checkpoint-dir ${FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_bounded_ref${S}.dump --tokens 40
            --prefill bounded)
  set_tests_properties(dsv41_forward_generate_bounded${S} PROPERTIES DEPENDS dsv41_forward_fixture${S})
  add_test(NAME dsv41_forward_test_bounded_e2e${S}
    COMMAND ${CMAKE_COMMAND} -E env ${FIXTURE_ENV} DGPP_DSV41_DUMP_ENGINE_STATES=${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states_bounded${S}.bin
            $<TARGET_FILE:dsv41_forward_test> --checkpoint-dir ${FIXTURE}
            --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_bounded_ref${S}.dump --relaxed --bounded
            --exact-dump ${CMAKE_CURRENT_BINARY_DIR}/dsv41_pure_ref${S}.dump
            --exact-states ${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states${S}.bin)
  set_tests_properties(dsv41_forward_test_bounded_e2e${S} PROPERTIES DEPENDS "dsv41_forward_generate_bounded${S};dsv41_forward_test_e2e${S}")
  add_test(NAME dsv41_forward_generate_teacher_bounded${S}
    COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
            --checkpoint-dir ${FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_bounded_ref${S}.dump --tokens 40
            --prefill bounded --teacher ${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states_bounded${S}.bin)
  set_tests_properties(dsv41_forward_generate_teacher_bounded${S} PROPERTIES DEPENDS dsv41_forward_test_bounded_e2e${S})
  add_test(NAME dsv41_forward_test_bounded${S}
    COMMAND ${ENV} $<TARGET_FILE:dsv41_forward_test> --checkpoint-dir ${FIXTURE}
            --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_bounded_ref${S}.dump --bounded)
  set_tests_properties(dsv41_forward_test_bounded${S} PROPERTIES DEPENDS dsv41_forward_generate_teacher_bounded${S})
endfunction()
dsv41_forward_chain("" "")
dsv41_forward_chain("_exl3" "DGPP_DSV41_FIXTURE=exl3")
set(DSV41_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/dsv41_forward_fixture)

# The DeepSeek-V4.1 session surface: decode vs the re-forward, interleaved
# slots, the chunked prefill, the prefix snapshots, the eager DSpark speculator.
add_executable(dsv41_decode_test tests/cuda/dsv41_decode_test.cpp)
target_include_directories(dsv41_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_decode_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_decode_test)
add_test(NAME dsv41_decode_test COMMAND dsv41_decode_test --fixture ${DSV41_FORWARD_FIXTURE})
set_tests_properties(dsv41_decode_test PROPERTIES DEPENDS dsv41_forward_fixture)
add_test(NAME dsv41_decode_test_exl3 COMMAND ${DSV41_EXL3_ENV} $<TARGET_FILE:dsv41_decode_test>
         --fixture ${DSV41_FORWARD_FIXTURE}_exl3 WORKING_DIRECTORY ${DSV41_EXL3_WORKDIR})
set_tests_properties(dsv41_decode_test_exl3 PROPERTIES DEPENDS dsv41_forward_fixture_exl3)

add_executable(dsv41_engram_test tests/cuda/dsv41_engram_test.cu)
target_include_directories(dsv41_engram_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_engram_test PRIVATE dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_engram_test)
add_test(NAME dsv41_engram_test COMMAND dsv41_engram_test)

add_executable(qwen_moe_test tests/cuda/qwen_moe_test.cu)
target_include_directories(qwen_moe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_moe_test PRIVATE dgpp_models_qwen_layers dgpp_models_kda CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_moe_test)
add_test(NAME qwen_moe_test COMMAND qwen_moe_test)

add_executable(qwen_loader_test tests/cuda/qwen_loader_test.cpp)
target_include_directories(qwen_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_loader_test PRIVATE dgpp_models_qwen_loader CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_loader_test)
add_test(NAME qwen_loader_test COMMAND qwen_loader_test)

add_executable(glm4_attn_test tests/cuda/glm4_attn_test.cu)
target_include_directories(glm4_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm4_attn_test PRIVATE dgpp_models_glm4 dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_attn_test)
add_test(NAME glm4_attn_test COMMAND glm4_attn_test)

add_executable(full_attn_test tests/cuda/full_attn_test.cu)
target_include_directories(full_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(full_attn_test PRIVATE dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(full_attn_test)
add_test(NAME full_attn_test COMMAND full_attn_test)
add_executable(qwen_full_attn_test tests/cuda/qwen_full_attn_test.cpp)
target_include_directories(qwen_full_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_full_attn_test PRIVATE dgpp_models_qwen_layers dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_full_attn_test)
add_test(NAME qwen_full_attn_test COMMAND qwen_full_attn_test)

# The Qwen3.8-27B forward gate (#84): fixture -> smoke; the python reference
# end to end (relaxed, the engine's states written) -> the reference
# teacher-forced on them -> the strict parity.
add_executable(qwen35_forward_test tests/cuda/qwen35_forward_test.cpp)
target_include_directories(qwen35_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen35_forward_test PRIVATE dgpp_models_qwen_layers dgpp_models_qwen_loader dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen35_forward_test)
set(QWEN35_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen35_forward_fixture)
add_test(NAME qwen35_forward_fixture COMMAND qwen35_forward_test --write-fixture ${QWEN35_FORWARD_FIXTURE})
add_test(NAME qwen35_forward_smoke COMMAND qwen35_forward_test --smoke ${QWEN35_FORWARD_FIXTURE})
set_tests_properties(qwen35_forward_smoke PROPERTIES DEPENDS qwen35_forward_fixture)
add_test(NAME qwen35_decode_rows_invariance COMMAND qwen35_forward_test --rows-invariance ${QWEN35_FORWARD_FIXTURE})
set_tests_properties(qwen35_decode_rows_invariance PROPERTIES DEPENDS qwen35_forward_fixture)
add_test(NAME qwen35_decode_rows_invariance_kvfp8 COMMAND qwen35_forward_test --rows-invariance ${QWEN35_FORWARD_FIXTURE} --kv-fp8)
set_tests_properties(qwen35_decode_rows_invariance_kvfp8 PROPERTIES DEPENDS qwen35_forward_fixture)
add_test(NAME qwen35_decode_rows_invariance_attnfp8 COMMAND qwen35_forward_test --rows-invariance ${QWEN35_FORWARD_FIXTURE} --attn-fp8)
set_tests_properties(qwen35_decode_rows_invariance_attnfp8 PROPERTIES DEPENDS qwen35_forward_fixture)
add_test(NAME qwen35_prefill_group_invariance COMMAND qwen35_forward_test --group-invariance ${QWEN35_FORWARD_FIXTURE})
set_tests_properties(qwen35_prefill_group_invariance PROPERTIES DEPENDS qwen35_forward_fixture)
add_test(NAME qwen35_forward_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_pure_ref.q35dump)
set_tests_properties(qwen35_forward_generate PROPERTIES DEPENDS qwen35_forward_fixture)
add_test(NAME qwen35_forward_test_e2e
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_pure_ref.q35dump --relaxed
          --engine-states ${CMAKE_CURRENT_BINARY_DIR}/qwen35_engine_states.bin)
set_tests_properties(qwen35_forward_test_e2e PROPERTIES DEPENDS qwen35_forward_generate)
add_test(NAME qwen35_forward_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_teacher_ref.q35dump
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/qwen35_engine_states.bin)
set_tests_properties(qwen35_forward_generate_teacher PROPERTIES DEPENDS qwen35_forward_test_e2e)
add_test(NAME qwen35_forward_test
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_teacher_ref.q35dump)
set_tests_properties(qwen35_forward_test PROPERTIES DEPENDS qwen35_forward_generate_teacher)
# The NVFP4 mixed release (#131): the same gate over the mixed-form fixture
# (channel-FP8 attention + late MLP + lm_head, NVFP4 early MLPs, BF16 draft).
set(QWEN35_MIXED_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_fixture COMMAND qwen35_forward_test --write-fixture-mixed ${QWEN35_MIXED_FIXTURE})
add_test(NAME qwen35_mixed_smoke COMMAND qwen35_forward_test --smoke ${QWEN35_MIXED_FIXTURE})
set_tests_properties(qwen35_mixed_smoke PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_resident_mtp COMMAND qwen35_forward_test --resident-mtp ${QWEN35_MIXED_FIXTURE})
set_tests_properties(qwen35_mixed_resident_mtp PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_rows_invariance COMMAND qwen35_forward_test --rows-invariance ${QWEN35_MIXED_FIXTURE})
set_tests_properties(qwen35_mixed_rows_invariance PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_rows_invariance_kvfp8 COMMAND qwen35_forward_test --rows-invariance ${QWEN35_MIXED_FIXTURE} --kv-fp8)
set_tests_properties(qwen35_mixed_rows_invariance_kvfp8 PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_rows_invariance_attnfp8 COMMAND qwen35_forward_test --rows-invariance ${QWEN35_MIXED_FIXTURE} --attn-fp8)
set_tests_properties(qwen35_mixed_rows_invariance_attnfp8 PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_ref.q35dump)
set_tests_properties(qwen35_mixed_generate PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_e2e
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_ref.q35dump --relaxed
          --engine-states ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_engine_states.bin)
set_tests_properties(qwen35_mixed_e2e PROPERTIES DEPENDS qwen35_mixed_generate)
add_test(NAME qwen35_mixed_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_teacher.q35dump
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_engine_states.bin)
set_tests_properties(qwen35_mixed_generate_teacher PROPERTIES DEPENDS qwen35_mixed_e2e)
add_test(NAME qwen35_mixed_test
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_teacher.q35dump)
set_tests_properties(qwen35_mixed_test PROPERTIES DEPENDS qwen35_mixed_generate_teacher)
# The MoE release (122B layout at toy widths: ModelOpt NVFP4 routed experts,
# BF16 rest): smoke/rows/group and independent reference/teacher parity.
set(QWEN35_MOE_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moe_fixture)
add_test(NAME qwen35_moe_fixture COMMAND qwen35_forward_test --write-fixture-moe ${QWEN35_MOE_FIXTURE})
add_test(NAME qwen35_moe_smoke COMMAND qwen35_forward_test --smoke ${QWEN35_MOE_FIXTURE})
set_tests_properties(qwen35_moe_smoke PROPERTIES DEPENDS qwen35_moe_fixture)
add_test(NAME qwen35_moe_rows_invariance COMMAND qwen35_forward_test --rows-invariance ${QWEN35_MOE_FIXTURE})
set_tests_properties(qwen35_moe_rows_invariance PROPERTIES DEPENDS qwen35_moe_fixture)
add_test(NAME qwen35_moe_group_invariance COMMAND qwen35_forward_test --group-invariance ${QWEN35_MOE_FIXTURE})
set_tests_properties(qwen35_moe_group_invariance PROPERTIES DEPENDS qwen35_moe_fixture)
add_test(NAME qwen35_moe_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MOE_FIXTURE} --tokens 16 --no-mtp
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moe_ref.q35dump)
set_tests_properties(qwen35_moe_generate PROPERTIES DEPENDS qwen35_moe_fixture)
add_test(NAME qwen35_moe_e2e
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MOE_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moe_ref.q35dump --relaxed
          --engine-states ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moe_states.bin)
set_tests_properties(qwen35_moe_e2e PROPERTIES DEPENDS qwen35_moe_generate)
add_test(NAME qwen35_moe_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MOE_FIXTURE} --tokens 16 --no-mtp
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moe_states.bin
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moe_teacher.q35dump)
set_tests_properties(qwen35_moe_generate_teacher PROPERTIES DEPENDS qwen35_moe_e2e)
add_test(NAME qwen35_moe_test
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MOE_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moe_teacher.q35dump)
set_tests_properties(qwen35_moe_test PROPERTIES DEPENDS qwen35_moe_generate_teacher)
set_tests_properties(qwen35_moe_fixture qwen35_moe_smoke qwen35_moe_rows_invariance
    qwen35_moe_group_invariance qwen35_moe_generate qwen35_moe_e2e
    qwen35_moe_generate_teacher qwen35_moe_test PROPERTIES RUN_SERIAL TRUE LABELS "gpu")
# Exercise the FP4 prefill dispatch at its 1024-row boundary.
# The short mixed fixture above stays on the decode/short-prefill kernel.
add_test(NAME qwen35_mixed_prefill_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MIXED_FIXTURE} --tokens 1024 --no-mtp
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_prefill_ref.q35dump)
set_tests_properties(qwen35_mixed_prefill_generate PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_prefill_e2e
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_prefill_ref.q35dump --relaxed
          --engine-states ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_prefill_states.bin)
set_tests_properties(qwen35_mixed_prefill_e2e PROPERTIES DEPENDS qwen35_mixed_prefill_generate)
add_test(NAME qwen35_mixed_prefill_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MIXED_FIXTURE} --tokens 1024 --no-mtp
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_prefill_teacher.q35dump
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_prefill_states.bin)
set_tests_properties(qwen35_mixed_prefill_generate_teacher PROPERTIES DEPENDS qwen35_mixed_prefill_e2e)
add_test(NAME qwen35_mixed_prefill_test
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_prefill_teacher.q35dump)
set_tests_properties(qwen35_mixed_prefill_test PROPERTIES DEPENDS qwen35_mixed_prefill_generate_teacher)
# A longer mixed-quant prefill spanning multiple tile groups.
add_test(NAME qwen35_mixed_long_prefill_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MIXED_FIXTURE} --tokens 3072 --no-mtp
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_long_prefill_ref.q35dump)
set_tests_properties(qwen35_mixed_long_prefill_generate PROPERTIES DEPENDS qwen35_mixed_fixture)
add_test(NAME qwen35_mixed_long_prefill_e2e
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_long_prefill_ref.q35dump --relaxed
          --engine-states ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_long_prefill_states.bin)
set_tests_properties(qwen35_mixed_long_prefill_e2e PROPERTIES DEPENDS qwen35_mixed_long_prefill_generate)
add_test(NAME qwen35_mixed_long_prefill_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen35_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN35_MIXED_FIXTURE} --tokens 3072 --no-mtp
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_long_prefill_teacher.q35dump
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_long_prefill_states.bin)
set_tests_properties(qwen35_mixed_long_prefill_generate_teacher PROPERTIES DEPENDS qwen35_mixed_long_prefill_e2e)
add_test(NAME qwen35_mixed_long_prefill_test
  COMMAND qwen35_forward_test --checkpoint-dir ${QWEN35_MIXED_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen35_mixed_long_prefill_teacher.q35dump)
set_tests_properties(qwen35_mixed_long_prefill_test PROPERTIES DEPENDS qwen35_mixed_long_prefill_generate_teacher)
# The 27B TP forwards (#86): loopback worlds of 2 and 4 over the tiny
# fixture against the world-1 forward (ports 29950, 29951).
add_executable(qwen35_tp_test tests/cuda/qwen35_tp_test.cpp)
target_include_directories(qwen35_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen35_tp_test PRIVATE dgpp_models_qwen_layers dgpp_models_qwen_loader dgpp_models_qwen dgpp_bus dgpp_loaders dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen35_tp_test)
add_test(NAME qwen35_tp_test COMMAND qwen35_tp_test)
add_test(NAME qwen35_tp_test_draft_screen
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_DRAFT_SCREEN=2 $<TARGET_FILE:qwen35_tp_test>)
set_tests_properties(qwen35_tp_test_draft_screen PROPERTIES RUN_SERIAL TRUE LABELS "gpu" SKIP_RETURN_CODE 2)
set_tests_properties(qwen35_tp_test PROPERTIES RUN_SERIAL TRUE LABELS "gpu")

add_executable(qwen_dflash2_test tests/cuda/qwen_dflash2_test.cpp)
target_include_directories(qwen_dflash2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_dflash2_test PRIVATE dgpp_models_qwen_layers dgpp_models_qwen_loader dgpp_models_qwen dgpp_bus dgpp_loaders dgpp_kernels dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_dflash2_test)
add_test(NAME qwen_dflash2_test COMMAND qwen_dflash2_test)
set_tests_properties(qwen_dflash2_test PROPERTIES RUN_SERIAL TRUE LABELS "gpu")

add_executable(dflash2_integration_test tests/cuda/dflash2_integration_test.cpp)
target_include_directories(dflash2_integration_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dflash2_integration_test PRIVATE dgpp_models_qwen_layers dgpp_bus dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dflash2_integration_test)
add_test(NAME dflash2_integration_test COMMAND dflash2_integration_test)
set_tests_properties(dflash2_integration_test PROPERTIES RUN_SERIAL TRUE LABELS "gpu")

add_test(NAME spec_ab_probe_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/spec_ab_probe_test.py)

add_executable(qwen35_loader_smoke tests/cuda/qwen35_loader_smoke.cpp)
target_include_directories(qwen35_loader_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen35_loader_smoke PRIVATE dgpp_models_qwen_loader dgpp_models_qwen dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(qwen35_loader_smoke)

add_executable(mimo_attn_test tests/cuda/mimo_attn_test.cu)
target_include_directories(mimo_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_attn_test PRIVATE dgpp_models_mimo dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_attn_test)
add_test(NAME mimo_attn_test COMMAND mimo_attn_test)

add_executable(flint_attn_test tests/cuda/flint_attn_test.cu)
target_include_directories(flint_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(flint_attn_test PRIVATE dgpp_models_flint dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(flint_attn_test)
add_test(NAME flint_attn_test COMMAND flint_attn_test)

add_executable(add_rmsnorm_test tests/cuda/add_rmsnorm_test.cu)
target_include_directories(add_rmsnorm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(add_rmsnorm_test PRIVATE dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(add_rmsnorm_test)
add_test(NAME add_rmsnorm_test COMMAND add_rmsnorm_test)

add_executable(dory_norm_test tests/cuda/dory_norm_test.cu)
target_include_directories(dory_norm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dory_norm_test PRIVATE dgpp_kernels dgpp_models_dory CUDA::cudart Threads::Threads)
dgpp_set_common(dory_norm_test)
add_test(NAME dory_norm_test COMMAND dory_norm_test)

add_executable(dory_attn_test tests/cuda/dory_attn_test.cu)
target_include_directories(dory_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dory_attn_test PRIVATE dgpp_kernels dgpp_models_dory CUDA::cudart Threads::Threads)
dgpp_set_common(dory_attn_test)
add_test(NAME dory_attn_test COMMAND dory_attn_test)

add_executable(dory_forward_test tests/cuda/dory_forward_test.cpp)
target_include_directories(dory_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dory_forward_test PRIVATE dgpp_models_dory_layers dgpp_models_dory_loader dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(dory_forward_test)
add_test(NAME dory_forward_test COMMAND dory_forward_test)
# The fixture ladder (docs/adding_a_model.md §6): the tiny release-layout
# checkpoint from the binding table, the CUDA model over it, and every
# logits row against the CPU fp64 host reference (generate and decode).
set(DORY_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/dory_forward_fixture)
add_test(NAME dory_forward_fixture COMMAND dory_forward_test --write-fixture ${DORY_FORWARD_FIXTURE})
add_test(NAME dory_forward_smoke COMMAND dory_forward_test --smoke ${DORY_FORWARD_FIXTURE})
set_tests_properties(dory_forward_smoke PROPERTIES DEPENDS dory_forward_fixture LABELS "gpu" RUN_SERIAL TRUE)
add_test(NAME dory_forward_reference COMMAND dory_forward_test --reference ${DORY_FORWARD_FIXTURE})
set_tests_properties(dory_forward_reference PROPERTIES DEPENDS dory_forward_fixture LABELS "gpu" RUN_SERIAL TRUE)

add_executable(glm4_loader_test tests/cuda/glm4_loader_test.cpp)
target_include_directories(glm4_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm4_loader_test PRIVATE dgpp_models_glm4_loader CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_loader_test)
add_test(NAME glm4_loader_test COMMAND glm4_loader_test)

add_executable(mimo_loader_test tests/cuda/mimo_loader_test.cpp)
target_include_directories(mimo_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_loader_test PRIVATE dgpp_models_mimo_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_loader_test)
add_test(NAME mimo_loader_test COMMAND mimo_loader_test)

# The GLM-4.7 forward (docs/glm47_plan.md G4): fixture -> smoke; the python
# reference chain follows (the same chain as the Qwen forward's).
add_executable(glm4_forward_test tests/cuda/glm4_forward_test.cpp)
target_include_directories(glm4_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm4_forward_test PRIVATE dgpp_models_glm4_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_forward_test)
set(GLM4_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/glm4_forward_fixture)
add_test(NAME glm4_forward_fixture COMMAND glm4_forward_test --write-fixture ${GLM4_FORWARD_FIXTURE})
add_test(NAME glm4_forward_smoke COMMAND glm4_forward_test --smoke ${GLM4_FORWARD_FIXTURE})
set_tests_properties(glm4_forward_smoke PROPERTIES DEPENDS glm4_forward_fixture)
add_test(NAME glm4_forward_generate
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm4_reference_dump.py
          gen-pure --checkpoint-dir ${GLM4_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm4_pure_ref.g4dump)
set_tests_properties(glm4_forward_generate PROPERTIES DEPENDS glm4_forward_fixture)
add_test(NAME glm4_forward_test
  COMMAND glm4_forward_test --checkpoint-dir ${GLM4_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm4_pure_ref.g4dump)
set_tests_properties(glm4_forward_test PROPERTIES DEPENDS glm4_forward_generate)

# The MiMo-V2.6-Flash forward (docs/mimo_v26_flash_plan.md G3): fixture ->
# smoke; the numpy reference chain follows (the GLM-4.7 chain's shape). The
# decode / tp / engine gates read ${MIMO_FORWARD_FIXTURE}.
add_executable(mimo_forward_test tests/cuda/mimo_forward_test.cpp)
target_include_directories(mimo_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_forward_test PRIVATE dgpp_models_mimo_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_forward_test)
set(MIMO_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/mimo_forward_fixture)
add_test(NAME mimo_forward_fixture COMMAND mimo_forward_test --write-fixture ${MIMO_FORWARD_FIXTURE})
add_test(NAME mimo_forward_smoke COMMAND mimo_forward_test --smoke ${MIMO_FORWARD_FIXTURE})
set_tests_properties(mimo_forward_smoke PROPERTIES DEPENDS mimo_forward_fixture)
add_test(NAME mimo_forward_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/mimo_reference_dump.py
          gen-pure --checkpoint-dir ${MIMO_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/mimo_pure_ref.mdump)
set_tests_properties(mimo_forward_generate PROPERTIES DEPENDS mimo_forward_fixture)
# End to end under the relaxed budgets (a routing near tie on the random
# fixture may flip and cascade: the rows from the first flip on are
# reported, not judged), writing the engine's states; then the reference
# teacher-forced on them (every layer fed the engine's own input) under the
# strict budgets — the DeepSeek ladder's design.
add_test(NAME mimo_forward_test_e2e
  COMMAND ${CMAKE_COMMAND} -E env DGPP_MIMO_DUMP_ENGINE_STATES=${CMAKE_CURRENT_BINARY_DIR}/mimo_engine_states.bin
          $<TARGET_FILE:mimo_forward_test> --checkpoint-dir ${MIMO_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/mimo_pure_ref.mdump --relaxed)
set_tests_properties(mimo_forward_test_e2e PROPERTIES DEPENDS mimo_forward_generate)
add_test(NAME mimo_forward_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/mimo_reference_dump.py
          gen-pure --checkpoint-dir ${MIMO_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/mimo_teacher_ref.mdump
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/mimo_engine_states.bin)
set_tests_properties(mimo_forward_generate_teacher PROPERTIES DEPENDS mimo_forward_test_e2e)
add_test(NAME mimo_forward_test
  COMMAND mimo_forward_test --checkpoint-dir ${MIMO_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/mimo_teacher_ref.mdump)
set_tests_properties(mimo_forward_test PROPERTIES DEPENDS mimo_forward_generate_teacher)

# The Flint loader (docs/flint_plan.md G2) and forward (G5): fixture ->
# smoke; the python reference end to end (relaxed) -> teacher-forced (strict).
add_executable(flint_loader_test tests/cuda/flint_loader_test.cpp)
target_include_directories(flint_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(flint_loader_test PRIVATE dgpp_models_flint_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(flint_loader_test)
add_test(NAME flint_loader_test COMMAND flint_loader_test)

add_executable(flint_forward_test tests/cuda/flint_forward_test.cpp)
target_include_directories(flint_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(flint_forward_test PRIVATE dgpp_models_flint_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(flint_forward_test)
set(FLINT_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/flint_forward_fixture)
add_test(NAME flint_forward_fixture COMMAND flint_forward_test --write-fixture ${FLINT_FORWARD_FIXTURE})
add_test(NAME flint_forward_smoke COMMAND flint_forward_test --smoke ${FLINT_FORWARD_FIXTURE})
set_tests_properties(flint_forward_smoke PROPERTIES DEPENDS flint_forward_fixture)
add_test(NAME flint_forward_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/flint_reference_dump.py
          gen-pure --checkpoint-dir ${FLINT_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/flint_pure_ref.fdump)
set_tests_properties(flint_forward_generate PROPERTIES DEPENDS flint_forward_fixture)
add_test(NAME flint_forward_test_e2e
  COMMAND ${CMAKE_COMMAND} -E env DGPP_FLINT_DUMP_ENGINE_STATES=${CMAKE_CURRENT_BINARY_DIR}/flint_engine_states.bin
          $<TARGET_FILE:flint_forward_test> --checkpoint-dir ${FLINT_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/flint_pure_ref.fdump --relaxed)
set_tests_properties(flint_forward_test_e2e PROPERTIES DEPENDS flint_forward_generate)
add_test(NAME flint_forward_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/flint_reference_dump.py
          gen-pure --checkpoint-dir ${FLINT_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/flint_teacher_ref.fdump
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/flint_engine_states.bin)
set_tests_properties(flint_forward_generate_teacher PROPERTIES DEPENDS flint_forward_test_e2e)
add_test(NAME flint_forward_test
  COMMAND flint_forward_test --checkpoint-dir ${FLINT_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/flint_teacher_ref.fdump)
set_tests_properties(flint_forward_test PROPERTIES DEPENDS flint_forward_generate_teacher)

# The full GLM-5.3 forward (docs/glm53_plan.md G5): fixture -> smoke; the
# python reference -> the dump parity.
add_executable(glm_dsa_forward_test tests/cuda/glm_dsa_forward_test.cpp)
target_include_directories(glm_dsa_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm_dsa_forward_test PRIVATE dgpp_models_glm_dsa_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(glm_dsa_forward_test)
set(GLM_DSA_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_forward_fixture)
add_test(NAME glm_dsa_forward_fixture COMMAND glm_dsa_forward_test --write-fixture ${GLM_DSA_FORWARD_FIXTURE})
add_test(NAME glm_dsa_forward_smoke COMMAND glm_dsa_forward_test --smoke ${GLM_DSA_FORWARD_FIXTURE})
set_tests_properties(glm_dsa_forward_smoke PROPERTIES DEPENDS glm_dsa_forward_fixture)
add_test(NAME glm_dsa_forward_generate
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_dsa_reference_dump.py
          gen-pure --checkpoint-dir ${GLM_DSA_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref.gdsdump)
set_tests_properties(glm_dsa_forward_generate PROPERTIES DEPENDS glm_dsa_forward_fixture)
add_test(NAME glm_dsa_forward_test
  COMMAND glm_dsa_forward_test --checkpoint-dir ${GLM_DSA_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref.gdsdump)
set_tests_properties(glm_dsa_forward_test PROPERTIES DEPENDS glm_dsa_forward_generate)

# Bulk packed prefill versus the same independent FP64 oracle and budgets.
# At 160 rows the packed GEMM is active. Dense attention and all-expert
# routing isolate arithmetic from selection changes and audit every row.
set(GLM_DSA_PREFILL_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_prefill_fixture)
add_test(NAME glm_dsa_prefill_fixture COMMAND glm_dsa_forward_test --write-prefill-fixture ${GLM_DSA_PREFILL_FIXTURE})
set_tests_properties(glm_dsa_prefill_fixture PROPERTIES LABELS "gpu")
add_test(NAME glm_dsa_prefill_generate
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_dsa_reference_dump.py
          gen-pure --checkpoint-dir ${GLM_DSA_PREFILL_FIXTURE} --tokens 160
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_prefill_ref.gdsdump)
set_tests_properties(glm_dsa_prefill_generate PROPERTIES DEPENDS glm_dsa_prefill_fixture LABELS "fixture")
add_test(NAME glm_dsa_prefill_test
  COMMAND glm_dsa_forward_test --checkpoint-dir ${GLM_DSA_PREFILL_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_prefill_ref.gdsdump)
set_tests_properties(glm_dsa_prefill_test PROPERTIES DEPENDS glm_dsa_prefill_generate LABELS "gpu")

# The full GLM-5.3 session surface (G5): decode over the fixture.
add_executable(glm_dsa_decode_test tests/cuda/glm_dsa_decode_test.cpp)
target_include_directories(glm_dsa_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm_dsa_decode_test PRIVATE dgpp_models_glm_dsa_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(glm_dsa_decode_test)
add_test(NAME glm_dsa_decode_test COMMAND glm_dsa_decode_test --fixture ${GLM_DSA_FORWARD_FIXTURE})
set_tests_properties(glm_dsa_decode_test PROPERTIES DEPENDS glm_dsa_forward_fixture)
# The NF4I8 contract's fixture (codebook routed experts, H32-rotated): the
# forward smoke and the decode gates run on it too.
set(GLM_DSA_NF4I8_FIXTURE ${CMAKE_BINARY_DIR}/glm_dsa_forward_fixture_nf4i8)
add_test(NAME glm_dsa_forward_fixture_nf4i8 COMMAND glm_dsa_forward_test --write-fixture-nf4i8 ${GLM_DSA_NF4I8_FIXTURE})
add_test(NAME glm_dsa_forward_smoke_nf4i8 COMMAND glm_dsa_forward_test --smoke ${GLM_DSA_NF4I8_FIXTURE})
set_tests_properties(glm_dsa_forward_smoke_nf4i8 PROPERTIES DEPENDS glm_dsa_forward_fixture_nf4i8)
add_test(NAME glm_dsa_decode_test_nf4i8 COMMAND glm_dsa_decode_test --fixture ${GLM_DSA_NF4I8_FIXTURE})
set_tests_properties(glm_dsa_decode_test_nf4i8 PROPERTIES DEPENDS glm_dsa_forward_fixture_nf4i8)
# The pure-double reference over the nf4i8 fixture (the codebook decode and
# the rotated routed inputs in tools/glm_dsa_reference_dump.py) and the
# engine's parity with it.
add_test(NAME glm_dsa_forward_generate_nf4i8
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_dsa_reference_dump.py
          gen-pure --checkpoint-dir ${GLM_DSA_NF4I8_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref_nf4i8.gdsdump)
set_tests_properties(glm_dsa_forward_generate_nf4i8 PROPERTIES DEPENDS glm_dsa_forward_fixture_nf4i8)
add_test(NAME glm_dsa_forward_test_nf4i8
  COMMAND glm_dsa_forward_test --checkpoint-dir ${GLM_DSA_NF4I8_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref_nf4i8.gdsdump)
set_tests_properties(glm_dsa_forward_test_nf4i8 PROPERTIES DEPENDS glm_dsa_forward_generate_nf4i8)
# The Mixed346 contract's fixture (every expert form on every packed layer,
# int8 activation codes): the forward smoke, the decode gate and the
# pure-reference parity run on it too.
set(GLM_DSA_M346_FIXTURE ${CMAKE_BINARY_DIR}/glm_dsa_forward_fixture_m346)
add_test(NAME glm_dsa_forward_fixture_m346 COMMAND glm_dsa_forward_test --write-fixture-m346 ${GLM_DSA_M346_FIXTURE})
add_test(NAME glm_dsa_forward_smoke_m346 COMMAND glm_dsa_forward_test --smoke ${GLM_DSA_M346_FIXTURE})
set_tests_properties(glm_dsa_forward_smoke_m346 PROPERTIES DEPENDS glm_dsa_forward_fixture_m346)
add_test(NAME glm_dsa_decode_test_m346 COMMAND glm_dsa_decode_test --fixture ${GLM_DSA_M346_FIXTURE})
set_tests_properties(glm_dsa_decode_test_m346 PROPERTIES DEPENDS glm_dsa_forward_fixture_m346)
add_test(NAME glm_dsa_forward_generate_m346
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_dsa_reference_dump.py
          gen-pure --checkpoint-dir ${GLM_DSA_M346_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref_m346.gdsdump)
set_tests_properties(glm_dsa_forward_generate_m346 PROPERTIES DEPENDS glm_dsa_forward_fixture_m346)
add_test(NAME glm_dsa_forward_test_m346
  COMMAND glm_dsa_forward_test --checkpoint-dir ${GLM_DSA_M346_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref_m346.gdsdump)
set_tests_properties(glm_dsa_forward_test_m346 PROPERTIES DEPENDS glm_dsa_forward_generate_m346)
# The int4-attention candidate's fixture (attention projections at 4 bits in
# the compressed-tensors contract, int8 shared expert): the same gates.
set(GLM_DSA_ATTN4_FIXTURE ${CMAKE_BINARY_DIR}/glm_dsa_forward_fixture_attn4)
add_test(NAME glm_dsa_forward_fixture_attn4 COMMAND glm_dsa_forward_test --write-fixture-attn4 ${GLM_DSA_ATTN4_FIXTURE})
add_test(NAME glm_dsa_forward_smoke_attn4 COMMAND glm_dsa_forward_test --smoke ${GLM_DSA_ATTN4_FIXTURE})
set_tests_properties(glm_dsa_forward_smoke_attn4 PROPERTIES DEPENDS glm_dsa_forward_fixture_attn4)
# (No decode gate on this fixture: its random 4-bit attention leaves the
# indexer at near-uniform logits, 10 of 13 decode rows flip their selection
# and the kept rows sit downstream of the flips — the fixture lesson of
# docs/glm53_plan.md; the 4-bit decode path is gated on the real weights by
# the teacher-forced forward check under --attention-weights int4.)
add_test(NAME glm_dsa_forward_generate_attn4
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_dsa_reference_dump.py
          gen-pure --checkpoint-dir ${GLM_DSA_ATTN4_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref_attn4.gdsdump)
set_tests_properties(glm_dsa_forward_generate_attn4 PROPERTIES DEPENDS glm_dsa_forward_fixture_attn4)
add_test(NAME glm_dsa_forward_test_attn4
  COMMAND glm_dsa_forward_test --checkpoint-dir ${GLM_DSA_ATTN4_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm_dsa_pure_ref_attn4.gdsdump)
set_tests_properties(glm_dsa_forward_test_attn4 PROPERTIES DEPENDS glm_dsa_forward_generate_attn4)

# The full GLM-5.3 TP forwards: loopback worlds 2 and 4 (ports 29952/29953).
add_executable(glm_dsa_tp_test tests/cuda/glm_dsa_tp_test.cpp)
target_include_directories(glm_dsa_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm_dsa_tp_test PRIVATE dgpp_models_glm_dsa_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(glm_dsa_tp_test)
add_test(NAME glm_dsa_tp_test COMMAND glm_dsa_tp_test)

# The full GLM-5.3 decode engines: eager and graph, world 2 (ports 29954-29956).
add_executable(glm_dsa_engine_test tests/cuda/glm_dsa_engine_test.cpp)
target_include_directories(glm_dsa_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm_dsa_engine_test PRIVATE dgpp_models_glm_dsa_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(glm_dsa_engine_test)
add_test(NAME glm_dsa_engine_test COMMAND glm_dsa_engine_test)
add_test(NAME glm_dsa_engine_test_draft_screen
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_DRAFT_SCREEN=2 $<TARGET_FILE:glm_dsa_engine_test>)
set_tests_properties(glm_dsa_engine_test_draft_screen PROPERTIES RUN_SERIAL TRUE LABELS "gpu" SKIP_RETURN_CODE 2)
# The wide gate at eight slots x depth 2 (24 rows; the cap is 32 since 2026-10-13).
add_test(NAME glm_dsa_engine_test_twenty_four_rows
  COMMAND ${CMAKE_COMMAND} -E env DGPP_WIDE_SLOTS=8 DGPP_WIDE_DEPTH=2 $<TARGET_FILE:glm_dsa_engine_test>)
set_tests_properties(glm_dsa_engine_test_twenty_four_rows PROPERTIES RUN_SERIAL TRUE LABELS "gpu" SKIP_RETURN_CODE 2)

add_executable(glm_dsa_forward_check apps/glm_dsa_forward_check.cpp)
target_link_libraries(glm_dsa_forward_check PRIVATE dgpp_models_glm_dsa_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(glm_dsa_forward_check)

# The GLM-4.7 session surface: incremental decode vs the re-forward, the
# prefix snapshots, the eager speculator through the draft block.
add_executable(glm4_decode_test tests/cuda/glm4_decode_test.cpp)
target_include_directories(glm4_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm4_decode_test PRIVATE dgpp_models_glm4_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_decode_test)
add_test(NAME glm4_decode_test COMMAND glm4_decode_test --fixture ${GLM4_FORWARD_FIXTURE})
set_tests_properties(glm4_decode_test PROPERTIES DEPENDS glm4_forward_fixture)

# The GLM-4.7 TP forwards: loopback worlds 2 and 4 vs world 1 (ports 29946, 29947).
add_executable(glm4_tp_test tests/cuda/glm4_tp_test.cpp)
target_include_directories(glm4_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm4_tp_test PRIVATE dgpp_models_glm4_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_tp_test)
add_test(NAME glm4_tp_test COMMAND glm4_tp_test)

# The GLM-4.7 engines: the graph engine's scalar, row-batched and MTP
# replays against the eager engine over a loopback world of 2 (ports 29948, 29949).
add_executable(glm4_engine_test tests/cuda/glm4_engine_test.cpp)
target_include_directories(glm4_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm4_engine_test PRIVATE dgpp_models_glm4_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_engine_test)
add_test(NAME glm4_engine_test COMMAND glm4_engine_test)
add_test(NAME glm4_engine_test_draft_screen
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_DRAFT_SCREEN=2 $<TARGET_FILE:glm4_engine_test>)
set_tests_properties(glm4_engine_test_draft_screen PROPERTIES RUN_SERIAL TRUE LABELS "gpu" SKIP_RETURN_CODE 2)

# ---- MiMo-V2.6-Flash: the session, TP and engine gates (2026-09-22) ---------
# The fixture the forward ladder writes (the same directory when that
# ladder is registered; the gates write their own copy when it is not).
if(NOT DEFINED MIMO_FORWARD_FIXTURE)
  set(MIMO_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/mimo_forward_fixture)
endif()
# The MiMo session surface: incremental decode vs the re-forward, the verify
# rows vs the scalar steps, the prefix snapshots, the eager speculator
# through the draft block, the sliding window through the paged cache.
add_executable(mimo_decode_test tests/cuda/mimo_decode_test.cpp)
target_include_directories(mimo_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_decode_test PRIVATE dgpp_models_mimo_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_decode_test)
add_test(NAME mimo_decode_test COMMAND mimo_decode_test --fixture ${MIMO_FORWARD_FIXTURE})
if(TEST mimo_forward_fixture)
  set_tests_properties(mimo_decode_test PROPERTIES DEPENDS mimo_forward_fixture)
endif()

# The Flint session surface (docs/flint_plan.md G4), the TP forwards on
# loopback worlds 2 and 4 (G5; ports 29964, 29965) and the engines on a
# loopback world of 2 (G6; port 29969).
add_executable(flint_decode_test tests/cuda/flint_decode_test.cpp)
target_include_directories(flint_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(flint_decode_test PRIVATE dgpp_models_flint_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(flint_decode_test)
add_test(NAME flint_decode_test COMMAND flint_decode_test --fixture ${FLINT_FORWARD_FIXTURE})
if(TEST flint_forward_fixture)
  set_tests_properties(flint_decode_test PROPERTIES DEPENDS flint_forward_fixture)
endif()
add_executable(flint_tp_test tests/cuda/flint_tp_test.cpp)
target_include_directories(flint_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(flint_tp_test PRIVATE dgpp_models_flint_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(flint_tp_test)
add_test(NAME flint_tp_test COMMAND flint_tp_test)
add_executable(flint_engine_test tests/cuda/flint_engine_test.cpp)
target_include_directories(flint_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(flint_engine_test PRIVATE dgpp_models_flint_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(flint_engine_test)
add_test(NAME flint_engine_test COMMAND flint_engine_test)

# The MiMo TP forwards: loopback worlds 2 and 4 vs world 1 (ports 29962, 29963).
add_executable(mimo_tp_test tests/cuda/mimo_tp_test.cpp)
target_include_directories(mimo_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_tp_test PRIVATE dgpp_models_mimo_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_tp_test)
add_test(NAME mimo_tp_test COMMAND mimo_tp_test)

# The MiMo engines: the graph engine's scalar, row-batched, MTP (depth 1
# and 2) and scheduled-depth replays against the eager engine over a
# loopback world of 2 (ports 29908, 29909, 29911, 29913).
add_executable(mimo_engine_test tests/cuda/mimo_engine_test.cpp)
target_include_directories(mimo_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_engine_test PRIVATE dgpp_models_mimo_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_engine_test)
add_test(NAME mimo_engine_test COMMAND mimo_engine_test)
add_test(NAME mimo_engine_test_draft_screen
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_DRAFT_SCREEN=2 $<TARGET_FILE:mimo_engine_test>)
set_tests_properties(mimo_engine_test_draft_screen PROPERTIES RUN_SERIAL TRUE LABELS "gpu" SKIP_RETURN_CODE 2)
# (the gpu;rdma labels and RUN_SERIAL join the loopback gates' list below)

add_executable(glm4_load_check apps/glm4_load_check.cpp)
target_link_libraries(glm4_load_check PRIVATE dgpp_models_glm4_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_load_check)

add_executable(mimo_load_check apps/mimo_load_check.cpp)
target_link_libraries(mimo_load_check PRIVATE dgpp_models_mimo_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_load_check)

add_executable(flint_load_check apps/flint_load_check.cpp)
target_link_libraries(flint_load_check PRIVATE dgpp_models_flint_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(flint_load_check)

add_executable(glm_dsa_loader_test tests/cuda/glm_dsa_loader_test.cpp)
target_include_directories(glm_dsa_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm_dsa_loader_test PRIVATE dgpp_models_glm_dsa_loader CUDA::cudart Threads::Threads)
dgpp_set_common(glm_dsa_loader_test)
add_test(NAME glm_dsa_loader_test COMMAND glm_dsa_loader_test)

add_executable(glm_dsa_load_check apps/glm_dsa_load_check.cpp)
target_link_libraries(glm_dsa_load_check PRIVATE dgpp_models_glm_dsa_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(glm_dsa_load_check)

# The DeepSeek-V4.1-Flash loader on its synthetic fixture (worlds 1/2/4
# byte-exact, the globals, the Engram table mappings, the image round trip).
add_executable(dsv41_loader_test tests/cuda/dsv41_loader_test.cpp)
target_include_directories(dsv41_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_loader_test PRIVATE dgpp_models_dsv41_loader dgpp_step_dsv41 CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_loader_test)
add_test(NAME dsv41_loader_test COMMAND dsv41_loader_test)

add_executable(dsv41_load_check apps/dsv41_load_check.cpp)
target_link_libraries(dsv41_load_check PRIVATE dgpp_models_dsv41_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_load_check)
# The per-layer dump on the real checkpoint for the torch cross-check.
add_executable(dsv41_forward_check apps/dsv41_forward_check.cpp)
target_link_libraries(dsv41_forward_check PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_forward_check)

# The DeepSeek-V4.1 TP forwards: loopback worlds 2 and 4 vs world 1 (ports 29958/29959).
add_executable(dsv41_tp_test tests/cuda/dsv41_tp_test.cpp)
target_include_directories(dsv41_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_tp_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_tp_test)
add_test(NAME dsv41_tp_test COMMAND dsv41_tp_test)
add_test(NAME dsv41_tp_test_exl3 COMMAND ${DSV41_EXL3_ENV} $<TARGET_FILE:dsv41_tp_test> WORKING_DIRECTORY ${DSV41_EXL3_WORKDIR})
set_tests_properties(dsv41_tp_test_exl3 PROPERTIES RUN_SERIAL TRUE)  # the same loopback ports as dsv41_tp_test

# The DeepSeek-V4.1 decode engines: the graph engine's scalar and row-batched
# replays (the Engram host node inside the walk) against the eager engine
# over a loopback world of 2 (ports 29961-29964: plain, the DSpark block, depth 2, the scheduled depth).
add_executable(dsv41_engine_test tests/cuda/dsv41_engine_test.cpp)
target_include_directories(dsv41_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_engine_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_engine_test)
add_test(NAME dsv41_engine_test COMMAND dsv41_engine_test)
add_test(NAME dsv41_engine_test_exl3 COMMAND ${DSV41_EXL3_ENV} $<TARGET_FILE:dsv41_engine_test> WORKING_DIRECTORY ${DSV41_EXL3_WORKDIR})
set_tests_properties(dsv41_engine_test_exl3 PROPERTIES RUN_SERIAL TRUE)  # the same loopback ports as dsv41_engine_test
add_test(NAME dsv41_engine_test_draft_screen
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_DRAFT_SCREEN=2 $<TARGET_FILE:dsv41_engine_test>)
set_tests_properties(dsv41_engine_test_draft_screen PROPERTIES RUN_SERIAL TRUE LABELS "gpu" SKIP_RETURN_CODE 2)

# The DeepSeek-V4-Flash loader on its synthetic fixture (worlds 1/2/4
# byte-exact, the globals, the image round trip).
add_executable(dsv4_loader_test tests/cuda/dsv4_loader_test.cpp)
target_include_directories(dsv4_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv4_loader_test PRIVATE dgpp_models_dsv4_loader CUDA::cudart Threads::Threads)
dgpp_set_common(dsv4_loader_test)
add_test(NAME dsv4_loader_test COMMAND dsv4_loader_test)

# The DeepSeek-V4 session surface at world 1: prefill == forward, decode vs
# the re-forward, verify + rollback, interleaved slots, the chunked prefill,
# the prefix snapshots, the group prefill, the eager DSpark draft.
add_executable(dsv4_model_test tests/cuda/dsv4_model_test.cpp)
target_include_directories(dsv4_model_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv4_model_test PRIVATE dgpp_models_dsv4_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv4_model_test)
add_test(NAME dsv4_model_test COMMAND dsv4_model_test)

# The DeepSeek-V4 TP forwards: loopback worlds 2 and 4 vs world 1 (ports
# 29991-29998, 30003, 30004).
add_executable(dsv4_tp_test tests/cuda/dsv4_tp_test.cpp)
target_include_directories(dsv4_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv4_tp_test PRIVATE dgpp_models_dsv4_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(dsv4_tp_test)
add_test(NAME dsv4_tp_test COMMAND dsv4_tp_test)

# The DeepSeek-V4 decode engines: the graph engine's scalar and row-batched
# replays against the eager engine over a loopback world of 2 (ports
# 29986-29990: plain, the DSpark block, depth 2, the scheduled depth, six slots).
add_executable(dsv4_engine_test tests/cuda/dsv4_engine_test.cpp)
target_include_directories(dsv4_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv4_engine_test PRIVATE dgpp_models_dsv4_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv4_engine_test)
add_test(NAME dsv4_engine_test COMMAND dsv4_engine_test)

add_executable(glm4_forward_check apps/glm4_forward_check.cpp)
target_link_libraries(glm4_forward_check PRIVATE dgpp_models_glm4_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_forward_check)

add_executable(mimo_forward_check apps/mimo_forward_check.cpp)
target_link_libraries(mimo_forward_check PRIVATE dgpp_models_mimo_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_forward_check)

add_executable(flint_forward_check apps/flint_forward_check.cpp)
target_link_libraries(flint_forward_check PRIVATE dgpp_models_flint_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(flint_forward_check)

add_executable(qwen_load_check apps/qwen_load_check.cpp)
target_link_libraries(qwen_load_check PRIVATE dgpp_models_qwen_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_load_check)

add_executable(qwen_tp_test tests/cuda/qwen_tp_test.cpp)
target_include_directories(qwen_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_tp_test PRIVATE dgpp_models_qwen_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_tp_test)
add_test(NAME qwen_tp_test COMMAND qwen_tp_test)

# The Qwen engines (Q6 stage B): the graph engine's scalar and row-batched
# replays against the eager engine over a loopback world of 2 (port 29944).
add_executable(qwen_engine_test tests/cuda/qwen_engine_test.cpp)
target_include_directories(qwen_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_engine_test PRIVATE dgpp_models_qwen_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_engine_test)
add_test(NAME qwen_engine_test COMMAND qwen_engine_test)
add_test(NAME qwen_engine_row_independent
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 DGPP_TEST_FILTER=wide
          $<TARGET_FILE:qwen_engine_test>)
# The graph engines with the bf16 decode weights' 12-bit companions on
# (engine.bf16_weights; DGPP_BF12 overrides the process setting): the packed
# multi launch and the packed GEMVs under capture and replay, against the
# same references — the form is bitwise, so every gate holds unchanged.
add_test(NAME qwen_engine_bf12
  COMMAND ${CMAKE_COMMAND} -E env DGPP_BF12=both $<TARGET_FILE:qwen_engine_test>)
# The draft head's screen (engine.draft_head screen; kernels/head_screen.hpp)
# through the loopback engines: the vocab-sharded head's masked rows at
# world 2 and the pick merge behind them, every gate unchanged (the verify
# reads the exact head; only the drafts can move).
add_test(NAME qwen_engine_draft_screen
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_DRAFT_SCREEN=2 $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_draft_screen PROPERTIES RESOURCE_LOCK qwen_engine_world RUN_SERIAL TRUE)
foreach(_rows 4 256)
  add_test(NAME qwen_engine_fp8_head_${_rows}
    COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=${_rows}
            DGPP_TEST_DENSE_FP8=1 DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
  set_tests_properties(qwen_engine_fp8_head_${_rows}
    PROPERTIES RESOURCE_LOCK qwen_engine_world RUN_SERIAL TRUE)
endforeach()
# The hybrid through the loopback engines (MTP == plain, the mmap'ed table,
# depth 2, wide slots, graph == eager) on its own fixture.
add_test(NAME qwen_engine_gptq
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_GPTQ=1 DGPP_TEST_DENSE_FP8=1 DGPP_TEST_FILTER=loopback_world_2
          $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_gptq PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)
foreach(_dense_fp8 IN ITEMS 0 1)
  add_test(NAME qwen_engine_nvfp4_dense_${_dense_fp8}
    COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_NVFP4=1 DGPP_TEST_DENSE_FP8=${_dense_fp8}
            DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
  set_tests_properties(qwen_engine_nvfp4_dense_${_dense_fp8}
    PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)
endforeach()
foreach(_filter IN ITEMS wide mixed_prefill)
  add_test(NAME qwen_engine_fp4local_${_filter}
    COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_FP4LOCAL=1 DGPP_TEST_DENSE_FP8=1
            DGPP_TEST_FILTER=${_filter} $<TARGET_FILE:qwen_engine_test>)
  set_tests_properties(qwen_engine_fp4local_${_filter}
    PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)
endforeach()
foreach(_filter IN ITEMS wide mixed_prefill)
  add_test(NAME qwen_engine_fp4local_${_filter}_tile
    COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_QWEN_FP4LOCAL=1 DGPP_TEST_DENSE_FP8=1
            DGPP_TEST_QWEN_PREFILL_TILE=1 DGPP_TEST_FILTER=${_filter} $<TARGET_FILE:qwen_engine_test>)
  set_tests_properties(qwen_engine_fp4local_${_filter}_tile
    PROPERTIES RESOURCE_LOCK qwen_engine_world RUN_SERIAL TRUE)
endforeach()
set_tests_properties(qwen_engine_test qwen_engine_row_independent qwen_engine_bf12 PROPERTIES RESOURCE_LOCK qwen_engine_world)
set_tests_properties(qwen_engine_row_independent qwen_engine_bf12 PROPERTIES LABELS "gpu;rdma" RUN_SERIAL TRUE)
add_test(NAME qwen_engine_fp8_wide
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=4 DGPP_TEST_DENSE_FP8=1 DGPP_TEST_FILTER=wide
          $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_fp8_wide PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)
add_test(NAME qwen_engine_fp8_mtp3_rows32
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=4 DGPP_TEST_DENSE_FP8=1
          DGPP_TEST_QWEN_ROWS32=1 DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
add_test(NAME qwen_engine_mtp3_rows32_row_independent
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 DGPP_TEST_DENSE_FP8=1
          DGPP_TEST_QWEN_ROWS32=1 DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_fp8_mtp3_rows32 qwen_engine_mtp3_rows32_row_independent
  PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)

add_test(NAME qwen_engine_fp8_mtp3_rows64
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=4 DGPP_TEST_DENSE_FP8=1
          DGPP_TEST_QWEN_ROWS64=1 DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
add_test(NAME qwen_engine_mtp3_rows64_row_independent
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 DGPP_TEST_DENSE_FP8=1
          DGPP_TEST_QWEN_ROWS64=1 DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_fp8_mtp3_rows64 qwen_engine_mtp3_rows64_row_independent
  PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)

add_test(NAME qwen_engine_compaction_disabled
  COMMAND ${CMAKE_COMMAND} -E env DGPP_TEST_COMPACT_BATCH=0 $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_compaction_disabled
  PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)

add_test(NAME qwen_engine_mixed_prefill_without_mtp
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 DGPP_TEST_FILTER=mixed_prefill_without_mtp
          $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_mixed_prefill_without_mtp
  PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)
add_test(NAME qwen_engine_mixed_prefill_scheduled_verify
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 DGPP_TEST_FILTER=mixed_prefill_scheduled_verify
          $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_mixed_prefill_scheduled_verify
  PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)

foreach(compact_candidates IN ITEMS 1 32)
  add_test(NAME qwen_engine_compact_sampling_${compact_candidates}
    COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 DGPP_TEST_COMPACT_BATCH=1
            DGPP_TEST_FILTER=wide DGPP_TEST_COMPACT_SAMPLING=${compact_candidates}
            $<TARGET_FILE:qwen_engine_test>)
  set_tests_properties(qwen_engine_compact_sampling_${compact_candidates}
    PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)
endforeach()

foreach(compact_policy IN ITEMS 0 1)
  add_test(NAME qwen_engine_compact_fallback_${compact_policy}
    COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256
            DGPP_TEST_COMPACT_BATCH=${compact_policy} DGPP_TEST_COMPACT_FALLBACK=1
            DGPP_TEST_COMPACT_SAMPLING=1 DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
  set_tests_properties(qwen_engine_compact_fallback_${compact_policy}
    PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)
endforeach()
add_test(NAME qwen_engine_compact_fallback_mtp3_rows64
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 DGPP_TEST_COMPACT_BATCH=1
          DGPP_TEST_COMPACT_FALLBACK=1 DGPP_TEST_COMPACT_SAMPLING=1
          DGPP_TEST_QWEN_ROWS64=1 DGPP_TEST_FILTER=wide $<TARGET_FILE:qwen_engine_test>)
set_tests_properties(qwen_engine_compact_fallback_mtp3_rows64
  PROPERTIES RESOURCE_LOCK qwen_engine_world LABELS "gpu;rdma" RUN_SERIAL TRUE)

add_executable(qwen_forward_check apps/qwen_forward_check.cpp)
target_link_libraries(qwen_forward_check PRIVATE dgpp_models_qwen_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_forward_check)

add_executable(qwen_head_check apps/qwen_head_check.cpp)
target_link_libraries(qwen_head_check PRIVATE dgpp_models_qwen_layers dgpp_bus dgpp_loaders dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_head_check)
add_test(NAME qwen_head_check_fixture
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/qwen_head_check_test.py
          $<TARGET_FILE:qwen_head_check> ${QWEN_FORWARD_FIXTURE})
set_tests_properties(qwen_head_check_fixture PROPERTIES DEPENDS qwen_forward_fixture LABELS "gpu" RUN_SERIAL TRUE)
add_test(NAME qwen_compact_check_fixture
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/qwen_head_check_test.py
          $<TARGET_FILE:qwen_head_check> ${QWEN_FORWARD_FIXTURE} --compaction)
set_tests_properties(qwen_compact_check_fixture PROPERTIES DEPENDS qwen_forward_fixture LABELS "gpu" RUN_SERIAL TRUE)

add_executable(glm_loader_test tests/cuda/glm_loader_test.cpp)
target_include_directories(glm_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_loader_test PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_loader_test)
add_test(NAME glm_loader_test COMMAND glm_loader_test)
set_tests_properties(glm_loader_test PROPERTIES SKIP_RETURN_CODE 2)

# Scale-aware GEMM parity (M4): block edges, saturation, NaN policy;
# --checkpoint-dir adds real-checkpoint slices (manual deployment test —
# the checkpoint-runner TU is host-only, minijson does not mix with nvcc).
add_executable(bf16_gemv_test tests/cuda/bf16_gemv_test.cu)
target_include_directories(bf16_gemv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bf16_gemv_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(bf16_gemv_test)
add_test(NAME bf16_gemv_test COMMAND bf16_gemv_test)
set_tests_properties(bf16_gemv_test PROPERTIES SKIP_RETURN_CODE 2)
add_executable(bf12_gemv_test tests/cuda/bf12_gemv_test.cu)
target_include_directories(bf12_gemv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bf12_gemv_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(bf12_gemv_test)
add_test(NAME bf12_gemv_test COMMAND bf12_gemv_test)
set_tests_properties(bf12_gemv_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(qwen_mtp_test tests/cuda/qwen_mtp_test.cu)
target_include_directories(qwen_mtp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen_mtp_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(qwen_mtp_test)
add_test(NAME qwen_mtp_test COMMAND qwen_mtp_test)
set_tests_properties(qwen_mtp_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(dsv41_dspark_test tests/cuda/dsv41_dspark_test.cu)
target_include_directories(dsv41_dspark_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dsv41_dspark_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(dsv41_dspark_test)
add_test(NAME dsv41_dspark_test COMMAND dsv41_dspark_test)
set_tests_properties(dsv41_dspark_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(mma_gemv_test tests/cuda/mma_gemv_test.cu)
target_include_directories(mma_gemv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mma_gemv_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(mma_gemv_test)
add_test(NAME mma_gemv_test COMMAND mma_gemv_test)
add_executable(fp4w_gemm_test tests/cuda/fp4w_gemm_test.cu)
target_include_directories(fp4w_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fp4w_gemm_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(fp4w_gemm_test)
add_test(NAME fp4w_gemm_test COMMAND fp4w_gemm_test)

add_executable(qwen35_fp4_prefill_test tests/cuda/qwen35_fp4_prefill_test.cu)
target_include_directories(qwen35_fp4_prefill_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qwen35_fp4_prefill_test PRIVATE dgpp_models_qwen_layers dgpp_kernels CUDA::cudart)
dgpp_set_common(qwen35_fp4_prefill_test)
add_test(NAME qwen35_fp4_prefill_test COMMAND qwen35_fp4_prefill_test)
add_executable(fp4_gemv_test tests/cuda/fp4_gemv_test.cu tests/cuda/fp4_gemv_checkpoint.cpp)
target_include_directories(fp4_gemv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fp4_gemv_test PRIVATE dgpp_kernels dgpp_loaders CUDA::cudart)
dgpp_set_common(fp4_gemv_test)
add_test(NAME fp4_gemv_test COMMAND fp4_gemv_test)
set_tests_properties(fp4_gemv_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(packq_gemv_test tests/cuda/packq_gemv_test.cu tests/cuda/packq_gemv_checkpoint.cpp)
target_include_directories(packq_gemv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(packq_gemv_test PRIVATE dgpp_kernels dgpp_loaders CUDA::cudart)
dgpp_set_common(packq_gemv_test)
add_test(NAME packq_gemv_test COMMAND packq_gemv_test)
set_tests_properties(packq_gemv_test PROPERTIES SKIP_RETURN_CODE 2)
add_executable(exl3_gemv_test tests/cuda/exl3_gemv_test.cu)
target_include_directories(exl3_gemv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(exl3_gemv_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(exl3_gemv_test)
add_test(NAME exl3_gemv_test COMMAND exl3_gemv_test)
set_tests_properties(exl3_gemv_test PROPERTIES SKIP_RETURN_CODE 2)
add_executable(head_screen_test tests/cuda/head_screen_test.cu)
target_include_directories(head_screen_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(head_screen_test PRIVATE dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(head_screen_test)
add_test(NAME head_screen_test COMMAND head_screen_test)
set_tests_properties(head_screen_test PROPERTIES SKIP_RETURN_CODE 2)
add_executable(packq_head_test tests/cuda/packq_head_test.cu)
target_include_directories(packq_head_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(packq_head_test PRIVATE dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(packq_head_test)
add_test(NAME packq_head_test COMMAND packq_head_test)
set_tests_properties(packq_head_test PROPERTIES SKIP_RETURN_CODE 2)
add_executable(packq_gemm_test tests/cuda/packq_gemm_test.cu)
target_include_directories(packq_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(packq_gemm_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(packq_gemm_test)
add_test(NAME packq_gemm_test COMMAND packq_gemm_test)
set_tests_properties(packq_gemm_test PROPERTIES SKIP_RETURN_CODE 2 LABELS "gpu")

add_executable(fp8_gemm_test tests/cuda/fp8_gemm_test.cu)
target_include_directories(fp8_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fp8_gemm_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(fp8_gemm_test)
add_test(NAME fp8_gemm_test COMMAND fp8_gemm_test)
add_executable(fp8w_gemm_test tests/cuda/fp8w_gemm_test.cu)
target_include_directories(fp8w_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fp8w_gemm_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(fp8w_gemm_test)
add_test(NAME fp8w_gemm_test COMMAND fp8w_gemm_test)
set_tests_properties(fp8_gemm_test PROPERTIES SKIP_RETURN_CODE 2 LABELS "gpu")

add_executable(scale_gemm_test
  tests/cuda/scale_gemm_test.cu
  tests/cuda/scale_gemm_checkpoint.cpp)
target_include_directories(scale_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scale_gemm_test PRIVATE dgpp_kernels dgpp_loaders CUDA::cudart)
dgpp_set_common(scale_gemm_test)
add_test(NAME scale_gemm_test COMMAND scale_gemm_test)
set_tests_properties(scale_gemm_test PROPERTIES SKIP_RETURN_CODE 2)

# mHC module parity (M4): compute/update/final-mean kernels vs the double
# oracle; real geometry, edge cases, determinism.
add_executable(glm_mhc_test tests/cuda/glm_mhc_test.cu)
target_include_directories(glm_mhc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_mhc_test PRIVATE dgpp_kernels dgpp_models_glm CUDA::cudart)
dgpp_set_common(glm_mhc_test)
add_test(NAME glm_mhc_test COMMAND glm_mhc_test)
set_tests_properties(glm_mhc_test PROPERTIES SKIP_RETURN_CODE 2)

# The on-device greedy pick (DESIGN §9): local top-2 vs glm_sample::local_max,
# verdict vs merge_greedy/judge_verify over a simulated world, digest check.
add_executable(det_math_test tests/cuda/det_math_test.cu)
target_include_directories(det_math_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(det_math_test PRIVATE CUDA::cudart)
dgpp_set_common(det_math_test)
add_test(NAME det_math_test COMMAND det_math_test)
set_tests_properties(det_math_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(glm_pick_test tests/cuda/glm_pick_test.cpp)
target_include_directories(glm_pick_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_pick_test PRIVATE dgpp_kernels dgpp_models_glm CUDA::cudart)
dgpp_set_common(glm_pick_test)
add_test(NAME glm_pick_test COMMAND glm_pick_test)
set_tests_properties(glm_pick_test PROPERTIES SKIP_RETURN_CODE 2)

# MoE module parity (M4): router vs oracle at real geometry, expert path on
# small geometry, swiglu clamps, accumulation order, determinism.
add_executable(moe_w4a4_test tests/cuda/moe_w4a4_test.cu)
target_include_directories(moe_w4a4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(moe_w4a4_test PRIVATE dgpp_models_glm_loader dgpp_models_glm CUDA::cudart)
dgpp_set_common(moe_w4a4_test)
add_test(NAME moe_w4a4_test COMMAND moe_w4a4_test --tokens 1024 --experts 64)
set_tests_properties(moe_w4a4_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(gdn_chunk_test tests/cuda/gdn_chunk_test.cu)
target_include_directories(gdn_chunk_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gdn_chunk_test PRIVATE dgpp_models_kda CUDA::cudart Threads::Threads)
dgpp_set_common(gdn_chunk_test)
add_test(NAME gdn_chunk_test COMMAND gdn_chunk_test)

add_executable(glm_moe_test tests/cuda/glm_moe_test.cu)
target_include_directories(glm_moe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_moe_test PRIVATE dgpp_models_glm_loader dgpp_models_glm CUDA::cudart)
dgpp_set_common(glm_moe_test)
add_test(NAME glm_moe_test COMMAND glm_moe_test)
set_tests_properties(glm_moe_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME moe_w4a4_layer_test COMMAND ${CMAKE_COMMAND} -E env
  DGPP_MOE_W4A4=1 --unset=DGPP_MOE_W4A4_F32_DOWN --unset=DGPP_MOE_W4A4_DYNAMIC
  DGPP_MOE_W4A4_MIN_ROWS=256 DGPP_MOE_SWIGLU_QUANT=1
  DGPP_TEST_FILTER=moe_w4a4_host_and_device_segmentation_match_both_output_formats
  $<TARGET_FILE:glm_moe_test>)
set_tests_properties(moe_w4a4_layer_test PROPERTIES SKIP_RETURN_CODE 2)
foreach(_mode IN ITEMS default 0 1)
  if(_mode STREQUAL "default")
    set(_w4a4_setting --unset=DGPP_MOE_W4A4)
  else()
    set(_w4a4_setting DGPP_MOE_W4A4=${_mode})
  endif()
  add_test(NAME moe_w4a4_policy_${_mode} COMMAND ${CMAKE_COMMAND} -E env
    ${_w4a4_setting} --unset=DGPP_MOE_W4A4_DYNAMIC DGPP_MOE_W4A4_MIN_ROWS=256
    DGPP_TEST_FILTER=moe_nvfp4_activation_scale_requires_opt_in $<TARGET_FILE:glm_moe_test>)
  set_tests_properties(moe_w4a4_policy_${_mode} PROPERTIES LABELS "gpu" RUN_SERIAL TRUE SKIP_RETURN_CODE 2)
endforeach()
add_test(NAME moe_w4a4_policy_dynamic COMMAND ${CMAKE_COMMAND} -E env
  DGPP_MOE_W4A4=1 DGPP_MOE_W4A4_DYNAMIC=1 DGPP_MOE_W4A4_MIN_ROWS=256
  DGPP_TEST_FILTER=moe_nvfp4_activation_scale_requires_opt_in $<TARGET_FILE:glm_moe_test>)
set_tests_properties(moe_w4a4_policy_dynamic PROPERTIES LABELS "gpu" RUN_SERIAL TRUE SKIP_RETURN_CODE 2)

# Full-model forward parity (M4 chunk 6): this test writes a synthetic
# mini-checkpoint, python computes the full-stack pure reference from the
# same checkpoint (one tensor-table source: the C++ binding), and the test
# runs the diagnostic forward against the dump. The torch backend on the
# real checkpoint is the manual deployment run (glm_forward_check).
add_executable(glm_forward_test tests/cuda/glm_forward_test.cpp)
target_link_libraries(glm_forward_test
  PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_forward_test)
set(GLM_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/glm_forward_fixture)
add_test(NAME glm_forward_fixture
  COMMAND glm_forward_test --write-fixture ${GLM_FORWARD_FIXTURE})
set_tests_properties(glm_forward_fixture PROPERTIES SKIP_RETURN_CODE 2)
add_executable(nvme_tier_families_test tests/cuda/nvme_tier_families_test.cpp)
target_include_directories(nvme_tier_families_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(nvme_tier_families_test PRIVATE dgpp_models_dory_layers dgpp_models_mimo_layers dgpp_models_flint_layers dgpp_models_qwen_layers dgpp_models_qwen_loader dgpp_models_qwen dgpp_models_glm_loader dgpp_kernels CUDA::cudart Threads::Threads)
dgpp_set_common(nvme_tier_families_test)
add_test(NAME nvme_tier_families_test COMMAND nvme_tier_families_test)
set_tests_properties(nvme_tier_families_test PROPERTIES RUN_SERIAL TRUE)
add_executable(prefix_snapshot_test tests/cuda/prefix_snapshot_test.cpp)
target_include_directories(prefix_snapshot_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(prefix_snapshot_test PRIVATE dgpp_models_qwen_layers dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(prefix_snapshot_test)
add_test(NAME prefix_snapshot_test
  COMMAND prefix_snapshot_test ${QWEN_FORWARD_FIXTURE} ${GLM_FORWARD_FIXTURE})
set_tests_properties(prefix_snapshot_test PROPERTIES DEPENDS "qwen_forward_fixture;glm_forward_fixture" RUN_SERIAL TRUE)
add_test(NAME glm_forward_generate
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_reference_dump.py
          gen-pure --checkpoint-dir ${GLM_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm_pure_ref.glmdump)
set_tests_properties(glm_forward_generate PROPERTIES
  DEPENDS glm_forward_fixture)
add_test(NAME glm_forward_test
  COMMAND glm_forward_test --checkpoint-dir ${GLM_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm_pure_ref.glmdump)
set_tests_properties(glm_forward_test PROPERTIES
  DEPENDS glm_forward_generate SKIP_RETURN_CODE 2)

# DSA reference-dump parity: generate a pure-python oracle dump, then run
# the layer against it (gen-torch is the manual deployment test).
add_test(NAME dsa_dump_generate
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsa_reference_dump.py
          gen-pure --out ${CMAKE_CURRENT_BINARY_DIR}/dsa_pure_ref.dsadump)
add_test(NAME dsa_dump_test
  COMMAND dsa_test --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsa_pure_ref.dsadump)
set_tests_properties(dsa_dump_test PROPERTIES
  DEPENDS dsa_dump_generate SKIP_RETURN_CODE 2)

# Reference-dump parity: generate a pure-python oracle dump, then run the
# layer against it. The torch backend (real checkpoint slices) is a manual
# deployment test documented in docs/benchmarking.md.
add_test(NAME kda_dump_generate
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/kda_reference_dump.py
          gen-pure --out ${CMAKE_CURRENT_BINARY_DIR}/kda_pure_ref.kdadump
          --heads 2 --head-dim 32 --hidden 64 --tokens 8)
add_test(NAME kda_dump_test
  COMMAND kda_test --dump-file ${CMAKE_CURRENT_BINARY_DIR}/kda_pure_ref.kdadump)
set_tests_properties(kda_dump_test PROPERTIES
  DEPENDS kda_dump_generate SKIP_RETURN_CODE 2)

add_executable(gpt_doll apps/gpt_doll.cpp)
target_link_libraries(gpt_doll PRIVATE dgpp_models_doll dgpp_loaders)
dgpp_set_common(gpt_doll)
# Fast GPU selftest; SKIP_RETURN_CODE 2 = no visible device, ctest skips.
add_test(NAME doll_selftest COMMAND gpt_doll --selftest --layers 2
  --hidden 1024 --vocab 8192 --inter 2048 --max-seq 512)
set_tests_properties(doll_selftest PROPERTIES SKIP_RETURN_CODE 2)

# Flag-protocol regression test (CUDA); skip without a device as above.
add_test(NAME flag_protocol_test COMMAND flag_protocol_test)

# M5 control-plane app smoke (loopback, in-process).
add_test(NAME roster_selftest COMMAND roster_check selftest)
set_tests_properties(flag_protocol_test PROPERTIES SKIP_RETURN_CODE 2)

# M5 data plane: deep loopback scenarios (verbs + consumers + contention)
# and the app-level smoke. Both skip without a visible device.
if(DGPP_ENABLE_IBV)
  add_executable(bus_test tests/cuda/bus_test.cu tests/cuda/bus_boundary_fixture.cpp)
  target_link_libraries(bus_test PRIVATE dgpp_bus)
  dgpp_set_common(bus_test)
  add_test(NAME bus_test COMMAND bus_test)
  set_tests_properties(bus_test PROPERTIES SKIP_RETURN_CODE 2
                       ENVIRONMENT "CUDA_DEVICE_MAX_CONNECTIONS=32")
  add_test(NAME bus_selftest COMMAND bus_check selftest)
  set_tests_properties(bus_selftest PROPERTIES SKIP_RETURN_CODE 2)
endif()

# Serving does not depend on the standalone benchmark sources.
if(DGPP_BUILD_BENCHMARKS)
  # Benchmark sources can be locally ignored or omitted from a checkout.
  # Check every source, including auxiliary translation units, before defining
  # the target so no target properties refer to a skipped benchmark.
  function(dgpp_add_benchmark target)
    cmake_parse_arguments(BENCH "" ""
      "SOURCES;LIBRARIES;INCLUDE_DIRECTORIES;SYSTEM_INCLUDE_DIRECTORIES" ${ARGN})
    foreach(source IN LISTS BENCH_SOURCES)
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        message(STATUS "Skipping benchmark ${target}: missing ${source}")
        return()
      endif()
    endforeach()
    add_executable(${target} ${BENCH_SOURCES})
    target_link_libraries(${target} PRIVATE ${BENCH_LIBRARIES})
    if(BENCH_INCLUDE_DIRECTORIES)
      target_include_directories(${target} PRIVATE ${BENCH_INCLUDE_DIRECTORIES})
    endif()
    if(BENCH_SYSTEM_INCLUDE_DIRECTORIES)
      target_include_directories(${target} SYSTEM PRIVATE ${BENCH_SYSTEM_INCLUDE_DIRECTORIES})
    endif()
    dgpp_set_common(${target})
  endfunction()

  dgpp_add_benchmark(host_penalties_bench
    SOURCES benchmarks/micro/host_penalties_bench.cpp)

  dgpp_add_benchmark(micro_mem_bw
    SOURCES benchmarks/micro/mem_bw.cu src/common/log.cpp
    LIBRARIES CUDA::cudart Threads::Threads)

  dgpp_add_benchmark(micro_zerocopy
    SOURCES benchmarks/micro/zerocopy.cu src/common/log.cpp
    LIBRARIES CUDA::cudart Threads::Threads)

  dgpp_add_benchmark(micro_gemv_bw
    SOURCES benchmarks/micro/gemv_bw.cu
    LIBRARIES dgpp_kernels CUDA::cudart)

  dgpp_add_benchmark(micro_head_screen
    SOURCES benchmarks/micro/head_screen_bench.cu
    LIBRARIES dgpp_kernels CUDA::cudart Threads::Threads
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})
  dgpp_add_benchmark(micro_mma_gemv_shapes
    SOURCES benchmarks/micro/mma_gemv_shapes.cu
    LIBRARIES dgpp_kernels CUDA::cudart)

  dgpp_add_benchmark(micro_gemm_peak
    SOURCES benchmarks/micro/gemm_peak.cu src/common/log.cpp
    LIBRARIES CUDA::cudart CUDA::cublasLt)

  dgpp_add_benchmark(kda_bench
    SOURCES benchmarks/micro/kda_bench.cu src/common/log.cpp
    LIBRARIES dgpp_models_kda CUDA::cudart CUDA::cublasLt)

  dgpp_add_benchmark(moe_slot_bench
    SOURCES benchmarks/micro/moe_slot_bench.cu src/common/log.cpp
    LIBRARIES dgpp_models_glm_loader dgpp_models_glm CUDA::cudart CUDA::cublasLt)
  dgpp_add_benchmark(qwen_dense_bench
    SOURCES benchmarks/micro/qwen_dense_bench.cu src/common/log.cpp
    LIBRARIES dgpp_kernels CUDA::cudart)
  dgpp_add_benchmark(mhc_site_bench
    SOURCES benchmarks/micro/mhc_site_bench.cu src/common/log.cpp
    LIBRARIES dgpp_kernels CUDA::cudart)
  dgpp_add_benchmark(moe_tile_bench
    SOURCES benchmarks/micro/moe_tile_bench.cu src/common/log.cpp
    LIBRARIES dgpp_models_glm_loader dgpp_models_glm CUDA::cudart CUDA::cublasLt)
  dgpp_add_benchmark(moe_prefill_bench
    SOURCES benchmarks/micro/moe_prefill_bench.cu benchmarks/micro/moe_prefill_variants.cu
    LIBRARIES dgpp_kernels CUDA::cudart)
  dgpp_add_benchmark(packq_prefill_bench
    SOURCES benchmarks/micro/packq_prefill_bench.cu
    LIBRARIES dgpp_kernels CUDA::cudart)

  dgpp_add_benchmark(fp8_gemm_bench
    SOURCES benchmarks/micro/fp8_gemm_bench.cu
    LIBRARIES dgpp_kernels CUDA::cudart CUDA::cublasLt)

  dgpp_add_benchmark(mimo_step_bench
    SOURCES benchmarks/micro/mimo_step_bench.cu src/common/log.cpp
    LIBRARIES dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart CUDA::cublasLt)

  dgpp_add_benchmark(dsa_select_bench
    SOURCES benchmarks/micro/dsa_select_bench.cu
    LIBRARIES dgpp_models_dsa CUDA::cudart)

  dgpp_add_benchmark(qsa_prefill_bench
    SOURCES benchmarks/micro/qsa_prefill_bench.cu
    LIBRARIES dgpp_kernels CUDA::cudart)

  dgpp_add_benchmark(qsa_fp8_bench
    SOURCES benchmarks/micro/qsa_fp8_bench.cu
    LIBRARIES dgpp_kernels CUDA::cudart)

  dgpp_add_benchmark(qsa_index_bench
    SOURCES benchmarks/micro/qsa_index_bench.cu
    LIBRARIES dgpp_models_qwen dgpp_kernels CUDA::cudart)

  dgpp_add_benchmark(csa2_select_bench
    SOURCES benchmarks/micro/csa2_select_bench.cu
    LIBRARIES dgpp_kernels CUDA::cudart)

  dgpp_add_benchmark(dsa_bench
    SOURCES benchmarks/micro/dsa_bench.cu src/common/log.cpp
    LIBRARIES dgpp_models_dsa CUDA::cudart CUDA::cublasLt)

  if(DGPP_ENABLE_IBV)
    dgpp_add_benchmark(micro_gdr_probe
      SOURCES benchmarks/micro/gdr_probe.cpp src/common/log.cpp
      LIBRARIES CUDA::cudart ${IBVERBS_LIBRARY}
      SYSTEM_INCLUDE_DIRECTORIES ${IBVERBS_INCLUDE_DIR})

    # The send-doorbell mapping probe (2026-09-19): mlx5 direct verbs, only
    # where libmlx5 is installed.
    find_library(MLX5_LIBRARY NAMES mlx5)
    if(MLX5_LIBRARY)
      dgpp_add_benchmark(micro_uar_probe
        SOURCES benchmarks/micro/uar_probe.cpp
        LIBRARIES ${MLX5_LIBRARY} ${IBVERBS_LIBRARY} CUDA::cudart
        SYSTEM_INCLUDE_DIRECTORIES ${IBVERBS_INCLUDE_DIR})
    endif()

    dgpp_add_benchmark(micro_ibv_smoke
      SOURCES benchmarks/micro/ibv_smoke.cpp src/net/nic_visibility.cu src/common/log.cpp
      LIBRARIES ${IBVERBS_LIBRARY} CUDA::cudart
      SYSTEM_INCLUDE_DIRECTORIES ${IBVERBS_INCLUDE_DIR})
  endif()
endif()

# Labels describe resources, not coverage guarantees. Some real-checkpoint
# cases still skip internally when their optional model data is unavailable.
get_property(DGPP_ALL_TESTS DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS DGPP_ALL_TESTS)
  set_tests_properties(${_test} PROPERTIES LABELS "gpu")
endforeach()
set_tests_properties(tokenizer_patterns_test word_cache_test bpe_test unit unit_tests http_server_test nvme_cache_test serve_test serve_test_wake fabric_serve_test fabric_serve_test_wake scheduler_test shutdown_watchdog_test engine_watchdog_test
  qwen_yarn_fixture_test qwen_w4a4_plan_default qwen_w4a4_plan_mode_0 qwen_w4a4_plan_mode_1
  qwen_w4a4_plan_mode_empty qwen_w4a4_plan_mode_invalid PROPERTIES LABELS "host")
set_tests_properties(checkpoint_audit_test chat_template_generator_test route_trace_test sampling_profile_test spec_ab_probe_test
  site_env_test script_reports_test timed_load_test serve_admission_bench_test qwen_yarn_release_check_test qwen_head_compare_test portability_test spark_cross_test setup_tools_test bootstrap_test launcher_test cluster_process_swap_test PROPERTIES LABELS "host;python")
set_tests_properties(cluster_process_swap_systemd_test PROPERTIES LABELS "systemd" TIMEOUT 180)
if(TARGET dgpp_serve_app)
  set_tests_properties(serve_startup_test PROPERTIES LABELS "host;python" TIMEOUT 180)
endif()
set_tests_properties(tokenizer_test qwen_tokenizer_test glm4_tokenizer_test dsv41_tokenizer_test
  chat_template_test qwen_chat_template_test glm4_chat_template_test dsv41_prompt_test
  dsv4_tokenizer_test dsv4_prompt_test glm_dsa_tokenizer_test mimo_tokenizer_test flint_tokenizer_test
  glm_dsa_chat_template_test mimo_chat_template_test flint_chat_template_test dory_grammar_repro
  PROPERTIES LABELS "host;checkpoint")
foreach(_test qwen_forward_fixture qwen_forward_generate glm4_forward_fixture
    glm4_forward_generate mimo_forward_fixture mimo_forward_generate mimo_forward_generate_teacher
    flint_forward_fixture flint_forward_generate flint_forward_generate_teacher
    glm_forward_fixture glm_forward_generate dsa_dump_generate kda_dump_generate)
  set_tests_properties(${_test} PROPERTIES LABELS "fixture")
endforeach()
if(DGPP_ENABLE_IBV)
  set_tests_properties(bus_test bus_selftest glm_tp_test glm4_tp_test glm4_engine_test
    qwen_tp_test qwen_engine_test qwen_engine_fp8_head_4 qwen_engine_fp8_head_256 dsv41_tp_test dsv41_engine_test
    dsv4_tp_test dsv4_engine_test mimo_tp_test mimo_engine_test flint_tp_test flint_engine_test
    PROPERTIES LABELS "gpu;rdma" RUN_SERIAL TRUE)
  set_tests_properties(nic_regress_selftest PROPERTIES LABELS "gpu;rdma" RUN_SERIAL TRUE)
endif()
set_tests_properties(qwen_engine_row_independent qwen_engine_bf12
  qwen_engine_nvfp4_dense_0 qwen_engine_nvfp4_dense_1
  qwen_engine_fp4local_wide qwen_engine_fp4local_mixed_prefill
  qwen_engine_fp4local_wide_tile qwen_engine_fp4local_mixed_prefill_tile
  qwen_engine_fp8_wide qwen_engine_fp8_mtp3_rows32
  qwen_engine_mtp3_rows32_row_independent qwen_engine_fp8_mtp3_rows64
  qwen_engine_mtp3_rows64_row_independent
  PROPERTIES LABELS "gpu;rdma" RUN_SERIAL TRUE)
set_tests_properties(roster_selftest PROPERTIES LABELS "host")

# GLM image encoder reference/parity runner (no language-model weights loaded).
add_executable(glm_vision_check apps/glm_vision_check.cpp)
target_link_libraries(glm_vision_check PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_vision_check)

add_executable(qwen_vision_check apps/qwen_vision_check.cpp)
target_link_libraries(qwen_vision_check PRIVATE dgpp_models_qwen_loader CUDA::cudart)
dgpp_set_common(qwen_vision_check)

add_executable(qwen_vision_test tests/cuda/qwen_vision_test.cpp)
target_link_libraries(qwen_vision_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(qwen_vision_test)
add_test(NAME qwen_vision_test COMMAND qwen_vision_test)
set_tests_properties(qwen_vision_test PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 2)

add_executable(glm_vision_test tests/cuda/glm_vision_test.cpp)
target_link_libraries(glm_vision_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(glm_vision_test)
add_test(NAME glm_vision_test COMMAND glm_vision_test)
set_tests_properties(glm_vision_test PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 2)

add_executable(glm_vision_stream_test tests/cuda/glm_vision_stream_test.cpp)
target_link_libraries(glm_vision_stream_test PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_vision_stream_test)
add_test(NAME glm_vision_stream_test COMMAND glm_vision_stream_test)
set_tests_properties(glm_vision_stream_test PROPERTIES LABELS "gpu;checkpoint" SKIP_RETURN_CODE 2 RUN_SERIAL TRUE)

add_executable(glm_vision_frontend_test tests/host/glm_vision_frontend_test.cpp)
target_link_libraries(glm_vision_frontend_test PRIVATE dgpp_serve)
dgpp_set_common(glm_vision_frontend_test)
add_test(NAME glm_vision_frontend_test COMMAND glm_vision_frontend_test)
set_tests_properties(glm_vision_frontend_test PROPERTIES LABELS "host;checkpoint" SKIP_RETURN_CODE 2)

# Isolated upstream-port regression: paged BF16/FP8 history and head rows.
add_executable(mimo_port_test tests/cuda/mimo_port_test.cpp)
target_include_directories(mimo_port_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_port_test PRIVATE dgpp_models_mimo_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_port_test)
add_test(NAME mimo_port_test COMMAND mimo_port_test)
set_tests_properties(mimo_port_test PROPERTIES LABELS "gpu" RUN_SERIAL TRUE)

add_executable(mimo_native_mtp_test tests/cuda/mimo_native_mtp_test.cpp)
target_include_directories(mimo_native_mtp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(mimo_native_mtp_test PRIVATE dgpp_models_mimo_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(mimo_native_mtp_test)
add_test(NAME mimo_native_mtp_test COMMAND mimo_native_mtp_test)
set_tests_properties(mimo_native_mtp_test PROPERTIES LABELS "gpu" RUN_SERIAL TRUE)

# Shared request lifetime and mixed execution across model architectures.
add_executable(shared_prefill_test tests/cuda/shared_prefill_test.cpp tests/cuda/glm_tp_fixture.cpp)
target_link_libraries(shared_prefill_test PRIVATE dgpp_models_glm_loader dgpp_models_glm4_layers
  dgpp_models_glm_dsa_layers dgpp_models_mimo_layers dgpp_models_qwen_layers
  dgpp_models_dsv4_layers dgpp_models_dsv41_layers dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(shared_prefill_test)
if(TARGET dgpp_bus)
  target_link_libraries(shared_prefill_test PRIVATE dgpp_bus dgpp_serve)
  target_compile_definitions(shared_prefill_test PRIVATE DGPP_TEST_HAS_BUS=1)
endif()
add_test(NAME shared_prefill_test COMMAND ${CMAKE_COMMAND} -E env DGPP_DENSE_GEMV_ROWS=256 $<TARGET_FILE:shared_prefill_test>)
set_tests_properties(shared_prefill_test PROPERTIES LABELS "gpu" RUN_SERIAL TRUE)

# Compact serving heads: allocation, FP64 oracle, groups, cache and drafts.
add_executable(compact_head_test tests/cuda/compact_head_test.cpp tests/cuda/glm_tp_fixture.cpp)
target_include_directories(compact_head_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compact_head_test PRIVATE dgpp_models_glm_loader dgpp_models_glm4_layers
  dgpp_models_glm_dsa_layers dgpp_models_mimo_layers dgpp_models_dsv41_layers CUDA::cudart Threads::Threads)
dgpp_set_common(compact_head_test)
add_test(NAME compact_head_test COMMAND compact_head_test)
add_test(NAME compact_head_all_rows COMMAND ${CMAKE_COMMAND} -E env DGPP_PREFILL_HEAD_ALL_ROWS=1 $<TARGET_FILE:compact_head_test>)
set_tests_properties(compact_head_test compact_head_all_rows PROPERTIES LABELS "gpu" RUN_SERIAL TRUE)

if(TARGET dgpp_bus)
  add_executable(serving_head_check apps/serving_head_check.cpp)
  target_link_libraries(serving_head_check PRIVATE dgpp_models_glm_loader dgpp_models_glm4_layers
    dgpp_models_glm_dsa_layers dgpp_models_mimo_layers dgpp_models_dsv41_layers dgpp_models_qwen_layers
    dgpp_bus dgpp_text CUDA::cudart Threads::Threads)
  dgpp_set_common(serving_head_check)
endif()
