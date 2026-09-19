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

add_library(dgpp_kernels STATIC
  src/kernels/gemm_cublaslt.cpp
  src/kernels/kernels.cu
  src/kernels/doll_attention.cu
  src/kernels/kda.cu
  src/kernels/dsa.cu
  src/kernels/fp8_dequant.cu
  src/kernels/fp4_gemv.cu
  src/kernels/packq_gemv.cu
  src/kernels/packq_gemm.cu
  src/kernels/scale_gemm.cu
  src/kernels/bf16_gemv.cu
  src/kernels/mma_gemv.cu
  src/kernels/l2_prefetch.cu
  src/kernels/glm_mhc.cu
  src/kernels/glm_moe.cu
  src/kernels/glm_vision.cu
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
  src/kernels/qwen_mtp.cu
  src/kernels/qsa.cu
  src/kernels/qsa_prefill.cu
  src/kernels/glm4_attn.cu)
target_link_libraries(dgpp_kernels PUBLIC CUDA::cudart CUDA::cublasLt dgpp_core)
dgpp_set_common(dgpp_kernels)
# The device sampler mirrors the host oracle bit for bit: no contraction of
# its multiplies and adds (the host builds with -ffp-contract=off).
set_source_files_properties(src/kernels/sample_pick.cu PROPERTIES
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
  src/text/tool_parser.cpp
  src/text/tool_grammar.cpp
  src/text/string_constraint.cpp
  src/text/json_grammar.cpp)
target_link_libraries(dgpp_text PUBLIC dgpp_loaders dgpp_pcre2)
dgpp_set_common(dgpp_text)

add_library(dgpp_sched STATIC
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
target_link_libraries(dgpp_models_glm4_loader PUBLIC dgpp_models_glm4 dgpp_loaders CUDA::cudart)
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
target_link_libraries(dgpp_models_glm_dsa_loader PUBLIC dgpp_models_glm_dsa dgpp_loaders CUDA::cudart)
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
target_link_libraries(dgpp_models_dsv41_loader PUBLIC dgpp_models_dsv41 dgpp_loaders CUDA::cudart)
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

# Qwen3.8-Flash-Next: the host-side model description.
add_library(dgpp_models_qwen STATIC
  src/models/qwen/config.cpp
  src/models/qwen/binding.cpp
  src/models/qwen/gdn_reference.cpp
  src/models/qwen/norm_reference.cpp
  src/models/qwen/gr_reference.cpp
  src/models/qwen/ple_reference.cpp
  src/models/qwen/qsa_reference.cpp)
target_include_directories(dgpp_models_qwen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_qwen PUBLIC dgpp_loaders dgpp_common)
dgpp_set_common(dgpp_models_qwen)

# The Qwen resident loader (Q2): the shared weight builder at the family's
# TP geometry, plus the n-gram table's own residency.
add_library(dgpp_models_qwen_loader STATIC
  src/models/qwen/loader.cpp)
target_include_directories(dgpp_models_qwen_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_qwen_loader PUBLIC dgpp_models_qwen dgpp_loaders CUDA::cudart)
dgpp_set_common(dgpp_models_qwen_loader)

# The Qwen layers (Q3): the MoE on the shared MoE layer, with its oracle.
add_library(dgpp_models_qwen_layers STATIC
  src/models/qwen/moe_layer.cpp
  src/models/qwen/moe_reference.cpp
  src/models/qwen/layers.cpp
  src/models/qwen/kv_pool.cpp
  src/models/qwen/forward.cpp)
target_include_directories(dgpp_models_qwen_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgpp_models_qwen_layers
  PUBLIC dgpp_models_qwen dgpp_models_qwen_loader dgpp_models_moe dgpp_models_glm dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_qwen_layers)

# The Qwen forward (Q4): fixture -> smoke; the python reference chain follows.
add_executable(qwen_forward_test tests/cuda/qwen_forward_test.cpp)
target_include_directories(qwen_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_forward_test PRIVATE dgpp_models_qwen_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_forward_test)
set(QWEN_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/qwen_forward_fixture)
add_test(NAME qwen_forward_fixture COMMAND qwen_forward_test --write-fixture ${QWEN_FORWARD_FIXTURE})
add_test(NAME qwen_forward_smoke COMMAND qwen_forward_test --smoke ${QWEN_FORWARD_FIXTURE})
set_tests_properties(qwen_forward_smoke PROPERTIES DEPENDS qwen_forward_fixture)
add_test(NAME qwen_forward_generate
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen_reference_dump.py
          gen-pure --checkpoint-dir ${QWEN_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/qwen_pure_ref.qwndump)
set_tests_properties(qwen_forward_generate PROPERTIES DEPENDS qwen_forward_fixture)
add_test(NAME qwen_forward_test
  COMMAND qwen_forward_test --checkpoint-dir ${QWEN_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/qwen_pure_ref.qwndump)
set_tests_properties(qwen_forward_test PROPERTIES DEPENDS qwen_forward_generate)

# The Qwen session surface (Q6 stage A): incremental decode vs the re-forward.
add_executable(qwen_decode_test tests/cuda/qwen_decode_test.cpp)
target_include_directories(qwen_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(qwen_decode_test PRIVATE dgpp_models_qwen_layers dgpp_models_glm_loader CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_decode_test)
add_test(NAME qwen_decode_test COMMAND qwen_decode_test --fixture ${QWEN_FORWARD_FIXTURE})
set_tests_properties(qwen_decode_test PROPERTIES DEPENDS qwen_forward_fixture)

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
  PUBLIC dgpp_models_glm dgpp_models_moe dgpp_kernels dgpp_models_kda dgpp_models_dsa)
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
  target_include_directories(dgpp_bus PUBLIC ${IBVERBS_INCLUDE_DIR})
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
                                             dgpp_models_dsv41_layers
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
  install(FILES docs/licenses/PCRE2.md docs/licenses/stb_image.md docs/licenses/pytorch.md DESTINATION docs/licenses)
  install(FILES deploy/nginx.example.conf DESTINATION deploy)
endif()

enable_testing()
add_executable(unit_tests
  tests/unit/image_inputs_test.cpp
  tests/unit/file_inputs_test.cpp
  tests/unit/log_test.cpp
  tests/unit/minijson_test.cpp
  tests/unit/arena_test.cpp
  tests/unit/safetensors_test.cpp
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
  tests/unit/qwen_binding_test.cpp
  tests/unit/glm4_config_test.cpp
  tests/unit/glm4_binding_test.cpp
  tests/unit/glm_dsa_config_test.cpp
  tests/unit/glm_dsa_binding_test.cpp
  tests/unit/dsv41_config_test.cpp
  tests/unit/dsv41_binding_test.cpp
  tests/unit/nvfp4_quant_test.cpp
  tests/unit/verify_schedule_test.cpp
  src/core/trace.cpp)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(unit_tests PRIVATE CUDA::cudart dgpp_loaders
                                        dgpp_models_glm dgpp_models_qwen dgpp_models_glm4
                                        dgpp_models_glm_dsa dgpp_models_dsv41
                                        dgpp_net dgpp_serve Threads::Threads)
target_compile_definitions(unit_tests PRIVATE DGPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
dgpp_set_common(unit_tests)
add_test(NAME unit_tests COMMAND unit_tests)
target_sources(unit_tests PRIVATE tests/unit/site_options_test.cpp)

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

# The DeepSeek-V4.1 prompt renderer's differential gate (no Jinja: the
# checkpoint's encoding.py is the golden source).
add_executable(dsv41_prompt_test tests/host/dsv41_prompt_test.cpp)
target_include_directories(dsv41_prompt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dsv41_prompt_test PRIVATE dgpp_text dgpp_models_glm Threads::Threads)
dgpp_set_common(dsv41_prompt_test)
add_test(NAME dsv41_prompt_test
  COMMAND dsv41_prompt_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dsv41_prompt_goldens.jsonl)
set_tests_properties(dsv41_prompt_test PROPERTIES SKIP_RETURN_CODE 2)
set_tests_properties(qwen_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm4_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm4_tokenizer_goldens.jsonl)
set_tests_properties(glm4_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm_dsa_tokenizer_test
  COMMAND tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm_dsa_tokenizer_goldens.jsonl)
set_tests_properties(glm_dsa_tokenizer_test PROPERTIES SKIP_RETURN_CODE 2)

# M6 Stage 3b: the chat-template differential gate (host). Skips with rc 2
# when the model cache is absent, same as above.
add_executable(chat_template_test tests/host/chat_template_test.cpp)
target_include_directories(chat_template_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(chat_template_test PRIVATE dgpp_models_glm Threads::Threads)
dgpp_set_common(chat_template_test)
add_test(NAME chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm_chat_template_goldens.jsonl)
set_tests_properties(chat_template_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME qwen_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen_chat_template_goldens.jsonl)
set_tests_properties(qwen_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm4_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm4_chat_template_goldens.jsonl)
set_tests_properties(glm4_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm_dsa_chat_template_test
  COMMAND chat_template_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/glm_dsa_chat_template_goldens.jsonl)
set_tests_properties(glm_dsa_chat_template_test PROPERTIES SKIP_RETURN_CODE 2)

# M6 Stage 4: the serving capstone — HTTP/SSE over epoll + the OpenAI
# generation service (host-only; the engine adapters live in the apps,
# dgpp_serve never links CUDA — CI runs the service gate with a fake
# engine and frontend, no model cache). Stage 4b adds the admission
# journal (fabric_serve.cpp over the host-only dgpp_net TCP primitives).
add_library(dgpp_serve STATIC
  src/serve/image_inputs.cpp
  src/serve/glm_vision_frontend.cpp
  src/serve/file_inputs.cpp
  src/serve/cluster_config.cpp
  src/serve/http_server.cpp
  src/serve/generation_service.cpp
  src/serve/fabric_serve.cpp
  src/serve/serve_stats.cpp)
target_link_libraries(dgpp_serve
  PUBLIC dgpp_core dgpp_models_glm dgpp_loaders dgpp_net Threads::Threads)
dgpp_set_common(dgpp_serve)

add_executable(http_server_test tests/host/http_server_test.cpp)
target_link_libraries(http_server_test PRIVATE dgpp_serve Threads::Threads)
dgpp_set_common(http_server_test)
add_test(NAME http_server_test COMMAND http_server_test)

# M6 Stage 4 service gate: the REAL HTTP/SSE service over a fake engine
# + fake frontend — the OpenAI surface, host-only, always runs.
add_executable(serve_test tests/host/serve_test.cpp)
target_link_libraries(serve_test PRIVATE dgpp_serve Threads::Threads)
dgpp_set_common(serve_test)
add_test(NAME serve_test COMMAND serve_test)

# M6 Stage 4b gate: the admission journal (the metronome), both loops,
# and the §11 identity — real localhost TCP + real HTTP, engine faked
# (host-only, always runs).
add_executable(fabric_serve_test tests/host/fabric_serve_test.cpp)
target_link_libraries(fabric_serve_test PRIVATE dgpp_serve
                                                   Threads::Threads)
dgpp_set_common(fabric_serve_test)
add_test(NAME fabric_serve_test COMMAND fabric_serve_test)

# M6 Stage 2b: the scheduler policy gate (host, no model, no skip path —
# the FakeEngine replaces the engine, so the policy is pinned everywhere
# CI runs).
add_executable(scheduler_test tests/host/scheduler_test.cpp)
target_include_directories(scheduler_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(scheduler_test PRIVATE dgpp_models_glm Threads::Threads)
dgpp_set_common(scheduler_test)
add_test(NAME scheduler_test COMMAND scheduler_test)

add_executable(flag_protocol_test tests/cuda/flag_protocol_test.cu)
target_link_libraries(flag_protocol_test PRIVATE CUDA::cudart)
dgpp_set_common(flag_protocol_test)

add_test(NAME unit COMMAND unit_tests)
add_test(NAME checkpoint_audit_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/checkpoint_audit_test.py)
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
add_test(NAME spark_cross_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/spark_cross_test.py)
add_test(NAME setup_tools_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/setup_tools_test.py)
add_test(NAME launcher_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/launcher_test.py)

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

add_executable(dsv41_engram_layer_test tests/cuda/dsv41_engram_layer_test.cpp)
target_include_directories(dsv41_engram_layer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_engram_layer_test PRIVATE dgpp_models_dsv41_layers CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_engram_layer_test)
add_test(NAME dsv41_engram_layer_test COMMAND dsv41_engram_layer_test)

add_executable(dsv41_model_test tests/cuda/dsv41_model_test.cpp)
target_include_directories(dsv41_model_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_model_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_model_test)
add_test(NAME dsv41_model_test COMMAND dsv41_model_test)

# The forward against the python reference dump (numpy: the repo's .venv
# when present, else the system python).
add_executable(dsv41_forward_test tests/cuda/dsv41_forward_test.cpp)
target_include_directories(dsv41_forward_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_forward_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_forward_test)
set(DSV41_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/dsv41_forward_fixture)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3)
  set(DSV41_PYTHON ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3)
else()
  set(DSV41_PYTHON ${Python3_EXECUTABLE})
endif()
add_test(NAME dsv41_forward_fixture COMMAND dsv41_forward_test --write-fixture ${DSV41_FORWARD_FIXTURE})
add_test(NAME dsv41_forward_smoke COMMAND dsv41_forward_test --smoke ${DSV41_FORWARD_FIXTURE})
set_tests_properties(dsv41_forward_smoke PROPERTIES DEPENDS dsv41_forward_fixture)
add_test(NAME dsv41_forward_generate
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
          --checkpoint-dir ${DSV41_FORWARD_FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_pure_ref.dump --tokens 40)
set_tests_properties(dsv41_forward_generate PROPERTIES DEPENDS dsv41_forward_fixture)
# End to end (the drift of eight layers with fp4 caches: relaxed budgets),
# writing the engine's layer states; then the teacher-forced dump (every
# reference layer fed the engine's input to it) under the strict budgets.
add_test(NAME dsv41_forward_test_e2e
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DSV41_DUMP_ENGINE_STATES=${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states.bin
          $<TARGET_FILE:dsv41_forward_test> --checkpoint-dir ${DSV41_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_pure_ref.dump --relaxed)
set_tests_properties(dsv41_forward_test_e2e PROPERTIES DEPENDS dsv41_forward_generate)
add_test(NAME dsv41_forward_generate_teacher
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
          --checkpoint-dir ${DSV41_FORWARD_FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_ref.dump --tokens 40
          --teacher ${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states.bin)
set_tests_properties(dsv41_forward_generate_teacher PROPERTIES DEPENDS dsv41_forward_test_e2e)
add_test(NAME dsv41_forward_test
  COMMAND dsv41_forward_test --checkpoint-dir ${DSV41_FORWARD_FIXTURE} --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_ref.dump)
set_tests_properties(dsv41_forward_test PROPERTIES DEPENDS dsv41_forward_generate_teacher)
# The bounded prefill (plan §1.8, the production mode): the same chain
# against the reference's --prefill bounded walk (the decoder over the
# last window rows), end to end then teacher-forced.
add_test(NAME dsv41_forward_generate_bounded
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
          --checkpoint-dir ${DSV41_FORWARD_FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_bounded_ref.dump --tokens 40
          --prefill bounded)
set_tests_properties(dsv41_forward_generate_bounded PROPERTIES DEPENDS dsv41_forward_fixture)
add_test(NAME dsv41_forward_test_bounded_e2e
  COMMAND ${CMAKE_COMMAND} -E env DGPP_DSV41_DUMP_ENGINE_STATES=${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states_bounded.bin
          $<TARGET_FILE:dsv41_forward_test> --checkpoint-dir ${DSV41_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_bounded_ref.dump --relaxed --bounded
          --exact-dump ${CMAKE_CURRENT_BINARY_DIR}/dsv41_pure_ref.dump
          --exact-states ${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states.bin)
set_tests_properties(dsv41_forward_test_bounded_e2e PROPERTIES DEPENDS "dsv41_forward_generate_bounded;dsv41_forward_test_e2e")
add_test(NAME dsv41_forward_generate_teacher_bounded
  COMMAND ${DSV41_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsv41_reference_dump.py gen-pure
          --checkpoint-dir ${DSV41_FORWARD_FIXTURE} --out ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_bounded_ref.dump --tokens 40
          --prefill bounded --teacher ${CMAKE_CURRENT_BINARY_DIR}/dsv41_engine_states_bounded.bin)
set_tests_properties(dsv41_forward_generate_teacher_bounded PROPERTIES DEPENDS dsv41_forward_test_bounded_e2e)
add_test(NAME dsv41_forward_test_bounded
  COMMAND dsv41_forward_test --checkpoint-dir ${DSV41_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsv41_teacher_bounded_ref.dump --bounded)
set_tests_properties(dsv41_forward_test_bounded PROPERTIES DEPENDS dsv41_forward_generate_teacher_bounded)

# The DeepSeek-V4.1 session surface: decode vs the re-forward, interleaved
# slots, the chunked prefill, the prefix snapshots, the eager DSpark speculator.
add_executable(dsv41_decode_test tests/cuda/dsv41_decode_test.cpp)
target_include_directories(dsv41_decode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_decode_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_decode_test)
add_test(NAME dsv41_decode_test COMMAND dsv41_decode_test --fixture ${DSV41_FORWARD_FIXTURE})
set_tests_properties(dsv41_decode_test PROPERTIES DEPENDS dsv41_forward_fixture)

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

add_executable(glm4_loader_test tests/cuda/glm4_loader_test.cpp)
target_include_directories(glm4_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(glm4_loader_test PRIVATE dgpp_models_glm4_loader CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_loader_test)
add_test(NAME glm4_loader_test COMMAND glm4_loader_test)

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

add_executable(glm4_load_check apps/glm4_load_check.cpp)
target_link_libraries(glm4_load_check PRIVATE dgpp_models_glm4_loader dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_load_check)

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
target_link_libraries(dsv41_loader_test PRIVATE dgpp_models_dsv41_loader CUDA::cudart Threads::Threads)
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

# The DeepSeek-V4.1 decode engines: the graph engine's scalar and row-batched
# replays (the Engram host node inside the walk) against the eager engine
# over a loopback world of 2 (ports 29961-29964: plain, the DSpark block, depth 2, the scheduled depth).
add_executable(dsv41_engine_test tests/cuda/dsv41_engine_test.cpp)
target_include_directories(dsv41_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda)
target_link_libraries(dsv41_engine_test PRIVATE dgpp_models_dsv41_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(dsv41_engine_test)
add_test(NAME dsv41_engine_test COMMAND dsv41_engine_test)

add_executable(glm4_forward_check apps/glm4_forward_check.cpp)
target_link_libraries(glm4_forward_check PRIVATE dgpp_models_glm4_layers dgpp_bus dgpp_loaders dgpp_sched dgpp_text CUDA::cudart Threads::Threads)
dgpp_set_common(glm4_forward_check)

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
set_tests_properties(qwen_engine_test qwen_engine_row_independent PROPERTIES RESOURCE_LOCK qwen_engine_world)
set_tests_properties(qwen_engine_row_independent PROPERTIES LABELS "gpu;rdma")

add_executable(qwen_forward_check apps/qwen_forward_check.cpp)
target_link_libraries(qwen_forward_check PRIVATE dgpp_models_qwen_layers dgpp_bus dgpp_loaders CUDA::cudart Threads::Threads)
dgpp_set_common(qwen_forward_check)

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

add_executable(mma_gemv_test tests/cuda/mma_gemv_test.cu)
target_include_directories(mma_gemv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mma_gemv_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(mma_gemv_test)
add_test(NAME mma_gemv_test COMMAND mma_gemv_test)
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
add_executable(packq_gemm_test tests/cuda/packq_gemm_test.cu)
target_include_directories(packq_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(packq_gemm_test PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(packq_gemm_test)
add_test(NAME packq_gemm_test COMMAND packq_gemm_test)
set_tests_properties(packq_gemm_test PROPERTIES SKIP_RETURN_CODE 2 LABELS "gpu")

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
add_executable(glm_moe_test tests/cuda/glm_moe_test.cu)
target_include_directories(glm_moe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_moe_test PRIVATE dgpp_models_glm_loader dgpp_models_glm CUDA::cudart)
dgpp_set_common(glm_moe_test)
add_test(NAME glm_moe_test COMMAND glm_moe_test)
set_tests_properties(glm_moe_test PROPERTIES SKIP_RETURN_CODE 2)

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
# deployment test documented in benchmarks/README.md.
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
  add_executable(bus_test tests/cuda/bus_test.cu)
  target_link_libraries(bus_test PRIVATE dgpp_bus)
  dgpp_set_common(bus_test)
  add_test(NAME bus_test COMMAND bus_test)
  set_tests_properties(bus_test PROPERTIES SKIP_RETURN_CODE 2
                       ENVIRONMENT "CUDA_DEVICE_MAX_CONNECTIONS=32")
  add_test(NAME bus_selftest COMMAND bus_check selftest)
  set_tests_properties(bus_selftest PROPERTIES SKIP_RETURN_CODE 2)
endif()

add_executable(micro_mem_bw benchmarks/micro/mem_bw.cu src/common/log.cpp)
target_link_libraries(micro_mem_bw PRIVATE CUDA::cudart Threads::Threads)
dgpp_set_common(micro_mem_bw)

add_executable(micro_zerocopy benchmarks/micro/zerocopy.cu src/common/log.cpp)
target_link_libraries(micro_zerocopy PRIVATE CUDA::cudart Threads::Threads)
dgpp_set_common(micro_zerocopy)

add_executable(micro_gemv_bw benchmarks/micro/gemv_bw.cu)
target_include_directories(micro_gemv_bw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(micro_gemv_bw PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(micro_gemv_bw)

add_executable(micro_gemm_peak benchmarks/micro/gemm_peak.cu src/common/log.cpp)
target_link_libraries(micro_gemm_peak PRIVATE CUDA::cudart CUDA::cublasLt)
dgpp_set_common(micro_gemm_peak)

add_executable(kda_bench benchmarks/micro/kda_bench.cu src/common/log.cpp)
target_link_libraries(kda_bench PRIVATE dgpp_models_kda CUDA::cudart CUDA::cublasLt)
dgpp_set_common(kda_bench)

add_executable(moe_slot_bench benchmarks/micro/moe_slot_bench.cu src/common/log.cpp)
add_executable(mhc_site_bench benchmarks/micro/mhc_site_bench.cu src/common/log.cpp)
target_include_directories(mhc_site_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mhc_site_bench PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(mhc_site_bench)
add_executable(moe_tile_bench benchmarks/micro/moe_tile_bench.cu src/common/log.cpp)
target_include_directories(moe_tile_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(moe_tile_bench PRIVATE dgpp_models_glm_loader dgpp_models_glm CUDA::cudart CUDA::cublasLt)
dgpp_set_common(moe_tile_bench)
add_executable(moe_prefill_bench benchmarks/micro/moe_prefill_bench.cu benchmarks/micro/moe_prefill_variants.cu)
target_include_directories(moe_prefill_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(moe_prefill_bench PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(moe_prefill_bench)
add_executable(packq_prefill_bench benchmarks/micro/packq_prefill_bench.cu)
target_include_directories(packq_prefill_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(packq_prefill_bench PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(packq_prefill_bench)
target_include_directories(moe_slot_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(moe_slot_bench PRIVATE dgpp_models_glm_loader dgpp_models_glm CUDA::cudart CUDA::cublasLt)
dgpp_set_common(moe_slot_bench)

add_executable(dsa_select_bench benchmarks/micro/dsa_select_bench.cu)
target_include_directories(dsa_select_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dsa_select_bench PRIVATE dgpp_models_dsa CUDA::cudart)
dgpp_set_common(dsa_select_bench)

add_executable(qsa_prefill_bench benchmarks/micro/qsa_prefill_bench.cu)
target_link_libraries(qsa_prefill_bench PRIVATE dgpp_kernels CUDA::cudart)
dgpp_set_common(qsa_prefill_bench)

add_executable(dsa_bench benchmarks/micro/dsa_bench.cu src/common/log.cpp)
target_link_libraries(dsa_bench PRIVATE dgpp_models_dsa CUDA::cudart CUDA::cublasLt)
dgpp_set_common(dsa_bench)

if(DGPP_ENABLE_IBV)
  add_executable(micro_gdr_probe benchmarks/micro/gdr_probe.cpp src/common/log.cpp)
  target_link_libraries(micro_gdr_probe PRIVATE CUDA::cudart)
  dgpp_set_common(micro_gdr_probe)
  target_link_libraries(micro_gdr_probe PRIVATE ${IBVERBS_LIBRARY})
  target_include_directories(micro_gdr_probe PRIVATE ${IBVERBS_INCLUDE_DIR})

  add_executable(micro_ibv_smoke
    benchmarks/micro/ibv_smoke.cpp
    src/net/nic_visibility.cu
    src/common/log.cpp)
  target_link_libraries(micro_ibv_smoke PRIVATE ${IBVERBS_LIBRARY} CUDA::cudart)
  target_include_directories(micro_ibv_smoke PRIVATE ${IBVERBS_INCLUDE_DIR})
  dgpp_set_common(micro_ibv_smoke)
endif()

# Labels describe resources, not coverage guarantees. Some real-checkpoint
# cases still skip internally when their optional model data is unavailable.
get_property(DGPP_ALL_TESTS DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS DGPP_ALL_TESTS)
  set_tests_properties(${_test} PROPERTIES LABELS "gpu")
endforeach()
set_tests_properties(unit unit_tests http_server_test serve_test fabric_serve_test scheduler_test
  PROPERTIES LABELS "host")
set_tests_properties(checkpoint_audit_test route_trace_test sampling_profile_test
  site_env_test script_reports_test portability_test spark_cross_test setup_tools_test launcher_test PROPERTIES LABELS "host;python")
set_tests_properties(tokenizer_test qwen_tokenizer_test glm4_tokenizer_test dsv41_tokenizer_test
  chat_template_test qwen_chat_template_test glm4_chat_template_test dsv41_prompt_test
  PROPERTIES LABELS "host;checkpoint")
foreach(_test qwen_forward_fixture qwen_forward_generate glm4_forward_fixture
    glm4_forward_generate glm_forward_fixture glm_forward_generate dsa_dump_generate kda_dump_generate)
  set_tests_properties(${_test} PROPERTIES LABELS "fixture")
endforeach()
if(DGPP_ENABLE_IBV)
  set_tests_properties(bus_test bus_selftest glm_tp_test glm4_tp_test glm4_engine_test
    qwen_tp_test qwen_engine_test dsv41_tp_test dsv41_engine_test PROPERTIES LABELS "gpu;rdma" RUN_SERIAL TRUE)
  set_tests_properties(nic_regress_selftest PROPERTIES LABELS "gpu;rdma" RUN_SERIAL TRUE)
endif()
set_tests_properties(roster_selftest PROPERTIES LABELS "host")

# GLM image encoder reference/parity runner (no language-model weights loaded).
add_executable(glm_vision_check apps/glm_vision_check.cpp)
target_link_libraries(glm_vision_check PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_vision_check)

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
