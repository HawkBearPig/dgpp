set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc-13)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++-13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/aarch64-linux-gnu-g++-13)
set(CMAKE_CUDA_FLAGS_INIT "--target-directory sbsa-linux")
set(CUDAToolkit_ROOT /usr/local/cuda-13.0)

# Ubuntu's cross compiler supplies its libc/libstdc++ search paths. Other
# dependencies must come from ARM64 multiarch directories or CUDA's SBSA root.
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu /usr/local/cuda-13.0/targets/sbsa-linux)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(IBVERBS_INCLUDE_DIR /usr/include CACHE PATH "ARM64 verbs headers")
set(IBVERBS_LIBRARY /usr/lib/aarch64-linux-gnu/libibverbs.so CACHE FILEPATH "ARM64 verbs library")
