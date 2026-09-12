cmake_minimum_required(VERSION 3.18)
project(cascade_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CASCADE_RUNTIME_RPATH "" CACHE STRING
    "Extra runtime library search path baked into the targets")
if(CASCADE_RUNTIME_RPATH)
    set(CMAKE_INSTALL_RPATH "${CASCADE_RUNTIME_RPATH}")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_library(CASCADE_GTL_LIB
    NAMES mpi_gtl_cuda
    HINTS
        ENV MPICH_DIR
        ENV MPI_HOME
        ENV LD_LIBRARY_PATH
        ${MPI_CXX_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64 gtl/lib ../../../gtl/lib
    DOC "GPU transport layer library for GPU-aware MPI")
if(CASCADE_GTL_LIB)
    set(CASCADE_GTL_LINK "${CASCADE_GTL_LIB}")
    message(STATUS "GPU transport layer: ${CASCADE_GTL_LIB}")
else()
    set(CASCADE_GTL_LINK "")
    message(STATUS
        "GPU transport layer library not found. GPU-aware MPI aborts at "
        "MPI_Init without it. Set CASCADE_GTL_LIB to its path.")
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler=${OpenMP_CXX_FLAGS} -allow-unsupported-compiler")

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)

option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenSSL_INCLUDE_DIR})

set(CASCADE_SOURCES
    src/cascade_core.cpp
    src/gpu_backend.cu
)

add_library(cascade_static STATIC ${CASCADE_SOURCES})
target_link_libraries(cascade_static
    CUDA::cudart
    OpenSSL::Crypto
    pthread
    stdc++fs
)
set_target_properties(cascade_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_PYTHON)
    set(CASCADE_CPP_SOURCES
        python/bindings.cpp
        src/cascade_core.cpp
        src/gpu_backend.cu
        src/sglang_slots.cu
    )
    if(USE_MPI)
        list(APPEND CASCADE_CPP_SOURCES src/distributed_backend.cpp)
    endif()

    pybind11_add_module(cascade_cpp ${CASCADE_CPP_SOURCES})
    target_link_libraries(cascade_cpp PRIVATE
        CUDA::cudart
        OpenSSL::Crypto
        OpenMP::OpenMP_CXX
        stdc++fs
    )
    if(USE_MPI)
        target_link_libraries(cascade_cpp PRIVATE
            MPI::MPI_CXX
            ${CASCADE_GTL_LINK}
        )
        target_compile_definitions(cascade_cpp PRIVATE USE_MPI)
    endif()
    target_compile_definitions(cascade_cpp PRIVATE CASCADE_MODULE_NAME=cascade_cpp)

    set_target_properties(cascade_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    pybind11_add_module(cascade_cc ${CASCADE_CPP_SOURCES})
    target_link_libraries(cascade_cc PRIVATE
        CUDA::cudart
        OpenSSL::Crypto
        OpenMP::OpenMP_CXX
        stdc++fs
    )
    if(USE_MPI)
        target_link_libraries(cascade_cc PRIVATE
            MPI::MPI_CXX
            ${CASCADE_GTL_LINK}
        )
        target_compile_definitions(cascade_cc PRIVATE USE_MPI CASCADE_CENTRALIZED)
    endif()
    target_compile_definitions(cascade_cc PRIVATE CASCADE_MODULE_NAME=cascade_cc)
    set_target_properties(cascade_cc PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    install(TARGETS cascade_cpp cascade_cc LIBRARY DESTINATION .)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark.cpp")
    add_executable(cascade_bench src/benchmark.cpp)
    target_link_libraries(cascade_bench
        cascade_static
        CUDA::cudart
        OpenSSL::Crypto
        pthread
        OpenMP::OpenMP_CXX
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/pure_memcpy_bench.cu")
    add_executable(pure_memcpy_bench src/pure_memcpy_bench.cu)
    target_link_libraries(pure_memcpy_bench CUDA::cudart)
    set_target_properties(pure_memcpy_bench PROPERTIES CUDA_ARCHITECTURES 80)
endif()

option(SITE_DEFAULTS "Apply the reference platform's build defaults" ON)

if(SITE_DEFAULTS)
    set(CMAKE_CUDA_ARCHITECTURES 80)

    add_definitions(-DLARGE_STRIPE_ALIGN)

    if(DEFINED ENV{NVIDIA_PATH})
        set(CMAKE_CUDA_COMPILER "$ENV{NVIDIA_PATH}/cuda/bin/nvcc")
    endif()
endif()

option(USE_IOURING "Enable io_uring for async I/O" OFF)

if(USE_IOURING)
    find_library(URING_LIB uring)
    if(URING_LIB)
        target_link_libraries(cascade_static ${URING_LIB})
        add_definitions(-DUSE_IOURING)
    endif()
endif()

option(USE_MPI "Enable MPI for distributed multi-node" ON)

if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_DIRS})

    add_library(cascade_distributed STATIC
        src/cascade_core.cpp
        src/gpu_backend.cu
        src/distributed_backend.cpp
    )
    target_compile_definitions(cascade_distributed PUBLIC USE_MPI)
    target_link_libraries(cascade_distributed
        CUDA::cudart
        OpenSSL::Crypto
        MPI::MPI_CXX
        ${CASCADE_GTL_LINK}
        pthread
        OpenMP::OpenMP_CXX
        stdc++fs
    )
    set_target_properties(cascade_distributed PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/distributed_benchmark.cpp")
        add_executable(distributed_bench src/distributed_benchmark.cpp)
        target_compile_definitions(distributed_bench PUBLIC USE_MPI)
        target_link_libraries(distributed_bench
            cascade_distributed
            CUDA::cudart
            OpenSSL::Crypto
            MPI::MPI_CXX
            ${CASCADE_GTL_LINK}
            pthread
            OpenMP::OpenMP_CXX
        )
    endif()

    message(STATUS "MPI enabled: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI include: ${MPI_CXX_INCLUDE_DIRS}")
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/full_benchmark.cpp")
    add_executable(full_bench src/full_benchmark.cpp)
    target_link_libraries(full_bench
        CUDA::cudart
        pthread
        OpenMP::OpenMP_CXX
    )
    set_target_properties(full_bench PROPERTIES CUDA_ARCHITECTURES 80)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/fair_tier_bench.cpp")
    add_executable(fair_tier_bench src/fair_tier_bench.cpp)
    target_link_libraries(fair_tier_bench
        CUDA::cudart
        pthread
        OpenMP::OpenMP_CXX
    )
    set_target_properties(fair_tier_bench PROPERTIES CUDA_ARCHITECTURES 80)
endif()
