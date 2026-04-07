cmake_minimum_required(VERSION 3.18)
project(cascade_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
        target_link_libraries(cascade_cpp PRIVATE MPI::MPI_CXX)
        target_compile_definitions(cascade_cpp PRIVATE USE_MPI)
    endif()

    set_target_properties(cascade_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    install(TARGETS cascade_cpp LIBRARY DESTINATION .)
endif()

option(USE_SLINGSHOT "Enable HPE Slingshot network optimizations" OFF)

if(USE_SLINGSHOT)
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
        pthread
        OpenMP::OpenMP_CXX
        stdc++fs
    )
    if(USE_SLINGSHOT)
        target_link_libraries(cascade_distributed -lmpi_gtl_cuda)
    endif()
    set_target_properties(cascade_distributed PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    message(STATUS "MPI enabled: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI include: ${MPI_CXX_INCLUDE_DIRS}")
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR}")
