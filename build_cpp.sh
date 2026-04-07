#!/bin/bash

set -e

source setup_env.sh

PKG_TMPDIR="/tmp/${USER}_pkg"
mkdir -p "$PKG_TMPDIR"
ln -sf /usr/bin/pkg-config "$PKG_TMPDIR/pkg-config"
export PATH="$PKG_TMPDIR:$PATH"

BUILD_DIR="cascade_Code/cpp/build_cascade_cpp"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON=ON \
    -DUSE_MPI=ON \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
    -DPKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config \
    -DUSE_SLINGSHOT=ON

echo "Building..."

make -j 258 || make -j 32

echo "Build complete!"
