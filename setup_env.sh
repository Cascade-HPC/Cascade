#!/bin/bash

module reset || true

module load python/3.9
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cascade_env

module load PrgEnv-gnu
module load gcc-native/12.3
module load cray-mpich
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load cmake

module unload darshan

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export CUDAHOSTCXX=$(which g++)
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

REPO_ROOT=$(pwd)
export PYTHONPATH=$REPO_ROOT/cascade_Code/cpp/build_cascade_cpp:$PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

echo "--------------------------------------------------------"
echo "Cascade GPU-Aware MPI Environment Loaded"
echo "--------------------------------------------------------"
echo "Python: $(which python3)"
echo "C++: $(which g++) ($(g++ --version | head -n 1))"
echo "MPI CC: $(which CC)"
echo "--------------------------------------------------------"
