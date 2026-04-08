# CASCADE
**CASCADE: Content-Addressed Distributed KV Cache Storage for Tiered Data Architecture LLM Inference (Submitted SC'26)**

## Introduction
CASCADE is a distributed KV cache management system designed for large-scale LLM inference on disaggregated architectures.  
It integrates GPU HBM, host DRAM, and parallel file systems into a unified tiered memory hierarchy with cross-node KV cache sharing.  
To efficiently support multi-node/GPU deployments, CASCADE introduces:
- A unified tiered memory hierarchy with data placement and promotion,
- Content-addressed block management using SHA-256 hashing with global deduplication across nodes,
- Multi-path zero-copy data movement via one-sided RDMA and CUDA P2P,
- Semantic-aware eviction that protects shared prefix blocks from eviction under memory pressure.


## Key Features
- Unified tiered storage hierarchy spanning GPU HBM, DRAM, and Lustre with tier promotion/demotion.
- Content-addressed deduplication eliminates redundant KV blocks across the entire cluster.
- Zero-copy RDMA transfers via Shadow Copy Buffers with FP16→INT8 compression.
- Semantic-aware eviction protects prefix blocks through a dedicated registry and dedup map.
- Cold data cascades to Lustre via aggregated O_DIRECT writes to avoid metadata contention.
- Decoupled control-data architecture separates metadata synchronization from data movement for scalable orchestration.
  
## Repository Structure
```
CASCADE/
├── cascade/                    # Python package
│   └── __init__.py
├── cascade_code/cpp/           # Core C++ backend
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── cascade.hpp                 # Core: CascadeConfig, ShardedIndex, GPU/SHM/Lustre backends
│   │   └── cascade_distributed.hpp     # Distributed: BlockLocation, DistributedStore
│   ├── src/
│   │   ├── cascade_core.cpp            # ShardedIndex, ShmBackend (mmap+SSE2), LustreBackend, KVCompressor, PrefetchPipeline
│   │   ├── distributed_backend.cpp     # Cross-node eviction, global dedup, locality-aware placement, RDMA
│   │   └── gpu_backend.cu              # GPU memory pool, multi-stream DMA, LRU eviction
│   └── python/
│       └── bindings.cpp                # pybind11 Python bindings
├── inference_benchmark/        # End-to-end LLM inference benchmark
│   ├── config.py               # Model and CASCADE configuration presets
│   ├── engine.py               # Inference engine (prefill, decode)
│   ├── kv_manager.py           # KV cache serialization and CASCADE PUT/GET
│   ├── workload.py             # ShareGPT workload generator
│   ├── run_inference.py        # CASCADE benchmark runner
│   ├── run_vllm_baseline.py    # vLLM baseline runner
│   └── run_vllm_lmcache.py    # vLLM + LMCache baseline runner
├── build_cpp.sh                # C++ build script
└── setup_env.sh                # Environment setup
```

## Core Components
CASCADE modifies and extends the following modules:
- `cascade_core.cpp` — Tiered store orchestration with LRU eviction, tier promotion/demotion, async prefetch pipeline, and FP16→INT8 KV compression.
- `distributed_backend.cpp` — Distributed backend with cross-node semantic-aware eviction, SHA-256 global deduplication, and locality-aware placement.
- `gpu_backend.cu` — CUDA GPU backend with pre-allocated memory pool and free-list recycling.
- `cascade.hpp` / `cascade_distributed.hpp` — Header definitions for all data structures, configuration, and API interfaces.
- `bindings.cpp` — pybind11 bindings exposing the C++ backend to Python.

## Installation

### Requirements
- CUDA 12.0+
- GPU-aware MPI (e.g., Cray MPICH, OpenMPI with CUDA support)
- OpenSSL (for SHA-256 block hashing)
- pybind11 (for Python bindings)
- CMake 3.18+
- Python 3.9+

### Build C++ Backend
```bash
git clone https://github.com/Cascade-HPC/Cascade.git
cd Cascade

# Set up environment (adjust for your system)
source setup_env.sh

# Build the C++ backend with MPI and CUDA support
bash build_cpp.sh
```

This produces `cascade_cpp.cpython-*.so` in the build directory.

### Verify Installation
```python
import cascade_cpp
store = cascade_cpp.CascadeStore(cascade_cpp.CascadeConfig())
print("CASCADE loaded successfully")
```

## Usage

### Basic Store Operations
```python
import cascade_cpp
import numpy as np

# Configure CASCADE store
cfg = cascade_cpp.CascadeConfig()
cfg.gpu_capacity_bytes = 32 * 1024**3    # 32 GB GPU
cfg.shm_capacity_bytes = 64 * 1024**3    # 64 GB DRAM
cfg.shm_path = "/dev/shm/cascade"
store = cascade_cpp.CascadeStore(cfg)

# PUT
data = np.random.bytes(1024 * 1024)  # 1 MB block
block_id = cascade_cpp.compute_block_id(np.frombuffer(data, dtype=np.uint8))
store.put(block_id, np.frombuffer(data, dtype=np.uint8), len(data), is_prefix=True)

# GET
out = np.zeros(len(data), dtype=np.uint8)
success = store.get(block_id, out)
```

### Inference Benchmark
See [inference_benchmark/README.md](inference_benchmark/README.md) for end-to-end benchmark usage and configuration.
