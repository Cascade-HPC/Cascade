# CASCADE
**CASCADE: Content-Addressed Distributed KV Cache Storage for Tiered Data Architecture LLM Inference (Submitted FAST'27)**

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
  
## Core Components
- `cascade_core.cpp` — Tiered store orchestration with LRU eviction, tier promotion/demotion, async prefetch pipeline, and FP16→INT8 KV compression.
- `distributed_backend.cpp` — Distributed backend with cross-node semantic-aware eviction, SHA-256 global deduplication, and locality-aware placement.
- `gpu_backend.cu` — CUDA GPU backend with pre-allocated memory pool and free-list recycling.
- `cascade.hpp` / `cascade_distributed.hpp` — Header definitions for all data structures, configuration, and API interfaces.
- `sglang_slots.cu` — CUDA kernels that gather and scatter a chunk of scattered KV token slots in one launch, so the device batch API can carry it.
- `bindings.cpp` — pybind11 bindings exposing the C++ backend to Python.
- `vllm/` and `sglang/` — adapters that let each engine keep its KV cache in CASCADE. Neither holds a cache policy of its own. Tier placement, eviction, deduplication and RDMA stay in the modules above, and an adapter only maps its engine's blocks onto them and answers that engine's scheduler. Both attach without modifying a file inside the engine's tree. See [cascade_code/vllm/README.md](cascade_code/vllm/README.md) and [cascade_code/sglang/README.md](cascade_code/sglang/README.md).

## Installation

### Requirements
- CUDA 12.0+
- GPU-aware MPI (a vendor MPI with GPU support, or OpenMPI built with CUDA)
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
The default configuration reserves 32 GB of GPU memory and 64 GB of DRAM, so
verify with a small host only store.

```python
import cascade_cpp

cfg = cascade_cpp.CascadeConfig()
cfg.use_gpu = False
cfg.shm_capacity_bytes = 64 * 1024**2
cfg.shm_path = "/dev/shm/cascade_check"
store = cascade_cpp.CascadeStore(cfg)
print("CASCADE loaded successfully")
```

## Usage

### Basic Store Operations
```python
import cascade_cpp
import numpy as np

cfg = cascade_cpp.CascadeConfig()
cfg.gpu_capacity_bytes = 8 * 1024**3
cfg.shm_capacity_bytes = 16 * 1024**3
cfg.shm_path = "/dev/shm/cascade"
store = cascade_cpp.CascadeStore(cfg)

data = np.frombuffer(np.random.bytes(1024 * 1024), dtype=np.uint8)
block_id = cascade_cpp.compute_block_id(data)

stored = store.put(block_id, data, is_prefix=True)

out = np.zeros(data.size, dtype=np.uint8)
found, size = store.get(block_id, out)
```

`compute_block_id` returns the SHA-256 of the block, which is the identity the
store deduplicates on. `is_prefix` marks a block as part of a shared prefix,
which the eviction policy protects. `get` returns whether the block was found
and how many bytes were written.

### Reproducing paper results
Two kinds of experiment are reproduced from this repository. The trace-driven
experiments replay a KV cache workload against the store and against each
baseline, covering throughput, tail latency, per-tier latency, block size and
sensitivity. The DeepCAM experiment is workload-specific reconfiguration, where
the store is driven by an ML training workload rather than by inference.

[script_example/](script_example/) holds one directory per experiment, each
with the Slurm scripts and driver for CASCADE and for the baselines it is
compared against. See [script_example/README.md](script_example/README.md).

### Serving engines
CASCADE attaches to vLLM and to SGLang as their external KV store. Neither
integration modifies a file inside the engine's tree. See
[cascade_code/vllm/README.md](cascade_code/vllm/README.md) and
[cascade_code/sglang/README.md](cascade_code/sglang/README.md) for the build,
the registration and the configuration of each.

## Repository Structure
```
CASCADE/
├── cascade_code/
│   ├── cpp/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   ├── cascade.hpp
│   │   │   ├── cascade_distributed.hpp
│   │   │   └── sglang_slots.hpp
│   │   ├── src/
│   │   │   ├── cascade_core.cpp
│   │   │   ├── distributed_backend.cpp
│   │   │   ├── gpu_backend.cu
│   │   │   └── sglang_slots.cu
│   │   └── python/
│   │       └── bindings.cpp
│   ├── vllm/
│   │   ├── README.md
│   │   └── cascade_vllm/
│   │       ├── connector.py
│   │       ├── manager.py
│   │       ├── handler.py
│   │       └── store.py
│   └── sglang/
│       ├── README.md
│       └── cascade_sglang/
│           ├── hook.py
│           ├── radix_cache.py
│           ├── backend.py
│           ├── shm_pool.py
│           ├── device_pool.py
│           ├── phase_timing.py
│           └── store.py
├── script_example/
│   ├── 00_setup/
│   ├── 01_throughput_scalability/
│   ├── 02_tail_latency_burst/
│   ├── 03_tier_latency/
│   ├── 04_variable_blocks/
│   ├── 05_sensitivity/
│   ├── 06_deepcam/
│   └── benchmark/
├── build_cpp.sh
└── setup_env.sh
```
