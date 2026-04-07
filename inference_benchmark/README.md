# Cascade Inference Benchmark

End-to-end LLM inference benchmark with Cascade KV cache backend. Runs actual Qwen-2.5-72B model inference and measures TTFT (Time To First Token) with Cascade's distributed KV cache management.

## How It Works

```
Request arrives (ShareGPT multi-turn prompt)
    │
    ├─ prefix_hash = SHA256(prefix_token_ids)
    ├─ Cascade GET(prefix_hash)?
    │
    ├── HIT:  deserialize → DynamicCache → partial prefill (new tokens only) → first token
    │         TTFT = Cascade GET + partial prefill
    │
    └── MISS: full GPU prefill → extract KV cache → serialize → Cascade PUT → first token
              TTFT = full prefill + Cascade PUT
```

## Prerequisites

### System Requirements
- HPC cluster with NVIDIA GPUs and high-speed interconnect (Slingshot/InfiniBand)
- NVIDIA A100 40GB (short/medium blocks) or 80GB (large blocks)
- 4 GPUs per node (for Qwen-2.5-72B TP=4)

### Software
- Python 3.11
- PyTorch 2.5+ with CUDA support
- Cascade C++ backend (`cascade_cpp.so` built for Python 3.11)

## Installation

### 1. Build Cascade C++ Backend (Python 3.11)

Submit the build job on a compute node:

```bash
sbatch inference_benchmark/scripts/build_cascade_py311.slurm
```

This produces `cascade_Code/cpp/build_py311/cascade_cpp.cpython-311-x86_64-linux-gnu.so`.

### 2. Install Python Dependencies

```bash
module load python/3.11-24.1.0

pip install --target=inference_benchmark/deps transformers accelerate bitsandbytes

rm -rf inference_benchmark/deps/numpy* inference_benchmark/deps/scipy*
```

### 3. Download Model Weights

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-72B', local_dir='models/Qwen2.5-72B')
"
```

This downloads ~136GB to `models/Qwen2.5-72B/`.

### 4. Verify ShareGPT Dataset

```bash
ls -lh benchmark/data_external/sharegpt/sharegpt_cleaned.json
```

## Usage

### Environment Setup (required before every run)

```bash
module load PrgEnv-gnu
module load python/3.11-24.1.0
module load cudatoolkit
module load cray-mpich

export PYTHONNOUSERSITE=1
export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/cascade_Code/cpp/build_py311:$PROJECT_DIR/inference_benchmark/deps:$PYTHONPATH
export HF_HOME=$PROJECT_DIR/models
```

### Quick Test (1 Node, Short Prefix)

```bash
sbatch inference_benchmark/scripts/run_qwen72b_1n.slurm
```

### Block Size Configurations

| Block Size | Tokens | SLURM Script (1N) | SLURM Script (Multi-N) | Hardware |
|---|---|---|---|---|
| Short (~MB) | ~50 | `run_qwen72b_1n.slurm` | `run_qwen72b_multi_fp16_short.slurm` | A100-40GB |
| 160MB | 512 | `run_qwen72b_1n_fp16_160mb_80g.slurm` | `run_qwen72b_multi_fp16_160mb_80g.slurm` | A100-80GB |
| 320MB | 1024 | `run_qwen72b_1n_fp16_longctx_80g.slurm` | `run_qwen72b_multi_fp16_longctx_80g.slurm` | A100-80GB |

### Multi-Node Scaling

```bash
# 2 nodes
sbatch -N 2 inference_benchmark/scripts/run_qwen72b_multi_fp16_longctx_80g.slurm

# 4 nodes
sbatch -N 4 inference_benchmark/scripts/run_qwen72b_multi_fp16_longctx_80g.slurm

# 8 nodes
sbatch -N 8 inference_benchmark/scripts/run_qwen72b_multi_fp16_longctx_80g.slurm
```


### Custom Run Example

```bash
srun -N 1 --gpus-per-node=4 python -m inference_benchmark.run_inference \
    --model models/Qwen2.5-72B \
    --num-sessions 50 \
    --turns-per-session 2 \
    --max-new-tokens 1 \
    --prefix-tokens 1024 \
    --gpu-capacity 40.0 \
    --shm-capacity 200.0 \
    --no-compression \
    --dataset benchmark/data_external/sharegpt/sharegpt_cleaned.json \
    --results-dir inference_benchmark/results
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `models/Qwen2.5-72B` | HuggingFace model path |
| `--num-sessions` | 50 | Number of user sessions |
| `--turns-per-session` | 2 | Conversation turns per session |
| `--max-new-tokens` | 1 | Tokens to generate (1 = TTFT only) |
| `--prefix-tokens` | 0 | System prompt length (0=short, 512=160MB, 1024=320MB) |
| `--gpu-capacity` | 0.0 | Cascade GPU pool per device (GB) |
| `--shm-capacity` | 200.0 | Cascade DRAM pool per node (GB) |
| `--quantization` | none | Model quantization (none/int4/int8) |
| `--no-compression` | false | Disable Cascade KV compression |
| `--warmup` | 1 | Warmup requests before measurement |

## Output

Results are saved as JSON in `inference_benchmark/results/`:

```json
{
  "config": { "model": "...", "kv_block_mb": 320, ... },
  "summary": {
    "total_requests": 81,
    "cache_hits": 49,
    "hit_rate": 0.605,
    "avg_miss_ttft_ms": 1233.0,
    "avg_hit_ttft_ms": 175.0,
    "speedup": 7.0
  },
  "per_request": [ ... ]
}
```

SLURM logs are in `inference_benchmark/logs/`.


## Notes

- **TTFT measurement**: Excludes decode time. Measures time from request arrival to first token generation.
- **Deserialize time**: Python-level bytes-to-tensor conversion is excluded from TTFT as it is benchmark glue code overhead, not Cascade performance. In production, this would be handled by zero-copy mechanisms.
- **Model loading**: Takes ~4 minutes per node. This is a one-time server startup cost, not included in TTFT.
- **A100-40GB vs 80GB**: 40GB nodes can run short prefix only. 160MB/320MB blocks require 80GB nodes to accommodate model weights + KV cache.
