# CASCADE — Example Scripts

Self-contained Slurm scripts, drivers and adapters for reproducing the CASCADE
results. One directory per experiment, each with the script for CASCADE and for
the baselines it is compared against.

## Layout

| Directory | What it measures |
|---|---|
| `00_setup/` | environment and build |
| `01_throughput_scalability/` | aggregate throughput as nodes are added |
| `02_tail_latency_burst/` | tail latency under bursty arrivals |
| `03_tier_latency/` | read latency of each storage tier |
| `04_variable_blocks/` | behaviour across block sizes |
| `05_sensitivity/prefix/` | shared prefix reuse |
| `05_sensitivity/oversubscription/` | a store smaller than the working set |
| `05_sensitivity/dedup/` | deduplication across nodes |
| `06_deepcam/` | an ML training workload rather than inference |
| `benchmark/` | shared adapters and driver |

## Environment

Set before submission.

```
export REPO_ROOT=/path/to/cascade
export SCRATCH=/your/scratch
```

Then build.

```
bash 00_setup/setup_env.sh
bash 00_setup/build_cpp.sh
```

Replace `#SBATCH -A <account>` with your allocation.

Each Slurm script prepends `${REPO_ROOT}/script_example` to `PYTHONPATH`, so a
driver's `from benchmark.run_benchmark import get_adapter` resolves to the
adapters shipped here.

## Data

Download ShareGPT to `${REPO_ROOT}/benchmark/data_external/sharegpt/sharegpt_cleaned.json`.

Trace-driven drivers fall back to an in-memory synthetic workload when no
extracted KV cache is present at `${SCRATCH}/cascade_kv_cache/`. To run against
a real one, stage it there by running a serving engine with the CASCADE
integration over ShareGPT at one output token per request.

## Running

```
sbatch 01_throughput_scalability/cascade_full.slurm
sbatch 02_tail_latency_burst/tail_{1,8,32,64}n.slurm
bash   02_tail_latency_burst/submit_burst_all.sh
sbatch 03_tier_latency/tier_{cascade,lmcache,hdf5,pdc}_{8n,64n}.slurm
bash   04_variable_blocks/submit_all.sh
sbatch 05_sensitivity/prefix/{cascade,lmcache,hdf5,pdc,redis}_prefix.slurm
sbatch 05_sensitivity/oversubscription/oversubscription_8n.slurm
sbatch 05_sensitivity/dedup/{cascade,hdf5,lmcache,pdc,redis}_8n.slurm
sbatch 06_deepcam/{original,nodedup,streaming}_{16n,64n}.slurm
```

End-to-end serving is not here. CASCADE attaches to vLLM and to SGLang as a
store the engine loads rather than as a job, so it is driven by the engine's
own benchmark. See [../cascade_code/vllm](../cascade_code/vllm) and
[../cascade_code/sglang](../cascade_code/sglang).

To change the node count, edit `#SBATCH -N` in the Slurm file and any node list
in the script.

## Baselines

| name | what it is |
|---|---|
| LMCache-Disk | LMCache with a shared filesystem backend |
| LMCache-Redis | LMCache with a centralized 128 GB Redis backend |
| HDF5 | parallel I/O in independent mode |
| PDC | object-centric data management |
| CASCADE | this work |

## Notes

- Model weights are not bundled. Download Llama-3-70B and Qwen-2.5-72B from
  HuggingFace into `${REPO_ROOT}/models/`.
  ```
  huggingface-cli login
  python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Meta-Llama-3-70B', local_dir='models/Llama-3-70B')"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-72B', local_dir='models/Qwen2.5-72B')"
  ```
- `06_deepcam/` requires the MLPerf HPC benchmark source and dataset
  (https://github.com/mlcommons/hpc/tree/main/deepcam). Clone it, export
  `DEEPCAM_SRC=/path/to/mlcommons_hpc/deepcam/src/deepCam`, and stage the
  512 GB dataset per the MLCommons instructions.
