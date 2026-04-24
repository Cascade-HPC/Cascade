#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -J vllm_lmcache_8n
#SBATCH -o benchmark/logs/vllm_lmcache_8n_%j.out
#SBATCH -e benchmark/logs/vllm_lmcache_8n_%j.err

cd ${REPO_ROOT}
source script_example/00_setup/setup_ml_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

srun -N 8 -n 8 --gpus-per-node=4 python3 script_example/06_e2e_inference/vllm_lmcache_driver.py \
    --model ${REPO_ROOT}/models/Qwen2.5-72B \
    --num-sessions 50 \
    --turns-per-session 3 \
    --max-new-tokens 1 \
    --prefix-tokens 1024 \
    --dataset ${REPO_ROOT}/benchmark/data_external/sharegpt/sharegpt_cleaned.json \
    --results-dir benchmark/results/vllm_lmcache_8n_${SLURM_JOB_ID}
