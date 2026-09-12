#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o benchmark/logs/tier_bench_lustre_%j.out
#SBATCH -e benchmark/logs/tier_bench_lustre_%j.err
#SBATCH -J tier_lustre

module load python cudatoolkit
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

echo "========================================"
echo "Tier Latency Microbenchmark: Lustre"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"
srun python3 script_example/03_tier_latency/tier_driver.py \
    --tier lustre \
    --block-size-mb 320 \
    --num-iterations 100 \
    --output-json benchmark/results/tier_latency_lustre.json

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
