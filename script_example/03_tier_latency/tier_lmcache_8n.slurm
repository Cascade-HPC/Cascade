#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J tier_lmcache_8n
#SBATCH -o benchmark/logs/tier_lmcache_8n_%j.out
#SBATCH -e benchmark/logs/tier_lmcache_8n_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEM="lmcache"
JOB_UID=${SLURM_JOB_ID}

for TIER_STATE in hot warm cold; do
    echo "[Tier: ${TIER_STATE}]"
    RUN_ID="${JOB_UID}_${TIER_STATE}"
    srun -N 8 -n 8 --ntasks-per-node=1 python3 script_example/01_throughput_scalability/throughput_driver.py \
        --system ${SYSTEM} \
        --num-requests 64 \
        --block-size-mb 160 \
        --tier-mode ${TIER_STATE} \
        --run-id ${RUN_ID}
    sleep 5
done
