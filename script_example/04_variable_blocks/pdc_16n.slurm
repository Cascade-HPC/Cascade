#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -J 16n_pdc
#SBATCH -o benchmark/logs/var_16n_pdc_%j.out
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

echo "=== Variable Block 16-Node Only: pdc ==="
if [ "pdc" == "redis" ]; then
    mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID
    hostname > benchmark/tmp/hosts_$SLURM_JOB_ID/redis_host
    ${HOME}/.conda/envs/cascade_env/bin/redis-server --port 16379 --protected-mode no --save "" --appendonly no --maxmemory 100gb --maxmemory-policy allkeys-lru &
    REDIS_PID=$!
    sleep 30
fi

srun -N 16 -n 16 python3 script_example/04_variable_blocks/variable_block_driver.py \
    --system pdc \
    --block-size-mb 160 --num-write-blocks 50 --num-read-ops 200 --sigma 0.8 \
    --redis-port 16379

if [ "pdc" == "redis" ]; then
    kill $REDIS_PID 2>/dev/null || true
    rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID
fi
