#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -J tail_cdf_${n}n
#SBATCH -o benchmark/logs/tail_cdf_${n}n_%j.out
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEMS="redis hdf5"
BLOCK_SIZES="160 320"

echo "=== Tail Latency CDF Data Collection (1 Nodes) ==="
for sys in $SYSTEMS; do
    if [ "$sys" == "redis" ]; then
        echo ">>> Starting Redis Server..."
        mkdir -p benchmark/tmp/hosts_$SLURM_JOB_ID
        hostname > benchmark/tmp/hosts_$SLURM_JOB_ID/redis_host
        redis-server --port 16379 --protected-mode no --save "" --appendonly no &
        REDIS_PID=$!
        sleep 5
    fi

    for bs in $BLOCK_SIZES; do
        model_name="Llama"
        if [ "$bs" -eq "320" ]; then model_name="Qwen"; fi
        echo " Testing $sys with $model_name ($bs MB blocks)"
        srun -N 1 -n 1 python3 script_example/02_tail_latency_burst/tail_driver.py \
            --system $sys --block-size-mb $bs --num-write-blocks 30 --num-read-ops 300
        sleep 5
    done

    if [ "$sys" == "redis" ]; then
        kill $REDIS_PID
        rm -rf benchmark/tmp/hosts_$SLURM_JOB_ID
    fi
done
