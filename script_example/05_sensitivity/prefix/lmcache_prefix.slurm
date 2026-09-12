#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J cascade_weak_prefixrep
#SBATCH -o benchmark/logs/cascade_weak_prefixrep_%j.out
#SBATCH -e benchmark/logs/cascade_weak_prefixrep_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEM="lmcache"
JOB_UID=${SLURM_JOB_ID}

echo "=== CASCADE WEAK SCALING 1-8N (Llama 160MB, 8 req/node) ==="
echo "Feature: prefix_replication=ON (broadcast_prefix_data)"
echo ""
for N in 1 2 4 8; do
    echo ""
    echo "--- Weak Scaling N=${N} nodes (REQS=$((8 * N))) ---"
    REQS=$((8 * N))
    RUN_ID="${JOB_UID}_WEAK_N${N}"

    srun --overlap -N $N -n $N --ntasks-per-node=1 \
        python3 script_example/01_throughput_scalability/throughput_driver.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID

    sleep 10
done

echo " Done: Cascade Weak Scaling 1-8N (prefix_replication)"
