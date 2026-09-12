#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J dedup_cascade_prefixrep
#SBATCH -o benchmark/logs/dedup_cascade_prefixrep_%j.out
#SBATCH -e benchmark/logs/dedup_cascade_prefixrep_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEM="cascade"
BLOCK_MB=160
PREFIX_BLOCKS=64

echo "=== Global Prefix Sharing (prefix_replication=ON) ==="
echo "System: ${SYSTEM}, ${PREFIX_BLOCKS} blocks x ${BLOCK_MB}MB"
echo "Feature: broadcast_prefix_data() — all nodes get local copies"
echo ""
for i in 1 2 4 8; do
    echo ""
    echo " SCALE: N=${i} nodes"
    srun --overlap -N $i -n $i --ntasks-per-node=1 \
        python3 script_example/05_sensitivity/prefix/dedup_prefix_driver.py \
        --system $SYSTEM \
        --block-size-mb $BLOCK_MB \
        --prefix-blocks $PREFIX_BLOCKS \
        --run-id "${SLURM_JOB_ID}_DEDUP_N${i}"
    sleep 5
done

echo " Done: Dedup Prefix Sharing (prefix_replication)"
