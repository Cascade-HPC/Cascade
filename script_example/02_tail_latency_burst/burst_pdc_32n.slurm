#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J bursty_pdc_32n
#SBATCH -o benchmark/logs/bursty_pdc_32n_%j.out
#SBATCH -e benchmark/logs/bursty_pdc_32n_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEM="pdc"
N=8
BLOCK_SIZE=160

echo "=== Bursty Traffic Test: ${SYSTEM} N=${N} ==="
srun -N $N -n $N --ntasks-per-node=1 \
    python3 script_example/02_tail_latency_burst/burst_driver.py \
    --system $SYSTEM \
    --node-count $N \
    --block-size-mb $BLOCK_SIZE

echo " Cleaning up data..."
rm -rf benchmark/pdc_bursty_${SLURM_JOB_ID}
rm -rf benchmark/tmp/bursty_res_${SLURM_JOB_ID}

echo " Done"
