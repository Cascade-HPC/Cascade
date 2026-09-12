#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J oversubscription_8n
#SBATCH -o benchmark/logs/oversubscription_8n_%j.out
#SBATCH -e benchmark/logs/oversubscription_8n_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEMS="cascade lmcache redis pdc hdf5"
RATIOS="1 2 4 8 10 16"

for SYSTEM in ${SYSTEMS}; do
    for RATIO in ${RATIOS}; do
        echo "[System: ${SYSTEM}, Oversub: ${RATIO}x]"
        TOTAL_GB=$(echo "${RATIO} * 160" | bc)
        srun -N 8 -n 8 --ntasks-per-node=1 python3 \
            script_example/05_sensitivity/oversubscription/oversubscription_driver.py \
            --system ${SYSTEM} \
            --node-count 8 \
            --block-size-mb 160 \
            --total-data-per-node-gb ${TOTAL_GB}
        sleep 5
    done
done
