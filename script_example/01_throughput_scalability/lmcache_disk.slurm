#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J lmcache_disk_32n
#SBATCH -o benchmark/logs/lmcache_disk_32n_%j.out
#SBATCH -e benchmark/logs/lmcache_disk_32n_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEM="lmcache"
NODES_LIST=(1 2 4)
JOB_UID=${SLURM_JOB_ID}

echo " Starting LMCache-Disk Scaling (1-32 nodes)"
for N in "${NODES_LIST[@]}"; do
    echo "   [Scale: $N Nodes]"
    REQS=$((8 * N))
    RUN_ID="${JOB_UID}_N${N}"

    srun --overlap -N $N -n $N --ntasks-per-node=1 python3 script_example/01_throughput_scalability/throughput_driver.py \
        --system $SYSTEM \
        --num-requests $REQS \
        --run-id $RUN_ID

    sleep 5
done
echo " Finished LMCache-Disk"
