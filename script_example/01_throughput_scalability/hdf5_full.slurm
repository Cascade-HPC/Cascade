#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J hdf5_indep_32n
#SBATCH -o benchmark/logs/hdf5_indep_32n_%j.out
#SBATCH -e benchmark/logs/hdf5_indep_32n_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEM="hdf5-indep"
NODES_LIST=(1 2 4 8 16 32 64)
JOB_UID=${SLURM_JOB_ID}

echo " Starting HDF5-Independent Scaling (1-32 nodes)"
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
echo " Finished HDF5-Indep"
