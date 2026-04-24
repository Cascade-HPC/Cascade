#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J redis_32n
#SBATCH -o benchmark/logs/redis_32n_%j.out
#SBATCH -e benchmark/logs/redis_32n_%j.err

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

echo "Starting Redis on 32 nodes..."
srun --overlap -N 4 -n 4 --ntasks-per-node=1 redis-server --port 16379 --bind 0.0.0.0 --protected-mode no --save "" &
sleep 20

SYSTEM="lmcache-redis"
NODES_LIST=(1 2 4)
JOB_UID=${SLURM_JOB_ID}

echo " Starting Redis Scaling (1-32 nodes)"
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

pkill -u $USER redis-server
echo " Finished Redis"
