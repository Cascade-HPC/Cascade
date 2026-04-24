#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J tail_cdf_8n
#SBATCH -o benchmark/logs/tail_cdf_8n_%j.out

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

SYSTEMS="cascade pdc lmcache"
BLOCK_SIZES="160 320"

echo "=== Tail Latency CDF Data Collection (8 Nodes) ==="
for sys in $SYSTEMS; do
    for bs in $BLOCK_SIZES; do
        model_name="Llama"
        if [ "$bs" -eq "320" ]; then model_name="Qwen"; fi

        echo " Testing $sys with $model_name ($bs MB blocks)"
        srun -N 8 -n 8 python3 script_example/02_tail_latency_burst/tail_driver.py \
            --system $sys \
            --block-size-mb $bs \
            --num-write-blocks 50 \
            --num-read-ops 500
        sleep 5
    done
done

echo " All CDF data collected in benchmark/tmp/tail_res_<JOBID>"
