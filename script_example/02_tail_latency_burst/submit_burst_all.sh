#!/bin/bash

cd ${REPO_ROOT}

echo "=========================================="
echo "Submitting Bursty Traffic Experiments"
echo "=========================================="
mkdir -p benchmark/logs

SYSTEMS="cascade hdf5 lmcache pdc redis"

for sys in $SYSTEMS; do
    script="benchmark/scripts/bursty_${sys}_32n.slurm"
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        sbatch "$script"
        sleep 1
    else
        echo "WARNING: Script not found: $script"
    fi
done

echo "=========================================="
echo "All bursty traffic jobs submitted!"
echo "Check status with: squeue -u \$USER"
echo "=========================================="
