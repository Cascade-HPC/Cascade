#!/bin/bash

echo "Submitting Variable Block Size Experiments to SLURM..."
for sys in cascade hdf5 lmcache pdc redis; do
    sbatch script_example/04_variable_blocks/${sys}_16n.slurm
done

echo "Variable block experiments submitted."
