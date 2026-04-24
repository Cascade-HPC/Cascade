#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J deepcam_cascade
#SBATCH -o benchmark/logs/deepcam_cascade_%j.out

cd ${REPO_ROOT}
source setup_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
export MPICH_GPU_SUPPORT_ENABLED=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export DATA_DIR="${REPO_ROOT}/benchmark/tmp/deepcam_dummy_data"
export OUT_DIR="${REPO_ROOT}/benchmark/results/deepcam_cascade_${SLURM_JOB_ID}"
export CASCADE_STORE_DIR="${REPO_ROOT}/benchmark/tmp/cascade_store_${SLURM_JOB_ID}"

echo "--------------------------------------------------------"
echo " DeepCAM Benchmark (CASCADE HOT)"
echo "--------------------------------------------------------"
srun python script_example/07_deepcam/deepcam_driver.py \
       --use_cascade \
       --cascade_lustre_path ${CASCADE_STORE_DIR} \
       --wireup_method "mpi" \
       --run_tag "deepcam_cascade_8n" \
       --data_dir_prefix ${DATA_DIR} \
       --output_dir ${OUT_DIR} \
       --model_prefix "segmentation" \
       --optimizer "LAMB" \
       --start_lr 0.0055 \
       --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
       --lr_warmup_steps 400 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --logging_frequency 10 \
       --save_frequency 0 \
       --max_epochs 1 \
       --max_inter_threads 4 \
       --batchnorm_group_size 1 \
       --local_batch_size 2

echo " Done: deepcam_cascade"
