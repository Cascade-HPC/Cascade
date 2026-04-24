#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J dc_repro_cas_opt_16n
#SBATCH -o benchmark/logs/deepcam_repro_cas_opt_16n_%j.out

cd ${REPO_ROOT}
source script_example/00_setup/setup_ml_env.sh
export PYTHONPATH="${REPO_ROOT}/script_example:${PYTHONPATH}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $RANDOM % 20000)

export DATA_DIR="${REPO_ROOT}/benchmark/tmp/deepcam_dummy_data_512gb"
export OUT_DIR="${REPO_ROOT}/benchmark/results/16n/cas_opt_${SLURM_JOB_ID}"
mkdir -p $OUT_DIR

echo "--------------------------------------------------------"
echo " DeepCAM 16-Node Benchmark: CAS_OPT"
echo "--------------------------------------------------------"
srun python script_example/07_deepcam/nodedup_driver.py \
       --system cascade \
       --wireup_method "mpi" \
       --run_tag "dc_repro_cas_opt_16n" \
       --data_dir_prefix ${DATA_DIR} \
       --output_dir ${OUT_DIR} \
       --model_prefix "segmentation" \
       --optimizer "Adam" \
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

echo " Done: dc_repro_cas_opt_16n"
