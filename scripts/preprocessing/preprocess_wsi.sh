#!/bin/sh
#SBATCH -p gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH -t 0:10:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem 16G 
#SBATCH -o logs/preprocess_wsi_%A_%a.out
#SBATCH -e logs/preprocess_wsi_%A_%a.err
#SBATCH --array=32

. ~/.bashrc
cd "$SLURM_SUBMIT_DIR"  # submit from the repository root
uv run python scripts/preprocessing/preprocess_wsi.py  $SLURM_ARRAY_TASK_ID
