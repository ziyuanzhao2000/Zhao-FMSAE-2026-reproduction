#!/bin/sh
#SBATCH -p gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH -t 0:15:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem 20G 
#SBATCH -o logs/extract_token_labels_%A_%a.out
#SBATCH -e logs/extract_token_labels_%A_%a.err
#SBATCH --array=15,21,34

. ~/.bashrc
cd "$SLURM_SUBMIT_DIR"  # submit from the repository root
/usr/bin/time -v uv run python scripts/features/extract_token_labels.py  $SLURM_ARRAY_TASK_ID
