#!/usr/bin/env bash
#SBATCH -J ms_demo_png
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -o /scratch/rl182/mutlisticker/logs/export_dual_lora_png_demos_%j.log

set -euo pipefail

cd /home/rl182/dl/V2L/Project-meme/MultiSticker

export HF_HOME=/scratch/rl182/meme/models/hf
export HUGGINGFACE_HUB_CACHE=/scratch/rl182/meme/models/hf
export TRANSFORMERS_CACHE=/scratch/rl182/meme/models/hf
export TORCH_HOME=/scratch/rl182/meme/models/torch
export XDG_CACHE_HOME=/scratch/rl182/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

module load GCCcore/13.3.0 FFmpeg/7.0.2 2>/dev/null || true
export FFMPEG_BIN="$(which ffmpeg 2>/dev/null || echo '')"

/scratch/rl182/envs/dl/bin/python -u scripts/export_dual_lora_png_demos.py --num-cases 50 --max-per-gold 1
