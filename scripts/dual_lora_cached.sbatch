#!/usr/bin/env bash
#SBATCH -J ms_lora_one
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -o /home/rl182/dl/V2L/Project-meme/MultiSticker/logs/dual_lora_cached_%j.log

mkdir -p /home/rl182/dl/V2L/Project-meme/MultiSticker/logs
export RUN_SCALE="${RUN_SCALE:-pilot}"
export RUN_NAME="${RUN_NAME:-rebuild_${RUN_SCALE}}"
export TUNING_MODE="${TUNING_MODE:-dual_lora}"
bash /home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/run_dual_lora_cached.sh
