#!/usr/bin/env bash
#SBATCH -J ms_rebuild_medium
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -o /scratch/rl182/mutlisticker/logs/rebuild_medium_%j.log

mkdir -p /scratch/rl182/mutlisticker/logs
export RUN_SCALE="${RUN_SCALE:-medium}"
bash /home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/run_rebuild_pilot.sh
