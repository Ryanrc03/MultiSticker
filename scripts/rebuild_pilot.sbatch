#!/usr/bin/env bash
#SBATCH -J ms_rebuild_pilot
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o /home/rl182/dl/V2L/Project-meme/MultiSticker/logs/rebuild_pilot_%j.log

mkdir -p /home/rl182/dl/V2L/Project-meme/MultiSticker/logs
bash /home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/run_rebuild_pilot.sh
