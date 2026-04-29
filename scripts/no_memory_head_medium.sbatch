#!/usr/bin/env bash
#SBATCH -J ms_nm_head
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -o /scratch/rl182/mutlisticker/logs/no_memory_head_medium_%j.log

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

PY=/scratch/rl182/envs/dl/bin/python
TRAIN_SCRIPT=/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py
ARTIFACT_ROOT=/scratch/rl182/mutlisticker

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/results" "$ARTIFACT_ROOT/embedding_cache"

"$PY" -u "$TRAIN_SCRIPT" \
  --tuning-mode head_only \
  --memory-strategy disabled \
  --device cuda \
  --supported-media ".png,.gif,.webm" \
  --max-train-samples 30000 \
  --max-val-samples 5000 \
  --max-stickers 8000 \
  --min-sticker-frequency 2 \
  --session-memories-file /scratch/rl182/meme/usticker_igsr/llm/session_memories_qwen32_gptq_v10_png_gif_webm_merged.jsonl \
  --sample-intents-file /scratch/rl182/meme/usticker_igsr/llm/sample_intents_qwen32_gptq_v10_png_gif_webm_merged.jsonl \
  --run-name rebuild_medium_no_memory_head \
  --force-rebuild \
  --epochs 5 \
  --train-batch-size 256 \
  --infer-batch-size 256 \
  --intent-clusters 64 \
  --num-workers "$OMP_NUM_THREADS" \
  --log-every 25 \
  --log-samples 8 \
  --log-dir "$ARTIFACT_ROOT/logs" \
  --results-dir "$ARTIFACT_ROOT/results" \
  --embedding-cache-dir "$ARTIFACT_ROOT/embedding_cache" \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-lr 0.0001 \
  --head-lr 0.001
