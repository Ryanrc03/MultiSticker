#!/usr/bin/env bash
# Run one LoRA experiment while reusing the run manifest and the shared
# embedding cache directory where applicable.
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

RUN_SCALE="${RUN_SCALE:-pilot}"
RUN_NAME="${RUN_NAME:-rebuild_${RUN_SCALE}}"
TUNING_MODE="${TUNING_MODE:-dual_lora}"
case "$TUNING_MODE" in
    image_lora|text_lora|dual_lora) ;;
    *)
        echo "[single_lora] Unknown TUNING_MODE=$TUNING_MODE (expected image_lora, text_lora, or dual_lora)" >&2
        exit 2
        ;;
esac
case "$RUN_SCALE" in
    pilot)
        DEFAULT_MAX_TRAIN=5000
        DEFAULT_MAX_VAL=2000
        DEFAULT_MAX_STICKERS=3000
        DEFAULT_EPOCHS=2
        DEFAULT_LORA_BS=64
        DEFAULT_INFER_BS=128
        ;;
    medium)
        DEFAULT_MAX_TRAIN=30000
        DEFAULT_MAX_VAL=5000
        DEFAULT_MAX_STICKERS=8000
        DEFAULT_EPOCHS=5
        DEFAULT_LORA_BS=96
        DEFAULT_INFER_BS=256
        ;;
    *)
        echo "[dual_lora] Unknown RUN_SCALE=$RUN_SCALE (expected pilot or medium)" >&2
        exit 2
        ;;
esac

MAX_TRAIN="${MAX_TRAIN:-$DEFAULT_MAX_TRAIN}"
MAX_VAL="${MAX_VAL:-$DEFAULT_MAX_VAL}"
MAX_STICKERS="${MAX_STICKERS:-$DEFAULT_MAX_STICKERS}"
MIN_FREQ="${MIN_FREQ:-2}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
LORA_BS="${LORA_BS:-$DEFAULT_LORA_BS}"
INFER_BS="${INFER_BS:-$DEFAULT_INFER_BS}"
NUM_WORKERS="${NUM_WORKERS:-$OMP_NUM_THREADS}"
LOG_EVERY="${LOG_EVERY:-25}"
LOG_SAMPLES="${LOG_SAMPLES:-8}"
INTENT_CLUSTERS="${INTENT_CLUSTERS:-64}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_LR="${LORA_LR:-0.0001}"
HEAD_LR="${HEAD_LR:-0.001}"
SESSION_MEMORIES="${SESSION_MEMORIES:-/scratch/rl182/meme/usticker_igsr/llm/session_memories_qwen32_gptq_v10_png_gif_webm_merged.jsonl}"
SAMPLE_INTENTS="${SAMPLE_INTENTS:-/scratch/rl182/meme/usticker_igsr/llm/sample_intents_qwen32_gptq_v10_png_gif_webm_merged.jsonl}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/rl182/dl/V2L/Project-meme/MultiSticker}"
LOGDIR="${LOGDIR:-$ARTIFACT_ROOT/logs}"
RESULTDIR="${RESULTDIR:-$ARTIFACT_ROOT/results}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-/scratch/rl182/mutlisticker/embedding_cache}"
ORGANIZED_DIR="${ORGANIZED_DIR:-$ARTIFACT_ROOT/organized_runs/$RUN_NAME}"
mkdir -p "$LOGDIR" "$RESULTDIR" "$EMBEDDING_CACHE_DIR" "$ORGANIZED_DIR"

LOG="$LOGDIR/${TUNING_MODE}_retrieved_topk_${RUN_NAME}.log"
PREFIX="am_${TUNING_MODE}_retrieved_topk_${RUN_NAME}"

echo "[single_lora] START mode=$TUNING_MODE scale=$RUN_SCALE run=$RUN_NAME train_bs=$LORA_BS infer_bs=$INFER_BS cache=$EMBEDDING_CACHE_DIR log=$LOG"
if "$PY" -u "$TRAIN_SCRIPT" \
    --tuning-mode "$TUNING_MODE" \
    --memory-strategy retrieved_topk \
    --device cuda \
    --supported-media ".png,.gif,.webm" \
    --max-train-samples "$MAX_TRAIN" \
    --max-val-samples "$MAX_VAL" \
    --max-stickers "$MAX_STICKERS" \
    --min-sticker-frequency "$MIN_FREQ" \
    --session-memories-file "$SESSION_MEMORIES" \
    --sample-intents-file "$SAMPLE_INTENTS" \
    --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --train-batch-size "$LORA_BS" \
    --infer-batch-size "$INFER_BS" \
    --intent-clusters "$INTENT_CLUSTERS" \
    --num-workers "$NUM_WORKERS" \
    --log-every "$LOG_EVERY" \
    --log-samples "$LOG_SAMPLES" \
    --log-dir "$LOGDIR" \
    --results-dir "$RESULTDIR" \
    --embedding-cache-dir "$EMBEDDING_CACHE_DIR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-lr "$LORA_LR" \
    --head-lr "$HEAD_LR" \
    >"$LOG" 2>&1
then
    echo "[single_lora] DONE mode=$TUNING_MODE run=$RUN_NAME"
    find "$LOGDIR" "$RESULTDIR" -maxdepth 1 -type f -name "*${PREFIX#am_}*" -exec cp -p -t "$ORGANIZED_DIR" {} +
    find "$RESULTDIR" -maxdepth 1 -type f -name "${PREFIX}*" -exec cp -p -t "$ORGANIZED_DIR" {} +
else
    status=$?
    echo "[single_lora] FAILED mode=$TUNING_MODE run=$RUN_NAME status=$status (see $LOG)"
    cp -p "$LOG" "$ORGANIZED_DIR/" 2>/dev/null || true
    exit "$status"
fi
