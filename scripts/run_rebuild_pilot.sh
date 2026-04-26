#!/usr/bin/env bash
# Rebuild pilot: runs all 5 experiment rows at pilot scale.
# Usage:
#   bash run_rebuild_pilot.sh                          # defaults
#   RUN_NAME=my_run MAX_TRAIN=5000 ... bash run_rebuild_pilot.sh
set -euo pipefail
cd /home/rl182/dl/V2L/Project-meme/MultiSticker

export HF_HOME=/scratch/rl182/meme/models/hf
export HUGGINGFACE_HUB_CACHE=/scratch/rl182/meme/models/hf
export TRANSFORMERS_CACHE=/scratch/rl182/meme/models/hf
export TORCH_HOME=/scratch/rl182/meme/models/torch
export XDG_CACHE_HOME=/scratch/rl182/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load GCCcore/13.3.0 FFmpeg/7.0.2 2>/dev/null || true
export FFMPEG_BIN="$(which ffmpeg 2>/dev/null || echo '')"

PY=/scratch/rl182/envs/dl/bin/python
TRAIN_SCRIPT=/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py
CLIP_SCRIPT=/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/eval_direct_clip.py
LOGDIR=/home/rl182/dl/V2L/Project-meme/MultiSticker/logs
mkdir -p "$LOGDIR"

RUN_NAME="${RUN_NAME:-rebuild_pilot}"
MAX_TRAIN="${MAX_TRAIN:-5000}"
MAX_VAL="${MAX_VAL:-2000}"
MAX_STICKERS="${MAX_STICKERS:-3000}"
MIN_FREQ="${MIN_FREQ:-2}"
EPOCHS="${EPOCHS:-2}"
SESSION_MEMORIES="${SESSION_MEMORIES:-/scratch/rl182/meme/usticker_igsr/llm/session_memories_qwen32_gptq_v10_png_gif_webm_merged.jsonl}"
SAMPLE_INTENTS="${SAMPLE_INTENTS:-/scratch/rl182/meme/usticker_igsr/llm/sample_intents_qwen32_gptq_v10_png_gif_webm_merged.jsonl}"

COMMON_ARGS=(
    --device cuda
    --supported-media ".png,.gif,.webm"
    --max-train-samples "$MAX_TRAIN"
    --max-val-samples "$MAX_VAL"
    --max-stickers "$MAX_STICKERS"
    --min-sticker-frequency "$MIN_FREQ"
    --session-memories-file "$SESSION_MEMORIES"
    --sample-intents-file "$SAMPLE_INTENTS"
    --run-name "$RUN_NAME"
)

run_train() {
    local MODE="$1"
    local MEM_STRATEGY="$2"
    local EXTRA_FLAGS="$3"
    local LOG="$LOGDIR/${MODE}_${MEM_STRATEGY}_${RUN_NAME}.log"
    local BS=64
    if [[ "$MODE" == *lora* ]]; then BS=32; fi
    echo "[rebuild] START mode=$MODE memory=$MEM_STRATEGY bs=$BS log=$LOG"
    # shellcheck disable=SC2086
    "$PY" -u "$TRAIN_SCRIPT" \
        --tuning-mode "$MODE" \
        --memory-strategy "$MEM_STRATEGY" \
        --epochs "$EPOCHS" \
        --train-batch-size "$BS" \
        --infer-batch-size 32 \
        --intent-clusters 64 \
        $EXTRA_FLAGS \
        "${COMMON_ARGS[@]}" \
        >"$LOG" 2>&1 \
        && echo "[rebuild] DONE mode=$MODE memory=$MEM_STRATEGY" \
        || echo "[rebuild] FAILED mode=$MODE memory=$MEM_STRATEGY (see $LOG)"
}

run_direct_clip() {
    local QUERY_MODE="$1"
    local EXTRA_FLAGS="$2"
    local LOG="$LOGDIR/direct_clip_${QUERY_MODE}_${RUN_NAME}.log"
    echo "[rebuild] START direct_clip query=$QUERY_MODE log=$LOG"
    # shellcheck disable=SC2086
    "$PY" -u "$CLIP_SCRIPT" \
        --query-mode "$QUERY_MODE" \
        --infer-batch-size 64 \
        $EXTRA_FLAGS \
        "${COMMON_ARGS[@]}" \
        >"$LOG" 2>&1 \
        && echo "[rebuild] DONE direct_clip query=$QUERY_MODE" \
        || echo "[rebuild] FAILED direct_clip query=$QUERY_MODE (see $LOG)"
}

# ── Step 1: memory baseline (head_only + retrieved_topk) ─────────────────────
# Force-rebuild builds the shared manifest for this run_name. All subsequent
# modes reuse it by pointing to the same run_name without --force-rebuild.
run_train "head_only" "retrieved_topk" "--force-rebuild"

# ── Step 2: direct CLIP zero-shot (reuses manifest built above) ──────────────
run_direct_clip "clip_context" ""

# ── Step 3: LoRA modes (all reuse the same manifest) ─────────────────────────
run_train "image_lora" "retrieved_topk" ""
run_train "text_lora"  "retrieved_topk" ""
run_train "dual_lora"  "retrieved_topk" ""

echo ""
echo "[rebuild] All experiments finished. Results in:"
echo "  /home/rl182/dl/V2L/Project-meme/MultiSticker/results/"
ls /home/rl182/dl/V2L/Project-meme/MultiSticker/results/*.json 2>/dev/null || true
