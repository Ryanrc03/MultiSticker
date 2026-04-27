#!/usr/bin/env bash
# Rebuild runner: runs all 5 experiment rows at pilot or medium scale.
# Usage:
#   bash run_rebuild_pilot.sh                          # medium defaults
#   RUN_SCALE=pilot bash run_rebuild_pilot.sh          # smaller smoke run
#   RUN_NAME=my_run MAX_TRAIN=5000 ... bash run_rebuild_pilot.sh
#   LORA_BS=32 INFER_BS=64 bash run_rebuild_pilot.sh   # lower batch size if OOM
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
CLIP_SCRIPT=/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/eval_direct_clip.py
ARCHIVE_SCRIPT=/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/archive_run_artifacts.py
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/scratch/rl182/mutlisticker}"
LOGDIR="${LOGDIR:-$ARTIFACT_ROOT/logs}"
RESULTDIR="${RESULTDIR:-$ARTIFACT_ROOT/results}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$ARTIFACT_ROOT/archives}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-$ARTIFACT_ROOT/embedding_cache}"
mkdir -p "$LOGDIR" "$RESULTDIR" "$ARCHIVE_ROOT" "$EMBEDDING_CACHE_DIR"

RUN_SCALE="${RUN_SCALE:-medium}"
RUN_NAME="${RUN_NAME:-rebuild_${RUN_SCALE}}"
case "$RUN_SCALE" in
    pilot)
        DEFAULT_MAX_TRAIN=5000
        DEFAULT_MAX_VAL=2000
        DEFAULT_MAX_STICKERS=3000
        DEFAULT_MIN_FREQ=2
        DEFAULT_EPOCHS=2
        DEFAULT_HEAD_BS=128
        DEFAULT_LORA_BS=64
        DEFAULT_INFER_BS=128
        ;;
    medium)
        DEFAULT_MAX_TRAIN=30000
        DEFAULT_MAX_VAL=5000
        DEFAULT_MAX_STICKERS=8000
        DEFAULT_MIN_FREQ=2
        DEFAULT_EPOCHS=5
        DEFAULT_HEAD_BS=256
        DEFAULT_LORA_BS=96
        DEFAULT_INFER_BS=256
        ;;
    *)
        echo "[rebuild] Unknown RUN_SCALE=$RUN_SCALE (expected pilot or medium)" >&2
        exit 2
        ;;
esac
MAX_TRAIN="${MAX_TRAIN:-$DEFAULT_MAX_TRAIN}"
MAX_VAL="${MAX_VAL:-$DEFAULT_MAX_VAL}"
MAX_STICKERS="${MAX_STICKERS:-$DEFAULT_MAX_STICKERS}"
MIN_FREQ="${MIN_FREQ:-$DEFAULT_MIN_FREQ}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
HEAD_BS="${HEAD_BS:-$DEFAULT_HEAD_BS}"
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

TRAIN_EXTRA_ARGS=(
    --num-workers "$NUM_WORKERS"
    --log-every "$LOG_EVERY"
    --log-samples "$LOG_SAMPLES"
    --log-dir "$LOGDIR"
    --results-dir "$RESULTDIR"
    --embedding-cache-dir "$EMBEDDING_CACHE_DIR"
    --lora-r "$LORA_R"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --lora-lr "$LORA_LR"
    --head-lr "$HEAD_LR"
)

archive_artifacts() {
    local KIND="$1"
    local PREFIX="$2"
    local LOG="$3"
    local MODE="${4:-}"
    local MEM_STRATEGY="${5:-}"
    local QUERY_MODE="${6:-}"
    "$PY" "$ARCHIVE_SCRIPT" \
        --artifact-prefix "$PREFIX" \
        --log-file "$LOG" \
        --result-dir "$RESULTDIR" \
        --archive-root "$ARCHIVE_ROOT" \
        --run-name "$RUN_NAME" \
        --run-scale "$RUN_SCALE" \
        --experiment-kind "$KIND" \
        --mode "$MODE" \
        --memory-strategy "$MEM_STRATEGY" \
        --query-mode "$QUERY_MODE" \
        --epochs "$EPOCHS" \
        --max-train-samples "$MAX_TRAIN" \
        --max-val-samples "$MAX_VAL" \
        --max-stickers "$MAX_STICKERS" \
        --min-sticker-frequency "$MIN_FREQ" \
        --supported-media ".png,.gif,.webm" \
        --train-batch-size "$TRAIN_BS_FOR_ARCHIVE" \
        --infer-batch-size "$INFER_BS" \
        --intent-clusters "$INTENT_CLUSTERS" \
        --num-workers "$NUM_WORKERS" \
        --log-every "$LOG_EVERY" \
        --log-samples "$LOG_SAMPLES" \
        --session-memories-file "$SESSION_MEMORIES" \
        --sample-intents-file "$SAMPLE_INTENTS" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout "$LORA_DROPOUT" \
        --lora-lr "$LORA_LR" \
        --head-lr "$HEAD_LR"
}

run_train() {
    local MODE="$1"
    local MEM_STRATEGY="$2"
    local EXTRA_FLAGS="$3"
    local LOG="$LOGDIR/${MODE}_${MEM_STRATEGY}_${RUN_NAME}.log"
    local PREFIX="am_${MODE}_${MEM_STRATEGY}_${RUN_NAME}"
    local BS="$HEAD_BS"
    if [[ "$MODE" == *lora* ]]; then BS="$LORA_BS"; fi
    TRAIN_BS_FOR_ARCHIVE="$BS"
    echo "[rebuild] START scale=$RUN_SCALE mode=$MODE memory=$MEM_STRATEGY train_bs=$BS infer_bs=$INFER_BS threads=$OMP_NUM_THREADS log=$LOG"
    # shellcheck disable=SC2086
    if "$PY" -u "$TRAIN_SCRIPT" \
        --tuning-mode "$MODE" \
        --memory-strategy "$MEM_STRATEGY" \
        --epochs "$EPOCHS" \
        --train-batch-size "$BS" \
        --infer-batch-size "$INFER_BS" \
        --intent-clusters "$INTENT_CLUSTERS" \
        $EXTRA_FLAGS \
        "${COMMON_ARGS[@]}" \
        "${TRAIN_EXTRA_ARGS[@]}" \
        >"$LOG" 2>&1
    then
        echo "[rebuild] DONE mode=$MODE memory=$MEM_STRATEGY"
        archive_artifacts "train_am" "$PREFIX" "$LOG" "$MODE" "$MEM_STRATEGY" ""
    else
        echo "[rebuild] FAILED mode=$MODE memory=$MEM_STRATEGY (see $LOG)"
    fi
}

run_direct_clip() {
    local QUERY_MODE="$1"
    local EXTRA_FLAGS="$2"
    local LOG="$LOGDIR/direct_clip_${QUERY_MODE}_${RUN_NAME}.log"
    local PREFIX="direct_clip_${QUERY_MODE}_${RUN_NAME}"
    TRAIN_BS_FOR_ARCHIVE=0
    echo "[rebuild] START scale=$RUN_SCALE direct_clip query=$QUERY_MODE infer_bs=$INFER_BS threads=$OMP_NUM_THREADS log=$LOG"
    # shellcheck disable=SC2086
    if "$PY" -u "$CLIP_SCRIPT" \
        --query-mode "$QUERY_MODE" \
        --infer-batch-size "$INFER_BS" \
        --results-dir "$RESULTDIR" \
        --embedding-cache-dir "$EMBEDDING_CACHE_DIR" \
        $EXTRA_FLAGS \
        "${COMMON_ARGS[@]}" \
        >"$LOG" 2>&1
    then
        echo "[rebuild] DONE direct_clip query=$QUERY_MODE"
        archive_artifacts "direct_clip" "$PREFIX" "$LOG" "" "" "$QUERY_MODE"
    else
        echo "[rebuild] FAILED direct_clip query=$QUERY_MODE (see $LOG)"
    fi
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
echo "  $RESULTDIR/"
echo "[rebuild] Archived successful experiment artifacts in:"
echo "  $ARCHIVE_ROOT/$RUN_NAME/"
ls "$RESULTDIR"/*.json 2>/dev/null || true
