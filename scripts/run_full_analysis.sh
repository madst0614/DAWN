#!/bin/bash
# =============================================================================
# DAWN Full Analysis Pipeline
# =============================================================================
# One-command script for running complete checkpoint analysis on TPU VM.
#
# Usage:
#   bash scripts/run_full_analysis.sh
#   bash scripts/run_full_analysis.sh --only health,routing,dead_neuron
#   bash scripts/run_full_analysis.sh --skip-upload
#
# tmux usage:
#   tmux new -s analysis bash scripts/run_full_analysis.sh
# =============================================================================

set -euo pipefail

# ---- Configuration (override via env vars) ----
DAWN_CHECKPOINT="${DAWN_CHECKPOINT:-gs://dawn-tpu-data-c4/checkpoints/dawn_v17_1_400M_c4_20B_v4_32/run_v17.1_20260210_160828_3201}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-gs://dawn-tpu-data-c4/checkpoints/baseline_400M_c4_20B_v4_32/run_vbaseline_20260215_113114_3201}"
VAL_DATA="${VAL_DATA:-gs://dawn-tpu-data-c4/c4_val.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis_results}"
GCS_UPLOAD_DIR="${GCS_UPLOAD_DIR:-gs://dawn-tpu-data-c4/analysis_results}"

# Analysis params
N_BATCHES="${N_BATCHES:-100}"
VAL_BATCHES="${VAL_BATCHES:-200}"
MAX_SENTENCES="${MAX_SENTENCES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# ---- Parse args ----
ONLY_FLAG=""
SKIP_UPLOAD=false
for arg in "$@"; do
    case $arg in
        --only=*) ONLY_FLAG="--only ${arg#*=}" ;;
        --only) shift; ONLY_FLAG="--only $1" ;;
        --skip-upload) SKIP_UPLOAD=true ;;
    esac
done

# ---- Helpers ----
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
elapsed() {
    local secs=$1
    printf '%dh %dm %ds' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

# ---- Start ----
PIPELINE_START=$(date +%s)

log "=========================================="
log "DAWN Full Analysis Pipeline"
log "=========================================="
log "DAWN checkpoint:     $DAWN_CHECKPOINT"
log "Baseline checkpoint: $BASELINE_CHECKPOINT"
log "Val data:            $VAL_DATA"
log "Output:              $OUTPUT_DIR"
log "n_batches=$N_BATCHES val_batches=$VAL_BATCHES max_sentences=$MAX_SENTENCES"
if [ -n "$ONLY_FLAG" ]; then
    log "Filter: $ONLY_FLAG"
fi
echo

# ---- Step 0: Git pull ----
log "[0/4] Git pull..."
cd "$(dirname "$0")/.."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    BRANCH=$(git branch --show-current)
    log "  Branch: $BRANCH"
    git pull origin "$BRANCH" --rebase 2>&1 | tail -3
    log "  Pull complete."
else
    log "  Not a git repo, skipping pull."
fi
echo

# ---- Step 1: DAWN analysis ----
log "[1/4] Running DAWN checkpoint analysis..."
STEP_START=$(date +%s)

python scripts/analysis/analyze_all_jax.py \
    --checkpoint "$DAWN_CHECKPOINT" \
    --compare_checkpoint "$BASELINE_CHECKPOINT" \
    --val_data "$VAL_DATA" \
    --output "${OUTPUT_DIR}/dawn" \
    --n_batches "$N_BATCHES" \
    --val_batches "$VAL_BATCHES" \
    --max_sentences "$MAX_SENTENCES" \
    --batch_size "$BATCH_SIZE" \
    $ONLY_FLAG \
    2>&1 | tee "${OUTPUT_DIR}/dawn_analysis.log"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
log "  DAWN analysis done in $(elapsed $STEP_ELAPSED)"
echo

# ---- Step 2: Baseline analysis ----
log "[2/4] Running Baseline checkpoint analysis..."
STEP_START=$(date +%s)

python scripts/analysis/analyze_all_jax.py \
    --checkpoint "$BASELINE_CHECKPOINT" \
    --val_data "$VAL_DATA" \
    --output "${OUTPUT_DIR}/baseline" \
    --n_batches "$N_BATCHES" \
    --val_batches "$VAL_BATCHES" \
    --batch_size "$BATCH_SIZE" \
    --only model_info,performance,health \
    2>&1 | tee "${OUTPUT_DIR}/baseline_analysis.log"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
log "  Baseline analysis done in $(elapsed $STEP_ELAPSED)"
echo

# ---- Step 3: Generation comparison ----
log "[3/4] Running generation comparison..."
STEP_START=$(date +%s)

python scripts/analysis/generate_jax.py \
    --checkpoint "$DAWN_CHECKPOINT" \
    --checkpoint2 "$BASELINE_CHECKPOINT" \
    --greedy \
    --prompts \
    "The mitochondria is the" \
    "Photosynthesis converts sunlight into" \
    "The human brain contains approximately" \
    "Shakespeare was born in" \
    "The chemical formula for water is" \
    "In a democracy, citizens have the right to" \
    "Machine learning models learn by" \
    "The Great Wall of China was built to" \
    "Antibiotics are used to treat" \
    "The speed of sound is" \
    --output "${OUTPUT_DIR}/generation" \
    2>&1 | tee "${OUTPUT_DIR}/generation.log"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
log "  Generation comparison done in $(elapsed $STEP_ELAPSED)"
echo

# ---- Step 4: Upload to GCS ----
if [ "$SKIP_UPLOAD" = false ]; then
    log "[4/4] Uploading results to GCS..."
    STEP_START=$(date +%s)

    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    GCS_DEST="${GCS_UPLOAD_DIR}/${TIMESTAMP}"

    gsutil -m cp -r "${OUTPUT_DIR}/" "${GCS_DEST}/" 2>&1 | tail -5
    STEP_ELAPSED=$(($(date +%s) - STEP_START))
    log "  Upload done in $(elapsed $STEP_ELAPSED)"
    log "  Results at: ${GCS_DEST}/"
else
    log "[4/4] Skipping GCS upload (--skip-upload)"
fi
echo

# ---- Done ----
TOTAL_ELAPSED=$(($(date +%s) - PIPELINE_START))
log "=========================================="
log "Pipeline complete! Total time: $(elapsed $TOTAL_ELAPSED)"
log "Results: ${OUTPUT_DIR}/"
log "=========================================="
