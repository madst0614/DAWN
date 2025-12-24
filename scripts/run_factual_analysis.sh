#!/bin/bash
# Factual Knowledge Neuron Analysis
# Runs routing_analysis.py on multiple factual prompts and compares results

CHECKPOINT=${1:-"checkpoint.pt"}
OUTPUT_DIR=${2:-"routing_analysis/factual"}
ITERATIONS=${3:-100}
LAYER=${4:-11}
POOL=${5:-"fv"}

echo "========================================"
echo "Factual Knowledge Neuron Analysis"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Iterations: $ITERATIONS"
echo "Layer: $LAYER"
echo "Pool: $POOL"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# Define prompts and target tokens
declare -a PROMPTS=(
    "The capital of France is"
    "The capital of Germany is"
    "The capital of Japan is"
    "The largest planet is"
    "Water freezes at"
    "The speed of light is"
    "Einstein developed the theory of"
    "The chemical symbol for gold is"
)

declare -a TARGETS=(
    "paris"
    "berlin"
    "tokyo"
    "jupiter"
    "0,zero,32"
    "speed,light,300"
    "relativity"
    "au"
)

declare -a NAMES=(
    "france_capital"
    "germany_capital"
    "japan_capital"
    "largest_planet"
    "water_freezing"
    "speed_of_light"
    "einstein_theory"
    "gold_symbol"
)

# Run analysis for each prompt
for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    TARGET="${TARGETS[$i]}"
    NAME="${NAMES[$i]}"

    echo ""
    echo "========================================"
    echo "[$((i+1))/${#PROMPTS[@]}] $NAME"
    echo "Prompt: '$PROMPT'"
    echo "Target: '$TARGET'"
    echo "========================================"

    # Handle multiple possible targets (comma-separated)
    IFS=',' read -ra TARGET_ARRAY <<< "$TARGET"

    for T in "${TARGET_ARRAY[@]}"; do
        OUTPUT_FILE="$OUTPUT_DIR/${NAME}_${T}.json"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping $T (already exists)"
            continue
        fi

        echo "  Running analysis for target: $T"

        python scripts/analysis/routing_analysis.py \
            --checkpoint "$CHECKPOINT" \
            --prompt "$PROMPT" \
            --target_token "$T" \
            --iterations "$ITERATIONS" \
            --layer "$LAYER" \
            --pool "$POOL" \
            --top_k 50 \
            --temperature 1.0 \
            --output "$OUTPUT_DIR" \
            2>&1 | tee "$OUTPUT_DIR/${NAME}_${T}.log"

        # Rename output file to match our naming convention
        GENERATED_FILE="$OUTPUT_DIR/token_analysis_${T}_${POOL}_layer${LAYER}.json"
        if [ -f "$GENERATED_FILE" ]; then
            mv "$GENERATED_FILE" "$OUTPUT_FILE"
            echo "  Saved: $OUTPUT_FILE"
        fi
    done
done

echo ""
echo "========================================"
echo "All analyses complete!"
echo "========================================"
echo ""

# Run comparison script
echo "Running comparison analysis..."
python scripts/analysis/compare_factual_neurons.py \
    --input_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/comparison_results.json"

echo ""
echo "Done! Results in: $OUTPUT_DIR"
