#!/bin/bash
# Generate answer-guided solutions with several models, one after another.
# The prompt and GPUs are set here; all other settings (sampling, ...) come from gen_data/gen.sh.
# Usage: bash gen_data/gen_models.sh [INPUT_FILE] [PROMPT_FILE] [GPUS]
set -e

cd "$(dirname "$0")/.."

INPUT_FILE=${1:-"data/math12k_train.json"}
PROMPT_FILE=${2:-"gen_data/prompt_simple.py"}   # or gen_data/prompt.py
GPUS=${3:-"2,3"}   # e.g. "2,3"; use 1, 2 or 4 GPUs (attention heads of these models must split evenly)
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
)


for MODEL_PATH in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Generating with $MODEL_PATH on $INPUT_FILE (prompt: $PROMPT_FILE, GPUs: $GPUS)"
    echo "=========================================="
    bash gen_data/gen.sh "$MODEL_PATH" "$INPUT_FILE" "$PROMPT_FILE" "$GPUS"
done

echo "=========================================="
echo "ALL MODELS DONE: ${MODELS[*]}"
echo "outputs: outputs/gen_data/<model>/$(basename "${INPUT_FILE%.*}")/$(basename "${PROMPT_FILE%.*}")/"
echo "=========================================="
