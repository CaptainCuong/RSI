#!/bin/bash
# Statistics of model responses (accuracy, truncation, hint leaks, lengths), compared across models.
# Usage: bash stats/stats.sh [DATA_NAME] [PROMPT_TYPE] [CFG]
#   e.g. bash stats/stats.sh math300_train prompt_simple t0.7_k4_s0_e300
set -e

cd "$(dirname "$0")/.."

DATA_NAME=${1:-"math300_train"}
PROMPT_TYPE=${2:-"prompt_simple"}          # prompt file name used in gen_data (prompt / prompt_simple)
CFG=${3:-"t0.7_k4_s0_e300"}
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
)

INPUT_FILES=()
for MODEL in "${MODELS[@]}"; do
    FILE="outputs/gen_data/$MODEL/$DATA_NAME/$PROMPT_TYPE/$CFG.jsonl"
    if [ -f "$FILE" ]; then
        INPUT_FILES+=("$FILE")
    else
        echo "skip (not found): $FILE"
    fi
done
if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "no input files found"; exit 1
fi

python stats/stats.py \
--input_files "${INPUT_FILES[@]}" \
--output_dir outputs/stats \
--count_tokens
