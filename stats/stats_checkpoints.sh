#!/bin/bash
# Statistics of responses from every checkpoint of a training run (gen_data/gen_checkpoints.sh outputs),
# compared side by side with the base model the run started from.
# Usage: bash stats/stats_checkpoints.sh [RUN_GEN_DIR] [DATA_NAME] [PROMPT_TYPE] [CFG] [BASE_MODEL]
#   e.g. bash stats/stats_checkpoints.sh outputs/gen_data/RSI/Qwen3-0.6B_math12k_prompt_simple_correct math12k_train prompt_simple t0.7_k4_s0_e12000
set -e

cd "$(dirname "$0")/.."

RUN_GEN_DIR=${1:-"outputs/gen_data/RSI/Qwen3-0.6B_math12k_prompt_simple_correct"}
DATA_NAME=${2:-"math12k_train"}
PROMPT_TYPE=${3:-"prompt_simple"}          # prompt file name used in gen_data (prompt / prompt_simple)
CFG=${4:-"t0.7_k4_s0_e12000"}
BASE_MODEL=${5:-"Qwen/Qwen3-0.6B"}         # baseline row and tokenizer for every file; "" to skip the baseline
RUN=$(basename "${RUN_GEN_DIR%/}")

INPUT_FILES=()
FILE="outputs/gen_data/$BASE_MODEL/$DATA_NAME/$PROMPT_TYPE/$CFG.jsonl"
if [ -n "$BASE_MODEL" ] && [ -f "$FILE" ]; then
    INPUT_FILES+=("$FILE")
elif [ -n "$BASE_MODEL" ]; then
    echo "skip baseline (not found): $FILE"
fi
for CKPT_DIR in $(ls -d "${RUN_GEN_DIR%/}"/checkpoint-* | sort -V); do
    FILE="$CKPT_DIR/$DATA_NAME/$PROMPT_TYPE/$CFG.jsonl"
    if [ -f "$FILE" ]; then
        INPUT_FILES+=("$FILE")
    else
        echo "skip (not found): $FILE"
    fi
done
if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "no input files found"; exit 1
fi

# checkpoint paths are not tokenizer names, so use the base model's tokenizer for every file
python stats/stats.py \
--input_files "${INPUT_FILES[@]}" \
--output_dir outputs/stats \
--output_name "${DATA_NAME}_${RUN}_${PROMPT_TYPE}_${CFG}" \
--count_tokens \
--tokenizer "${BASE_MODEL:-Qwen/Qwen3-0.6B}"
