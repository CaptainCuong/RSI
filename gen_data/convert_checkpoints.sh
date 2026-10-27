#!/bin/bash
# Convert gen_data/gen_checkpoints.sh outputs (one per checkpoint of a training run) into SFT datasets
# (one random correct response per question) and register them in data/dataset_info.json.
# Inputs: outputs/gen_data/<parent>/<run>/checkpoint-*/<data>/<prompt>/*.jsonl
# Usage: bash gen_data/convert_checkpoints.sh [RUN_GEN_DIR] [DATA_NAME] [PROMPT_NAME] [SELECT]
set -e

cd "$(dirname "$0")/.."

RUN_GEN_DIR=${1:-"outputs/gen_data/RSI/Qwen3-0.6B_math12k_prompt_simple_correct"}
DATA_NAME=${2:-"math12k_train"}     # stem of the generate.py input file
PROMPT_NAME=${3:-"prompt_simple"}   # stem of the prompt file (prompt_simple or prompt)
SELECT=${4:-"correct"}              # correct: correct only | clean: correct + complete + no hint leak | all: every response
RUN=$(basename "${RUN_GEN_DIR%/}")

CKPT_DIRS=($(ls -d "${RUN_GEN_DIR%/}"/checkpoint-* | sort -V))
if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
    echo "no checkpoint-* found in $RUN_GEN_DIR"
    exit 1
fi

for CKPT_DIR in "${CKPT_DIRS[@]}"; do
    CKPT=$(basename "$CKPT_DIR")
    for INPUT_FILE in "$CKPT_DIR/$DATA_NAME/$PROMPT_NAME"/*.jsonl; do
        if [ ! -f "$INPUT_FILE" ]; then
            echo "no outputs for $CKPT in $CKPT_DIR/$DATA_NAME/$PROMPT_NAME, skipping"
            continue
        fi
        # the default name would only use the checkpoint dir as the model, so include the run name:
        # <data>_<run>_<checkpoint>_<prompt>_<cfg>_<select>
        CFG=$(basename "${INPUT_FILE%.*}")
        DATASET_NAME=$(echo "${DATA_NAME}_${RUN}_${CKPT}_${PROMPT_NAME}_${CFG}_${SELECT}" | sed -E 's/[^A-Za-z0-9_.]+/_/g')
        echo "=========================================="
        echo "Converting $INPUT_FILE -> $DATASET_NAME"
        echo "=========================================="
        python gen_data/convert_to_sft.py \
        --input_file "$INPUT_FILE" \
        --data_dir data \
        --dataset_name "$DATASET_NAME" \
        --select "$SELECT" \
        --max_per_question 1   # one random correct response per question; -1: keep all
        # add --strip_think to keep only the final solution
    done
done
