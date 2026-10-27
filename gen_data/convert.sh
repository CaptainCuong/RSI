#!/bin/bash
# Convert gen_data/generate.py outputs of several models into SFT datasets (one random correct response per question)
# and register them in data/dataset_info.json. Models and output layout match gen_data/gen_models.sh.
# Usage: bash gen_data/convert.sh [DATA_NAME] [PROMPT_NAME] [SELECT]
set -e

cd "$(dirname "$0")/.."

DATA_NAME=${1:-"math12k_train"}     # stem of the generate.py input file
PROMPT_NAME=${2:-"prompt_simple"}   # stem of the prompt file (prompt_simple or prompt)
SELECT=${3:-"correct"}              # correct: correct only | clean: correct + complete + no hint leak | all: every response
GEN_DIR="outputs/gen_data"
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
)

for MODEL_PATH in "${MODELS[@]}"; do
    for INPUT_FILE in "$GEN_DIR/$MODEL_PATH/$DATA_NAME/$PROMPT_NAME"/*.jsonl; do
        if [ ! -f "$INPUT_FILE" ]; then
            echo "no outputs for $MODEL_PATH in $GEN_DIR/$MODEL_PATH/$DATA_NAME/$PROMPT_NAME, skipping"
            continue
        fi
        echo "=========================================="
        echo "Converting $INPUT_FILE (select: $SELECT)"
        echo "=========================================="
        # dataset name is derived from the path: <data>_<model>_<prompt>_<cfg>_<select>
        python gen_data/convert_to_sft.py \
        --input_file "$INPUT_FILE" \
        --data_dir data \
        --select "$SELECT" \
        --max_per_question 1   # one random correct response per question; -1: keep all
        # add --strip_think to keep only the final solution
    done
done
