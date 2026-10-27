#!/bin/bash
# Generate answer-guided solutions with every checkpoint of a training run, one checkpoint per GPU in parallel.
# Checkpoint i runs on GPU i % #GPUS; checkpoints sharing a GPU run one after another. Sampling settings come
# from gen_data/gen.sh. Outputs: outputs/gen_data/<parent>/<run>/<checkpoint>/<data>/<prompt>/*.jsonl
# Usage: bash gen_data/gen_checkpoints.sh [RUN_DIR] [INPUT_FILE] [PROMPT_FILE] [GPUS]
set -e

cd "$(dirname "$0")/.."

RUN_DIR=${1:-"/helios-storage/helios4-data/cuong/model/RSI/Qwen3-0.6B_math12k_prompt_simple_correct"}
INPUT_FILE=${2:-"data/math12k_train.json"}
PROMPT_FILE=${3:-"gen_data/prompt_simple.py"}   # or gen_data/prompt.py
GPUS=${4:-"2,3"}                            # comma-separated GPU ids, one checkpoint per GPU at a time
LOG_DIR="outputs/gen_data/logs"
mkdir -p "$LOG_DIR"

CHECKPOINTS=($(ls -d "${RUN_DIR%/}"/checkpoint-* | sort -t- -k2 -n))
IFS=',' read -ra GPU_LIST <<< "$GPUS"
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "no checkpoint-* found in $RUN_DIR"
    exit 1
fi

echo "=========================================="
echo "Generating on $INPUT_FILE (prompt: $PROMPT_FILE) with ${#CHECKPOINTS[@]} checkpoints on GPUs $GPUS"
echo "=========================================="
for g in "${!GPU_LIST[@]}"; do
    (
        for i in "${!CHECKPOINTS[@]}"; do
            [ $((i % ${#GPU_LIST[@]})) -eq "$g" ] || continue
            CKPT=${CHECKPOINTS[$i]}
            LOG="$LOG_DIR/$(basename "$RUN_DIR")_$(basename "$CKPT")_$(basename "${INPUT_FILE%.*}").log"
            echo "[GPU ${GPU_LIST[$g]}] $CKPT -> $LOG"
            bash gen_data/gen.sh "$CKPT" "$INPUT_FILE" "$PROMPT_FILE" "${GPU_LIST[$g]}" > "$LOG" 2>&1 \
                || echo "[GPU ${GPU_LIST[$g]}] FAILED: $CKPT (see $LOG)"
        done
    ) &
done
wait

echo "=========================================="
echo "ALL CHECKPOINTS DONE"
echo "outputs: outputs/gen_data/$(basename "$(dirname "${RUN_DIR%/}")")/$(basename "${RUN_DIR%/}")/checkpoint-*/$(basename "${INPUT_FILE%.*}")/$(basename "${PROMPT_FILE%.*}")/"
echo "=========================================="
