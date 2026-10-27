#!/bin/bash
# Evaluate the interpolated models made by lmc/interpolate.sh, one model per GPU in parallel.
# Model i runs on GPU i % #GPUS; models sharing a GPU run one after another. Sampling settings come from eval/eval.sh.
# The models run in fp32 by default (PRECISION): in bf16 or fp16 the fp32 weights from lmc/interpolate.py are rounded,
# mostly to one of the two endpoints.
# Outputs: outputs/eval_<PRECISION>/RSI/<END_MODEL parent>_lmc_<checkpoint>/alpha-<alpha>/metrics_test_t0.7_n8.json
# Usage: bash eval/eval_lmc.sh [END_MODEL] [DATA_NAMES] [GPUS] [PRECISION]
set -e

cd "$(dirname "$0")/.."

END_MODEL=${1:-"/helios-storage/helios4-data/cuong/model/RSI/Qwen3-0.6B_math12k_prompt_simple_correct/checkpoint-28565"}
DATA_NAMES=${2:-"math,aime,amc,minerva,gpqa"}
GPUS=${3:-"2,3"}                            # comma-separated GPU ids, one model per GPU at a time
PRECISION=${4:-"fp32"}                      # bf16 | fp16 | fp32, see eval/eval.sh
END_MODEL=${END_MODEL%/}
LMC_DIR="$(dirname "$END_MODEL")_lmc_$(basename "$END_MODEL")"   # same as OUTPUT_DIR in lmc/interpolate.sh
LOG_DIR="outputs/eval/logs"
mkdir -p "$LOG_DIR"

MODELS=($(ls -d "$LMC_DIR"/alpha-* | sort -V))
IFS=',' read -ra GPU_LIST <<< "$GPUS"
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "no alpha-* found in $LMC_DIR; run lmc/interpolate.sh first"
    exit 1
fi

echo "=========================================="
echo "Evaluating ${#MODELS[@]} interpolated models in $LMC_DIR on $DATA_NAMES with GPUs $GPUS"
echo "=========================================="
for g in "${!GPU_LIST[@]}"; do
    (
        for i in "${!MODELS[@]}"; do
            [ $((i % ${#GPU_LIST[@]})) -eq "$g" ] || continue
            MODEL=${MODELS[$i]}
            LOG="$LOG_DIR/$(basename "$LMC_DIR")_$(basename "$MODEL").log"
            echo "[GPU ${GPU_LIST[$g]}] $MODEL -> $LOG"
            bash eval/eval.sh "$MODEL" "$DATA_NAMES" "${GPU_LIST[$g]}" "$PRECISION" > "$LOG" 2>&1 \
                || echo "[GPU ${GPU_LIST[$g]}] FAILED: $MODEL (see $LOG)"
        done
    ) &
done
wait

echo "=========================================="
echo "ALL INTERPOLATED MODELS DONE"
echo "metrics: outputs/eval_$PRECISION/$(basename "$(dirname "$LMC_DIR")")/$(basename "$LMC_DIR")/alpha-*/"
echo "=========================================="
