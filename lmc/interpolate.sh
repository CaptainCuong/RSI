#!/bin/bash
# Create models on the straight line between the base model (alpha = 0) and a fine-tuned checkpoint (alpha = 1),
# to check linear mode connectivity. Runs on CPU. Weights are saved in fp32 (see lmc/interpolate.py).
# Outputs: <END_MODEL parent>_lmc_<checkpoint>/alpha-<alpha>/
# Usage: bash lmc/interpolate.sh [END_MODEL] [START_MODEL] [ALPHAS]
set -e

cd "$(dirname "$0")/.."

END_MODEL=${1:-"/helios-storage/helios4-data/cuong/model/RSI/Qwen3-0.6B_math12k_prompt_simple_correct/checkpoint-28565"}
START_MODEL=${2:-"Qwen/Qwen3-0.6B"}
ALPHAS=${3:-"0.2,0.4,0.6,0.8"}      # weight of END_MODEL
END_MODEL=${END_MODEL%/}
OUTPUT_DIR="$(dirname "$END_MODEL")_lmc_$(basename "$END_MODEL")"

python lmc/interpolate.py \
--start_model "$START_MODEL" \
--end_model "$END_MODEL" \
--output_dir "$OUTPUT_DIR" \
--alphas "$ALPHAS"
