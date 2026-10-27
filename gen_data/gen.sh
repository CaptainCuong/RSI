#!/bin/bash
# Generate answer-guided solutions (problem + gold answer -> solution) with vLLM.
# Usage: bash gen_data/gen.sh [MODEL_PATH] [INPUT_FILE] [PROMPT_FILE] [GPUS]
set -e

cd "$(dirname "$0")/.."

MODEL_PATH=${1:-"Qwen/Qwen3-0.6B"}
INPUT_FILE=${2:-"data/math300_train.json"}
PROMPT_FILE=${3:-"gen_data/prompt.py"}   # output goes to <model>/<data>/<prompt file name>/
GPUS=${4:-"0,1,2,3"}                     # comma-separated GPU ids; the model is split across all of them
OUTPUT_DIR="./outputs/gen_data"

CUDA_VISIBLE_DEVICES="$GPUS" \
python gen_data/generate.py \
--model_name_or_path "$MODEL_PATH" \
--input_file "$INPUT_FILE" \
--output_dir "$OUTPUT_DIR" \
--prompt_file "$PROMPT_FILE" \
--start_idx 0 \
--end_idx -1 \
--n_sampling 4 \
--temperature 0.7 \
--top_p 0.95 \
--max_tokens 16384 \
--gpu_memory_utilization 0.9 \
--chunk_size 500 \
--num_workers 32 \
--seed 0
