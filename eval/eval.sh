#!/bin/bash
# Evaluate a model on math benchmarks (eval/data/<name>/test.jsonl).
# Usage: bash eval/eval.sh [MODEL_PATH] [DATA_NAMES] [GPUS] [PRECISION] [OUTPUT_DIR] [GPU_MEM]
set -e

cd "$(dirname "$0")/.."

MODEL_PATH=${1:-"Qwen/Qwen3-0.6B"}
DATA_NAMES=${2:-"math,aime,amc,minerva,gpqa"}   # also: cn_math_2024,gaokao,gpqa,grade_school_math,kaoyan,math12k
GPUS=${3:-"0"}                                   # comma-separated GPU ids
PRECISION=${4:-"bf16"}                           # bf16 | fp16 | fp32; fp32 for lmc/ models (bf16 rounds their weights)
case "$PRECISION" in
    bf16) DTYPE=bfloat16; DEFAULT_OUTPUT_DIR="./outputs/eval" ;;
    fp16) DTYPE=float16;  DEFAULT_OUTPUT_DIR="./outputs/eval_fp16" ;;
    fp32) DTYPE=float32;  DEFAULT_OUTPUT_DIR="./outputs/eval_fp32" ;;
    *) echo "PRECISION must be bf16, fp16 or fp32 (got '$PRECISION')"; exit 1 ;;
esac
OUTPUT_DIR=${5:-$DEFAULT_OUTPUT_DIR}             # results are reused per dir, so each precision gets its own
GPU_MEM=${6:-"0.9"}                              # fraction of each GPU's memory vLLM may use

CUDA_VISIBLE_DEVICES="$GPUS" \
python eval/eval.py \
--model_name_or_path "$MODEL_PATH" \
--data_names "$DATA_NAMES" \
--output_dir "$OUTPUT_DIR" \
--prompt_file eval/prompt.py \
--split test \
--start_idx 0 \
--end_idx -1 \
--n_sampling 8 \
--k 1,4 \
--temperature 0.7 \
--top_p 0.95 \
--max_tokens 16384 \
--gpu_memory_utilization "$GPU_MEM" \
--num_workers 32 \
--seed 0 \
--dtype "$DTYPE"
# add --disable_thinking to evaluate Qwen3-style models without <think>
