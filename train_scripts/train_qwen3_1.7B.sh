#!/bin/bash
# Full SFT of Qwen/Qwen3-1.7B on its own answer-guided generations for math12k_train
# (one random correct response per question, built by gen_data/convert.sh). Settings follow
# SLM_REASONING/train_script/train_adaptive_high.sh, without the per-epoch data selection.
# Usage: bash train_scripts/train_qwen3_1.7B.sh [GPUS]
set -e

cd "$(dirname "$0")/.."

GPUS=${1:-"2,3"}
MODEL_NAME=Qwen/Qwen3-1.7B
TRAIN_DATA=math12k_train_Qwen3_1.7B_prompt_simple_t0.7_k4_s0_e12000_correct
NUM_EPOCHS=5
OUTPUT_DIR=/helios-storage/helios4-data/cuong/model/RSI/Qwen3-1.7B_math12k_prompt_simple_correct
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$GPUS llamafactory-cli train \
    --model_name_or_path $MODEL_NAME \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --deepspeed train_scripts/ds_z3_config.json \
    --flash_attn fa2 \
    --gradient_checkpointing true \
    --dataset_dir data \
    --dataset $TRAIN_DATA \
    --cutoff_len 8192 \
    --overwrite_cache true \
    --preprocessing_num_workers 64 \
    --template qwen3 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 1 \
    --save_strategy epoch \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.25e-6 \
    --num_train_epochs $NUM_EPOCHS \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --save_only_model true
