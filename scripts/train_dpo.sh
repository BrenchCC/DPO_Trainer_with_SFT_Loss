#!/usr/bin/env bash

set -euo pipefail

# Override these values with environment variables when needed.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
TRAIN_DATA="${TRAIN_DATA:-data_examples/dpo_data_example.json}"
EVAL_DATA="${EVAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dpo}"
LORA="${LORA:-true}"
SFT_LOSS_WEIGHT="${SFT_LOSS_WEIGHT:-0.1}"

ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --train_data_path "${TRAIN_DATA}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate 5e-6
    --logging_steps 1
    --save_strategy no
    --report_to none
    --model_max_length 512
    --lora "${LORA}"
    --dpo true
    --dpo_beta 0.1
    --sft_loss_weight "${SFT_LOSS_WEIGHT}"
)

if [[ -n "${EVAL_DATA}" ]]; then
    ARGS+=(--eval_data_path "${EVAL_DATA}" --eval_strategy epoch)
fi

python train.py "${ARGS[@]}" "$@"
