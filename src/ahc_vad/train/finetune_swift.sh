#!/usr/bin/env bash
# LoRA fine-tune via ms-swift. Fallback / longer-run alternative to finetune_unsloth.py,
# for paid GPUs (Modal / Lightning) where a CLI-driven run is more convenient than a notebook.
#
# Prereqs: pip install ms-swift, and data/processed/train.jsonl + val.jsonl built via
# `python -m ahc_vad.data.build_sft_dataset`.
set -euo pipefail

# NOTE: T4 (Turing SM75) has NO bf16 support. TORCH_DTYPE defaults to float16 for that
# reason; override to bfloat16 only on Ampere+ (A100/L4/L40S).
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
TRAIN_DATA="${TRAIN_DATA:-data/processed/train.jsonl}"
VAL_DATA="${VAL_DATA:-data/processed/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-vl-7b-lora}"

# IMAGE_MAX_TOKEN_NUM / FPS_MAX_FRAMES are the memory and latency dials - tune down for
# real-time inference budgets, tune up if accuracy is low on multi-event clips.
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-8}"

swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN_DATA" \
  --val_dataset "$VAL_DATA" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype "$TORCH_DTYPE" \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing true \
  --max_length 4096 \
  --output_dir "$OUTPUT_DIR"
