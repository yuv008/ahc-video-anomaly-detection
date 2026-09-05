# Primer — Small VLMs for Video Anomaly Detection

Source: `AHC Visual Intelligence Hackathon (2).pdf`

## Reading list (VAD)

1. Alert-CLIP — Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for VAD
2. A similar approach using fine-grained prompting
3. Cerberus — Real-Time Video Anomaly Detection via Cascaded Vision-Language Models
4. TAU-R1 — Traffic Anomaly Understanding

(Links were embedded in the source PDF as "Link" text without visible URLs when extracted —
re-check the original PDF/slide deck for the actual hyperlinks if needed.)

## Fine-tuning framework options

### Unsloth — fastest start
Free Colab notebooks for Qwen3-VL 8B, Gemma 3 4B, Qwen2.5-VL 7B, Llama 3.2 Vision 11B,
Pixtral 12B.

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers   = False,  # frozen encoder
    finetune_language_layers = True,
    r = 16, lora_alpha = 16,
    target_modules = "all-linear",
)
```

Tips:
- Use `UnslothVisionDataCollator` with `train_on_responses_only = True`.
- Build the dataset with a list comprehension, **not** `dataset.map()` — mapping breaks on
  multi-image samples.
- Docs: unsloth.ai/docs/basics/vision-fine-tuning

### ms-swift — CLI-driven, broad model coverage
Training + inference + eval + export in one tool.

```bash
swift sft --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset train.jsonl --val_dataset val.jsonl \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 --learning_rate 1e-4 \
  --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 --gradient_checkpointing true \
  --max_length 4096 --output_dir output
```

Tip: `IMAGE_MAX_TOKEN_NUM` and `FPS_MAX_FRAMES` are the memory/latency dials.
Docs: swift.readthedocs.io, dataset formats, Qwen3-VL guide.

### HF TRL + PEFT — reference stack
Set `max_length=None` in `SFTConfig`, or truncation silently cuts image tokens.
Docs: TRL VLM SFT guide, HF Cookbook.

## Our default pick

Qwen2.5-VL 7B / Qwen3-VL 4B via **Unsloth** for the first fine-tune (fastest iteration on free
T4 Colab/Kaggle), with an ms-swift LoRA config kept as a fallback/CLI alternative for longer
runs on paid GPUs (Modal/Lightning). See `configs/` and `src/ahc_vad/train/`.
