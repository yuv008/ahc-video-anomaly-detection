"""LoRA fine-tune a small VLM with Unsloth. Intended to run on a free Kaggle/Colab T4.

Usage (in a Kaggle/Colab cell or locally with a GPU):
    python -m ahc_vad.train.finetune_unsloth --dataset-root data/raw --output-dir outputs/qwen2.5-vl-7b-lora
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen2.5-vl-7b-lora"))
    # 7B over 3B: measured FASTER on a T4 despite being larger (the 3B is deeper/narrower -
    # 36 layers vs 28 - and depth dominates latency at batch 1). See architecture 5.4.
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--normal-ratio", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    # Pick precision from the device. CAREFUL: plain torch.cuda.is_bf16_supported() returns
    # True on a T4, because it counts EMULATED bf16. Measured on an actual T4 (SM75):
    #     fp16 20.98 TFLOP/s | bf16 2.28 TFLOP/s | fp32 4.08 TFLOP/s
    # so emulated bf16 is ~9x slower than fp16 and slower even than fp32. Selecting bf16
    # off the naive check would quietly make training ~9x slower.
    if not torch.cuda.is_available():
        use_bf16 = False
    else:
        try:
            use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # older torch has no such kwarg; bf16 is native from SM80 (Ampere)
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    print(f"precision: {'bf16' if use_bf16 else 'fp16'}")

    import random

    from ahc_vad.data.build_sft_dataset import balance_windows, build_unsloth_examples, build_windows

    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # frozen encoder - keep it cheap
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
    )

    # WINDOWS, not whole clips: labels come from temporal overlap with the annotated
    # interval, which recovers correctly-labelled background from inside anomaly clips
    # (67% of road_spill windows, 61% of vehicle_blocking). Feeding whole Events here
    # would silently bypass all of that and train the model to over-fire.
    windows = build_windows(args.dataset_root)
    windows = balance_windows(windows, random.Random(args.seed), normal_ratio=args.normal_ratio)
    print(f"training windows: {len(windows)}")
    # List comprehension, not dataset.map() - multi-image samples break .map().
    train_examples = build_unsloth_examples(windows, num_frames=args.num_frames)

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer, train_on_responses_only=True),
        train_dataset=train_examples,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            bf16=use_bf16,
            fp16=not use_bf16,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(args.output_dir),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=4096,  # never leave this at the SFTConfig default - it truncates image tokens
        ),
    )

    trainer.train()
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
