"""LoRA fine-tune Qwen2.5-VL-7B on pre-extracted window frames. Runs on the T4 via colab exec.

Consumes the frame pack produced by scripts/export_frames.py:
    /content/export_train/frames/<window_id>/f*.jpg
    /content/export_train/index.jsonl

No video decoding and no dataset logic here - windows are already labelled by temporal
overlap locally (docs/architecture.md 3.1) and frames are already at the deployment
resolution, so there is no train/serve skew.

Model choice (docs/architecture.md 5.4): 7B, not 3B. Measured on this exact GPU the 7B is
FASTER despite being larger - the 3B is deeper and narrower (36 layers vs 28) and depth
dominates latency at batch 1.
"""

import argparse
import json
from pathlib import Path

ANOMALY_CLASSES = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
SYSTEM_PROMPT = (
    "You are a real-time visual anomaly detector for city drone, CCTV and dashcam footage. "
    "Given a short sequence of frames from one time window, decide whether they show one of "
    "these anomalies: " + ", ".join(ANOMALY_CLASSES) + ", or normal if nothing of concern is "
    "happening. Most footage is ordinary and should be called normal. "
    'Reply with a single JSON object: {"is_anomaly": true|false, "class_name": "<label>"}.'
)
USER_PROMPT = "What is happening in this window?"


def target_json(class_name):
    return json.dumps({"is_anomaly": class_name != "normal", "class_name": class_name},
                      separators=(",", ":"))


class LazyWindowDataset:
    """Frames are decoded per __getitem__, never all at once.

    Loading eagerly is not an option: 2,998 windows x 8 frames at 588x336x3 is ~14.2 GB of
    decoded pixels, against roughly 13 GB of VM RAM. `Image.open` alone is lazy, but the
    `.convert("RGB")` needed for the model forces the decode, so the whole set would
    materialise before the first training step and OOM.

    Implements __len__/__getitem__ so SFTTrainer treats it as a map-style dataset.
    """

    def __init__(self, export_dir: Path):
        rows = [json.loads(l) for l in (export_dir / "index.jsonl").open(encoding="utf-8")]
        frames_root = export_dir / "frames"
        self.items = []
        for r in rows:
            paths = sorted((frames_root / r["id"]).glob("f*.jpg"),
                           key=lambda p: int(p.stem[1:]))
            if paths:
                self.items.append((paths, r["class_name"]))

    def __len__(self):
        return len(self.items)

    def class_counts(self):
        from collections import Counter
        return Counter(c for _, c in self.items)

    def __getitem__(self, idx):
        from PIL import Image

        paths, class_name = self.items[idx]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return {"messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
                                        + [{"type": "text", "text": USER_PROMPT}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": target_json(class_name)}]},
        ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=Path("/content/export_train"))
    ap.add_argument("--output-dir", type=Path, default=Path("/content/qwen7b-lora"))
    # Qwen3-VL-8B, not Qwen2.5-VL-7B. Every top team on the AI City 2026 Track-3
    # leaderboard used Qwen3-VL-8B / Qwen3.5 (0.679, 0.678, 0.670); we had anchored on
    # 2.5 only because that is what was benchmarked first.
    # NOTE: Qwen3-VL uses patch 16 (1024 px/token) vs 2.5-VL patch 14 (784), so the same
    # 588x336 frame costs 192 tokens here instead of 252 - cheaper per frame. It is also
    # 36 layers vs 28, and depth dominates batch-1 latency (architecture 5.4), so net
    # speed must be MEASURED, not assumed. Chat template is still ChatML, so the
    # response-masking markers below are unchanged.
    ap.add_argument("--base-model", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=50)
    # Resume after a machine switch or a reclaimed session. Checkpoints live on the
    # persistent volume, so training survives moving between T4 and H100.
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir, or 'auto' to pick the newest in output-dir")
    args = ap.parse_args()

    import torch
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    # is_bf16_supported() alone returns True on a T4 because it counts EMULATION. Measured
    # here: fp16 20.98 TFLOP/s vs bf16 2.28 - picking bf16 would be ~9x slower.
    try:
        use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    print(f"device: {torch.cuda.get_device_name(0)} | precision: {'bf16' if use_bf16 else 'fp16'}",
          flush=True)

    dataset = LazyWindowDataset(args.export_dir)
    print(f"training windows: {len(dataset)}", flush=True)
    for c, n in dataset.class_counts().most_common():
        print(f"  {c:35s} {n}", flush=True)

    model, processor = FastVisionModel.from_pretrained(
        args.base_model, load_in_4bit=True, use_gradient_checkpointing="unsloth"
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,   # frozen encoder keeps this cheap on a T4
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0, bias="none",
        random_state=3407, target_modules="all-linear",
    )

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        # train_on_responses_only masks the prompt so loss is computed only on the answer -
        # important here, because the system prompt is long and identical on every example,
        # so training on it would spend most of the gradient re-learning a constant.
        # It REQUIRES the chat-template markers that delimit the response; without them
        # unsloth_zoo asserts (instruction_part/response_part must be str). Qwen2.5-VL uses
        # ChatML, so the assistant turn opens with "<|im_start|>assistant\n".
        data_collator=UnslothVisionDataCollator(
            model, processor,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            bf16=use_bf16, fp16=not use_bf16,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(args.output_dir),
            save_steps=args.save_steps,
            save_total_limit=2,
            save_strategy="steps",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=4096,  # leaving this at the default silently truncates image tokens
        ),
    )

    resume = args.resume
    if resume == "auto":
        cks = sorted(args.output_dir.glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[1]))
        resume = str(cks[-1]) if cks else None
        print(f"resuming from: {resume or 'scratch'}", flush=True)

    stats = trainer.train(resume_from_checkpoint=resume)
    print("train_runtime:", stats.metrics.get("train_runtime"), flush=True)
    print("train_loss:", stats.metrics.get("train_loss"), flush=True)

    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"saved adapter -> {args.output_dir}", flush=True)


main()
