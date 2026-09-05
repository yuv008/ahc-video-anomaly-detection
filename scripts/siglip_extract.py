"""Extract mean-pooled SigLIP image embeddings for every window in an export pack.

Separated from siglip_infer.py's zero-shot text-prompt scoring so the same embeddings can be
reused to fit a proper trained linear head (docs/architecture.md Stage 1 always specified a
*trained* head on a frozen encoder, never raw zero-shot cosine similarity - that was tried
separately and over-fires at 96% anomaly rate, unusable).

Output: one .npy of shape [N, dim] (L2-normalised, mean-pooled over each window's frames) and
a parallel meta.jsonl (id, video_id, start, end, class_name-or-null) so embeddings and labels
line up by row index.
"""

import argparse
import json
from pathlib import Path

MODEL_ID = "google/siglip-base-patch16-256"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"loading {MODEL_ID} on {dev}/{dtype} ...", flush=True)

    model = AutoModel.from_pretrained(MODEL_ID).to(dev, dtype=dtype)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    def features(x):
        return x.pooler_output if hasattr(x, "pooler_output") else x

    rows = [json.loads(l) for l in (args.export_dir / "index.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    frames_root = args.export_dir / "frames"
    print(f"{len(rows)} windows to embed", flush=True)

    embs = []
    meta = []
    import time
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        paths = sorted((frames_root / r["id"]).glob("f*.jpg"), key=lambda p: int(p.stem[1:]))
        if not paths:
            continue
        imgs = [Image.open(p).convert("RGB") for p in paths]
        with torch.inference_mode():
            im_in = proc(images=imgs, return_tensors="pt").to(dev, dtype=dtype)
            im_emb = features(model.get_image_features(**im_in))
            im_emb = im_emb / im_emb.norm(dim=-1, keepdim=True)
            win_emb = im_emb.mean(dim=0)
            win_emb = win_emb / win_emb.norm()
        embs.append(win_emb.float().cpu().numpy())
        meta.append({
            "id": r["id"], "video_id": r["video_id"], "start": r["start"], "end": r["end"],
            "class_name": r.get("class_name"),
        })
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(rows)}  {el/(i+1):.3f}s/win", flush=True)

    arr = np.stack(embs)
    np.save(str(args.out_prefix) + ".npy", arr)
    with open(str(args.out_prefix) + "_meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"wrote {arr.shape} to {args.out_prefix}.npy", flush=True)


main()
