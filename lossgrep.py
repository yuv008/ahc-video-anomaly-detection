import pathlib, re
txt = pathlib.Path("/content/train.log").read_text(errors="replace")
loss = [l for l in txt.splitlines() if "'loss'" in l]
print(f"loss log lines: {len(loss)}")
for l in loss[-4:]: print("  ", l[:150])
prog = [l for l in txt.splitlines() if "/300" in l]
print("latest step:", prog[-1][:90] if prog else "none")
ck = sorted(pathlib.Path("/content/qwen7b-lora").glob("checkpoint-*")) if pathlib.Path("/content/qwen7b-lora").exists() else []
print("checkpoints:", [c.name for c in ck])
