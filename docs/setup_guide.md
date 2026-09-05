# Prerequisites & Resources

Source: `AHC Visual Intelligence Hackathon (3).pdf`

## Two things needed before the event

1. **Coding setup** — Claude Code, OpenCode, Codex, Cursor, or any coding agent you're
   comfortable with.
2. **Model / compute access** — hosted APIs, free model providers, cloud GPU runtimes, local
   hardware.

## Free GPU runtimes (chosen for this repo: Kaggle + Colab, plus Modal for paid overflow)

- **Kaggle Notebooks** — verify phone number to unlock GPU T4 x2, 30 GPU hrs/week. Save
  checkpoints to `/kaggle/working/`.
- **Google Colab** — Runtime → Change runtime type → T4 GPU. Save to Google Drive
  (`from google.colab import drive; drive.mount('/content/drive')`) since sessions can
  disconnect.
- **Lightning AI** — 5 free credits, +25 with a card on file (~$30 total). CPU instance for
  setup, switch to GPU (T4/L4/L40S/A100) only when training.
- **Modal** — serverless GPU, $30/month free credit on the Starter plan, requires a card on
  file. `pip install modal && modal setup`, decorate a function with `@app.function(gpu="T4")`.
  Use Secrets for API keys, Volumes for persistence, Budgets to cap spend.

## Hosted model APIs (no GPU needed — for large-model comparison / distillation data only,
per the brief's constraint that large models can't be part of the runtime detector)

- **AI Grants India x Flytbase** — link provided on hackathon day, ~4 RPM rate limit, gives an
  OpenAI-compatible key.
- **NVIDIA NIM** (build.nvidia.com) — 100+ hosted models incl. vision, free, phone-verify
  required, OpenAI-SDK compatible (`base_url="https://integrate.api.nvidia.com/v1"`), ~40 RPM.
- **Gemini API free tier** (aistudio.google.com) — accepts video/images directly, Flash +
  Flash-Lite only on free tier, limits are per-project not per-key.

## Claude Code + OpenRouter quick setup (if not already using another agent/provider)

```bash
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="<your-api-key>"
export ANTHROPIC_MODEL="nvidia/nemotron-3.5-lightning:free"
claude -p "Reply with exactly: OK"
```

PowerShell equivalent uses `$env:ANTHROPIC_BASE_URL` etc. (see `scripts/` for a saved snippet).

## Pre-event checklist

- [ ] Kaggle account created and phone-verified
- [ ] Colab opened at least once
- [ ] Modal account set up (card on file, budget set)
- [ ] NVIDIA NIM API key generated and saved
- [ ] Gemini API key generated and saved
- [ ] One test call made successfully with each
- [ ] Dataset downloaded locally (`scripts/download_dataset.py`)
- [ ] Fine-tuning environment smoke-tested (`src/ahc_vad/train/`)
