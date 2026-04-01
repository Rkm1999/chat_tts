# 읽어주는 별코코 — Discord TTS Bot

Discord TTS bot and standalone GUI using [Faster Qwen3-TTS](https://github.com/ex3ndr/faster-qwen3-tts). Reads messages in voice channels with per-user voice cloning, 9 speakers, and 10 languages.

**Runs on WSL2 (Ubuntu 24.04) with an NVIDIA GPU.**

---

## Requirements

- Windows 10/11 with WSL2 enabled
- NVIDIA GPU with CUDA 12.4 support (RTX 20xx or newer)
- Latest NVIDIA Windows driver (no separate CUDA install needed inside WSL)
- ~8 GB free disk space for models

---

## Setup

### Step 1 — Verify GPU is visible in WSL

```bash
nvidia-smi
```

Your GPU should appear. If not, update your Windows NVIDIA driver.

### Step 2 — Install system packages

```bash
sudo apt-get update && sudo apt-get install -y \
    python3-venv python3-dev \
    portaudio19-dev libopus-dev ffmpeg \
    git git-lfs
```

### Step 3 — Clone the repo and create the venv

```bash
git clone <repo-url> ~/chat_tts
cd ~/chat_tts
git lfs pull

python3 -m venv ~/chat_tts_venv
```

> If working from the Windows filesystem (`/mnt/c/...`), create the venv in the Linux filesystem (`~/chat_tts_venv`) for performance — do not put it under `/mnt/c/`.

### Step 4 — Install dependencies

```bash
bash setup_wsl.sh
```

This installs PyTorch (CUDA 12.4), all requirements, and `qwen3-tts-triton`.

### Step 5 — Create `.env`

```bash
cat > faster_qwen/.env << 'EOF'
DISCORD_BOT_TOKEN=your_token_here
DEFAULT_SPEAKER=sohee
DEFAULT_LANGUAGE=Korean
EOF
```

### Step 6 — Run

```bash
source ~/chat_tts_venv/bin/activate
cd faster_qwen

# Discord bot
python bot.py

# Standalone GUI (for testing TTS without Discord)
python gui.py
```

**First startup takes ~75 extra seconds** while `torch.compile` builds its kernel cache. Subsequent startups are fast.

---

## Configuration

### `faster_qwen/.env`

| Key | Default | Description |
|-----|---------|-------------|
| `DISCORD_BOT_TOKEN` | — | **Required.** Bot token from Discord Developer Portal |
| `MODEL_NAME` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | GPU streaming model |
| `CPU_MODEL_NAME` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CPU model for voice reference generation |
| `DEFAULT_SPEAKER` | `sohee` | Default speaker voice |
| `DEFAULT_LANGUAGE` | `Korean` | Default language |
| `DEFAULT_INSTRUCT` | — | Default speaking style instruction |
| `MAX_QUEUE_PER_GUILD` | `20` | Max queued messages per server |

### Optimization flags

At the top of `bot.py` and `gui.py`:

| Flag | Default | Effect |
|------|---------|--------|
| `INT8_QUANTIZE` | `True` | int8 weight quantization + `torch.compile` fusion. ~2x realtime on RTX 3060. Mutually exclusive with `HYBRID_MODE`. |
| `HYBRID_MODE` | `True` | Hand-written Triton kernel patches (RMSNorm, SwiGLU, M-RoPE). ~1.7x realtime. Used when `INT8_QUANTIZE = False`. |

When `INT8_QUANTIZE = True`, `HYBRID_MODE` is automatically skipped.

---

## Available voices

**Speakers:** aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian

**Languages:** Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish

---

## Discord slash commands (`/별코코`)

| Command | Description |
|---------|-------------|
| `들어와` | Join your current voice channel |
| `나가` | Leave the voice channel |
| `말투` | Set your personal speaking style instruction |
| `목소리목록` | List available speakers |
| `내목소리` | Set your personal voice reference (voice cloning) |
| `서버목소리` | Set server-default voice reference (admin only) |
| `스킵` | Skip current TTS output |
| `설정` | Show current server settings |

The bot only reads messages sent in the **text channel with the same ID as the voice channel** it joined.

---

## Troubleshooting

**`sox: not found`** — harmless warning from a `qwen-tts` dependency. Does not affect functionality.

**`GPU device discovery failed` (onnxruntime)** — harmless warning. The CPU model uses ONNX for speaker embedding; GPU discovery fails in WSL but falls back to CPU automatically.

**`Not enough SMs to use max_autotune_gemm mode`** — harmless. Inductor falls back to a compatible kernel configuration for your GPU.

**First synthesis takes 36–75 seconds** — `torch.compile` is building the kernel cache on first run. Normal after that.

**`AutoProcessor` import error** — caused by torchao version mismatch. Ensure `torchao==0.9.0` is installed: `pip install torchao==0.9.0`.
