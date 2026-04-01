# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Discord TTS (Text-to-Speech) bot that reads messages in voice channels using the Faster Qwen3-TTS model. Supports 9 speakers, 10 languages, per-user voice customization, and real-time audio streaming to Discord. There is also a standalone Tkinter GUI (`gui.py`) for testing TTS without Discord. The GUI loads the smaller `0.6B-Base` model (vs `1.7B-Base` in the bot) and uses `sounddevice` for local audio playback. The GUI also supports VB-CABLE virtual audio device (can auto-download the driver) to route TTS output into a virtual microphone.

## Runtime Environment

Runs on **WSL2 (Ubuntu 24.04)** with Python 3.12 and an NVIDIA GPU (CUDA 12.4). The git repo lives on the Windows filesystem (`/mnt/c/...`) but the venv must be created in the Linux filesystem (`~/chat_tts_venv`) for I/O performance.

## Running the Project

```bash
# Initial setup (installs PyTorch + all deps + qwen3-tts-triton)
bash setup_wsl.sh

# Discord bot
cd faster_qwen
python bot.py

# Standalone GUI
cd faster_qwen
python gui.py
```

Requires `faster_qwen/.env` with `DISCORD_BOT_TOKEN` set. On startup, models load in two stages: CPU model (`CustomVoice`) first (needed to generate anchor WAVs), then GPU model (`Base`) for real-time streaming. The `engine_ready` asyncio event gates all message processing until both are loaded.

**First startup takes ~75 extra seconds** while `torch.compile` builds its kernel cache. Subsequent startups are fast.

`bot_config.py` parses `.env` manually with `os.environ.setdefault` — it does **not** call `python-dotenv`'s `load_dotenv`.

### Performance flags (top of `bot.py` and `gui.py`)

| Flag | Default | Effect |
|------|---------|--------|
| `INT8_QUANTIZE` | `True` | int8 weight quantization + `torch.compile` fusion. ~2× realtime on RTX 3060. |
| `HYBRID_MODE` | `True` | Triton kernel patches (RMSNorm, SwiGLU, M-RoPE). Can now run alongside `INT8_QUANTIZE` — patches run after quantization, before CUDA graph capture. Triton patches cover norms/activations; int8 covers matmuls — complementary. |

## Architecture

All active code lives in `faster_qwen/`. The `final/` directory (deleted in git index) was a previous version.

**Data flow**: Discord message → `bot.py` preprocessing → `GuildStateManager` queue → `bot_tts_worker.py` synthesis → `bot_audio.py` conversion → Discord voice channel

### Key modules

| File | Role |
|------|------|
| `bot.py` | Discord bot entry point; slash commands; message handler |
| `bot_config.py` | `.env` config loader; `BotConfig` dataclass |
| `bot_guild.py` | Per-guild runtime state + JSON persistence (`bot_guild_data.json`) |
| `bot_tts_worker.py` | Async TTS worker; synthesizes audio with noise/hallucination guards |
| `bot_audio.py` | Converts 24kHz mono float32 TTS output → 48kHz stereo int16 Discord PCM frames |
| `gui.py` | Standalone Tkinter GUI for TTS testing |

### Discord slash commands (command group: `/별코코`)
- `들어와` / `나가` — join/leave voice channel
- `말투` — set per-user speaking style instruction
- `목소리목록` — list available speakers
- `내목소리` — set personal voice reference WAV (voice cloning)
- `서버목소리` — set server-default voice reference (admin only)
- `스킵` — skip current TTS playback
- `설정` — show current settings

### Speakers & Languages
- **Speakers**: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
- **Languages**: Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish

### Message preprocessing (`preprocess_message`)
- Strips mentions, custom emotes, URLs, markdown formatting, and leading command prefixes (`./,;:` etc.)
- Expands Korean informal shorthand: `ㅎㅇ` → 하이, `ㅋ+` → 킥킥킥, `ㅎ+` → 히히
- Auto-appends `.` if text lacks terminal punctuation — prevents TTS hallucination at sentence end

### Audio pipeline
- `StreamingAudioSource` in `bot_audio.py` feeds audio chunk-by-chunk from the TTS iterator into Discord via a thread-safe queue. A legacy `FrameBufferSource` (pre-computed frame list) also exists but is unused by the bot.
- `GuildSettings` has `volume` (0.0–2.0, default 1.0) and `max_chars` (default 400) fields; `max_chars` truncation is applied both at enqueue time and again in `_synthesize_blocking`
- `_upsample_2x()` uses fast linear interpolation for 24kHz → 48kHz resampling
- Voice references (for cloning) stored as WAV files in `faster_qwen/references/{guild_id}_{user_id}.wav`
- The worker **prefetches** the next queued message for synthesis as soon as the GPU finishes the current one, eliminating silence gaps between consecutive TTS outputs
- The bot only reads text messages where `message.channel.id == voice_client.channel.id` — i.e. the text channel that shares the same Discord snowflake ID as the voice channel it joined
- Voice reference resolution priority in `_voice_args`: `user_references[user_id]` → `server_reference` → `default_reference` (global `default_server.wav`)
- `state.skip_event` is a `threading.Event` (not `asyncio.Event`) because it is checked from executor threads inside `_synthesize_blocking`
- Synthesis always uses ICL mode (`xvec_only=False`): both the reference audio and `ANCHOR_TEXT` are sent to the model for reliable voice cloning
- `SILENCE_LEAD_FRAMES` is currently `0` in `bot_audio.py` (comment says 40 ms but constant is 0)

### Noise / hallucination guard (`bot_tts_worker.py`)
`_synthesize_blocking` applies two guards to cut off runaway/noisy model output:
1. **Duration guard** — computes `max_duration` from text length and a per-language `chars/sec` table (e.g. Korean 6 cps, English 13 cps), multiplied by 3× to cover slowest expressive delivery; minimum floor of 4 s. When available, calibrates `chars/sec` from the actual anchor WAV duration instead of the table.
2. **Spectral flatness guard** — measures Wiener entropy of each audio chunk after 1.5 s of audio has accumulated. Real speech scores 0.2–0.5; noise/hallucination scores 0.7–1.0. Threshold is `min(ref_flatness + 0.25, 0.65)`, so the guard self-calibrates to each cloned voice.

Reference audio stats (`chars_per_sec`, `spectral_flatness`) are read once from each reference WAV and cached in `_ref_stats_cache` (keyed by path). The cache is invalidated via `invalidate_ref_stats_cache()` whenever a reference WAV is replaced, and `_invalidate_voice_cache()` in `bot.py` also clears the GPU model's internal `_voice_prompt_cache` for the same path.

### Voice/tone change approval flow
When a user runs `/별코코 내목소리`, `/별코코 말투`, or `/별코코 서버목소리`, the bot:
1. Generates a preview WAV using the CPU model in an executor
2. **Sends the preview as a DM** with Approve / Regenerate buttons (`AnchorApprovalView`, `ToneApprovalView`, `ServerAnchorApprovalView`)
3. Only saves the reference to `GuildSettings` after the user clicks Approve

If the user has DMs disabled, the command reports failure and no reference is saved.

## Configuration

`faster_qwen/.env` keys:
```
DISCORD_BOT_TOKEN=...
DEFAULT_SPEAKER=sohee
DEFAULT_LANGUAGE=Korean
DEFAULT_INSTRUCT=...
MAX_QUEUE_PER_GUILD=20
MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base          # GPU streaming model
CPU_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice  # CPU model for anchor WAV generation
```

Per-guild settings (speakers, user mappings, voice references) are persisted to `faster_qwen/bot_guild_data.json`.

## Dependencies

```bash
pip install -r requirements.txt

# PyTorch must be installed separately with CUDA support (versions must match CUDA toolkit):
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Key packages: `discord.py[voice]`, `onnxruntime-gpu`, `sounddevice`, `numpy`, `scipy`, `tokenizers`. The `faster_qwen3_tts` and `qwen_tts` packages are imported at runtime.

**Hybrid mode (Triton acceleration):** `qwen3-tts-triton` enables hybrid mode. Triton itself ships with PyTorch on Linux/WSL. After `FasterQwen3TTS.from_pretrained()`, the bot and GUI call `apply_triton_kernels(find_patchable_model(model.model))` to patch RMSNorm/SwiGLU/M-RoPE/Norm+Residual kernels before the first inference, so faster_qwen3_tts's CUDA graph capture includes the fused kernels. If the patch fails it logs a warning and falls back to non-Triton mode. Install with: `pip install qwen3-tts-triton --ignore-requires-python --no-deps` (the package declares Python 3.12+ but works on 3.12 in practice).

Git LFS is used for large model files (`.gguf`, `.onnx`, `.npy`, `.safetensors`, `.dll`, `.exe`).
