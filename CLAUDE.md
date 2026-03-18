# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Discord TTS (Text-to-Speech) bot that reads messages in voice channels using the Faster Qwen3-TTS model. Supports 9 speakers, 10 languages, per-user voice customization, and real-time audio streaming to Discord. There is also a standalone Tkinter GUI (`gui.py`) for testing TTS without Discord.

## Running the Project

```bash
# Discord bot
cd faster_qwen
python bot.py

# Standalone GUI
cd faster_qwen
python gui.py
```

Requires `faster_qwen/.env` with `DISCORD_BOT_TOKEN` set. Two models are loaded on startup: a GPU model for streaming synthesis and a CPU model for voice reference (anchor WAV) generation.

## Architecture

All active code lives in `faster_qwen/`. The `final/` directory (deleted in git index) was a previous version.

**Data flow**: Discord message → `bot.py` preprocessing → `GuildStateManager` queue → `bot_tts_worker.py` synthesis → `bot_audio.py` conversion → Discord voice channel

### Key modules

| File | Role |
|------|------|
| `bot.py` | Discord bot entry point; slash commands; message handler |
| `bot_config.py` | `.env` config loader; `BotConfig` dataclass |
| `bot_guild.py` | Per-guild runtime state + JSON persistence (`bot_guild_data.json`) |
| `bot_tts_worker.py` | Async TTS worker; handles voice cloning vs. custom voice synthesis |
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

### Audio pipeline
- `StreamingAudioSource` in `bot_audio.py` feeds audio chunk-by-chunk from the TTS iterator into Discord via a thread-safe queue
- `_upsample_2x()` uses fast linear interpolation for 24kHz → 48kHz resampling
- Voice references (for cloning) stored as WAV files in `faster_qwen/references/{guild_id}_{user_id}.wav`
- The worker **prefetches** the next queued message for synthesis as soon as the GPU finishes the current one, eliminating silence gaps between consecutive TTS outputs
- The bot only reads text messages posted in the **text channel that shares the same ID as the voice channel** it joined

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

Install via:
```bash
pip install -r requirements.txt
```

Key packages: `discord.py[voice]`, `onnxruntime-directml`, `sounddevice`, `numpy`, `scipy`, `tokenizers`. The `faster_qwen3_tts` package and `torch` are imported at runtime and must be installed separately.

Git LFS is used for large model files (`.gguf`, `.onnx`, `.npy`, `.safetensors`, `.dll`, `.exe`).
