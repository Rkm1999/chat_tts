"""
bot_config.py - Bot configuration loaded from .env
"""
import os
from dataclasses import dataclass


@dataclass
class BotConfig:
    token: str = ""
    default_speaker: str = "sohee"
    default_language: str = "Korean"
    default_instruct: str = "Read quickly and clearly, at a fast pace."
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    cpu_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    max_queue_per_guild: int = 20


def load_config() -> BotConfig:
    """Load config from .env file in the same directory as this file."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    return BotConfig(
        token=os.environ.get("DISCORD_BOT_TOKEN", ""),
        default_speaker=os.environ.get("DEFAULT_SPEAKER", "sohee"),
        default_language=os.environ.get("DEFAULT_LANGUAGE", "Korean"),
        default_instruct=os.environ.get("DEFAULT_INSTRUCT", "Read quickly and clearly, at a fast pace."),
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        cpu_model_name=os.environ.get("CPU_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        max_queue_per_guild=int(os.environ.get("MAX_QUEUE_PER_GUILD", "20")),
    )
