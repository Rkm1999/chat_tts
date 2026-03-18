"""
bot_guild.py - Per-guild settings and state management.
"""
import asyncio
import json
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

import discord


@dataclass
class GuildSettings:
    speaker: str = "sohee"
    language: str = "Korean"
    instruct: str = "Speak naturally and expressively."
    volume: float = 1.0          # 0.0–2.0 (1.0 = 100%)
    max_chars: int = 400
    user_speakers: dict = field(default_factory=dict)   # str(user_id) → speaker name
    user_instructs: dict = field(default_factory=dict)  # str(user_id) → instruct string
    user_references: dict = field(default_factory=dict) # str(user_id) → wav path str
    server_reference: Optional[str] = None              # server-default reference wav path


@dataclass
class GuildState:
    guild_id: int
    settings: GuildSettings
    voice_client: Optional[discord.VoiceClient] = field(default=None, compare=False)
    tts_queue: Optional[asyncio.Queue] = field(default=None, compare=False)
    worker_task: Optional[asyncio.Task] = field(default=None, compare=False)
    is_speaking: bool = False
    skip_event: threading.Event = field(default_factory=threading.Event, compare=False)


_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_guild_data.json")


def _load_persisted() -> dict:
    if os.path.exists(_DATA_FILE):
        try:
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[GuildManager] Failed to load guild data: {e}")
    return {}


def _save_persisted(data: dict):
    try:
        with open(_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[GuildManager] Failed to save guild data: {e}")


class GuildStateManager:
    def __init__(self, default_settings_factory):
        """
        Args:
            default_settings_factory: callable() → GuildSettings
        """
        self._states: dict[int, GuildState] = {}
        self._default_factory = default_settings_factory
        self._persisted: dict = _load_persisted()

    def get_or_create(self, guild_id: int) -> GuildState:
        if guild_id not in self._states:
            settings = self._restore_or_default(guild_id)
            self._states[guild_id] = GuildState(guild_id=guild_id, settings=settings)
        return self._states[guild_id]

    def _restore_or_default(self, guild_id: int) -> GuildSettings:
        saved = self._persisted.get(str(guild_id))
        if saved:
            valid_fields = GuildSettings.__dataclass_fields__.keys()
            return GuildSettings(**{k: v for k, v in saved.items() if k in valid_fields})
        return self._default_factory()

    def save(self, guild_id: int):
        state = self._states.get(guild_id)
        if state:
            self._persisted[str(guild_id)] = asdict(state.settings)
            _save_persisted(self._persisted)
