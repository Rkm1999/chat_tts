"""
bot.py - Discord TTS bot entry point (faster-qwen3-tts backend).

Usage:
    python bot.py

Requires a .env file in the same directory with at minimum:
    DISCORD_BOT_TOKEN=your_token_here
"""
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot_config import load_config, BotConfig
from bot_guild import GuildSettings, GuildState, GuildStateManager
from bot_tts_worker import TTSRequest, guild_tts_worker, _to_float32, ANCHOR_TEXT

# ── Speaker / Language constants ─────────────────────────────────────────────

REFERENCES_DIR = Path(__file__).parent / "references"
REFERENCES_DIR.mkdir(exist_ok=True)

SPEAKERS = sorted([
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
])

# ── Config & globals ──────────────────────────────────────────────────────────

config = load_config()

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)

model = None
cpu_model = None   # Qwen3TTSModel on CPU for anchor WAV generation
_default_server_reference: Optional[str] = None
guild_manager: GuildStateManager = None
engine_ready = asyncio.Event()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_default_settings() -> GuildSettings:
    return GuildSettings(
        speaker=config.default_speaker,
        language=config.default_language,
        instruct=config.default_instruct,
        server_reference=_default_server_reference,
    )


def _strip_emoji(text: str) -> str:
    """Remove emoji and non-text Unicode symbols."""
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)          # supplementary plane (💗 etc.)
    text = re.sub(r"[\u2600-\u27BF\u2B00-\u2BFF\uFE00-\uFE0F\u200D]", "", text)  # misc symbols, dingbats, variation selectors
    return text


def preprocess_message(text: str) -> str:
    """Strip mentions, custom emotes, URLs, and markdown formatting."""
    text = _strip_emoji(text)
    text = re.sub(r"<@!?\d+>", "", text)               # user mentions
    text = re.sub(r"<#\d+>", "", text)                  # channel mentions
    text = re.sub(r"<@&\d+>", "", text)                 # role mentions
    text = re.sub(r"<a?:[a-zA-Z0-9_]+:\d+>", "", text) # custom emotes
    text = re.sub(r"https?://\S+", "", text)             # remove URLs
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text) # bold / italic
    text = re.sub(r"__(.+?)__", r"\1", text)            # underline
    text = re.sub(r"~~(.+?)~~", r"\1", text)            # strikethrough
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)       # inline / code blocks
    text = re.sub(r"\s+", " ", text).strip()

    # Skip messages that start with bot-command prefixes or leading punctuation
    text = text.lstrip("./,;:~|#$%^&*@`").strip()
    if not text:
        return ""

    # Punctuation → spoken form (only when message contains nothing but ? and !)
    if re.fullmatch(r'[?!\s]+', text):
        text = re.sub(r'\?', '물음표', text)
        text = re.sub(r'!', '느낌표', text)

    # Korean shorthand → readable form (order matters: specific before general)
    text = re.sub(r'ㅎㅇㄹ', '하이루', text)
    text = re.sub(r'ㅎㅇ', '하이', text)
    text = re.sub(r'ㅋ+', '킥킥킥', text)
    text = re.sub(r'ㅎ+', '히히', text)

    # Ensure text ends with sentence-ending punctuation to prevent TTS hallucination
    if text and text[-1] not in '.。!?~…':
        text += '.'

    return text


async def _ensure_guild_worker(state: GuildState):
    """Start the guild's TTS worker task if it isn't already running."""
    if state.tts_queue is None:
        state.tts_queue = asyncio.Queue(maxsize=config.max_queue_per_guild)
    if state.worker_task is None or state.worker_task.done():
        state.worker_task = asyncio.create_task(
            guild_tts_worker(state, model, config, _default_server_reference)
        )


async def _stop_guild_worker(state: GuildState):
    """Send a poison pill to the worker and wait for it to exit."""
    if state.tts_queue is not None:
        await state.tts_queue.put(None)
    if state.worker_task and not state.worker_task.done():
        try:
            await asyncio.wait_for(state.worker_task, timeout=3.0)
        except asyncio.TimeoutError:
            state.worker_task.cancel()
    state.worker_task = None
    state.tts_queue = None


# ── Engine loaders (blocking, run in executor) ────────────────────────────────

def _load_gpu_engine(model_name: str):
    import torch
    from faster_qwen3_tts import FasterQwen3TTS
    return FasterQwen3TTS.from_pretrained(model_name, device="cuda", dtype=torch.bfloat16)


def _load_cpu_engine(model_name: str):
    import torch
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)


# ── Events ────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    global model, cpu_model, _default_server_reference, guild_manager
    print(f"[Bot] Logged in as {bot.user} (id={bot.user.id})")

    guild_manager = GuildStateManager(_make_default_settings)

    # Sync slash commands immediately so Discord's client sees up-to-date definitions
    # before the (slow) model load completes.
    await bot.tree.sync()
    print("[Bot] Slash commands synced.")

    loop = asyncio.get_event_loop()

    # Stage 1: CPU model (CustomVoice) for anchor WAV generation
    print(f"[Bot] Loading CPU model {config.cpu_model_name!r} ...")
    try:
        cpu_model = await loop.run_in_executor(None, lambda: _load_cpu_engine(config.cpu_model_name))
        print("[Bot] CPU model ready.")
    except Exception as e:
        print(f"[Bot] CPU model load failed: {e}")
        return

    # Stage 2: GPU model (Base) for real-time streaming TTS
    print(f"[Bot] Loading GPU model {config.model_name!r} ...")
    try:
        model = await loop.run_in_executor(None, lambda: _load_gpu_engine(config.model_name))
        print("[Bot] GPU model ready.")
    except Exception as e:
        print(f"[Bot] GPU model load failed: {e}")
        return

    # Stage 3: default server reference WAV (used as fallback for guilds with no reference)
    default_ref_path = REFERENCES_DIR / "default_server.wav"
    if not default_ref_path.exists():
        default_lang = config.default_language.title()
        print(f"[Bot] Generating default server reference WAV ({config.default_speaker}/{default_lang}) ...")
        try:
            await loop.run_in_executor(
                None, _generate_anchor_wav, cpu_model,
                config.default_speaker, default_lang, default_ref_path, config.default_instruct,
            )
            print("[Bot] Default server reference WAV ready.")
        except Exception as e:
            print(f"[Bot] Failed to generate default server reference WAV: {e}")
    _default_server_reference = str(default_ref_path)

    engine_ready.set()
    print("[Bot] All models ready.")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return
    if not engine_ready.is_set():
        return

    state = guild_manager.get_or_create(message.guild.id)

    if not state.voice_client or not state.voice_client.is_connected():
        return

    # Auto-read: only messages posted inside the voice channel the bot is in
    if message.channel.id != state.voice_client.channel.id:
        return

    # Skip messages that are purely attachments/files with no text
    if not message.content and message.attachments:
        return

    text = preprocess_message(message.content)
    if not text:
        return

    if state.tts_queue is not None and not state.tts_queue.full():
        truncated = text[: state.settings.max_chars]
        print(f"[{message.guild.name}] #{message.channel.name} | {message.author.display_name}: {truncated!r}")
        await state.tts_queue.put(
            TTSRequest(text=truncated, guild_id=message.guild.id, user_id=message.author.id)
        )
    elif state.tts_queue is not None and state.tts_queue.full():
        print(f"[{message.guild.name}] Queue full — dropped message from {message.author.display_name}")


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    if guild_manager is None:
        return

    # ── Bot was disconnected/kicked ───────────────────────────────────────────
    if member.id == bot.user.id:
        if before.channel is None or after.channel is not None:
            return
        # Wait briefly — discord.py auto-reconnects within ~1 s on voice server migrations.
        await asyncio.sleep(3)
        if member.guild.voice_client and member.guild.voice_client.is_connected():
            return  # reconnected — do nothing
        state = guild_manager.get_or_create(member.guild.id)
        print(f"[{member.guild.name}] Bot disconnected from voice — resetting state")
        await _stop_guild_worker(state)
        state.voice_client = None
        state.is_speaking = False
        return

    # ── Human member events ───────────────────────────────────────────────────
    if member.bot:
        return

    state = guild_manager.get_or_create(member.guild.id)
    if not state.voice_client or not state.voice_client.is_connected():
        return

    bot_channel_id = state.voice_client.channel.id

    # Member joined bot's channel
    if after.channel is not None and after.channel.id == bot_channel_id:
        if before.channel is None or before.channel.id != bot_channel_id:
            announcement = f"{_strip_emoji(member.display_name)}님이 입장하셨습니다."
            print(f"[{member.guild.name}] {member.display_name} joined — queuing announcement")
            if engine_ready.is_set() and state.tts_queue is not None and not state.tts_queue.full():
                await state.tts_queue.put(TTSRequest(text=announcement, guild_id=member.guild.id, user_id=0))
            return

    # Member left bot's channel
    if before.channel is not None and before.channel.id == bot_channel_id:
        if after.channel is None or after.channel.id != bot_channel_id:
            human_members = [m for m in state.voice_client.channel.members if not m.bot]
            if len(human_members) == 0:
                print(f"[{member.guild.name}] Channel empty — leaving")
                await _stop_guild_worker(state)
                await state.voice_client.disconnect()
                state.voice_client = None
                state.is_speaking = False
            else:
                announcement = f"{_strip_emoji(member.display_name)}님이 퇴장하셨습니다."
                print(f"[{member.guild.name}] {member.display_name} left — queuing announcement")
                if engine_ready.is_set() and state.tts_queue is not None and not state.tts_queue.full():
                    await state.tts_queue.put(TTSRequest(text=announcement, guild_id=member.guild.id, user_id=0))


# ── Anchor WAV helpers & approval views ──────────────────────────────────────

def _invalidate_voice_cache(path: Path):
    """Remove stale cache entries for a given reference WAV path."""
    if model is not None and hasattr(model, '_voice_prompt_cache'):
        path_str = str(path)
        stale = [k for k in model._voice_prompt_cache if k[0] == path_str]
        for k in stale:
            del model._voice_prompt_cache[k]
        if stale:
            print(f"[Cache] Invalidated {len(stale)} stale voice cache entry(ies) for {path_str}")


def _generate_anchor_wav(mdl, speaker: str, language: str, path: Path, instruct: Optional[str] = None):
    """Blocking — run in executor. Generates anchor WAV for a given speaker using the CPU model."""
    import soundfile as sf

    wavs, sr = mdl.generate_custom_voice(text=ANCHOR_TEXT, speaker=speaker, language=language, instruct=instruct)
    if wavs:
        sf.write(str(path), wavs[0].astype("float32"), sr)


class AnchorApprovalView(discord.ui.View):
    def __init__(self, guild_id: int, user_id: int, wav_path: Path, speaker: str, instruct: Optional[str] = None):
        super().__init__(timeout=300)
        self.guild_id = guild_id
        self.user_id  = user_id
        self.wav_path = wav_path
        self.speaker  = speaker
        self.instruct = instruct

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green)
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        state = guild_manager.get_or_create(self.guild_id)
        # Guard: speaker may have changed since this view was created
        if state.settings.user_speakers.get(str(self.user_id)) != self.speaker:
            self.stop()
            await interaction.response.edit_message(
                content="Speaker was changed — this preview is outdated. Run `/별코코 내목소리` again.",
                view=None,
            )
            return
        state.settings.user_references[str(self.user_id)] = str(self.wav_path)
        _invalidate_voice_cache(self.wav_path)
        guild_manager.save(self.guild_id)
        self.stop()
        await interaction.response.edit_message(
            content="Voice reference saved! Future TTS will use this voice.", view=None
        )

    @discord.ui.button(label="Regenerate", style=discord.ButtonStyle.blurple)
    async def regenerate(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="Generating a new sample...", view=None)
        self.stop()
        loop = asyncio.get_event_loop()
        language = guild_manager.get_or_create(self.guild_id).settings.language
        await loop.run_in_executor(
            None, _generate_anchor_wav, cpu_model, self.speaker, language, self.wav_path, self.instruct
        )
        _invalidate_voice_cache(self.wav_path)
        view = AnchorApprovalView(self.guild_id, self.user_id, self.wav_path, self.speaker, self.instruct)
        await interaction.followup.send(
            "Here's a new sample — does this sound good?",
            file=discord.File(str(self.wav_path), filename="voice_preview.wav"),
            view=view,
        )

    async def on_timeout(self):
        pass  # Reference stays cleared; user must re-run /내목소리


class ServerAnchorApprovalView(discord.ui.View):
    def __init__(self, guild_id: int, wav_path: Path, speaker: str, instruct: Optional[str] = None):
        super().__init__(timeout=300)
        self.guild_id = guild_id
        self.wav_path = wav_path
        self.speaker  = speaker
        self.instruct = instruct

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green)
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        state = guild_manager.get_or_create(self.guild_id)
        state.settings.server_reference = str(self.wav_path)
        _invalidate_voice_cache(self.wav_path)
        guild_manager.save(self.guild_id)
        self.stop()
        await interaction.response.edit_message(
            content="Server default voice reference saved!", view=None
        )

    @discord.ui.button(label="Regenerate", style=discord.ButtonStyle.blurple)
    async def regenerate(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="Generating a new sample...", view=None)
        self.stop()
        loop = asyncio.get_event_loop()
        state = guild_manager.get_or_create(self.guild_id)
        await loop.run_in_executor(
            None, _generate_anchor_wav, cpu_model, self.speaker, state.settings.language, self.wav_path, self.instruct
        )
        _invalidate_voice_cache(self.wav_path)
        view = ServerAnchorApprovalView(self.guild_id, self.wav_path, self.speaker, self.instruct)
        await interaction.followup.send(
            "New server voice sample:",
            file=discord.File(str(self.wav_path), filename="server_voice_preview.wav"),
            view=view,
        )

    async def on_timeout(self):
        pass


# ── Slash commands ────────────────────────────────────────────────────────────

class TTS(app_commands.Group, name="별코코", description="TTS 음성 컨트롤"):

    @app_commands.command(name="들어와", description="음성 채널에 들어와서 채팅을 읽기 시작합니다")
    async def join(self, interaction: discord.Interaction):
        if not engine_ready.is_set():
            await interaction.response.send_message("Model is still loading, please wait a moment.", ephemeral=True)
            return

        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message("You must be in a voice channel first.", ephemeral=True)
            return

        await interaction.response.defer()

        state = guild_manager.get_or_create(interaction.guild_id)
        channel = interaction.user.voice.channel

        existing_vc = interaction.guild.voice_client
        if existing_vc and existing_vc.is_connected():
            state.voice_client = existing_vc
            if existing_vc.channel.id == channel.id:
                await interaction.followup.send(f"Already in **{channel.name}**.", ephemeral=True)
                return
            print(f"[{interaction.guild.name}] Moving to voice channel: {channel.name}")
            await existing_vc.move_to(channel)
            await interaction.guild.change_voice_state(channel=channel, self_deaf=True)
        else:
            if existing_vc:
                try:
                    await existing_vc.disconnect(force=True)
                except Exception:
                    pass
            print(f"[{interaction.guild.name}] Joining voice channel: {channel.name} (requested by {interaction.user.display_name})")
            state.voice_client = await channel.connect(self_deaf=True)

        await _ensure_guild_worker(state)
        print(f"[{interaction.guild.name}] Auto-read active on #{channel.name}")
        await interaction.followup.send(
            f"Joined **{channel.name}**. I'll read every message posted in this channel's text chat."
        )

    @app_commands.command(name="나가", description="음성 채널에서 나갑니다")
    async def leave(self, interaction: discord.Interaction):
        state = guild_manager.get_or_create(interaction.guild_id)
        if not state.voice_client or not state.voice_client.is_connected():
            await interaction.response.send_message("Not currently in a voice channel.", ephemeral=True)
            return

        print(f"[{interaction.guild.name}] Leaving voice channel (requested by {interaction.user.display_name})")
        await _stop_guild_worker(state)
        await state.voice_client.disconnect()
        state.voice_client = None
        state.is_speaking = False
        await interaction.response.send_message("Disconnected.")

    @app_commands.command(name="말투", description="Set your personal speaking style/emotion instruction (omit to reset to server default)")
    @app_commands.describe(instruct="Style instruction, e.g. 'Speak with excitement and energy'")
    async def tone(self, interaction: discord.Interaction, instruct: Optional[str] = None):
        state = guild_manager.get_or_create(interaction.guild_id)
        uid = str(interaction.user.id)
        speaker = state.settings.user_speakers.get(uid)
        has_reference = uid in state.settings.user_references

        if instruct is None:
            state.settings.user_instructs.pop(uid, None)
            guild_manager.save(interaction.guild_id)
            if speaker and has_reference:
                await interaction.response.defer(ephemeral=True)
                loop = asyncio.get_event_loop()
                wav_path = Path(state.settings.user_references[uid])
                await loop.run_in_executor(
                    None, _generate_anchor_wav, cpu_model, speaker, state.settings.language, wav_path, None
                )
                _invalidate_voice_cache(wav_path)
                await interaction.followup.send("Your personal tone reset and voice reference updated.", ephemeral=True)
            else:
                await interaction.response.send_message("Your personal tone reset to server default.", ephemeral=True)
        else:
            state.settings.user_instructs[uid] = instruct
            guild_manager.save(interaction.guild_id)
            if speaker and has_reference:
                await interaction.response.defer(ephemeral=True)
                loop = asyncio.get_event_loop()
                wav_path = Path(state.settings.user_references[uid])
                await loop.run_in_executor(
                    None, _generate_anchor_wav, cpu_model, speaker, state.settings.language, wav_path, instruct
                )
                _invalidate_voice_cache(wav_path)
                await interaction.followup.send(f"Your personal tone set and voice reference updated: *{instruct}*", ephemeral=True)
            else:
                note = "" if speaker else " (Set a speaker with `/별코코 내목소리` for it to take effect on your voice)"
                await interaction.response.send_message(f"Your personal tone set to: *{instruct}*{note}", ephemeral=True)

    @app_commands.command(name="스킵", description="Stop current playback and cancel its synthesis")
    async def skip(self, interaction: discord.Interaction):
        state = guild_manager.get_or_create(interaction.guild_id)
        if state.voice_client and state.voice_client.is_playing():
            state.skip_event.set()
            state.voice_client.stop()
            await interaction.response.send_message("Skipped.")
        else:
            await interaction.response.send_message("Nothing is currently playing.", ephemeral=True)

    @app_commands.command(name="설정", description="Show current TTS settings for you and this server")
    async def config_cmd(self, interaction: discord.Interaction):
        state = guild_manager.get_or_create(interaction.guild_id)
        s = state.settings
        uid = str(interaction.user.id)

        personal_speaker = s.user_speakers.get(uid)
        personal_tone = s.user_instructs.get(uid)

        lines = [
            "**— Your Settings —**",
            f"**Speaker:** {personal_speaker or f'{s.speaker} (server default)'}",
            f"**Tone:** {personal_tone or f'{s.instruct} (server default)'}",
            "",
            "**— Server Defaults —**",
            f"**Speaker:** {s.speaker}",
            f"**Tone:** {s.instruct}",
            f"**Max chars per message:** {s.max_chars}",
        ]
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @app_commands.command(name="목소리목록", description="List all available speaker names")
    async def speakers(self, interaction: discord.Interaction):
        await interaction.response.send_message("**Available speakers:** " + ", ".join(SPEAKERS), ephemeral=True)

    @app_commands.command(name="내목소리", description="Set your personal TTS speaker (omit to reset to server default)")
    @app_commands.describe(name="Speaker name")
    @app_commands.choices(name=[app_commands.Choice(name=s, value=s) for s in SPEAKERS])
    async def myvoice(self, interaction: discord.Interaction, name: Optional[str] = None):
        state = guild_manager.get_or_create(interaction.guild_id)
        uid = str(interaction.user.id)

        if name is None:
            state.settings.user_speakers.pop(uid, None)
            state.settings.user_references.pop(uid, None)
            guild_manager.save(interaction.guild_id)
            await interaction.response.send_message("Your voice reset to server default.", ephemeral=True)
            return

        if not engine_ready.is_set():
            await interaction.response.send_message("Model still loading.", ephemeral=True)
            return

        # Set speaker, clear old reference
        state.settings.user_speakers[uid] = name
        state.settings.user_references.pop(uid, None)
        guild_manager.save(interaction.guild_id)

        await interaction.response.defer(ephemeral=True)

        wav_path = REFERENCES_DIR / f"{interaction.guild_id}_{uid}.wav"
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, _generate_anchor_wav, cpu_model, name, state.settings.language, wav_path,
                state.settings.user_instructs.get(uid),
            )
        except Exception as e:
            await interaction.followup.send(f"Failed to generate voice sample: {e}", ephemeral=True)
            return

        view = AnchorApprovalView(interaction.guild_id, int(uid), wav_path, name, state.settings.user_instructs.get(uid))
        try:
            await interaction.user.send(
                f"Here's a preview of the **{name}** voice. Approve it to use it, or regenerate for a new sample.",
                file=discord.File(str(wav_path), filename="voice_preview.wav"),
                view=view,
            )
            await interaction.followup.send("Check your DMs for your voice preview!", ephemeral=True)
        except discord.Forbidden:
            await interaction.followup.send(
                f"Voice set to **{name}** but I couldn't DM you (DMs disabled). No reference saved.",
                ephemeral=True,
            )

    @app_commands.command(name="서버목소리", description="서버 기본 TTS 목소리 레퍼런스를 설정합니다 (관리자 전용)")
    @app_commands.describe(name="Speaker name")
    @app_commands.choices(name=[app_commands.Choice(name=s, value=s) for s in SPEAKERS])
    @app_commands.checks.has_permissions(administrator=True)
    async def servervoice(self, interaction: discord.Interaction, name: Optional[str] = None):
        state = guild_manager.get_or_create(interaction.guild_id)

        if name is None:
            state.settings.server_reference = None
            guild_manager.save(interaction.guild_id)
            await interaction.response.send_message("Server default voice reference cleared.", ephemeral=True)
            return

        if not engine_ready.is_set():
            await interaction.response.send_message("Model still loading.", ephemeral=True)
            return

        state.settings.speaker = name
        state.settings.server_reference = None
        guild_manager.save(interaction.guild_id)

        await interaction.response.defer(ephemeral=True)

        wav_path = REFERENCES_DIR / f"{interaction.guild_id}_server.wav"
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, _generate_anchor_wav, cpu_model, name, state.settings.language, wav_path,
                state.settings.instruct,
            )
        except Exception as e:
            await interaction.followup.send(f"Failed to generate server voice sample: {e}", ephemeral=True)
            return

        view = ServerAnchorApprovalView(interaction.guild_id, wav_path, name, state.settings.instruct)
        try:
            await interaction.user.send(
                f"Server default voice preview — **{name}**. Approve to set as server default, or regenerate.",
                file=discord.File(str(wav_path), filename="server_voice_preview.wav"),
                view=view,
            )
            await interaction.followup.send("Check your DMs for the server voice preview!", ephemeral=True)
        except discord.Forbidden:
            await interaction.followup.send(
                f"Server speaker set to **{name}** but I couldn't DM you. No reference saved.",
                ephemeral=True,
            )

    @servervoice.error
    async def servervoice_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("관리자만 서버 기본 목소리를 설정할 수 있습니다.", ephemeral=True)


bot.tree.add_command(TTS())


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not config.token:
        print("ERROR: DISCORD_BOT_TOKEN is not set. Create a .env file (see .env.example).")
        sys.exit(1)
    bot.run(config.token)


if __name__ == "__main__":
    main()
