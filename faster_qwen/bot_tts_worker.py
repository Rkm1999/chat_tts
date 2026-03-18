"""
bot_tts_worker.py - Per-guild async TTS worker (faster-qwen3-tts backend).
"""
import asyncio
from dataclasses import dataclass
from typing import Optional

import numpy as np


ANCHOR_TEXT = "안녕하세요, 읽어주는 별코코입니다. 오늘도 좋은 하루 되세요."

INIT_PHRASES = {
    "Korean":     "안녕하세요, 반갑습니다.",
    "English":    "Hello, nice to meet you.",
    "Japanese":   "こんにちは、よろしくお願いします。",
    "Chinese":    "你好，很高兴认识你。",
    "German":     "Hallo, schön Sie kennenzulernen.",
    "French":     "Bonjour, enchanté de vous rencontrer.",
    "Spanish":    "Hola, encantado de conocerte.",
    "Russian":    "Привет, приятно познакомиться.",
    "Italian":    "Ciao, piacere di conoscerti.",
    "Portuguese": "Olá, prazer em conhecê-lo.",
}

_QUEUE_EMPTY = object()  # sentinel distinguishing "empty queue" from None poison pill


def _to_float32(audio) -> np.ndarray:
    """Convert torch tensor or numpy array to float32 ndarray."""
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            return audio.cpu().float().numpy()
    except ImportError:
        pass
    return np.asarray(audio, dtype=np.float32)


@dataclass
class TTSRequest:
    text: str
    guild_id: int
    user_id: int = 0  # Discord user ID for per-user voice lookup


def _voice_args(state, req: TTSRequest, default_reference: Optional[str] = None) -> tuple[str, Optional[str]]:
    language  = state.settings.language.title()  # normalize: "korean" → "Korean"
    reference = (state.settings.user_references.get(str(req.user_id))
                 or state.settings.server_reference
                 or default_reference)
    return language, reference


def _synthesize_blocking(model, source, text: str, language: str,
                          guild_id: int = 0, skip_event=None,
                          reference: Optional[str] = None):
    """
    Runs in executor. Always uses generate_voice_clone_streaming with the provided reference WAV.
    If no reference is available, logs a warning and finishes the source without audio.
    """
    import os
    import time

    if not reference or not os.path.exists(reference):
        print(f"[TTS Worker {guild_id}] No reference WAV — skipping")
        source.finish()
        return

    t0 = time.monotonic()
    first_sr = None
    total_samples = 0

    iterator = model.generate_voice_clone_streaming(
        text=text, language=language,
        ref_audio=reference, ref_text=ANCHOR_TEXT,
        chunk_size=4,
        xvec_only=False,   # ICL mode: use full audio context + ref_text for reliable voice cloning
    )

    for audio_chunk, sr, _timing in iterator:
        if skip_event and skip_event.is_set():
            print(f"[TTS Worker {guild_id}] Synthesis aborted by skip")
            break
        audio = _to_float32(audio_chunk)
        if first_sr is None:
            first_sr = sr
            elapsed = time.monotonic() - t0
            print(f"[TTS Worker {guild_id}] First chunk in {elapsed:.3f}s  ({len(audio)/sr*1000:.0f}ms audio)")
        source.feed(audio)
        total_samples += len(audio)

    elapsed = time.monotonic() - t0
    duration = total_samples / first_sr if first_sr else 0.0
    print(f"[TTS Worker {guild_id}] Done — {duration:.1f}s audio in {elapsed:.2f}s")


async def guild_tts_worker(state, model, config, default_reference: Optional[str] = None) -> None:
    """
    Per-guild async worker that consumes TTSRequests from the guild's queue
    and plays them sequentially in the connected voice channel.

    Pipeline: as soon as current synthesis finishes (GPU free), the next queued
    message is synthesized immediately while current playback is still draining.
    This eliminates the silence gap between consecutive messages.

    Exits when a None sentinel (poison pill) is received.
    """
    from bot_audio import StreamingAudioSource

    loop = asyncio.get_event_loop()

    # Prefetched next item: (TTSRequest, StreamingAudioSource, Future)
    prefetched: Optional[tuple] = None
    stop = False

    while not stop:
        # ── Acquire next request ───────────────────────────────────────────────
        if prefetched is not None:
            req, source, synth_fut = prefetched
            prefetched = None
        else:
            req = await state.tts_queue.get()
            if req is None:
                break
            if not state.voice_client or not state.voice_client.is_connected():
                continue
            language, reference = _voice_args(state, req, default_reference)
            source = StreamingAudioSource(volume=state.settings.volume)
            state.skip_event.clear()
            print(f"[TTS Worker {state.guild_id}] Synthesizing (ref={reference!r}): {req.text[:80]!r}")
            synth_fut = loop.run_in_executor(
                None, _synthesize_blocking, model, source,
                req.text[:state.settings.max_chars], language, state.guild_id,
                state.skip_event, reference,
            )

        # ── Start playback ─────────────────────────────────────────────────────
        if not state.voice_client or not state.voice_client.is_connected():
            continue

        done = asyncio.Event()

        def _after(error, _done=done):
            if error:
                print(f"[TTS Worker {state.guild_id}] Playback error: {error}")
            loop.call_soon_threadsafe(_done.set)

        state.is_speaking = True
        state.voice_client.play(source, after=_after)

        # ── Wait for synthesis to finish ───────────────────────────────────────
        try:
            await synth_fut
        except Exception as e:
            print(f"[TTS Worker {state.guild_id}] Synthesis error: {e}")
        finally:
            source.finish()

        # ── GPU is free; prefetch next message while playback drains ──────────
        try:
            next_req = state.tts_queue.get_nowait()
        except asyncio.QueueEmpty:
            next_req = _QUEUE_EMPTY

        if next_req is None:
            # Poison pill arrived — finish current playback then stop.
            stop = True
        elif next_req is not _QUEUE_EMPTY:
            if state.voice_client and state.voice_client.is_connected():
                next_language, next_reference = _voice_args(state, next_req, default_reference)
                next_source = StreamingAudioSource(volume=state.settings.volume)
                state.skip_event.clear()
                print(f"[TTS Worker {state.guild_id}] Prefetching (ref={next_reference!r}): {next_req.text[:80]!r}")
                next_fut = loop.run_in_executor(
                    None, _synthesize_blocking, model, next_source,
                    next_req.text[:state.settings.max_chars],
                    next_language, state.guild_id,
                    state.skip_event, next_reference,
                )
                prefetched = (next_req, next_source, next_fut)

        # ── Wait for Discord to drain the current source ───────────────────────
        while not done.is_set():
            await asyncio.sleep(0.5)
            vc = state.voice_client
            if vc is None or not vc.is_connected() or not vc.is_playing():
                break  # voice client stopped without firing _after; unblock
        state.is_speaking = False
