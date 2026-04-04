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

# ── Hallucination / noise guard ───────────────────────────────────────────────

# Natural speech rates (chars/second) by language.
# CJK: ~1 char per syllable. Latin/Cyrillic: avg word ~5 chars at ~2.5 words/sec.
_CHARS_PER_SEC: dict[str, float] = {
    "Korean":     6.0,
    "Japanese":   6.0,
    "Chinese":    7.0,
    "English":    13.0,
    "German":     10.0,
    "French":     11.0,
    "Spanish":    14.0,
    "Italian":    13.0,
    "Portuguese": 12.0,
    "Russian":    13.0,
}
_DEFAULT_CHARS_PER_SEC  = 10.0  # fallback for unlisted languages
_DURATION_FLOOR_SEC     = 4.0   # minimum: even 1-word messages must allow 4 s
_DURATION_MULTIPLIER    = 3.0   # 3× natural rate covers slowest expressive delivery

_FLATNESS_THRESHOLD     = 0.65  # spectral flatness above this → noise (speech: 0.2–0.5, noise: 0.7–1.0)
_FLATNESS_MARGIN        = 0.25  # how much above ref flatness before calling it noise
_FLATNESS_CHECK_AFTER   = 1.5   # seconds of audio to accumulate before first flatness check
_FLATNESS_ENERGY_GATE   = 0.01  # skip flatness on near-silent chunks (RMS below this)


def _estimate_max_duration(text: str, language: str, ref_chars_per_sec: float | None = None) -> float:
    """Upper-bound audio duration for the given text (seconds).

    Uses ref_chars_per_sec (derived from reference WAV) when available,
    falling back to the language table. This accounts for tone prompts
    that change speaking speed.
    """
    rate = ref_chars_per_sec if ref_chars_per_sec is not None else _CHARS_PER_SEC.get(language, _DEFAULT_CHARS_PER_SEC)
    return max(_DURATION_FLOOR_SEC, len(text) / rate * _DURATION_MULTIPLIER)


def _spectral_flatness(audio: np.ndarray) -> float:
    """Wiener entropy of an audio chunk.

    Returns value in [0, 1]:
      ~0.2–0.5  real speech (harmonic structure, concentrated energy)
      ~0.7–1.0  noise / hallucination (flat, spread spectrum)

    Only numpy is required.
    """
    spectrum = np.abs(np.fft.rfft(audio))
    spectrum = np.maximum(spectrum, 1e-10)  # guard against log(0)
    geometric_mean = np.exp(np.mean(np.log(spectrum)))
    arithmetic_mean = np.mean(spectrum)
    return float(geometric_mean / arithmetic_mean)


# Reference audio stats cache: path → (chars_per_sec, spectral_flatness)
_ref_stats_cache: dict[str, tuple[float, float]] = {}


def _get_ref_audio_stats(ref_path: str) -> tuple[float, float] | None:
    """Read reference WAV and return (chars_per_sec, spectral_flatness).

    chars_per_sec is derived from ANCHOR_TEXT length / ref audio duration,
    calibrated to the actual speaking speed of this voice + tone.
    spectral_flatness is the Wiener entropy of the reference, defining
    what 'real speech' looks like for this cloned voice.

    Returns None if the file cannot be read.
    Cached by path; invalidate with invalidate_ref_stats_cache().
    """
    if ref_path in _ref_stats_cache:
        return _ref_stats_cache[ref_path]

    try:
        import scipy.io.wavfile
        sr, data = scipy.io.wavfile.read(ref_path)
        if data.ndim > 1:
            data = data[:, 0]
        audio = data.astype(np.float32)
        if np.issubdtype(data.dtype, np.integer):
            audio = audio / np.iinfo(data.dtype).max

        duration = len(audio) / sr
        chars_per_sec = len(ANCHOR_TEXT) / duration if duration > 0 else _DEFAULT_CHARS_PER_SEC
        flatness = _spectral_flatness(audio)

        result = (chars_per_sec, flatness)
        _ref_stats_cache[ref_path] = result
        return result
    except Exception as exc:
        print(f"[Noise Guard] Could not read reference stats from {ref_path!r}: {exc}")
        return None


def invalidate_ref_stats_cache(path) -> None:
    """Remove stale stats cache entry when a reference WAV is replaced."""
    _ref_stats_cache.pop(str(path), None)


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
    print(f"[Latency] executor thread started at {t0:.3f}")
    first_sr = None
    total_samples = 0
    ref_stats      = _get_ref_audio_stats(reference)
    ref_cps        = ref_stats[0] if ref_stats else None
    ref_flat       = ref_stats[1] if ref_stats else None
    max_duration   = _estimate_max_duration(text, language, ref_cps)
    flat_threshold = min(ref_flat + _FLATNESS_MARGIN, _FLATNESS_THRESHOLD) if ref_flat is not None else _FLATNESS_THRESHOLD
    flatness_values: list[float] = []
    halted_by_noise = False
    print(f"[TTS Worker {guild_id}] Max duration: {max_duration:.1f}s "
          f"(ref_cps={f'{ref_cps:.1f}' if ref_cps is not None else 'n/a'}, "
          f"flat_thresh={flat_threshold:.3f}) "
          f"for {len(text)} chars ({language})")

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
            t_feed = time.monotonic()
            source.feed(audio)
            print(f"[Latency] pcm convert + queue put: {time.monotonic() - t_feed:.4f}s")
        else:
            source.feed(audio)
        total_samples += len(audio)
        current_duration = total_samples / sr

        # ── Duration guard ──────────────────────────────────────────────────
        if current_duration > max_duration:
            print(
                f"[TTS Worker {guild_id}] NOISE ABORT (duration) — "
                f"{current_duration:.1f}s > {max_duration:.1f}s limit "
                f"for {len(text)}-char {language} input"
            )
            halted_by_noise = True
            break

        # ── Spectral flatness guard ─────────────────────────────────────────
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > _FLATNESS_ENERGY_GATE:
            flatness_values.append(_spectral_flatness(audio))

        if current_duration >= _FLATNESS_CHECK_AFTER and flatness_values:
            avg_flatness = float(np.mean(flatness_values))
            if avg_flatness > flat_threshold:
                print(
                    f"[TTS Worker {guild_id}] NOISE ABORT (spectral flatness={avg_flatness:.3f}) — "
                    f"after {current_duration:.1f}s of {len(text)}-char {language} input"
                )
                halted_by_noise = True
                break

    elapsed = time.monotonic() - t0
    duration = total_samples / first_sr if first_sr else 0.0
    if halted_by_noise:
        print(f"[TTS Worker {guild_id}] Aborted (noise guard) — {duration:.1f}s audio in {elapsed:.2f}s")
    else:
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
            import time as _time
            t_dequeued = _time.monotonic()
            t_queued = getattr(req, '_t_queued', None)
            if t_queued is not None:
                print(f"[Latency] queue wait: {t_dequeued - t_queued:.3f}s")
            language, reference = _voice_args(state, req, default_reference)
            source = StreamingAudioSource(volume=state.settings.volume)
            state.skip_event.clear()
            print(f"[TTS Worker {state.guild_id}] Synthesizing (ref={reference!r}): {req.text[:80]!r}")
            t_executor = _time.monotonic()
            print(f"[Latency] submitting executor at +{t_executor - t_dequeued:.3f}s from dequeue")
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
