"""
bot_audio.py - Audio conversion utilities for Discord voice output.

Converts 24kHz mono float32 TTS audio to 48kHz stereo int16 PCM frames
suitable for py-cord's AudioSource interface.
"""
import queue as _queue
import time as _time

import numpy as np
import discord


DISCORD_FRAME_SIZE = 3840  # 20ms of 48kHz stereo int16 = 48000 * 0.02 * 2ch * 2bytes

# Cached TCP RTT to the Discord voice server (ms), updated on each voice join and per-message.
_voice_rtt_ms: float | None = None
_voice_server_endpoint: tuple[str, int] | None = None  # (host, port)

def set_voice_rtt(ms: float) -> None:
    global _voice_rtt_ms
    _voice_rtt_ms = ms

def set_voice_endpoint(host: str, port: int) -> None:
    global _voice_server_endpoint
    _voice_server_endpoint = (host, port)
SILENCE_LEAD_FRAMES = 0   # 40 ms — gives Discord time to process speaking state before audio


def _upsample_2x(audio: np.ndarray) -> np.ndarray:
    """
    Fast 2x upsample via linear interpolation (24kHz → 48kHz).
    ~5-10x faster than resample_poly; perceptually identical for speech.
    Interleaves original samples with their midpoints: [a, (a+b)/2, b, ...]
    """
    a = audio.astype(np.float32)
    out = np.empty(len(a) * 2, dtype=np.float32)
    out[0::2] = a
    out[1::2] = np.concatenate([(a[:-1] + a[1:]) * 0.5, a[-1:]])
    return out


def pcm_to_discord_frames(audio: np.ndarray, volume: float = 1.0) -> list[bytes]:
    """
    Convert 24kHz mono float32 audio to a list of 3840-byte Discord PCM frames.

    Steps:
      1. Linear 2x upsample 24k→48k
      2. Clip to [-1, 1] and apply volume
      3. Mono → stereo (stack channel with itself)
      4. float32 → int16
      5. Split into 3840-byte chunks, zero-pad the last chunk if needed
    """
    # 1. Resample 24kHz → 48kHz
    resampled = _upsample_2x(audio)

    # 2. Volume + clip
    if volume != 1.0:
        resampled = resampled * volume
    resampled = np.clip(resampled, -1.0, 1.0)

    # 3. Mono → stereo  [N] → [N, 2]
    stereo = np.stack([resampled, resampled], axis=1)

    # 4. float32 → int16
    pcm_int16 = (stereo * 32767).astype(np.int16)

    # 5. Slice into 3840-byte frames
    raw = pcm_int16.tobytes()
    frames: list[bytes] = []
    for i in range(0, len(raw), DISCORD_FRAME_SIZE):
        chunk = raw[i : i + DISCORD_FRAME_SIZE]
        if len(chunk) < DISCORD_FRAME_SIZE:
            chunk = chunk + b"\x00" * (DISCORD_FRAME_SIZE - len(chunk))
        frames.append(chunk)

    return frames


class FrameBufferSource(discord.AudioSource):
    """
    An AudioSource backed by a pre-computed list of 3840-byte PCM frames.

    Discord's VoiceClient calls read() every 20ms and expects either a
    3840-byte chunk or b'' to signal end of stream.
    """

    def __init__(self, frames: list[bytes]):
        silence = b"\x00" * DISCORD_FRAME_SIZE
        self._frames = [silence] * SILENCE_LEAD_FRAMES + frames
        self._pos = 0

    def read(self) -> bytes:
        if self._pos >= len(self._frames):
            return b""
        frame = self._frames[self._pos]
        self._pos += 1
        return frame

    def is_opus(self) -> bool:
        return False


class StreamingAudioSource(discord.AudioSource):
    """
    AudioSource fed chunk-by-chunk via feed() from the TTS _chunk_callback.
    Inserts silence when queue is temporarily empty (still generating).
    Ends when finish() is called and the buffer drains.
    Thread-safe: feed() called from proxy listener thread, read() from Discord send thread.
    """

    def __init__(self, volume: float = 1.0):
        silence = b"\x00" * DISCORD_FRAME_SIZE
        self._q: _queue.Queue = _queue.Queue()
        for _ in range(SILENCE_LEAD_FRAMES):
            self._q.put(silence)
        self._volume = volume
        self._raw_tail = b""  # leftover bytes from a partial last frame
        self._t_first_feed: float | None = None
        self._first_read_logged: bool = False
        try:
            self._opus_encoder = discord.opus.Encoder()
        except Exception:
            self._opus_encoder = None

    def feed(self, audio: np.ndarray):
        """Called from chunk callback thread."""
        if self._t_first_feed is None:
            self._t_first_feed = _time.monotonic()
        frames = pcm_to_discord_frames(audio, volume=self._volume)
        for f in frames:
            self._q.put(f)

    def finish(self):
        """Signal end-of-stream. Must be called exactly once after synthesis completes."""
        self._q.put(None)  # sentinel

    def read(self) -> bytes:
        """Called by Discord every 20ms."""
        try:
            frame = self._q.get(timeout=0.015)  # 15ms — stay within Discord's 20ms cadence
        except _queue.Empty:
            return b"\x00" * DISCORD_FRAME_SIZE  # still generating; keep connection alive
        if frame is None:
            return b""  # sentinel — done
        if not self._first_read_logged and self._t_first_feed is not None:
            feed_to_read = _time.monotonic() - self._t_first_feed
            opus_ms = None
            if self._opus_encoder is not None:
                t_enc = _time.monotonic()
                self._opus_encoder.encode(frame, 960)
                opus_ms = (_time.monotonic() - t_enc) * 1000
            rtt = _voice_rtt_ms
            print(
                f"[Latency] feed→AudioPlayer read: {feed_to_read:.4f}s"
                + (f" | Opus encode: {opus_ms:.3f}ms" if opus_ms is not None else "")
                + (f" | voice server RTT: {rtt:.1f}ms (~{rtt/2:.1f}ms one-way)" if rtt is not None else "")
            )
            self._first_read_logged = True
        return frame

    def is_opus(self) -> bool:
        return False
