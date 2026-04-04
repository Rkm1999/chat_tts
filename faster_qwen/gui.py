"""
gui.py - Tkinter chat-style GUI for Qwen3-TTS (faster-qwen3-tts backend)

Uses faster-qwen3-tts with CUDA graph capture for 4-10x speedup over GGUF/ONNX.
Single process, two threads: main (UI) + synthesis.
"""
import gc
import logging
import os
import sys
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
from faster_qwen3_tts import FasterQwen3TTS

logger = logging.getLogger(__name__)

# ── Speaker / Language constants ──────────────────────────────────────────────

SPEAKERS = sorted([
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
])

LANGUAGES = [
    "Chinese", "English", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
]

DEFAULT_SPEAKER  = "sohee"
DEFAULT_LANGUAGE = "Korean"

DEFAULT_INSTRUCT = "Speak naturally and expressively, with clear emotion and a warm tone."

GPU_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
ANCHOR_TEXT    = "안녕하세요, 읽어주는 별코코입니다. 오늘도 좋은 하루 되세요."
REFERENCES_DIR = Path(__file__).parent / "references"

TTS_SR = 24000  # faster-qwen3-tts output sample rate

HYBRID_MODE = True    # Set False to disable Triton kernel patching (qwen3-tts-triton)
INT8_QUANTIZE = True  # Set False to disable torchao int8 weight-only quantization
KV_CACHE_INT8 = True  # Set False to disable int8 KV cache quantization (~48% KV VRAM reduction)

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_output_devices() -> list[tuple[int, str]]:
    try:
        return [(i, f"[{i}] {d['name']}")
                for i, d in enumerate(sd.query_devices())
                if d['max_output_channels'] > 0]
    except Exception:
        return []


def _to_float32(audio) -> np.ndarray:
    """Convert torch tensor or numpy array to float32 ndarray."""
    if isinstance(audio, torch.Tensor):
        return audio.cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32)


# ── App ───────────────────────────────────────────────────────────────────────

class TTSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Qwen3 TTS")
        self.root.minsize(480, 520)

        self.model: FasterQwen3TTS | None = None
        self.synthesizing = False
        self.ui_queue: queue.Queue = queue.Queue()

        self._stop_event   = threading.Event()
        self._voice_ready  = False
        self._previewing   = False
        self._discord_tip_shown  = False
        self._meta_tag_counter = 0
        self._speak_queue: queue.Queue = queue.Queue()  # (text, speaker, language, tag)

        self._reference_path: str | None = None
        self._ref_speaker:    str | None = None
        self._ref_language:   str | None = None
        self._ref_instruct:   str | None = None

        # Speaker output — ring buffer drained by OutputStream callback
        self._spk_buf      = np.zeros(0, dtype=np.float32)
        self._spk_buf_lock = threading.Lock()
        self._spk_out_stream: sd.OutputStream | None = None
        self._spk_device_id: int | None = None   # None → default device
        self._spk_sr: int = TTS_SR
        self._spk_out_device_map: dict[str, int] = {}

        # Mic output — ring buffer drained by OutputStream callback
        self._mic_device_id: int | None = None
        self._mic_device_map: dict[str, int] = {}
        self._mic_target_sr: int = TTS_SR
        self._mic_buf      = np.zeros(0, dtype=np.float32)
        self._mic_buf_lock = threading.Lock()
        self._mic_out_stream: sd.OutputStream | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_engine()
        self.root.after(100, self._poll_queue)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)   # chat log expands

        # ── Top bar: dropdowns ────────────────────────────────────────────────
        top = ttk.Frame(self.root)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))

        ttk.Label(top, text="Voice:").pack(side="left")
        self.voice_var = tk.StringVar(value=DEFAULT_SPEAKER)
        self.voice_cb  = ttk.Combobox(top, textvariable=self.voice_var,
                                       values=SPEAKERS, state="readonly", width=13)
        self.voice_cb.pack(side="left", padx=(4, 14))
        self.voice_cb.bind("<<ComboboxSelected>>", self._on_voice_preview)

        ttk.Label(top, text="Language:").pack(side="left")
        self.lang_var = tk.StringVar(value=DEFAULT_LANGUAGE)
        self.lang_cb  = ttk.Combobox(top, textvariable=self.lang_var,
                                      values=LANGUAGES, state="readonly", width=15)
        self.lang_cb.pack(side="left", padx=(4, 0))
        self.lang_cb.bind("<<ComboboxSelected>>", self._on_language_change)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=(10, 10))
        ttk.Label(top, text="Spk Out:").pack(side="left")
        self.spk_out_var = tk.StringVar(value="— default —")
        self.spk_out_cb  = ttk.Combobox(top, textvariable=self.spk_out_var,
                                         state="readonly", width=22)
        self.spk_out_cb.pack(side="left", padx=(4, 8))
        self.spk_out_cb.bind("<<ComboboxSelected>>", self._on_spk_out_selected)

        ttk.Label(top, text="Mic Out:").pack(side="left")
        self.mic_var = tk.StringVar(value="— disabled —")
        self.mic_cb  = ttk.Combobox(top, textvariable=self.mic_var,
                                     state="readonly", width=22)
        self.mic_cb.pack(side="left", padx=(4, 4))
        self.mic_cb.bind("<<ComboboxSelected>>", self._on_mic_selected)
        self.mic_refresh_btn = ttk.Button(top, text="↺", width=2,
                                          command=self._refresh_mic_devices)
        self.mic_refresh_btn.pack(side="left")

        self.stop_btn = ttk.Button(top, text="Stop", command=self._on_stop)
        self.stop_btn.pack(side="right")

        # ── Volume / mute row ─────────────────────────────────────────────────
        vol = ttk.Frame(self.root)
        vol.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 0))

        self.spk_mute_var = tk.BooleanVar(value=False)
        self.spk_mute_btn = ttk.Checkbutton(vol, text="Mute Spk",
                                             variable=self.spk_mute_var)
        self.spk_mute_btn.pack(side="left")

        ttk.Label(vol, text="Spk Vol:").pack(side="left", padx=(10, 2))
        self.spk_vol_var = tk.IntVar(value=100)
        ttk.Scale(vol, from_=0, to=200, variable=self.spk_vol_var,
                  orient="horizontal", length=110,
                  command=self._on_speaker_vol).pack(side="left")
        self.spk_vol_lbl = ttk.Label(vol, text="100%", width=5)
        self.spk_vol_lbl.pack(side="left")

        ttk.Separator(vol, orient="vertical").pack(side="left", fill="y", padx=(12, 12))

        ttk.Label(vol, text="Mic Vol:").pack(side="left", padx=(0, 2))
        self.mic_vol_var = tk.IntVar(value=100)
        ttk.Scale(vol, from_=0, to=200, variable=self.mic_vol_var,
                  orient="horizontal", length=110,
                  command=self._on_mic_vol).pack(side="left")
        self.mic_vol_lbl = ttk.Label(vol, text="100%", width=5)
        self.mic_vol_lbl.pack(side="left")


        # ── Tone prompt row ───────────────────────────────────────────────────
        prompt_frame = ttk.Frame(self.root)
        prompt_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 0))
        prompt_frame.columnconfigure(1, weight=1)
        ttk.Label(prompt_frame, text="Tone:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.instruct_var = tk.StringVar(value=DEFAULT_INSTRUCT)
        self.instruct_entry = ttk.Entry(prompt_frame, textvariable=self.instruct_var,
                                        font=("Segoe UI", 9))
        self.instruct_entry.grid(row=0, column=1, sticky="ew")
        self.tone_test_btn = ttk.Button(prompt_frame, text="Test", width=7,
                                        command=self._on_tone_test)
        self.tone_test_btn.grid(row=0, column=2, padx=(6, 0))
        self.tone_regen_btn = ttk.Button(prompt_frame, text="Regenerate", width=10,
                                         command=self._on_tone_regen)
        self.tone_regen_btn.grid(row=0, column=3, padx=(4, 0))
        self.tone_reset_btn = ttk.Button(prompt_frame, text="Reset", width=5,
                                         command=self._on_tone_reset)
        self.tone_reset_btn.grid(row=0, column=4, padx=(4, 0))

        # ── Chat log ──────────────────────────────────────────────────────────
        log_frame = ttk.Frame(self.root)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=6)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.chat_log = tk.Text(
            log_frame, state="disabled", wrap="word",
            font=("Segoe UI", 10), relief="flat",
            background="#f5f5f5", foreground="#1a1a1a",
            cursor="arrow",
        )
        self.chat_log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.chat_log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.chat_log.config(yscrollcommand=scroll.set)

        self.chat_log.tag_config("bubble",
            lmargin1=8, lmargin2=8, rmargin=8, spacing1=4, spacing3=4)
        self.chat_log.tag_config("meta",
            foreground="#888888", font=("Segoe UI", 8), lmargin1=8, spacing3=2)
        self.chat_log.tag_config("system",
            foreground="#555555", font=("Segoe UI", 9, "italic"),
            lmargin1=8, spacing1=2, spacing3=2)

        # ── Input row ─────────────────────────────────────────────────────────
        inp_frame = ttk.Frame(self.root)
        inp_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 4))
        inp_frame.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        self.entry = ttk.Entry(inp_frame, textvariable=self.input_var,
                               font=("Segoe UI", 10))
        self.entry.grid(row=0, column=0, sticky="ew", ipady=4)
        self.entry.bind("<Return>",       self._on_enter)
        self.entry.bind("<Shift-Return>", lambda e: None)

        self.send_btn = ttk.Button(inp_frame, text="Send", command=self._on_speak)
        self.send_btn.grid(row=0, column=1, padx=(6, 0))

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Starting engine…")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               anchor="w", relief="sunken", font=("Segoe UI", 8))
        status_bar.grid(row=5, column=0, sticky="ew")

        self._refresh_mic_devices()

    # ── Device management ─────────────────────────────────────────────────────

    def _refresh_mic_devices(self):
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        devices = _get_output_devices()

        # Spk Out combobox
        self._spk_out_device_map = {label: dev_id for dev_id, label in devices}
        spk_labels = ["— default —"] + list(self._spk_out_device_map)
        self.spk_out_cb.config(values=spk_labels)
        if self.spk_out_var.get() not in spk_labels:
            self.spk_out_var.set("— default —")

        # Mic Out combobox
        self._mic_device_map = {label: dev_id for dev_id, label in devices}
        labels = ["— disabled —"] + list(self._mic_device_map)
        self.mic_cb.config(values=labels)
        if self.mic_var.get() not in labels:
            self.mic_var.set("— disabled —")
            self._mic_device_id = None
            self._close_mic_stream()

        # Auto-select CABLE Input if nothing is selected
        if self._mic_device_id is None:
            target = next(
                (l for l in self._mic_device_map if "cable input" in l.lower()), None)
            if target:
                self.mic_var.set(target)
                self._mic_device_id = self._mic_device_map[target]
                self._open_mic_stream(self._mic_device_id)
                if not self._discord_tip_shown:
                    self._discord_tip_shown = True
                    self._append_system(
                        "Discord tip: set your mic input to CABLE Output, then go to "
                        "Settings → Voice & Video → disable 'Automatically determine "
                        "input sensitivity' and set it to 0, then disable Noise Suppression."
                    )

    def _on_spk_out_selected(self, _event=None):
        label = self.spk_out_var.get()
        self._spk_device_id = self._spk_out_device_map.get(label)  # None → default
        self._reopen_spk_stream()

    def _open_spk_stream(self, device_id: int | None = None):
        self._close_spk_stream()

        def _spk_callback(outdata, frames, _time, _status):
            vol = 0.0 if self.spk_mute_var.get() else self.spk_vol_var.get() / 100.0
            with self._spk_buf_lock:
                n = min(len(self._spk_buf), frames)
                if n > 0:
                    outdata[:n, 0] = self._spk_buf[:n] * vol
                    self._spk_buf = self._spk_buf[n:]
                if n < frames:
                    outdata[n:].fill(0)

        try:
            self._spk_out_stream = sd.OutputStream(
                samplerate=self._spk_sr, channels=1, dtype='float32',
                device=device_id, callback=_spk_callback, blocksize=512,
            )
            self._spk_out_stream.start()
        except Exception as exc:
            self.ui_queue.put(("system", f"Speaker stream open error: {exc}"))
            self._spk_out_stream = None

    def _close_spk_stream(self):
        stream = self._spk_out_stream
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            self._spk_out_stream = None

    def _reopen_spk_stream(self):
        self._close_spk_stream()
        self._open_spk_stream(self._spk_device_id)

    def _on_mic_selected(self, _event=None):
        self._mic_device_id = self._mic_device_map.get(self.mic_var.get())
        if self._mic_device_id is not None:
            self._open_mic_stream(self._mic_device_id)
        else:
            self._close_mic_stream()

    def _on_speaker_vol(self, _val=None):
        self.spk_vol_lbl.config(text=f"{self.spk_vol_var.get()}%")

    def _on_mic_vol(self, _val=None):
        self.mic_vol_lbl.config(text=f"{self.mic_vol_var.get()}%")

    def _open_mic_stream(self, device_id: int):
        self._close_mic_stream()
        try:
            self._mic_target_sr = int(sd.query_devices(device_id)['default_samplerate'])

            def _mic_callback(outdata, frames, _time, _status):
                with self._mic_buf_lock:
                    n = min(len(self._mic_buf), frames)
                    if n > 0:
                        outdata[:n, 0] = self._mic_buf[:n]
                        self._mic_buf = self._mic_buf[n:]
                    if n < frames:
                        outdata[n:].fill(0)

            self._mic_out_stream = sd.OutputStream(
                samplerate=self._mic_target_sr, channels=1, dtype='float32',
                device=device_id, callback=_mic_callback, blocksize=2048,
            )
            self._mic_out_stream.start()
        except Exception as exc:
            self.ui_queue.put(("system", f"Mic stream open error: {exc}"))
            self._mic_out_stream = None

    def _close_mic_stream(self):
        stream = self._mic_out_stream
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            self._mic_out_stream = None
        with self._mic_buf_lock:
            self._mic_buf = np.zeros(0, dtype=np.float32)

    def _on_mic_chunk(self, audio: np.ndarray, src_sr: int):
        """Push one audio chunk into the mic ring buffer with volume + resample."""
        if self._mic_out_stream is None:
            return
        vol = self.mic_vol_var.get() / 100.0
        if vol == 0.0:
            return
        if vol != 1.0:
            audio = np.clip(audio * vol, -1.0, 1.0)
        if self._mic_target_sr != src_sr:
            orig_len = len(audio)
            target_len = int(orig_len * self._mic_target_sr / src_sr)
            audio = np.interp(
                np.linspace(0, orig_len - 1, target_len),
                np.arange(orig_len),
                audio,
            ).astype(np.float32)
        with self._mic_buf_lock:
            self._mic_buf = np.concatenate([self._mic_buf, audio])

    def _push_spk(self, audio: np.ndarray):
        with self._spk_buf_lock:
            self._spk_buf = np.concatenate([self._spk_buf, audio])

    # ── Chat log helpers ──────────────────────────────────────────────────────

    def _append_chat_pending(self, text: str, speaker: str, language: str,
                             status: str = "Synthesizing…") -> str:
        self.chat_log.config(state="normal")
        self._meta_tag_counter += 1
        tag = f"_pending_meta_{self._meta_tag_counter}"
        meta = f"{speaker.title()}  ·  {language}  ·  {status}"
        self.chat_log.insert("end", meta + "\n", ("meta", tag))
        self.chat_log.insert("end", text + "\n\n", "bubble")
        self.chat_log.config(state="disabled")
        self.chat_log.see("end")
        return tag

    def _update_meta_tag(self, tag: str, new_meta: str):
        """Replace the meta line for tag. Safe to call from UI thread only."""
        self.chat_log.config(state="normal")
        ranges = self.chat_log.tag_ranges(tag)
        if ranges:
            start = ranges[0]
            line_end = self.chat_log.index(f"{start} lineend")
            self.chat_log.delete(start, line_end)
            self.chat_log.insert(start, new_meta, "meta")
        self.chat_log.config(state="disabled")

    def _finalize_chat_meta(self, tag: str, speaker: str, language: str,
                            duration: float | None):
        if not tag:
            return
        self.chat_log.config(state="normal")
        meta = f"{speaker.title()}  ·  {language}"
        meta += f"  ·  {duration:.1f}s" if duration is not None else "  ·  no audio"
        ranges = self.chat_log.tag_ranges(tag)
        if ranges:
            start = ranges[0]
            line_end = self.chat_log.index(f"{start} lineend")
            self.chat_log.delete(start, line_end)
            self.chat_log.insert(start, meta, "meta")
            self.chat_log.tag_delete(tag)
        self.chat_log.config(state="disabled")

    def _append_system(self, msg: str):
        self.chat_log.config(state="normal")
        self.chat_log.insert("end", msg + "\n", "system")
        self.chat_log.config(state="disabled")
        self.chat_log.see("end")

    # ── Engine init ───────────────────────────────────────────────────────────

    def _start_engine(self):
        self._set_input_state("disabled")
        threading.Thread(target=self._init_engine_thread, daemon=True).start()

    def _generate_reference_blocking(self, speaker, language, instruct):
        """Swap GPU models: unload Base → load CustomVoice → generate WAV → unload → reload Base."""
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        REFERENCES_DIR.mkdir(exist_ok=True)
        ref_path = REFERENCES_DIR / "gui_reference.wav"

        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

        ref_model = None
        try:
            ref_model = Qwen3TTSModel.from_pretrained(
                REF_MODEL_NAME, device_map="cuda", torch_dtype=torch.bfloat16)
            wavs, sr = ref_model.generate_custom_voice(
                text=ANCHOR_TEXT, speaker=speaker, language=language,
                instruct=instruct or None)
            if wavs:
                sf.write(str(ref_path), wavs[0].astype("float32"), sr)
                return str(ref_path)
            return None
        except Exception as exc:
            self.ui_queue.put(("system", f"Reference generation failed: {exc}"))
            return None
        finally:
            if ref_model is not None:
                del ref_model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                self.model = FasterQwen3TTS.from_pretrained(
                    GPU_MODEL_NAME, device="cuda", dtype=torch.bfloat16)
                if INT8_QUANTIZE:
                    try:
                        from qwen3_tts_triton.models.patching import find_patchable_model
                        from torchao.quantization import Int8WeightOnlyConfig, quantize_
                        internal = find_patchable_model(self.model.model)
                        quantize_(internal, Int8WeightOnlyConfig())
                        print("[GUI] Int8 weight-only quantization applied")
                    except Exception as e:
                        print(f"[GUI] Int8 skipped: {e}")
                if HYBRID_MODE:
                    try:
                        from qwen3_tts_triton.models.patching import find_patchable_model, apply_triton_kernels
                        internal = find_patchable_model(self.model.model)
                        apply_triton_kernels(internal)
                        print("[GUI] Triton kernels applied — hybrid mode active")
                    except Exception as triton_exc:
                        print(f"[GUI] Triton patching skipped: {triton_exc}")
                if INT8_QUANTIZE:
                    try:
                        # Compile after all patches so inductor sees the full patched graph
                        # (int8 dequant+matmul + Triton norms) and fuses across them.
                        self.model.predictor_graph.pred_model = torch.compile(
                            self.model.predictor_graph.pred_model, mode="max-autotune-no-cudagraphs")
                        self.model.talker_graph.model = torch.compile(
                            self.model.talker_graph.model, mode="max-autotune-no-cudagraphs")
                        print("[GUI] torch.compile applied — kernels ready for CUDA graph capture")
                    except Exception as e:
                        print(f"[GUI] torch.compile skipped: {e}")
                if KV_CACHE_INT8:
                    try:
                        from kv_cache_quant import replace_static_caches
                        n = replace_static_caches(self.model)
                        print(f"[GUI] KV cache int8 quantization applied ({n} layers)")
                    except Exception as e:
                        print(f"[GUI] KV cache int8 skipped: {e}")
            except Exception as exc:
                self.ui_queue.put(("error", f"Failed to reload Base model: {exc}"))

    def _init_engine_thread(self):
        try:
            self.ui_queue.put(("system", "Generating voice reference…"))
            instruct = self.instruct_var.get().strip() or None
            ref_path = self._generate_reference_blocking(DEFAULT_SPEAKER, DEFAULT_LANGUAGE, instruct)
            if ref_path is None:
                self.ui_queue.put(("error", "Failed to generate voice reference."))
                return
            self._reference_path = ref_path
            self._ref_speaker    = DEFAULT_SPEAKER
            self._ref_language   = DEFAULT_LANGUAGE
            self._ref_instruct   = instruct

            self.ui_queue.put(("system", f"Warming up {DEFAULT_SPEAKER.title()} voice…"))
            self._warmup(DEFAULT_SPEAKER, DEFAULT_LANGUAGE, ref_path)
        except Exception as exc:
            self.ui_queue.put(("error", f"Engine error: {exc}"))

    def _warmup(self, speaker: str, language: str, reference_path: str):
        """Run a short synthesis to capture CUDA graphs; opens the speaker stream."""
        try:
            init_text = INIT_PHRASES.get(language, "Hello.")
            first_sr = None
            for audio_chunk, sr, _timing in self.model.generate_voice_clone_streaming(
                text=init_text, language=language,
                ref_audio=reference_path, ref_text=ANCHOR_TEXT,
                chunk_size=4, xvec_only=False,
            ):
                audio_chunk = _to_float32(audio_chunk)
                if first_sr is None:
                    first_sr = sr
                    if sr != self._spk_sr:
                        self._spk_sr = sr
                    self._open_spk_stream(self._spk_device_id)
                self._push_spk(audio_chunk)
            if first_sr is None:
                # No output — open stream anyway so device is ready
                self._open_spk_stream(self._spk_device_id)
            self._voice_ready = True
            self.ui_queue.put(("ready", f"Ready — {speaker.title()} / {language}"))
        except Exception as exc:
            self.ui_queue.put(("error", f"Warmup failed: {exc}"))

    # ── Input handlers ────────────────────────────────────────────────────────

    def _on_enter(self, _event):
        self._on_speak()
        return "break"

    def _on_speak(self):
        text = self.input_var.get().strip()
        if not text:
            return

        speaker  = self.voice_var.get()
        language = self.lang_var.get()
        self.input_var.set("")

        if self.synthesizing or self._previewing:
            tag = self._append_chat_pending(text, speaker, language, status="Queued…")
            self._speak_queue.put((text, speaker, language, tag))
            n = self._speak_queue.qsize()
            self._status(f"Synthesizing…  ({n} queued)")
            return

        tag = self._append_chat_pending(text, speaker, language)
        self.synthesizing = True
        self._stop_event.clear()
        self._status("Synthesizing…")

        threading.Thread(
            target=self._speak_thread,
            args=(text, speaker, language, tag),
            daemon=True,
        ).start()

    def _speak_thread(self, text: str, speaker: str, language: str, tag: str):
        duration = None
        try:
            ref_path = self._reference_path
            if not ref_path:
                self.ui_queue.put(("system", "No voice reference — skipping."))
                self.ui_queue.put(("finalize_meta", (tag, speaker, language, None)))
                return

            print(f"[TTS] Speaking: '{text}'")
            t0 = time.monotonic()
            total_samples = 0
            first_sr = None

            for audio_chunk, sr, _timing in self.model.generate_voice_clone_streaming(
                text=text, language=language,
                ref_audio=ref_path, ref_text=ANCHOR_TEXT,
                chunk_size=4, xvec_only=False,
            ):
                if self._stop_event.is_set():
                    break
                audio_chunk = _to_float32(audio_chunk)
                if first_sr is None:
                    first_sr = sr
                    print(f"[TTS] First chunk in {time.monotonic() - t0:.3f}s  "
                          f"({len(audio_chunk)/sr*1000:.0f}ms audio)")
                self._push_spk(audio_chunk)
                if self._mic_device_id is not None and self._mic_out_stream is not None:
                    self._on_mic_chunk(audio_chunk, sr)
                total_samples += len(audio_chunk)

            if first_sr and total_samples > 0:
                duration = total_samples / first_sr
            elapsed = time.monotonic() - t0
            dur_str = f"{duration:.2f}s audio" if duration is not None else "no audio"
            print(f"[TTS] Done — {dur_str} in {elapsed:.2f}s")

            self.ui_queue.put(("finalize_meta", (tag, speaker, language, duration)))
        except Exception as exc:
            self.ui_queue.put(("error", f"Synthesis error: {exc}"))
        finally:
            self.synthesizing = False
            self._dequeue_next()

    def _dequeue_next(self):
        """Called from synthesis thread after finishing. Starts next queued item or idles."""
        try:
            text, speaker, language, tag = self._speak_queue.get_nowait()
            self.synthesizing = True
            self._stop_event.clear()
            self.ui_queue.put(("update_meta", (tag, f"{speaker.title()}  ·  {language}  ·  Synthesizing…")))
            n = self._speak_queue.qsize()
            status = f"Synthesizing…  ({n} queued)" if n else "Synthesizing…"
            self.ui_queue.put(("status", status))
            threading.Thread(
                target=self._speak_thread,
                args=(text, speaker, language, tag),
                daemon=True,
            ).start()
        except queue.Empty:
            self.ui_queue.put(("enable_input", None))

    def _on_tone_test(self):
        if self._previewing or not self._voice_ready or not self._reference_path:
            return
        self._previewing = True
        self._status("Playing reference audio…")
        threading.Thread(target=self._tone_test_thread, daemon=True).start()

    def _tone_test_thread(self):
        try:
            import soundfile as sf
            audio, sr = sf.read(self._reference_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != TTS_SR:
                orig_len = len(audio)
                target_len = int(orig_len * TTS_SR / sr)
                audio = np.interp(
                    np.linspace(0, orig_len - 1, target_len),
                    np.arange(orig_len), audio).astype(np.float32)
            self._push_spk(audio)
            if self._mic_device_id is not None and self._mic_out_stream is not None:
                self._on_mic_chunk(audio, TTS_SR)
        except Exception as exc:
            self.ui_queue.put(("system", f"Reference playback error: {exc}"))
        finally:
            self._previewing = False
            self.ui_queue.put(("status", "Ready"))

    def _on_tone_regen(self):
        if self.synthesizing or self._previewing or not self._voice_ready:
            return
        speaker  = self.voice_var.get()
        language = self.lang_var.get()
        instruct = self.instruct_var.get().strip() or None
        self._previewing = True
        self._status(f"Regenerating reference for {speaker.title()}…")
        threading.Thread(target=self._voice_change_thread,
                         args=(speaker, language, instruct), daemon=True).start()

    def _on_tone_reset(self):
        self.instruct_var.set(DEFAULT_INSTRUCT)

    def _on_voice_preview(self, _event=None):
        if self.synthesizing or self._previewing or not self._voice_ready:
            return
        speaker  = self.voice_var.get()
        language = self.lang_var.get()
        instruct = self.instruct_var.get().strip() or None
        self._previewing = True
        threading.Thread(target=self._voice_change_thread,
                         args=(speaker, language, instruct), daemon=True).start()

    def _on_language_change(self, _event=None):
        if self.synthesizing or self._previewing or not self._voice_ready:
            return
        speaker  = self.voice_var.get()
        language = self.lang_var.get()
        instruct = self.instruct_var.get().strip() or None
        self._previewing = True
        threading.Thread(target=self._voice_change_thread,
                         args=(speaker, language, instruct), daemon=True).start()

    def _voice_change_thread(self, speaker: str, language: str, instruct: str | None):
        try:
            self.ui_queue.put(("status", f"Generating reference for {speaker.title()}…"))
            ref_path = self._generate_reference_blocking(speaker, language, instruct)
            if ref_path is None:
                self.ui_queue.put(("system", f"Reference generation failed for {speaker}."))
                return
            self._reference_path = ref_path
            self._ref_speaker    = speaker
            self._ref_language   = language
            self._ref_instruct   = instruct

            self.ui_queue.put(("status", f"Testing {speaker.title()} voice…"))
            init_text = INIT_PHRASES.get(language, "Hello.")
            for audio_chunk, sr, _timing in self.model.generate_voice_clone_streaming(
                text=init_text, language=language,
                ref_audio=ref_path, ref_text=ANCHOR_TEXT,
                chunk_size=4, xvec_only=False,
            ):
                audio = _to_float32(audio_chunk)
                self._push_spk(audio)
                if self._mic_device_id is not None and self._mic_out_stream is not None:
                    self._on_mic_chunk(audio, sr)
        except Exception as exc:
            self.ui_queue.put(("system", f"Voice change error: {exc}"))
        finally:
            self._previewing = False
            self.ui_queue.put(("ready", f"Ready — {speaker.title()} / {language}"))

    def _on_stop(self):
        self._stop_event.set()
        # Cancel all queued items
        cancelled = []
        while True:
            try:
                cancelled.append(self._speak_queue.get_nowait())
            except queue.Empty:
                break
        for text, speaker, language, tag in cancelled:
            self._update_meta_tag(tag, f"{speaker.title()}  ·  {language}  ·  Cancelled")
        with self._spk_buf_lock:
            self._spk_buf = np.zeros(0, dtype=np.float32)
        with self._mic_buf_lock:
            self._mic_buf = np.zeros(0, dtype=np.float32)
        self.synthesizing = False
        self._set_input_state("normal")
        self._status("Stopped.")

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "system":
                    self._append_system(payload)
                    self._status(payload)
                elif kind == "error":
                    self._append_system(f"⚠ {payload}")
                    self._status(f"Error: {payload}")
                    self._set_input_state("normal")
                    self.synthesizing = False
                elif kind == "ready":
                    self._status(payload)
                    self._set_input_state("normal")
                    self.entry.focus()
                elif kind == "finalize_meta":
                    tag, speaker, language, duration = payload
                    self._finalize_chat_meta(tag, speaker, language, duration)
                    n = self._speak_queue.qsize()
                    status = f"Ready — {speaker.title()} / {language}"
                    if duration is not None:
                        status += f"   {duration:.1f}s audio"
                    if n:
                        status += f"   ({n} queued)"
                    self._status(status)
                elif kind == "update_meta":
                    tag, new_meta = payload
                    self._update_meta_tag(tag, new_meta)
                elif kind == "status":
                    self._status(payload)
                elif kind == "enable_input":
                    self._set_input_state("normal")
                    self.entry.focus()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.status_var.set(msg)

    def _set_input_state(self, state: str):
        # entry + send always follow state (disabled during init, normal after ready)
        self.entry.config(state=state)
        self.send_btn.config(state=state)
        # These controls are independent of synthesis — only disabled during init
        for w in (self.stop_btn, self.voice_cb, self.lang_cb,
                  self.spk_out_cb, self.mic_cb, self.mic_refresh_btn,
                  self.tone_test_btn, self.tone_regen_btn, self.tone_reset_btn,
                  self.instruct_entry):
            w.config(state=state)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_event.set()
        self._close_mic_stream()
        self._close_spk_stream()
        self.root.destroy()


def main():
    root = tk.Tk()
    TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
