"""
gui.py - Tkinter chat-style GUI for Qwen3-TTS
"""
import ctypes
import os
import subprocess
import sys
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import urllib.request
import webbrowser
import zipfile
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult
from qwen3_tts_gguf.inference.schema.constants import SPEAKER_MAP, LANGUAGE_MAP

SPEAKERS  = sorted(SPEAKER_MAP.keys())
LANGUAGES = sorted(LANGUAGE_MAP.keys())

DEFAULT_SPEAKER  = "sohee"
DEFAULT_LANGUAGE = "korean"
INIT_TEXT        = "안녕하세요, 저는 소희입니다."

DEFAULT_INSTRUCT = "Speak naturally and expressively, with clear emotion and a warm tone."

CFG = TTSConfig(temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45)

_VBCABLE_ZIP_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"


def _cable_installed() -> bool:
    try:
        return any(
            "cable input" in d['name'].lower() and d['max_output_channels'] > 0
            for d in sd.query_devices()
        )
    except Exception:
        return False


def _get_output_devices() -> list[tuple[int, str]]:
    try:
        return [(i, f"[{i}] {d['name']}")
                for i, d in enumerate(sd.query_devices())
                if d['max_output_channels'] > 0]
    except Exception:
        return []

INIT_PHRASES = {
    "korean":          "안녕하세요, 반갑습니다.",
    "english":         "Hello, nice to meet you.",
    "japanese":        "こんにちは、よろしくお願いします。",
    "chinese":         "你好，很高兴认识你。",
    "german":          "Hallo, schön Sie kennenzulernen.",
    "french":          "Bonjour, enchanté de vous rencontrer.",
    "spanish":         "Hola, encantado de conocerte.",
    "russian":         "Привет, приятно познакомиться.",
    "italian":         "Ciao, piacere di conoscerti.",
    "portuguese":      "Olá, prazer em conhecê-lo.",
    "sichuan_dialect": "你好，幸会幸会。",
    "beijing_dialect": "你好，认识你很高兴。",
}


class TTSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Qwen3 TTS")
        self.root.minsize(480, 520)

        self.engine = None
        self.stream = None
        self.last_result: TTSResult | None = None
        self.synthesizing = False
        self.ui_queue: queue.Queue = queue.Queue()

        self._active_speaker  = DEFAULT_SPEAKER
        self._active_language = DEFAULT_LANGUAGE
        self._voice_ready     = False
        self._pending_meta_tag: str | None = None
        self._meta_tag_counter = 0

        self._cable_check_done = False
        self._previewing = False
        self._discord_tip_shown = False

        self._mic_device_id: int | None = None
        self._mic_device_map: dict[str, int] = {}
        self._mic_target_sr: int = 24000
        self._mic_buf      = np.zeros(0, dtype=np.float32)
        self._mic_buf_lock = threading.Lock()
        self._mic_out_stream: sd.OutputStream | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_engine()
        self.root.after(100, self._poll_queue)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)   # chat log expands

        # ── Top bar: dropdowns ─────────────────────────────────────────────
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

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=(10, 10))
        ttk.Label(top, text="Mic Out:").pack(side="left")
        self.mic_var = tk.StringVar(value="— disabled —")
        self.mic_cb  = ttk.Combobox(top, textvariable=self.mic_var, state="readonly", width=28)
        self.mic_cb.pack(side="left", padx=(4, 4))
        self.mic_cb.bind("<<ComboboxSelected>>", self._on_mic_selected)
        self.mic_refresh_btn = ttk.Button(top, text="↺", width=2, command=self._refresh_mic_devices)
        self.mic_refresh_btn.pack(side="left")

        self.stop_btn = ttk.Button(top, text="Stop", command=self._on_stop)
        self.stop_btn.pack(side="right")

        # ── Volume / mute row ──────────────────────────────────────────────
        vol = ttk.Frame(self.root)
        vol.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 0))

        self.spk_mute_var = tk.BooleanVar(value=False)
        self.spk_mute_btn = ttk.Checkbutton(vol, text="Mute Spk",
                                             variable=self.spk_mute_var,
                                             command=self._on_speaker_mute)
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

        # ── Tone prompt row ────────────────────────────────────────────────
        prompt_frame = ttk.Frame(self.root)
        prompt_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 0))
        prompt_frame.columnconfigure(1, weight=1)
        ttk.Label(prompt_frame, text="Tone:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.instruct_var = tk.StringVar(value=DEFAULT_INSTRUCT)
        self.instruct_entry = ttk.Entry(prompt_frame, textvariable=self.instruct_var,
                                        font=("Segoe UI", 9))
        self.instruct_entry.grid(row=0, column=1, sticky="ew")
        self.tone_test_btn = ttk.Button(prompt_frame, text="Test", width=5,
                                        command=self._on_tone_test)
        self.tone_test_btn.grid(row=0, column=2, padx=(6, 0))
        self.tone_reset_btn = ttk.Button(prompt_frame, text="Reset", width=5,
                                         command=self._on_tone_reset)
        self.tone_reset_btn.grid(row=0, column=3, padx=(4, 0))

        # ── Chat log ────────────────────────────────────────────────────────
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

        # Tags for bubble styling
        self.chat_log.tag_config("bubble",
            lmargin1=8, lmargin2=8, rmargin=8,
            spacing1=4, spacing3=4,
        )
        self.chat_log.tag_config("meta",
            foreground="#888888", font=("Segoe UI", 8),
            lmargin1=8, spacing3=2,
        )
        self.chat_log.tag_config("system",
            foreground="#555555", font=("Segoe UI", 9, "italic"),
            lmargin1=8, spacing1=2, spacing3=2,
        )

        # ── Input row ──────────────────────────────────────────────────────
        inp_frame = ttk.Frame(self.root)
        inp_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 4))
        inp_frame.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        self.entry = ttk.Entry(inp_frame, textvariable=self.input_var,
                               font=("Segoe UI", 10))
        self.entry.grid(row=0, column=0, sticky="ew", ipady=4)
        self.entry.bind("<Return>",       self._on_enter)
        self.entry.bind("<Shift-Return>", lambda e: None)  # let shift-enter do nothing

        self.send_btn = ttk.Button(inp_frame, text="Send", command=self._on_speak)
        self.send_btn.grid(row=0, column=1, padx=(6, 0))

        # ── Status bar ─────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Starting engine…")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               anchor="w", relief="sunken",
                               font=("Segoe UI", 8))
        status_bar.grid(row=5, column=0, sticky="ew")

        self._refresh_mic_devices()

    def _refresh_mic_devices(self):
        # Force PortAudio to re-enumerate devices (picks up newly installed drivers).
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        devices = _get_output_devices()
        self._mic_device_map = {label: dev_id for dev_id, label in devices}
        labels = ["— disabled —"] + list(self._mic_device_map)
        self.mic_cb.config(values=labels)
        if self.mic_var.get() not in labels:
            self.mic_var.set("— disabled —")
            self._mic_device_id = None
            self._close_mic_stream()
        # Auto-select CABLE Input if found and nothing is currently selected.
        # Prefer "cable input" (VB-Audio send side) over other cable devices.
        if self._mic_device_id is None:
            candidates = list(self._mic_device_map)
            target = next((l for l in candidates if "cable input" in l.lower()), None)
            if target:
                self.mic_var.set(target)
                self._mic_device_id = self._mic_device_map[target]
                self._open_mic_stream(self._mic_device_id)
                if not self._discord_tip_shown:
                    self._discord_tip_shown = True
                    self._append_system(
                        "Discord tip: set your mic input to CABLE Output, then go to "
                        "Settings → Voice & Video → disable 'Automatically determine input sensitivity' "
                        "and set it to 0, then disable Noise Suppression."
                    )

    def _on_mic_selected(self, _event=None):
        self._mic_device_id = self._mic_device_map.get(self.mic_var.get())
        if self._mic_device_id is not None:
            self._open_mic_stream(self._mic_device_id)
        else:
            self._close_mic_stream()

    def _on_speaker_vol(self, _val=None):
        vol = self.spk_vol_var.get()
        self.spk_vol_lbl.config(text=f"{vol}%")
        if self.stream and self.stream.decoder:
            self.stream.decoder._speaker_volume = vol / 100.0

    def _on_speaker_mute(self):
        muted = self.spk_mute_var.get()
        if self.stream and self.stream.decoder:
            self.stream.decoder._speaker_muted = muted

    def _on_mic_vol(self, _val=None):
        vol = self.mic_vol_var.get()
        self.mic_vol_lbl.config(text=f"{vol}%")

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
        if self.stream and self.stream.decoder:
            self.stream.decoder._chunk_callback = None
        with self._mic_buf_lock:
            self._mic_buf = np.zeros(0, dtype=np.float32)

    def _on_mic_chunk(self, audio: np.ndarray):
        """Called from proxy's listener thread for each decoded audio chunk."""
        if self._mic_out_stream is None:
            return
        vol = self.mic_vol_var.get() / 100.0
        if vol == 0.0:
            return
        if vol != 1.0:
            audio = np.clip(audio * vol, -1.0, 1.0)
        if self._mic_target_sr != 24000:
            orig_len = len(audio)
            target_len = int(orig_len * self._mic_target_sr / 24000)
            audio = np.interp(
                np.linspace(0, orig_len - 1, target_len),
                np.arange(orig_len),
                audio,
            ).astype(np.float32)
        with self._mic_buf_lock:
            self._mic_buf = np.concatenate([self._mic_buf, audio])

    # ── Chat log helpers ───────────────────────────────────────────────────

    def _append_chat_pending(self, text: str, speaker: str, language: str):
        """Insert bubble immediately with 'Synthesizing…'; tag the meta line for later patch."""
        self.chat_log.config(state="normal")
        meta = f"{speaker.title()}  ·  {language}  ·  Synthesizing…"
        self._meta_tag_counter += 1
        tag = f"_pending_meta_{self._meta_tag_counter}"
        self._pending_meta_tag = tag
        # Apply both "meta" (style) and the unique tag so we can find this exact line later
        self.chat_log.insert("end", meta + "\n", ("meta", tag))
        self.chat_log.insert("end", text + "\n\n", "bubble")
        self.chat_log.config(state="disabled")
        self.chat_log.see("end")

    def _finalize_chat_meta(self, speaker: str, language: str, rtf: float | None):
        """Replace only the meta line text; leave the bubble text untouched."""
        tag = self._pending_meta_tag
        if not tag:
            return
        self.chat_log.config(state="normal")
        meta = f"{speaker.title()}  ·  {language}"
        meta += f"  ·  RTF {rtf:.2f}" if rtf is not None else "  ·  no audio"
        ranges = self.chat_log.tag_ranges(tag)
        if ranges:
            start = ranges[0]
            # lineend stops before the \n — so delete replaces only the meta text
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

    # ── Engine init ────────────────────────────────────────────────────────

    def _start_engine(self):
        self._set_input_state("disabled")
        threading.Thread(target=self._init_engine_thread, daemon=True).start()

    def _init_engine_thread(self):
        try:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            engine = TTSEngine(models_dir, onnx_provider="DML", verbose=False)
            stream = engine.create_stream()
            if stream is None:
                self.ui_queue.put(("error", "Engine init failed — check model path."))
                return
            self.engine = engine
            self.stream = stream
            self.ui_queue.put(("system", f"Initializing {DEFAULT_SPEAKER.title()} voice…"))
            self._init_voice_thread(DEFAULT_SPEAKER, DEFAULT_LANGUAGE, INIT_TEXT)
        except Exception as exc:
            self.ui_queue.put(("error", f"Engine error: {exc}"))

    def _init_voice_thread(self, speaker: str, language: str, init_text: str):
        try:
            self.stream.set_voice(speaker, text=init_text, language=language)
            self.stream.join()
            self._active_speaker  = speaker
            self._active_language = language
            self._voice_ready = True
            self.ui_queue.put(("ready", f"Ready — {speaker.title()} / {language}"))
        except Exception as exc:
            self.ui_queue.put(("error", f"Voice init failed: {exc}"))

    # ── Input handlers ─────────────────────────────────────────────────────

    def _on_enter(self, _event):
        self._on_speak()
        return "break"   # prevent default newline insertion

    def _on_speak(self):
        if self.synthesizing:
            return
        text = self.input_var.get().strip()
        if not text:
            return

        speaker  = self.voice_var.get()
        language = self.lang_var.get()

        self.input_var.set("")
        self._append_chat_pending(text, speaker, language)  # show immediately

        self.synthesizing = True
        self._set_input_state("disabled")
        self._status("Synthesizing…")

        threading.Thread(
            target=self._speak_thread,
            args=(text, speaker, language),
            daemon=True,
        ).start()

    def _speak_thread(self, text: str, speaker: str, language: str):
        rtf = None
        try:
            if speaker != self._active_speaker or language != self._active_language:
                self.ui_queue.put(("system", f"Switching to {speaker.title()} / {language}…"))
                init_text = INIT_PHRASES.get(language, "Hello.")
                self._init_voice_thread(speaker, language, init_text)

            if not self._voice_ready:
                self.ui_queue.put(("error", "Voice not ready — try again."))
                return

            # Wire mic chunk callback and sync volume/mute right before synthesis
            if self.stream and self.stream.decoder:
                dec = self.stream.decoder
                dec._chunk_callback = (
                    self._on_mic_chunk
                    if self._mic_device_id is not None and self._mic_out_stream is not None
                    else None
                )
                dec._speaker_volume = self.spk_vol_var.get() / 100.0
                dec._speaker_muted  = self.spk_mute_var.get()

            instruct = self.instruct_var.get().strip() or None
            result = self.stream.clone(text, config=CFG, instruct=instruct)
            self.stream.join()
            self.last_result = result

            if result:
                rtf = result.rtf if result.rtf != 0.0 else None
                if rtf is None:
                    self.ui_queue.put(("system", "No audio generated (RTF 0.00). Try a fuller sentence."))

            self.ui_queue.put(("done", (speaker, language, rtf)))

        except RuntimeError as exc:
            self.ui_queue.put(("error", f"Error: {exc}"))
        except Exception as exc:
            self.ui_queue.put(("error", f"Unexpected error: {exc}"))
        finally:
            self.synthesizing = False
            self.ui_queue.put(("enable_input", None))

    def _on_tone_test(self):
        if self.synthesizing or self._previewing or not self._voice_ready:
            return
        speaker  = self.voice_var.get()
        language = self.lang_var.get()
        text     = INIT_PHRASES.get(language, "Hello.")
        self._previewing = True
        self._status(f"Testing tone for {speaker.title()}…")
        threading.Thread(target=self._tone_test_thread, args=(speaker, language, text),
                         daemon=True).start()

    def _tone_test_thread(self, speaker: str, language: str, text: str):
        try:
            if self.stream and self.stream.decoder:
                self.stream.decoder._chunk_callback = None
            instruct = self.instruct_var.get().strip() or None
            self.stream.custom(text, speaker, language=language, instruct=instruct, config=CFG)
            self.stream.join()
        except Exception as exc:
            self.ui_queue.put(("system", f"Tone test error: {exc}"))
        finally:
            self._previewing = False
            self.ui_queue.put(("system", "Tone test done."))

    def _on_tone_reset(self):
        self.instruct_var.set(DEFAULT_INSTRUCT)

    def _on_voice_preview(self, _event=None):
        if self.synthesizing or self._previewing or not self._voice_ready:
            return
        speaker  = self.voice_var.get()
        language = self.lang_var.get()
        self._previewing = True
        self._status(f"Previewing {speaker.title()}…")
        threading.Thread(
            target=self._preview_thread,
            args=(speaker, language),
            daemon=True,
        ).start()

    def _preview_thread(self, speaker: str, language: str):
        try:
            # Suppress mic routing so preview plays only through the speaker.
            if self.stream and self.stream.decoder:
                self.stream.decoder._chunk_callback = None

            init_text = INIT_PHRASES.get(language, "Hello.")
            self._init_voice_thread(speaker, language, init_text)
        finally:
            self._previewing = False

    def _on_stop(self):
        if self.stream:
            try:
                self.stream.reset()
            except Exception:
                pass
        self.synthesizing = False
        self._set_input_state("normal")
        self._status("Stopped.")

    # ── Queue polling ──────────────────────────────────────────────────────

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
                    if not self._cable_check_done:
                        self._cable_check_done = True
                        if not _cable_installed():
                            self._offer_cable_install()
                elif kind == "done":
                    speaker, language, rtf = payload
                    self._finalize_chat_meta(speaker, language, rtf)
                    status = f"Ready — {speaker.title()} / {language}"
                    if rtf is not None:
                        status += f"   RTF: {rtf:.2f}"
                    self._status(status)
                elif kind == "enable_input":
                    self._set_input_state("normal")
                    self.entry.focus()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.status_var.set(msg)

    def _set_input_state(self, state: str):
        for w in (self.send_btn, self.stop_btn, self.voice_cb, self.lang_cb,
                  self.mic_cb, self.mic_refresh_btn,
                  self.tone_test_btn, self.tone_reset_btn):
            w.config(state=state)
        self.entry.config(state=state)
        self.instruct_entry.config(state=state)

    # ── VB-Audio Virtual Cable install ─────────────────────────────────────

    def _offer_cable_install(self):
        if not messagebox.askyesno(
            "VB-Audio Virtual Cable not found",
            "CABLE Input device not detected.\n\n"
            "Install VB-Audio Virtual Cable now?\n"
            "(Requires internet + admin approval)",
            icon="question",
        ):
            return
        self._append_system("Downloading VB-Audio Virtual Cable installer…")
        threading.Thread(target=self._cable_install_thread, daemon=True).start()

    def _cable_install_thread(self):
        # ── Option A: Chocolatey (if available) ──────────────────────────────
        try:
            r = subprocess.run(
                ["choco", "install", "vb-cable", "-y"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                self.ui_queue.put(("system",
                    "VB-Audio Virtual Cable installed via Chocolatey. Click ↺ to refresh."))
                return
        except FileNotFoundError:
            pass  # choco not installed — fall through to direct download

        # ── Option B: Direct download + elevated installer ────────────────────
        try:
            # Use mkdtemp (no auto-cleanup) so the folder persists while the
            # elevated installer process (launched async) still has the exe open.
            tmp = tempfile.mkdtemp()
            zip_path = os.path.join(tmp, "VBCABLE.zip")
            self.ui_queue.put(("system", "Downloading VBCABLE_Driver_Pack43.zip…"))
            urllib.request.urlretrieve(_VBCABLE_ZIP_URL, zip_path)

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)

            exe = os.path.join(tmp, "VBCABLE_Setup_x64.exe")
            if not os.path.exists(exe):
                exe = os.path.join(tmp, "VBCABLE_Setup.exe")  # 32-bit fallback

            # ShellExecuteW with "runas" prompts UAC and runs the installer elevated.
            # It returns immediately (async) — user must click ↺ after install finishes.
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", exe, "/VERYSILENT /NORESTART", os.path.dirname(exe), 1
            )
            self.ui_queue.put(("system",
                "Installer launched. Approve the UAC + driver dialogs, "
                "then click ↺ to refresh devices."))
        except Exception as exc:
            webbrowser.open("https://vb-audio.com/Cable/")
            self.ui_queue.put(("system",
                f"Download failed ({exc}). Browser opened — install manually, then click ↺."))

    # ── Shutdown ───────────────────────────────────────────────────────────

    def _on_close(self):
        self._close_mic_stream()
        if self.engine:
            try:
                self.engine.shutdown()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required for PyInstaller + multiprocessing on Windows
    main()
