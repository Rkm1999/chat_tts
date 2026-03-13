# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# ── Extra binaries ──────────────────────────────────────────────────────────
# llama.cpp DLLs — include all DLLs from the inference/bin folder
inference_bin = os.path.join('qwen3_tts_gguf', 'inference', 'bin')
llama_dlls = [
    (os.path.join(inference_bin, f), os.path.join('qwen3_tts_gguf', 'inference', 'bin'))
    for f in os.listdir(inference_bin)
    if f.endswith('.dll')
]

# PortAudio DLL (sounddevice)
portaudio_dir = os.path.join(
    '..', '.venv', 'Lib', 'site-packages',
    '_sounddevice_data', 'portaudio-binaries'
)
portaudio_dlls = [
    (os.path.join(portaudio_dir, f), '_sounddevice_data/portaudio-binaries')
    for f in os.listdir(portaudio_dir)
    if f.endswith('.dll')
]

binaries = llama_dlls + portaudio_dlls

# ── Data files ──────────────────────────────────────────────────────────────
datas = (
    collect_data_files('tokenizers')
    + collect_data_files('onnxruntime')
    + [
        # _sounddevice_data helper module
        (
            os.path.join('..', '.venv', 'Lib', 'site-packages', '_sounddevice_data'),
            '_sounddevice_data',
        ),
    ]
)

# ── Analysis ────────────────────────────────────────────────────────────────
a = Analysis(
    ['gui.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'sounddevice',
        '_sounddevice',
        'tokenizers',
        'tokenizers.implementations',
        'onnxruntime',
        'numpy',
        'numpy.core._multiarray_umath',
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'qwen3_tts_gguf',
        'qwen3_tts_gguf.inference',
        'qwen3_tts_gguf.inference.schema',
        'qwen3_tts_gguf.inference.schema.constants',
        'qwen3_tts_gguf.inference.utils',
        'qwen3_tts_gguf.inference.workers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pandas', 'PIL', 'cv2'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ChatTTS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ChatTTS',
)
