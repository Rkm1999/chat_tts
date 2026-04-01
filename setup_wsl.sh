#!/bin/bash
set -e
VENV="$HOME/chat_tts_venv"
REPO="$(cd "$(dirname "$0")" && pwd)"
PIP="$VENV/bin/pip"

echo '--- Creating venv ---'
python3 -m venv "$VENV"
echo 'VENV_OK'

echo '--- Upgrading pip ---'
$PIP install --upgrade pip -q

echo '--- Installing PyTorch with CUDA 12.4 ---'
$PIP install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -q
echo 'TORCH_OK'

echo '--- Installing requirements ---'
$PIP install -r "$REPO/requirements.txt" -q
echo 'REQS_OK'

echo '--- Installing qwen3-tts-triton ---'
$PIP install qwen3-tts-triton==0.1.0 --ignore-requires-python --no-deps -q
echo 'TRITON_OK'

echo ''
echo 'Done! Activate with: source ~/chat_tts_venv/bin/activate'
