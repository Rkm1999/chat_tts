#!/bin/bash
set -e
PIP=/home/ryu/chat_tts_venv/bin/pip

echo '--- Upgrading pip ---'
$PIP install --upgrade pip -q

echo '--- Installing PyTorch with CUDA 12.4 ---'
$PIP install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -q
echo 'TORCH_OK'

echo '--- Installing requirements.txt ---'
$PIP install -r /mnt/c/git/chat_tts/requirements.txt -q
echo 'REQS_OK'

echo '--- Installing qwen3-tts-triton ---'
$PIP install qwen3-tts-triton==0.1.0 --ignore-requires-python --no-deps -q
echo 'TRITON_OK'
