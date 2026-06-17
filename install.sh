#!/usr/bin/env bash
set -euo pipefail

python --version
python -m pip --version

python -m pip install https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.10.0+cu130" \
    "torchvision==0.25.0+cu130" \
    "torchaudio==2.10.0+cu130"

git submodule update --init --recursive --remote
python -m pip install -e .
python -m pip install -e ./lingbotvla/models/vla/vision_models/lingbot-depth/ --no-deps
python -m pip install \
    "huggingface_hub>=0.34.2,<0.36.0" \
    "numpy>=2.0.0,<2.3.0" \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \
    "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6"
python -m pip install -e ./lingbotvla/models/vla/vision_models/MoGe/ --no-deps

if command -v nvcc >/dev/null 2>&1 && python -c 'import sys, torch; sys.exit(0 if torch.version.cuda else 1)'; then
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}" 
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn==2.8.3 --no-build-isolation
else
    echo "Skipping flash-attn: CUDA build environment not detected. Ensure nvcc is on PATH and torch is CUDA-enabled."
fi
