FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    TOKENIZERS_PARALLELISM=false

# Системные зависимости: espeak-ng (их packages.txt), плюс аудио тулзы
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3.10-venv git \
    espeak-ng ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2) Copy requirements
COPY requirements.txt .

# PyTorch 2.6.0 / CUDA 12.4 (важно: torchaudio==2.6.0 у них в requirements)
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

RUN python3 -m pip install -r /app/requirements.txt

# RunPod SDK
RUN python3 -m pip install runpod

COPY . .

# Кеш директории (ускоряет cold start)
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch

COPY --chmod=755 start_standalone.sh /start.sh
ENTRYPOINT ["/start.sh"]
