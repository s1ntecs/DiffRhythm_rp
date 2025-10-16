# ===== Base =====
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    HF_HOME=/.cache/huggingface \
    TORCH_HOME=/.cache/torch \
    TOKENIZERS_PARALLELISM=false

# ----- System deps -----
# ffmpeg/libsndfile — для аудио; espeak-ng — нужен DiffRhythm; dos2unix — лечим CRLF у скриптов
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3.10-venv \
    git git-lfs curl wget ca-certificates \
    ffmpeg libsndfile1 espeak-ng dos2unix \
 && rm -rf /var/lib/apt/lists/* && git lfs install

WORKDIR /workspace

# ----- Torch cu124 + тащим зависимости -----
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Ставим зависимости проекта (если есть)
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# RunPod SDK
RUN python3 -m pip install runpod

# ----- Копируем весь проект в -----
COPY . .

RUN dos2unix start_standalone.sh && chmod 755 start_standalone.sh
COPY start_standalone.sh /start.sh
RUN chmod 755 /start.sh

ENTRYPOINT ["/start.sh"]
