# ===== Base =====
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
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
COPY requirements.txt /workspace/requirements.txt
RUN if [ -f /workspace/requirements.txt ]; then \
      python3 -m pip install -r /workspace/requirements.txt ; \
    fi

# RunPod SDK
RUN python3 -m pip install runpod

# ----- Копируем весь проект в /workspace -----
COPY . /workspace/

# Чиним строки и права у стартового скрипта
RUN dos2unix /workspace/start_standalone.sh && \
    chmod 755  /workspace/start_standalone.sh

# Кэши на будущее
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch

# (опц.) предзагрузка весов — ускорит cold start, но увеличит образ
# RUN python3 -m pip install huggingface_hub && python3 /workspace/download_models.py

# Лучше явно указать PYTHONPATH на корень (на случай смены WORKDIR)
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# ----- Запуск -----
ENTRYPOINT ["/workspace/start_standalone.sh"]
