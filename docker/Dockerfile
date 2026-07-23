###############################################################################
# Voice2Voice Dockerfile — CUDA 12.1 + Python 3.10 + RVC
# Target: Standard_NC4as_T4_v3 (NVIDIA T4 GPU)
# NOTE: Python 3.10 required — fairseq 0.12.2 incompatible with Python 3.11
###############################################################################

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ─── System Dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    g++ \
    git \
    curl \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*
# The ffmpeg *-dev headers above are required to build PyAV, which the `rvc`
# package depends on. Without them `pip install rvc` fails at the wheel build.

# Use Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ─── Working Directory ──────────────────────────────────────────────────────
WORKDIR /app

# ─── Python Dependencies ────────────────────────────────────────────────────
COPY DVC/voice-conversion/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir pip==23.3.2 && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ─── Vendored RVC-WebUI ─────────────────────────────────────────────────────
# Copied BEFORE the app code so the ~1.9GB of model downloads below stay cached
# when only application code changes.
#
# It lives at /app/rvc-webui, NOT /app/rvc, and must not be importable as the
# top-level `rvc` package. This app is written against the `rvc` PyPI package,
# which provides VC() and vc_single(..., index_file=, hubert_path=). Placing the
# WebUI at /app/rvc and symlinking rvc/modules made
# `import rvc.modules.vc.modules` resolve to the WebUI instead, silently
# swapping in a different, incompatible engine whose vc_single() takes
# file_index/file_index2 and no hubert_path.
# The WebUI is still used for TRAINING (subprocess with cwd=/app/rvc-webui).
COPY DRW/rvc-webui /app/rvc-webui

# ─── Inference models (~370MB) ──────────────────────────────────────────────
RUN mkdir -p /app/rvc-webui/assets/hubert /app/rvc-webui/assets/rmvpe \
             /app/voice-conversion/pretrained && \
    curl -fL -o /app/rvc-webui/assets/hubert/hubert_base.pt \
      https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt && \
    curl -fL -o /app/rvc-webui/assets/rmvpe/rmvpe.pt \
      https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt && \
    cp /app/rvc-webui/assets/hubert/hubert_base.pt /app/voice-conversion/pretrained/hubert_base.pt

# ─── Pretrained BASE models for TRAINING (~1.5GB) ───────────────────────────
# Required by rvc_trainer. RVC fine-tunes from these; without them training
# starts from random weights and produces a valid-looking but badly degraded
# voice with no error. Their absence is what corrupted a retrained model here.
RUN mkdir -p /app/rvc-webui/assets/pretrained_v2 && \
    for f in f0G32k.pth f0D32k.pth f0G40k.pth f0D40k.pth f0G48k.pth f0D48k.pth; do \
      curl -fL -o /app/rvc-webui/assets/pretrained_v2/$f \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/$f || exit 1; \
    done

# ─── Copy Application Code (changes often — kept after the big downloads) ────
COPY DVC/voice-conversion /app/voice-conversion

# ─── RVC-WebUI training scripts resolve assets relative to the working dir ──
RUN ln -sf /app/rvc-webui/assets /app/voice-conversion/assets

# ─── Copy RVC configs to voice-conversion working directory ─────────────────
# RVC-WebUI's Config reads configs/inuse/<ver>/<sr>.json and copies from
# configs/<ver>/ if absent — but shutil.copy cannot create the intermediate
# inuse/<ver> directory, so it must already exist and be populated.
RUN mkdir -p /app/voice-conversion/configs/inuse && \
    cp -r /app/rvc-webui/configs/v1 /app/voice-conversion/configs/ 2>/dev/null || true && \
    cp -r /app/rvc-webui/configs/v2 /app/voice-conversion/configs/ 2>/dev/null || true && \
    cp -r /app/rvc-webui/configs/v1 /app/voice-conversion/configs/inuse/ 2>/dev/null || true && \
    cp -r /app/rvc-webui/configs/v2 /app/voice-conversion/configs/inuse/ 2>/dev/null || true && \
    cp /app/rvc-webui/configs/config.json /app/voice-conversion/configs/ 2>/dev/null || true

# ─── Set Python path ────────────────────────────────────────────────────────
# /app/rvc-webui is deliberately absent: it must not shadow the `rvc` package.
ENV PYTHONPATH="/app/voice-conversion:/app:${PYTHONPATH}"

# ─── Create Required Directories ────────────────────────────────────────────
RUN mkdir -p /app/voice-conversion/voices \
             /app/voice-conversion/uploads \
             /app/voice-conversion/pretrained

# ─── Expose Port ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ─── Health Check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── Start Application ──────────────────────────────────────────────────────
WORKDIR /app/voice-conversion
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "16777216"]
