# =============================================================================
#  Voice Conversion Service — production image
#  Base: CUDA 12.4 runtime (matches the torch 2.6.0 cu124 wheels)
#  Host requirement: NVIDIA driver >= 550 + nvidia-container-toolkit
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model caches. These MUST match the values used by prewarm_models.py,
    # and must not be overridden at run time or the app re-downloads weights.
    HF_HOME=/opt/models/hf \
    TORCH_HOME=/opt/models/torch \
    # Belt-and-braces: refuse to phone home if a cache miss ever happens.
    # Remove this line if you deliberately want runtime downloads.
    HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------
# System packages
#   python3.11        - target interpreter (chatterbox-tts requires >=3.10)
#   libsndfile1       - required by soundfile, used by librosa
#   ffmpeg            - librosa/audioread fallback decoder for mp3/m4a/ogg
#                       reference voices uploaded via /api/voices/upload
#   git               - torch.hub may shell out to git for the Silero repo
#   curl              - container health probe
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        libsndfile1 \
        ffmpeg \
        git \
        curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies (own layer — changes rarely, ~6-8 GB, cache it)
# ---------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Bake model weights into the image (own layer — see prewarm_models.py)
# Adds roughly 1-2 GB. Removes all runtime egress to huggingface.co/github.com.
# ---------------------------------------------------------------------------
COPY prewarm_models.py /app/prewarm_models.py
# HF_HUB_OFFLINE=1 is set in the ENV block at the top of this file, which also
# applies at BUILD time — so this step, whose whole job is to download the
# weights, was forbidden from reaching the network and failed every build with
#     OfflineModeIsEnabled: Cannot reach https://huggingface.co/ResembleAI/...
# Overriding the flag for this one RUN keeps the runtime guard intact while
# letting the prewarm actually download. Do not move the ENV line instead:
# editing it invalidates the cache for the ~6-8 GB pip layer above.
RUN HF_HUB_OFFLINE=0 python /app/prewarm_models.py

# ---------------------------------------------------------------------------
# Application source
#
# Build context is the PROJECT ROOT (the directory holding backend/ and
# frontend/), not the backend directory:
#     docker build -t voice-vc:1.0.0 .
#
# IMPORTANT — the layout is not arbitrary.
#   * config.py sets VOICE_REF_DIR = "voices", a RELATIVE path, so the process
#     working directory MUST be the directory containing the .py files.
#   * main.py resolves the web client as  os.path.dirname(__file__)/../frontend
#     so the static assets must sit one level ABOVE the backend directory.
# The image reproduces the repository layout exactly: /app/backend + /app/frontend.
# ---------------------------------------------------------------------------
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Reference voices are resolved as ./voices relative to the working directory
# below, i.e. /app/backend/voices — matching backend/voices/ in the repo.
# NOTE: if a second voices/ directory exists at the project root, it is NOT
# the one the service reads. See README, item K-11.
WORKDIR /app/backend

EXPOSE 8131

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8131/api/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
#
# We invoke uvicorn directly rather than `python main.py`. The __main__ block
# in main.py hardcodes ssl_keyfile="../certs/key.pem", which does not exist in
# this image — TLS is terminated by Azure ingress / Application Gateway, not by
# the application. See README § Known Issues, item K-03.
#
# --workers is intentionally absent: the model is a single shared, MUTABLE
# GPU object. Do not add workers without reading README § Known Issues, K-05.
# --ws-ping-interval / --ws-ping-timeout keep idle sockets alive through the
# Azure ingress idle timeout.
# ---------------------------------------------------------------------------
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8131", \
     "--log-level", "info", \
     "--timeout-keep-alive", "75", \
     "--ws-ping-interval", "20", \
     "--ws-ping-timeout", "20"]
