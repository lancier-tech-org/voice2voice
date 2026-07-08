# Real-Time Voice Conversion System

## Architecture Overview

A production-grade real-time voice conversion system that transfers vocal timbre (tone color) from a trained target voice onto a live speaker's audio, preserving the live speaker's accent, speed, rhythm, and pitch contour.

## System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser (User B)                   │
│  ┌──────────┐    WebSocket     ┌──────────────────────┐  │
│  │ Mic Input ├────────────────►│ Converted Audio Out  │  │
│  └──────────┘   (binary PCM)   └──────────────────────┘  │
└────────────────────┬──────────────────▲──────────────────┘
                     │                  │
                     ▼                  │
┌─────────────────────────────────────────────────────────┐
│               FastAPI Backend (Python)                    │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              WebSocket Handler                       │ │
│  │  Audio In → Buffer → Convert → Stream Out            │ │
│  └──────────┬──────────────────────────────┬───────────┘ │
│             │                              │             │
│  ┌──────────▼──────────┐  ┌────────────────▼───────────┐ │
│  │  Content Encoder     │  │   Vocoder + Voice Model   │ │
│  │  (ContentVec/HuBERT) │  │   (HiFi-GAN + target)    │ │
│  └──────────┬──────────┘  └────────────────▲───────────┘ │
│             │                              │             │
│  ┌──────────▼──────────┐  ┌────────────────┴───────────┐ │
│  │  Pitch Extractor     │  │  Target Voice Checkpoint  │ │
│  │  (RMVPE)             │  │  (loaded from disk)       │ │
│  └─────────────────────┘  └────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Training Pipeline (async)                  │ │
│  │  Upload → Preprocess → Train → Save Checkpoint       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
voice-conversion/
├── README.md
├── requirements.txt
├── config.py                  # All configuration constants
│
├── models/                    # ML model definitions
│   ├── __init__.py
│   ├── content_encoder.py     # ContentVec/HuBERT wrapper
│   ├── pitch_extractor.py     # RMVPE F0 extraction
│   ├── synthesizer.py         # HiFi-GAN vocoder + speaker conditioning
│   └── speaker_encoder.py     # Speaker embedding extraction
│
├── training/                  # Voice training pipeline
│   ├── __init__.py
│   ├── preprocessor.py        # Audio cleaning, slicing, feature extraction
│   ├── dataset.py             # PyTorch dataset for training
│   └── trainer.py             # Training loop + checkpoint saving
│
├── inference/                 # Real-time conversion
│   ├── __init__.py
│   ├── voice_converter.py     # Core conversion pipeline
│   └── stream_processor.py    # Chunked streaming with overlap-crossfade
│
├── api/                       # FastAPI server
│   ├── __init__.py
│   ├── main.py                # App entry point
│   ├── routes_train.py        # Training endpoints (upload, status)
│   ├── routes_convert.py      # WebSocket conversion endpoint
│   └── routes_voices.py       # Voice management (list, delete, select)
│
├── frontend/                  # Web UI
│   └── index.html             # Single-page app
│
├── voices/                    # Trained voice checkpoints (auto-created)
├── uploads/                   # Temporary upload storage (auto-created)
└── pretrained/                # Pre-trained base models (downloaded on first run)
```

## Setup Instructions

### 1. System Requirements
- Python 3.10+
- CUDA 11.8+ with cuDNN
- NVIDIA GPU with 8GB+ VRAM (RTX 6000 Ada is ideal)
- ffmpeg installed (`apt install ffmpeg`)

### 2. Run Setup
```bash
cd voice-conversion
chmod +x setup.sh
./setup.sh
```

This creates a virtual environment, installs all dependencies (with CUDA-enabled PyTorch if a GPU is detected), and downloads pre-trained models (~1.4GB total):
- ContentVec (content encoder) ~1.2GB
- RMVPE (pitch extractor) ~180MB

### 3. Run the Server
```bash
source venv/bin/activate
python -m api.main
```
Server starts at http://localhost:8000

### 5. Open the Web UI
Navigate to http://localhost:8000 in your browser.

## Usage Flow

1. **Train a voice**: Upload 10-15 minutes of clean audio for User A
2. **Wait for training**: Auto-starts on upload, ~20-30 min on RTX 6000
3. **Select the voice**: Choose from trained voices dropdown
4. **Start conversion**: Click "Start" → speak into your mic → hear converted output

## Key Design Decisions

- **ContentVec over HuBERT**: Better speaker-independent content features
- **RMVPE for pitch**: Most accurate F0 extractor, handles cross-gender well
- **HiFi-GAN v2 vocoder**: Best quality-to-speed ratio for real-time
- **160ms chunks with 50% overlap**: Sweet spot for latency vs smoothness
- **WebSocket binary streaming**: Lowest overhead for browser ↔ server audio
- **Separate F0 conditioning**: Enables cross-gender by keeping pitch from source
