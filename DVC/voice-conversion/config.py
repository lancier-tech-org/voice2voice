"""
Central configuration for the voice conversion system.
All tunable parameters in one place.
"""

import os
from pathlib import Path

# =============================================================
# PATHS
# =============================================================
BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voices"
UPLOADS_DIR = BASE_DIR / "uploads"
PRETRAINED_DIR = BASE_DIR / "pretrained"

# Auto-create directories
for d in [VOICES_DIR, UPLOADS_DIR, PRETRAINED_DIR]:
    d.mkdir(exist_ok=True)

# =============================================================
# AUDIO SETTINGS
# =============================================================
SAMPLE_RATE = 40000            # Model's native sample rate
INPUT_SAMPLE_RATE = 16000      # ContentVec expects 16kHz input
HOP_LENGTH = 320               # Hop size for feature extraction
WIN_LENGTH = 1280              # Window size

# =============================================================
# STREAMING / REAL-TIME SETTINGS
# =============================================================
CHUNK_DURATION_MS = 160        # Size of each audio chunk in milliseconds
OVERLAP_RATIO = 0.5            # 50% overlap between chunks for crossfade
CROSSFADE_DURATION_MS = 40     # Crossfade window at chunk boundaries

# Derived values
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)      # 6400 samples
OVERLAP_SAMPLES = int(CHUNK_SAMPLES * OVERLAP_RATIO)              # 3200 samples
CROSSFADE_SAMPLES = int(SAMPLE_RATE * CROSSFADE_DURATION_MS / 1000)  # 1600 samples

# WebSocket buffer: accumulate this many ms before processing
WS_BUFFER_MS = 320             # Process every 320ms (2 chunks worth)

# =============================================================
# MODEL SETTINGS
# =============================================================
# ContentVec
CONTENTVEC_MODEL = "contentvec_base"  # 768-dim content features
CONTENTVEC_DIM = 768

# Speaker embedding
SPEAKER_EMBED_DIM = 256

# Synthesizer (HiFi-GAN based)
SYNTH_HIDDEN_DIM = 512
SYNTH_UPSAMPLE_RATES = [10, 8, 2, 2]  # Total: 320x (matches HOP_LENGTH)
SYNTH_UPSAMPLE_KERNELS = [20, 16, 4, 4]
SYNTH_RESBLOCK_KERNEL_SIZES = [3, 7, 11]
SYNTH_RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

# =============================================================
# TRAINING SETTINGS
# =============================================================
TRAINING_EPOCHS = 200
TRAINING_BATCH_SIZE = 8
TRAINING_LR = 2e-4
TRAINING_LR_DECAY = 0.999
TRAINING_SEGMENT_SIZE = 32000  # ~800ms segments for training (longer = better quality)

# Preprocessing
MIN_AUDIO_DURATION_SEC = 300   # Minimum 5 minutes of audio
MAX_AUDIO_DURATION_SEC = 3600  # Maximum 1 hour
SILENCE_THRESHOLD_DB = -40     # Below this is considered silence
MIN_SEGMENT_DURATION_SEC = 2   # Minimum segment length after slicing
MAX_SEGMENT_DURATION_SEC = 15  # Maximum segment length

# =============================================================
# PITCH (F0) SETTINGS
# =============================================================
F0_METHOD = "rmvpe"            # Most accurate for cross-gender
F0_MIN = 50                    # Hz - lowest expected pitch
F0_MAX = 1100                  # Hz - highest expected pitch (covers soprano)

# Cross-gender pitch shift (semitones) - applied at inference
PITCH_SHIFT_MALE_TO_FEMALE = 12
PITCH_SHIFT_FEMALE_TO_MALE = -12

# =============================================================
# SERVER SETTINGS
# =============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 500       # Max upload file size
ALLOWED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# =============================================================
# GPU SETTINGS
# =============================================================
DEVICE = "cuda"                # Change to "cpu" if no GPU
GPU_ID = 0                     # Which GPU to use
HALF_PRECISION = True          # Use FP16 for faster inference (RTX 6000 supports it)

# =============================================================
# PRE-TRAINED MODEL URLS
# =============================================================
PRETRAINED_URLS = {
    "contentvec": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
    "rmvpe": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
}