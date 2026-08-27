"""
Configuration for Voice Conversion Pipeline
Chatterbox VC — audio in, audio out. No STT.
"""

# ─── Audio ───────────────────────────────────────────────────
SAMPLE_RATE_IN = 16000        # Browser mic capture rate
SAMPLE_RATE_OUT = 24000       # Chatterbox VC output rate
CHANNELS = 1
DTYPE_IN = "int16"

# ─── VAD (Silero) ────────────────────────────────────────────
VAD_THRESHOLD = 0.45
VAD_MIN_SPEECH_MS = 500       # Ignore speech shorter than this
VAD_MIN_SILENCE_MS = 400      # Silence needed to end a segment
VAD_WINDOW_SIZE = 512

# ─── VC (Chatterbox) ────────────────────────────────────────
XTTS_DEVICE = "cuda"
XTTS_LANGUAGE = "en"

# ─── Voice Reference ────────────────────────────────────────
VOICE_REF_DIR = "voices"
DEFAULT_VOICE_REF = None

# ─── Server ──────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8131
