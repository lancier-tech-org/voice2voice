# Real-Time Voice Conversion Service

A GPU-backed WebSocket service that converts a live microphone stream into a different speaker's voice in near real time, using Resemble AI's [Chatterbox](https://github.com/resemble-ai/chatterbox) model.

**No speech recognition is involved.** Audio goes in, audio comes out. Nothing is transcribed to text, so the output can't hallucinate words — the speaker's original timing, rhythm, prosody and emotion are preserved. Only the vocal identity changes.

---

## How It Works

1. A browser captures microphone audio and streams raw PCM over a WebSocket.
2. The server uses a voice-activity detector (Silero VAD) to find where each phrase begins and ends.
3. When a phrase completes, the Chatterbox model re-synthesises it in a different voice.
4. The converted audio is streamed back over the same socket and played in the browser.

The target voice is defined by a short **reference recording** — a `.wav` file of the voice you want to sound like.

Conversion is **phrase-by-phrase, not continuous**. The listener hears nothing until the speaker pauses, then the entire phrase arrives at once. End-to-end latency is roughly **1.2 seconds** from the moment the speaker stops.

---

## Project Structure

```
stt-voice-clone/
├── backend/
│   ├── main.py              FastAPI app: HTTP routes + /ws endpoint
│   ├── pipeline.py          Per-session orchestration and statistics
│   ├── vad_engine.py        Silero VAD — phrase segmentation
│   ├── tts_engine.py        Chatterbox VC — the conversion model
│   ├── config.py            All tunables
│   └── voices/              Reference voice recordings
│       ├── vineesha.wav
│       └── SW_pavan_sai_12.wav
│
├── frontend/
│   └── index.html           Browser client (mic capture, WebSocket, playback)
│
├── Dockerfile
├── docker-compose.yml
├── prewarm_models.py        Bakes model weights into the Docker image
└── requirements.txt
```

---

## Requirements

- **Python** 3.11 (3.10–3.12 supported)
- **NVIDIA GPU** with CUDA 12.x (validated on T4 16 GB)
- **Driver** ≥ 550 + `nvidia-container-toolkit` for Docker
- **RAM** 16 GB minimum, 28 GB recommended
- **Disk** ~15 GB for the Docker image

---

## Quick Start

### Docker (recommended)

```bash
docker compose build
docker compose up -d
docker compose logs -f
```

### Local

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r ../requirements.txt
python main.py
```

The server starts on **port 8131**. The browser client is served at `http://localhost:8131`. Note: browsers require HTTPS for microphone access, so local dev uses self-signed certs from `certs/`.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness check — returns model status and available voices |
| `/api/voices` | GET | Lists available reference voice files |
| `/api/voices/upload` | POST | Upload a new reference voice (multipart `file` field) |
| `/ws` | WebSocket | The voice conversion stream |

### WebSocket Flow

1. **Connect** to `/ws`
2. **Send config** (text): `{"voice": "vineesha.wav"}`
3. **Receive ready** (text): `{"status": "ready", "output_sr": 24000}`
4. **Stream audio** (binary): 16 kHz mono int16 PCM, little-endian
5. **Receive converted audio** (binary): 24 kHz mono int16 PCM + `segment_done` events

---

## Configuration

All settings are in `config.py`. The main tuning dials:

| Setting | Default | What it does |
|---|---|---|
| `VAD_THRESHOLD` | `0.45` | Speech probability cutoff. Lower = catches quiet speech but admits noise. |
| `VAD_MIN_SILENCE_MS` | `400` | Silence needed to end a phrase. **Biggest latency lever** — lower = snappier but may split mid-sentence. |
| `VAD_MIN_SPEECH_MS` | `500` | Minimum phrase length to convert. |
| `VOICE_REF_DIR` | `"voices"` | Directory for reference voice files (relative to `backend/`). |
| `DEFAULT_VOICE_REF` | `None` | Set this to your primary voice to avoid a 15 s warm-up on first connection. |

---

## Reference Voice Guidelines

- Clean, single speaker, no background noise or music.
- **Only the first 10 seconds are used** — put the best audio first.
- Any sample rate works (resampled internally to 24 kHz).
- A clean 8-second sample beats a noisy 30-second one.

---

## Good to Know

- **One session at a time.** The target voice is shared global state — concurrent sessions with different voices will corrupt each other's output.
- **Use headphones when testing.** The mic will pick up playback and re-convert it, causing feedback loops.
- **Processing time is flat (~0.85 s)** regardless of how long the phrase is.
- **First connection after startup is slow (~15 s)** due to CUDA warm-up. Set `DEFAULT_VOICE_REF` to move this into startup time.
- **The last phrase is dropped** when the user presses stop (known issue).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No NVIDIA driver found | `docker run --gpus all`; install `nvidia-container-toolkit` |
| CUDA out of memory | One replica per GPU |
| Startup hangs then errors to huggingface.co | Rebuild with `prewarm_models.py` to bake weights into the image |
| Empty voices list | Make sure you're running from `backend/`; check volume mount |
| Audio sent, nothing comes back | Confirm 16 kHz mono int16 LE; check mic level |
| Chipmunk-pitched output | Use `output_sr` from the `ready` message, don't hardcode the sample rate |
| Noise bursts between phrases | Wear headphones (mic is picking up playback) |
| Last phrase never plays | Known issue — socket closes before server finishes converting |