"""
FastAPI server — Voice Conversion mode.
Mic audio → ChatterboxVC → cloned audio. No STT, no text.
"""

import os
import json
import asyncio
import logging
import tempfile
import numpy as np
from pathlib import Path

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import HOST, PORT, VOICE_REF_DIR
from tts_engine import TTSEngine
from pipeline import Pipeline

# ─── Reference-voice upload rules ────────────────────────────
# Anything librosa/soundfile can decode is accepted, then re-encoded to wav.
# m4a/aac/webm additionally need ffmpeg installed on the host.
ALLOWED_UPLOAD_EXT = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm",
}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024     # 25 MB — generous for a 10-30s clip
REF_SR = 24000                          # ChatterboxVC works at 24 kHz
REF_WINDOW_S = 10                       # only the first 10s build the embedding
MIN_REF_SECONDS = 1.0
TRIM_TOP_DB = 35                        # leading/trailing silence threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

app = FastAPI(title="Voice Clone — Chatterbox VC")
os.makedirs(VOICE_REF_DIR, exist_ok=True)

vc_engine = None


@app.on_event("startup")
async def startup():
    global vc_engine
    log.info("═══ Loading Chatterbox VC ═══")
    vc_engine = TTSEngine()
    log.info(f"═══ Model loaded (sr={vc_engine.output_sr}) ═══")


def _safe_voice_path(filename: str) -> str:
    """Resolve a filename inside VOICE_REF_DIR, refusing path traversal."""
    name = Path(filename).name
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    root = os.path.abspath(VOICE_REF_DIR)
    dest = os.path.abspath(os.path.join(root, name))
    if os.path.commonpath([root, dest]) != root:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return dest


def _prepare_reference(src_path: str) -> np.ndarray:
    """
    Decode any supported audio file into a model-ready reference.

    Normalising here rather than at inference time matters because the speaker
    embedding is built ONCE from the first REF_WINDOW_S seconds. Leading
    silence inside that window is wasted, and a quiet recording yields a weak
    embedding, so both are corrected on the way in.
    """
    import librosa

    audio, _ = librosa.load(src_path, sr=REF_SR, mono=True)
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="File contains no audio")

    # Drop leading/trailing silence only — internal pauses are left intact so
    # the speaker's natural delivery is preserved.
    trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    if trimmed.size:
        audio = trimmed

    if len(audio) / REF_SR < MIN_REF_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(audio)/REF_SR:.1f}s of audio after trimming "
                   f"silence; need at least {MIN_REF_SECONDS:.0f}s",
        )

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = (audio * (0.95 / peak)).astype(np.float32)
    return audio


@app.get("/api/voices")
async def list_voices():
    if vc_engine is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    return {"voices": vc_engine.get_available_voices()}


@app.post("/api/voices/upload")
async def upload_voice(file: UploadFile = File(...)):
    """
    Accept a reference recording in any supported format and store it as wav.

    Everything is re-encoded to wav so the stored set is uniform and always
    decodable — the previous behaviour wrote the raw bytes under their original
    extension, which failed later, during set_target_voice, as an unhandled 500.
    """
    if vc_engine is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        shown = ext or "(no extension)"
        raise HTTPException(
            status_code=400,
            detail="Unsupported format " + shown + ". Supported: "
                   + ", ".join(sorted(ALLOWED_UPLOAD_EXT)),
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(content)/1e6:.1f} MB; limit is "
                   f"{MAX_UPLOAD_BYTES/1e6:.0f} MB",
        )

    # Decode from a temp file, so a bad upload never lands in voices/.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            audio = await asyncio.to_thread(_prepare_reference, tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            log.warning(f"Could not decode upload '{file.filename}': {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode this file as audio. "
                       f"If it is m4a/aac/webm, ffmpeg must be installed. ({exc})",
            )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    import soundfile as sf

    safe_name = Path(file.filename).stem + ".wav"
    dest = _safe_voice_path(safe_name)
    await asyncio.to_thread(sf.write, dest, audio, REF_SR)

    duration = len(audio) / REF_SR
    used = min(duration, REF_WINDOW_S)
    ready = vc_engine.load_voice(safe_name)
    if not ready:
        # Should not happen — the file was just written and already decoded.
        log.error(f"Voice '{safe_name}' written but failed to load")

    log.info(f"Voice uploaded: {safe_name} ({duration:.1f}s, using {used:.1f}s)")
    return {
        "filename": safe_name,
        "converted_from": ext,
        "duration_sec": round(duration, 2),
        "used_sec": round(used, 2),
        "truncated": duration > REF_WINDOW_S,
        "ready": ready,
        "voices": vc_engine.get_available_voices(),
    }


@app.delete("/api/voices/{filename}")
async def delete_voice(filename: str):
    """Remove a reference voice from disk."""
    if vc_engine is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    dest = _safe_voice_path(filename)
    if not os.path.exists(dest):
        raise HTTPException(status_code=404, detail=f"No such voice: {filename}")

    os.remove(dest)

    # Drop the cache entry so a later session with the same name reloads rather
    # than silently reusing the deleted voice's embedding.
    if getattr(vc_engine, "_current_voice", None) == Path(filename).name:
        vc_engine._current_voice = None

    log.info(f"Voice deleted: {Path(filename).name}")
    return {
        "deleted": Path(filename).name,
        "voices": vc_engine.get_available_voices(),
    }


@app.get("/api/health")
async def health():
    return {
        "vc": vc_engine is not None,
        "vc_sr": vc_engine.output_sr if vc_engine else None,
        "voices": vc_engine.get_available_voices() if vc_engine else [],
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    try:
        config_raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        config = json.loads(config_raw)
    except Exception as e:
        await ws.send_text(json.dumps({"error": f"Expected config JSON: {e}"}))
        await ws.close()
        return

    voice = config.get("voice")
    if not voice:
        await ws.send_text(json.dumps({"error": "Missing 'voice' in config"}))
        await ws.close()
        return

    log.info(f"Session config: voice={voice}")

    pipe = Pipeline(tts=vc_engine, voice=voice)

    await ws.send_text(json.dumps({
        "status": "ready",
        "output_sr": vc_engine.output_sr,
    }))

    try:
        while True:
            data = await ws.receive_bytes()

            async for event_type, event_data in pipe.process_audio(data):
                if event_type == "audio":
                    pcm = (event_data * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_bytes(pcm.tobytes())

                elif event_type == "info":
                    await ws.send_text(json.dumps({
                        "type": "info",
                        "message": event_data,
                    }))

                elif event_type == "done":
                    await ws.send_text(json.dumps({"type": "segment_done"}))

                elif event_type == "error":
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": event_data,
                    }))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception:
        log.exception("WebSocket error")
    finally:
        async for event_type, event_data in pipe.flush():
            try:
                if event_type == "audio":
                    pcm = (event_data * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_bytes(pcm.tobytes())
            except Exception:
                break
        log.info(f"Session stats: {pipe.get_stats()}")


# ─── Serve frontend ─────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", host=HOST, port=PORT, reload=False, log_level="info",
        ssl_keyfile="../certs/key.pem", ssl_certfile="../certs/cert.pem",
    )