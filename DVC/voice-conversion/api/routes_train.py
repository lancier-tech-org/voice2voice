"""
Training API routes — RVC backend.

POST /api/train/upload — Upload audio + auto-start training
GET  /api/train/status/{voice_name} — Check training progress
"""

import asyncio
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

import config

router = APIRouter()


@router.post("/upload")
async def upload_and_train(
    request: Request,
    file: UploadFile = File(...),
    voice_name: str = Form(...),
):
    state = request.app.state.app_state

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_AUDIO_FORMATS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported format: {ext}")

    voice_name = voice_name.strip().replace(" ", "_").lower()
    if not voice_name:
        raise HTTPException(status_code=400, detail="Voice name required.")

    # Check if voice already exists
    voice_dir = config.VOICES_DIR / voice_name
    if voice_dir.exists() and list(voice_dir.glob("*.pth")):
        raise HTTPException(status_code=409,
            detail=f"Voice '{voice_name}' already exists. Delete it first.")

    if state.trainer.is_training:
        raise HTTPException(status_code=409,
            detail="Training already running. Wait for it to complete.")

    # Save uploaded file
    upload_dir = config.UPLOADS_DIR / voice_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"raw{ext}"

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413,
            detail=f"File too large: {size_mb:.1f}MB")

    with open(str(upload_path), "wb") as f:
        f.write(content)

    state.training_tasks[voice_name] = {
        "status": "preprocessing",
        "progress": 0.0,
        "message": "Starting training...",
    }

    asyncio.create_task(
        _train_background(state, voice_name, str(upload_path))
    )

    return JSONResponse({
        "status": "started",
        "voice_name": voice_name,
        "message": "Upload received. Training started.",
        "file_size_mb": round(size_mb, 1),
    })


async def _train_background(state, voice_name, audio_path):
    try:
        def progress_callback(progress, message):
            state.training_tasks[voice_name] = {
                "status": "training",
                "progress": progress,
                "message": message,
            }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: state.trainer.train(
                voice_name=voice_name,
                audio_path=audio_path,
                sr=48000,
                epochs=400,
                batch_size=8,
                progress_callback=progress_callback,
            )
        )

        if result["status"] == "success":
            state.training_tasks[voice_name] = {
                "status": "complete",
                "progress": 1.0,
                "message": "Training complete!",
                "metadata": result.get("metadata"),
            }
            # Cleanup uploads
            try:
                upload_dir = config.UPLOADS_DIR / voice_name
                if upload_dir.exists():
                    shutil.rmtree(upload_dir, ignore_errors=True)
            except:
                pass
        else:
            state.training_tasks[voice_name] = {
                "status": "error",
                "progress": 0,
                "message": result.get("message", "Training failed"),
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        state.training_tasks[voice_name] = {
            "status": "error",
            "progress": 0,
            "message": f"Error: {str(e)}",
        }


@router.get("/status/{voice_name}")
async def training_status(voice_name: str, request: Request):
    state = request.app.state.app_state

    if voice_name not in state.training_tasks:
        voice_dir = config.VOICES_DIR / voice_name
        if voice_dir.exists() and list(voice_dir.glob("*.pth")):
            return {"status": "complete", "progress": 1.0,
                    "message": "Voice model is ready."}
        raise HTTPException(status_code=404, detail="No training task found.")

    return state.training_tasks[voice_name]