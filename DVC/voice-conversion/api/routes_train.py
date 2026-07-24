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

    # Remember the source audio so a paused run can be resumed later.
    if not hasattr(state, "training_audio"):
        state.training_audio = {}
    state.training_audio[voice_name] = str(upload_path)

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


def _purge_voice_artifacts(voice_name: str):
    """Delete everything a cancelled/failed training left behind, so the name is
    free to reuse: the output voice dir, the RVC training logs (checkpoints), the
    saved weight checkpoints, and the uploaded audio."""
    rvc_webui = Path(config.BASE_DIR).parent / "rvc-webui"
    targets = [
        config.VOICES_DIR / voice_name,
        config.UPLOADS_DIR / voice_name,
        rvc_webui / "logs" / voice_name,
    ]
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    weights = rvc_webui / "assets" / "weights"
    if weights.is_dir():
        for f in weights.glob(f"{voice_name}*.pth"):
            try:
                f.unlink()
            except OSError:
                pass


async def _train_background(state, voice_name, audio_path, resume=False):
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
                # Use the configured value instead of a hardcoded 400. Measured on
                # anil_v2: quality peaks around epoch 200 (HNR 11.1) and does not
                # improve by 400 (10.9, within run-to-run noise), so 400 just
                # doubled training time — 140 min instead of ~70.
                epochs=config.TRAINING_EPOCHS,
                batch_size=config.TRAINING_BATCH_SIZE,
                progress_callback=progress_callback,
                resume=resume,
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
            if hasattr(state, "training_audio"):
                state.training_audio.pop(voice_name, None)

        elif result["status"] == "stopped":
            mode = result.get("mode", "cancel")
            if mode == "pause":
                # Keep everything (checkpoints + audio) so it can be resumed.
                state.training_tasks[voice_name] = {
                    "status": "paused",
                    "progress": result.get("progress", 0),
                    "message": "Training paused. Resume to continue from the last checkpoint.",
                }
            else:
                # Cancel: wipe it so the name is free.
                _purge_voice_artifacts(voice_name)
                state.training_tasks.pop(voice_name, None)
                if hasattr(state, "training_audio"):
                    state.training_audio.pop(voice_name, None)

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


@router.get("/active")
async def active_training(request: Request):
    """Return the training that is currently in progress or paused, if any, so the
    UI can re-attach its progress bar and controls after a page reload/restart."""
    state = request.app.state.app_state
    live = ("preprocessing", "training", "resuming", "paused")
    for name, t in state.training_tasks.items():
        if t.get("status") in live:
            return {"voice_name": name, **t}
    return {"voice_name": None}


@router.post("/stop")
async def stop_training(request: Request,
                        voice_name: str = Form(...),
                        mode: str = Form("cancel")):
    """
    Stop the running training.

      mode=cancel : discard it (mistaken/unwanted upload). Artifacts are deleted
                    and the name is freed.
      mode=pause  : keep the checkpoints so it can be resumed later.

    The background task observes the stop and updates status; we kill the process
    group here so RVC's GPU workers die too.
    """
    state = request.app.state.app_state
    mode = mode if mode in ("cancel", "pause") else "cancel"

    if not state.trainer.is_training:
        raise HTTPException(status_code=409, detail="No training is running.")

    active_voice = state.trainer.current_voice
    if voice_name and active_voice and voice_name != active_voice:
        raise HTTPException(status_code=409,
            detail=f"Training running is '{active_voice}', not '{voice_name}'.")

    state.trainer.stop_training(mode=mode)
    return {"status": "stopping", "mode": mode, "voice_name": active_voice or voice_name}


@router.post("/resume")
async def resume_training(request: Request, voice_name: str = Form(...)):
    """Resume a paused training from its last saved checkpoint."""
    state = request.app.state.app_state
    voice_name = voice_name.strip().replace(" ", "_").lower()

    if state.trainer.is_training:
        raise HTTPException(status_code=409,
            detail="Another training is already running.")

    # Need the extracted features/checkpoints from the earlier run to resume.
    rvc_webui = Path(config.BASE_DIR).parent / "rvc-webui"
    exp_dir = rvc_webui / "logs" / voice_name
    if not exp_dir.is_dir() or not (exp_dir / "filelist.txt").exists():
        raise HTTPException(status_code=404,
            detail=f"Nothing to resume for '{voice_name}'.")

    audio_path = getattr(state, "training_audio", {}).get(voice_name, "")

    state.training_tasks[voice_name] = {
        "status": "training",
        "progress": 0.44,
        "message": "Resuming from last checkpoint...",
    }
    asyncio.create_task(
        _train_background(state, voice_name, audio_path, resume=True)
    )
    return {"status": "resuming", "voice_name": voice_name}