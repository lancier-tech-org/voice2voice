"""
Voice management API routes — RVC backend.
"""

import json
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request

import config

router = APIRouter()


@router.get("/")
async def list_voices(request: Request):
    state = request.app.state.app_state
    voices = []

    # Check subdirectories (RVC style: voices/name/name.pth)
    for voice_dir in sorted(config.VOICES_DIR.iterdir()):
        if voice_dir.is_dir():
            pth_files = list(voice_dir.glob("*.pth"))
            if pth_files:
                meta = {}
                meta_path = voice_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)

                voices.append({
                    "name": voice_dir.name,
                    "active": (state.converter.current_voice_name == voice_dir.name
                              if state.converter else False),
                    "has_index": bool(list(voice_dir.glob("*.index"))),
                    "epochs": meta.get("epochs"),
                    "training_time_min": meta.get("training_time_min"),
                })

    # Also check for loose .pth files (from old training)
    for pth in sorted(config.VOICES_DIR.glob("*.pth")):
        name = pth.stem
        if not any(v["name"] == name for v in voices):
            voices.append({
                "name": name,
                "active": (state.converter.current_voice_name == name
                          if state.converter else False),
                "has_index": False,
                "epochs": None,
                "training_time_min": None,
            })

    return {"voices": voices, "count": len(voices)}


@router.get("/{name}")
async def get_voice(name: str):
    voice_dir = config.VOICES_DIR / name
    pth_files = list(voice_dir.glob("*.pth")) if voice_dir.is_dir() else []

    # Check loose file
    if not pth_files:
        loose = config.VOICES_DIR / f"{name}.pth"
        if loose.exists():
            pth_files = [loose]

    if not pth_files:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")

    meta = {}
    meta_path = voice_dir / "metadata.json" if voice_dir.is_dir() else None
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        "name": name,
        "file_size_mb": round(pth_files[0].stat().st_size / (1024*1024), 1),
        "has_index": bool(list(voice_dir.glob("*.index"))) if voice_dir.is_dir() else False,
        "metadata": meta,
    }


@router.delete("/{name}")
async def delete_voice(name: str, request: Request):
    state = request.app.state.app_state

    voice_dir = config.VOICES_DIR / name
    loose_pth = config.VOICES_DIR / f"{name}.pth"

    if voice_dir.is_dir():
        if (state.stream_processor and state.stream_processor.is_active
                and state.converter.current_voice_name == name):
            raise HTTPException(status_code=409,
                detail="Cannot delete voice while active.")
        shutil.rmtree(voice_dir)
    elif loose_pth.exists():
        loose_pth.unlink()
        meta = config.VOICES_DIR / f"{name}_meta.json"
        if meta.exists():
            meta.unlink()
    else:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")

    if state.converter and state.converter.current_voice_name == name:
        state.converter.current_voice_name = None
        state.converter._voice_loaded = False

    return {"status": "deleted", "voice": name}