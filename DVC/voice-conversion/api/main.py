"""
FastAPI application — RVC Backend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config
from inference.rvc_converter import RVCConverter
from inference.stream_processor import StreamProcessor
from training.rvc_trainer import RVCTrainer

from api.routes_train import router as train_router
from api.routes_convert import router as convert_router
from api.routes_voices import router as voices_router

app = FastAPI(
    title="Voice Conversion System (RVC)",
    description="Real-time voice conversion powered by RVC",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    converter: RVCConverter = None
    stream_processor: StreamProcessor = None
    trainer: RVCTrainer = None
    training_tasks: dict = {}

state = AppState()


@app.on_event("startup")
async def startup():
    print("=" * 60)
    print("  Voice Conversion System (RVC) — Starting Up")
    print("=" * 60)

    state.converter = RVCConverter(device=config.DEVICE)
    state.stream_processor = StreamProcessor(state.converter)
    state.trainer = RVCTrainer()

    if config.DEVICE == "cuda":
        print("[Startup] Pre-loading RVC engine...")
        try:
            state.converter.load_models()
            print("[Startup] RVC engine ready.")
        except Exception as e:
            print(f"[Startup] Warning: {e}")
            print("[Startup] RVC will initialize on first use.")

    print(f"[Startup] Server ready at http://localhost:{config.API_PORT}")
    print("=" * 60)


app.state.app_state = state

app.include_router(train_router, prefix="/api/train", tags=["Training"])
app.include_router(convert_router, prefix="/api/convert", tags=["Conversion"])
app.include_router(voices_router, prefix="/api/voices", tags=["Voices"])


@app.get("/")
async def serve_frontend():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(str(frontend_path), media_type="text/html")


@app.get("/test_audio")
async def serve_test_audio():
    return FileResponse("/tmp/test_ws_audio.html", media_type="text/html")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "backend": "RVC",
        "device": config.DEVICE,
        "models_loaded": state.converter._models_loaded if state.converter else False,
        "current_voice": state.converter.current_voice_name if state.converter else None,
        "training_active": state.trainer.is_training if state.trainer else False,
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        ws_max_size=16 * 1024 * 1024,
    )