"""
FastAPI application — RVC Backend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

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
            # Load and warm one voice now, so the cost is paid here rather than by
            # the user. Cold warm-up measured ~12s (CUDA autotuning, rmvpe load) and
            # ~1s once the engine has run; without this it landed on session start.
            voices = state.converter.list_voices()
            if voices:
                state.converter.load_voice(voices[0]["path"])
                print(f"[Startup] Warmed on '{voices[0]['name']}'.")
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


#
# A/B listening page. The assistant cannot hear audio, so every audio change has
# to be judged by ear against real recordings; this just makes the comparison
# files playable in a browser. Serves ONLY the explicitly named comparison files
# from uploads/debug — not the directory — so the rest of the captured recordings
# stay unreachable. Delete this block once the comparison is done.
#
def _ab_clips():
    """Only files written by the sweep (SW_*.wav). The rest of uploads/debug is
    the user's raw captured recordings and must stay unreachable."""
    d = config.UPLOADS_DIR / "debug"
    return sorted(p.name for p in d.glob("SW_*.wav")) if d.is_dir() else []


@app.get("/ab")
async def ab_page():
    rows = []
    for n in _ab_clips():
        stem = n[3:-4]
        if "input" in stem:
            label = "YOUR VOICE — unconverted, for reference"
        elif stem.startswith("40_"):
            label = "1. YOUR VOICE &mdash; the mic input from your 12:30 session"
        elif stem.startswith("41_"):
            label = ("<b>2. WHAT YOU ACTUALLY HEARD</b> &nbsp;<i>(your live output, saved "
                     "server-side during that session)</i>")
        elif stem.startswith("42_"):
            label = ("3. SAME AUDIO, converted OFFLINE in one call &nbsp;<i>(no chunking "
                     "&mdash; the ceiling)</i>")
        elif stem.startswith("43_"):
            label = ("4. SAME AUDIO through the LIVE path on current code")
        elif "LIVE" in stem:
            label = (f"LIVE streaming path, {stem.split('_')[-1]} semitones "
                     f"&nbsp;<i>(real-time, old bare-chunk path)</i>")
        else:
            label = (f"offline whole-file, {stem.split('_')[-1]} semitones"
                     + (" &nbsp;<b>&larr; currently deployed</b>"
                        if stem.endswith("-13") and "index" not in stem else ""))
        rows.append(
            f'<h3>{label}</h3><audio controls preload="none" src="/ab/{n}" '
            f'style="width:100%;max-width:640px"></audio>'
        )
    return HTMLResponse(
        "<html><head><title>Where is the disturbance?</title>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<style>body{font-family:system-ui;margin:24px;max-width:720px;line-height:1.5}"
        "h3{margin:22px 0 6px;font-size:1rem}</style></head><body>"
        "<h2>Where is the disturbance?</h2>"
        "<p>All four are the SAME 16 seconds of your speech, voice <b>vamsi</b> at "
        "<b>-12</b> &mdash; the voice and pitch your live session actually used.</p>"
        "<p><b>2</b> is your real live output, recorded server-side while you were "
        "speaking, so it is literally what came out of your speakers. <b>3</b> is that "
        "same audio converted in one call with no chunking at all. <b>4</b> is the live "
        "path on the code running now.</p>"
        "<p><b>What to tell me:</b></p>"
        "<ul><li>Is the disturbance in <b>2</b>? Then it is in the audio, and comparing "
        "2 with 4 says whether the current code already fixed it.</li>"
        "<li>Is <b>3</b> clean while <b>2</b> and <b>4</b> are not? Then it is the "
        "streaming, and I keep working on the live path.</li>"
        "<li>Are <b>3</b> and <b>4</b> both disturbed? Then it is the model or the -12 "
        "shift, and no amount of streaming work will help.</li>"
        "<li>Are they ALL clean to you? Then the disturbance is added after the server "
        "&mdash; browser or network &mdash; and the Breaks/Sync numbers on the main page "
        "are what I need.</li></ul>"
        + "".join(rows) + "</body></html>"
    )


@app.get("/ab/{name}")
async def ab_file(name: str):
    if name not in _ab_clips():          # membership check = no path traversal
        raise HTTPException(status_code=404, detail="unknown clip")
    return FileResponse(str(config.UPLOADS_DIR / "debug" / name),
                        media_type="audio/wav", filename=name)


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