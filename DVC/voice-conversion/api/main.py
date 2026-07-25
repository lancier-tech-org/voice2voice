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
        elif "LIVEFIX" in stem:
            label = (f"<b>LIVE with real context, {stem.split('_')[-1]} semitones</b> "
                     f"&nbsp;<i>(the new fix — compare against the two below)</i>")
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
        "<html><head><title>Pitch-shift ladder</title>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<style>body{font-family:system-ui;margin:24px;max-width:720px;line-height:1.5}"
        "h3{margin:22px 0 6px;font-size:1rem}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;"
        "padding:4px 10px;text-align:left;font-size:.9rem}</style></head><body>"
        "<h2>How far down can the pitch go before it falls apart?</h2>"
        "<p>Same 16s of your speech, same voice model. The only variable is how many "
        "semitones the pitch is shifted down. <b>-13</b> is deployed; <b>0</b> is no "
        "shift.</p>"
        "<p><b>Two kinds of clip:</b> <i>offline whole-file</i> = converted in one "
        "call, no chunking — the best this model can do. <i>LIVE streaming path</i> = "
        "the actual real-time pipeline at the same pitch. Compare the two at the same "
        "number to hear what real-time costs you.</p>"
        "<table><tr><th>shift</th><th>offline</th><th>live</th></tr>"
        "<tr><td>-7</td><td>0.549</td><td>0.593</td></tr>"
        "<tr><td>-9</td><td>0.532</td><td>0.587</td></tr>"
        "<tr><td>-13</td><td>0.478</td><td>0.490</td></tr></table>"
        "<p>(periodicity = how much harmonic structure survives; your own voice is "
        "0.68. Real-time is not costing it — the pitch shift is.)</p>"
        "<hr><h3 style='margin-top:20px'>The live fix, at -13</h3>"
        "<p>Pitch stays at -13 (correct for f&rarr;m). Three clips to compare at that "
        "same pitch: <b>LIVE with real context</b> (the fix), <b>LIVE streaming path</b> "
        "(what you have now), and <b>offline whole-file</b> (the ceiling). Measured: "
        "speech lost to dropouts fell from 2.07% to 0.17%, pacing unchanged, but "
        "smoothness metrics barely moved — whether the WORDS come out better is the "
        "part no metric can see, so that is what to listen for.</p>"
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