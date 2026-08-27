"""
prewarm_models.py — bake model weights into the container image.

WHY THIS EXISTS
---------------
This service downloads TWO sets of weights from the public internet at runtime,
which is unacceptable in a production Azure deployment:

  1. Chatterbox VC  — tts_engine.py calls ChatterboxVC.from_pretrained(), which
     hits huggingface.co for `s3gen.safetensors` and `conds.pt`.
     Happens once, on FastAPI startup.

  2. Silero VAD     — vad_engine.py calls torch.hub.load("snakers4/silero-vad"),
     which hits github.com for a source archive on the FIRST call. A new
     VADEngine is constructed per Pipeline (i.e. per WebSocket session), so
     this runs on every connection; later calls hit the hub cache rather than
     the network, but on a cold container with no egress the very first
     connection fails outright. See README § Known Issues, item K-08.

Running this script at Docker BUILD time populates the same caches the app
reads at runtime, so the running container needs no egress to HuggingFace or
GitHub, starts faster, and cannot be broken by an upstream outage.

USAGE
-----
    # in the Dockerfile, after `pip install -r requirements.txt`
    RUN python prewarm_models.py

Requires HF_HOME and TORCH_HOME to be set to the SAME paths at build time and
at run time, otherwise the app will re-download into a different cache.
"""

import os
import sys

HF_HOME = os.environ.get("HF_HOME", "/opt/models/hf")
TORCH_HOME = os.environ.get("TORCH_HOME", "/opt/models/torch")

os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(TORCH_HOME, exist_ok=True)


def prewarm_chatterbox() -> None:
    """Download the Chatterbox VC checkpoints into the HuggingFace cache."""
    from huggingface_hub import hf_hub_download

    repo_id = "ResembleAI/chatterbox"
    for filename in ("s3gen.safetensors", "conds.pt"):
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [chatterbox] {filename:<22} {size_mb:8.1f} MB  ->  {path}")


def prewarm_silero() -> None:
    """Download the Silero VAD hub repo + weights into the torch hub cache."""
    import torch

    torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    print(f"  [silero-vad] cached under {TORCH_HOME}/hub")


if __name__ == "__main__":
    print(f"Pre-warming models  (HF_HOME={HF_HOME}  TORCH_HOME={TORCH_HOME})")
    try:
        prewarm_chatterbox()
        prewarm_silero()
    except Exception as exc:  # fail the build loudly, not the deployment quietly
        print(f"PREWARM FAILED: {exc}", file=sys.stderr)
        raise
    print("Pre-warm complete.")
