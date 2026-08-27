"""
Chatterbox Voice Conversion — audio in, audio out.
No STT, no text, no hallucinations. Preserves timing, rhythm, emotion.
"""

import os
import time
import numpy as np
import torch
import logging
from pathlib import Path

from config import (
    XTTS_DEVICE,
    VOICE_REF_DIR,
    DEFAULT_VOICE_REF,
    SAMPLE_RATE_IN,
)

log = logging.getLogger(__name__)


def _patch_perth():
    try:
        import perth

        class _SafeNoOpWatermarker:
            def __init__(self, *a, **kw):
                pass
            def apply(self, audio, *a, **kw):
                return audio
            def apply_watermark(self, audio, *a, **kw):
                return audio
            def __call__(self, audio, *a, **kw):
                return audio

        perth.PerthImplicitWatermarker = _SafeNoOpWatermarker
        log.info("Using safe no-op watermarker stub")
    except Exception:
        pass

_patch_perth()


class TTSEngine:
    def __init__(self):
        from chatterbox.vc import ChatterboxVC

        log.info(f"Loading Chatterbox VC on {XTTS_DEVICE}...")
        self.model = ChatterboxVC.from_pretrained(device=XTTS_DEVICE)
        self.output_sr = self.model.sr
        self.input_sr = SAMPLE_RATE_IN
        log.info(f"Chatterbox VC loaded (sr={self.output_sr})")

        self._current_voice = None

        if DEFAULT_VOICE_REF:
            self.load_voice(DEFAULT_VOICE_REF)

    def load_voice(self, voice_filename: str) -> bool:
        voice_path = os.path.join(VOICE_REF_DIR, voice_filename)
        if not os.path.exists(voice_path):
            log.error(f"Voice reference not found: {voice_path}")
            return False
        self.model.set_target_voice(voice_path)
        self._current_voice = voice_filename
        log.info(f"Voice '{voice_filename}' loaded and cached")
        return True

    def get_available_voices(self) -> list[str]:
        voice_dir = Path(VOICE_REF_DIR)
        if not voice_dir.exists():
            return []
        return sorted(
            f.name
            for f in voice_dir.iterdir()
            if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        )

    def convert_speech(self, audio: np.ndarray, voice_filename: str) -> np.ndarray | None:
        t0 = time.monotonic()

        if self._current_voice != voice_filename:
            self.load_voice(voice_filename)

        try:
            with torch.inference_mode():
                audio_16 = torch.from_numpy(audio).float().to(self.model.device)[None,]
                s3_tokens, _ = self.model.s3gen.tokenizer(audio_16)
                wav, _ = self.model.s3gen.inference(
                    speech_tokens=s3_tokens,
                    ref_dict=self.model.ref_dict,
                )
                result = wav.squeeze(0).detach().cpu().numpy()

            elapsed = time.monotonic() - t0
            duration_in = len(audio) / self.input_sr
            duration_out = len(result) / self.output_sr
            log.info(
                f"VC done: {duration_in:.2f}s in → {duration_out:.2f}s out "
                f"({elapsed:.2f}s processing, {elapsed/duration_in:.2f}x realtime)"
            )
            return result

        except Exception as e:
            log.exception("Voice conversion failed")
            return None