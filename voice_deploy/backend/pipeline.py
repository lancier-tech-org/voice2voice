"""
Pipeline — Voice Conversion mode:
  Mic audio → VAD → ChatterboxVC → converted audio

No STT, no text, no hallucinations. Preserves timing, rhythm, emotion.
One Pipeline instance per WebSocket session.
"""

import time
import numpy as np
import logging
from dataclasses import dataclass

from vad_engine import VADEngine
from tts_engine import TTSEngine

log = logging.getLogger(__name__)


@dataclass
class SessionStats:
    segments_processed: int = 0
    total_audio_in: float = 0.0
    total_audio_out: float = 0.0
    total_latency: float = 0.0
    errors: int = 0


class Pipeline:
    """One per WebSocket connection. Holds its own VAD state."""

    def __init__(self, tts: TTSEngine, voice: str):
        self.tts = tts
        self.vad = VADEngine()
        self.voice = voice
        self.stats = SessionStats()
        self.output_sr = tts.output_sr

        self.tts.load_voice(voice)

    async def process_audio(self, pcm_bytes: bytes):
        """
        Feed raw PCM bytes from the browser.
        Yields (event_type, data) tuples:
          ("audio", np.ndarray)  — converted audio chunk
          ("info", str)          — status message
          ("done", None)         — segment complete
          ("error", str)
        """
        segments = self.vad.feed(pcm_bytes)

        for speech_audio in segments:
            t0 = time.monotonic()
            duration_in = len(speech_audio) / 16000
            self.stats.total_audio_in += duration_in

            yield ("info", f"Converting {duration_in:.1f}s of speech...")

            # ── Voice Conversion (no text involved) ──────
            try:
                converted = self.tts.convert_speech(
                    audio=speech_audio,
                    voice_filename=self.voice,
                )
            except Exception as e:
                log.exception("VC failed")
                self.stats.errors += 1
                yield ("error", f"Voice conversion error: {e}")
                continue

            if converted is None:
                self.stats.errors += 1
                yield ("error", "Voice conversion returned empty")
                continue

            latency = time.monotonic() - t0
            self.stats.total_latency += latency
            self.stats.total_audio_out += len(converted) / self.output_sr
            self.stats.segments_processed += 1

            log.info(f"Segment converted: {duration_in:.1f}s → latency {latency:.2f}s")

            yield ("audio", converted)
            yield ("done", None)

    async def flush(self):
        """Process any remaining speech on disconnect."""
        remaining = self.vad.flush()
        if remaining is not None:
            converted = self.tts.convert_speech(remaining, self.voice)
            if converted is not None:
                yield ("audio", converted)
                yield ("done", None)

    def get_stats(self) -> dict:
        s = self.stats
        avg_lat = (
            s.total_latency / s.segments_processed
            if s.segments_processed > 0
            else 0
        )
        return {
            "segments": s.segments_processed,
            "audio_in_sec": round(s.total_audio_in, 1),
            "audio_out_sec": round(s.total_audio_out, 1),
            "avg_latency_sec": round(avg_lat, 2),
            "errors": s.errors,
        }
