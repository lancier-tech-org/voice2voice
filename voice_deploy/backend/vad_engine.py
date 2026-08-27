"""
Silero VAD — detects when the speaker finishes a sentence/phrase.
Accumulates audio chunks, emits complete speech segments.
"""

import numpy as np
import torch
import logging

from config import (
    SAMPLE_RATE_IN,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
    VAD_WINDOW_SIZE,
)

log = logging.getLogger(__name__)


class VADEngine:
    def __init__(self):
        log.info("Loading Silero VAD...")
        self.model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self.model.eval()
        log.info("Silero VAD loaded")

        self._sr = SAMPLE_RATE_IN
        self._threshold = VAD_THRESHOLD
        self._min_speech_samples = int(VAD_MIN_SPEECH_MS * self._sr / 1000)
        self._min_silence_samples = int(VAD_MIN_SILENCE_MS * self._sr / 1000)
        self._window = VAD_WINDOW_SIZE  # Silero requires exactly 512 at 16kHz

        self.reset()

    def reset(self):
        """Clear all state for a new session."""
        self.model.reset_states()
        self._buffer = np.array([], dtype=np.float32)
        self._speech_buf = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_samples = 0

    def feed(self, pcm_i16: bytes) -> list[np.ndarray]:
        """
        Feed raw 16-bit PCM bytes from the browser.
        Returns a list of completed speech segments (float32 numpy arrays).
        Each segment is one sentence/phrase worth of audio, ready for STT.
        """
        # Convert int16 bytes → float32
        audio = np.frombuffer(pcm_i16, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, audio])

        completed_segments = []

        # Process in windows of exactly 512 samples
        while len(self._buffer) >= self._window:
            chunk = self._buffer[: self._window]
            self._buffer = self._buffer[self._window :]

            # Run VAD
            tensor = torch.from_numpy(chunk).float()
            prob = self.model(tensor, self._sr).item()

            if prob >= self._threshold:
                # Speech detected
                if not self._is_speaking:
                    self._is_speaking = True
                    self._silence_samples = 0
                    log.debug("Speech start")
                self._speech_buf = np.concatenate([self._speech_buf, chunk])
                self._silence_samples = 0

            else:
                if self._is_speaking:
                    # We're in speech but this window is silent
                    self._silence_samples += self._window
                    self._speech_buf = np.concatenate([self._speech_buf, chunk])

                    if self._silence_samples >= self._min_silence_samples:
                        # Enough silence — segment complete
                        if len(self._speech_buf) >= self._min_speech_samples:
                            log.info(
                                f"Speech segment: {len(self._speech_buf)/self._sr:.2f}s"
                            )
                            completed_segments.append(self._speech_buf.copy())
                        else:
                            log.debug("Segment too short, discarding")

                        self._speech_buf = np.array([], dtype=np.float32)
                        self._is_speaking = False
                        self._silence_samples = 0
                        self.model.reset_states()

        return completed_segments

    def flush(self) -> np.ndarray | None:
        """Flush any remaining speech (e.g. on disconnect)."""
        if len(self._speech_buf) >= self._min_speech_samples:
            seg = self._speech_buf.copy()
            self.reset()
            return seg
        self.reset()
        return None
