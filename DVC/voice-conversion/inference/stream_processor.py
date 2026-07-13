"""
Stream Processor with overlap-crossfade for smooth RVC streaming.
Includes lightweight noise suppression (high-pass filter) and output gating.
"""

import numpy as np


class StreamProcessor:
    def __init__(self, voice_converter):
        self.converter = voice_converter
        self.input_buffer = np.array([], dtype=np.float32)
        self.is_active = False
        self.pitch_shift = 0.0
        self.input_sr = 48000

        # Process 1.5s chunks with 0.5s overlap for smooth transitions
        self.chunk_sec = 1.5
        self.overlap_sec = 0.5
        self.chunk_samples = None
        self.overlap_samples = None
        self.prev_tail = None

        # VAD
        self.vad_threshold = 0.003
        self.speech_active = False
        self.silence_count = 0
        self.silence_hangover = 0

        # High-pass filter for noise suppression
        self._hp_sos = None

        # Timing for health endpoint
        self.last_inference_ms = 0

    def start(self, voice_path, pitch_shift=0, input_sr=48000):
        self.converter.load_voice(voice_path)
        self.pitch_shift = pitch_shift
        self.input_sr = input_sr
        self.input_buffer = np.array([], dtype=np.float32)
        self.chunk_samples = int(self.chunk_sec * input_sr)
        self.overlap_samples = int(self.overlap_sec * input_sr)
        self.stride_samples = self.chunk_samples - self.overlap_samples
        self.prev_tail = None
        self.speech_active = False
        self.silence_count = 0
        self.is_active = True

        # Init high-pass filter (80Hz, removes hum/rumble)
        try:
            from scipy.signal import butter
            self._hp_sos = butter(4, 80, btype='high', fs=input_sr, output='sos')
            print("[StreamProcessor] High-pass noise filter enabled")
        except ImportError:
            self._hp_sos = None

        self.fade_in = np.linspace(0, 1, self.overlap_samples, dtype=np.float32)
        self.fade_out = np.linspace(1, 0, self.overlap_samples, dtype=np.float32)

        print(f"[StreamProcessor] Started | pitch={pitch_shift} | "
              f"chunk={self.chunk_samples} overlap={self.overlap_samples} stride={self.stride_samples}")

    def stop(self):
        self.is_active = False
        self.input_buffer = np.array([], dtype=np.float32)
        self.prev_tail = None
        print("[StreamProcessor] Stopped")

    def _gate_output(self, audio):
        """
        Frame-level gating: silence any 50ms frame with RMS below threshold.
        This removes RVC artifacts from near-silent regions while keeping speech intact.
        Uses smooth transitions to avoid clicks.
        """
        gate_threshold = 0.02
        frame_ms = 50
        frame_samples = int(frame_ms * self.input_sr / 1000)
        fade_samples = int(5 * self.input_sr / 1000)  # 5ms fade

        result = audio.copy()
        i = 0
        while i < len(result) - frame_samples:
            frame = result[i:i + frame_samples]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < gate_threshold:
                # Fade out over 5ms, silence the frame, fade in over 5ms
                if i >= fade_samples:
                    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
                    result[i:i + fade_samples] *= fade_out
                result[i + fade_samples:i + frame_samples - fade_samples] = 0
                if i + frame_samples + fade_samples <= len(result):
                    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
                    result[i + frame_samples - fade_samples:i + frame_samples] *= fade_in
            i += frame_samples

        return result

    def process(self, audio_bytes):
        if not self.is_active:
            return None

        incoming = np.frombuffer(audio_bytes, dtype=np.float32)
        self.input_buffer = np.concatenate([self.input_buffer, incoming])

        if len(self.input_buffer) < self.chunk_samples:
            return None

        chunk = self.input_buffer[:self.chunk_samples].copy()
        self.input_buffer = self.input_buffer[self.stride_samples:]

        # VAD
        rms = np.sqrt(np.mean(chunk ** 2))
        is_speech_now = rms > self.vad_threshold

        if is_speech_now:
            self.speech_active = True
            self.silence_count = 0
        else:
            self.silence_count += 1
            if self.silence_count > self.silence_hangover:
                self.speech_active = False
                self.prev_tail = None
                return None

        if not self.speech_active:
            return None

        try:
            # High-pass noise filter (<1ms overhead)
            if self._hp_sos is not None:
                from scipy.signal import sosfilt
                chunk = sosfilt(self._hp_sos, chunk).astype(np.float32)

            import time as _t; _t0 = _t.perf_counter()
            converted = self.converter.convert_streaming(
                chunk, sr=self.input_sr, pitch_shift=self.pitch_shift,
            )
            self.last_inference_ms = (_t.perf_counter() - _t0) * 1000
            if converted is None or len(converted) == 0:
                return None

            # Apply crossfade with previous chunk's tail
            if self.prev_tail is not None:
                overlap_len = min(len(self.prev_tail), self.overlap_samples, len(converted))
                if overlap_len > 0:
                    converted[:overlap_len] = (
                        converted[:overlap_len] * self.fade_in[:overlap_len] +
                        self.prev_tail[:overlap_len] * self.fade_out[:overlap_len]
                    )

            # Save tail for next crossfade
            if len(converted) > self.overlap_samples:
                self.prev_tail = converted[-self.overlap_samples:].copy()
                output = converted[:len(converted) - self.overlap_samples]
            else:
                self.prev_tail = None
                output = converted

            # Match output to stride length
            target_len = self.stride_samples
            if len(output) > target_len:
                output = output[:target_len]
            elif len(output) < target_len:
                output = np.pad(output, (0, target_len - len(output)))

            # Only silence entire chunks that are clearly not speech
            chunk_rms = np.sqrt(np.mean(output ** 2))
            if chunk_rms < 0.01:
                return None

            return output.astype(np.float32).tobytes()

        except Exception as e:
            print(f"[StreamProcessor] Conversion error: {e}")
            return None

    def get_latency_ms(self):
        return self.chunk_sec * 1000
