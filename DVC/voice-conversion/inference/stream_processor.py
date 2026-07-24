"""
Stream Processor with overlap-crossfade for smooth RVC streaming.
Includes frame-level output gating to eliminate RVC artifacts.
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

        # Output gate keyed to the CLEAN INPUT: silence output wherever the input
        # had no speech, to kill the noise RVC hallucinates at pause boundaries.
        # Threshold sits BELOW the quietest real speech (measured: noise floor
        # ~0.0005, quietest speech frames ~0.006) so speech is never cut — the
        # mistake in an earlier attempt was a threshold (0.006) above median speech.
        # 0.004 chosen by measuring this against a real recording: pause-noise max
        # 0.12 -> 0.027 (spikes killed) with only ~1.3% speech change. Higher (0.008+)
        # starts cutting quiet speech; lower (0.002) barely helps.
        self.gate_enabled = True
        self.gate_threshold = 0.0040
        self._gate_prev = 1.0

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
        self._gate_prev = 1.0
        self.is_active = True

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

    def _gate_to_input(self, output, ref):
        """
        Silence `output` wherever the clean input `ref` has no speech.

        Keyed to the INPUT (which is clean) rather than the converted output, at a
        threshold below the quietest speech, so real speech is never removed — it
        only removes RVC's hallucinated noise in pauses. Smooth 20ms-frame ramp,
        seeded from the previous chunk so there are no clicks at chunk seams.
        """
        n = len(output)
        if n == 0:
            return output
        ref = ref[:n]
        if len(ref) < n:
            ref = np.pad(ref, (0, n - len(ref)))

        frame = max(1, int(0.02 * self.input_sr))
        nf = max(1, n // frame)
        env = np.sqrt(np.mean(ref[:nf * frame].reshape(nf, frame) ** 2, axis=1))
        gate = (env > self.gate_threshold).astype(np.float32)

        xf = (np.arange(nf) + 0.5) * frame
        smooth = np.interp(
            np.arange(n),
            np.concatenate(([0.0], xf, [float(n - 1)])),
            np.concatenate(([self._gate_prev], gate, [gate[-1]])),
        ).astype(np.float32)
        self._gate_prev = float(gate[-1])
        return (output * smooth).astype(np.float32)

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
                # Emit silence for this stride instead of dropping it. Dropping
                # consumed 1 stride of input but sent 0 output, so the browser
                # played the rest back-to-back — 55.6s of speech came out as 32s
                # and sounded rushed/fast. Emitting silence keeps input and output
                # the same length; we also skip RVC on pauses, avoiding the noise
                # it hallucinates there.
                return np.zeros(self.stride_samples, dtype=np.float32).tobytes()

        if not self.speech_active:
            return np.zeros(self.stride_samples, dtype=np.float32).tobytes()

        try:
            converted = self.converter.convert_streaming(
                chunk, sr=self.input_sr, pitch_shift=self.pitch_shift,
            )
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

            # Silence output where the INPUT had no speech — removes RVC's
            # hallucinated noise at pause boundaries without touching speech.
            if self.gate_enabled:
                output = self._gate_to_input(output, chunk[:len(output)])

            # If the converted chunk is essentially silent, emit silence (keeps the
            # timeline aligned) rather than dropping it (which compressed time).
            chunk_rms = np.sqrt(np.mean(output ** 2))
            if chunk_rms < 0.01:
                return np.zeros(self.stride_samples, dtype=np.float32).tobytes()

            return output.astype(np.float32).tobytes()

        except Exception as e:
            print(f"[StreamProcessor] Conversion error: {e}")
            # Keep the timeline even on error — emit silence for this stride.
            return np.zeros(self.stride_samples, dtype=np.float32).tobytes()

    def get_latency_ms(self):
        return self.chunk_sec * 1000