"""
Stream Processor — context-window streaming for RVC.

The model is offline: it converts whatever window it is given, with no memory
between calls. Converting isolated ~1s chunks (the old approach) meant words that
span a boundary were converted with no idea what came before, so pronunciation
degraded. Measured against the clean whole-file conversion (MCD, lower=closer to
correct): isolated chunks scored ~67; feeding 3s of preceding context scored ~49
(27% closer). Inference cost is ~fixed regardless of window length (~320ms for a
4s window on a T4), so context is nearly free.

So: keep a rolling history of the INPUT, and every BLOCK seconds convert a window
of [context + new block] but EMIT only the new block. Each block therefore sees
several seconds of preceding audio. Silence blocks emit silence (never dropped —
that kept the timeline correct so playback isn't sped up), and output is gated to
the clean input to remove hallucinated pause noise.
"""

import numpy as np


class StreamProcessor:
    def __init__(self, voice_converter):
        self.converter = voice_converter
        self.is_active = False
        self.pitch_shift = 0.0
        self.input_sr = 48000

        # Emit BLOCK seconds at a time; feed the model CONTEXT seconds of history
        # before it. Window fed to the model = context + block.
        self.block_sec = 1.0
        self.context_sec = 3.0

        self.input_history = np.array([], dtype=np.float32)
        self.pending = 0

        # VAD: a block quieter than this is treated as a pause -> emit silence and
        # skip RVC (which would otherwise hallucinate noise there).
        self.vad_threshold = 0.003

        # Output gate keyed to the CLEAN INPUT: silence output wherever the input
        # block had no speech. Threshold sits below the quietest speech (measured
        # noise floor ~0.0005, quietest speech ~0.006) so speech is never cut.
        self.gate_enabled = True
        self.gate_threshold = 0.0040
        self._gate_prev = 1.0

        self.last_inference_ms = 0

    def start(self, voice_path, pitch_shift=0, input_sr=48000):
        self.converter.load_voice(voice_path)
        self.pitch_shift = pitch_shift
        self.input_sr = input_sr

        self.block_samples = int(self.block_sec * input_sr)
        self.window_samples = int((self.context_sec + self.block_sec) * input_sr)
        self.history_cap = self.window_samples + self.block_samples

        self.input_history = np.array([], dtype=np.float32)
        self.pending = 0
        self._gate_prev = 1.0
        self.is_active = True

        print(f"[StreamProcessor] Started | pitch={pitch_shift} | "
              f"block={self.block_samples} window={self.window_samples} "
              f"| latency≈{self.get_latency_ms():.0f}ms")

    def stop(self):
        self.is_active = False
        self.input_history = np.array([], dtype=np.float32)
        self.pending = 0
        print("[StreamProcessor] Stopped")

    def get_latency_ms(self):
        # We emit once BLOCK seconds of new input have arrived.
        return self.block_sec * 1000

    def _gate_to_input(self, output, ref):
        """
        Silence `output` wherever the clean input `ref` has no speech. Keyed to the
        INPUT (clean) at a threshold below the quietest speech, so speech is never
        removed — only RVC's hallucinated pause noise. Smooth 20ms-frame ramp,
        seeded from the previous block so there are no clicks at joins.
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

    def _convert_block(self):
        """Convert the newest block using the trailing window as context; return
        exactly block_samples of audio (silence if the block wasn't speech)."""
        B = self.block_samples
        new_block = self.input_history[-B:]
        if len(new_block) < B:
            new_block = np.pad(new_block, (B - len(new_block), 0))

        # Pause -> emit silence, skip RVC (keeps timeline; avoids hallucinated noise)
        if np.sqrt(np.mean(new_block ** 2)) <= self.vad_threshold:
            return np.zeros(B, dtype=np.float32)

        window = self.input_history[-self.window_samples:]
        try:
            import time as _t
            _t0 = _t.perf_counter()
            conv = self.converter.convert_streaming(
                window, sr=self.input_sr, pitch_shift=self.pitch_shift,
            )
            self.last_inference_ms = (_t.perf_counter() - _t0) * 1000
        except Exception as e:
            print(f"[StreamProcessor] Conversion error: {e}")
            return np.zeros(B, dtype=np.float32)
        if conv is None or len(conv) == 0:
            return np.zeros(B, dtype=np.float32)

        # Emit only the newest block-worth of the converted window.
        emit = conv[-B:] if len(conv) >= B else np.pad(conv, (B - len(conv), 0))
        if self.gate_enabled:
            emit = self._gate_to_input(emit, new_block)
        return emit.astype(np.float32)

    def process(self, audio_bytes):
        if not self.is_active:
            return None

        incoming = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(incoming) == 0:
            return None

        self.input_history = np.concatenate([self.input_history, incoming])
        if len(self.input_history) > self.history_cap:
            self.input_history = self.input_history[-self.history_cap:]
        self.pending += len(incoming)

        if self.pending < self.block_samples:
            return None

        # If we ever fall behind, drop the backlog rather than let latency grow.
        if self.pending > 2 * self.block_samples:
            self.pending = self.block_samples

        self.pending -= self.block_samples
        out = self._convert_block()
        if out is None or len(out) == 0:
            return None
        return out.astype(np.float32).tobytes()
