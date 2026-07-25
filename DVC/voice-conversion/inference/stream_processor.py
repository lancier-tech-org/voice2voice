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

        # REAL context around each emitted block.
        #
        # RVC's pipeline reflect-pads every call with x_pad seconds of TIME-REVERSED
        # audio (x_pad=3 on this T4, so a bare 1.5s chunk was inferred as ~7.5s of
        # which only 20% was real, and RMVPE estimated F0 over the mirror too). The
        # words at each chunk edge were therefore conditioned on phonetic nonsense —
        # that is the boundary garbling. It is also why crossfade and SOLA could not
        # help: consecutive chunks mirrored DIFFERENT audio, so the shared overlap
        # was rendered two genuinely different ways, leaving nothing to align.
        #
        # So put real audio there instead. History before the block is already in
        # the buffer, so it costs NO latency; a short lookahead after it gives the
        # emitted frames right-context too. Only the middle is emitted and the
        # context margins are discarded, so nothing the listener hears sits next to
        # fabricated audio.
        #
        # Latency is emit + lookahead = 1.5s, the same as the old 1.5s chunk buffer.
        # Sizes chosen by measuring HuBERT content similarity against the input —
        # i.e. "are the words still the spoken words" — which is the one thing the
        # roughness/level metrics are blind to. Measured on a 16s dense-speech
        # excerpt plus two longer recordings:
        #   hist 2.0 HURTS (0.799 vs 0.827) — more past context is not better
        #   look 1.0 helps, most of all on the worst 10% of frames (the garbled words)
        #   emit 0.5 hurts, confirming the earlier "smaller blocks are worse" result
        self.hist_sec = 1.0     # real past context (free — already buffered)
        self.emit_sec = 1.0     # block actually sent (smaller measured worse)
        # Was tried at 1.0: better content on paper, but it pushed latency to 2.0s AND
        # made the inference tail far worse (p95 327->462ms, worst call 1354->3900ms
        # against a 1.0s budget). Since inference runs on the event loop, a spike that
        # long stalls sending entirely — the user heard exactly that as breaks during
        # long speech. Back to 0.5s: latency 1.5s, and a much shorter tail.
        self.look_sec = 0.5     # real future context
        self.lap_sec = 0.010    # short join between consecutive blocks
        self.pos = 0
        self.prev_lap = None

        # VAD
        self.vad_threshold = 0.003

        # Output gate keyed to the CLEAN INPUT: silence output wherever the input
        # had no speech, to kill the noise RVC hallucinates at pause boundaries.
        # Threshold sits BELOW the quietest real speech (measured: noise floor
        # ~0.0005, quietest speech frames ~0.006) so speech is never cut — the
        # mistake in an earlier attempt was a threshold (0.006) above median speech.
        # 0.004 chosen by measuring this against a real recording: pause-noise max
        # 0.12 -> 0.027 (spikes killed) with only ~1.3% speech change. Higher (0.008+)
        # starts cutting quiet speech; lower (0.002) barely helps.
        self.gate_enabled = True
        self.gate_threshold = 0.0040   # legacy fixed value, kept for reference only
        self._gate_prev = 1.0

        # ADAPTIVE gate threshold.
        #
        # A single absolute threshold cannot work here: across the user's own
        # recordings the input peak spans 0.036 to 0.954 — a 26x range — because
        # autoGainControl is off in the browser. 0.0040 was calibrated against a
        # recording whose median speech frame was 0.0148; on a quiet one whose median
        # speech frame is 0.0051 it sat ABOVE most of the speech and deleted it.
        # Measured speech loss with the fixed threshold: 11.2% / 17.7% / 70.4% on
        # three quiet recordings. That is the "words suddenly not playing", and as a
        # speaker tires and drops in level it also explains the output growing
        # quieter and gappier over a long session.
        #
        # So track this speaker's own noise floor and speech level and place the
        # threshold between them (geometric mean, biased toward keeping speech).
        # Both estimates are slow EMAs, so it follows a drifting mic level instead of
        # progressively deleting more speech.
        self.gate_adaptive = True
        self.gate_bias = 0.6        # <1 favours keeping speech over killing noise
        self.gate_abs_floor = 0.0008
        self._noise_est = None
        self._speech_est = None

    def start(self, voice_path, pitch_shift=0, input_sr=48000):
        self.converter.load_voice(voice_path)
        self.pitch_shift = pitch_shift
        self.input_sr = input_sr
        self.input_buffer = np.array([], dtype=np.float32)

        self.hist_samples = int(self.hist_sec * input_sr)
        self.emit_samples = int(self.emit_sec * input_sr)
        self.look_samples = int(self.look_sec * input_sr)
        self.lap_samples = int(self.lap_sec * input_sr)
        # Kept under the old names so callers/metrics keep working: one emitted
        # block per call, and this much input must arrive before the first one.
        self.stride_samples = self.emit_samples
        self.chunk_samples = self.emit_samples + self.look_samples

        self.pos = 0
        self.prev_lap = None
        self._gate_prev = 1.0
        self._noise_est = None
        self._speech_est = None
        self.is_active = True

        self.lap_in = np.linspace(0, 1, self.lap_samples, dtype=np.float32)
        self.lap_out = 1.0 - self.lap_in

        print(f"[StreamProcessor] Started | pitch={pitch_shift} | "
              f"hist={self.hist_samples} emit={self.emit_samples} "
              f"look={self.look_samples} window="
              f"{self.hist_samples + self.emit_samples + self.look_samples}")

    def stop(self):
        self.is_active = False
        self.input_buffer = np.array([], dtype=np.float32)
        self.pos = 0
        self.prev_lap = None
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

    def _level_threshold(self, env):
        """
        Threshold separating this speaker's speech from their own silence.

        Placed at the geometric mean of the tracked noise floor and speech level,
        biased below it, so it adapts to a 26x spread in input level instead of
        deleting speech on quiet input. Slow EMAs so a drifting mic level is
        followed rather than accumulating loss.
        """
        if not self.gate_adaptive:
            return self.gate_threshold
        lo = float(np.percentile(env, 10))
        hi = float(np.percentile(env, 90))
        if self._noise_est is None:
            self._noise_est, self._speech_est = lo, hi
        else:
            # ASYMMETRIC on purpose. A symmetric EMA of each block's 10th percentile
            # is wrong: in a block that is entirely speech that percentile IS speech
            # level, so the "noise floor" climbs and the threshold starts cutting
            # words — measured as speech loss rising 0.17% -> 3.02% on in_105612.
            # So the noise floor follows evidence of quiet quickly but rises only
            # very slowly, and the speech level does the reverse.
            self._noise_est = (0.7 * self._noise_est + 0.3 * lo) if lo < self._noise_est \
                else (0.995 * self._noise_est + 0.005 * lo)
            self._speech_est = (0.7 * self._speech_est + 0.3 * hi) if hi > self._speech_est \
                else (0.99 * self._speech_est + 0.01 * hi)
        thr = self.gate_bias * float(np.sqrt(
            max(self._noise_est, 1e-9) * max(self._speech_est, 1e-9)))
        # Never above half the speech level (that would cut speech), never below the
        # absolute floor (that would gate nothing at all).
        thr = min(thr, 0.5 * self._speech_est)
        return max(thr, self.gate_abs_floor)

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
        gate = (env > self._level_threshold(env)).astype(np.float32)

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

        # Need the block plus its lookahead before anything can be emitted. The
        # history behind `pos` is already there and is never waited for.
        if len(self.input_buffer) < self.pos + self.emit_samples + self.look_samples:
            return None

        emit_in = self.input_buffer[self.pos:self.pos + self.emit_samples].copy()
        w0 = max(0, self.pos - self.hist_samples)
        w1 = self.pos + self.emit_samples + self.look_samples
        window = self.input_buffer[w0:w1].copy()
        off = self.pos - w0                     # emitted block's offset in window

        # Advance, keeping only as much past audio as the history margin needs.
        self.pos += self.emit_samples
        if self.pos > self.hist_samples:
            drop = self.pos - self.hist_samples
            self.input_buffer = self.input_buffer[drop:]
            self.pos -= drop

        # VAD on the EMITTED region only — a pause inside the context margins must
        # not suppress a block that does contain speech.
        #
        # Decided on the LOUDEST 20ms frame, not the block average: a short quiet word
        # surrounded by silence averages below any sensible threshold, and averaging
        # would drop the whole 1s block and with it the word. The only job here is
        # "does this block contain any speech at all" — _gate_to_input handles
        # silencing the non-speech parts within it.
        _f = max(1, int(0.02 * self.input_sr))
        _n = max(1, len(emit_in) // _f)
        _env = np.sqrt(np.mean(emit_in[:_n * _f].reshape(_n, _f) ** 2, axis=1))
        if float(_env.max()) <= self.vad_threshold:
            self.prev_lap = None
            # Silence in place, never a dropped block: dropping consumed input but
            # sent nothing, so the browser played the rest back-to-back and the
            # output ran ahead of the speaker. Every path below returns exactly one
            # block, which is what keeps output in step with speech.
            return np.zeros(self.emit_samples, dtype=np.float32).tobytes()

        try:
            converted = self.converter.convert_streaming(
                window, sr=self.input_sr, pitch_shift=self.pitch_shift,
            )
            if converted is None or len(converted) == 0:
                return np.zeros(self.emit_samples, dtype=np.float32).tobytes()

            # Cut the emitted block out of the middle. RVC may resample, so map
            # window offsets through the length ratio rather than assuming 1:1.
            ratio = len(converted) / float(len(window))
            s = int(round(off * ratio))
            need = self.emit_samples + self.lap_samples
            seg = converted[s:s + need]
            if len(seg) < need:
                seg = np.pad(seg, (0, need - len(seg)))
            seg = seg.astype(np.float32).copy()

            # Short join with the previous block. Both sides of this seam were
            # generated from the same real audio with the same real context, so
            # unlike the old 500ms crossfade of two independently-padded chunks
            # they are near-duplicates and blend without combing.
            if self.prev_lap is not None and self.lap_samples > 0:
                seg[:self.lap_samples] = (
                    seg[:self.lap_samples] * self.lap_in
                    + self.prev_lap * self.lap_out
                )
            self.prev_lap = seg[self.emit_samples:need].copy()
            output = seg[:self.emit_samples]

            # Silence output where the INPUT had no speech — removes the noise RVC
            # hallucinates in pauses. Keyed to the clean input, now exactly aligned
            # with the emitted block.
            if self.gate_enabled:
                output = self._gate_to_input(output, emit_in)

            return output.astype(np.float32).tobytes()

        except Exception as e:
            print(f"[StreamProcessor] Conversion error: {e}")
            # Keep the timeline even on error — emit silence for this block.
            return np.zeros(self.emit_samples, dtype=np.float32).tobytes()

    def get_latency_ms(self):
        return (self.emit_sec + self.look_sec) * 1000