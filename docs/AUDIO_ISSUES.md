# Live-audio quality — issue outline

Distinct problems in the live streaming path, to be fixed ONE AT A TIME and each
validated against a real recording (assistant cannot hear audio). Offline
whole-file conversion is clean (10.65 dB); everything below is live-only.

Two categories:
- **STRUCTURAL** — inherent to slicing an offline model into independent chunks.
  Real fix = give the model context across chunks (tensor realtime engine).
- **ISOLATED** — self-contained bugs in the current pipeline; each fixable and
  testable on its own, lower risk.

| # | Issue (symptom) | Cause | Type | Fix direction | Risk |
|---|---|---|---|---|---|
| 1 | Words on chunk boundaries mispronounced/garbled | Each 1.5 s chunk converted with NO preceding context; phonemes split across boundaries converted blind | STRUCTURAL | Input-context window / realtime engine keeps history | high effort |
| 2 | Pitch wobble at boundaries | RMVPE re-estimates F0 from scratch per chunk; edge F0 unreliable | STRUCTURAL | Cache pitch across calls (realtime engine does this) | high effort |
| 3 | Clicks / discontinuity at seams | Crossfade of two independently-converted overlapping chunks that don't align in phase → comb/click | STRUCTURAL | Proper stitching WITH shared context (naive crossfade + SOLA-without-context both fail) | high effort |
| 4 | Amplitude pumping / noise floor breathes | `rvc_converter.convert()` peak-normalizes EACH chunk to 0.95 → gain differs per chunk | ISOLATED | **FIXED 2026-07-25** (commit `d735c45`). Level fixed on the OUTPUT, not the input: converted RMS rescaled to 3.5x the input chunk's RMS. Input normalization kept — reducing input drive measurably raised roughness. Level step p95 5.3→2.1 / 8.1→4.9 / 10.0→3.0 dB on three recordings; per-frame roughness and spectrum unchanged; pacing and pause silence unchanged | low |
| 5 | ~11% of speech lost, words jammed together | `stream_processor` VAD returns None on chunks it judges silent → never sent → browser plays rest back-to-back (measured 30.0 s out for 33.7 s in) | ISOLATED | Silence in place instead of dropping (keep timeline) — must NOT over-gate | low-med |
| 6 | Background noise during pauses | RVC hallucinates sound with no speech input (input was ~39% silence) | ISOLATED | Correctly-calibrated input-keyed gate. CAUTION: a prior gate at 0.006 silenced ~21% of real speech (median frame 0.0057). Threshold must sit ~3x noise floor (~0.0015), below quietest speech | med (broke things before) |
| 7 | Bursty output / breaks in continuous speech | GPU inference ran ON the asyncio event loop; during conversion the server could neither send nor recv | ISOLATED | **FIXED 2026-07-25** — single-worker ThreadPoolExecutor (one worker, so blocks cannot overtake each other and GPU access stays serialized). Had to ship WITH #8 | low |
| 8 | Gaps / underruns in playback | Browser scheduled only 20 ms ahead, which cannot absorb a p95 arrival spread of ~330 ms | ISOLATED | **FIXED 2026-07-25** — 250 ms jitter buffer, plus an underrun counter shown in the UI so the user's own session reports breaks. On a 220 s recording, starvation 978 ms → 383 ms | low (frontend) |
| 10 | ~4 s stall on the FIRST conversion of a session | CUDA autotuning/lazy alloc: call 0 measured 3929 ms, calls 1-119 a steady 268-284 ms | ISOLATED | **FIXED 2026-07-25** — throwaway conversion at container startup (`_warm_up`), so the cost is paid before any user connects, not on their first words | low |
| 9 | ~1.9 s latency | 1.5 s chunk buffering + ~0.29 s inference + playback buffer | STRUCTURAL-ish | Smaller emit block + context, or realtime engine (~300–500 ms floor) | high effort |

## Why SOLA failed, mechanistically (2026-07-25)
`rvc/pipeline.py` pads every call with `x_pad` seconds of **reflect** (time-reversed)
audio — on a T4 `is_half=True` → `x_pad = 3`, so each 1.5 s chunk is inferred as
~7.5 s of which only 20 % is real, and RMVPE estimates F0 over the mirrored signal
too. Boundary frames are therefore conditioned on *fabricated* context, not merely
missing context. Consecutive chunks mirror *different* audio, so the shared 0.5 s
overlap is rendered two genuinely different ways — SOLA's correlation search has no
redundancy to lock onto, picks an arbitrary shift, and combs the spectrum. SOLA is
a stitching method and presupposes near-duplicate overlaps; it cannot work until
real shared context exists. This also explains why inference cost is ~fixed
regardless of window length (0.35 s→242 ms, 2.0 s→276 ms): the padding dominates.

Corollary — the past audio needed as left context is already in the buffer and is
thrown away at `stream_processor.py` (`input_buffer[stride:]`). Supplying it costs
**zero** latency. The 2 s-window attempt that "failed" bundled that with a 0.5 s
emit block (measured worse: 1244 vs 928 discontinuities) and per-chunk
normalization, so two of three changes were regressions.

## Negative results already established (do not repeat)
- Larger chunks (3 s) made it WORSE → fewer boundaries is not the fix by itself.
- SOLA without shared context DESTROYS voice identity (mechanism above).
- Stacking multiple changes made it impossible to attribute regressions — the
  reason for "one change at a time".

## Suggested order (each validated against the recording before moving on)
Phase A — isolated, low-risk, measurable individually:
  1. #5 stop dropping speech — DONE (`b0aa343`)
  2. #6 input-keyed pause gate @0.0040 — DONE (`b0aa343`)
  3. #4 per-chunk normalization — DONE (`d735c45`)
  4. #7 + #8 together — event-loop→executor AND the 20 ms playback buffer. Must be
     one step: #7 alone measured worse last time, because without the executor
     arrival was accidentally regular, and freeing the loop made it bursty against
     a 20 ms buffer.
Phase B — the real ceiling, in this order (each validated separately):
  5. Free left context: keep the past stride instead of discarding it, emit the
     SAME 1.0 s block, so emitted frames stop being conditioned on reflect-pad.
     Zero latency cost.
  6. Right-context margin (~0.5 s), emit the window's middle, butt-join instead of
     the 500 ms crossfade. Costs latency the current design already spends.
  7. SOLA at the joint — only meaningful once 5 and 6 give it shared context.
  8. Alternative with a higher ceiling: cut at silence (VAD phrase boundaries)
     rather than on a clock, converting whole phrases. Every seam then lands in a
     pause, so nothing is split and the reflect pad mirrors silence. Gives offline
     quality at the cost of variable, phrase-length latency.

## Measurement harness (use this, don't guess)
Replay a real recording through `StreamProcessor` in the same 4096-sample packets
the browser sends, and measure: pacing ratio + drops, per-stride level step (dB,
speech frames only — averaging whole strides mixes in the pause gate and swamps
it), speech frames lost, pause noise, and roughness/spectrum **per 20 ms frame
normalized by that frame's own RMS**. That last point matters: an absolute-
threshold jump count rewards any change that merely lowers the level, and it
reported a spurious 67 % regression on the #4 fix. Real inputs are captured to
`uploads/debug/in_*.wav` by the `dbg_on` path in `routes_convert.py`.

Baseline to return to at any time: tag `audio-work-start` / `bash ~/RESTORE-KNOWN-GOOD.sh`.

## Real-time measurement (2026-07-25) — the offline harness is not enough
The offline replay harness runs as fast as the GPU allows, so it is **structurally
blind** to real-time failures: it cannot see that a call occasionally overruns its
budget while blocking the event loop. Use `rt_client.py`, which streams a recording
over the real websocket at exactly 1x and simulates the browser's scheduler to
report *starvation* — the moments playback would have had nothing to play. That is
the metric that corresponds to "breaks".

## Content preservation — the metric for "are these the words I said"
Roughness/periodicity/level are all blind to pronunciation: a word converted with
the wrong context is perfectly smooth and simply wrong. Use HuBERT feature cosine
similarity between output and input (hubert_base.pt is already loaded for
inference, so no new dependency). It ranks sensibly — offline whole-file 0.891 >
context path 0.82 > old bare-chunk 0.802 — and it was the only metric that could
see the context change working.

**Run-to-run variance is ~±0.02** (CUDA nondeterminism). Differences smaller than
that are noise: a single run made a 1.0s lookahead look like a clear win, and on
repeats it was indistinguishable from 0.5s. Always repeat before believing a knob.
Reproducible so far: `index_rate` 0.75 → 0.2 is worth about +0.02 consistently.
