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
| 7 | Bursty output / breaks in continuous speech | GPU inference runs ON the asyncio event loop (not executor); during ~290 ms conversion the server can't send/recv → bursts, worse on slow GPU | ISOLATED | Run conversion in a worker thread (run_in_executor) | low |
| 8 | Gaps / underruns in playback | Browser schedules only 20 ms ahead (`nextPlayTime`), so uneven arrival underruns | ISOLATED | Larger jitter buffer (trade a little latency for smoothness) | low (frontend) |
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
