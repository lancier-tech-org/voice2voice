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
| 4 | Amplitude pumping / noise floor breathes | `rvc_converter.convert()` peak-normalizes EACH chunk to 0.95 → gain differs per chunk | ISOLATED | Steady/fixed gain instead of per-chunk normalization | low |
| 5 | ~11% of speech lost, words jammed together | `stream_processor` VAD returns None on chunks it judges silent → never sent → browser plays rest back-to-back (measured 30.0 s out for 33.7 s in) | ISOLATED | Silence in place instead of dropping (keep timeline) — must NOT over-gate | low-med |
| 6 | Background noise during pauses | RVC hallucinates sound with no speech input (input was ~39% silence) | ISOLATED | Correctly-calibrated input-keyed gate. CAUTION: a prior gate at 0.006 silenced ~21% of real speech (median frame 0.0057). Threshold must sit ~3x noise floor (~0.0015), below quietest speech | med (broke things before) |
| 7 | Bursty output / breaks in continuous speech | GPU inference runs ON the asyncio event loop (not executor); during ~290 ms conversion the server can't send/recv → bursts, worse on slow GPU | ISOLATED | Run conversion in a worker thread (run_in_executor) | low |
| 8 | Gaps / underruns in playback | Browser schedules only 20 ms ahead (`nextPlayTime`), so uneven arrival underruns | ISOLATED | Larger jitter buffer (trade a little latency for smoothness) | low (frontend) |
| 9 | ~1.9 s latency | 1.5 s chunk buffering + ~0.29 s inference + playback buffer | STRUCTURAL-ish | Smaller emit block + context, or realtime engine (~300–500 ms floor) | high effort |

## Negative results already established (do not repeat)
- Larger chunks (3 s) made it WORSE → fewer boundaries is not the fix by itself.
- SOLA without shared context DESTROYS voice identity.
- Stacking multiple changes made it impossible to attribute regressions — the
  reason for "one change at a time".

## Suggested order (each validated against the recording before moving on)
Phase A — isolated, low-risk, measurable individually:
  1. #7 event-loop → executor
  2. #4 per-chunk normalization
  3. #8 playback jitter buffer
  4. #5 stop dropping speech (silence-in-place, carefully)
  5. #6 gate ONLY if pause-noise remains, with a correctly measured threshold
Phase B — structural (the real ceiling), only after A is exhausted:
  6. #1/#2/#3/#9 via the tensor realtime engine, on a branch, step by step.

Baseline to return to at any time: tag `audio-work-start` / `bash ~/RESTORE-KNOWN-GOOD.sh`.
