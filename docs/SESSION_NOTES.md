# voice2voice — Engineering Session Notes / Handoff

Durable record of what was diagnosed, fixed, measured, and what remains open.
Kept in-repo so it is preserved with the code.

Last updated: 2026-07-24.

---

## 1. What the system is
Real-time voice conversion built on **RVC (Retrieval-based Voice Conversion)**.
A target voice is trained once; a live speaker's mic audio is re-cast in the
target's timbre while keeping the speaker's words/prosody. Female→male here is
cross-gender (hardest RVC case), needs a large pitch shift (~-12/-13 semitones).

- Backend: FastAPI (`DVC/voice-conversion/`), WebSocket streaming conversion.
- Engine: the **`rvc` PyPI package 0.3.5** (NOT the vendored RVC-WebUI).
- Vendored RVC-WebUI at `DRW/rvc-webui` is used only for **training** subprocesses.
- Deploy: Docker on an Azure T4 VM, behind nginx at **https://v2v.lancieretech.com**.
- Dev machine that produced the original good model: `landevaiser` (RTX 6000).

## 2. Root causes found & fixed (all committed)
1. **Training silently produced broken voices.** RVC fine-tunes from pretrained
   base models (`f0G/f0D48k.pth`); they were never downloaded, so training ran
   from random weights and the trainer *silently* blanked the `-pg/-pd` flags.
   Fix: Dockerfile downloads all 6 base models; trainer raises if they're missing.
   Proof: `anil` (no base) HNR **6.96 dB** vs `anil_v2` (fine-tuned) **10.65 dB**.
2. **Wrong RVC engine in the container.** Code targets the `rvc` pip package
   (`VC()`, `vc_single(..., index_file=, hubert_path=)`), which was never in
   requirements. A Dockerfile symlink made `import rvc` resolve to RVC-WebUI (a
   different, incompatible engine). Fix: `rvc==0.3.5` pinned; WebUI moved to
   `/app/rvc-webui`, dropped from PYTHONPATH. This also pins numpy<2 (fairseq
   needs it).
3. **Crash bugs:** missing `configs/inuse/<ver>/`; HuBERT relative-path
   (`assets/hubert/...`) not found; `get_vc` doubled an absolute path; frontend
   hardcoded `wss://` broke the mic over HTTPS (now protocol-adaptive).
4. **Training controls added:** Pause (keep checkpoints) / Cancel (discard) /
   Resume-from-checkpoint, killing the whole process group; UI re-attaches to a
   running/paused job after refresh. All verified live end-to-end.
5. Checkpoints persist across rebuilds (added volumes). Epochs read config (200,
   not a hardcoded 400 — quality ties at 200 vs 400, half the time).

## 3. Key measurements (source input = 12.37 dB HNR)
- pavan_sai good model: **9.75 dB**; anil_v2: **10.65 dB**; broken models: ~6.3–7 dB.
- User voice (vineesha/source) median F0 **~263 Hz**; pavan_sai target **~125 Hz** →
  needs **-13 semitones** (round). anil target ~125 Hz too.
- Inference on T4 is ~**fixed cost ~230–290 ms** regardless of window length
  (0.35 s→242 ms, 2.0 s→276 ms). So long context is nearly free; calling more
  often is what costs.
- Offline whole-file conversion is clean (10.65 dB). Live streaming is worse:
  measured 30.0 s out for 33.7 s in (**~11% speech lost**) + more discontinuity.

## 4. THE OPEN PROBLEM: live streaming quality
Symptoms (live only, not offline): disturbances/noise, breaks during continuous
speech, mispronounced/missing words, ~1.9 s latency.

Root cause is structural: `stream_processor.py` converts **independent 1.5 s
chunks** with no context between them, so words spanning boundaries are converted
blind and pitch tracking restarts each chunk. This is inherent to slicing an
offline model's input.

### What was tried and FAILED (do not repeat blindly)
- Noise gate (threshold 0.006) — sat above the user's median speech frame
  (0.0057) → silenced ~21% of speech = "missing words". REVERTED.
- Per-chunk peak normalization already in `convert()` pumps the noise floor;
  shrinking the block made it worse (gain changes more often). 
- 80 Hz high-pass per chunk (no carried state) → transient click each chunk.
- Executor + 150 ms jitter buffer, equal-power crossfade, context-window rewrite
  (2 s window / 0.5 s block) — all REVERTED; each made it worse to the user's ear.
- Larger chunks (3 s) — **made it worse**, so chunk-boundary count is NOT the
  whole story. Important negative result.
- SOLA — the user tried it pre-deployment: it *destroyed voice identity* (with no
  shared context between independent conversions, its correlation search combs
  the spectrum). Confirmed dead end without context continuity.

### The honest constraint
The assistant cannot hear audio. Every change must be validated against a REAL
recording of the user's voice, measured, ONE change at a time. Guessing from
metrics failed repeatedly. The proven-correct path if pursued: the tensor-based
realtime engine (`DRW/rvc-webui/tools/rvc_for_realtime.py`) which keeps model +
pitch state in memory (no temp WAV, no per-call setup) — but that is a real
integration, done on a branch, validated step by step.

## 5. Restore points & backups (all on this VM)
- Tags: `working-original-audio` (current good audio path), `known-good-2026-07-23`.
- Backup branches: `backup/working-original-audio`, `backup/known-good-2026-07-23`.
- One-command restore: `bash ~/RESTORE-KNOWN-GOOD.sh`.
- Code snapshots: `~/code-snapshots/*.tar.gz`.
- **Model master copies (gitignored, not in repo):** `~/voice-models-master/`
  (pavan_sai good = md5 5f0a4ae2…, anil_v2, plus _bad_jul21_retrain for reference).

## 6. Open items
- **Push:** branch `fix/rvc-engine-and-training-pretrained-models` (+tags) is
  committed locally only — no GitHub creds on this VM. `git push -u origin …`.
- **Models off-machine:** only on this VM + host copy. Move to blob/LFS.
- `anil` (6.96 dB, pre-fix broken) still selectable — consider deleting.
- Live streaming quality — the main remaining work (see §4).

## 7. Working principles for this codebase (learned the hard way)
- ONE change at a time; validate each against a real recording before the next.
- Never stack unvalidated audio changes. Keep a known-good tag before touching it.
- The audio path files: `inference/stream_processor.py`, `inference/rvc_converter.py`,
  `api/routes_convert.py`, `frontend/index.html` (playback). Training is separate.
- Trained models are BUILD ARTIFACTS — copy at deploy, never retrain on the server.

---

## 8. Working configuration (2026-07-27) — user-verified
Tag `audio-good-vamsi8-phone`. User: "i tested the voice in phone at -8 vamsi voice and
its clear".

**Use the phone as the microphone, and `vamsi` at -8 semitones.**

Measured on their real session: silence sits **-24 dB** below speech with only 3% of
silent frames non-zero (the laptop mic gave -2 dB and 77%); periodicity 0.654 against
their input's 0.775, i.e. 84% of the harmonic structure retained (`pavan_sai` at -12
gave 0.42); pacing 0.983; capture 100%.

**The two things that mattered were both OUTSIDE the conversion code:**

1. **Microphone.** Phone: noise floor 0.00000, speech 0.093. Laptop: floor 0.0004,
   speech 0.003 — only 4-10x apart, i.e. the user's quiet speech and the laptop's noise
   OVERLAP in level. Eight gate strategies were tried and measured; every one fails the
   same way, because no threshold can separate two things that overlap. The laptop mic
   was the entire "background noise" complaint.
2. **Target voice pitch, not the shift amount.** Required shift = gap between source and
   TARGET F0. Source 262 Hz. `pavan_sai` 125 Hz -> -13 (periodicity 0.42);
   `vamsi` 169 Hz -> **-8** (periodicity 0.65) and still clearly male. Measure a
   candidate target's F0 from `rvc-webui/logs/<voice>/0_gt_wavs` BEFORE choosing it.
   Do not shrink the shift below what the target needs — pick a better-matched target.

**Residual, accepted:** chopping at ~2.5 abrupt drops/s vs 0.2 offline; ~3.8% speech
loss; ~1.5 s latency. Every cheap lever swept and negative (see AUDIO_ISSUES.md). The
only remaining fix is the vendored tensor engine with cached pitch state — not attempted.

## 9. Hardening done 2026-07-27
- **Per-session StreamProcessor.** Previously ONE instance served every websocket, so two
  concurrent connections (second tab, or Start pressed before the old socket closed) wrote
  into the same input buffer and read position and corrupted each other. Each connection
  now builds its own. The RVCConverter is still shared and holds one loaded voice, so the
  newest session wins via `state.session_token` and older ones exit cleanly — verified:
  older session closed after 1 block, newer ran isolated with its own contiguous sequence.
- **NVIDIA driver pinned.** All 22 nvidia/libnvidia packages `apt-mark hold`'d.
  unattended-upgrades swapping 535 for 580 mid-session broke the GPU on 2026-07-27.
- **`nvidia-cdi-refresh.service`** (new, enabled): regenerates `/var/run/cdi/nvidia.yaml`
  before docker.service starts. The second half of that outage was the CDI spec still
  listing deleted 535 library paths, which makes every `--gpus` container fail with
  "failed to fulfil mount request". Recovery then needed a manual module reload plus
  `nvidia-ctk cdi generate`. Verified: spec references match the running driver.
