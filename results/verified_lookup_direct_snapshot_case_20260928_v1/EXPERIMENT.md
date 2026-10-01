# Verified lookup reuse and direct checkpoint copies

Baseline: unmodified pristine vLLM + LMCache. Experimental changes are confined
to the existing experimental source trees/branches. The previous convolution
scalar-readback fix remains enabled.

## Changes in this run

1. ReplaceTailConnector passes the confirmed (server-verified) Full KV hit to
   TailProbeConnector. After the separately verified tail-state hit, the latter
   reuses both existing read leases and does not enter the ordinary state
   namespace lookup. No unverified local-index hint is accepted as a cache hit.
   Pending, missing and unverified cases retain the previous fallback.
2. recurrent_with_checkpoints optionally copies each intermediate recurrent
   state directly into the caller's reserved snapshot slot. The next segment
   still reads the kernel-produced state, not the snapshot slot. This removes
   clone-then-copy; recurrence segmentation and multi-segment concatenation
   remain. Single-segment output no longer needs concatenation.

No changes to native baseline, CPU object layout, H2D submission, completion
events, snapshot block reservation or checkpoint maintenance policy.
KV lookup followed by tail lookup remains sequential; this change eliminates
the redundant ordinary-state lookup, not all lookup round trips.

## Correctness checks before launch

- test_tail_verified_lookup.py executes the real method via AST extraction:
  verified hit skips parent lookup, pending does not fabricate a hit, missing
  and unverified states fall back, insufficient Full KV coverage fails closed.
- test_checkpoint_capture.py exercises 39/(32), 872/(528,864), 1054/(528),
  1054/(1040), 151/(144), 663/(528). Direct destinations and old clone-based
  execution produce bit-identical outputs, final states and snapshots.
  Overwriting the final running state leaves snapshots unchanged. Independent
  prefix/final recurrence comparisons remain within relative L2 threshold 0.025.
- Both pristine source trees were clean; diff --check passed for experiments.

## Measurement

One real ShareGPT session WRAImOg_0, seed 872 / resume 903 tokens, three paired
repetitions per arm, one generated token. GPU 1; eager in BOTH arms. Ten unrelated
warmup requests per startup, five formal GPU resets per arm. CPU cache retained
within each pair and cleared between pairs. Same CPU affinity 0-11,24-35.
Experimental arm first, fresh native arm second; no reused baseline numbers.
Native prefill budget 528 vs experimental 2048: system comparison, not a pure
single-factor ablation. One startup per arm and three repetitions are exploratory.

Expected resume: native KV/state 528, forward 375; experimental KV/state 864,
forward 39, next checkpoint S896 still maintained. First-token comparison is
only a smoke check, not full-sequence equivalence. The source manifest includes
modified connector, GDN, allocator and scheduler files.

## Stopped before measurement

The model launch was interrupted during weight loading under severe disk I/O
pressure, then the user explicitly requested stopping. No warmup or formal
TTFT measurements completed; the native arm was not launched. The runner's
failed status reflects SIGINT, not a demonstrated model correctness failure.
All experiment processes exited; code and logs are retained.

A separate 24-layer recurrence component check (T39, checkpoint32, 12 alternating
repetitions after 4 warmups, includes snapshot slot copy) measured clone+copy
15.678 ms versus direct copy 15.018 ms. This excludes the model and H2D and ran
while the service was waiting for disk; it is not an end-to-end speedup claim.
