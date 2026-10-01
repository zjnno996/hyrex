# Single real-session case

## Final comparison: freshly rerun native baseline

Use this table, not the older reused-baseline comparison below, for the final
within-session result. Native sources remained unmodified and were verified clean.

| Repetition | Fresh native seed ms | Optimized seed ms | Fresh native resume ms | Optimized resume ms |
|---|---:|---:|---:|---:|
| 1 | 195.11 | 215.88 | 143.27 | 173.93 |
| 2 | 171.95 | 215.45 | 142.52 | 156.18 |
| 3 | 186.58 | 210.55 | 126.94 | 167.00 |
| Mean | 184.55 | 213.96 | 137.58 | 165.70 |

Optimized resume remains 28.13 ms / 20.44% slower than the freshly rerun native
baseline. Against the previous split-forward tail implementation (227.24 ms),
resume TTFT decreased by 61.53 ms / 27.08%. That historical comparison is
sequential and not an alternating-order statistical estimate.

### Changes verified in this run

- Seed: one model forward over 872 tokens, capturing S528 and S864.
- Resume: all three requests log `state=864 full=864`, and exactly one
  `SINGLE_FORWARD start=864 count=39 checkpoints=[(528, 0), (896, 3)]`.
- Tail snapshots use a separate GPU slot. One guard slot plus one snapshot slot
  are reserved through the native Mamba block manager, with admission accounting
  and native STORE-future-controlled release. Approximate added GPU payload:
  99 MiB across 24 recurrent layers. Extra CPU tail checkpoint: 49.5 MiB.
- The old synchronous tail-save barrier is absent (`SAVE_BARRIER_MS` count zero).
  This removes that explicit device-wide synchronization, not all possible syncs.
- Projection, attention and MLP execute once per request prefill. GDN recurrence
  is still internally segmented at snapshot boundaries; the 7-token final
  segment uses the existing recurrent kernel with final-state-only output.
- New S896 maintenance remains enabled. This is not a skip-new-state ablation.

### Checks and limitations

- Both measured arms: 10 startup warmups, 6 formal requests, 5 successful GPU
  resets, CPU retained within each seed/resume pair and cleared between pairs.
- All six paired first tokens agree. Full-sequence equivalence is not established.
- GPU recurrence tests passed for all lengths 1..15 against FP32 reference;
  initial state stays unchanged. Checkpoint tests at lengths 79, 151, 1054,
  1071 and 1056 passed, with checkpoint tensors unchanged by the short-tail
  optimization and output/final-state relative errors below 0.025.
- The 24-layer component benchmark was T=79: 23.134 ms versus 14.383 ms.
  It is not a measurement of the 39-token model forward or TTFT.
- Actual block-manager method tests passed for reservation accounting and
  snapshot/decode separation. Experimental mode supports one request and
  1..32 generated tokens; longer generation is explicitly rejected.
- Native prefill budget=528, optimized=2048. This is a system comparison,
  not a single-factor ablation. One real session with three repetitions;
  no claim of general speedup, throughput benefit or complete profiling.
- Remaining overhead is not causally decomposed by this run. Short-prefill
  execution, GDN capture/copies, lookup and transfer require GPU/CPU timelines.
- GPU 1 was released after the experiment.

### Fresh baseline provenance

`3_baseline_fresh` was run immediately after `1_optimized`, using the exact command
and clean environment recorded in
`/root/hyrex_results/coarse_single_case_20260928_v2/2_baseline/command.json`,
changing only `--output-dir` to this run's `3_baseline_fresh` directory.
The native trace, sources, budget, repetitions, warmup and reset rules were unchanged.

Additional runtime SHA256, complementing source_sha256.json:
- single_type_kv_cache_manager.py: d65e88731dc7097c2526ad3c6a54ef6c836115becfa6d345f137fe486a57525a
- gpu_model_runner.py: 54ae117d980367f36a8f4229649d80a4c6d9de3a3be05f9cc86b53d9897f2180
- checkpoint_capture.py: 0ff27495323198129c0337fe531679df8b7193be9719703627d983c61fbdef35
- fused_recurrent.py: af9d0fd184d94bff1f6a5cd7b3a642f4d2e5a98249d054392daa53e78b6d5416

## Earlier automatic comparison against reused baseline (historical)

Additive tail checkpoint: True; single model forward: True. Session: WRAImOg_0. Three seed/resume repetitions per arm; 10 unrelated startup warmups; GPU reset between requests; CPU reset between repetitions.

Resume mean TTFT: native 125.043 ms; deep 165.703 ms; reduction -32.52%.

First generated token agrees; this does not establish full-sequence equivalence. One startup per arm: preliminary case, not a robust speedup claim. Baseline reused from /root/hyrex_results/coarse_single_case_20260928_v2/2_baseline (None means newly run). Native prefill budget=528; experimental=2048. Raw per-request data in online.jsonl.
