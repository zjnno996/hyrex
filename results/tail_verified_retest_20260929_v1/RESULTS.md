# Single real-session case

Additive tail checkpoint: True; single model forward: True. Session: WRAImOg_0. Three seed/resume repetitions per arm; 10 unrelated startup warmups; GPU reset between requests; CPU reset between repetitions.

Resume mean TTFT: native 125.600 ms; deep 155.860 ms; reduction -24.09%.

First generated token agrees; this does not establish full-sequence equivalence. One startup per arm: preliminary case, not a robust speedup claim. Baseline reused from /root/hyrex_results/deep_native_retest_20260929_v1/2_baseline (None means newly run). Native prefill budget=528; experimental=2048. Raw per-request data in online.jsonl.
