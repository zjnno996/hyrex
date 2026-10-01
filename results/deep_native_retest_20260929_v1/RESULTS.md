# Single real-session case

## Verified native versus deep-KV replay result

| Repetition | Native seed ms | Deep seed ms | Native resume ms | Deep resume ms |
|---|---:|---:|---:|---:|
| 1 | 198.81 | 148.32 | 135.52 | 129.00 |
| 2 | 192.56 | 147.66 | 103.17 | 139.57 |
| 3 | 188.29 | 167.20 | 138.11 | 142.85 |
| Mean | 193.22 | 154.39 | 125.60 | 137.14 |

The deep-KV resume mean is 11.54 ms / 9.19% slower, not faster. Individual
repetitions vary substantially, especially native repetition 2; do not infer
statistical significance from three observations or discard the low value.

Both first turns: prompt 872, no cache hit, zero saved prefix tokens.
Each resume: prompt 903; native common KV/state hit 528, 375 forward tokens;
deep KV hit 864 and state hit 528, runtime SINGLE_FORWARD start=528 count=375.
Thus 336 additional token positions reuse Full-Attention K/V projections in
each of 8 Full-Attention layers (2688 token-layer projection positions), but
whole-model forward tokens saved = 0. Q, attention output, recurrent layers,
and MLP are not all skipped for those positions. The 864 cached_tokens API
field is NOT a claim that only 39 tokens underwent model forward.

All six first tokens agree; this is a smoke check, not full-sequence equivalence.
Each arm has 10 startup warmups and 5 successful formal GPU prefix resets.
Both eager, GPU 1, same CPU affinity; CPU cache retained within seed/resume
pairs and cleared between repetitions. Baseline source trees were clean before
and after measurement. No historical baseline reused for this comparison.

The runner's `single model forward: False` below is the --fused-tail flag,
not the actual deep-replay execution mode: VLLM_HYREX_SINGLE_FORWARD=1 was
enabled and all six deep forwards above were verified in runtime logs.
Native budget 528 versus deep 2048 also affects seed scheduling; seed latency
must not be presented as cache-recovery speedup. No pure H2D timing was captured.

## Automatic runner summary

Additive tail checkpoint: False; single model forward: False. Session: WRAImOg_0. Three seed/resume repetitions per arm; 10 unrelated startup warmups; GPU reset between requests; CPU reset between repetitions.

Resume mean TTFT: native 125.600 ms; deep 137.140 ms; reduction -9.19%.

First generated token agrees; this does not establish full-sequence equivalence. One startup per arm: preliminary case, not a robust speedup claim. Baseline reused from None (None means newly run). Native prefill budget=528; experimental=2048. Raw per-request data in online.jsonl.
