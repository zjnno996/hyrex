# Single real-session case

## Measured results: new baseline, matched CPU affinity

| Repetition | Native seed ms | Optimized seed ms | Native resume ms | Optimized resume ms |
|---|---:|---:|---:|---:|
| 1 | 222.24 | 144.56 | 146.66 | 159.80 |
| 2 | 194.84 | 163.81 | 131.10 | 146.47 |
| 3 | 201.15 | 167.72 | 134.76 | 155.18 |
| Mean | 206.08 | 158.70 | 137.51 | 153.82 |

Resume remains 16.31 ms (11.86%) slower than the fresh native baseline.
Every native resume reports cache hit 528 (375 forward tokens); every optimized
resume reports KV/state hit 864 and SINGLE_FORWARD start=864 count=39. Thus
336 whole-model forward tokens (89.6%) are avoided. Next checkpoint S896 is
still maintained. The optimization does not skip future checkpoint maintenance.

The previous implementation measured 165.70 ms resume mean; the new 153.82 ms
is 11.89 ms / 7.17% lower. This historical sequential comparison is not a causal
ablation: it cannot assign all of that difference to scalar-readback removal.

Seed latency is also reported, but native prefill budget 528 versus experimental
2048 changes scheduling. Seed speedup is not evidence of recovery speedup.
Both arms now use the same CPU affinity through the same runner; no baseline
result was reused. Three repetitions with one startup per arm are preliminary.

Checks: both arms completed 10 unrelated warmups, 5 successful formal GPU
resets, and 6 first-token comparisons. Baseline source trees remained clean.
CPU cache is retained within each pair, cleared between pairs. Component tests
cover exact convolution histories, ownership and absence of scalar readback in
the helper; they do not establish full-model sequence equivalence or absence
of synchronization elsewhere. GPU 1 was released at completion.

Remaining candidates, not measured attribution: recurrent segmentation and
intermediate snapshot copies; additional Full-KV transfer/padded tail layout;
lookup and short-prefill launch overhead. No end-to-end timeline was captured
in this run, so the remaining 16.31 ms cannot yet be apportioned among them.

See EXPERIMENT.md for the code change and test protocol, source_sha256.json for
the frozen source manifest, and each arm's online.jsonl for raw measurements.

## Runner summary

Additive tail checkpoint: True; single model forward: True. Session: WRAImOg_0. Three seed/resume repetitions per arm; 10 unrelated startup warmups; GPU reset between requests; CPU reset between repetitions.

Resume mean TTFT: native 137.507 ms; deep 153.817 ms; reduction -11.86%.

First generated token agrees; this does not establish full-sequence equivalence. One startup per arm: preliminary case, not a robust speedup claim. Baseline reused from None (None means newly run). Native prefill budget=528; experimental=2048. Raw per-request data in online.jsonl.
