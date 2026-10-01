# Fixed native baseline versus optimized deep KV, 2026-09-29

Existing run_coarse_validation.py --single-case, no tail-checkpoint flag.
GPU 1, Qwen3.5-9B, ShareGPT WRAImOg_0 first two turns (872/903 tokens),
three seed/resume repetitions per arm. Deep runs first, freshly started native
baseline second. Ten unrelated warmups per service, GPU prefix reset between
formal requests, CPU cache retained inside each pair and cleared between pairs.
CPU affinity 0-11,24-35. Both eager; native prefill budget 528, deep 2048.

Baseline source trees were clean at launch:
- vLLM 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
- LMCache 140819c9d57a975dbc5678a6459a218e544cb58b

This arm is deep KV plus state replay, NOT the additive-tail state variant.
Expected resume: KV864/state528, 375 whole-model forward tokens, with K/V
projection reuse for 336 additional prefix positions in Full-Attention layers.
Actual counters and outputs must be verified before reporting these expectations
as measured results. Tail-state optimization, if measured afterward, must be
reported separately and identify which fresh native result it reuses.

At launch global I/O PSI some/full avg10 was 0.61/0.41; pressure rose during
imports. No completed request measurements yet when this note was created.
Do not interpret loading time or host retrieve-submit logs as H2D DMA time.
One startup and three repetitions per arm give an exploratory case, not a
general performance claim. First-token agreement is not full-sequence validation.
