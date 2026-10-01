# Qwen3.5-9B: 16-session, concurrency-1 screening run

## Setup

- 16 independent real ShareGPT sessions, 5--10 turns each.
- 120 requests total per treatment; 104 are continuation/recovery requests.
- Requests are round-robin and sequential (`concurrency=1`).
- GPU prefix cache is reset after every request; the 8 GiB LMCache CPU cache is retained.
- One pass in order `native -> tail`; this is a screening run, not an ABBA significance result.
- Qwen3.5-9B BF16 eager on GPU 2; one generated token per measured request.

## Main result

| Metric (104 continuations) | Native 528 alignment | Tail state | Change |
|---|---:|---:|---:|
| Mean cached prefix | 842.8 tok | 1075.1 tok | +232.3 tok |
| Mean full-model replay | 391.5 tok | 159.2 tok | -232.3 tok (-59.3%) |
| Mean TTFT | 159.38 ms | 167.84 ms | +8.46 ms (+5.3%) |
| Median TTFT | 143.68 ms | 165.64 ms | +21.97 ms (+15.3%) |
| P95 TTFT | 216.74 ms | 189.45 ms | -27.29 ms (-12.6%) |
| Requests where tail is faster | -- | 27/104 | 26.0% |

The unconditional tail path substantially reduces replay and improves P95, but its
fixed implementation cost makes the median request slower.

## Break-even by avoided replay

| Full-model tokens avoided | Samples | Mean TTFT delta (tail-native) | Tail faster |
|---:|---:|---:|---:|
| 0--127 | 30 | +24.79 ms | 2/30 |
| 128--255 | 28 | +20.96 ms | 2/28 |
| 256--383 | 26 | -2.73 ms | 9/26 |
| 384--512 | 20 | -18.96 ms | 14/20 |

This run crosses over around 256 avoided tokens. The previous 4-session ABBA run
crossed over around 384 tokens, so a learned/measured cost model is preferable to a
fixed global threshold.

## Offline cost-aware policy on the paired observations

| Use tail only when avoided tokens >= | Tail selections | Mean TTFT | Median TTFT | P95 TTFT |
|---:|---:|---:|---:|---:|
| Never (native) | 0 | 159.38 | 143.68 | 216.74 |
| 192 | 60 | 157.82 | 157.75 | 188.41 |
| 256 | 46 | 155.05 | 153.81 | 189.45 |
| 320 | 32 | 153.08 | 147.43 | 211.22 |
| Always (tail) | 104 | 167.84 | 165.64 | 189.45 |

The 320-token rule lowers mean TTFT by 3.95% in the raw run. The baseline contains
one formal-request JIT outlier (450.44 ms); after excluding that contaminated pair,
the same rule lowers mean TTFT by 2.30% (156.56 -> 152.96 ms). These threshold rows
are an offline policy analysis over one observation per path, not a new online run.

## Correctness and measurement warnings

- Prompt token counts match for all 120 pairs.
- 119/120 first generated tokens match exactly.
- One pair (`rcgl3wR_0`, turn 4, 1309 prompt tokens) flips from a space to a double
  newline. In native, those two candidates have exactly equal reported logprob
  (-1.784638); in tail, newline leads space by 0.125. The Top-5 candidate set is the
  same. This is consistent with a BF16/chunk-order near-tie, but is retained as a
  correctness warning rather than declared a pass.
- After the warmup reset boundary, native reports one Triton JIT warning and tail
  reports zero. Therefore the raw native mean is slightly inflated; medians,
  percentiles, bins, and the sensitivity result above are all reported.
- Both treatments used the same 8 GiB LMCache setting and both fell back from SHM to
  pickle because this container has only 64 MiB `/dev/shm`.

## Interpretation

This larger workload confirms the mechanism but not an unconditional speedup:

1. Coarse 528-token state alignment causes hundreds of replay tokens per resumed
   request.
2. A tail state removes 59.3% of that work and reduces P95 TTFT.
3. Current tail capture/restore overhead dominates small and medium gaps.
4. The implementable policy is `coarse + one replaceable tail`, with independent
   KV/state lookup and a cost-aware path selector; use native recovery for small
   gaps and tail recovery for large gaps.

