# Motivation: coarse recovery vs. one tail checkpoint

## Experiment

- Model: Qwen3.5-9B, BF16, eager.
- Workload: four real ShareGPT sessions, ten cumulative turns per session.
- Comparison: unmodified aligned recovery vs. the current prototype that replaces
  the last coarse state with one 16-token-aligned tail state
  (`--replace-tail-checkpoint`). This is an ablation, not the final retention policy.
- Protocol: ABBA (`native -> tail -> tail -> native`), warm-up before each service, and GPU cache reset after every request while retaining CPU cache.
- Pairing: the two native and two tail observations are averaged for every identical session/turn. First turns are excluded from recovery statistics.
- Correctness: all 160 first-token checks passed; no formal JIT warnings were reported.

## Main result

| Metric (36 continuation pairs) | Mean | Median | P95 |
|---|---:|---:|---:|
| Native full-model replay | 400.0 tok | 387.0 | 671.0 |
| Tail full-model replay | 132.5 tok | 108.5 | 270.0 |
| Full-model tokens avoided | 267.6 tok | 272.0 | 496.0 |
| Replay reduction | 63.3% | 66.5% | 89.6% |
| Native TTFT | 152.37 ms | 143.24 ms | 208.76 ms |
| Tail TTFT | 170.10 ms | 161.16 ms | 290.36 ms |
| Tail - native TTFT | +17.73 ms | +19.43 ms | +156.15 ms |

The coarse 528-token recovery boundary causes substantial replay: **400.0 tokens per
continuation on average**. One tail checkpoint avoids **267.6 full-model tokens
(63.3%)**, with positive token saving in 35/36 cases. However, the current
unconditional implementation is slower on average: **152.37 -> 170.10 ms
(+11.6%)**, and is faster in only 9/36 pairs.

Two paired examples make the distinction concrete:

| Prompt | Native hit/replay | Tail hit/replay | Avoided | Native -> tail TTFT |
|---:|---:|---:|---:|---:|
| 1,096 | 528 / 568 | 992 / 104 | 464 (81.7%) | 203.04 -> 134.43 ms |
| 830 | 528 / 302 | 720 / 110 | 192 (63.6%) | 134.21 -> 290.36 ms |

The first case realizes the expected speedup; the second is a measured outlier that
shows why token hit length alone is not a safe recovery objective.

## Per-turn averages

Turn 1 is the cold request and is intentionally omitted.

| Turn | Prompt tok | Native replay | Tail replay | Avoided | Reduction | Native TTFT | Tail TTFT | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 848.2 | 320.2 | 72.2 | 248.0 | 74.7% | 131.70 | 152.82 | +21.12 |
| 3 | 944.0 | 416.0 | 108.0 | 308.0 | 73.3% | 125.96 | 188.69 | +62.73 |
| 4 | 1053.8 | 525.8 | 117.8 | 408.0 | 78.5% | 167.54 | 148.34 | -19.20 |
| 5 | 1134.0 | 342.0 | 90.0 | 252.0 | 63.8% | 156.87 | 161.57 | +4.70 |
| 6 | 1229.2 | 305.2 | 105.2 | 200.0 | 55.7% | 141.24 | 158.26 | +17.02 |
| 7 | 1379.0 | 455.0 | 159.0 | 296.0 | 61.8% | 150.51 | 149.72 | -0.79 |
| 8 | 1569.2 | 513.2 | 197.2 | 316.0 | 65.6% | 180.05 | 214.22 | +34.18 |
| 9 | 1736.0 | 284.0 | 176.0 | 108.0 | 33.6% | 149.94 | 170.83 | +20.89 |
| 10 | 1890.8 | 438.8 | 166.8 | 272.0 | 62.4% | 167.50 | 186.44 | +18.94 |

## Break-even evidence

| Avoided full-model tokens | Samples | Mean avoided | Mean TTFT delta | Tail faster |
|---:|---:|---:|---:|---:|
| 0-127 | 7 | 61.7 | +18.34 | 0/7 |
| 128-255 | 9 | 188.4 | +27.65 | 1/9 |
| 256-383 | 10 | 304.0 | +27.42 | 1/10 |
| 384-inf | 10 | 446.4 | -1.31 | 7/10 |

The tail path is not profitable for every recovery. In this run, cases avoiding at
least 384 tokens are the only group with a lower mean TTFT, and 7/10 of those cases
are faster. This is evidence for a **cost-aware checkpoint policy**, not for adding
a tail checkpoint after every request.

## Measured cost context

| Cost item | Measured value | Interpretation |
|---|---:|---|
| One padded tail SSM checkpoint in CPU cache | 49.5 MiB/session; 198 MiB for four sessions | Intrinsic capacity and D2H cost |
| Earlier synchronous save barrier | 14.59 ms mean | Implementation cost; later fused path removes this explicit barrier |
| Fused direct-snapshot transient GPU storage | 99 MiB | Guard plus in-flight snapshot across 24 recurrent layers |
| Fine page-by-page Full-KV H2D | 18.753 ms for 33 MiB | Bad transfer organization |
| Coalesced Full-KV H2D | 1.853 ms for the same 33 MiB | Fine matching must use coarse DMA |

The proposed extra tail does not require loading two recurrent states during
recovery: the system selects and loads one state. Its intrinsic added costs are checkpoint
capture, one extra CPU-resident state, D2H at save time, lookup metadata, and any
deeper Full-KV transfer. Split forwards, repeated concatenation, scalar syncs, and
page-by-page DMA are avoidable implementation costs.

## Proposed design

1. Keep the existing 528-token coarse checkpoints and add **at most one replaceable
   tail state per active session**, aligned to the deepest reusable 16-token KV
   boundary.
2. Index KV and recurrent state independently and return `(L_KV, L_state)`. Recover
   from the state that minimizes predicted TTFT, rather than always choosing the
   deepest hit.
3. Capture the tail in the normal single forward, snapshot to an immutable buffer,
   and perform D2H asynchronously after the critical TTFT path. Match at 16-token
   granularity but coalesce H2D/D2H transfers.
4. Admit or retain a tail only when its expected saved compute exceeds capture,
   transfer, and capacity cost:

   `p_reuse * (T_forward(delta_tokens) - T_extra_KV_H2D) > T_capture + T_D2H + lambda * 49.5 MiB`.

5. For chat/agent workloads, prioritize turn ends, tool-result boundaries, branch
   roots, and rollback points. Do not checkpoint every intermediate token range.

This yields the paper's core motivation: **coarse state alignment wastes substantial
recomputation, but maximum cache hit is not equivalent to minimum TTFT; hybrid
recovery must jointly choose state placement, independent matches, transfer layout,
and the fastest valid recovery path.**
