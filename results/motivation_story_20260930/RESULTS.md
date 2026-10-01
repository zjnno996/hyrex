# Paper-facing motivation evidence

## Boundary opportunity

Across 104 continuation requests from 16 real ShareGPT sessions, an exact
tail state advances the executable recovery boundary in
**102/104 requests
(98.1%)**. The advance is
**232.3 tokens on average**, with median
216.0, P95 480, and maximum
512 tokens.

This is executable-boundary opportunity, not proof that the unmodified baseline
physically retained the same fine-grained state.

## TTFT break-even

| Avoided full-model tokens | Requests | Mean avoided | Tail - Native TTFT | Tail faster |
|---:|---:|---:|---:|---:|
| 0-127 | 30 | 58.7 | +24.79 | 2/30 |
| 128-255 | 28 | 185.7 | +20.96 | 2/28 |
| 256-383 | 26 | 314.5 | -2.73 | 9/26 |
| 384-512 | 20 | 451.2 | -18.96 | 14/20 |

Exact recovery has conditional value: its fixed realization cost dominates short
gaps, while the 384--512-token group is faster on average. Therefore maximum hit
length is not a safe policy objective.

## Offline policy opportunity

A leave-one-session-out selector learns its token threshold on 15 sessions and
applies it to the held-out session. It chooses thresholds [336, 352],
uses Tail for 29/104 requests, and changes mean
TTFT from 159.38 to 153.05 ms
(-3.97%). After excluding the single known maximum
Native/JIT-contaminated observation, the change is
156.56 to 154.14 ms
(-1.54%).

This is an offline opportunity study, not an online HyRex speedup. It motivates a
runtime recovery planner and Native fallback.

## Safe paper claim

The evidence supports: coarse recovery leaves frequent executable-prefix
opportunities; unconditional exact recovery is not profitable; and recovery value
crosses a measurable break-even point. It does not yet establish the final HyRex
end-to-end gain or a capacity-aware policy.
