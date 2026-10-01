# Single real-session case

Additive tail checkpoint: True. Session: WRAImOg_0. Three seed/resume repetitions per arm; 10 unrelated startup warmups; GPU reset between requests; CPU reset between repetitions.

Resume mean TTFT: native 125.043 ms; deep 227.237 ms; reduction -81.73%.

First generated token agrees; this does not establish full-sequence equivalence. One startup per arm: preliminary case, not a robust speedup claim. Baseline reused from /root/hyrex_results/coarse_single_case_20260928_v2/2_baseline (None means newly run). Native prefill budget=528; experimental=528. Raw per-request data in online.jsonl.

## Additive-tail result (not the previous deep-KV replay implementation)

| Repetition | Native seed ms | Tail seed ms | Native resume ms | Tail resume ms |
|---|---:|---:|---:|---:|
| 1 | 190.47 | 264.60 | 135.80 | 239.92 |
| 2 | 188.11 | 281.42 | 134.62 | 220.18 |
| 3 | 184.47 | 274.69 | 104.71 | 221.61 |
| Mean | 187.68 | 273.57 | 125.04 | 227.24 |

Native two-turn TTFT sum averages 312.73 ms; additive tail averages 500.81 ms.
Neither seed nor resume latency improves with this initial split-forward implementation.

Verified from runtime logs, for all three continuations:
- `TAIL_PROBE HIT state=864 full=864 base=528`: the original coarse state remains available.
- `TAIL_PROBE STEP start=864 count=32 end=896`, followed by `start=896 count=7 end=903`.
- `TAIL_PROBE STORE boundary=896` and `skip_new_states=False`: the new tail is maintained.
- Actual forward-token count falls from 375 to 39 (336 fewer, 89.6%). This time the
  reduction applies to all model layers, not just K/V projection.

The initial implementation explicitly splits the forward and synchronously saves
the tail before the running-state slot is overwritten. Resume save-barrier host
times are 16.106, 14.608 and 13.055 ms (mean 14.590 ms). This includes waiting
and synchronization; it is NOT a pure D2H GPU measurement. It cannot alone explain
the 102.193 ms mean TTFT gap. The two short forward invocations, extra transfer,
lookup and scheduling overhead require separate profiling before assigning the
remaining gap. Fewer token operations do not imply faster short matrix operations
or fewer per-layer kernel launches.

One additional tail SSM checkpoint is 49.5 MiB in the native padded CPU transport
layout. This adds three state objects, while keeping coarse checkpoints; it is not
an RSS measurement. No tail eviction policy beyond the existing cache policy and
pair reset was added. See EXPERIMENT.md for the implementation and scope.

First-token agreement holds for all six requests; probabilities differ. This
does not establish full-sequence correctness. GPU 1 was released after completion.
Boundary checks passed, including seed 528/864/872 and resume 896/903. The first
startup attempt failed a prefill-budget safety check; v2 uses 528 for both arms.

Next implementation target: independently owned snapshot storage, capturing the
tail inside a single forward and saving asynchronously, without overwriting an
in-flight snapshot. Do not remove the current barrier without solving buffer
ownership. This experiment validates the recovery boundary, not the performance
of that future fused implementation.
