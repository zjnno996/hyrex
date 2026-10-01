# Hybrid recovery motivation review

## Decision

The current evidence supports an **architectural motivation** and a
**preliminary action-selection motivation**:

1. A Hybrid cache has two recovery semantics. FullAttention consumes
   token-indexed KV history, whereas Mamba/GDN can resume a continuation from
   its endpoint state.
2. Consequently, one CPU-only interval admits three safe recovery actions:
   `P2=Replay`, `P1=All Load`, and `P5=Full-history Load + terminal recurrent
   state Load`.
3. Historical static timings suggest different offline winners, but they are
   not yet cross-action correctness-cleared at multi-request scale. They are
   a hypothesis for the online Adaptive experiment, not its result.

The contribution should therefore be framed as **semantic- and cost-aware
recovery for tiered Hybrid-model prefix caches**, not as a claim that a Linear
layer can be skipped independently or that terminal-state transfer always
reduces TTFT.

## Novelty boundary against prior work

Marconi already addresses Hybrid prefix-cache admission and eviction: it
scores exact-match cache entries by estimated reuse and compute savings per
memory footprint.  It must therefore be treated as the baseline for any
claim about what to retain or evict, rather than as a system that only stores
Hybrid states.  Its paper does not describe CPU-tier offload, CPU-to-GPU
recovery, a load-versus-recompute choice, or recovery queue scheduling.

KVPR instead partitions a Transformer KV recovery into transferred and
recomputed regions so the two can overlap, and includes a profiler and a
scheduler for that execution plan.  A generic claim about scheduling transfer
and recomputation would therefore overlap KVPR.  The distinction to validate
is the Hybrid semantic asymmetry between token-indexed FullAttention history
and a recurrent endpoint state, combined with CPU/GPU tier residency and
multi-request contention for H2D and prefill resources.

The defensible system question is therefore a two-timescale policy:

1. On a CPU-tier Hybrid hit, choose a correctness-safe recovery action
   (`Replay`, `All Load`, or `terminal-state Load`) and schedule its H2D and
   prefill work against the current queues.  A KVPR-style partial
   Load+recompute action is a required baseline for this decision.
2. When capacity is scarce, retain or evict a tiered state according to its
   expected *future recovery value*--reuse probability times the cheapest
   queue-aware recovery cost--rather than Marconi's GPU exact-hit compute
   saving alone.

The first decision is the primary contribution.  The second is a valid
extension only after the experiment demonstrates that queue-aware recovery
value predicts future TTFT better than recency/FLOPs-per-byte.

## Evidence for the scheduling question

The present raw JSONL evidence is sufficient to motivate the question, but
not yet to claim an online scheduler improvement.  The historical offline
map below is a *performance hypothesis*: its old first-token/cache-state gate
is weaker than the current 16-token cross-action gate.

| Actual miss / concurrency | Safe winner | Evidence |
| --- | --- | --- |
| K=2, C=1 | P1 All Load | 169.864 ms versus P2 265.888 ms |
| K=2, C=8 | P1 All Load | 823.014 ms versus P2 37,279.698 ms |
| K=3, C=1 | P5 terminal state | 182.881 ms; P1 213.284 ms, P2 301.854 ms |
| K=3, C=4 | P2 Replay | 16,608.363 ms versus P1 17,411.515 ms across three P1/P2 repetitions |

The K=2 C1/C8 pair suggests the queue externality: P2's standalone
prefill is about 264--270 ms/request, but concurrent replay occupies the
prefill path long enough for median TTFT to reach 37.3 s at C8.  P1 has a
normal CPU-to-GPU recovery cost (about 36--97 ms measured H2D completion) and
keeps that cell below 1.1 s p99.  K=3 C1 then shows why the action cannot be
only Load versus Replay: recurrent endpoint recovery halves P1's 198 MiB
materialization to 99 MiB and becomes the fastest safe action.

The K=3 C4 P1/P2 reversal is deliberately weak evidence (about 4.8% median
gap); it establishes that the action map is not monotone in the small sample,
not a standalone crossover claim.  The next experiment must create a mixed
arrival workload and report offline-oracle regret, tail TTFT, H2D bytes, and
GPU/CPU cache residency.  That directly tests the missing question: when a
Hybrid CPU hit arrives under queue pressure, should it load, replay, or retain
its state for a later request?

## Latest cache-state control: logical 1K interval

The first arbitrary-length control separates logical request length from the
physical cache page.  It uses a 528-token GPU-resident shared prefix and a
1024-token logical suffix; the LMCache page size is 528 tokens.  With 64
eviction requests, all four P1 requests passed the actual-state gate:

| Quantity | Observed value |
| --- | ---: |
| GPU-resident prefix | 528 tokens |
| CPU-resident prefix | 1056 tokens |
| CPU-only recovery interval | 528 tokens (K=1) |
| P1 transferred state | 66 MiB/request |
| P1 TTFT p50 | 164.683 ms |
| P2 TTFT p50 | 155.203 ms |

This is a useful systems finding: a logical 1K continuation does **not** mean
that 1K tokens are loaded.  The first cacheable 528-token page is recovered
from CPU and the remaining 496 tokens are naturally prefetched/computed.  A
general planner must therefore record both logical length and physical hit
extent, rather than treating `L=1K` as a cache-state bucket.

The P2 trace was added specifically to validate that it is not a zero-prefix
control.  It reported `local_prefix=528` for every request and zero H2D, so it
is a genuine replay-from-local-prefix control.  Its two fresh-process runs
gave the identical first-token signature.  However P1's first-token signature
for the same four prompts differs on two requests.  Thus this cell currently
**fails cross-action numerical equivalence**.  Its timings are not evidence
that either action wins; the valid conclusion is that a Hybrid recovery action
needs a cross-policy logits/output gate, not merely an in-policy completion
check.  A P1 repeat with the same eviction recipe was also correctly rejected
because two requests became K=0 rather than K=1, illustrating why every row
must be filtered by actual cache state.

The controlled-reset rerun fixes that cache-state instability: it populates
CPU, resets only vLLM's local prefix hashes, restores the 528-token shared
prefix, and uses suffixes whose first physical 528-token pages are distinct.
It produced four out of four `local_gpu=528`, `cpu=1056`, and `need_h2d=528`
P1 requests, with all four Hybrid object groups transferred.  P2 reported four
out of four native `local_prefix=528` values and zero H2D.  The full 16-token
outputs still disagree for two request IDs (P1 signature `3054dcd7b49cd9a4`,
P2 `d73d3871c5169777`).  This is now a reproducible K=1 recovery correctness
diagnostic, not a TTFT result: P1's 166.650 ms and P2's 187.082 ms must not be
used as a speedup claim.

The same protocol has two longer, token-level multi-request controls.  They
show that multi-request isolation remains the implementation boundary:

| Cell | Cache-state gate | First-token gate | 16-token output gate | First divergent token |
| --- | --- | --- | --- | --- |
| K=2/C1 P1 versus P2 | pass (4/4, `need_h2d=1056`) | pass (`80f1de9cc34d7d21`) | fail (1/4 requests) | token 9 |
| K=3/C1 P5 versus P2 | pass (4/4, `need_h2d=1584`) | pass (`fd9af8c6805d5e37`) | fail (2/4 requests) | tokens 5 and 11 |

The K=2 P1 run transferred all four groups for two pages (132 MiB/request),
and the K=3 P5 run transferred 99 MiB/request: three FullAttention pages and
one terminal page per recurrent group.  Both therefore validate routing and
the intended state-size asymmetry.  They do **not** validate end-to-end state
semantics.  The next implementation task is to compare restored recurrent
boundary state against suffix replay at the first divergent decode step;
neither the K=2 263.486/279.123 ms P1/P2 pair nor the K=3 170.952/344.614 ms
P5/P2 pair may be used as a latency result until that succeeds.

### New single-request isolation result

The multi-request failures do not imply that an individual CPU-state recovery
is intrinsically invalid.  I re-ran P1 and P2 as separate fresh processes,
with one request, a 528-token retained GPU prefix, one distinct missing
528-token CPU page, `LMCACHE_MP_STRICT_LAYER_LOAD=1`, and serialized P1 H2D.
Both selected prompts (ShareGPT start indices 0 and 1; the latter was one of
the previous first-token mismatches) have byte-identical 16-token greedy
completions across P1 and P2:

| Prompt index | P1 TTFT | P2 TTFT | H2D state / correctness |
| ---: | ---: | ---: | --- |
| 0 | 428.139 ms | 336.436 ms | 4/4 Hybrid groups, 66 MiB; 16/16 tokens equal |
| 1 | 393.162 ms | 259.441 ms | 4/4 Hybrid groups, 66 MiB; 16/16 tokens equal |

These are deliberately **not** speedup measurements: startup-adjacent queue
time dominates a single sample.  They establish a narrower, important fact:
P1 is semantically recoverable for isolated requests, so the K=1/K=2/K=3
four-request mismatches should be investigated as batch/request-isolation or
cache-lifecycle behavior.  This gives the concurrency-aware policy a concrete
safety gate: before choosing Load under queue pressure, the runtime must only
batch recoveries whose state materialization is isolation-safe.

A four-request sequential P1 rerun with serialized H2D retained the old P1
completion signature (`3054dcd7b49cd9a4`), while a same-config P2 rerun
retained the P2 signature (`d73d3871c5169777`).  Thus serializing copies does
not remove the cross-action mismatch.  A follow-up physical-block trace showed
distinct destination IDs for the two recovered requests observed (`41--44`,
then `65--68`), so there was no direct destination-block alias in that trace.
However, that trace was correctly rejected because only 2/4 requests had the
required CPU-tier hit; the other two reported `cpu_tokens=528` and
`need_h2d_tokens=0`.  This confirms that CPU-tier population/visibility is an
independent experimental gate, not an outcome to average away.

## Verified Hybrid state asymmetry

The Qwen3.5-9B aligned layout has three Mamba/GDN groups and one
FullAttention group. The verified median materialization volumes per request
from correctness-gated rows are:

| Missing pages | P1 Full | P1 Mamba | P1 total | P5 Full | P5 Mamba | P5 total | P5 reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K=2 | 33 MiB | 99 MiB | 132 MiB | 33 MiB | 49.5 MiB | 82.5 MiB | 37.5% |
| K=3 | 49.5 MiB | 148.5 MiB | 198 MiB | 49.5 MiB | 49.5 MiB | 99 MiB | 50.0% |

P5 retains every missing FullAttention page but loads only one terminal page
per Mamba/GDN group. The reduction therefore comes from state semantics, not
from omitting the decoder computation for those layers. Every layer still
executes for the new suffix.

## Preliminary offline action map

The command below retains only rows that pass the runner's cache-state and
first-token gates, then defines a cell oracle as the policy with the lowest
median TTFT among the policies available in that exact protocol cell.

```bash
.venv/bin/python benchmarks/reproductions/analyze_hybrid_recovery.py \
  /root/qwen35_goal_p1p2_1568_c1_repeat2.jsonl \
  /root/qwen35_goal_p1p2_1568_c4_repeat2.jsonl \
  /root/qwen35_goal_p1p2_1568_c8_repeat2.jsonl \
  /root/qwen35_goal_p1p2_2112_c1_settle3_probe.jsonl \
  /root/qwen35_goal_p5_2112_c1_settle3.jsonl \
  /root/qwen35_goal_p1p2_2112_c4_repeat1.jsonl \
  /root/qwen35_goal_p1p2_2112_c4_repeat2.jsonl \
  /root/qwen35_goal_p1p2_2112_c4_repeat3.jsonl \
  /root/qwen35_goal_p1p2p5_2112_c4_settle3.jsonl \
  --architecture
```

Its current output contains these representative cells:

| Cell | P1 TTFT p50 | P2 TTFT p50 | P5 TTFT p50 | Offline winner | Meaning |
| --- | ---: | ---: | ---: | --- | --- |
| K=2, C=1 | 169.864 ms | 265.888 ms | unavailable | P1 | ordinary replay cost exceeds normal H2D recovery |
| K=2, C=8 | 823.014 ms | 37,279.698 ms | unavailable | P1 | Replay queue dominates under burst concurrency |
| K=3, C=1 | 206.429 ms | 318.893 ms | 191.963 ms | P5 | endpoint state halves materialization and wins |
| K=3, C=4 | 17,411.515 ms | 16,608.363 ms | excluded: output mismatch | P2 | P1/P2 can reverse under pressure |

These rows are intentionally labelled *preliminary*: they are collected by
independent services and not every P5 row has a matched normal-concurrency
P1/P2/P5 repetition. The new cross-action mismatch above additionally means
that this offline map is a performance hypothesis, not a correctness-cleared
Adaptive claim, until it is re-run with cross-policy output equality. The
K=3/C4 P1/P2 gap is only about 4.8% in the current raw files, so it is evidence
against a universal static rule, not a robust standalone speedup claim.

## What the final motivation must prove

## Single-layer context sweep: 1K--128K

The isolated pinned-memory microbenchmark was extended to one FullAttention
layer and one GDN layer at C=1.  The FullAttention payload is historical KV
and grows with context; the GDN payload is one endpoint state and stays at
2.047 MiB.  Values are p50 GPU-event times in milliseconds:

| Context | Full bytes | Full Load | Full recompute | GDN bytes | GDN Load | GDN recompute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 4.00 MiB | 0.220 | 0.459 | 2.047 MiB | 0.127 | 0.689 |
| 4K | 16.00 MiB | 0.848 | 1.065 | 2.047 MiB | 0.114 | 0.893 |
| 8K | 32.00 MiB | 1.746 | 1.757 | 2.047 MiB | 0.124 | 1.760 |
| 16K | 64.00 MiB | 3.627 | 3.493 | 2.047 MiB | 0.115 | 3.476 |
| 32K | 128.00 MiB | 6.763 | 7.026 | 2.047 MiB | 0.109 | 7.042 |
| 64K | 256.00 MiB | 14.325 | 13.914 | 2.047 MiB | 0.112 | 13.946 |
| 128K | 512.00 MiB | 28.310 | 28.106 | 2.047 MiB | 0.123 | 28.109 |

The result supports two separate design inputs.  First, FullAttention has a
context-dependent Load/Replay crossover around 8K--16K in this lower-bound
measurement; at 64K--128K the two costs are close again, so a fixed threshold
is unsafe without queue and implementation overheads.  Second, GDN endpoint
Load remains nearly constant while replay grows with context, giving a much
stronger Load preference as the recovered recurrent interval grows.  These
are layer-level costs, not end-to-end TTFT: actual decisions must add lookup,
H2D queueing, page alignment, and the full-block compute cost.

The Load column is a real local H2D measurement: the benchmark allocates a
page-locked CPU tensor, a CUDA destination tensor, performs
`device.copy_(host, non_blocking=True)`, and measures CUDA events.  The
independent bidirectional check reports matching CPU-to-GPU medians; for
example, Full@32K is 6.763 ms direct H2D versus 6.761 ms in the round-trip
check, and Full@128K is 28.310 versus 28.374 ms.  No artificial bandwidth
model or GPU-to-GPU copy is used in these rows.

### Concurrency effect on the layer cost

The same single-layer experiment was repeated with C=1/4/8.  The table shows
direct pinned H2D p50 and recompute p50 in milliseconds; the H2D column is an
independent CPU-to-GPU measurement from the bidirectional check.

| Type | Context | C=1 Load/Compute | C=4 Load/Compute | C=8 Load/Compute |
| --- | ---: | ---: | ---: | ---: |
| Full | 1K | 0.230/0.465 | 0.852/0.936 | 1.800/1.697 |
| Full | 8K | 1.756/1.766 | 7.159/6.750 | 14.197/13.384 |
| Full | 32K | 7.076/6.937 | 28.414/27.782 | 57.343/55.331 |
| GDN state | 1K | 0.127/0.455 | 0.480/0.932 | 0.949/1.698 |
| GDN state | 8K | 0.128/1.744 | 0.505/6.754 | 0.920/13.390 |
| GDN state | 32K | 0.133/6.909 | 0.469/27.772 | 0.942/55.300 |

This confirms that concurrency changes the decision boundary: at 1K and C=8,
Full replay is already slightly cheaper than its H2D, while GDN endpoint
state Load remains cheaper.  At 32K, Full replay is cheaper in this isolated
lower-bound test for all three concurrency levels.  The final serving cost
must additionally include one prefix-match/lookup cost per request,
H2D-queue waiting, and any remaining suffix computation.  Match cost is not a
per-layer multiplier: it is normally paid once per request, whereas state
transfer and queueing are paid for each selected group/descriptor.

The target is not lower bytes alone. The final claim must be:

> Different Hybrid recovery actions are optimal for different cache states and
> serving conditions; selecting between them reduces TTFT tail latency and
> SLO misses relative to any fixed action.

This requires the following minimal experiment, all using normal local H2D
(no artificial bandwidth throttling):

1. Freeze four anchor cells: K=2/C1, K=2/C8, K=3/C1, and K=3/C4.
2. For each cell, run P1/P2/P5 where P5 has passed correctness, three times
   with the same prompts, cache-state protocol, warmup count, and output
   check. Require *cross-policy* equality against suffix replay for the same
   request IDs; exclude an action when this gate fails.
3. Store per-request TTFT, not only p50/p99 aggregates. The offline oracle is
   the lowest median safe action for each request class.
4. Calibrate `T_load` from normal LMCache lookup, object wait, queue, and H2D
   completion; use raw PCIe bandwidth only as a physical lower bound.
   Calibrate `T_replay` from prompt prefill and scheduler backlog.
5. Run Adaptive over a mixed workload containing all four classes. Compare it
   against static P1, P2, and P5 using TTFT p50/p99, SLO miss rate, H2D bytes,
   action mix, correctness, and regret versus the offline oracle.

## Non-claims and safety boundary

- P5's H2D-byte reduction does not imply universal TTFT improvement.
- P3/P4 selective state-type runs are routing diagnostics, not valid
  layer-skipping baselines; Hybrid decoder layers remain data dependent.
- The current normal-concurrency K=3 P5 path is excluded where its output gate
  fails. Serialized P5 is a separate safe fallback, not a substitute for that
  missing result.
- The current evidence is Qwen3.5-9B aligned-layout evidence; it should not be
  generalized to all Hybrid architectures without another layout/model.
