# Qwen3.5-9B hybrid recovery results

## Validated setup

- Qwen3.5-9B, BF16, one RTX 4090 (the goal-run used physical GPU 2),
  `gpu_memory_utilization=0.9`.
- LMCache local CPU tier (24 GiB), 528-token aligned hybrid page.
- Real ShareGPT token-id prompts: a 528-token GPU-resident shared prefix plus a
  CPU-resident unique suffix.
- P1 loads all state groups; P2 replays all; P3 loads FullAttention group 3 and
  replays Mamba groups 0--2; P4 does the reverse.

A recorded recovery cell is considered valid only when its intended cache state
is observed. P1/P3/P4 require actual LMCache H2D after eviction for suffixes of
at least one page; P2 must have no H2D. P3/P4 also require the suffix-only
replay marker.

## Correctness and state-selection check

For suffix=784, output=1, C1, four real requests, all policies produced the
same deterministic first-token signature: `e6ca34b4c7f2b072`.

| Policy | TTFT p50 (ms) | Actual H2D groups | Suffix-only replay |
| --- | ---: | --- | --- |
| P1 All Load | 124.376 | 0, 1, 2, 3 | no |
| P2 All Replay | 129.395 | none | no |
| P3 Full Load + Linear Replay | 180.426 | 3 only | yes (4/4) |
| P4 Full Replay + Linear Load | 185.058 | 0, 1, 2 only | yes (4/4) |

Thus the mixed policies are real state-selective recovery paths, not a
GPU-prefix hit or an unverified legacy full replay path.

The signature check is a low-concurrency correctness check. At C16, policies
use different batching/recovery schedules and their first-token signatures are
not identical; that difference needs a future per-token or logit-tolerance
check and is not used as correctness evidence.

## Missing-length effect (P1 vs P2, output=1, C1)

| CPU suffix tokens | P1 / P2 TTFT p50 (ms) | Observation |
| ---: | ---: | --- |
| 64 | 75.051 / 73.379 | No full page; Replay wins. |
| 256 | 88.291 / 72.418 | No full page; Replay wins. |
| 528 | 121.142 / 88.663 | One full page; Replay still wins at C1. |
| 784 | 110.239 / 128.336 | Actual H2D; Load wins. |
| 1568 | 100.435 / 145.557 | Two full pages; Load wins. |

The 64/256-token rows contain no actual H2D group transfer; the 528-token row
is the first row with actual H2D. This verifies that cache page alignment, not
only nominal missing-token count, changes the recovery cost.

## State-type and concurrency result (suffix=784, output=1)

| Effective concurrency | P1 | P2 | P3 | P4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 124.376 | 129.395 | 180.426 | 185.058 |
| 4 | 352.423 | 308.563 | 442.950 | 453.098 |
| 8 | 435.469 | 425.740 | 698.596 | 729.625 |
| 9 | -- | -- | 849.670 | -- |
| 12 | -- | -- | deferred/stalled | -- |
| 16 | 507.296 | 738.301 | deferred/stalled | 1157.135 |

All values are TTFT p50 in milliseconds. Static P3/P4 are currently slower:
they pay both a real H2D path and replay. P3 loads the one FullAttention group
and replays the three Mamba groups; it runs at C9 but twice stalled at C12/C16
with remaining requests Deferred after H2D. P4 loads the three Mamba groups
and replays FullAttention; it completes at C16. This is a policy-specific
scheduling/capacity effect, not yet a proven property of the attention
algorithms themselves.

The result disproves a fixed rule such as "Full always Load" or "Linear always
Replay" on this hardware. It motivates a state/segment-aware Adaptive policy
and a scheduler that accounts for recovery dependencies and capacity.

## Hybrid type × missing-length interaction

The following older cells requested C16 but contained eight requests, so their
effective concurrency is C8. Their recovery semantics are valid: P3 transfers
only group 3 and P4 transfers only groups 0--2.

| CPU suffix tokens | P3: Full Load + Mamba Replay | P4: Mamba Load + Full Replay |
| ---: | ---: | ---: |
| 528 | 394.072 | 415.549 |
| 784 | 695.494 | 695.358 |
| 1568 | 1051.137 | 993.184 |

The relative ordering flips as missing length grows: P3 is faster at one page,
the two policies tie near 784 tokens, and P4 is faster at two pages. Neither
mixed policy beats P1 in the corresponding runs, but the flip is direct
evidence that type alone is insufficient: the decision depends on type and
recoverable length together.

## Long-output E2E result (suffix=784, output=1024)

With four simultaneous requests (the old files were labelled C16 but had only
four requests), P1 is 1.301 s p50, versus P2 19.614 s, P3 17.474 s, and P4
18.066 s. Long decoding makes full replay strongly harmful to TTFT on the
measured system. At C1 the same ordering is much closer: P1 138.065 ms, P2
139.026 ms, P3 184.948 ms, P4 164.939 ms.

## Supported claim and remaining work

## Metric choice for the motivation experiment

TTFT is the primary latency metric because the recovery decision is paid before
the first generated token. Report TTFT p50 and p99 for the P1/P2/P5 oracle
comparison. End-to-end request latency and TPOT are secondary serving metrics:
they are useful for long outputs, but decode time can hide the recovery cost.
For P5, H2D bytes and GPU-capacity boundary are co-primary resource metrics,
because the validated benefit is currently bandwidth/capacity rather than a
clear TTFT reduction.

The runner now records the complete streamed completion per request in
`completion_texts`. Use `--require-full-output` in the offline analyzer for a
strict P1/P2(/P5) cell; the older rows only provide first-token evidence and
must not be described as full-output correctness.

## H2D timing interpretation

The `LMCache H2D batch complete` line is an end-to-end retrieve-future wait,
not a bare `cudaMemcpyAsync` duration. It includes LMCache worker scheduling,
prefetched CPU-object access, transfer-kernel enqueue/completion, and the
intentional strict barrier before the first model forward. In the recent C8
probe, one request and seven requests completed with `wait_ms=84.265` and
`166.084`, while the final `cuda_sync_ms` was only `0.276` and `0.073` ms.
The corresponding transferred volume was 132 MiB/request. A direct pinned
BF16 CPU-to-GPU microbenchmark on the same RTX 4090 measured about 20.0 GB/s
(64 MiB in about 3.1 ms and 256 MiB in about 12.5 ms). Therefore these LMCache
wait values should be reported as recovery-path latency, not as PCIe H2D
bandwidth. The strict barrier is useful for correctness, but its wait is a
TTFT cost and should be separated from the raw copy time in future profiling.

## Architecture-level state-size validation

The cache-state-valid JSONL rows were re-aggregated by actual missing pages and
object-group type. Values below are median transferred MiB per measured request.
P1/P2/P5 signature equivalence is checked separately by the oracle pass; a P5
row with a signature mismatch remains useful for transfer accounting but is
not a correctness or latency conclusion.

| Policy | Missing pages | MambaSpec groups | FullAttentionSpec group | Total |
| --- | ---: | ---: | ---: | ---: |
| P1 | 1 | 49.5 | 16.5 | 66.0 |
| P1 | 2 | 99.0 | 33.0 | 132.0 |
| P1 | 3 | 148.5 | 49.5 | 198.0 |
| P5 | 1 | 49.5 | 16.5 | 66.0 |
| P5 | 2 | 49.5 | 33.0 | 82.5 |
| P5 | 3 | 49.5 | 49.5 | 99.0 |

This is direct evidence for the architectural motivation of TSM: the three
Mamba/linear groups transfer one terminal page under P5, while the
FullAttention group still transfers all missing pages. It proves a state-size
and bandwidth difference, not by itself a TTFT improvement. The reproducible
command is:

```bash
.venv/bin/python benchmarks/reproductions/analyze_hybrid_recovery.py \
  --self-check --architecture /root/qwen35_goal_*.jsonl
```

Supported: Hybrid cache recovery should make a Load-or-Replay decision using
actual recoverable pages, missing length, state/segment type, and serving
pressure. Uniform All Load, All Replay, and fixed Full/Linear splitting are
all suboptimal in at least part of the measured space.

Not yet supported: that H2D queueing alone causes a universal Replay win; that
Linear Replay is intrinsically better than Linear Load; or that a runtime
Adaptive policy beats the static policies. LMCache does not yet expose reliable
per-request H2D queue/service time through the vLLM metrics used here.

Next implementation/evaluation step: collect lookup, H2D queue/service, and
replay-compute time per request; choose the lower estimated cost; compare that
Adaptive policy with P1--P4 and an offline oracle. Activation-assisted layer
skipping remains a separate extension because KV/state restoration alone does
not skip a decoder layer.

## Strict synchronized GPU0 validation (BF16, output=256)

After the asynchronous MP retrieve path produced different P1/P2 first-token
signatures, strict mode was strengthened to wait for all retrieve futures and
call `torch.cuda.synchronize()` immediately after `start_load_kv`. This is a
correctness baseline: it removes H2D/compute overlap, so its P1 latency should
not be interpreted as the final adaptive-policy performance.

The following cells used LMCache local CPU, SLRU (`protected_ratio=0.8`),
GPU-memory-utilization 0.9, Qwen3.5-9B BF16, suffix-only requests, and the
GPU0 device. Both 784-token rows are from the same strict configuration and
have identical first-token signatures.

| Suffix | Policy | Requests/C | TTFT p50 (ms) | TPOT p50 (ms) | First-token signature |
| ---: | --- | ---: | ---: | ---: | --- |
| 784 | P1 All Load | 8/1 | 126.802 | 29.613 | `c1c53cd11b84779a` |
| 784 | P2 All Replay | 8/1 | 143.945 | 30.447 | `c1c53cd11b84779a` |
| 784 | P1 All Load | 2/1 | 203.819 | 30.873 | `ee2a1e4daf221305` |
| 784 | P2 All Replay | 2/1 | 204.131 | 30.771 | `ee2a1e4daf221305` |
| 784 | P1 All Load | 2/2 | 352.923 | 33.894 | `ee2a1e4daf221305` |
| 784 | P2 All Replay | 2/2 | 4218.088 | 30.913 | `ee2a1e4daf221305` |
| 1056 | P1 All Load | 4/1 | 177.174 | 32.126 | `0d12b50abe639ce2` |
| 1056 | P2 All Replay | 4/1 | 97.198 | 31.102 | `a5b3519aaa6d27cf` |
| 784 | P1 All Load | 4/2 | 283.712 | 34.111 | `9038d882fa0d5bb2` |
| 784 | P2 All Replay | 4/2 | 7607.557 | 29.702 | `7968cf1b1c1b72b8` |

For suffix=784, P1 passed the strict materialization gate for all eight
requests: `local_gpu_tokens=528`, `cpu_tokens=1056`, and
`need_h2d_tokens=528`; groups 0/1/2 (`MambaSpec`) and group 3
(`FullAttentionSpec`) each transferred eight objects. P2 issued zero H2D
retrieves. Thus the measured P1 TTFT is about 11.9% lower than P2 while the
outputs are equal.

The smaller paired cells provide the accepted concurrency comparison. At C1,
P1 and P2 are effectively tied (203.819 vs. 204.131 ms p50) with identical
signatures. At C2, P1 remains correct and rises to 352.923 ms, while P2 rises
to 4218.088 ms; P1 is 91.6% lower in TTFT p50. The corresponding TPOT values
are 33.894 ms for P1 and 30.913 ms for P2, so the P1 advantage is specifically
the first-token/recovery path, not a faster decode kernel.

The new connector trace decomposes the accepted P1 cells. Lookup latency was
3.1--3.8 ms per request. Strict H2D completion was 19.6--80.8 ms per request
(the two-request median is about 50 ms), including only about 0.05--0.16 ms
of the final CUDA synchronization; all four state groups transferred. P2
issued no lookup and no H2D, so its C2 TTFT increase is replay/scheduling
cost. This is the first strict result showing a serving-pressure effect: a
policy that is approximately tied with Load at C1 becomes dramatically worse
under C2, even though the measured H2D service remains on the order of tens
of milliseconds.

The suffix=1056 pair is rejected: although P1 passed the recovery gate and
transferred all four state groups, its signature (`0d12...`) differs from P2
(`a5b3...`). The four-request C2 suffix=784 pair is also rejected for the
same reason (`9038...` versus `7968...`), despite P1 passing the gate and P2
issuing zero H2D retrieves. These rows are retained as debugging evidence,
not as performance claims. A suffix=256 attempt was rejected because the 528-token
aligned cache page was already locally available (`need_h2d_tokens=0`), so it
did not represent an actual CPU recovery. A suffix=1568 attempt was rejected
when the shared GPU prefix was evicted (`local_gpu_tokens=0`).

Therefore the strict synchronized comparisons currently accepted are
suffix=784 at C1 and C2 with two measured requests. The failed larger-batch
pairs expose two implementation requirements before claiming a broader
concurrency or missing-length crossover:
per-request/logit-level correctness must be stable under batched recovery, and
the benchmark must distinguish nominal suffix length from aligned cache-page
materialization.

Earlier C4 attempts (including a shifted sample window) had one first-token
mismatch and are retained only as debugging evidence. The later true-
concurrency C4 rerun with four requests passed the signature gate and is the
C4 result used in the extended sweep below. The intermittent earlier mismatch
still reinforces that the current batched replay/recovery path needs
per-request logit-level debugging before accepting every C8+ cell.

The strict C1 type-selective cells are also retained only as diagnostic data:
P3 measured 202.880 ms TTFT p50 and P4 measured 181.085 ms, but both produced
the same signature (`d895...`) that differs from the accepted P1/P2 signature
(`c1c53...`). Their traces do confirm the intended routing—P3 transferred
only FullAttention group 3, while P4 transferred only Mamba groups 0--2—and
both replayed the complementary groups. Until their output/logit mismatch is
resolved, these timings cannot support a claim that Full versus Linear loading
is independently beneficial.

## Extended true-concurrency sweep (GPU0, BF16, output=256)

This sweep uses the same number of measured requests as the target
concurrency (`requests=C`), so C1/C4/C8/C16 are not confounded by a long
serial tail. All cells use suffix=784, shared prefix=528, LMCache local CPU,
SLRU, and GPU-memory-utilization 0.9.

| Concurrency | P1 TTFT / TPOT p50 (ms) | P2 TTFT / TPOT p50 (ms) | P1/P2 signature gate | Status |
| ---: | ---: | ---: | --- | --- |
| 1 | 299.983 / 34.288 | 297.225 / 34.262 | pass: `8f9b4c14` | accepted; nearly tied |
| 4 | 368.811 / 31.844 | 13096.930 / 29.747 | pass: `7968cf1b` | accepted; Load wins |
| 8 | 892.799 / 36.315 | 27987.571 / 30.451 | fail: `39071769` vs `c1c53cd1` | rejected correctness |
| 16 | unavailable | 58966.811 / 30.652 | P1 prefix gate failed | capacity failure, not paired |

For the accepted C4 point, P1 transfers all four state groups and its strict
H2D completion is 34.4--45.9 ms, while P2 performs no H2D. P1 is about 35.5x
faster in TTFT p50. C8 shows the same qualitative latency growth, but its
signature mismatch means it is diagnostic only. At C16, even reducing the
eviction filler set from 16 to 8 could not preserve the 528-token GPU shared
prefix for all requests (`local_gpu_tokens=0` appeared); P2's 59-second result
is therefore a replay-only pressure datapoint, not a P1/P2 comparison.

The extended sweep strengthens the motivation for a pressure-aware policy but
also identifies two separate system limits: batched recovery correctness must
be fixed before accepting C8+, and GPU prefix capacity must be budgeted before
interpreting C16 as a CPU-recovery experiment.

## Corrected C8 rerun after LMCache port isolation (GPU3, BF16, output=256)

The runner previously assigned LMCache control ports from the fixed
`5555/8080 + concurrency` range. After an interrupted cell, a stale local
LMCache server could remain on that port and a new vLLM process could attach to
the wrong server. The runner now derives unique LMCache ports from the vLLM
port for each run. The following C8 pair was rerun after that fix with
`VLLM_FLOAT32_MATMUL_PRECISION=highest` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

| Concurrency | P1 TTFT / TPOT p50 (ms) | P2 TTFT / TPOT p50 (ms) | Signature gate | Status |
| ---: | ---: | ---: | --- | --- |
| 8 | 845.805 / 46.659 | 36118.219 / 39.015 | pass: `c1c53cd1` | accepted; Load wins |

P1 again passed the full materialization gate for all eight requests:
`local_gpu_tokens=528`, `cpu_tokens=1056`, `need_h2d_tokens=528`; all four
groups transferred eight objects. P2 issued no H2D retrieves. Their complete
first-token signatures and per-request first-token hashes matched. P1 is about
42.7x lower in TTFT p50 than P2. The earlier GPU0 C8 mismatch is superseded
and should not be used as a correctness or performance conclusion; it was run
before the port-isolation fix and is retained only as historical diagnostic
evidence.

## Clean no-explicit-eviction matrix progress (GPU3, BF16, output=256)

These cells use 72 natural warmup requests and a final shared-prefix touch;
they do not pass `--evict-gpu-cache`. Each row uses
`requests=concurrency`, LMCache local CPU 24 GiB, SLRU, and GPU-memory
utilization 0.9.

| Suffix | C | P1 TTFT p50 (ms) | P2 TTFT p50 (ms) | Correctness | State |
| ---: | ---: | ---: | ---: | --- | --- |
| 784 | 1 | 246.082 | 249.430 | pass | accepted |
| 784 | 4 | 490.764 | 15537.651 | pass | accepted; Load wins |
| 784 | 8 | 544.452 | 36683.376 | fail | diagnostic only |
| 1056 | 1 | 298.660 | 296.213 | pass | accepted; nearly tied |
| 1056 | 4 | 15850.447 | 15365.757 | fail | diagnostic only |
| 1056 | 8 | 29908.101 | 36636.060 | fail | diagnostic only |

All P1 rows above passed the intended GPU-prefix/CPU-state gate when they
were recorded: `local_gpu_tokens=528`, `cpu_tokens=1056`, and
`need_h2d_tokens=528`; all four groups transferred. The failed rows are not
used for final speedup claims until the high-concurrency numerical/correctness
variation is isolated. C16 at utilization 0.9 did not form a valid pair:
natural workload pressure reduced some requests to `local_gpu_tokens=0`, so it
was a GPU-prefix capacity failure rather than a Load-vs-Replay result.

## Clean state-type correctness check (GPU3, suffix=784, C1)

The latest no-explicit-eviction selective checks passed against the P1
signature `8f9b4c14031dae15`:

| Policy | H2D groups | P1/Px signature | Correctness |
| --- | --- | --- | --- |
| P3 Full Load + Linear Replay | group 3 FullAttentionSpec | `8f9b4c14` | pass |
| P4 Full Replay + Linear Load | groups 0/1/2 MambaSpec | `8f9b4c14` | pass |

The earlier P3/P4 mismatch rows were collected before the LMCache port
isolation fix and remain historical diagnostics. C4/C8 state-type performance
cells still need to be rerun after GPU3 is available.

## Selective recovery C4 rerun (GPU1, BF16, output=256)

This rerun uses the current isolated-port runner, LMCache local CPU (24 GiB),
SLRU, no explicit GPU eviction, 72 natural warmup requests, and four measured
requests at C4. Every measured request had `local_gpu_tokens=528`,
`cpu_tokens=1056`, and `need_h2d_tokens=528`.

| Policy | TTFT p50 (ms) | Actual H2D groups | Correctness | Status |
| --- | ---: | --- | --- | --- |
| P3 Full Load + Linear Replay | 16772.056 | FullAttention group 3 only | fail: `9038d882` vs accepted P1-C4 `7968cf1b` | diagnostic only |
| P4 Full Replay + Linear Load | 16190.244 | Mamba groups 0/1/2 only | pass: `7968cf1b` | accepted recovery semantics |

P3 skipped all three Mamba groups and replayed the suffix-only segment for all
four requests; P4 skipped FullAttention group 3 and likewise replayed the
suffix-only segment for all four requests. Thus routing is correct for both
policies. P3 nevertheless differs on one request's first token under C4,
whereas P4 matches the accepted P1-C4 signature. Neither P3 nor P4 timing is
used to infer an intrinsic Full-versus-Linear advantage until P3's batched
correctness issue is fixed. The connector's aggregate `h2d_cpu_to_gpu_bytes`
field remains zero on this LMCache-v1 path, so actual transferred/skipped group
traces, rather than that aggregate field, establish materialization.

### P3-C4 confirmation rerun (GPU0)

The same P3 cell was repeated on GPU0 with the same inputs and configuration.
It passed the complete first-token gate: `7968cf1b1c1b72b8`, matching P1-C4
and P4-C4 for all four requests. Its TTFT p50 was 16641.745 ms; all requests
again had a 528-token GPU prefix, a 1056-token CPU hit, and a 528-token
materialization interval. Only FullAttention group 3 transferred (eight trace
objects); Mamba groups 0--2 were skipped and suffix-only replay ran 4/4.

This supersedes the earlier GPU1 P3-C4 mismatch as a correctness conclusion:
the P3 path can be correct at C4, but correctness must remain a per-cell gate.
Both P3 and P4 C4 requests were serialized by Mamba align-mode scheduling
(`Running: 1`, remaining requests `Deferred`), which explains their roughly
16-second TTFT and prevents interpreting their close timing as a state-type
algorithm comparison.

## Selective recovery C8 (GPU0, BF16, output=256)

With the same suffix-only CPU-hit/GPU-prefix-miss construction, eight measured
requests were run at C8. Every request had `local_gpu_tokens=528`,
`cpu_tokens=1056`, and `need_h2d_tokens=528`.

| Policy | TTFT p50 / p90 (ms) | Actual H2D groups | P3/P4 first-token gate | Status |
| --- | ---: | --- | --- | --- |
| P3 Full Load + Linear Replay | 35990.866 / 63994.126 | FullAttention group 3 only | `d895863a` | routing + pairwise output pass |
| P4 Full Replay + Linear Load | 35642.339 / 63422.414 | Mamba groups 0/1/2 only | `d895863a` | routing + pairwise output pass |

The connector recorded only group 3 transfers for P3 (16 trace objects) and
only groups 0--2 for P4 (16 objects per group); each skipped the complementary
groups and replayed the suffix-only segment 8/8. P3 and P4 have identical
per-request first tokens, so this cell validates the selective routing and
their pairwise functional equivalence.

Their roughly 36-second p50s are not a Full-vs-Mamba cost crossover: align-mode
logs show recovery scheduling with one running request and the remainder
deferred, serializing the eight requests. Also, this cell's signature differs
from the earlier separately-run P1/P2-C8 corpus signature, so a same-cell P1
baseline is still required before using C8 for a final P1/P3/P4 correctness or
speed comparison.

## Complete four-policy C8 cell (GPU0, BF16, output=256)

The complete P1--P4 comparison was then rerun under one invocation with the
same workload construction (C8, 72 warmups, suffix 784, shared GPU prefix 528,
CPU hit 1056, materialization interval 528, LMCache local CPU 24 GiB/SLRU).

| Policy | TTFT p50 (ms) | Recovery trace |
| --- | ---: | --- |
| P1 All Load | 1006.423 | all Mamba groups and FullAttention transferred |
| P2 All Replay | 35841.869 | no CPU lookup or H2D retrieve |
| P3 Full Load + Linear Replay | 35821.825 | only FullAttention group 3 transferred; groups 0--2 skipped |
| P4 Full Replay + Linear Load | 36984.567 | only Mamba groups 0--2 transferred; group 3 skipped |

This is the current strongest C8 performance result: in this implementation,
all-load is 35.6x faster than all-replay and 35.6--36.8x faster than the two
selective policies. P3 is essentially tied with P2, while P4 is slightly
slower. The selective policies inherit Mamba align-mode scheduling that runs
one recovery request at a time; consequently this measures a current system
limitation, not a general statement that Full and Mamba states have equal
materialization cost.

All requests used greedy decoding (`temperature=0`) with a fixed seed.
Nevertheless P1/P2/P3/P4 produced four different first-token signatures in
their independent vLLM services; crucially P2 has no prefix hit, lookup, or
H2D at all. Thus the C8 cross-service signature variation cannot be assigned
to selective recovery alone. Treat C8 route/transfer validation as passed, and
treat bitwise output equality at C8 as an unresolved experimental
non-determinism issue in the Qwen GDN `align` path. C1 is useful for small,
fixed-prompt functional checks, but a logits-based tolerance check or
deterministic kernel configuration is still required for a general correctness
claim.

## Page-boundary crossover probe (GPU2, C1, BF16, output=256)

The Hybrid model enforces a common 528-token page for Mamba and FullAttention.
An initial `shared=528, suffix=528` cell was deliberately rejected by the
runner: it had `local_gpu_tokens=528`, `cpu_tokens=528`, and `need_h2d=0`.
The terminal page is not a recoverable cache block, so a logical request ending
exactly at a page boundary does not create a second materializable page.

Increasing the suffix by exactly one token (`shared=528, suffix=529`) produced
the first valid two-page recovery request: `local_gpu_tokens=528`,
`cpu_tokens=1056`, `need_h2d_tokens=528`. Both policies have the identical
first-token signature `fd9cdd05c75e533d`.

| Policy | TTFT p50 (ms) | Recovery evidence |
| --- | ---: | --- |
| P1 All Load | 252.610 | all four groups loaded; LMCache H2D total 76.867 ms |
| P2 All Replay | 219.345 | no lookup and no H2D |

This single-request probe is a diagnostic, not a final crossover claim. It
shows that at one recoverable page the Load--Replay gap can be small enough
that input-specific and run-to-run effects matter. The meaningful decision
granularity is therefore the *recoverable page or continuous segment*, not the
logical suffix length alone.

### C1x4 replication of the 529-token cell (GPU2)

The same configuration was repeated with four independent ShareGPT suffixes,
issued serially (C1) and summarized by p50. All four P1 requests passed the
CPU-hit/GPU-prefix-miss gate: `local_gpu_tokens=528`, `cpu_tokens=1056`,
`need_h2d_tokens=528`; each transferred all four state groups. P2 issued no
lookup or H2D.

| Policy | TTFT p50 (ms) | Difference |
| --- | ---: | ---: |
| P1 All Load | 135.821 | 18.5% lower than P2 |
| P2 All Replay | 166.677 | baseline |

The larger sample reverses the single-request ordering: all-load wins by
30.856 ms at the smallest valid materialization page. Thus the current local
pinned-memory setup has **not** established a stable C1 length crossover.
What it does establish is a narrow decision region: the one-page gap is tens
of milliseconds, while C8 replay/selective recovery costs tens of seconds due
to serial align-mode scheduling. The first token matched for three of four
suffixes; one suffix differed across independent services, so this cell is
valid for transfer semantics and latency but remains unsuitable for a bitwise
correctness claim.

### State-type comparison at the one-page boundary (GPU2, C1x4)

P3/P4 were run with exactly the same four suffixes and configuration as the
preceding P1/P2 C1x4 cell. Every request again had a 528-token local GPU
prefix, a 1056-token CPU hit, and one 528-token materialization interval.

| Policy | TTFT p50 (ms) | Actual materialization | First-token relation |
| --- | ---: | --- | --- |
| P1 All Load | 135.821 | Full + all Mamba groups | P3-equivalent |
| P2 All Replay | 166.677 | no CPU lookup/H2D | P4-equivalent |
| P3 Full Load + Linear Replay | 177.403 | Full group 3 only; Mamba 0--2 skipped | exactly matches P1 map |
| P4 Full Replay + Linear Load | 184.741 | Mamba 0--2 only; Full group 3 skipped | exactly matches P2 map |

P3 and P4 therefore validate type-selective routing and preserve the output
behavior of the corresponding recovered state: loading FullAttention (P3)
matches P1, while replaying FullAttention (P4) matches P2. But neither mixed
policy wins in this implementation: P3 is 6.4% slower than P2, and P4 is
10.8% slower than P2. At C1 the reason is not H2D bandwidth; LMCache H2D
batches take only 6--69 ms. The remaining penalty is the present
suffix-only/align recovery scheduling and materialization path. This is a
useful negative result: state-aware policy must optimize *valid continuous
recovery segments and scheduler behavior*, not merely transfer fewer state
groups.

## Long-context and higher-concurrency probe (GPU2, BF16, output=256)

We next increased the logical suffix to 1568 tokens.  The final cacheable
prefix was 1584 tokens (three 528-token Hybrid pages): every valid request
kept the first 528 tokens on GPU, found 1584 tokens in the local-CPU LMCache,
and therefore had a 1056-token CPU-to-GPU recovery interval.  This is a
substantially longer materialization segment than the one-page probe above.

At C8, all eight P1 and Adaptive requests passed that precondition.  P1's
eight H2D retrieves were batched as 1 and 7 requests, with LMCache H2D totals
of 81.286 and 293.346 ms.  The corresponding Adaptive batches were 1 and 7,
with totals of 65.310 and 159.855 ms.  The three policies produced the same
first-token signature (`bd11450b1323d04e`).

| Policy | Decision | TTFT p50 (ms) | TTFT p90 (ms) | Recovery evidence |
| --- | --- | ---: | ---: | --- |
| P1 All Load | Load 1056 tokens | 1726.986 | 2314.167 | 8/8 CPU hits and all four groups retrieved |
| P2 All Replay | Replay | 32848.662 | 62116.313 | no lookup or H2D |
| Adaptive (threshold=528) | Load 1056 tokens | 793.722 | 992.353 | 8/8 CPU hits; no adaptive replay event |

The current Adaptive prototype is deliberately minimal: it chooses replay
only when the measured missing interval is no more than the configured
threshold.  Thus the same 528-token threshold that chose replay for the
one-page C1 smoke test chooses load for this 1056-token C8 interval.  It does
not yet use H2D queue pressure, so this is a functional missing-length-aware
baseline rather than the final contention-aware policy.  Its lower observed
TTFT than P1 comes from independent-service variance and should not be read
as an adaptive optimization gain; the reliable result is the verified branch
selection and the large long-segment gap between load and replay.

We also tested the capacity boundary at the same length.  C10 was valid:
10/10 requests retained the GPU 528-token prefix, had a CPU hit of 1584
tokens, retrieved 1056 tokens, and P1 achieved 937.208 ms p50 TTFT.  In
contrast, C12 had 2/12 requests and C16 had 4/16 requests with
`local_gpu_tokens=0`; those requests would recover the entire 1584-token
prefix rather than the intended CPU-suffix interval.  The runner rejected
both cells, correctly keeping them out of the comparable recovery matrix.
On this single GPU at 0.9 memory utilization, the valid C8--C10 range is
therefore the present high-concurrency regime for this 1568-token experiment;
C12+ requires either more GPU cache capacity or a separate full-CPU-recovery
experiment.

### Long-context Adaptive replay branch after tracing fix

With the threshold raised to 1056, the same C8 workload took the replay branch
for all 8 requests.  The post-fix trace contains exactly 8 prefix matches and
8 adaptive decisions, with zero H2D retrieves; the first-token signature again
matched P2.  TTFT p50 was 34600.887 ms (p90 64215.811 ms), consistent with
replay rather than load.  One request had an actual 528-token missing interval
after prompt/page alignment, while the other seven had 1056 tokens; all were
correctly classified by the measured interval.  This confirms both branches
of the prototype at C8 and exposes why the controller must use the actual
cacheable missing segment, not only the nominal requested suffix length.

The Adaptive lookup/replay path also received a small correctness-of-tracing
fix: once a request has selected replay, later scheduler polls no longer redo
the same lookup or append duplicate hit/decision records.  The C1 and C8
regressions passed after this change.

## Contention-proxy probe (GPU2, long context, C8)

To exercise the contention dimension, we disabled the length trigger
(`threshold=0`) and enabled a scheduler-side in-flight proxy:
Adaptive selects replay when at least one *other* request is already in
`WAITING_FOR_LOAD`.  This is a proxy for H2D pressure, not the worker's exact
CUDA queue depth.  The workload was the same 1568-token suffix and 1056-token
recoverable interval as above.

| Policy | TTFT p50 (ms) | H2D/replay behavior | First-token signature |
| --- | ---: | --- | --- |
| P1 All Load | 971.745 | 8 H2D retrieves, batched 1+7 | `bd11450b1323d04e` |
| P2 All Replay | 33726.039 | no lookup/H2D | `bd11450b1323d04e` |
| Adaptive, max in-flight=1 | 17271.518 | 5 loads + 3 contention-triggered replays | `bd11450b1323d04e` |

The three replay decisions all observed `pending_h2d=1`; the other five
requests loaded.  This proves the contention branch and mixed execution path
under C8.  It is also an important negative result: on this hardware the
observed P1 H2D batches completed in about 19--82 ms, while replaying a
1056-token segment is tens of seconds.  A queue-only threshold therefore
caused a large regression (Adaptive was 17.8x slower than P1), even though it
reduced H2D requests.  The final policy must compare estimated remaining
`T_load = lookup + queue + H2D` against segment replay cost, using queue
pressure as one input rather than as a standalone override.

### Length-protected contention policy: short-segment C8

The first contention probe used an unbounded queue override.  The controller
was then tightened so that contention may select replay only for a missing
interval no larger than a separate `contention_replay_below_tokens` cap.  This
is the minimal threshold implementation of the cost-model observation above:
queue pressure cannot force a known-expensive long segment to replay.  We set
that cap to one Hybrid page (528 tokens) and tested `shared=528, suffix=529`,
where each valid recovery interval is one 528-token page.

| Policy | TTFT p50 (ms) | H2D/replay behavior |
| --- | ---: | --- |
| P1 All Load | 4640.382 | 8 loads; two H2D batches totaling 73.550 and 124.800 ms |
| P2 All Replay | 38200.075 | no lookup/H2D |
| Adaptive, in-flight=1, cap=528 | 6343.369 | 5 loads + 3 contention-triggered replays |

The three Adaptive decisions again all saw `pending_h2d=1`, and all had a
528-token interval, validating the new length guard and the desired mixed
behavior.  Adaptive is still slower than all-load in this run, while much
faster than all-replay.  This reinforces the experimental conclusion: local
pinned-CPU H2D is cheap even under C8, so replay must not be used merely
because a small queue exists.  The C8 independent-service first-token
signatures differed between P1 and P2; Adaptive matched P2.  As documented
above, C8 cross-service GDN align-mode output signatures remain a
non-determinism issue, so this cell establishes routing and latency semantics,
not bitwise correctness across services.

## State-size measurement and the next design direction

### Why the aligned Hybrid page is 528 tokens

The 528-token granularity is not an arbitrary Linear-Attention restriction.
In vLLM `mamba_cache_mode=align`, one Mamba/GDN state page must fit in the
same byte-sized page as the corresponding Full-Attention KV group; the runtime
then rounds up for kernel alignment.  The Qwen3.5-9B configuration makes this
constraint directly observable:

| Quantity | Measured from Qwen3.5-9B config |
| --- | ---: |
| One GDN layer's conv + recurrent state | 2,146,304 B (2.047 MiB) |
| One 8-layer GDN group state | 17,170,432 B (16.375 MiB) |
| All 8 Full-Attention layers' KV per token | 32,768 B (32 KiB) |
| Minimum equivalent Full-KV tokens | 524 |
| Aligned page selected by the runtime | 528 tokens / 17,301,504 B |

A 512-token Full-KV page is 393,216 B too small for the 8-layer GDN state;
528 tokens is the first aligned size that fits, with only 131,072 B padding.
Thus a smaller global page is possible only if the runtime changes the Hybrid
grouping or permits a Mamba state object to span multiple Full-KV pages.  It
is not a limitation shared by all Linear-Attention models: a standalone
recurrent model can checkpoint state at any chosen interval, whereas this
Hybrid `align` layout chooses a common page boundary for Full KV and GDN state.

### Isolated state-type microbenchmark (GPU2)

To separate state geometry from the stalled end-to-end model-load environment,
we measured local pinned H2D against GPU replay kernels without loading model
weights.  Full replay covers Q/K/V plus causal attention for all eight Full
layers. Linear replay covers all 24 GDN recurrences but deliberately excludes
MLP, norm, residual, lookup, and scheduling; it is therefore favorable to
Replay. Linear transfer is the minimal 24-layer endpoint-state representation
(49.12 MiB), rather than the current aligned-page materialization.

| Recoverable segment | C | Full: Load / Replay (ms) | Linear: Load / Replay (ms) |
| ---: | ---: | ---: | ---: |
| 528 | 1 | 0.784 / 2.603 | 2.418 / 9.420 |
| 528 | 4 | 3.125 / 7.429 | 9.296 / 9.642 |
| 528 | 8 | 6.346 / 14.202 | 19.585 / 20.119 |
| 1056 | 1 | 1.652 / 4.594 | 2.308 / 10.024 |
| 1056 | 4 | 6.958 / 14.892 | 10.108 / 19.734 |
| 1056 | 8 | 15.378 / 28.911 | 22.781 / 38.765 |

The 528-token Linear C8 point is a near crossover even under this optimistic
replay microbenchmark (19.585 vs. 20.119 ms).  At 1056 tokens, Linear replay
grows with segment length while endpoint-state transfer remains nearly fixed,
and Load regains a 1.70--4.34x advantage.  This is not an end-to-end TTFT
claim: it omits LMCache lookup/queueing and the current align-mode scheduler.
It is direct evidence that the right decision inputs are state type, actual
segment length, and transfer pressure rather than a fixed Full/Linear rule.

We added real LMCache worker-side byte tracing and measured one verified C1
recovery with GPU prefix 528, CPU hit 1584, and a 1056-token H2D interval.
Each of the three Mamba/GDN object groups and the FullAttention object group
transferred exactly 34,603,008 bytes (33 MiB), for 132 MiB total.  Qwen3.5-9B
therefore does **not** exhibit the simplistic "Full is large, Linear is
small" split at this grouping granularity.

The model configuration instead reveals eight interleaved
`Linear x3 -> FullAttention x1` decoder tiles.  This, together with the
current connector's global strict H2D barrier, motivates the next pure-KV
design: restore Hybrid states in execution-order tile bundles and overlap the
H2D of tile `i+1` with suffix computation in tile `i`.  It preserves the
required inter-layer dependencies and avoids the invalid claim that loading
one state type permits arbitrary layer skipping.  The detailed proposal and
correctness-safe evaluation plan are in
`HYBRID_STATE_STREAMING_IDEA.md`.

### EOSS prototype: execution-order structural configuration runs correctly

The four-layer execution-tile layout was then exercised through the real
LMCache local-CPU P1 path on the same valid cell (`GPU=528`, `CPU=1584`,
`need_h2d=1056`, C1).  LMCache registered eight object groups, one for each
`Linear x3 -> FullAttention` tile.  The runtime transferred all eight groups;
each contained two state objects and measured 17,301,504 bytes (16.5 MiB).
The total remained 138,412,032 bytes (132 MiB), exactly matching the original
four-group state-family layout, while the first-token signature remained
`f194f88525b7a18f`.

This is a positive structural-correctness result: an execution-order
configuration can preserve the intended cache-hit condition, transfer volume,
and observed output. It is not proof that baseline state-family CPU objects
can be re-materialized as tiles, because later metadata inspection showed that
the prototype also changes the physical kernel/page groups. It is deliberately
not reported as an EOSS latency result. The correct next mechanism is
cache-plane layer slices plus a completion event per tile and a
correctness-safe wait only at that tile's first decoder layer.

### Superseded structural reconfiguration probe (not a layout-only ablation)

We then repeated the exact one-page suffix-only cell with the same BF16,
0.9 GPU-utilization, 24-GiB local-CPU SLRU configuration, eight requests,
and 256 output tokens. Both configurations had eight valid recoveries with
`local_gpu=528`, `cpu=1056`, and `need_h2d=528`; their first-token signature
was identical (`dcca7ae1da9dc115`).

| Layout | H2D object groups | TTFT p50 / p90 (ms) |
| --- | ---: | ---: |
| State-family configuration | 4 kernel groups of 8 layers | 114.326 / 155.016 |
| Current execution-tile configuration | 8 kernel groups of 4 layers | 130.745 / 180.314 |

Post-run worker metadata inspection revealed that this is **not** an
object-layout-only comparison: setting `execution_tile_size=4` also split the
underlying kernel/page groups from four contiguous 8-layer groups to eight
4-layer groups. It therefore changes cache tensor geometry, transfer-kernel
launch structure, and scheduler-visible engine groups. The 14.4% C1 slowdown
is a valid structural observation, but it cannot be assigned solely to object
count or CPU-object scheduling.

One valid C8 structural sample was likewise 540.181 ms p50 for the baseline
and 2084.138 ms for the altered 8-kernel-group configuration. Both recovered
all eight requests and transferred 528 MiB, but this sample has the same
confound and is not a causal EOSS result. A second altered-configuration C8
attempt completed on the server but did not yield a complete client result
row, so it is not used as a statistical replicate.

The stronger data-supported design constraint is: **recovery-object
granularity must be decoupled from the engine's physical KV kernel-group
layout.** A Hybrid cache hit becomes useful only when its state is materialized
at the consumer's execution-ready granularity, but that cache-plane choice
must not silently change scheduler-visible page geometry. The adaptive choice
should consequently be between complete *materialization plans* (coarse
barrier load, future execution-order streaming, or replay), using missing
length and contention, rather than independently choosing Load/Replay for
Full and Linear state families.

### Unconfounded cache-plane LayerSlice recovery (verified)

We implemented a separate cache-plane-only switch (`hybrid_cache_slice_size=4`)
that preserves the baseline four physical vLLM kernel/page groups. In a real
P1 C1 suffix-only recovery (`GPU=528`, `CPU=1056`, `need_h2d=528`, 256 output
tokens), LMCache registered four physical groups but eight cache objects:
`[[0], [0], [1], [1], [2], [2], [3], [3]]`. Each slice copied 8,650,752 B;
all eight together copied 69,206,016 B. The unmodified baseline copied four
17,301,504-B objects, the same 69,206,016-B total. Both runs produced the
same first-token signature, `9cb3ee3a671b7e05`.

This is the first valid proof that an execution-order cache layout can be
materialized without changing engine page geometry. It is a correctness and
abstraction result, not a latency claim: current strict loading still waits
for all objects, and both independent runs included first-use Triton JIT.
The next experiment must add per-tile completion events plus a Mamba/GDN
consumer-side wait before measuring EOSS TTFT.

### EOSS execution path status (not yet a performance result)

The per-tile retrieval protocol and Qwen Mamba/GDN layer-entry readiness hook
are now implemented behind `--hybrid-eoss`, which requires the unconfounded
`--hybrid-cache-slice-size 4` layout and currently permits P1 only.  It
uses `--hybrid-eoss-window w` (default two): it initially submits only the
first `w` tiles and each tile entry submits the next tile before waiting for
its own state. vLLM's pre-model load transition waits only for that initial
window; all later readiness is owned by the layer-entry hook. This avoids a
deadlock in which the scheduler waits for tiles that cannot be submitted until
the model begins execution. Unit checks cover layer-to-tile mapping, window
advancement, and this scheduler-boundary semantics.

After switching checkpoint loading to `auto`, a clean EOSS C1 run completed
on GPU2. It had `local_gpu=528`, `cpu=1056`, `need_h2d=528`, SLRU local CPU,
eight 8,650,752-B tile transfers, and all eight `EOSS tile ready` entries.
Its first-token signature was `9cb3ee3a671b7e05`, identical to the strict
barrier LayerSlice P1 control under the same recovery precondition. This is
the first end-to-end correctness validation of the bounded (`w=2`) EOSS path.

The corresponding one-shot TTFTs were 285.949 ms (EOSS) and 628.529 ms
(barrier). They are **not** a publishable speedup: both independent processes
logged first-use Triton JIT, and the barrier cell's 26.808-ms strict H2D wait
does not account for the 342.580-ms gap. They establish functional behavior,
not a causal latency result. Reuse a warmed server or collect repetitions
before reporting performance.

We then varied the bounded prefetch window in separate C1 processes. All
three EOSS windows transferred the same eight 8,650,752-B tile objects and
produced the same signature; the window changes scheduling granularity, not
the recovery contents.

| Recovery schedule | Window | TTFT p50 (ms) | TPOT p50 (ms) | All 8 tiles ready | Signature |
| --- | ---: | ---: | ---: | :---: | --- |
| EOSS | 1 | 1395.188 | 35.279 | yes | `9cb3ee3a671b7e05` |
| EOSS | 2 | 285.949 | 36.039 | yes | `9cb3ee3a671b7e05` |
| EOSS | 4 | 286.862 | 35.491 | yes | `9cb3ee3a671b7e05` |
| Strict barrier | — | 628.529 | 36.907 | yes | `9cb3ee3a671b7e05` |

The large C1 difference between `w=1` and `w=2/4` is a useful motivating
signal: fully just-in-time per-tile retrieval can expose per-tile scheduling
overhead, while a small bounded window avoids it. It is not yet a controlled
latency claim because these are one-shot independent processes with
first-use kernel JIT and model-startup effects. The correct follow-up is a
warmed server with repeated requests at C1/C4/C8, plus explicit aggregate
per-tile lookup, queue, H2D service, and layer-compute timings. Note that the
generic aggregate H2D byte/time fields are empty for EOSS; its eight tile
transfers are recorded in `lmcache_eoss_*` and
`lmcache_actual_h2d_object_group_bytes`.

To reduce the startup confound, we repeated the comparison in one service
process with two warmup requests and four measured C1 requests. Because the
runner retains the shared GPU prefix, two measured requests were genuine
CPU-suffix recoveries and two were already GPU-resident hits; both counts are
reported explicitly below.

| EOSS window | TTFT p50 (ms) | TTFT p90 (ms) | H2D recoveries | GPU-only hits | Tiles ready |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 174.479 | 248.814 | 2 | 2 | 8/8 per recovery |
| 2 | 194.005 | 267.007 | 2 | 2 | 8/8 per recovery |
| 4 | 171.661 | 229.478 | 2 | 2 | 8/8 per recovery |

These warm C1 numbers do **not** show a stable window-size advantage: all
three p50 values are within the noise of a small, mixed hit set. They do,
however, validate the measurement protocol and show why the earlier 1,395-ms
one-shot `w=1` number must not be interpreted as per-tile recovery cost. The
next useful performance experiment is repeated C4/C8 with all requests forced
to have a CPU suffix hit, followed by a controlled H2D-bandwidth/queue sweep.

### C4 barrier recovery: concurrent H2D becomes visible

Using four warmup requests to populate the four measured suffixes, followed
by a small 16-request GPU-cache pressure phase, the C4 P1 barrier cell
satisfied the full recovery precondition for every request:
`local_gpu=528`, `cpu=1056`, and `need_h2d=528`. The result was
`TTFT p50=14,927.073 ms`, `p90=26,236.087 ms`, with `TPOT p50=36.363 ms`.
The connector recorded two H2D batches with waits of 25.560 ms and 50.680 ms
(`total_ms=26.510` and `51.477`), and transferred all eight cache-plane tile
objects for all four requests.

This is a valid motivating observation for contention-aware recovery: at C1,
the local H2D wait was only about 0--3 ms in the measured cells, whereas C4
exposed tens of milliseconds of queued transfer time and a much larger TTFT
tail. It is not yet an EOSS-vs-barrier comparison. The matching C4 EOSS run
was blocked during model startup by the host overlayfs safetensors read at
25% progress, before any EOSS tile request; it produced no result row and is
excluded from all statistics.

The runner was also hardened for this matrix: each cell has a unique log,
measured prefix/retrieve entries are filtered by the backend request ID (with
the EngineCore `-0-hash` suffix), and duplicate lookup lines for one request
are collapsed. Warmup requests are no longer mistaken for measured recovery.

The runner no longer forces safetensors background prefetch on this overlayfs
model path. Its default `--safetensors-load-strategy auto` follows vLLM's
native choice (no prefetch for a non-network filesystem); `prefetch` remains
an explicit opt-in, rather than a hidden source of startup-I/O variance.

### P5 prototype: terminal-state materialization

The current all-load path materializes all four 528-token state pages for every
missed page: three Mamba/GDN objects plus one FullAttention object. This is
unnecessarily sequence-shaped for a recurrent state. A new opt-in `P5`
(`terminal_linear_load`) prototype therefore submits two LMCache retrieves for
one CPU-hit interval:

| State family | Retrieved interval | Why |
| --- | --- | --- |
| FullAttention KV | Every missing 528-token page | Attention needs the full token-indexed history. |
| Mamba/GDN | Only the final 528-token page | The state at the CPU-hit boundary is sufficient to start the new suffix. |

The prototype is intentionally restricted to Qwen3.5's verified aligned
geometry: every engine group has exactly one 528-token paged block per LMCache
chunk and CPU objects are split by state family. It does not claim that every
Linear-Attention implementation has this representation. The implementation
fails closed when either condition is absent. It records object-group H2D bytes
separately, so a valid P5 row must show all Full KV pages and exactly one page
per Mamba/GDN group. Correctness and E2E latency remain pending because the
host overlayfs model-load stall currently prevents a new model server from
reaching the recovery phase.

For a `K`-page CPU-only interval, the expected transfer volume is:

| Recovery | Expected H2D bytes |
| --- | --- |
| P1 aligned all-load | `65.625 MiB × K` |
| P5 terminal-state materialization | `16.5 MiB × K + 49.125 MiB` |

Thus P5 is intentionally neutral at one page (both are 65.625 MiB), while it
removes 49.125 MiB for every additional missed page: 82.125 MiB versus 131.25
MiB at two pages, and 115.125 MiB versus 262.5 MiB at four pages. This is the
specific long-miss/transfer-pressure opportunity to test against P1 and P2.

### Goal-run: deterministic P1/P2/P5 checks on GPU2

The goal-run uses the real LMCache local-CPU backend, BF16 Qwen3.5-9B,
`gpu_memory_utilization=0.9`, 24 GiB CPU cache, SLRU, and a 528-token chunk.
The runner now fixes `VLLM_FLOAT32_MATMUL_PRECISION=highest` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; without these settings, fresh processes
occasionally produced different greedy first tokens in the Qwen GDN align path.

The 784-token smoke run used four requests at C1 and output=1. P1 had three
actual CPU recoveries and one natural GPU hit; P2 issued no H2D. Both had the
same first-token signature `7968cf1b1c1b72b8`, so this pair passes the small
correctness gate. P1 TTFT p50 was 154.364 ms and P2 was 156.555 ms. Because
the CPU-recovery count was 3/4, this is a correctness smoke result, not a
strict all-request recovery point.

The 1568-token run is the first valid P5 byte-reduction point. All four
requests had `local_gpu=528`, `cpu=1584`, and `need_h2d=1056`; all three
policies produced the same first-token signature `80f1de9cc34d7d21`.

| Policy | TTFT p50 (ms) | H2D object bytes, 4 requests | H2D group bytes | Correctness |
| --- | ---: | ---: | --- | --- |
| P1 All Load | 148.883 | 528.0 MiB | Mamba 0/1/2: 132 MiB each; Full: 132 MiB | pass vs P2 |
| P2 All Replay | 219.104 | 0 | none | reference |
| P5 Full history + terminal Mamba | 165.789 | 330.0 MiB | Mamba 0/1/2: 66 MiB each; Full: 132 MiB | pass vs P2 |

P5 therefore reduces H2D volume by 37.5% versus P1 at two missed pages,
while remaining faster than P2 in this C1 sample. The result supports the
more precise claim that recurrent state recovery can be terminal-state based;
it does not support “Linear always loads” or “Linear always replays”. The
784-token P5 smoke was retained as debugging evidence because one request's
signature differed despite the one-page transfer geometry; the two-page P5
cell above is the accepted P5 correctness/byte result.

The raw rows are in `/root/qwen35_goal_correctness_c1.jsonl`,
`/root/qwen35_goal_p1p2p5_c1.jsonl`, and
`/root/qwen35_goal_p1p2p5_1568_c1.jsonl`. The earlier 784-token C1/C4/C8
output=256 run is in `/root/qwen35_goal_p1p2_784.jsonl`; its timing and
recovery traces are useful diagnostics, but it predates the deterministic
environment and its cross-policy signatures must not be used as correctness
evidence.

### Goal-run: completed P1/P2 page-count × concurrency matrix

After fixing the deterministic CUDA environment, the output=256 matrix was
rerun on one physical GPU through LMCache local CPU. The shared GPU prefix was
528 tokens and the CPU suffix was either 784 tokens (one actual 528-token
page) or 1568 tokens (two actual pages). No explicit GPU eviction was used.
Each row below compares P1 and P2 at the same concurrency and uses the same
first-token map as the correctness gate.

| Suffix / pages | Requests/C | P1 TTFT p50 (ms) | P2 TTFT p50 (ms) | P1 H2D | H2D wait observed | Signature gate | Classification |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 784 / 1 | 8/1 | 145.694 | 178.058 | 528 MiB | 12.9–74.4 ms | fail: P1 `2d4d...` vs P2 `c1c5...` | diagnostic |
| 784 / 1 | 8/4 | 396.410 | 31,321.034 | 528 MiB | 17.4–38.6 ms | fail: P1 `0c90...` vs P2 `c1c5...` | diagnostic |
| 784 / 1 | 8/8 | 851.168 | 35,263.987 | 462 MiB | 31.8–113.3 ms | fail: P1 `0c90...` vs P2 `c1c5...` | diagnostic: 7/8 recovery |
| 1568 / 2 | 4/1 | 175.223 | 270.939 | 528 MiB | 31.8–103.1 ms | pass: `80f1de9cc34d7d21` | valid |
| 1568 / 2 | 4/4 | 637.714 | 16,053.305 | 528 MiB | 78.6–100.4 ms | pass: `80f1de9cc34d7d21` | valid |
| 1568 / 2 | 8/8 | 817.933 | 34,064.130 | 1,056 MiB | 91.7–171.1 ms | pass: `bd11450b1323d04e` | valid |

The 1568/C8 cell used eight requests, whereas the 1568/C1 and C4 cells used
four requests; comparisons across those rows are therefore not a
request-count-controlled sweep.

The 784-token rows are retained for latency and transfer debugging only: their
cross-policy signatures did not match, so no 784-token P1/P2 speedup is claimed
from this run. For the correctness-passing 1568-token four-request rows, P1 is
35.3% lower than P2 at C1 and 96.0% lower at C4. At C8 with eight requests,
P1 and P2 are also output-identical and P1 is about 97.6% lower in TTFT, but
that row is reported separately because its request count differs from the
C1/C4 pair.

The queue trace supports, but does not by itself prove, a contention effect:
P1's observed H2D wait reaches about 113 ms at one-page C8 and 171 ms at
two-page C8. P2 has no lookup or H2D by design, so its tens-of-seconds TTFT
comes from replay/scheduling rather than transfer. The current data therefore
supports “page count and serving pressure must enter the recovery decision,”
not “queueing universally makes Replay faster.”

One additional 1568-token, eight-request C1 attempt was rejected before
timing because one request lost the GPU shared prefix
(`local_gpu_tokens=0`, `cpu_tokens=1584`, `need_h2d_tokens=1584`). This is a
capacity/retention failure under the no-explicit-eviction protocol, not a
valid recovery point. It is retained in the raw run log
`/root/qwen35_goal_p1p2_1568_deterministic.jsonl`. The accepted 1568/C8 row
is in `/root/qwen35_goal_p1_1568_c8_capacity.jsonl` and
`/root/qwen35_goal_p2_1568_c8.jsonl`; the completed 784 matrix is in
`/root/qwen35_goal_p1p2_784_deterministic_rerun.jsonl`, and the four-request
1568 pair is in `/root/qwen35_goal_p1p2_1568_deterministic_4req.jsonl`.

### Goal-run: three-page P5 under controlled GPU pressure

The corpus has no prompt with a suffix of 2640 or more tokens. Therefore the
longest real workload available with a 528-token retained shared prefix is
suffix=2112, which produces `need_h2d_tokens=1584` (three missing pages). A
controlled GPU filler phase was used so that the shared prefix stayed on GPU
while the suffix was found in LMCache CPU; the CPU tier was not cleared.

| Policy | Requests/C | TTFT p50 (ms) | H2D bytes | Group bytes | Signature | Classification |
| --- | ---: | ---: | ---: | --- | --- | --- |
| P1 All Load | 4/1 | 189.706 | 792 MiB | Full 198; Mamba 0/1/2 198 each | `fd9af8c6805d5e37` | valid |
| P2 All Replay | 4/1 | 325.281 | 0 | none | `fd9af8c6805d5e37` | valid reference |
| P5 terminal Mamba | 4/1 | 182.639 | 396 MiB | Full 198; Mamba 0/1/2 66 each | `fd9af8c6805d5e37` | valid |
| P5 terminal Mamba | 4/4 | 15,961.798 | 396 MiB | Full 198; Mamba 0/1/2 66 each | `fd9af8c6805d5e37` | valid |

The C1 P5 row reduces H2D volume by 50% versus P1 at three actual missing
pages and remains below P2 TTFT. The C4 row confirms the same object selection
and output signature under concurrent requests, but its very large TTFT is a
pressure observation and should be repeated before treating it as a stable
p50 claim. The P5 bytes match the terminal-state prediction: Full transfers K
=3 pages and each recurrent group transfers one endpoint page.

The requested four-page (`need_h2d_tokens=2112`) real-ShareGPT point could not
be run: constructing it requires suffix=2640, but the dataset contains zero
eligible prompts in `[2640, 6736]`. This is a workload-availability failure,
not a model correctness or GPU-capacity result. The failed attempt was cleaned
up and produced no result row; it must remain separate from the valid
three-page evidence.

The three-page P1/P2/P5 C1 rows are in
`/root/qwen35_goal_p1p2p5_2112_c1_pressure.jsonl`; P5/C4 is in
`/root/qwen35_goal_p5_2112_c4_pressure.jsonl`.

The separate K=3/C4 P1/P2 run had complete individual cache gates (P1 had
4/4 CPU recoveries and 792 MiB H2D), but the signatures differed:
`d3cffb256dd38b0d` for P1 versus `fd9af8c6805d5e37` for P2. Its TTFT p50 was
12,031.383 ms versus 16,843.413 ms. It is therefore diagnostic only; the
latency difference is not used as a valid P1/P2 result. The raw pair is in
`/root/qwen35_goal_p1p2_2112_c4_repeat1.jsonl`.

### Repeat audit for the one-page matrix

A second output=256 run used the same eight-request workload. Its P1 latency
was 147.329/352.211/807.237 ms at C1/C4/C8. C1 had only 6/8 CPU recoveries;
C4 and C8 had 8/8. The P2 latencies were 174.008/32,755.968/35,981.806 ms.
The C4 and C8 signatures still differed between P1 and P2, and the C1 P1 row
was mixed, so these repeat rows are diagnostic rather than main-table evidence.
They nevertheless reproduce the large replay penalty at higher concurrency;
that observation is only used after the valid 1568 and three-page pairs have
passed the signature gate. The repeat file is
`/root/qwen35_goal_p1p2_784_repeat2.jsonl`.

An auxiliary output=1 K=1 pressure run confirmed the materialization semantics
at small output length: P1/C1 and P1/C4 both had 4/4 CPU recoveries and
signature `7968cf1b1c1b72b8` (TTFT 152.638 and 456.084 ms). The corresponding
P2 process exceeded the 900-second startup timeout while loading overlayfs
checkpoint shards, so no P2 row was written. This supports a narrow
correctness smoke check only; it does not repair the output=256 K=1
cross-policy mismatch and is excluded from the latency main table.

### Goal-run: second valid K=2 P1/P2 repetition

Using the same 528-token shared prefix, actual `need_h2d_tokens=1056`, and
output length 256, a second pressure-safe repetition was completed with
filler=4 for C1 and filler=8 for C4/C8. Every measured request passed
`local_gpu=528`, CPU `=1584`, and nonzero H2D; P2 issued no LMCache retrieve.
The first-token signatures matched within every pair:

| Concurrency | P1 TTFT p50 (ms) | P2 TTFT p50 (ms) | P1 H2D | Signature gate | Classification |
| ---: | ---: | ---: | ---: | --- | --- |
| C1, 4 requests | 175.553 | 260.712 | 528 MiB | pass: `80f1de9cc34d7d21` | valid |
| C4, 4 requests | 694.133 | 16,794.896 | 528 MiB | pass: `80f1de9cc34d7d21` | valid |
| C8, 8 requests | 952.865 | 32,448.600 | 1,056 MiB | pass: `bd11450b1323d04e` | valid |

The raw files are `/root/qwen35_goal_p1p2_1568_c1_repeat1.jsonl`,
`/root/qwen35_goal_p1p2_1568_c4_repeat1b.jsonl`, and
`/root/qwen35_goal_p1p2_1568_c8_repeat1.jsonl`. The earlier attempt with
filler=16 was rejected because one request lost the shared GPU prefix; it is a
capacity/control failure and produced no row. The safe filler values are
therefore part of the cache-state protocol, not a tunable latency result.

Across the two valid K=2 repetitions, the within-run paired TTFT medians are:

| Concurrency | P1 median p50 (ms) | P2 median p50 (ms) | P1 reduction | H2D volume |
| ---: | ---: | ---: | ---: | ---: |
| C1 | 175.388 | 265.826 | 34.0% | 528 MiB / 4 requests |
| C4 | 665.924 | 16,424.101 | 95.9% | 528 MiB / 4 requests |
| C8 | 885.399 | 33,256.365 | 97.3% | 1,056 MiB / 8 requests |

### Goal-run: third valid K=2 P1/P2 repetition

A third pressure-safe repetition used the same actual `need_h2d_tokens=1056`,
output length 256, and per-concurrency filler protocol. Every request passed
the GPU-prefix/CPU-suffix gate, and the first-token signatures matched within
each P1/P2 pair:

| Concurrency | P1 TTFT p50 (ms) | P2 TTFT p50 (ms) | P1 H2D | Signature gate | Classification |
| ---: | ---: | ---: | ---: | --- | --- |
| C1, 4 requests | 169.864 | 265.888 | 528 MiB | pass: `80f1de9cc34d7d21` | valid |
| C4, 4 requests | 710.982 | 17,404.484 | 528 MiB | pass: `80f1de9cc34d7d21` | valid |
| C8, 8 requests | 823.014 | 37,279.698 | 1,056 MiB | pass: `bd11450b1323d04e` | valid |

The raw files are `/root/qwen35_goal_p1p2_1568_c1_repeat2.jsonl`,
`/root/qwen35_goal_p1p2_1568_c4_repeat2.jsonl`, and
`/root/qwen35_goal_p1p2_1568_c8_repeat2.jsonl`. The runner's new
endpoint-backed timing fields are populated in these rows. For example, at
C8 P1 measured 2.042 s aggregate prefill and 3.982 s aggregate scheduler
queue time, while P2 measured 2.111 s aggregate prefill and 305.568 s
aggregate queue time; these are sums over eight requests, not per-request
latencies.

Across all three valid K=2 repetitions, the paired TTFT p50 medians are:

| Concurrency | P1 median p50 (ms) | P2 median p50 (ms) | P1 reduction | H2D volume |
| ---: | ---: | ---: | ---: | ---: |
| C1 | 175.223 | 265.888 | 34.1% | 528 MiB / 4 requests |
| C4 | 694.133 | 16,794.896 | 95.9% | 528 MiB / 4 requests |
| C8 | 823.014 | 34,064.130 | 97.6% | 1,056 MiB / 8 requests |

These results support a motivating observation for the tested local-CPU
configuration: replay becomes rapidly queue-dominated with concurrency while
all-load H2D remains bounded. They do not establish that Load always wins or
that a crossover is impossible; longer valid missing-page points and a
correctness-safe Adaptive policy are still required.

The runner now snapshots vLLM Prometheus counters immediately before and after
each measured phase. The third-repeat rows validate measured-phase
`request_prefill_time`, `request_queue_time`, `request_inference_time`, and
`e2e_request_latency` deltas; P2 records the prefill delta as its replay
compute-time proxy. These values are aggregate counter deltas for the measured
requests and are not substituted for TTFT percentiles.

### Goal-run: three-page C8 correctness diagnostic

The same pressure protocol was also run at actual `need_h2d_tokens=1584`
(three pages), with eight measured requests at C8. P1 satisfied the cache
precondition for all eight requests (`local_gpu=528`, CPU `=2112`,
`need_h2d=1584`) and transferred 1,584 MiB in total: 396 MiB per state group.
Its TTFT p50 was 30,629.059 ms and the observed LMCache H2D waits were
42.184--205.602 ms. P2 had no lookup or H2D and TTFT p50 38,515.566 ms.

The pair is **diagnostic, not a valid comparison**: the first-token signatures
were `0cd0fb7df15756c3` (P1) and `d0dcd3115c7de5dc` (P2). The per-request map
shows one differing first token, so the apparent 20.5% P1 TTFT reduction is
not reported as a recovery speedup. This is evidence that the current
high-concurrency asynchronous materialization path still needs a correctness
investigation before C8 rows can enter the main table; it is not evidence that
H2D queueing makes Replay preferable. The raw pair is in
`/root/qwen35_goal_p1p2_2112_c8_repeat1.jsonl`.

### Goal-run: second valid K=3/C1 P1/P2/P5 repetition

At output length 256, actual `need_h2d_tokens=1584` (three missing pages), a
second C1 run passed the cache and signature gates for all three policies:

| Policy | TTFT p50 (ms) | H2D volume | Signature | Classification |
| --- | ---: | ---: | --- | --- |
| P1 All Load | 213.284 | 792 MiB | `fd9af8c6805d5e37` | valid |
| P2 All Replay | 301.854 | 0 | `fd9af8c6805d5e37` | valid |
| P5 terminal-state materialization | 182.881 | 396 MiB | `fd9af8c6805d5e37` | valid |

P1 transferred three pages for each of the four state groups. P5 transferred
three Full pages and one terminal page for each of the three Mamba groups,
reducing materialization bytes from 792 MiB to 396 MiB (50%). The raw file is
`/root/qwen35_goal_p1p2p5_2112_c1_repeat2.jsonl`. P5's object selection and
output match are therefore reproduced at K=3/C1, while a third K=3/C1 repeat
and broader valid concurrency coverage remain pending.

The next K=3/C1 repeat completed the cache gates but was diagnostic for
correctness: P1 produced signature `ea62b817fb56a82a`, whereas P2 and P5 both
produced `fd9af8c6805d5e37` (the differing request was record 417). The raw
file is `/root/qwen35_goal_p1p2p5_2112_c1_repeat3.jsonl`. This strengthens the
need to treat multi-page All Load as correctness-gated: the row is not used in
the latency or P5 repetition counts, even though P1 transferred the expected
three-page groups. The fact that P5 matched P2 in this run is a useful
diagnostic signal for terminal-state materialization, not yet a general proof.

A settle=3 P1/P2 correctness probe at K=3/C1 then passed with p50
206.429/318.893 ms, uniform `[1584, 1584, 1584, 1584]` recovery tokens, and
common signature `fd9af8c6805d5e37`. The raw file is
`/root/qwen35_goal_p1p2_2112_c1_settle3_probe.jsonl`. Together with the two
earlier valid K=3/C1 pairs, this is the third valid P1/P2 repetition at this
length. A separate settle=3 P5 probe, `/root/qwen35_goal_p5_2112_c1_settle3.jsonl`,
passed at 191.963 ms with the same signature, uniform K=3, and 396 MiB H2D;
this supplies the third P5/P2 correctness point. The earlier P1 mismatch is
still diagnostic and not included in these counts.

### K=3/C4 settle=3 follow-up

With `--warmup-settle-seconds=3`, the K=3/C4 P1/P2 pair had uniform actual
K=3 and matching signature `fd9af8c6805d5e37`: P1 p50 was 17,411.515 ms and
P2 p50 was 16,375.518 ms. P5 passed its cache gate and transferred the
expected 396 MiB, but its signature was `d3cffb256dd38b0d`; one request
differed from P1/P2. The raw file is
`/root/qwen35_goal_p1p2p5_2112_c4_settle3.jsonl`. Thus this is a valid P1/P2
pair but a P5 correctness failure, and none of its apparent timing
differences is used as a P5 speedup claim.

### K=3/C8 serialized-terminal follow-up

With settle=3 and `--p5-serialize-terminal`, all eight requests had uniform
K=3. P1 was diagnostic (`0cd0fb7df15756c3`), while P2 and P5 matched at
`d0dcd3115c7de5dc`. P2 p50 was 36,260.322 ms and P5 p50 was 37,206.112 ms;
P5 transferred 792 MiB (Full 396 MiB plus three terminal Mamba groups of
132 MiB each). The raw file is
`/root/qwen35_goal_p1p2p5_2112_c8_serial_terminal.jsonl`. This is a valid
P2/P5 correctness-safe point, not a P1 comparison; P1's one-request mismatch
keeps the normal All Load C8 result diagnostic.

### K=3/C4 P1/P2 repeated matrix

Three settle=3 repetitions now pass the P1/P2 cache, page-count, H2D, and
signature gates at C4: `/root/qwen35_goal_p1p2_2112_c4_settle3.jsonl`,
`/root/qwen35_goal_p1p2_2112_c4_repeat2.jsonl`, and
`/root/qwen35_goal_p1p2_2112_c4_repeat3.jsonl`. Their P1/P2 p50 values are
17,411.515/16,375.518 ms, 17,831.277/16,608.363 ms, and
15,633.205/17,492.879 ms. The medians are therefore 17,411.515 ms for P1
and 16,608.363 ms for P2; P2 is about 4.8% lower. This is a valid high-queue
P1/P2 point, but not a Load win. It supports treating missing length and
contention as decision variables rather than assuming All Load always wins.
The normal concurrent P5 result from the first file remains diagnostic; the
serialized-terminal P5 fallback is reported separately below.

### K=3/C4 serialized-terminal correctness probe

The opt-in `--p5-serialize-terminal` mode serialized each request's Full and
terminal H2D copies, then synchronized CUDA before the next request. In this
mode P2 p50 was 18,653.444 ms and P5 p50 was 16,754.925 ms; both had signature
`fd9af8c6805d5e37`, uniform K=3, and P5 transferred 396 MiB. The raw file is
`/root/qwen35_goal_p2p5_2112_c4_serial_terminal.jsonl`. This is a valid
correctness-safe fallback probe, not a normal-P5 speedup result: the
serialization overhead is part of the measured P5 latency, and the default
concurrent P5 C4 row remains diagnostic.

### Goal-run: K=2 P5 concurrency sweep (output=256)

The output=256 terminal-state sweep produced uniform actual K=2 P5/P2 pairs:

| Concurrency | P2 TTFT p50 (ms) | P5 TTFT p50 (ms) | P5 H2D | Signature gate | Classification |
| ---: | ---: | ---: | ---: | --- | --- |
| C1, 4 requests | 246.842 | 159.184 | 330 MiB | pass: `80f1de9cc34d7d21` | valid |
| C4, 4 requests | 18,378.942 | 461.480 | 330 MiB | pass: `80f1de9cc34d7d21` | valid |
| C8, 8 requests | 41,698.756 | 680.482 | 660 MiB | pass: `bd11450b1323d04e` | valid |

For K=2, P1 transfers 528 MiB for four requests and 1,056 MiB for eight;
P5 transfers 330 MiB and 660 MiB, respectively. Thus P5 reduces H2D volume by
37.5% while retaining the P2 first-token signature. The C4 raw file also
contains a mixed P1 row (one request had `need_h2d=528` and three had 1056),
so that P1 row is excluded; the P5 row itself had uniform K=2 and is valid
against P2. Raw files are `/root/qwen35_goal_p1p2p5_1568_c1_repeat2.jsonl`,
`/root/qwen35_goal_p1p2p5_1568_c4_repeat2.jsonl`, and
`/root/qwen35_goal_p2p5_1568_c8_repeat1.jsonl`.

The runner now records `lmcache_need_h2d_tokens`, `lmcache_missing_pages`, and
`lmcache_recovery_length_uniform` in each LMCache row. This prevents a nominal
suffix setting from being mistaken for a uniform actual missing-page bucket.
It also exposes `--warmup-settle-seconds` (default 3 s) so asynchronous CPU
stores are allowed to complete before the deliberate GPU-cache pressure.

### Goal-run: K=2 Adaptive length branch

With `adaptive_replay_below_tokens=1056`, the K=2 workload produced a uniform
`need_h2d_tokens=1056` (two actual pages) and selected the length-triggered
Replay branch for all eight measured requests at C1, C4, and C8. All three
rows passed the GPU-prefix/CPU-hit/page-count gates and had zero H2D; their
first-token signature was `bd11450b1323d04e`, matching the established P2
signature for this deterministic prompt set.

| Concurrency | Adaptive TTFT p50 (ms) | p99 (ms) | Replay decisions | H2D | Replay prefill proxy (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| C1, 8 requests | 250.816 | 335.098 | 8/8, length | 0 | 1,963.805 |
| C4, 8 requests | 26,912.178 | 32,472.750 | 8/8, length | 0 | 2,138.259 |
| C8, 8 requests | 32,480.165 | 68,706.620 | 8/8, length | 0 | 1,924.426 |

The raw files are `/root/qwen35_goal_adaptive_k2_threshold1056_c1_probe.jsonl`
and `/root/qwen35_goal_adaptive_k2_threshold1056_c4c8.jsonl`. These rows
validate the length-aware decision and its replay timing trace; they do not
prove an Adaptive speedup because the exact same protocol was not rerun with
P1 and P2 in the same file. In particular, the C8 result shows that removing
H2D does not remove the scheduler queue: its aggregate vLLM queue time was
271,263 ms over eight requests.

The attempted K=3 Adaptive probe is excluded. A large-filler variant failed
before measurement because the corpus had only 245 eligible long prompts for
the requested 1,024 filler requests; the 64-filler variant left only the
shared 528-token prefix in CPU cache (`cpu_tokens=528`), so it did not satisfy
the intended 4-page CPU-suffix precondition. Neither is a performance result.

### K=1/C1 serialized All Load correctness probe

To test whether the one-page mismatch was only caused by concurrent H2D
staging, P1 was rerun with per-request All Load serialization and compared to
P2 under the same output=256 protocol. Both rows had four requests with
`local_gpu_tokens=528`, uniform `need_h2d_tokens=528`, and the P1 row
transferred 276,824,064 bytes (four groups, 69,206,016 bytes each). P1/P2
TTFT p50 was 155.230/184.220 ms and the signatures were
`a0d2400bba2b816d` versus `7968cf1b1c1b72b8`. The pair is therefore diagnostic,
not a valid speed comparison; serializing All Load does not resolve the
single-page cross-policy correctness discrepancy. Raw evidence is
`/root/qwen35_goal_p1p2_784_c1_serial_all_load.jsonl`.

### Offline P1/P2 oracle over correctness-safe cells

As a trace-level oracle, each valid repeated P1/P2 cell was assigned to the
policy with the lower paired TTFT p50. The oracle is not an online policy and
uses measured outcomes only to establish the decision boundary:

| Actual missing pages | Concurrency | P1 median p50 (ms) | P2 median p50 (ms) | Oracle choice | Margin |
| ---: | ---: | ---: | ---: | --- | ---: |
| 2 | C1 | 175.223 | 265.888 | P1 Load | 34.1% |
| 2 | C4 | 694.133 | 16,794.896 | P1 Load | 95.9% |
| 2 | C8 | 823.014 | 34,064.130 | P1 Load | 97.6% |
| 3 | C1 | 206.429 | 318.893 | P1 Load | 35.2% |
| 3 | C4 | 17,411.515 | 16,608.363 | P2 Replay | 4.8% |

This small oracle already rejects a fixed policy: the winner changes at the
same model and page type when concurrency/queue pressure changes. It also
rejects the current fixed Adaptive threshold as a performance claim: at K=2
the threshold=1056 branch selected Replay, while the measured oracle selected
Load at all tested C1/C4/C8 points. The observation is a motivation for a
cost model using measured queue and replay cost, not evidence that the current
threshold controller is good.

### K=3/C8 serialized All Load diagnosis

For comparison, `--serialize-all-load` was used to submit and synchronize one
complete P1 materialization per request. P1 still produced
`0cd0fb7df15756c3`, while P2 produced `d0dcd3115c7de5dc`, with uniform K=3;
P1 p50 was 36,082.459 ms and P2 p50 was 39,025.214 ms. The raw file is
`/root/qwen35_goal_p1p2_2112_c8_serial_all_load.jsonl`. Since serialization
did not remove the mismatch, cross-request H2D overlap is not the sole cause;
this remains a correctness diagnostic rather than a speedup result.

### K=1/start=4 eviction probe (excluded)

The same strict K=1 protocol was tried on ShareGPT entries starting at index 4
with GPU-cache eviction. The trace reported
`need_h2d_tokens=[528,0,528,528]`: three requests recovered one CPU page, but
one request still had the suffix on GPU. P1 was therefore rejected as a
non-uniform recovery bucket; this is precisely the case the strengthened
`--expected-missing-pages` gate now rejects. The output is retained at
`/root/qwen35_goal_p1p2_784_c1_start4_evicted_serial.jsonl` and is not a
performance result.
### K=1 warmup-repeat correction and short-length matrix update

The K=1 probe showed that a single warmup pass can leave only a subset of the
unique suffix pages in LMCache CPU tier. The runner therefore gained
`--warmup-repeats`; with sequential warmup, it also waits after each request
so asynchronous D2H storage cannot race GPU-page reuse. This changes only
cache population, not the measured P1/P2 path.

With `warmup_repeats=2`, serialized All Load, and output=256, K=1/C1 passed
the uniform-page and output gates: P1 p50 was 159.836 ms and P2 p50 was
191.478 ms; all four requests had `need_h2d_tokens=528` and the common first
token signature was `7968cf1b1c1b72b8`. P1 transferred one page in each of the
three Mamba groups and the Full group per request.

At C4, P1 with `warmup_repeats=3` passed the cache gate and produced p50
495.960 ms with the same `7968cf1b1c1b72b8` signature as the established P2 C4
baseline (16,677.625 ms). This is accepted as a correctness-safe pair, with
the warmup protocol recorded explicitly. A repeat-2 C4 P1 attempt had the
`9038...` signature and remains diagnostic.

At C8, even after `warmup_repeats=3`, P1 remained diagnostic (`d674...`) while
P2 produced `c1c53...`; all eight P1 requests nevertheless had uniform
`need_h2d_tokens=528` and complete four-group H2D traces. The corresponding
P2 repeat-3 run confirmed `c1c53...`. Thus the C8 mismatch is not a missing-page
gate failure, and neither row is used for a P1/P2 speed claim.

Raw evidence:
- `/root/qwen35_goal_p1_784_c1_repeatwarmup_serial.jsonl`
- `/root/qwen35_goal_p2_784_c1_repeatwarmup.jsonl`
- `/root/qwen35_goal_p1_784_c4_repeat3_serial.jsonl`
- `/root/qwen35_goal_p1_784_c8_repeat3_serial.jsonl`
- `/root/qwen35_goal_p2_784_c8_repeat3.jsonl`

### K=2/C10 suffix-only metadata and pressure point

The K=2/C10 pair was rerun with the explicit `--suffix-only` protocol after
the analyzer was corrected to require that flag only for P3/P4 mixed replay.
Both policies passed the uniform two-page recovery/control and output gates
with common signature `247a563bac96db57`.

| Policy | TTFT p50 (ms) | TPOT p50 (ms) | H2D | Queue/replay evidence |
| --- | ---: | ---: | ---: | --- |
| P1 All Load | 876.727 | 42.914 | 1.289 GiB / 10 requests | H2D batches 34.003/212.307 ms |
| P2 All Replay | 43,070.703 | 40.696 | 0 | aggregate scheduler queue 438.255 s |

All ten P1 requests had `local_gpu_tokens=528`, `cpu_tokens=1584`, and
`need_h2d_tokens=1056`; P1 transferred two objects per group per request.
This is a valid high-contention P1/P2 point and strengthens the pressure-aware
observation, but it does not imply that Load always wins: the valid K=3/C4
pair still favors Replay by about 4.8%.

Raw evidence: `/root/qwen35_goal_p1p2_1568_c10_suffixonly_repeat2.jsonl`.

### K=2/C10 P5 terminal-state materialization

P5 was then measured at the same K=2/C10 workload. P2, P5, and the preceding
P1 row all used the same ten prompts and produced the common signature
`247a563bac96db57`; all ten P5 requests had `local_gpu_tokens=528`,
`cpu_tokens=1584`, and `need_h2d_tokens=1056`.

| Policy | TTFT p50 (ms) | TPOT p50 (ms) | H2D |
| --- | ---: | ---: | ---: |
| P1 All Load | 876.727 | 42.914 | 1.289 GiB |
| P5 TSM | 885.384 | 44.899 | 0.805 GiB |
| P2 All Replay | 43,168.268 | 41.309 | 0 |

P5 transferred two Full pages per request and one terminal page in each of the
three Mamba groups, reducing materialization from 1.289 GiB to 0.805 GiB
(37.5%) while preserving the output signature. TTFT is essentially tied with
P1 at this point, so this result supports a byte/capacity benefit rather than
a claimed latency win. Raw evidence:
`/root/qwen35_goal_p2p5_1568_c10_suffixonly_repeat2.jsonl`.

### K=1/start=8 C4/C8 correctness-safe pair

The previous K=1/start=0 C4/C8 rows were diagnostic. To test whether that
failure was workload-specific, the same one-page protocol was run on a
different deterministic ShareGPT suffix set (`start_index=8`). Both
concurrency points passed the cache-state, uniform-page, H2D, and signature
gates with common signature `3163847c9065406e`.

| Concurrency | P1 TTFT p50 (ms) | P2 TTFT p50 (ms) | P1 H2D | Classification |
| ---: | ---: | ---: | ---: | --- |
| C4 | 392.373 | 30,213.724 | 528 MiB / 8 requests | valid |
| C8 | 581.270 | 38,048.725 | 528 MiB / 8 requests | valid |

All eight P1 requests at both points had `local_gpu_tokens=528`,
`cpu_tokens=1056`, and `need_h2d_tokens=528`. These rows are valid evidence
for the one-page workload, but are kept under `start_index=8`; they are not
merged with the start=0 diagnostic rows.

Raw evidence: `/root/qwen35_goal_p1p2_784_start8_c4c8_suffixonly_repeat3.jsonl`.

### K=2/C1 state-type selection diagnostic

The K=2/C1 P3/P4 probe verified the intended group-level selection and
suffix-only replay markers, but neither mixed policy passed output correctness
against P1/P2. P1/P2 shared signature `80f1de9cc34d7d21`, while both P3 and P4
produced `67f4fc18605c4e61`.

| Policy | TTFT p50 (ms) | H2D groups | Replay marker | Classification |
| --- | ---: | --- | ---: | --- |
| P3 Full Load + Linear Replay | 252.880 | Full group only | 4/4 | diagnostic: signature mismatch |
| P4 Full Replay + Linear Load | 269.573 | 3 Mamba groups only | 4/4 | diagnostic: signature mismatch |

The raw trace proves selective H2D object-group routing, not that a state type
can independently replace the corresponding decoder computation. These rows
are excluded from all speedup claims.

Raw evidence: `/root/qwen35_goal_p1p2p3p4_1568_c1_suffixonly.jsonl`.

### K=3/C8 start=8 multi-page correctness diagnosis

Using a second prompt set and a feasible 16-request filler, the K=3/C8 P1/P2
pair passed uniform actual-page and H2D/no-H2D gates, but the signatures still
differed: P1 `9f8fb4d476a9e45d` versus P2 `a45ee06d699a318d`. This reproduces
the high-concurrency multi-page All Load correctness issue across prompt sets;
the row is retained as diagnostic and is not used for a latency comparison.

The attempted 64-request filler was rejected before measurement because only
237 qualifying K=3 suffix candidates exist while the selector needed 512.
That run is a workload-availability failure, not a model-performance result.

Raw evidence: `/root/qwen35_goal_p1p2_2112_start8_c8_suffixonly_filler16.jsonl`.

### K=2/C10 Adaptive threshold diagnostic (non-evicted protocol)

An initial same-command P1/P2/Adaptive run accidentally omitted
`--evict-gpu-cache`, so it is not comparable to the established correctness-safe
K=2/C10 P1/P2 pair. The raw rows still passed their individual cache-state and
H2D-routing checks, but P1's signature was `873865596b548b4b`, whereas P2 and
Adaptive shared `247a563bac96db57`; the analyzer therefore produced no
comparable cell.

| Policy | TTFT p50 (ms) | Actual missing pages | H2D / replay | Classification |
| --- | ---: | ---: | --- | --- |
| P1 All Load | 1,136.261 | 2 | 1.289 GiB object bytes | diagnostic: signature mismatch |
| P2 All Replay | 42,600.025 | 2 | 0 H2D; replay prefill 2,340.052 ms | valid control row |
| Adaptive, threshold=1056 | 43,920.796 | 2 | 0 H2D; all 10 chose replay | diagnostic: no paired correctness cell |

The Adaptive decisions were all `reason=length`, with `need_h2d_tokens=1056`
and no pending H2D. This is not evidence of an Adaptive speedup; it is a
useful warning that a length-only threshold can select Replay in a workload
where the established evicted K=2/C10 P1 reference is much faster. A corrected
Adaptive run with GPU eviction is retained separately.

Raw evidence: `/root/qwen35_goal_p1p2adaptive_1568_c10_suffixonly.jsonl`.

### K=2/C10 Adaptive with GPU eviction (protocol-correct routing, pairing pending)

The corrected Adaptive-only rerun added `--evict-gpu-cache` and therefore
reached the same shared-GPU-prefix plus CPU-suffix state as the established
K=2/C10 P1/P2 reference. All ten requests had `local_gpu_tokens=528`,
`cpu_tokens=1584`, and `need_h2d_tokens=1056`; the output signature was
`247a563bac96db57`. The fixed length threshold still selected Replay for all
ten requests (`reason=length`), with no H2D and a replay prefill proxy of
2,507.243 ms.

| Policy | TTFT p50 (ms) | H2D | Replay decisions | Classification |
| --- | ---: | ---: | ---: | --- |
| Adaptive, threshold=1056 | 42,463.860 | 0 | 10/10 | valid routing/correctness row |

The established same-workload P1/P2 reference uses `warmup_requests=10`,
whereas this Adaptive run used `warmup_requests=72`. Consequently the strict
analyzer keeps it out of the formal paired cell until an Adaptive row with
warmup=10 is collected. As a provisional diagnostic, its TTFT is close to P2
(42.464 s) and vastly slower than the reference P1 (0.877 s); this reinforces
that a length-only threshold is not a cost model, but is not yet a formal
Adaptive regret result.

Raw evidence: `/root/qwen35_goal_adaptive_1568_c10_suffixonly_evict.jsonl`.
