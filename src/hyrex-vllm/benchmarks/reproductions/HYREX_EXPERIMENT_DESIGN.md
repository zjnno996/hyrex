# HyRex experiment design

## Claim

HyRex jointly manages GPU and CPU cache residency for heterogeneous Hybrid
model state, then chooses Full-KV load, recurrent-state load, terminal-state
restore, or replay from measured H2D and GPU-compute pressure. The primary
claim is concurrency-sensitive: one fixed recovery policy is not optimal as
the H2D and compute queues change.

Layers with the same layout may share kernels, buffer pools, transfer
descriptors, profiles, and batched DMA submissions. Numerical cache state is
shared only when model revision, layer id, prefix, positions, adapter, dtype,
and layout match. A shared CPU source does not imply a shared GPU destination.

## System variants

The main comparison uses identical model, trace, arrival process, cache
capacity, GPU memory fraction, output length, and correctness reference.

| Variant | Cache placement | Recovery decision |
| --- | --- | --- |
| Marconi | cache-management baseline | all available state is loaded |
| Tail Replay | fixed | fixed Full-load/Linear-replay policy |
| KVPR-H | CPU cache | checkpoint-aligned prefix load and suffix replay |
| CacheFlow-H | CPU cache | batch/chunk recovery scheduling |
| HyRex | GPU/CPU state-aware cache | online heterogeneous load/replay choice |

The wording above describes the executable adapters in this repository; it
does not claim that a conservative adapter reproduces every optimization in
the corresponding paper system.

## Main concurrency matrix

Run paired cells at concurrency `1,4,8,16,32`, with at least three repetitions
and alternating baseline/HyRex order. Concurrency 64 is a separate stress cell
because it may exceed the memory gate on a 24 GiB GPU. The formal trace must
contain at least 512 sessions and must induce both cache admission and
eviction.

Use the same matrix under these arrival processes:

* Poisson arrivals for the main result;
* bursty arrivals for queue-pressure sensitivity;
* Zipf session popularity for shared-prefix and cache-value sensitivity.

The primary command is:

```bash
.venv/bin/python benchmarks/reproductions/run_hyrex_formal_campaign.py \
  --trace TRACE.jsonl --output-dir RESULTS --concurrencies 1,4,8,16,32
```

Run `--concurrencies 64` separately after the resource preflight succeeds.

## Metrics and acceptance gates

User-facing metrics:

* TTFT p50, p95, and p99;
* TPOT p50, p95, and p99;
* output-token throughput;
* TTFT SLO violation rate when an SLO is configured.

Mechanism metrics:

* policy counts by concurrency;
* GPU-, CPU-, partial-, and cold-hit counts;
* observed H2D bytes, service time, and queue time;
* modeled H2D and compute queue debt at decision time;
* replay milliseconds and loaded bytes;
* admitted, evicted, and rejected cache blocks;
* logical versus physical loads once GPU materialization fan-out is enabled.

A formal cell is valid only when outputs match the paired reference, transfer
completion is fenced, calibration provenance is measured, cache pressure is
observed, and every requested repetition completes. The main claim should be
reported using p99 TTFT and SLO violations; median-only improvements are not
sufficient under concurrency.

## Required ablations

1. fixed all-load versus dynamic recovery;
2. configured constants versus online H2D/replay feedback;
3. request-local decision versus batch queue-aware decision;
4. uniform LRU versus state-aware CPU/GPU admission;
5. per-layer transfers versus same-layout layer bundles;
6. source lookup reuse versus physical GPU materialization reuse.

## Expected interpretation

At low concurrency, all-load may win when H2D is idle. As concurrency grows,
H2D queueing can make mixed replay preferable; if GPU compute is saturated,
the decision can switch back to load. A useful result therefore shows both
latency and the accompanying policy/queue transition, rather than asserting
that one action always wins.

## Implemented path

The native worker already bundles all layer data references in one KV group
into a single asynchronous transfer submission, so the same-layout layer
optimization reuses that path rather than adding a second bundler. HyRex
calibrates H2D by transfer-size and serving-concurrency buckets, calibrates
replay by state/token/concurrency buckets, and falls back to the nearest
measured bucket before using configured cold-start values.

Selective per-state CPU admission is enabled for HyRex. Full-KV and recurrent
groups are looked up independently, so one missing state type does not erase a
useful hit in the other. Admission keeps a state type only when its predicted
reload is no slower than replay, using the nearest transfer-size/concurrency
and replay-token/concurrency profiles with configured cold-start fallbacks.
The decision is deliberately per state type rather than per layer: layers with
similar layouts share transfer submission, buffers, kernels, and profiles, but
their numerical states are reusable only when the source content is identical.

HyRex batch-wide mode uses one scheduler-step lookup barrier. It collects the
real Full-KV and recurrent CPU hits for all waiting requests, chooses their
load/replay actions together using shared H2D and compute clocks, and only then
allocates destination GPU blocks. Sources are revalidated after the barrier.
Configure the native CPU backend with `eviction_policy=hyrex` to evict by
measured recovery-time benefit per byte; LRU and ARC remain comparison modes.

For unequal state hit depths, HyRex evaluates all-load at the common boundary
and each mixed policy at the selected source's own boundary. This safely uses
the longer independent hit without presenting an unrestored group as cached.
The stronger two-checkpoint merge (load the shorter recurrent checkpoint and
replay only its gap to the longer Full-KV boundary) remains outside vLLM's
single external-token boundary interface and is not claimed by this prototype.
