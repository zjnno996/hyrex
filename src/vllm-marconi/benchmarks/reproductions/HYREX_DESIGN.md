# HyRex: Hybrid Recovery Execution Scheduler

## Research claim

HyRex treats CPU-offloaded Hybrid cache recovery as a **batch scheduling** problem,
not a per-request cache hit decision. A recovery batch contains heterogeneous
Full-Attention KV segments and recurrent/SSM states. For every segment, HyRex
tracks:

```text
source readiness + logical source sharing + GPU materialization target
        + H2D queue/service cost + replay cost + request urgency
```

It then chooses `LOAD`, `TERMINAL_LOAD`, or `REPLAY`, and orders the resulting
tasks on the shared H2D and GPU-compute resources. `TERMINAL_LOAD` is valid only
for a recurrent state with a verified endpoint checkpoint; it is never applied
to Full-Attention KV.

The key distinction from ordinary prefix caching is that a shared CPU source
does not automatically mean a shared GPU transfer: two requests may need the
same conversation segment in different GPU blocks. HyRex therefore coalesces
lookup/metadata work by `source_key`, but only coalesces H2D by an explicit
`materialization_key`.

## Why this is publishable

The contribution is the interaction of four facts that are usually evaluated
separately:

1. Hybrid state has non-equivalent recovery semantics (all Full KV pages versus
   one recurrent endpoint).
2. Multi-turn users create shared CPU sources but independent GPU destinations.
3. Readiness is dynamic: a cache hit can be present but not yet readable while a
   store/promotion is in flight.
4. Concurrent recovery tasks contend for two different resources, so the
   locally cheapest action need not minimize batch P99 TTFT.

Marconi is the cache admission/eviction baseline; KVPR/HCache are
load-versus-recompute baselines for homogeneous KV; HyRex takes their measured
costs as inputs and solves the missing cross-request Hybrid recovery decision.

## vLLM implementation stages

### Stage 1: decision layer (current branch)

`vllm/v1/kv_offload/hyrex_scheduler.py` provides a backend-independent planner.
It is intentionally not enabled by default. This makes the policy testable and
keeps existing vLLM recovery behavior unchanged.

The same module now also provides `match_hybrid_prefix`: it scans contiguous
Full-KV page hits independently from recurrent checkpoint hits and returns
`(L_KV, L_S)`. The first prototype chooses the deepest `L_S <= L_KV`, then
marks `[L_S, L_KV)` for a normal decoder forward that preserves loaded Full KV
as read-only, followed by `[L_KV, prompt_end)` for ordinary prefill. For
example, `L_KV=8192` and `L_S=6272` requires 1920 tokens of state-restoring
forward, *not* 1920 tokens of whole-model computation avoided. The final
prompt token is excluded from the matched prefix to produce logits once.

The native CPU offload connector now has an opt-in experimental execution path
(`VLLM_MOONCAKE_HYBRID_POLICY=independent_full_kv`). It loads the jointly
available CPU checkpoint first, then queries/loads any deeper Full-KV-only
pages and replays from the checkpoint while leaving those Full-KV slots
read-only. It falls back to the normal path if no deeper Full KV is found.
This uses two serial transfer phases to stay within vLLM's single-prefix
connector interface. It is *not* enabled for LMCache, not yet validated by a
9B end-to-end correctness/TTFT run, and cannot invent CPU objects missing
from the aligned store. That verification is required before any speedup claim.

### Stage 2: observation adapter

At `OffloadingConnectorScheduler.get_num_new_matched_tokens()` and
`update_state_after_alloc()`, create one `RecoverySegment` per recoverable
group/segment. Populate `source_ready`, actual missing tokens, bytes, lookup
time, H2D queue estimate, and replay estimate. The adapter must retain the
actual group/block keys; nominal suffix length is not sufficient.

### Stage 3: executor integration

Before `build_connector_meta()`, pass pending segments to HyRex. Keep the
existing transfer protocol and completion fences. The first executor version
may reorder independent load jobs and choose only among already-correct P1/P2/P5
paths. A true partial Full-KV replay path requires hidden-state checkpoints and
is a separate extension, not silently assumed here.

### Stage 4: P99-aware online policy

Use an exponentially weighted calibration table keyed by
`(state_kind, missing_pages, concurrency_bucket, queue_bucket)`. During warm-up,
fall back to the analytical model. Decisions and predictions must be logged
per request, together with actual TTFT and H2D/replay breakdown.

## Evaluation that can support the claim

Use identical multi-turn traces and strict correctness/cache-state gates. Compare:

* P1: all load;
* P2: all replay;
* P5: Full all-page load + recurrent terminal-state load;
* HyRex online planner;
* offline oracle: lowest measured completion time among valid actions.

Report p50/p95/p99 TTFT, throughput, H2D bytes, replay work, source-sharing
ratio, physical-transfer-sharing ratio, readiness stalls, and oracle regret.
Sweep missing pages (1/2/3/4), concurrency (1/4/8/10), and mixed conversation
reuse. Rows failing either cache-state or full-output checks remain diagnostics
and are excluded from speedup/regret tables.
