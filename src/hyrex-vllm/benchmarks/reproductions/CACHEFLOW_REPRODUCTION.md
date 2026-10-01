# CacheFlow reproduction branch

CacheFlow models restoration as a batch problem across H2D and recomputation,
and prioritizes work with the largest marginal reduction in recomputation cost.
This branch clean-room reproduces that public scheduling principle in the
isolated recovery-policy interface:

1. keep a forward compute pointer and reverse I/O pointer per request;
2. after every chunk, rank requests by remaining recomputation cost;
3. use one H2D and one compute resource clock, assigning the selected chunk
   to the resource that makes it ready first.

Every returned plan additionally validates the execution contract needed by a
runtime adapter: each chunk occurs exactly once, replay chunks advance from
the head, load chunks retreat from the tail, and work on each resource is
serial.  This is intentionally a plan-level gate; vLLM still owns actual
transfer/replay submission and completion fencing.

`CacheFlowExecutor` now dispatches the already-validated load and replay task
queues concurrently through backend callbacks. It does not implement a second
transfer backend: a future vLLM adapter must supply callbacks that submit the
existing H2D/replay jobs and preserve their normal completion fences.

The unified evaluation branch additionally provides a conservative native
Hybrid binding. After a real CPU lookup and before GPU external-token
allocation, it runs the two-pointer policy at the recurrent checkpoint
granularity, restores only the selected number of Hybrid chunks, and lets
ordinary prefill replay the rest. Within one scheduler step, actual H2D jobs
are ordered across requests by remaining replay work. Request logs expose the
available/load/replay token counts, so this path cannot be confused with an
all-load run.

The native vLLM interface currently represents only a contiguous recovered
prefix. Consequently, the number of CacheFlow tail-load chunks is mapped to
an equal-size prefix load, and H2D completes before suffix prefill. This is a
runnable single-GPU CacheFlow-H adaptation, not the paper's exact arbitrary
tail-page, layer/GPU-parallel, overlapped executor; that remains a separate
implementation requirement.

The paper was released in April 2026 and no public source artifact was located
in this audit.  Therefore this is a **paper-level scheduler baseline**, not a
source-level port.  It must not be reported as CacheFlow's full 3D distributed
runtime.  A faithful end-to-end comparison additionally needs its layer/GPU
parallel executor, distributed transport and the same correctness/cache-state
gates as HyRex.

Structural check:

```bash
.venv/bin/pytest -q tests/v1/kv_offload/test_cacheflow_policy.py \
  --confcutdir=tests/v1/kv_offload
```
