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
