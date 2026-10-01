# Baseline feasibility audit

This audit is code-level only.  It does not claim an end-to-end experiment.
Each baseline remains in its own branch and uses the lazy policy registry in
`vllm/v1/kv_offload/recovery_policy.py`; no baseline changes another
baseline's execution path.

| Baseline | Current status | Feasible next implementation | Not valid to claim yet |
| --- | --- | --- | --- |
| Marconi | Native cache-policy adapter and compressed radix index are implemented. | Feed identical token traces to the adapter and compare admission/eviction against LRU/SLRU. | Paper workload numbers or recovery scheduling results. |
| KVPR | The published analytical split is reproduced. | A homogeneous Transformer executor can regenerate a chosen prefix interval from a verified layer-input activation while loading the complementary KV interval. | A full Hybrid Attention implementation, or end-to-end KVPR in Qwen3.5 without an activation-to-KV writer. |
| HCache | A paper-level split planner exists; the main worktree has safe hidden/residual checkpoint storage. | Add a layer-input checkpoint format and write projected K/V directly into the exact vLLM page/slot mapping, with stream fences. | Treating the existing post-layer restore hook as HCache: it skips a decoder layer and is activation-assisted layer skipping, not HCache KV reconstruction. |
| CacheFlow | A paper-level two-resource scheduler exists. | Supply real per-unit dependency metadata and dispatch already-correct load/replay jobs in that order. | The paper's full 3D distributed runtime, layer/GPU parallel executor, or its reported numbers. |

## Why KVPR and HCache are not drop-in Hybrid policies

`Qwen3NextAttention` has a fused `qkv_proj` and vLLM owns the physical KV page
update through `unified_kv_cache_update`.  The public connector exposes cache
transfer, but no API accepts a verified activation interval and writes only
the corresponding K/V range into those pages.  The Hybrid model additionally
contains GDN/Mamba recurrent layers, whose state is not a token-indexed K/V
range.  Therefore applying a Transformer-only KVPR/HCache executor to every
Hybrid group would be semantically wrong.

The existing `HybridActivationContext` captures post-layer hidden/residual and
restores them immediately before the same layer.  Its documented behavior is
to skip that whole layer.  It is useful for an explicitly separate
activation-assisted experiment, but must remain outside P1/P2/P5 and HCache
comparisons until a K/V-only reconstruction path is added.

## Minimal gates before an end-to-end baseline result

### Marconi

1. Use the upstream token trace and fixed cache capacity.
2. Verify prefix-match, split/merge, and eviction sequence, then compare cache
   hit rate separately from recovery TTFT.

### KVPR / HCache

1. Restrict the first executor to a homogeneous Full-Attention model or to
   Qwen3.5's `full_attention` groups only; recurrent groups remain on their
   verified P1/P5 path.
2. Record checkpoint provenance `(model, dtype, layer, token interval)` and
   reject partial or stale checkpoints.
3. Use the same `slot_mapping` as normal attention to write the regenerated
   K/V pages; never bypass page ownership.
4. Fence the projection/write stream before the first attention read.
5. Require P1/P2/baseline first-token and full-output agreement, and account
   separately for activation H2D, K/V projection, and fallback replay.

### CacheFlow

1. Build units only from the already-correct P1/P2/P5 actions.
2. Record each unit's bytes, replay cost, source readiness, destination key,
   dependency predecessor, and request deadline.
3. Verify the dispatched order equals the scheduler plan and preserve the
   normal vLLM completion fence.
4. Compare p50/p99 TTFT and throughput only after all requests satisfy the
   cache-state and output gates in `goal.md`.

## Decision

All four baselines are feasible as **separate comparison tracks**.  Marconi is
the only one with a native vLLM cache-policy adapter today.  KVPR and HCache
need an activation-to-KV executor before they can produce fair end-to-end
numbers; CacheFlow needs the per-unit observation/dispatch adapter before it
can be called a runtime scheduler.  HyRex's main P1/P2/P5 recovery result does
not depend on pretending that any of these missing executors already exists.

## Reproducible CPU semantic gates

From the Marconi worktree, run all four isolated baseline checks with:

```bash
.venv/bin/python benchmarks/reproductions/run_baseline_semantic_checks.py
```

To add the public Marconi artifact-to-adapter parity check, pass its clone:

```bash
.venv/bin/python benchmarks/reproductions/run_baseline_semantic_checks.py \
  --marconi-artifact /path/to/marconi
```

Neither command starts a model server, allocates KV cache, or produces a
serving-performance result.
