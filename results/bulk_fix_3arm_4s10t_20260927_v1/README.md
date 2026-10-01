# Full16 bulk-transfer repair

The existing baseline/decoupled/optimized worktrees are unchanged. New branch
`motivation/bulk-transfer-fix-20260927` exists in both repositories:

- vLLM harness: `/root/exp-vllm-bulk-fix`, tested commit `a021f685e`.
- LMCache repair: `/root/exp-lmcache-bulk-fix`, tested commit `237c78f`.

Repair scope: keep 16-token CPU keys, but gather/scatter GPU pages in staging-
capacity batches, independently of contiguous CPU allocation runs. Add D2H
batching, not just H2D. Copies honor current-stream ordering and lazy allocator
host-registration boundaries. Existing producer/completion events and all-
state-ready scheduling stay unchanged. Unsupported layouts, partially reserved
stores, GDS and incompatible sizes fall back before touching buffers.

Not implemented in this patch: combined lookup/retrieve RPC or layer-ready
compute overlap. Those must be isolated in later branches, not silently mixed
into this transfer ablation.

Preflight tests: 12 layout/round-trip/audit tests (CPU and CUDA, fragmented or
contiguous host objects, arbitrary GPU page IDs, multiple staging batches),
plus 2 CUDA copy stream-order tests passed. These do not replace model checks.

Three sequential arms: baseline; frozen previous deep Q-only/no-cat; repaired
deep Q-only/no-cat. Same GPU1 Qwen3.5-9B BF16 eager, CPU 2GiB, 4 sessions x
10 turns, unrelated warmup, 39 GPU prefix resets with CPU retained per arm.
One output token, first-token logprobs, 36 continuation TTFTs averaged per arm.
Baseline includes the shared CUDA fallback correctness fix and reset-success
reporting, not a byte-identical pristine upstream checkout. No profiler.
Exact source versions, trace hash, command and settings are in each design.json;
PYTHONSAFEPATH and imports.log prevent cwd import shadowing. Preserve outliers.
Single sequential passes cannot establish statistical significance.
