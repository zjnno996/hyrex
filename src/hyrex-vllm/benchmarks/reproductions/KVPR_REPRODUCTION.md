# KVPR reproduction branch

This branch clean-room reproduces KVPR's profile-based partial-recomputation
split in vLLM's isolated recovery-policy extension point.

Upstream artifact: `https://github.com/chaoyij/KVPR`, commit
`1712a52042262fe1646799322e45792b109bd020`.  It is a FlexGen/OPT runtime,
not a vLLM implementation.  Its split scans `r` recomputed tokens and minimizes:

```text
activation_H2D(r) + max(KV_H2D(total-r), GPU_recompute(r))
```

The adapter uses the same BF16 activation/KV byte accounting and MHA FLOP
model, with its published default 32 GB/s H2D and 312 TFLOP/s GPU profile.
It accepts measured H2D bandwidth from `RecoveryTelemetry`.

Artifact-level check: the upstream function
`get_optimal_split_point(1024, 32, 5120)` returns `671`; the adapter's default
profile returns the same split.  At 64 GB/s the adapter intentionally selects
`499`, because that is a different measured transport profile rather than the
paper's 32 GB/s reference configuration.

## Executable vLLM adaptation

`VLLM_KVPR_ENABLED=1` enables a conservative executable adaptation in the
native CPU-offloading scheduler. It loads KV for an aligned contiguous prefix
and lets vLLM's ordinary prefill path compute the remaining suffix. The plan
is block-aligned before it changes the external-prefix length, so the
scheduler never claims a partial block was restored.

This is a runnable **KVPR contiguous-prefix adaptation**, not the paper's
activation-H2D plus overlapped K/V projection executor: vLLM does not expose a
safe activation-to-arbitrary-KV-page writer yet. It is intentionally rejected
for models containing Mamba/recurrent groups, including Qwen3.5 Hybrid,
because treating recurrent state as a token-indexed partial KV range would be
incorrect. Use it only in the homogeneous Full-Attention restoration table.

The branch retains the paper-level analytical executor as a dependency-model
test. A future exact KVPR executor requires layer-input checkpoint provenance,
partial KV page ownership, stream fences, and output equivalence against both
all-load and all-replay controls.

## Hybrid extension

The isolated branch `hyrec/baseline-kvpr-hybrid` adds `kvpr_hybrid`, a
Hybrid-aware baseline rather than claiming that the original KVPR supports
recurrent state. It aligns replay candidates to the 528-token recurrent
checkpoint boundary, loads the complementary Full-Attention KV tail, and
chooses either terminal recurrent-state load or replay. H2D and replay are
scored on their shared critical path.

The unified HyRex evaluation branch now binds a conservative executable form
after the real native CPU lookup and before external-token allocation. It
loads all Hybrid state for a checkpoint-aligned prefix and lets ordinary vLLM
prefill recompute the suffix. The request log records available, loaded, and
replayed token counts, so an all-load execution cannot be reported as a KVPR-H
split. This binding is output-correct by construction but serializes the H2D
fence and suffix prefill; it does **not** claim KVPR's layer-wise H2D/compute
overlap. That overlap still requires layer-input checkpoint provenance and a
safe partial KV-page writer.

Run the structural reproduction checks with:

```bash
.venv/bin/pytest -q tests/v1/kv_offload/test_kvpr_policy.py \
  --confcutdir=tests/v1/kv_offload
```
