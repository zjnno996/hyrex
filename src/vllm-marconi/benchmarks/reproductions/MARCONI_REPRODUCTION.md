# Marconi reproduction branch

This branch is an isolated, clean-room adapter for the Marconi Hybrid cache
policy.  The upstream artifact is used only as an external reproduction
reference; its CC BY-NC source is not copied into vLLM.

## Artifact smoke reproduction

Reference repository:

```text
https://github.com/ruipeterpan/marconi
```

Reference commit used for the smoke run:

```text
08016617b1524e6bf6ac29b680641cc945bda7f0
```

Command:

```bash
marconi_dir=$(mktemp -d /tmp/marconi-artifact.XXXXXX)
git clone --depth 1 https://github.com/ruipeterpan/marconi.git "$marconi_dir"
.venv/bin/python "$marconi_dir/toy_example.py"
```

Environment used here:

```text
Python: project .venv
PyTorch: 2.11.0+cu128
GPU: not required by toy_example.py
```

Observed behavior:

* the first Princeton prompt creates one radix-tree leaf;
* prompts with the same first five tokens reuse that shared prefix;
* changing the suffix creates sibling nodes below the shared prefix;
* a shorter prompt shares no more than its exact token prefix;
* later insertions create additional branch-off nodes and state checkpoints.

This validates the artifact's basic Hybrid radix-tree prefix matching and
branching behavior.  It is not the paper's full workload result: the full
LMSys/ShareGPT/SWEBench trace sweep requires the separately distributed trace
archive and the artifact's vLLM+ environment.

## vLLM adapter status

The branch contains a clean-room `MarconiPolicy` implementing the paper's
retention utility over the vLLM cache-entry interface, plus a compressed
`MarconiIndex` that preserves the original split/merge behavior:

* a partial branch splits a compressed edge;
* a node with one child can be evicted by absorbing that child edge;
* V2/V3 touch only the matched terminal node;
* branching nodes are protected until their descendants are removed.

```text
normalized recency
+ alpha * normalized compute-savings / bytes
```

It is registered dynamically as `marconi`, so it does not alter the HyRex
branch or other baseline branches.  The Marconi branch also exposes
`eviction_policy=marconi` through the native CPU offloader.  Its block adapter
preserves ref-count and protected-block safety; when a scheduler supplies
Hybrid metadata, `observe_metadata` enables the recency/compute-savings
utility.  Without that metadata it intentionally falls back to deterministic
block recency rather than inventing state costs.  The structural checks can be run
without a GPU:

```bash
.venv/bin/pytest -q tests/v1/kv_offload/test_marconi_index.py \
  --confcutdir=tests/v1/kv_offload
```

The current run passed 4 tests in 39.88s.  The next reproduction step is to
feed the same token traces through the vLLM adapter and compare hit rate and
TTFT against LRU/SLRU under a matched cache capacity.  Those results must be
kept separate from recovery-scheduler results.

## Artifact-to-adapter semantic check

The following CPU-only verifier loads the upstream radix artifact and compares
each prefix lookup and post-insert node count with the clean-room adapter on a
trace containing exact matches, mid-edge branch-offs, and extension requests:

```bash
.venv/bin/python benchmarks/reproductions/verify_marconi_artifact.py \
  /path/to/marconi-artifact
```

It is a structural/semantic reproduction gate only. It does not establish
cache capacity behavior, cache-hit rate on a paper trace, output correctness,
or TTFT.
