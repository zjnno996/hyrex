# Execution-Order State Streaming for Hybrid KV Recovery

## The idea

**Execution-Order State Streaming (EOSS)** restores Hybrid cache state by
*decoder execution position*, rather than by state type.  It bundles the
linear/GDN state and FullAttention KV needed by a small contiguous decoder
tile, prefetches those bundles in execution order, and waits for each bundle
only immediately before its tile runs.

For Qwen3.5-9B, the decoder repeats eight times:

```
Linear(3 layers) -> FullAttention(1 layer)
```

EOSS uses each four-layer pattern as its natural recovery tile.  While tile
`i` computes the new suffix, the H2D stream transfers tile `i+1`.  A
prefetch-window controller selects how many later tiles may be in flight.

## Why this, instead of "Full Load + Linear Replay"

Pure KV/state restoration cannot skip an arbitrary intermediate decoder
layer.  The next layer needs the preceding layer's hidden/residual output for
the new suffix, even if its historical KV or recurrent state was restored.
Qwen's Linear and Full layers are interleaved, so a state-type decision is
not an executable compute cut.  This explains the existing P3/P4 result:
they route the intended state groups, but inherit replay/alignment scheduling
and do not remove the full dependency chain.

The current LMCache path also restores all object groups under one strict
H2D barrier before the first forward operation.  It therefore serializes:

```
copy(tile 0..7) -> compute(tile 0..7)
```

EOSS changes the schedule, not the model semantics:

```
copy(tile 0) -> compute(tile 0) || copy(tile 1) -> compute(tile 1) || ...
```

Its ideal recovery time is bounded by the copy/compute pipeline rather than
their sum:

```
T_EOSS ~= copy(tile 0) + sum_i max(compute(tile i), copy(tile i+1))
```

No activation checkpoint is required, because every tile still executes for
the new suffix.  Activation checkpoints are a separate extension for
skipping tiles entirely.

## Evidence from the current Qwen3.5-9B system

The model has 32 decoder layers: eight `linear_attention x3 + full_attention`
tiles.  Under a verified CPU hit of 1584 tokens and GPU prefix of 528 tokens,
the missing interval is 1056 tokens.  Real LMCache local-CPU H2D tracing
measured:

| Object group | State family | H2D bytes |
| --- | --- | ---: |
| 0 | Mamba/GDN | 34,603,008 B |
| 1 | Mamba/GDN | 34,603,008 B |
| 2 | Mamba/GDN | 34,603,008 B |
| 3 | FullAttention | 34,603,008 B |

The total is 132 MiB.  Thus this model does **not** support the simplistic
claim that the FullAttention group is inherently the large state and every
Linear group is inherently small.  The central asymmetry is dependency and
readiness: all four state families are needed repeatedly along the decoder
path, but they are currently materialized as one barrier.

### Phase-1 structural prototype validation (implemented)

We implemented an opt-in four-layer execution-tile layout and exercised it
through the real LMCache local-CPU recovery path, with the same verified
`GPU=528, CPU=1584, H2D=1056` token cell.  LMCache registered eight object
groups (one per tile), and each group contained the two state objects needed
by that tile.  Every group was restored successfully:

| Layout | Object groups | Bytes per group | Total H2D bytes | First-token signature |
| --- | ---: | ---: | ---: | --- |
| Existing state-family layout | 4 | 34,603,008 B (33 MiB) | 138,412,032 B (132 MiB) | `f194f88525b7a18f` |
| EOSS tile layout (4 layers/tile) | 8 | 17,301,504 B (16.5 MiB) | 138,412,032 B (132 MiB) | `f194f88525b7a18f` |

This validates that the altered execution-tile *engine configuration* can
complete CPU lookup, H2D transfer, and output generation. It does **not**
validate compatibility with the baseline CPU object format: the prototype also
changes physical kernel/page groups. The current runtime still applies its
global strict wait, so this is a structural correctness result only, not an
EOSS latency result. A correct Phase 2 must introduce cache-plane slices,
per-tile completion events, and just-in-time waits while retaining baseline
engine groups.

The first C1/C8 comparisons must not be interpreted as a tile-object overhead
measurement. Inspection of the worker metadata showed that the current
prototype changes *both* planes: it turns four physical 8-layer kernel/page
groups into eight 4-layer kernel/page groups and then makes one object per
group. Thus its 114.326 ms versus 130.745 ms C1 result (and its C8 result) is
a structural-reconfiguration probe, not a causal EOSS performance result.

This discovery sharpens EOSS's required abstraction: it is a
**readiness-overlap** proposal, not a layout compression proposal. The engine
plane must retain its original page tensors, scheduler groups, and transfer
kernel geometry; only the cache plane may expose execution-order pieces.

## Minimal implementation plan

### Phase-2a: cache-plane LayerSlice transfer (implemented and verified)

The first Phase-2 mechanism now exists in the active LMCache runtime.  The
new `VLLM_MOONCAKE_HYBRID_CACHE_SLICE_SIZE=4` switch is deliberately distinct
from the old `...EXECUTION_TILE_SIZE` switch: it leaves vLLM's physical engine
groups unchanged and constructs CPU-cache objects from `(kernel_group_id,
layer_positions)` slices.  The Python transfer fallback copies each slice with
a pointer subset and a `PageBufferShapeDesc` whose `nl` equals the slice's
layer count.  Native object-group transfer is intentionally bypassed for a
slice until it receives the same descriptor support.

One real Qwen3.5-9B BF16 local-CPU P1 recovery was run with `GPU=528`,
`CPU=1056`, `need H2D=528`, C1, 256 output tokens, 0.9 GPU utilization, and a
24-GiB SLRU CPU tier.  It establishes the following facts:

| Cache-plane layout | Physical kernel groups | H2D objects | Object bytes | Total bytes | First-token signature |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline state-family | 4 | 4 | 17,301,504 B | 69,206,016 B | `9cb3ee3a671b7e05` |
| LayerSlice, 4 layers | 4 | 8 | 8,650,752 B | 69,206,016 B | `9cb3ee3a671b7e05` |

The worker registered the slice layout as
`kernel_groups=4, object_groups=[[0],[0],[1],[1],[2],[2],[3],[3]]`; the
repeated physical-group identifiers are expected, and demonstrate that only
the cache plane changed.  All eight slice objects were retrieved successfully.
The matching first-token signature against the unsliced baseline establishes
the required transfer correctness.  The observed one-shot H2D/TTFT timings
are **not** reported as a performance comparison: both cells included
first-use Triton JIT and independent service startup.

This is still a barrier-load implementation.  It makes execution-ready
objects available but retains the global strict H2D wait, so it cannot yet
produce EOSS overlap or a latency gain.  The next required feature is the
per-tile completion event and Mamba/GDN entry wait described below.

### EOSS software MVP (implemented and functionally validated)

The required software path now exists behind an opt-in flag:

```bash
--hybrid-cache-slice-size 4 --hybrid-eoss --policies P1
```

It preserves the four physical vLLM groups, creates eight cache-plane tile
objects, and sends one LMCache retrieve per tile. The scheduler sees only the
initial-window future: it must complete before vLLM can enter the model. The
Qwen decoder then calls the existing connector wait hook at **every** layer
entry, including Mamba/GDN layers; later tiles are submitted and waited there.
The first layer of each four-layer tile waits only for that tile. The normal
strict global H2D barrier is rejected when EOSS is enabled.

`--hybrid-eoss-window w` is now a bounded prefetch controller (default `w=2`):
`start_load` submits tiles `[0, w)`, then entry to tile `i` submits tile
`i+w` before waiting for tile `i`. Thus the copy of a future tile can overlap
the current tile's suffix compute without pre-queuing all eight transfers.
A real E2E C1 run now completes on the healthy single-GPU path. All eight
tiles are observed at their layer-entry waits and the output signature agrees
with the strict barrier control. Initial one-shot measurements are recorded
in `QWEN35_HYBRID_RECOVERY_RESULTS.md`; they are functional/motivating
measurements rather than a final latency claim because they use independent
processes and include first-use JIT/startup effects.

1. **Completed:** introduce a cache-plane `LayerSlice` descriptor:
   `(kernel_group_id, layer_indices)` inside an unchanged physical kernel
   group. A tile object is a list of these slices, not a new engine group.
2. **Completed:** make the H2D/D2H transfer kernel operate on a `LayerSlice` so
   object keys and completion events can be per tile while vLLM's original
   four kernel groups remain intact.
3. **Completed:** submit tile H2Ds on one copy stream with one completion event per
   tile, then replace the global strict wait with a per-tile wait at the first layer of
   that tile.  This is required for correctness; merely disabling strict wait
   is not a valid experiment.
4. **Current experiment:** use prefetch window `w` as the initial controller. `w=8` reproduces
   all-load; `w=1` is fully just-in-time; `w=2/4` expose overlap.

The only adaptive decision needed at first is the window size: increase it
when H2D is slower than preceding tile compute, decrease it under GPU-memory
or H2D-queue pressure.  This is more faithful than forcing long segments to
full replay.

## Phase-2 implementation boundary

The layout alone is intentionally insufficient. Inspection of the active MP
path identifies the two mechanisms that EOSS must add:

1. A normal `LoadStoreOp` currently requests every object group and yields one
   completion future. EOSS needs an internal `object_group_ids`/`tile_id`
   field so one logical prefix lookup can produce eight tile retrieve jobs,
   each with its own CUDA completion event. The CPU object keys and block-ID
   lists remain unchanged; only the selected cache-plane `LayerSlice` subset
   is passed to the H2D worker. The request is reported complete only after
   every tile job completes or fails.
2. vLLM's existing `wait_for_layer_load` hook currently surrounds
   FullAttention only. A Qwen3.5 tile begins with a Mamba/GDN layer, so its
   first state-consuming operation needs the same model-neutral wait hook.
   The hook maps `layer_name -> tile_id`, waits for that tile's event, and
   submits the next tile when the prefetch window permits it. Waiting only at
   FullAttention would be incorrect: the first three Linear layers could read
   their recurrent state before its copy completed.

This yields a small, correctness-preserving MVP schedule for `w=1`:

```
start_load: submit tile 0
tile i entry: submit tile i+1 (if any); wait(tile i); execute tile i
```

The copy stream orders `tile i -> tile i+1`; after `wait(tile i)` releases
the compute stream, the H2D of tile `i+1` can proceed concurrently with tile
`i`'s suffix work. `w>1` simply allows more future tile jobs to be submitted.
This is safe only after per-tile events and the Mamba entry hook exist. An
environment flag that merely removes the global wait is not an EOSS variant
and must not be benchmarked as one.

## Evaluation

Compare only correctness-safe variants:

| Variant | Meaning |
| --- | --- |
| Barrier Load | Current strict all-group H2D before forward |
| EOSS w=1/2/4/8 | Per-tile event-driven H2D/compute pipeline |
| All Replay | Compute-only control |
| Activation-assisted skip | Separate extension, with checkpoint cost included |

Use verified `GPU prefix hit + CPU suffix hit` cells; report per-tile copy
service/wait, hidden copy time, scheduler queue, TTFT, and logits/first-token
correctness.  Sweep recoverable interval (528/1056/... tokens) and C1/C8/C10.
The current local pinned-CPU system is expected to show limited gain because
H2D is already cheap; a throttled or remote/contended tier is needed to expose
the overlap benefit.  That limitation is an experimental result, not an
assumption.
