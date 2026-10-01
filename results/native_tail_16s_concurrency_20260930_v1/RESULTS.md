# Qwen3.5-9B: 16-session recovery across concurrency 1/4/8/16

## Setup

- 16 independent real ShareGPT sessions, 5--10 turns each.
- 120 requests and 104 continuations per treatment and concurrency.
- Same-turn requests form one concurrent wave; a session's next turn is never
  submitted before its previous turn completes.
- GPU prefix cache is cleared after each wave; the 8 GiB LMCache CPU cache is retained.
- Native uses its required 528-token prefill budget. Tail uses the established
  single-forward 2048-token budget, 16-token Full-KV pages, and replacement tail state.
- Every concurrency has an unrelated seed/resume warmup at the same batch size.
- Concurrency 1 comes from the immediately preceding run; 4/8/16 share one resident
  service per treatment to avoid repeated model-load effects.

## End-to-end result

| Concurrency | Native mean TTFT | Tail mean TTFT | Delta | Native P95 | Tail P95 | Native recovery req/s | Tail recovery req/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 159.38 | 167.84 | +5.3% | 216.74 | 189.45 | 6.27 | 5.95 |
| 4 | 323.07 | 358.27 | +10.9% | 486.01 | 466.72 | 9.68 | 9.49 |
| 8 | 472.41 | 576.26 | +22.0% | 733.30 | 783.00 | 11.70 | 11.29 |
| 16 | 738.42 | 915.89 | +24.0% | 1305.23 | 1375.31 | 12.46 | 11.00 |

Recovery throughput excludes the artificial two-second cache-reset sleeps and uses
the maximum request completion time in each wave. TTFT includes real in-wave queueing.

## Recomputed work

| Concurrency | Native replay | Tail replay | Reduction |
|---:|---:|---:|---:|
| 1 | 391.5 tok | 159.2 tok | 59.3% |
| 4 | 391.5 tok | 159.2 tok | 59.3% |
| 8 | 391.5 tok | 159.2 tok | 59.3% |
| 16 | 401.6 tok | 159.2 tok | 60.4% |

At concurrency 16, native loses one 528-token checkpoint for two consecutive turns
of one session (`QpV93PD_42`), increasing its mean replay by 10.2 tokens. Tail retains
the expected hit for every concurrency.

## Where tail still helps

| Concurrency | Avoided replay range | Samples | Mean tail-native TTFT |
|---:|---:|---:|---:|
| 1 | 384--512 | 20 | -18.96 ms |
| 4 | 384--512 | 20 | -19.05 ms |
| 8 | 384--512 | 20 | +58.06 ms |
| 16 | 384+ | 22 | +111.45 ms |

At concurrency 4, choosing tail only when it avoids at least 384 tokens lowers mean
TTFT from 323.07 to 319.41 ms (1.13%). At concurrency 8 and 16, no ordinary
replay-gap threshold makes the current tail implementation profitable.

## Why the benefit disappears at high concurrency

The newly enabled multi-request path is correct in its ownership model but is not a
batched checkpoint kernel. For every GDN layer it currently:

1. slices each request from the packed prefill;
2. runs that request's recurrent segments separately;
3. clones its checkpoint state;
4. concatenates all outputs back into packed order.

Native executes the packed recurrent batch together. Tail therefore replaces roughly
one packed recurrent invocation per layer with up to `N` request-local invocations
(and sometimes two segments per request). Kernel launches, tensor materialization and
loss of GPU occupancy grow with concurrency, overwhelming the 59% token reduction.
This is an implementation cost, not an index lookup cost.

## Correctness and measurement warnings

- Prompt-token counts match for every native/tail pair.
- First-token mismatches: 1/120 at concurrency 1 and 2/120 at each of 4/8/16.
- Every mismatch is a near-tie among the same Top-5 candidates; several native logits
  are exactly tied at the displayed precision. This is compatible with BF16 batching
  and segmentation order, but remains a correctness warning until full-logit tolerance
  and longer-generation tests pass.
- Formal-request JIT warnings: native has one at concurrency 1 and one at concurrency
  4; all other native levels and all tail levels have zero.
- LMCache falls back to pickle because the container has only 64 MiB `/dev/shm`.
  Both treatments share this constraint, but serialization can disproportionately hurt
  the path that moves more objects.

## Required optimization

The next implementation should preserve independent request boundaries without
serializing requests:

1. Build a packed segmented-recurrent call using `cu_seqlens`, grouping requests by
   checkpoint segment and executing one kernel per segment depth, not per request.
2. Scatter all final recurrent/conv states into destination checkpoint blocks with one
   batched copy kernel per layer.
3. Keep fine-grained lookup but coalesce Full-KV and state transfers across requests.
4. Let the scheduler disable tail creation/recovery when predicted batching loss or
   PCIe contention exceeds saved replay compute.

The main systems result is therefore: **deeper recovery reduces work, but a recovery
mechanism that breaks batching becomes increasingly worse with concurrency.** A useful
HyReX design must be concurrency-aware and batch-preserving, not merely finer-grained.

