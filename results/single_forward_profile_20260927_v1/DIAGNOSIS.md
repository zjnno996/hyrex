# Recovery slowdown diagnosis

## Scope

No model, connector, or pristine baseline source was changed. Reused frozen commands with eight real requests per arm and profiled request index 5: ShareGPT `Pr8nMeM_0`, second turn, 831 input tokens. Its previous prompt has 756 tokens, giving a complete Full-KV boundary of 752 and coarse-state boundary of 528. Each service received the existing unrelated warmup and GPU prefix resets. All three profiled first-token texts agree.

Scripts: `/root/hyrex_results/profile_single_forward_20260927.py` and `/root/hyrex_results/analyze_single_forward_profile_20260927.py`. Raw traces are under each arm's `profile/`. Machine-readable results: `profile_summary.json`.

## 1. Deeper KV saves a narrow operation, not the full replay forward

Native and deep both run 303 tokens (831-528). Deep reuses KV for the 224-token interval [528,752), but Q/gate, attention output, GDN and MLP still need to run to reconstruct exact state. The native Full-Attention projection is one `[303,4096] x [4096,10240]` GEMM; deep splits it into Q/gate `[303,4096] x [4096,8192]` and new-KV `[79,4096] x [4096,2048]`.

Across Full-Attention layers in this trace, projection kernel sums are 1.626 ms native versus 1.495+0.246=1.741 ms deep. Overall engine kernel sums are 47.224 versus 47.405 ms. Splitting reduces arithmetic but introduces less favorable shapes/additional launches. This concrete request does not show a GPU-time saving from Q-only projection.

The model has 32 layers, only 8 Full-Attention layers. MLP GEMMs remain unchanged between native and deep; their two principal GEMM groups alone total about 26.2 ms in this trace.

## 2. Tail state saves real GPU work but incurs expensive checkpoint execution

Tail restores both states at 752, running only 79 tokens. To preserve the next tail state at 816, each GDN layer runs 64 tokens then 15 tokens. This is one transformer forward, but two recurrent operator calls per GDN layer.

| Diagnostic metric | Deep | Tail |
|---|---:|---:|
| Forward tokens | 303 | 79 |
| GPU kernel sum, ms | 47.405 | 26.662 |
| GPU kernel count | 1882 | 2096 |
| GDN recurrent calls | 24 | 48 |
| Clone operations | 105 | 201 |
| Concatenations | 5 | 29 |
| Copy operations | 658 | 791 |
| GDN attention-core inclusive CPU duration, ms | 37.978 | 56.502 |
| Model execution CPU range, ms | 113.634 | 124.962 |

This is evidence of lower GPU arithmetic time but more host/operator work in checkpoint capture. Inclusive CPU times overlap; do not add rows to reconstruct TTFT. First-profile effects are visible in native CPU durations, so native CPU-range time is not used as a quantitative speed comparison.

Source: `qwen_gdn_linear_attn.py:1324` enters checkpoint capture, and `:1528` selects the segmented recurrence. `checkpoint_capture.py:24` calls the complete chunk operator separately for each segment, makes contiguous slices, clones state, and concatenates outputs.

Tail also adds 48 CUDA stream synchronizations in this trace, with about 0.201 ms summed API duration. A nested trace verifies `bool(has_initial_state[0])` performs a device-to-host scalar read. This is avoidable serialization but is not, by itself, evidence for a 19-ms synchronization penalty.

In the formal 108-continuation-request sample per arm, deep captures new checkpoints in only 24 requests; tail captures them in all 108. Thus a small per-capture cost applies much more frequently to tail. Deep capture/no-capture groups differ in prompt shape and cannot be treated as a controlled causal ablation.

## 3. Transfer/control path is also heavier than native, but does not explain tail-vs-deep by itself

Formal LMCache retrieve-handler logs: native has one call per continuation, experimental arms two (state plus Full KV, in different order). The summed handler-duration means per continuation are approximately 2.94 ms native, 9.56 ms deep, 9.36 ms tail. These are rounded host handler measurements, not complete GPU H2D times or additive TTFT components. The last 108/216/216 retrieve records correspond to formal requests; earlier records belong to warmup.

Tail additionally inherits serial lookup stages: Full-KV lookup -> tail-state namespace lookup -> parent ordinary-state lookup. The ordinary-state lookup is redundant on a verified tail hit because this mode stores state in independent namespaces. Each adapter submission/status check can wait on messaging futures. This is a source-identified optimization opportunity, not a measured attribution of the whole regression.

## Prioritized next changes (not implemented in this diagnosis)

1. Reduce checkpoint-capture overhead while retaining the exact checkpoint: avoid unnecessary old-convolution-history reads when the offset already contains the full history; avoid redundant snapshot clones/copies and single-output concatenation; investigate a lighter exact recurrent tail kernel or intermediate-state output from one chunk call. Preserve separate snapshot lifetime until asynchronous STORE completes.
2. On a verified tail hit, remove the unused ordinary-state lookup; preserve existence verification, lock accounting, fallback, and correct state/Full-KV allocation.
3. For deep KV, compare fused QKV against split Q-only under matched restore/checkpoint settings; do not assume fewer FLOPs means lower latency.

Further unprofiled ablations are required to assign exact milliseconds or claim recovered TTFT gains. The current result demonstrates an implementation bottleneck, not that tail checkpointing is inherently faster or inherently useless.
