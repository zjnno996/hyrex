# Stock vLLM + LMCache hybrid-prefix motivation result

## Setup

- Clean upstream vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`; official LMCache PyPI `0.5.3`.
- Qwen3.5-9B BF16, text-only, one RTX 4090; official `lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector` with Hybrid KV manager enabled.
- LMCache MP CPU L1: 8 GiB, 528-token chunks, separate object groups, LRU.
- ShareGPT: 32 sessions with three turns each, 96 seed prompts followed by 96 resume prompts. Generation is one token per request so it minimally perturbs the stored prompt state.
- After seeding, `POST /reset_prefix_cache?reset_external=false` clears only vLLM's local prefix cache. The same LMCache MP server and CPU cache remain alive for all resume requests.
- Raw measurements: `/root/hyrex_results/motivation_lmcache_boundary_clean/sharegpt_mp_default_32s/{seed.jsonl,resume.jsonl,audited.jsonl,summary.json,vllm.log,lmcache.log}`.

## Measured result

The vLLM log reports `Successfully reset prefix cache`. All **96** post-reset resume requests have an LMCache prefetch completion from **L1 CPU** (zero L2 keys). The prompt-token counts returned by vLLM match the trace tokenizer counts for both phases.

For each resume prompt, the audit re-tokenizes all earlier seed and resume prompts and finds its longest actual common token prefix `P`. In all 96 cases, the observed cache hit is exactly `floor(P/528) × 528`; none fell below this boundary. Thus CPU capacity/eviction did not reduce the recoverable prefix in these samples.

| Metric | Observed |
|---|---:|
| Sessions / resume requests | 32 / 96 |
| 528-token hits | 47 |
| 1,056-token hits | 36 |
| 1,584-token hits | 13 |
| Requests with a non-recovered prior-prefix tail | 96 / 96 |
| Tail length, mean / median / maximum | 258.54 / 236 / 511 tokens |
| Total prior-prefix tail across requests | 24,820 tokens |

Example: session `ipe1nhQ_5`, turn 2, has a 1,532-token common prefix with an earlier prompt, but restores only 1,056 tokens; 476 prior-prefix tokens are recomputed. These are **not** 476 Full-KV tokens proven to be present in CPU memory.

## Mechanism and what this does *not* prove

The clean vLLM runtime logs that it raises the Full-Attention page size to **528 tokens** to match the GDN/mamba page size. The official MP connector's `LMCacheMPRequestMetadata.GetStoreMetadata` takes the minimum available length across all engine groups and emits only full 528-token chunks. Its lookup reports one chunk-aligned prefix. Consequently, this working baseline couples heterogeneous states at the **storage and lookup interface**, before any independent Full-KV recovery choice can be exercised.

The official MP connector contains no reads of `discard_partial_chunks` or `save_unfull_chunk`. Those switches belong to another LMCache connector path; for this Hybrid-capable MP path, a “tail-retention enabled” configuration is not a valid independent treatment. The older `LMCacheConnectorV1` path fails the Hybrid-manager compatibility check, but that does **not** mean stock vLLM + LMCache cannot run Qwen3.5-9B: the official MP connector ran this workload successfully.

This experiment therefore supports the narrower, paper-safe claim: **stock Hybrid offload/recovery exposes only a common 528-token boundary, leaving reusable conversational prefix tails outside the CPU hit**. It does not show `L_KV > L_S` in CPU, does not establish that a deeper stored Full KV was discarded by the scheduler, and does not establish the isolated TTFT cost of the tail. HyRex's first technical step should include independent Full-KV and recurrent-state *storage/indexing* as well as decoupled recovery, not only replay after an assumed deeper KV hit.

## Continuous multi-turn replay (additional experiment)

The preceding seed/resume experiment is **not** an online dialogue replay. For an online trial, the same 32 three-turn ShareGPT sessions were replayed once in original `arrival_index` order, interleaving users while preserving each session's turn order. Each event sent its `resume_prompt`, generated 16 tokens, and completed before the next event. No manual GPU reset or concentrated seed phase was used. The CPU L1 limit was 2 GiB; GPU KV capacity was 17,261 tokens. This is multi-user interleaving at **concurrency 1**, not a high-concurrency throughput benchmark. The next turn uses the reference ShareGPT conversation (teacher-forced trace), not the model's newly generated 16-token reply.

| Turn | Requests | Requests with a prior ≥528-token aligned prefix | Requests with a cache hit | Median request latency, including 16-token decode |
|---|---:|---:|---:|---:|
| 0 | 32 | 0 | 0 | 621.19 ms |
| 1 | 32 | 32 | 6 | 718.45 ms |
| 2 | 32 | 32 | 19 | 740.18 ms |

All 96 requests succeeded and generated exactly 16 tokens. Every nonzero API cache hit remained a multiple of 528. An independent tokenizer audit found that all 64 later-turn prompts had at least one complete 528-token common-prefix block with a previously sent prompt, yet **39/64** restored fewer complete blocks than that prefix permits (26 in turn 1; 13 in turn 2). The LMCache server logged 25 successful external prefetches, all from CPU L1 and none from L2, and repeatedly triggered L1 eviction above its 80% watermark. This establishes CPU-cache pressure in the online trial, but the logs do not attribute every individual missed block uniquely to eviction versus other write/lookup effects. API `cached_tokens` is not a per-request GPU-vs-CPU breakdown.

The first request took 23.36 s due to cold-start/JIT work and is not a steady-state latency sample. Latency medians are descriptive only: prompt lengths increase across turns, and this trial has no same-request uncached control. It also cannot be numerically compared as a controlled cache-policy experiment against the 8 GiB concentrated seed/resume run because both workload order and CPU capacity changed.

Raw online records and tokenizer audit: `/root/hyrex_results/motivation_lmcache_boundary_clean/sharegpt_mp_online_32s_2gb/{online.jsonl,online_audited.jsonl,online_summary.json,lmcache.log,vllm.log}`.

## Controlled 4-session, 10-turn CPU recovery and TTFT

Four fixed real ShareGPT sessions were replayed one at a time, ten turns per session (40 requests, concurrency 1). Each request generated one token. After every request except the last, the runner waited two seconds and called `reset_prefix_cache?reset_external=false`; the GPU prefix cache was cleared while the same 2 GiB LMCache MP CPU L1 stayed alive. The next turn used the dataset's reference conversation, not the model's newly generated token. The clean vLLM source and official LMCache MP connector, model, 528-token chunks, and 4096-token context limit were otherwise unchanged.

All 40 requests succeeded. The vLLM log confirms 39 local prefix-cache resets. The LMCache log records 36 successful prefetches for the 36 resumed turns, all from CPU L1 (none from L2), with no observed eviction event. Exact Qwen3.5-9B retokenization shows that every resumed request restored precisely `floor(P / 528) * 528` tokens, where `P` is its longest same-session historical token prefix. Thus the previous 528-alignment table is valid for this stock connector's *common stored/retrievable prefix*. It does **not** establish a deeper independently stored Full-Attention KV prefix in CPU than recurrent state.

| Session | Resume turns | Historical tail not restored, total tokens | Median resumed TTFT |
|---|---:|---:|---:|
| `Ud3L2sd_124` | 9 | 2,428 | 156.89 ms |
| `Pr8nMeM_0` | 9 | 2,237 | 163.99 ms |
| `WRAImOg_0` | 9 | 2,749 | 162.15 ms |
| `WnjND3T_0` | 9 | 2,552 | 158.88 ms |
| **All** | **36** | **9,966** | **160.25 ms** |

For every turn below, `P` is the exact common token prefix with an earlier turn of the **same session**, `H` is the observed CPU-restored prefix, and `P-H` is historical text that had to be recomputed. `Prompt` includes both historical and newly added tokens; it is not the amount of historical text eligible for reuse.

| Session | Turn | Prompt | P | H | P-H | TTFT (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `Ud3L2sd_124` | 0 | 651 | 0 | 0 | 0 | 29,194.47 |
| `Ud3L2sd_124` | 1 | 732 | 651 | 528 | 123 | 506.66 |
| `Ud3L2sd_124` | 2 | 830 | 732 | 528 | 204 | 143.08 |
| `Ud3L2sd_124` | 3 | 889 | 830 | 528 | 302 | 125.52 |
| `Ud3L2sd_124` | 4 | 942 | 889 | 528 | 361 | 124.45 |
| `Ud3L2sd_124` | 5 | 1,054 | 942 | 528 | 414 | 153.90 |
| `Ud3L2sd_124` | 6 | 1,191 | 1,054 | 528 | 526 | 212.48 |
| `Ud3L2sd_124` | 7 | 1,708 | 1,191 | 1,056 | 135 | 1,295.34 |
| `Ud3L2sd_124` | 8 | 1,823 | 1,708 | 1,584 | 124 | 156.89 |
| `Ud3L2sd_124` | 9 | 1,922 | 1,823 | 1,584 | 239 | 160.01 |
| `Pr8nMeM_0` | 0 | 756 | 0 | 0 | 0 | 182.54 |
| `Pr8nMeM_0` | 1 | 831 | 756 | 528 | 228 | 145.73 |
| `Pr8nMeM_0` | 2 | 994 | 831 | 528 | 303 | 138.29 |
| `Pr8nMeM_0` | 3 | 1,096 | 994 | 528 | 466 | 211.12 |
| `Pr8nMeM_0` | 4 | 1,259 | 1,096 | 1,056 | 40 | 160.01 |
| `Pr8nMeM_0` | 5 | 1,336 | 1,259 | 1,056 | 203 | 151.67 |
| `Pr8nMeM_0` | 6 | 1,514 | 1,336 | 1,056 | 280 | 163.99 |
| `Pr8nMeM_0` | 7 | 1,645 | 1,514 | 1,056 | 458 | 222.03 |
| `Pr8nMeM_0` | 8 | 1,782 | 1,645 | 1,584 | 61 | 172.22 |
| `Pr8nMeM_0` | 9 | 1,954 | 1,782 | 1,584 | 198 | 171.36 |
| `WRAImOg_0` | 0 | 872 | 0 | 0 | 0 | 187.35 |
| `WRAImOg_0` | 1 | 903 | 872 | 528 | 344 | 131.74 |
| `WRAImOg_0` | 2 | 973 | 903 | 528 | 375 | 143.84 |
| `WRAImOg_0` | 3 | 1,031 | 973 | 528 | 445 | 142.33 |
| `WRAImOg_0` | 4 | 1,081 | 1,031 | 528 | 503 | 284.00 |
| `WRAImOg_0` | 5 | 1,174 | 1,081 | 1,056 | 25 | 160.48 |
| `WRAImOg_0` | 6 | 1,282 | 1,174 | 1,056 | 118 | 162.15 |
| `WRAImOg_0` | 7 | 1,340 | 1,282 | 1,056 | 226 | 185.73 |
| `WRAImOg_0` | 8 | 1,485 | 1,340 | 1,056 | 284 | 162.50 |
| `WRAImOg_0` | 9 | 1,740 | 1,485 | 1,056 | 429 | 216.09 |
| `WnjND3T_0` | 0 | 849 | 0 | 0 | 0 | 187.49 |
| `WnjND3T_0` | 1 | 927 | 849 | 528 | 321 | 135.17 |
| `WnjND3T_0` | 2 | 979 | 927 | 528 | 399 | 145.62 |
| `WnjND3T_0` | 3 | 1,199 | 979 | 528 | 451 | 212.35 |
| `WnjND3T_0` | 4 | 1,254 | 1,199 | 1,056 | 143 | 157.71 |
| `WnjND3T_0` | 5 | 1,353 | 1,254 | 1,056 | 198 | 173.22 |
| `WnjND3T_0` | 6 | 1,529 | 1,353 | 1,056 | 297 | 165.12 |
| `WnjND3T_0` | 7 | 1,584 | 1,529 | 1,056 | 473 | 158.88 |
| `WnjND3T_0` | 8 | 1,854 | 1,584 | 1,584 | 0 | 154.13 |
| `WnjND3T_0` | 9 | 1,947 | 1,854 | 1,584 | 270 | 165.28 |

Turn 0 includes a 29.2-second cold-start/JIT outlier and is excluded from resumed-turn medians. Turn 7 is another latency outlier despite a small 135-token tail. The observed TTFT is **not** a controlled estimate of the tail's marginal cost: prompt length, CPU transfer, and runtime noise also vary. This experiment validates a reproducible boundary loss under verified CPU restoration; measuring HyRex's TTFT gain still requires the new recovery path and same-request paired runs, including checkpoint creation/transfer cost.

Raw trace: `/root/hyrex_results/motivation_sharegpt_4s10t_trace.jsonl`. Raw measurements and audit: `/root/hyrex_results/motivation_ttft_4s10t_cpu/{online.jsonl,online_audited.jsonl,online_summary.json,lmcache.log,vllm.log}`.

## Strict per-request GPU reset: stock versus independent Full-KV indexing

One controlled 9B trial used the same four ShareGPT sessions and ten turns each, round-robin by turn (40 sequential requests). Both runs used greedy one-token generation, 2 GiB LMCache CPU L1, eight unrelated warmup requests followed by a CPU-cache clear, and then a two-second wait plus a **confirmed successful** `reset_prefix_cache?reset_external=false` between measured requests. Each run has 39/39 successful measured GPU resets; the CPU cache and servers remained live. This is a single run per treatment, not a repeated-sample performance estimate.

The stock treatment used clean vLLM with official LMCache MP and the common 528-token boundary. The experimental treatment independently indexed Full-Attention KV at 16-token pages while retaining recurrent-state checkpoints at 528-token boundaries, with no artificial state cap. The only change in the stock vLLM checkout was to make the development reset endpoint return the engine's actual success boolean so the runner could reject failed clears; the stock inference/cache logic was unchanged.

| Metric | Stock vLLM + LMCache | Independent Full KV + 528-state |
|---|---:|---:|
| Completed requests | 40/40 | 40/40 |
| Confirmed measured GPU resets | 39/39 | 39/39 |
| Resumed requests with CPU cache hit | 36/36 | 36/36 |
| Sum of API-reported resumed cached tokens | 32,736 | 42,368 |
| Median resumed TTFT | 157.96 ms | 218.58 ms |
| Resumed requests with different first-token text from stock | reference | 16/36 |

The experimental API reported a deeper hit on 35 of 36 resumed requests (equal on one), adding 9,632 API-reported cached tokens in total; the median paired difference was 272 tokens. For example, session `Ud3L2sd_124` turn 1 restored 528 tokens in stock versus 640 experimental tokens. These counts demonstrate that independent CPU storage/lookup reaches past the common boundary. They **do not** prove 9,632 tokens of compute were saved: the experimental implementation can replay recurrent computation while preserving Full-KV pages, and `cached_tokens` is not a per-layer FLOP counter.

Critically, the current experimental recovery path is **not yet correctness-qualified**: greedy first-token text differs on 16 of 36 resumed requests, even though all four cold turns match. Therefore its 218.58 ms TTFT median is diagnostic only and cannot be presented as a valid speedup (it is slower even before resolving correctness). A one-token textual match on the other 20 resumed requests also does not establish full-state equivalence. The next implementation step is to fix and verify cross-boundary Full-KV/state replay, then repeat paired TTFT and checkpoint/transfer-cost measurement. No conclusion about TTFT benefit or loss from the proposed policy is justified by this run.

Raw records: `/root/hyrex_results/motivation_9b_clean_4s10t_strict_reset/{online.jsonl,resets.jsonl,warmup.jsonl,vllm.log,lmcache.log}` and `/root/hyrex_results/motivation_9b_full16_4s10t_strict_reset/{online.jsonl,resets.jsonl,warmup.jsonl,vllm.log,lmcache.log}`.
