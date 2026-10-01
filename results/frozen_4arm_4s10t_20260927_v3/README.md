# Isolated branch comparison

Qwen3.5-9B BF16 eager, GPU1, 2 GiB CPU cache, four real ShareGPT sessions,
ten turns each, round-robin but single request at a time. Unrelated warmup;
39 successful GPU prefix resets required per arm, CPU retained. One output
token with logprobs. Primary metric: arithmetic mean of 36 continuation TTFTs.
Keep all outliers; one sequential pass per arm is not statistical significance.

| Source | Worktree | Branch | Commit |
|---|---|---|---|
| vLLM baseline | /root/exp-vllm-baseline | motivation/baseline-20260927 | e582e17e6 |
| vLLM independent | /root/exp-vllm-decoupled | motivation/decoupled-20260927 | 3ec66cb48 |
| vLLM optimized | /root/exp-vllm-kv-opt | motivation/kv-opt-20260927 | 89624310f |
| LMCache baseline | /root/exp-lmcache-baseline | motivation/baseline-20260927 | 1559f56 |
| LMCache independent | /root/exp-lmcache-decoupled | motivation/decoupled-20260927 | ebda940 |
| LMCache optimized | /root/exp-lmcache-kv-opt | motivation/kv-opt-20260927 | ebda940 |

Original dirty worktrees and indexes remain unchanged. Independent snapshots
also retain existing inactive diagnostic/tail code. Optimized vLLM differs
from independent only in qwen3_next.py and the experiment harness. LMCache
independent and optimized commits intentionally match. Compiled extensions
are shared via symlinks to the existing builds, not rebuilt for each arm.

Baseline uses upstream aligned recovery plus reset-success reporting and a
common CUDA fallback stream-order correctness fix. It is NOT byte-for-byte
pristine upstream. Experimental arms additionally use independent Full16
objects, coalesced transport and last-state-only restore. The shallow arm
caps Full restore to SSM; deep_qkv uses deep KV without projection skipping;
deep_qonly adds Q-only and no-cat. No extra tail checkpoint is enabled.

Every arm records clean source commits, command, trace hash and explicit
PYTHONPATH. PYTHONSAFEPATH=1 prevents cwd from shadowing these paths; imports
are asserted before starting each arm and recorded in imports.log. Source
cleanliness and commits are checked again after each run.

v1 failed before model launch due to missing native_storage_ops symlink.
v2 was stopped when cwd import-shadowing risk was identified. Neither attempt
is included in this comparison. v3 uses verified import isolation throughout.

Timing is client HTTP request to first nonempty SSE text. Warmup, startup and
the explicit GPU resets are outside each request's timing. This run has no
profiler; it does not by itself provide GPU DMA or per-phase TTFT accounting.

## Completed results

All 160 requests completed; each arm passed all 39 resets and source checks.
All 40 first-token texts match across four arms. Baseline/shallow hit lengths
match per request, as do the two deep arms. First-token agreement is a limited
correctness check, not proof of full generated sequences or state equality.

| Arm | Mean continuation TTFT ms | Change vs baseline |
|---|---:|---:|
| baseline | 167.0922 | — |
| shallow | 171.4272 | +2.59% |
| deep_qkv | 174.6614 | +4.53% |
| deep_qonly | 172.4136 | +3.18% |

deep_qonly saves 2.2478 ms vs deep_qkv, but costs 0.9864 ms vs shallow and
5.3214 ms vs baseline. This pass does not demonstrate net acceleration over
native aligned recovery. Single sequential passes cannot establish statistical
significance. No outliers removed. Average extra KV hit: 267.556 tokens,
8.361 MiB; this does not eliminate the same number of full forward tokens.

See summary.json, matched_comparison.json, and all per-request online.jsonl.
The tested source commits above remain the authoritative run versions even
if a later documentation-only commit records these results on the opt branch.
