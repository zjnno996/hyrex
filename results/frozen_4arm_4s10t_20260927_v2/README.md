# Frozen source comparison

All variants use Qwen3.5-9B BF16 eager, GPU1, CPU cache 2 GiB, the same
4 ShareGPT sessions x 10 turns, one request at a time, unrelated warmup,
and GPU reset between requests while retaining CPU cache. First-token
text and top-5 logprobs are retained. This is one sequential pass per arm,
not a randomized or statistically conclusive speedup measurement.

| Source | Worktree | Branch | Commit |
|---|---|---|---|
| vLLM baseline | /root/exp-vllm-baseline | motivation/baseline-20260927 | e582e17e6 |
| vLLM independent recovery | /root/exp-vllm-decoupled | motivation/decoupled-20260927 | 3ec66cb48 |
| vLLM Q-only + no-cat | /root/exp-vllm-kv-opt | motivation/kv-opt-20260927 | 6730b5fec |
| LMCache baseline | /root/exp-lmcache-baseline | motivation/baseline-20260927 | 1559f56 |
| LMCache independent | /root/exp-lmcache-decoupled | motivation/decoupled-20260927 | ebda940 |
| LMCache optimized arm | /root/exp-lmcache-kv-opt | motivation/kv-opt-20260927 | ebda940 |

The original dirty worktrees and their indexes were not changed. Snapshots
preserve their existing changes, including inactive tail-probe code.
The independent vLLM version restores the upstream qwen3_next projection
file; optimized projection changes are isolated to that file (plus harness).
LMCache independent and optimized versions intentionally have identical code.
Compiled extensions are linked to the same existing builds for all variants.

Baseline recovery code is upstream except reset-success reporting in vLLM
and the common CUDA fallback stream-order correctness fix in LMCache. It is
not a byte-for-byte pristine baseline. No independent Full16 indexing or
last-state-only optimization is enabled in the baseline arm.

Arms: baseline; shallow (independent transport but cap Full hit to state);
deep_qkv (deeper Full hit, fused QKV); deep_qonly (deeper Full hit, optimized
projection without cat). Per-arm design.json stores exact commits, command,
settings, and trace hash. Code trees are checked clean before and after runs.

The v1 attempt failed before model startup because the newly created
worktrees lacked native_storage_ops linkage. Existing extension binaries
were then linked; v2 is the first full runtime attempt. No v1 timings are used.
