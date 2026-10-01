# HyRex validation code index

This file distinguishes the formal HyRex system from the smaller controlled
prototype used to establish the paper motivation. Both are retained because
they answer different questions.

## 1. Formal HyRex implementation and validation

The formal implementation is in `src/hyrex-vllm` and `src/hyrex-lmcache`.

Primary campaign and workload drivers:

- `src/hyrex-vllm/benchmarks/reproductions/run_hyrex_formal_campaign.py`
- `src/hyrex-vllm/benchmarks/reproductions/run_online_session_matrix.py`
- `src/hyrex-vllm/benchmarks/reproductions/run_online_session_cell.py`
- `src/hyrex-vllm/benchmarks/reproductions/run_native_independent_cells.py`
- `src/hyrex-vllm/benchmarks/reproductions/run_partial_prefix_cells.py`
- `src/hyrex-vllm/benchmarks/reproductions/sharegpt_hybrid_trace_e2e.py`
- `src/hyrex-vllm/benchmarks/reproductions/build_online_main_table.py`
- `src/hyrex-vllm/benchmarks/reproductions/HYREX_EXPERIMENT_DESIGN.md`

Core implementation:

- `src/hyrex-vllm/vllm/v1/kv_offload/hyrex_scheduler.py`
- `src/hyrex-vllm/vllm/v1/kv_offload/hyrex_vllm.py`
- `src/hyrex-vllm/vllm/v1/kv_offload/cpu/policies/hyrex.py`
- `src/hyrex-vllm/vllm/v1/kv_offload/cpu/manager.py`
- `src/hyrex-vllm/vllm/v1/core/sched/scheduler.py`
- `src/hyrex-lmcache/lmcache/integration/vllm/lmcache_mp_connector.py`
- `src/hyrex-lmcache/lmcache/integration/vllm/tail_probe_connector.py`
- `src/hyrex-lmcache/lmcache/v1/multiprocess/modules/lookup.py`
- `src/hyrex-lmcache/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`

Focused tests:

- `src/hyrex-vllm/tests/v1/kv_offload/test_hyrex_scheduler.py`
- `src/hyrex-vllm/tests/v1/kv_offload/test_hyrex_vllm.py`
- `src/hyrex-vllm/tests/v1/kv_offload/cpu/test_manager.py`
- `src/hyrex-vllm/tests/v1/kv_connector/unit/test_native_offloading_policy.py`
- `src/hyrex-lmcache/tests/v1/multiprocess/test_hyrex_bulk_full_pages.py`
- `src/hyrex-lmcache/tests/v1/test_tail_probe.py`
- `src/hyrex-lmcache/tests/v1/test_torch_ops_stream_order.py`

Fast source-level check:

```bash
cd src/hyrex-vllm
python benchmarks/reproductions/run_hyrex_formal_campaign.py --self-check
python -m pytest -q \
  tests/v1/kv_offload/test_hyrex_scheduler.py \
  tests/v1/kv_offload/test_hyrex_vllm.py
```

## 2. Controlled Motivation prototype

The Motivation prototype is intentionally separate from the formal scheduler.
It isolates the recovery mechanism and compares aligned recovery, deeper
Full-KV matching, and exact tail-state recovery.

- `results/run_controlled_gap_sweep_20261001.py`
- `results/analyze_controlled_gap_sweep_20261001.py`
- `src/vllm-hyrex/benchmarks/motivation/audit_real_sharegpt_mp.py`
- `src/lmcache-hyrex/tests/v1/test_tail_probe.py`
- `results/profile_single_forward_20260927.py`
- `results/analyze_single_forward_profile_20260927.py`

The frozen output is under `results/controlled_gap_sweep_20261001_v2/`.

## 3. Real traces, concurrency, and CPU-capacity scripts

- `results/run_16s_c1_native_tail_20260930.py`: 16-session, concurrency-1
  comparison.
- `results/run_16s_concurrency_sweep_20260930.py`: concurrency 1/4/8/16 sweep.
- `results/run_motivation_capacity_20260930.py`: constrained CPU-cache sweep.
- `results/analyze_motivation_story_20260930.py`: aggregation for the
  Motivation story.
- `results/motivation_sharegpt_10turn_trace.jsonl`: ten-turn conversation
  trace.
- `results/motivation_bfcl_agent_1s10r_trace.jsonl`: compact agent trace.

## 4. Correctness rule

A performance cell is publishable only if it uses the same model and prompts,
resets GPU prefix state at the declared boundary, retains only the intended CPU
cache, records hit/replay lengths, and matches the reference first token. For
the final paper, repeat the controlled measurements with at least 6 GiB
`/dev/shm`; the archived run used the same pickle fallback for all arms because
the source container exposed only 64 MiB shared memory.
