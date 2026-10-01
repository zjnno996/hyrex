# HyRex container migration bundle

This directory is a self-contained snapshot of the code, paper, traces, and
results needed to continue the HyRex project. Source trees are copied from the
working directories, so current uncommitted changes are included. Git object
databases, virtual environments, model weights, build products, and caches are
excluded.

## CORE: use these first

| Path | Purpose |
|---|---|
| `src/hyrex-vllm` | Formal HyRex vLLM implementation: heterogeneous recovery scheduler, CPU/GPU cache policy, campaign drivers, and tests |
| `src/hyrex-lmcache` | Formal HyRex LMCache implementation: independent hybrid-state lookup/transfer path and tests |
| `src/vllm-hyrex` | Controlled Motivation vLLM prototype for Deep-KV and exact tail-state recovery |
| `src/lmcache-hyrex` | Controlled Motivation LMCache prototype paired with `src/vllm-hyrex` |
| `src/vllm-native` | Clean vLLM baseline used by the controlled experiment |
| `src/lmcache-native` | Clean LMCache baseline used by the controlled experiment |
| `paper/` | WWW manuscript, motivation figure source, and compiled PDF |
| `results/controlled_gap_sweep_20261001_v2/` | Final five-arm controlled motivation experiment |
| `results/HYREX_TTFT_REALIZATION_GAP.md` | Main diagnosis and paper story |
| `datasets/hyrex_traces/` | Curated conversation/agent traces |
| `datasets/BFCL/` | Small BFCL multi-turn workload snapshot |
| `model-metadata/Qwen3.5-9B/` | Model config and tokenizer, without weights |
| `baselines/` | Source artifacts for the papers/baselines to reproduce |

The controlled experiment entry points are:

- `results/run_controlled_gap_sweep_20261001.py`
- `results/analyze_controlled_gap_sweep_20261001.py`
- `src/vllm-hyrex/benchmarks/motivation/audit_real_sharegpt_mp.py`

The formal HyRex campaign entry point is:

- `src/hyrex-vllm/benchmarks/reproductions/run_hyrex_formal_campaign.py`

See `VALIDATION_CODE.md` for the complete validation index and
`HYREX_PAPER_AND_EXPERIMENT_PLAN.md` for the research/evaluation plan.

## REFERENCE: useful but not the active implementation

- `src/vllm-marconi/` preserves the Marconi/offloading baseline branch and its
  current HyRex design notes.
- `baselines/marconi/` is the upstream Marconi artifact.
- `baselines/sparse-prefix/` is the Sparse Prefix artifact.
- `baselines/cacheflow/` is the CacheFlow baseline worktree used locally.
- `baselines/kvpr/` is the KVPR artifact snapshot.
- The other directories below `results/` preserve profiling, ABBA,
  concurrency, ShareGPT, and implementation-debugging evidence. The final
  controlled result remains `controlled_gap_sweep_20261001_v2`.

## Runtime expected by the measured results

- Python 3.13.2
- PyTorch 2.11.0+cu128
- vLLM 0.23.1.dev1+gfab9cce2e.d20260816
- NVIDIA driver 570.211.01
- Four RTX 4090 GPUs were visible; the controlled run used one GPU.
- Qwen3.5-9B BF16 eager

Exact package versions are in `environment/pip-freeze.txt`; source baselines
and dirty-file inventories are in `environment/SOURCE_STATE.md`.

## Restore checklist

1. Place Qwen3.5-9B weights under `/root/models/Qwen3.5-9B`, or update the
   model path in the experiment driver.
2. Run `./restore_layout.sh` if the new container should retain the absolute
   paths used by the current experiment scripts.
3. Build a fresh environment rather than copying the old `.venv`.
4. Put `src/hyrex-lmcache` and `src/hyrex-vllm` first on `PYTHONPATH` for formal
   HyRex runs. Use `src/lmcache-hyrex` and `src/vllm-hyrex` only for the
   controlled Motivation experiment, and use the two `*-native` trees for its
   clean baseline.
5. Start the container with at least 6 GiB shared memory, for example
   `--shm-size=6g`. The old container had only 64 MiB and therefore used the
   slower pickle IPC fallback.
6. Run the controlled experiment, then its analyzer. Result files should be
   compared with `results/controlled_gap_sweep_20261001_v2/RESULTS.md`.
7. Verify copied files with `sha256sum -c SHA256SUMS` from this directory.

## Intentionally excluded

See `environment/EXCLUDED.md`. The largest exclusions are model weight files,
the old virtual environment, raw WildChat/agent datasets, Git object stores,
and generated build/cache directories.
