# HyRex

HyRex is a cost-aware heterogeneous prefix-recovery prototype for hybrid LLM
serving. This repository is a reproducibility snapshot containing the active
vLLM/LMCache implementation, clean native baselines, paper artifacts,
motivation experiments, curated traces, and related baseline source code.

Start with:

- [`README_MIGRATION.md`](README_MIGRATION.md): directory map and restoration
  instructions.
- [`src/hyrex-vllm`](src/hyrex-vllm): formal HyRex vLLM implementation,
  scheduler, policies, experiment campaign, and tests.
- [`src/hyrex-lmcache`](src/hyrex-lmcache): formal HyRex LMCache implementation
  and tests.
- [`src/vllm-hyrex`](src/vllm-hyrex) and
  [`src/lmcache-hyrex`](src/lmcache-hyrex): controlled motivation prototype used
  for the Deep-KV and exact tail-state experiments. These are not the formal
  HyRex implementation above.
- [`VALIDATION_CODE.md`](VALIDATION_CODE.md): index of validation and experiment
  entry points.
- [`HYREX_PAPER_AND_EXPERIMENT_PLAN.md`](HYREX_PAPER_AND_EXPERIMENT_PLAN.md):
  paper claim, current evidence, reproducibility protocol, and remaining work.
- [`results/controlled_gap_sweep_20261001_v2/RESULTS.md`](results/controlled_gap_sweep_20261001_v2/RESULTS.md): latest controlled motivation results.
- [`paper/manuscript.pdf`](paper/manuscript.pdf): current WWW manuscript draft.

Model weights, virtual environments, and machine-specific CUDA binaries are
not tracked. Their versions and restore requirements are recorded under
[`environment/`](environment/).
