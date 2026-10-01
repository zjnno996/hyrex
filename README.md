# HyRex

HyRex is a cost-aware heterogeneous prefix-recovery prototype for hybrid LLM
serving. This repository is a reproducibility snapshot containing the active
vLLM/LMCache implementation, clean native baselines, paper artifacts,
motivation experiments, curated traces, and related baseline source code.

Start with:

- [`README_MIGRATION.md`](README_MIGRATION.md): directory map and restoration
  instructions.
- [`src/vllm-hyrex`](src/vllm-hyrex): active vLLM changes.
- [`src/lmcache-hyrex`](src/lmcache-hyrex): active LMCache changes.
- [`results/controlled_gap_sweep_20261001_v2/RESULTS.md`](results/controlled_gap_sweep_20261001_v2/RESULTS.md): latest controlled motivation results.
- [`paper/manuscript.pdf`](paper/manuscript.pdf): current WWW manuscript draft.

Model weights, virtual environments, and machine-specific CUDA binaries are
not tracked. Their versions and restore requirements are recorded under
[`environment/`](environment/).
