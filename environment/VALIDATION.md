# Bundle validation

Validation performed before archive creation:

- The three main modified implementation files were byte-compared with their
  original working-tree files:
  - `src/vllm-hyrex/vllm/v1/worker/gpu_model_runner.py`
  - `src/lmcache-hyrex/lmcache/integration/vllm/lmcache_mp_connector.py`
  - `src/lmcache-hyrex/lmcache/integration/vllm/replace_tail_connector.py`
- The controlled-run driver, analyzer, and request harness passed
  `python -m py_compile`.
- Six source-only tail/recovery tests passed; the two model-state import tests
  were deselected because the bundle intentionally excludes approximately
  1.3 GiB of machine-specific precompiled vLLM CUDA/FlashAttention `.so`
  artifacts. They require installing/building vLLM in the destination
  container, as expected for a source migration.
- All bundled files are covered by the root `SHA256SUMS` manifest.

The already completed 9B run is preserved under
`results/controlled_gap_sweep_20261001_v2`. Its diagnostics report zero formal
JIT warnings and no first-token mismatch across the measured continuation
requests.
