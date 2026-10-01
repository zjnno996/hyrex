# Excluded large or regenerable content

The bundle intentionally omits:

- `/root/models/**/model*.safetensors` (the full model directory was about
  70 GiB; Qwen3.5-9B alone has four shards totaling about 18.5 GiB).
- `/root/hybrid-model-offloading/.venv` and package caches.
- `/root/dataset/WildChat` (about 1.5 GiB).
- `/root/dataset/lmcache-agentic-traces/raw` (about 2.3 GiB).
- `/root/paper-artifacts/envs/marconi` (about 6.4 GiB Conda environment); the
  Marconi source and environment specification are included instead.
- Git object databases and worktree pointer files.
- `__pycache__`, compiled objects/shared libraries, build directories, and
  `/root/hybrid-model-offloading/vllm/vllm_flash_attn` vendored/build content.
- LaTeX auxiliary files; the manuscript source and final PDFs are included.

These exclusions are not required to inspect the results or continue the
HyRex code. Model weights and a rebuilt Python/CUDA environment are required
to rerun inference.
