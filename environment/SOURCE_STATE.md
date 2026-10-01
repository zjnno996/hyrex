# Source state at bundle creation

The snapshots contain the working-tree files, including the listed local
modifications. `.git` metadata is intentionally excluded because four source
directories were linked Git worktrees whose `.git` files pointed outside the
tree and were not portable.

| Snapshot | Original branch | Base commit |
|---|---|---|
| `src/vllm-hyrex` | `motivation/single-forward-20260927` | `cd2977fc04e0ab36656f964dccc6986833b01706` |
| `src/lmcache-hyrex` | `motivation/single-forward-20260927` | `7a832182564f609bf52f3ab505dbcd58ac0cfa27` |
| `src/vllm-native` | `motivation/pristine-3way-20260927` | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| `src/lmcache-native` | `motivation/pristine-3way-20260927` | `140819c9d57a975dbc5678a6459a218e544cb58b` |
| `src/vllm-marconi` | `hyrec/baseline-marconi` | `76af32295db87d562cc453c9260bd72d0b628c89` |

## Active HyRex local changes

`src/vllm-hyrex`:

- `benchmarks/motivation/audit_real_sharegpt_mp.py`
- `benchmarks/motivation/run_frozen_comparison.py`
- `benchmarks/motivation/make_bfcl_agent_trace.py`
- `benchmarks/motivation/make_sharegpt_16s_varied_trace.py`
- `vllm/v1/worker/gpu_model_runner.py`

`src/lmcache-hyrex`:

- `lmcache/integration/vllm/lmcache_mp_connector.py`
- `lmcache/integration/vllm/replace_tail_connector.py`
- `tests/v1/test_tail_probe.py`

The native source snapshots were clean. The Marconi snapshot also contains
local scheduler/offloading changes and the design documents under
`benchmarks/reproductions/`; use the snapshot itself as the authoritative
working state.

## Paper/baseline artifact versions

| Snapshot | Branch | Commit |
|---|---|---|
| `baselines/marconi` | `main` | `08016617b1524e6bf6ac29b680641cc945bda7f0` |
| `baselines/sparse-prefix` | `main` | `d3050d44ed9d878c396cf3fcf632d63743ffed5b` |
| `baselines/cacheflow` | `hyrec/baseline-cacheflow` | `7c73ced6501ee48083ad83dc18b4b69d74881989` |
| `baselines/kvpr` | detached artifact | `1712a52042262fe1646799322e45792b109bd020` |

These snapshots were clean when copied. Their Git object databases are omitted;
the exact source files and commit identifiers are retained.
