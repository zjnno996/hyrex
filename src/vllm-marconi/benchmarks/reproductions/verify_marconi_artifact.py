# SPDX-License-Identifier: Apache-2.0
"""Compare the public Marconi radix artifact with the vLLM adapter.

This is deliberately a CPU-only semantic check.  It verifies the observable
prefix reuse and node creation behavior without importing a model or claiming
an end-to-end serving result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path


TRACE = (
    (1, 2, 3, 4),
    (1, 2, 3, 5),
    (1, 2, 3, 6),
    (1, 9),
    (8, 2),
    (1, 2, 3, 6, 7),
)


def _load_artifact(artifact_dir: Path):
    """Load the cache class while stubbing its unused transformers import."""
    if not (artifact_dir / "radix_cache_hybrid.py").is_file():
        raise ValueError(f"not a Marconi artifact directory: {artifact_dir}")
    sys.path.insert(0, str(artifact_dir))
    sys.modules.setdefault(
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=object, AutoTokenizer=object),
    )
    return importlib.import_module("radix_cache_hybrid").RadixCache


def verify(artifact_dir: Path) -> dict[str, object]:
    RadixCache = _load_artifact(artifact_dir)
    from vllm.v1.kv_offload.policies.marconi_index import MarconiIndex

    reference = RadixCache(
        capacity_bytes=10**12,
        num_ssm_layers=1,
        num_attn_layers=1,
        num_mlp_layers=1,
        d=8,
        n=2,
        evict_policy_version=1,
    )
    adapter = MarconiIndex()
    matched_tokens: list[int] = []
    for now, tokens in enumerate(TRACE, start=1):
        reference_prefix, _, _, _ = reference.match_prefix(list(tokens))
        adapter_prefix = adapter.lookup(tuple(map(str, tokens)))
        if len(reference_prefix) != len(adapter_prefix):
            raise AssertionError(
                f"prefix mismatch for {tokens}: artifact={reference_prefix}, "
                f"adapter={adapter_prefix}"
            )
        matched_tokens.append(len(adapter_prefix))
        reference.insert(
            list(tokens), state_at_leaf=now, state_at_branchoff=now
        )
        adapter.insert(
            tuple(map(str, tokens)),
            state_kind="hybrid",
            token_count=len(tokens),
            byte_size=len(tokens),
            last_access_ms=float(now),
        )
        if reference.num_nodes != len(adapter.nodes()):
            raise AssertionError(
                f"node mismatch after {tokens}: artifact={reference.num_nodes}, "
                f"adapter={len(adapter.nodes())}"
            )
    return {"matched_tokens": matched_tokens, "nodes": len(adapter.nodes())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact_dir), sort_keys=True))


if __name__ == "__main__":
    main()
