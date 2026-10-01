#!/usr/bin/env python3
"""Record the stock vLLM + LMCache hybrid-cache capability boundary."""

import ast
import importlib.metadata
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
connector_path = root / "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"
factory_path = root / "vllm/distributed/kv_transfer/kv_connector/factory.py"
tree = ast.parse(connector_path.read_text())
connector = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "LMCacheConnectorV1"
)
supports_hma = any(ast.unparse(base).endswith("SupportsHMA") for base in connector.bases)
match_method = next(
    node
    for node in connector.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "get_num_new_matched_tokens"
)
return_shape = ast.unparse(match_method.returns)
factory_source = factory_path.read_text()

result = {
    "vllm_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "lmcache_version": importlib.metadata.version("lmcache"),
    "connector": "LMCacheConnectorV1",
    "supports_hybrid_memory_allocator": supports_hma,
    "match_api_return": return_shape,
    "recovery_boundary_shape": "one scalar token count per request",
}

assert supports_hma is False
assert return_shape == "tuple[int | None, bool]"
assert "does not support HMA" in factory_source
print(json.dumps(result, indent=2))
