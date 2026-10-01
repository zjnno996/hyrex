"""Run directly with Python; no LMCache/GPU imports required."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/checkpoint_index.py"
spec = importlib.util.spec_from_file_location("checkpoint_index", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
index = module.CheckpointIndex()
tokens = list(range(2500))
index.record(tokens, 544, "a")
assert index.candidates(tokens, 544, "a") == [544]
index.record(tokens, 1040, "a")
assert len(index.slots) == 1
assert index.candidates(tokens, 1088, "a") == [1040]
assert index.candidates(tokens, 528, "a") == []
index.record(tokens, 1184, "a")
assert index.candidates(tokens, 1200, "a") == [1184, 1040]
assert index.candidates(tokens, 2500, "b") == []
branch = tokens.copy()
branch[1100] = -1
assert index.candidates(branch, 1200, "a") == [1040]
assert len(module.checkpoint_salt(tokens, 1184, "a")) <= 128
print("checkpoint replacement, deeper-KV lookup, and branch isolation passed")
