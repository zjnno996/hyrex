"""Run with Python; exercises actual scheduler source without loading a model."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
tree = ast.parse((root / "vllm/v1/core/sched/scheduler.py").read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_mamba_block_aligned_split")
for arg in method.args.args:
    arg.annotation = None
method.returns = None
scope = {"os": os}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "scheduler", "exec"), scope)
split = scope[method.name]
os.environ["VLLM_HYREX_REPLACE_TAIL"] = "1"
scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=528), use_eagle=False)
for length, restored, expected in (
    (1054, 0, [528, 1040, 1054]),
    (1191, 1040, [1184, 1191]),
    (1708, 0, [528, 1056, 1584, 1696, 1708]),
    (1056, 0, [528, 1056]),
):
    request = SimpleNamespace(num_prompt_tokens=length, num_tokens=length, num_computed_tokens=restored)
    endpoints = []
    while request.num_computed_tokens < length:
        count = split(scheduler, request, min(528, length-request.num_computed_tokens))
        assert count > 0
        request.num_computed_tokens += count
        endpoints.append(request.num_computed_tokens)
    assert endpoints == expected, (length, restored, endpoints)
print("replacement scheduler boundary checks passed")
