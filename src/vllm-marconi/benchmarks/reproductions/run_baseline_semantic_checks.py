# SPDX-License-Identifier: Apache-2.0
"""Run the CPU-only semantic gates for all isolated baseline branches."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKTREES = {
    "marconi": ROOT,
    "kvpr": ROOT.parent / "hybrid-model-offloading-kvpr",
    "hcache": ROOT.parent / "hybrid-model-offloading-hcache",
    "cacheflow": ROOT.parent / "hybrid-model-offloading-cacheflow",
}
TESTS = {
    "marconi": ("tests/v1/kv_offload/test_marconi_index.py",),
    "kvpr": (
        "tests/v1/kv_offload/test_kvpr_policy.py",
        "tests/v1/kv_offload/test_kvpr.py",
    ),
    "hcache": (
        "tests/v1/kv_offload/test_hcache_policy.py",
        "tests/v1/kv_offload/test_hcache_storage.py",
        "tests/v1/kv_offload/test_hcache_executor.py",
        "tests/v1/kv_offload/test_hcache_vllm.py",
    ),
    "cacheflow": ("tests/v1/kv_offload/test_cacheflow_policy.py",),
}


def _run(worktree: Path, tests: tuple[str, ...]) -> None:
    if not worktree.is_dir():
        raise FileNotFoundError(f"missing baseline worktree: {worktree}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *tests,
        "--confcutdir=tests/v1/kv_offload",
    ]
    print("+", " ".join(command), flush=True)
    env = os.environ | {"PYTHONPATH": str(worktree)}
    subprocess.run(command, cwd=worktree, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--marconi-artifact",
        type=Path,
        help="optional upstream artifact directory for the extra parity check",
    )
    args = parser.parse_args()
    for name, worktree in WORKTREES.items():
        _run(worktree, TESTS[name])
    if args.marconi_artifact:
        verifier = ROOT / "benchmarks/reproductions/verify_marconi_artifact.py"
        command = [sys.executable, str(verifier), str(args.marconi_artifact)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
