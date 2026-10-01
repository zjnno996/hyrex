"""Matched-transfer 4-session/10-turn replay ablation after the stream fix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["shallow", "deep_qkv", "deep_qonly"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("LMCACHE_TAIL_") or key == "VLLM_HYREX_TAIL_PROBE":
            env.pop(key)
    settings = {
        "LMCACHE_HYREX_COALESCE_FULL_PAGES": "1",
        "LMCACHE_HYREX_LAST_STATE_ONLY": "1",
        "LMCACHE_HYREX_FULL_LOAD_TO_STATE": "1" if args.arm == "shallow" else "0",
        "VLLM_HYREX_Q_ONLY_REPLAY": "1" if args.arm == "deep_qonly" else "0",
        "VLLM_HYREX_MASK_FULL_KV": "1",
        "LMCACHE_HYREX_VERIFY_FULL_H2D": "0",
    }
    env.update(settings)
    command = [sys.executable, str(Path(__file__).with_name("audit_real_sharegpt_mp.py")),
        "--trace", "/root/hyrex_results/motivation_sharegpt_4s10t_trace.jsonl",
        "--mode", "default", "--workflow", "online", "--sessions", "4",
        "--min-session-turns", "10", "--limit", "0", "--round-robin-sessions",
        "--reset-between-requests", "--gpu", "1", "--cpu-gb", "2",
        "--experimental-lmcache-source", "/root/lmcache-hyrex",
        "--experimental-full-page-size", "16", "--first-token-logprobs",
        "--max-output-tokens", "1", "--vllm-port", "8761", "--lmcache-port", "8762",
        "--lmcache-http-port", "8763", "--output-dir", str(args.output_dir)]
    (args.output_dir / "design.json").write_text(json.dumps({
        "arm": args.arm, "settings": settings, "command": command,
        "scope": "matched experimental transfer stack, not pristine native baseline",
        "stream_order_fix": True, "tail_checkpoint": False,
    }, indent=2))
    with (args.output_dir / "runner.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = [json.loads(x) for x in (args.output_dir / "online.jsonl").read_text().splitlines()]
    resets = [json.loads(x) for x in (args.output_dir / "resets.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and sum(x["turn_index"] > 0 for x in rows) == 36
    assert len(resets) == 39 and all(x["success"] for x in resets)


if __name__ == "__main__":
    main()
