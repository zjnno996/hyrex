#!/usr/bin/env python3
"""Matched-budget CPU-cache capacity sweep for native and tail recovery."""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/hyrex_results/motivation_capacity_16s5t_20260930_v1")
RUNNER = Path("/root/exp-vllm-single-forward/benchmarks/motivation/audit_real_sharegpt_mp.py")
TRACE = Path("/root/hyrex_results/motivation_sharegpt_16s_varied_trace.jsonl")
PYTHON = Path("/root/hybrid-model-offloading/.venv/bin/python")
CAPACITIES = (1, 2, 4, 8)
EXPECTED = 80


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def status(state, **fields):
    (ROOT / "status.json").write_text(json.dumps(
        {"state": state, "updated_unix": time.time(), **fields}, indent=2
    ) + "\n")


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    summary = {"design": {
        "model": "Qwen3.5-9B BF16 eager",
        "sessions": 16,
        "turns_per_session": 5,
        "requests_per_arm": EXPECTED,
        "prefill_budget_both_arms": 2112,
        "gpu_reset_between_requests": True,
        "cpu_capacity_gib": CAPACITIES,
    }, "results": {}}
    status("running")
    for capacity in CAPACITIES:
        summary["results"][str(capacity)] = {}
        for arm in ("native", "tail"):
            out = ROOT / f"cpu{capacity}_{arm}"
            out.mkdir()
            vllm = Path("/root/exp-vllm-pristine-3way" if arm == "native" else "/root/exp-vllm-single-forward")
            lmcache = Path("/root/exp-lmcache-pristine-3way" if arm == "native" else "/root/exp-lmcache-single-forward")
            command = [
                str(PYTHON), str(RUNNER), "--trace", str(TRACE),
                "--mode", "default", "--workflow", "online",
                "--sessions", "16", "--min-session-turns", "5", "--limit", str(EXPECTED),
                "--round-robin-sessions", "--reset-between-requests",
                "--gpu", "2", "--cpu-gb", str(capacity), "--max-output-tokens", "1",
                "--first-token-logprobs", "--online-repetitions", "1",
                "--output-dir", str(out), "--vllm-port", "8961",
                "--lmcache-port", "8962", "--lmcache-http-port", "8963",
                "--vllm-source", str(vllm), "--experimental-lmcache-source", str(lmcache),
                "--prefill-budget", "2112",
            ]
            env = {key: value for key, value in os.environ.items()
                   if "HYREX" not in key and not key.startswith("LMCACHE_")}
            env.update(PYTHONSAFEPATH="1", PYTHONPATH=f"{lmcache}:{vllm}", CUDA_VISIBLE_DEVICES="2")
            if arm == "tail":
                env.update(
                    VLLM_HYREX_SINGLE_FORWARD="1",
                    LMCACHE_HYREX_LAST_STATE_ONLY="1",
                    LMCACHE_HYREX_FULL_LOAD_TO_STATE="0",
                    LMCACHE_HYREX_BATCH_FULL_PAGES="1",
                    LMCACHE_HYREX_COALESCE_FULL_PAGES="1",
                    VLLM_HYREX_Q_ONLY_REPLAY="1",
                    VLLM_HYREX_SHORT_TAIL="1",
                )
                command += ["--experimental-full-page-size", "16", "--replace-tail-checkpoint"]
            (out / "command.json").write_text(json.dumps({
                "command": command,
                "environment": {key: value for key, value in env.items()
                                if "HYREX" in key or key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")},
            }, indent=2) + "\n")
            status("running", capacity_gib=capacity, arm=arm)
            print(f"START cpu={capacity} arm={arm}", flush=True)
            with (out / "runner.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = load(out / "online.jsonl")
            resets = load(out / "resets.jsonl")
            assert len(rows) == EXPECTED
            assert len(resets) == EXPECTED - 1 and all(row["success"] for row in resets)
            continued = [row for row in rows if row["turn_index"] > 0]
            summary["results"][str(capacity)][arm] = {
                "requests": len(rows),
                "continuations": len(continued),
                "mean_cached_tokens": statistics.mean(row["cached_tokens"] for row in continued),
                "mean_replay_tokens": statistics.mean(
                    row["prompt_tokens"] - row["cached_tokens"] for row in continued),
                "mean_ttft_ms": statistics.mean(row["ttft_ms"] for row in continued),
                "median_ttft_ms": statistics.median(row["ttft_ms"] for row in continued),
            }
            (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary["results"][str(capacity)][arm]), flush=True)

        native_rows = {(row["session_id"], row["turn_index"]): row
                       for row in load(ROOT / f"cpu{capacity}_native" / "online.jsonl")}
        tail_rows = {(row["session_id"], row["turn_index"]): row
                     for row in load(ROOT / f"cpu{capacity}_tail" / "online.jsonl")}
        keys = [key for key in native_rows if key[1] > 0]
        summary["results"][str(capacity)]["paired"] = {
            "first_text_mismatches": sum(native_rows[key]["first_text"] != tail_rows[key]["first_text"] for key in keys),
            "tail_minus_native_mean_ttft_ms": statistics.mean(
                tail_rows[key]["ttft_ms"] - native_rows[key]["ttft_ms"] for key in keys),
            "tail_faster": sum(tail_rows[key]["ttft_ms"] < native_rows[key]["ttft_ms"] for key in keys),
        }
        (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    status("completed")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            status("failed", error=repr(error))
        raise
