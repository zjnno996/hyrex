#!/usr/bin/env python3
"""One-pass native/tail screening run on 16 varied ShareGPT sessions."""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/hyrex_results/native_tail_16s_c1_20260930_v1")
RUNNER = Path("/root/exp-vllm-single-forward/benchmarks/motivation/audit_real_sharegpt_mp.py")
TRACE = Path("/root/hyrex_results/motivation_sharegpt_16s_varied_trace.jsonl")
PYTHON = Path("/root/hybrid-model-offloading/.venv/bin/python")
GPU = "2"
EXPECTED = 120


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_status(state, **extra):
    (ROOT / "status.json").write_text(json.dumps({
        "state": state, "updated_unix": time.time(), **extra,
    }, indent=2) + "\n")


def command(output, arm):
    baseline = arm == "baseline"
    result = [
        str(PYTHON), str(RUNNER),
        "--trace", str(TRACE), "--mode", "default", "--workflow", "online",
        "--sessions", "16", "--min-session-turns", "5", "--limit", "0",
        "--round-robin-sessions", "--reset-between-requests",
        "--gpu", GPU, "--cpu-gb", "8", "--max-output-tokens", "1",
        "--first-token-logprobs", "--online-repetitions", "1",
        "--output-dir", str(output), "--vllm-port", "8861",
        "--lmcache-port", "8862", "--lmcache-http-port", "8863",
        "--vllm-source", "/root/exp-vllm-pristine-3way" if baseline else "/root/exp-vllm-short-tail",
        "--experimental-lmcache-source",
        "/root/exp-lmcache-pristine-3way" if baseline else "/root/exp-lmcache-single-forward",
        "--prefill-budget", "528" if baseline else "2048",
    ]
    if not baseline:
        result += ["--experimental-full-page-size", "16", "--replace-tail-checkpoint"]
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    write_status("running", arm="baseline")
    summary = {}
    all_rows = {}
    for arm in ("baseline", "tail"):
        output = ROOT / arm
        output.mkdir()
        cmd = command(output, arm)
        env = {k: v for k, v in os.environ.items()
               if "HYREX" not in k and not k.startswith("LMCACHE_")}
        env.update(PYTHONSAFEPATH="1", CUDA_VISIBLE_DEVICES=GPU)
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
        source = cmd[cmd.index("--vllm-source") + 1]
        cache_source = cmd[cmd.index("--experimental-lmcache-source") + 1]
        env["PYTHONPATH"] = f"{cache_source}:{source}"
        (output / "command.json").write_text(json.dumps({
            "command": cmd,
            "environment": {key: value for key, value in env.items()
                            if "HYREX" in key or key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")},
        }, indent=2) + "\n")
        print(f"START {arm}", flush=True)
        write_status("running", arm=arm)
        with (output / "runner.log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        measured = rows(output / "online.jsonl")
        resets = rows(output / "resets.jsonl")
        assert len(measured) == EXPECTED
        assert len(resets) == EXPECTED - 1 and all(row["success"] for row in resets)
        continued = [row for row in measured if row["turn_index"] > 0]
        all_rows[arm] = measured
        summary[arm] = {
            "requests": len(measured),
            "continuations": len(continued),
            "mean_ttft_ms": statistics.mean(row["ttft_ms"] for row in continued),
            "median_ttft_ms": statistics.median(row["ttft_ms"] for row in continued),
            "mean_cached_tokens": statistics.mean(row["cached_tokens"] for row in continued),
            "mean_replay_tokens": statistics.mean(
                row["prompt_tokens"] - row["cached_tokens"] for row in continued),
        }
        (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"DONE {arm}: {json.dumps(summary[arm])}", flush=True)

    baseline = {(row["session_id"], row["turn_index"]): row for row in all_rows["baseline"]}
    tail = {(row["session_id"], row["turn_index"]): row for row in all_rows["tail"]}
    assert baseline.keys() == tail.keys()
    assert all(
        baseline[key]["prompt_tokens"] == tail[key]["prompt_tokens"]
        and baseline[key]["first_text"] == tail[key]["first_text"]
        for key in baseline
    ), "trace or first-token mismatch"
    summary["tail_ttft_reduction_percent"] = 100 * (
        1 - summary["tail"]["mean_ttft_ms"] / summary["baseline"]["mean_ttft_ms"]
    )
    summary["mean_replay_tokens_avoided"] = (
        summary["baseline"]["mean_replay_tokens"] - summary["tail"]["mean_replay_tokens"]
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_status("completed", requests=EXPECTED * 2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            write_status("failed", error=repr(error))
        raise
