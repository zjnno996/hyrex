#!/usr/bin/env python3
"""Run native and tail once each for concurrency 4/8/16."""

import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path("/root/hyrex_results/native_tail_16s_concurrency_20260930_v1")
RUNNER = Path("/root/exp-vllm-single-forward/benchmarks/motivation/audit_real_sharegpt_mp.py")
TRACE = Path("/root/hyrex_results/motivation_sharegpt_16s_varied_trace.jsonl")
PYTHON = Path("/root/hybrid-model-offloading/.venv/bin/python")
GPU = "2"
CONCURRENCIES = (4, 8, 16)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values, q):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * q) - 1]


def status(state, **extra):
    (ROOT / "status.json").write_text(json.dumps({
        "state": state, "updated_unix": time.time(), **extra,
    }, indent=2) + "\n")


def command(output, arm):
    baseline = arm == "baseline"
    result = [
        str(PYTHON), str(RUNNER), "--trace", str(TRACE),
        "--mode", "default", "--workflow", "online",
        "--sessions", "16", "--min-session-turns", "5", "--limit", "0",
        "--round-robin-sessions", "--reset-between-requests",
        "--concurrency-sweep", "4,8,16", "--gpu", GPU, "--cpu-gb", "8",
        "--max-output-tokens", "1", "--first-token-logprobs",
        "--online-repetitions", "1", "--output-dir", str(output),
        "--vllm-port", "8961", "--lmcache-port", "8962",
        "--lmcache-http-port", "8963",
        "--vllm-source", "/root/exp-vllm-pristine-3way" if baseline else "/root/exp-vllm-short-tail",
        "--experimental-lmcache-source",
        "/root/exp-lmcache-pristine-3way" if baseline else "/root/exp-lmcache-single-forward",
        "--prefill-budget", "528" if baseline else "2048",
    ]
    if not baseline:
        result += ["--experimental-full-page-size", "16", "--replace-tail-checkpoint"]
    return result


def summarize(rows):
    result = {}
    for concurrency in CONCURRENCIES:
        selected = [row for row in rows
                    if row["concurrency"] == concurrency and row["turn_index"] > 0]
        ttft = [row["ttft_ms"] for row in selected]
        replay = [row["prompt_tokens"] - row["cached_tokens"] for row in selected]
        waves = {}
        for row in selected:
            waves.setdefault(row["wave_index"], []).append(row)
        service_ms = sum(max(row["elapsed_ms"] for row in wave) for wave in waves.values())
        result[str(concurrency)] = {
            "continuations": len(selected),
            "mean_ttft_ms": statistics.mean(ttft),
            "median_ttft_ms": statistics.median(ttft),
            "p95_ttft_ms": percentile(ttft, 0.95),
            "mean_replay_tokens": statistics.mean(replay),
            "recovery_requests_per_second_excluding_resets": len(selected) * 1000 / service_ms,
        }
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = (json.loads((ROOT / "summary.json").read_text())
               if (ROOT / "summary.json").exists() else {})
    raw = {}
    for arm in ("baseline", "tail"):
        output = ROOT / arm
        if (output / "online.jsonl").exists():
            measured = read_rows(output / "online.jsonl")
            if len(measured) == 120 * len(CONCURRENCIES):
                raw[arm] = measured
                summary[arm] = summarize(measured)
                diagnostics = read_rows(output / "sweep_diagnostics.jsonl")
                summary[arm]["diagnostics"] = diagnostics
                print(f"REUSE {arm}: {json.dumps(summary[arm])}", flush=True)
                continue
        output.mkdir(exist_ok=False)
        cmd = command(output, arm)
        env = {key: value for key, value in os.environ.items()
               if "HYREX" not in key and not key.startswith("LMCACHE_")}
        env.update(PYTHONSAFEPATH="1", CUDA_VISIBLE_DEVICES=GPU)
        if arm == "tail":
            env.update(
                VLLM_HYREX_SINGLE_FORWARD="1",
                VLLM_HYREX_MULTI_REQUEST="1",
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
        status("running", arm=arm)
        print(f"START {arm}", flush=True)
        with (output / "runner.log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        measured = read_rows(output / "online.jsonl")
        assert len(measured) == 120 * len(CONCURRENCIES)
        diagnostics = read_rows(output / "sweep_diagnostics.jsonl")
        assert len(diagnostics) == len(CONCURRENCIES)
        raw[arm] = measured
        summary[arm] = summarize(measured)
        summary[arm]["diagnostics"] = diagnostics
        (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"DONE {arm}: {json.dumps(summary[arm])}", flush=True)

    correctness = {}
    for concurrency in CONCURRENCIES:
        baseline = {(row["session_id"], row["turn_index"]): row for row in raw["baseline"]
                    if row["concurrency"] == concurrency}
        tail = {(row["session_id"], row["turn_index"]): row for row in raw["tail"]
                if row["concurrency"] == concurrency}
        assert baseline.keys() == tail.keys()
        assert all(baseline[key]["prompt_tokens"] == tail[key]["prompt_tokens"]
                   for key in baseline)
        correctness[str(concurrency)] = {
            "first_token_mismatches": sum(
                baseline[key]["first_text"] != tail[key]["first_text"] for key in baseline),
            "requests": len(baseline),
        }
    summary["correctness"] = correctness
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    status("completed_with_warning" if any(
        value["first_token_mismatches"] for value in correctness.values()) else "completed",
        requests=120 * len(CONCURRENCIES) * 2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            status("failed", error=repr(error))
        raise
