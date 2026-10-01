#!/usr/bin/env python3
"""Controlled hybrid-recovery gap sweep with one service start per arm."""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get(
    "HYREX_CONTROLLED_ROOT",
    "/root/hyrex_results/controlled_gap_sweep_20261001_v2",
))
RUNNER = Path(os.environ.get(
    "HYREX_AUDIT_RUNNER",
    "/root/exp-vllm-single-forward/benchmarks/motivation/audit_real_sharegpt_mp.py",
))
PYTHON = Path(os.environ.get("HYREX_PYTHON", sys.executable))
GAPS = (0, 64, 128, 256, 384, 512)
COARSE = 1056
APPEND = 128
REPETITIONS = 5
GPU = os.environ.get("HYREX_GPU", "2")


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def status(state, **fields):
    (ROOT / "status.json").write_text(json.dumps(
        {"state": state, "updated_unix": time.time(), **fields}, indent=2
    ) + "\n")


def make_trace():
    rows = []
    arrival = 0
    for index, gap in enumerate(GAPS):
        seed_len = COARSE + gap
        seed = [10000 + index] + [20000 + index] * (seed_len - 1)
        resume = seed + [30000 + index] * APPEND
        sid = f"gap-{gap}"
        for turn, prompt in enumerate((seed, resume)):
            rows.append({
                "session_id": sid,
                "turn_index": turn,
                "arrival_index": arrival,
                "cache_prompt": prompt,
                "resume_prompt": prompt,
                "controlled_gap": gap,
            })
            arrival += 1
    trace = ROOT / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return trace


def sources(arm):
    if arm == "native":
        return Path("/root/exp-vllm-pristine-3way"), Path("/root/exp-lmcache-pristine-3way")
    return Path("/root/exp-vllm-single-forward"), Path("/root/exp-lmcache-single-forward")


def run_arm(trace, arm):
    out = ROOT / arm
    out.mkdir()
    vllm, lmcache = sources(arm)
    command = [
        str(PYTHON), str(RUNNER), "--trace", str(trace),
        "--mode", "default", "--workflow", "online",
        "--sessions", str(len(GAPS)), "--min-session-turns", "2", "--limit", "0",
        "--round-robin-sessions", "--reset-between-requests",
        "--gpu", GPU, "--cpu-gb", "4", "--max-output-tokens", "1",
        "--first-token-logprobs", "--online-repetitions", str(REPETITIONS),
        "--output-dir", str(out), "--vllm-port", "9061",
        "--lmcache-port", "9062", "--lmcache-http-port", "9063",
        "--vllm-source", str(vllm), "--experimental-lmcache-source", str(lmcache),
        "--prefill-budget", "528" if arm == "native" else "2112",
    ]
    env = {key: value for key, value in os.environ.items()
           if "HYREX" not in key and not key.startswith("LMCACHE_")}
    env.update(PYTHONSAFEPATH="1", PYTHONPATH=f"{lmcache}:{vllm}", CUDA_VISIBLE_DEVICES=GPU)
    if arm != "native":
        env.update(
            VLLM_HYREX_SINGLE_FORWARD="1",
            LMCACHE_HYREX_LAST_STATE_ONLY="1",
            LMCACHE_HYREX_FULL_LOAD_TO_STATE="1" if arm == "aligned" else "0",
            LMCACHE_HYREX_BATCH_FULL_PAGES="1",
            LMCACHE_HYREX_COALESCE_FULL_PAGES="1",
            VLLM_HYREX_Q_ONLY_REPLAY="0" if arm == "aligned" else "1",
        )
        command += ["--experimental-full-page-size", "16"]
    if arm.startswith("exact"):
        command += ["--replace-tail-checkpoint"]
    if arm == "exact_recovery_only":
        env["LMCACHE_HYREX_RECOVERY_ONLY"] = "1"
    (out / "command.json").write_text(json.dumps({
        "command": command,
        "environment": {key: value for key, value in env.items()
                        if "HYREX" in key or key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")},
    }, indent=2) + "\n")
    status("running", arm=arm)
    print(f"START {arm}", flush=True)
    with (out / "runner.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = load(out / "online.jsonl")
    resets = load(out / "resets.jsonl")
    expected = len(GAPS) * 2 * REPETITIONS
    assert len(rows) == expected
    assert len(resets) == expected - 1 and all(row["success"] for row in resets)
    print(f"DONE {arm}", flush=True)
    return rows


def summarize(all_rows):
    summary = {
        "design": {
            "model": "Qwen3.5-9B BF16 eager",
            "gaps": GAPS,
            "coarse_boundary": COARSE,
            "appended_tokens": APPEND,
            "repetitions": REPETITIONS,
            "prefill_budget": {"native": 528, "other_arms": 2112},
            "matched_comparison": "aligned/deep/exact arms share the same single-forward implementation and 2112-token budget",
            "gpu_reset_between_requests": True,
            "cpu_cache_retained_within_repetition": True,
            "shm_warning": "64 MiB /dev/shm; all arms use the same pickle fallback",
        },
        "by_gap": [],
        "correctness": {},
    }
    indexed = {
        arm: {(row["repetition"], row["session_id"], row["turn_index"]): row
              for row in rows}
        for arm, rows in all_rows.items()
    }
    native = indexed["native"]
    for arm, rows in indexed.items():
        mismatches = 0
        compared = 0
        for key, row in rows.items():
            if key not in native:
                continue
            compared += 1
            mismatches += row["first_text"] != native[key]["first_text"]
        summary["correctness"][arm] = {
            "compared_first_tokens": compared,
            "first_text_mismatches": mismatches,
        }
    for gap in GAPS:
        sid = f"gap-{gap}"
        row = {"gap_tokens": gap}
        for arm, rows in indexed.items():
            samples = [rows[(rep, sid, 1)] for rep in range(REPETITIONS)]
            ttfts = [sample["ttft_ms"] for sample in samples]
            hits = [sample["cached_tokens"] for sample in samples]
            row[arm] = {
                "mean_ttft_ms": statistics.mean(ttfts),
                "median_ttft_ms": statistics.median(ttfts),
                "min_ttft_ms": min(ttfts),
                "max_ttft_ms": max(ttfts),
                "mean_cached_tokens": statistics.mean(hits),
                "mean_forward_tokens": statistics.mean(
                    sample["prompt_tokens"] - sample["cached_tokens"] for sample in samples),
                "samples": ttfts,
            }
        summary["by_gap"].append(row)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    trace = make_trace()
    all_rows = {}
    try:
        for arm in ("native", "aligned", "deep", "exact_recovery_only", "exact_steady"):
            all_rows[arm] = run_arm(trace, arm)
        summary = summarize(all_rows)
        status("completed")
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
