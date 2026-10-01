"""Three-arm, isolated real seed/resume TTFT probe (not a 40-request run)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--arms", nargs="+", default=["baseline", "deep", "replacement"],
                        choices=["baseline", "deep", "replacement"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path("/root/hyrex_results/deep_kv_no_cat_probe_20260927_v1/trace.jsonl")
    trace = args.output_dir / "trace.jsonl"
    trace.write_bytes(source.read_bytes())
    base_vllm = Path("/root/exp-vllm-pristine-3way")
    base_lmc = Path("/root/exp-lmcache-pristine-3way")
    exp_vllm = Path("/root/exp-vllm-replacement-3way")
    exp_lmc = Path("/root/exp-lmcache-replacement-3way")
    roots = [base_vllm, base_lmc, exp_vllm, exp_lmc]
    snapshots = {}
    for root in roots:
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        if status:
            raise RuntimeError(f"Uncommitted experiment tree {root}: {status}")
        snapshots[str(root)] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    (args.output_dir / "design.json").write_text(json.dumps({
        "trees": snapshots, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "scope": "isolated real seed/resume pair; CPU cleared between repetitions; not continuous 4x10",
        "selection": "1054 -> 1191 tokens; near-max alignment gap; NOT representative average",
        "replacement": "replace S528 with exact S1040 in seed; capture S1184 in resume; old tail retained until pair clear",
        "output": "seed 1 token, resume 32 tokens; compare complete greedy continuation",
        "metric": "client HTTP to first nonempty SSE text; seed overhead included separately",
        "warmup": "8 unrelated length/repeat requests plus unrelated 1054->1191 shape pair, all 32 output tokens; GPU reset each; CPU/GPU cleared before measurement",
    }, indent=2))
    results = {}
    for arm in args.arms:
        out = args.output_dir / arm
        out.mkdir()
        vllm, lmc = (base_vllm, base_lmc) if arm == "baseline" else (exp_vllm, exp_lmc)
        env = {k: v for k, v in os.environ.items()
               if "HYREX" not in k and not k.startswith("LMCACHE_TAIL_")}
        env.update(PYTHONSAFEPATH="1", PYTHONPATH=f"{lmc}:{vllm}", CUDA_VISIBLE_DEVICES="1")
        if arm != "baseline":
            env.update(LMCACHE_HYREX_LAST_STATE_ONLY="1", LMCACHE_HYREX_FULL_LOAD_TO_STATE="0",
                       LMCACHE_HYREX_COALESCE_FULL_PAGES="1", LMCACHE_HYREX_BATCH_FULL_PAGES="1",
                       VLLM_HYREX_Q_ONLY_REPLAY="1", VLLM_HYREX_Q_ONLY_NO_CAT="1")
        check = ("import torch, inspect, vllm, lmcache, lmcache.c_ops as ops; "
                 "print(vllm.__file__, lmcache.__file__, ops.multi_layer_block_kv_transfer); "
                 f"assert vllm.__file__.startswith('{vllm}/'); "
                 f"assert lmcache.__file__.startswith('{lmc}/'); "
                 "assert inspect.isbuiltin(ops.multi_layer_block_kv_transfer), 'native CUDA extension not bound'")
        with (out / "imports.log").open("w") as log:
            subprocess.run([sys.executable, "-c", check], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, str(Path(__file__).with_name("audit_real_sharegpt_mp.py")),
                   "--trace", str(trace), "--mode", "default", "--workflow", "online", "--limit", "0",
                   "--reset-between-requests", "--gpu", "1", "--cpu-gb", "2",
                   "--vllm-source", str(vllm), "--experimental-lmcache-source", str(lmc),
                   "--vllm-port", "8761", "--lmcache-port", "8762", "--lmcache-http-port", "8763",
                   "--output-dir", str(out), "--first-token-logprobs", "--online-repetitions", str(args.repetitions)]
        if arm != "baseline":
            command += ["--experimental-full-page-size", "16"]
        if arm == "replacement":
            command += ["--replace-tail-checkpoint"]
        (out / "command.json").write_text(json.dumps(command, indent=2))
        print(f"Starting {arm}", flush=True)
        with (out / "runner.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (out / "online.jsonl").read_text().splitlines()]
        resets = [json.loads(line) for line in (out / "resets.jsonl").read_text().splitlines()]
        assert len(rows) == 2 * args.repetitions and len(resets) == len(rows) - 1
        expected = 528 if arm == "baseline" else 1040
        assert all(r["cached_tokens"] == expected for r in rows if r["turn_index"] == 1), rows
        results[arm] = rows
        summary = {}
        for name, measured in results.items():
            summary[name] = {
                "seed_mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in measured if r["turn_index"] == 0),
                "resume_mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in measured if r["turn_index"] == 1),
                "all_mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in measured),
            }
            if "baseline" in results:
                summary[name]["all_generated_text_equal_baseline"] = all(
                    a["generated_text"] == b["generated_text"] for a, b in zip(results["baseline"], measured))
                summary[name]["resume_ttft_reduction_pct"] = 100 * (
                    1 - summary[name]["resume_mean_ttft_ms"] / summary["baseline"]["resume_mean_ttft_ms"])
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
