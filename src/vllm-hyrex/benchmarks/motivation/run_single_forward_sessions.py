"""Frozen native/deep/tail comparison: 4 sessions x 10 turns x repetitions."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline = (Path("/root/exp-vllm-pristine-3way"), Path("/root/exp-lmcache-pristine-3way"))
    experiment = (Path("/root/exp-vllm-single-forward"), Path("/root/exp-lmcache-single-forward"))
    commits = {}
    for root in (*baseline, *experiment):
        assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]), root
        commits[str(root)] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    (args.output_dir / "design.json").write_text(json.dumps({
        "commits": commits, "sessions": 4, "turns": 10, "repetitions": args.repetitions,
        "baseline_budget": 528, "experimental_budget": 2048,
        "budget_difference": "Native LMCache requires [528,1056); only experimental in-layer snapshots lift this restriction. This is a combined-method comparison, not an equal-budget ablation.",
        "output_tokens": 1, "warmup_output_tokens": 32,
        "reset": "GPU reset between every request, CPU retained; both cleared between repetitions",
        "correctness_scope": "Every first text compared to native; separate 32-token smoke required before launch",
    }, indent=2))
    all_rows, summary = {}, {}
    for arm in ("baseline", "deep", "tail"):
        vllm, lmc = baseline if arm == "baseline" else experiment
        out = args.output_dir / arm
        out.mkdir()
        env = {k: v for k, v in os.environ.items() if "HYREX" not in k and not k.startswith("LMCACHE_TAIL_")}
        env.update(PYTHONSAFEPATH="1", PYTHONPATH=f"{lmc}:{vllm}", CUDA_VISIBLE_DEVICES="1")
        if arm != "baseline":
            env.update(VLLM_HYREX_SINGLE_FORWARD="1", LMCACHE_HYREX_LAST_STATE_ONLY="1",
                LMCACHE_HYREX_FULL_LOAD_TO_STATE="0", LMCACHE_HYREX_BATCH_FULL_PAGES="1",
                LMCACHE_HYREX_COALESCE_FULL_PAGES="1", VLLM_HYREX_Q_ONLY_REPLAY="1")
        command = [sys.executable, str(Path(__file__).with_name("audit_real_sharegpt_mp.py")),
            "--trace", "/root/hyrex_results/motivation_sharegpt_4s10t_trace.jsonl",
            "--mode", "default", "--workflow", "online", "--sessions", "4", "--min-session-turns", "10",
            "--limit", "0", "--round-robin-sessions", "--reset-between-requests",
            "--gpu", "1", "--cpu-gb", "2", "--max-output-tokens", "1", "--first-token-logprobs",
            "--vllm-source", str(vllm), "--experimental-lmcache-source", str(lmc),
            "--prefill-budget", "528" if arm == "baseline" else "2048",
            "--online-repetitions", str(args.repetitions), "--output-dir", str(out),
            "--vllm-port", "8761", "--lmcache-port", "8762", "--lmcache-http-port", "8763"]
        if arm != "baseline":
            command += ["--experimental-full-page-size", "16"]
        if arm == "tail":
            command += ["--replace-tail-checkpoint"]
        (out / "command.json").write_text(json.dumps(command, indent=2))
        print("Starting", arm, flush=True)
        with (out / "runner.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (out / "online.jsonl").read_text().splitlines()]
        resets = [json.loads(line) for line in (out / "resets.jsonl").read_text().splitlines()]
        assert len(rows) == 40 * args.repetitions
        assert len(resets) == len(rows)-1 and all(r["success"] for r in resets)
        all_rows[arm] = rows
        matches = all((a["session_id"], a["turn_index"], a["first_text"]) ==
                      (b["session_id"], b["turn_index"], b["first_text"])
                      for a, b in zip(all_rows["baseline"], rows))
        summary[arm] = {"requests": len(rows), "first_text_equal_native": matches,
            "all_mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in rows),
            "first_turn_mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in rows if r["turn_index"] == 0),
            "continuation_mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in rows if r["turn_index"] > 0)}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if not matches:
            raise RuntimeError(f"{arm} output differs from native; do not report speedup")


if __name__ == "__main__":
    main()
