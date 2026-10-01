"""One real ShareGPT seed/resume pair; call existing CPU-offload audit runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["shallow", "deep_qkv", "deep_qonly", "tail", "tail_no_new_states", "cold_fused_reference"], required=True)
    p.add_argument("--profile-resume", action="store_true")
    p.add_argument("--verify-state-transfer", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--max-output-tokens", type=int, default=1)
    args = p.parse_args()
    if args.profile_resume and args.repetitions < 2:
        p.error("profile requires a shape-warmup pair before the profiled pair")
    rows = [json.loads(line) for line in Path(
        "/root/hyrex_results/motivation_sharegpt_4s10t_trace.jsonl").read_text().splitlines()]
    seed, resume = rows[5], rows[6]
    tok = AutoTokenizer.from_pretrained("/root/models/Qwen3.5-9B")
    a, b = [tok.encode(row["resume_prompt"]) for row in (seed, resume)]
    shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    boundary = len(a) // 16 * 16
    assert shared >= boundary, (shared, boundary)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trace = args.output_dir / "trace.jsonl"
    trace.write_text("".join(json.dumps(dict(row, turn_index=i,
                             max_output_tokens=1 if i == 0 else args.max_output_tokens)) + "\n"
                             for i, row in enumerate((seed, resume))))
    (args.output_dir / "design.json").write_text(json.dumps({
        "arm": args.arm, "seed_tokens": len(a), "resume_tokens": len(b),
        "exact_shared_tokens": shared, "tail_boundary": boundary,
        "coarse_boundary": len(a)//528*528,
        "diagnostic_byte_audit": args.verify_state_transfer,
        "diagnostic_profile_run": args.profile_resume,
        "q_only_no_cat": os.getenv("VLLM_HYREX_Q_ONLY_NO_CAT", "1") == "1",
        "selection": "mechanism probe: near-full 528-token tail, not representative average",
    }, indent=2))
    env = dict(os.environ, LMCACHE_HYREX_COALESCE_FULL_PAGES="1",
               LMCACHE_HYREX_LAST_STATE_ONLY="1", LMCACHE_HYREX_FULL_LOAD_TO_STATE="1",
               VLLM_HYREX_Q_ONLY_REPLAY="0")
    deep = args.arm in ("deep_qkv", "deep_qonly")
    if deep:
        env["LMCACHE_HYREX_FULL_LOAD_TO_STATE"] = "0"
    if args.arm == "deep_qonly":
        env["VLLM_HYREX_Q_ONLY_REPLAY"] = "1"
    env["LMCACHE_TAIL_SKIP_NEW_STATES"] = "1" if args.arm == "tail_no_new_states" else "0"
    env["LMCACHE_TAIL_VERIFY_STATE"] = "1" if args.verify_state_transfer else "0"
    env["LMCACHE_TAIL_VERIFY_MODEL_STATE"] = "1" if args.verify_state_transfer else "0"
    env["LMCACHE_HYREX_VERIFY_FULL_H2D"] = "1" if args.verify_state_transfer else "0"
    env["LMCACHE_TAIL_COLD_FUSED_REFERENCE"] = "1" if args.arm == "cold_fused_reference" else "0"
    command = [sys.executable, str(Path(__file__).with_name("audit_real_sharegpt_mp.py")),
               "--trace", str(trace), "--mode", "default", "--workflow", "online",
               "--limit", "0", "--reset-between-requests", "--gpu", "1", "--cpu-gb", "2",
               "--experimental-lmcache-source", "/root/lmcache-hyrex",
               "--experimental-full-page-size", "16", "--output-dir", str(args.output_dir),
               "--vllm-port", "8761", "--lmcache-port", "8762", "--lmcache-http-port", "8763",
               "--first-token-logprobs", "--online-repetitions", str(args.repetitions),
               "--max-output-tokens", str(args.max_output_tokens)]
    if args.arm not in ("shallow", "deep_qkv", "deep_qonly"):
        command += ["--tail-probe-boundary", str(boundary)]
    if args.arm == "cold_fused_reference":
        command += ["--no-warmup"]  # existing warmup asserts CPU cache hits
    if args.profile_resume:
        command += ["--profile-request-index", "3"]
    (args.output_dir / "command.json").write_text(json.dumps(command))
    with (args.output_dir / "runner.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    measured = [json.loads(line) for line in (args.output_dir / "online.jsonl").read_text().splitlines()]
    expected = boundary if args.arm != "shallow" else len(a)//528*528
    if args.arm == "cold_fused_reference":
        expected = 0
    for row in measured:
        if row["turn_index"] == 1 and row["cached_tokens"] != expected:
            raise RuntimeError(f"invalid recovery boundary: expected {expected}, got {row}")


if __name__ == "__main__":
    main()
