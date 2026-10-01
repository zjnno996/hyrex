#!/usr/bin/env python3
"""Summarize measured CPU recovery on the ShareGPT MP motivation run."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def common_prefix(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="/root/models/Qwen3.5-9B")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    trace = {(r["session_id"], r["turn_index"]): r for r in read_jsonl(args.trace)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokens(row, field):
        return tokenizer.encode(row[field], add_special_tokens=False)

    if args.online:
        results = read_jsonl(args.run_dir / "online.jsonl")
        prior = {}
        audited = []
        for result in results:
            row = trace[result["session_id"], result["turn_index"]]
            ids = tokens(row, "resume_prompt")
            if len(ids) != result["prompt_tokens"]:
                raise ValueError("online tokenizer length does not match vLLM")
            common = max((common_prefix(ids, prev)
                          for prev in prior.get(result["session_id"], [])), default=0)
            aligned = common // 528 * 528
            if result["cached_tokens"] > aligned:
                raise ValueError("online hit exceeds prior aligned prefix")
            audited.append({
                **result,
                "max_prior_common_prefix": common,
                "aligned_prior_prefix": aligned,
                "missing_aligned_tokens": aligned - result["cached_tokens"],
                "unrecovered_prior_tokens": common - result["cached_tokens"],
            })
            prior.setdefault(result["session_id"], []).append(ids)
        summary = {
            "sessions": len({r["session_id"] for r in audited}),
            "requests": len(audited),
            "all_hits_chunk_aligned": all(r["cached_tokens"] % 528 == 0 for r in audited),
            "by_turn": {
                str(turn): {
                    "requests": len(group),
                    "cache_hit_requests": sum(r["cached_tokens"] > 0 for r in group),
                    "eligible_aligned_requests": sum(r["aligned_prior_prefix"] > 0 for r in group),
                    "missing_aligned_requests": sum(r["missing_aligned_tokens"] > 0 for r in group),
                    "median_elapsed_ms": round(statistics.median(r["elapsed_ms"] for r in group), 2),
                    "median_ttft_ms": round(statistics.median(r["ttft_ms"] for r in group), 2),
                    "median_unrecovered_prior_tokens": statistics.median(
                        r["unrecovered_prior_tokens"] for r in group
                    ),
                }
                for turn in sorted({r["turn_index"] for r in audited})
                if (group := [r for r in audited if r["turn_index"] == turn])
            },
        }
        (args.run_dir / "online_audited.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in audited)
        )
        (args.run_dir / "online_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return

    seeds = read_jsonl(args.run_dir / "seed.jsonl")
    resumes = read_jsonl(args.run_dir / "resume.jsonl")

    prior = []
    for result in seeds:
        row = trace[result["session_id"], result["turn_index"]]
        ids = tokens(row, "cache_prompt")
        if len(ids) != result["prompt_tokens"]:
            raise ValueError("seed tokenizer length does not match vLLM")
        prior.append(ids)

    audited = []
    for result in resumes:
        row = trace[result["session_id"], result["turn_index"]]
        ids = tokens(row, "resume_prompt")
        if len(ids) != result["prompt_tokens"]:
            raise ValueError("resume tokenizer length does not match vLLM")
        observed = result["cached_tokens"]
        max_available_prefix = max(common_prefix(ids, prev) for prev in prior)
        if observed > max_available_prefix:
            raise ValueError("observed hit exceeds every previously sent prompt")
        audited.append({
            **result,
            "max_prior_common_prefix": max_available_prefix,
            "aligned_prior_prefix": max_available_prefix // 528 * 528,
            "unrecovered_prior_tokens": max_available_prefix - observed,
        })
        prior.append(ids)

    losses = [r["unrecovered_prior_tokens"] for r in audited]
    summary = {
        "sessions": len({r["session_id"] for r in resumes}),
        "seed_requests": len(seeds),
        "resume_requests": len(resumes),
        "all_hits_chunk_aligned": all(r["cached_tokens"] % 528 == 0 for r in audited),
        "hit_histogram": dict(sorted(Counter(r["cached_tokens"] for r in audited).items())),
        "below_aligned_prior_prefix": sum(
            r["cached_tokens"] < r["aligned_prior_prefix"] for r in audited
        ),
        "positive_gap_requests": sum(value > 0 for value in losses),
        "mean_gap_tokens": round(statistics.mean(losses), 2),
        "median_gap_tokens": statistics.median(losses),
        "max_gap_tokens": max(losses),
        "sum_gap_tokens": sum(losses),
    }
    (args.run_dir / "audited.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in audited)
    )
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
