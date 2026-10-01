#!/usr/bin/env python3
"""Turn the 16-session native/tail run into paper-facing motivation evidence."""

import json
import math
import statistics
from pathlib import Path


SOURCE = Path("/root/hyrex_results/native_tail_16s_c1_20260930_v1")
OUT = Path("/root/hyrex_results/motivation_story_20260930")


def load(arm):
    return {(row["session_id"], row["turn_index"]): row for row in map(
        json.loads, (SOURCE / arm / "online.jsonl").read_text().splitlines())}


def pctl(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def select_loso(rows):
    thresholds = list(range(0, 529, 16)) + [10**9]
    selected = []
    chosen = []
    for session in sorted({row["session"] for row in rows}):
        train = [row for row in rows if row["session"] != session]
        threshold = min(
            (statistics.mean(row["tail_ttft"] if row["gap"] >= candidate
                             else row["native_ttft"] for row in train), candidate)
            for candidate in thresholds
        )[1]
        chosen.append(threshold)
        selected.extend({**row, "threshold": threshold,
                         "selected_ttft": row["tail_ttft"] if row["gap"] >= threshold
                         else row["native_ttft"]}
                        for row in rows if row["session"] == session)
    return selected, chosen


def policy_summary(rows):
    selected, thresholds = select_loso(rows)
    native = statistics.mean(row["native_ttft"] for row in selected)
    policy = statistics.mean(row["selected_ttft"] for row in selected)
    return {
        "requests": len(selected),
        "thresholds_tokens": sorted(set(thresholds)),
        "tail_selections": sum(row["gap"] >= row["threshold"] for row in selected),
        "native_mean_ttft_ms": native,
        "policy_mean_ttft_ms": policy,
        "mean_ttft_change_percent": 100 * (policy / native - 1),
    }


def main():
    native, tail = load("baseline"), load("tail")
    assert native.keys() == tail.keys()
    rows = []
    for key in native:
        if key[1] == 0:
            continue
        n, t = native[key], tail[key]
        assert n["prompt_tokens"] == t["prompt_tokens"]
        rows.append({
            "session": key[0], "turn": key[1], "prompt": n["prompt_tokens"],
            "native_hit": n["cached_tokens"], "tail_hit": t["cached_tokens"],
            "gap": t["cached_tokens"] - n["cached_tokens"],
            "native_ttft": n["ttft_ms"], "tail_ttft": t["ttft_ms"],
        })
    gaps = [row["gap"] for row in rows]
    bins = []
    for lower, upper in ((0, 127), (128, 255), (256, 383), (384, 512)):
        sample = [row for row in rows if lower <= row["gap"] <= upper]
        bins.append({
            "range": f"{lower}-{upper}", "requests": len(sample),
            "mean_gap": statistics.mean(row["gap"] for row in sample),
            "mean_tail_minus_native_ms": statistics.mean(
                row["tail_ttft"] - row["native_ttft"] for row in sample),
            "tail_faster": sum(row["tail_ttft"] < row["native_ttft"] for row in sample),
        })
    max_native = max(rows, key=lambda row: row["native_ttft"])
    clean_rows = [row for row in rows if row is not max_native]
    summary = {
        "source": str(SOURCE),
        "continuations": len(rows),
        "gap": {
            "positive_requests": sum(gap > 0 for gap in gaps),
            "positive_percent": 100 * sum(gap > 0 for gap in gaps) / len(gaps),
            "mean_tokens": statistics.mean(gaps),
            "median_tokens": statistics.median(gaps),
            "p95_tokens": pctl(gaps, .95),
            "max_tokens": max(gaps),
        },
        "bins": bins,
        "loso_policy_raw": policy_summary(rows),
        "loso_policy_without_max_native_ttft": {
            "excluded": {key: max_native[key] for key in ("session", "turn", "native_ttft")},
            **policy_summary(clean_rows),
        },
        "limitations": [
            "Tail hit is the prototype's exact executable boundary, not a claim that native LMCache physically stored the same finer-grained state.",
            "The LOSO selector is an offline opportunity analysis, not an online HyRex result.",
            "One first-token near-tie differs across 120 total paired requests.",
        ],
    }
    OUT.mkdir(exist_ok=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    bin_rows = "\n".join(
        f"| {row['range']} | {row['requests']} | {row['mean_gap']:.1f} | "
        f"{row['mean_tail_minus_native_ms']:+.2f} | {row['tail_faster']}/{row['requests']} |"
        for row in bins
    )
    raw, clean = summary["loso_policy_raw"], summary["loso_policy_without_max_native_ttft"]
    report = f"""# Paper-facing motivation evidence

## Boundary opportunity

Across {len(rows)} continuation requests from 16 real ShareGPT sessions, an exact
tail state advances the executable recovery boundary in
**{summary['gap']['positive_requests']}/{len(rows)} requests
({summary['gap']['positive_percent']:.1f}%)**. The advance is
**{summary['gap']['mean_tokens']:.1f} tokens on average**, with median
{summary['gap']['median_tokens']:.1f}, P95 {summary['gap']['p95_tokens']}, and maximum
{summary['gap']['max_tokens']} tokens.

This is executable-boundary opportunity, not proof that the unmodified baseline
physically retained the same fine-grained state.

## TTFT break-even

| Avoided full-model tokens | Requests | Mean avoided | Tail - Native TTFT | Tail faster |
|---:|---:|---:|---:|---:|
{bin_rows}

Exact recovery has conditional value: its fixed realization cost dominates short
gaps, while the 384--512-token group is faster on average. Therefore maximum hit
length is not a safe policy objective.

## Offline policy opportunity

A leave-one-session-out selector learns its token threshold on 15 sessions and
applies it to the held-out session. It chooses thresholds {raw['thresholds_tokens']},
uses Tail for {raw['tail_selections']}/{raw['requests']} requests, and changes mean
TTFT from {raw['native_mean_ttft_ms']:.2f} to {raw['policy_mean_ttft_ms']:.2f} ms
({raw['mean_ttft_change_percent']:.2f}%). After excluding the single known maximum
Native/JIT-contaminated observation, the change is
{clean['native_mean_ttft_ms']:.2f} to {clean['policy_mean_ttft_ms']:.2f} ms
({clean['mean_ttft_change_percent']:.2f}%).

This is an offline opportunity study, not an online HyRex speedup. It motivates a
runtime recovery planner and Native fallback.

## Safe paper claim

The evidence supports: coarse recovery leaves frequent executable-prefix
opportunities; unconditional exact recovery is not profitable; and recovery value
crosses a measurable break-even point. It does not yet establish the final HyRex
end-to-end gain or a capacity-aware policy.
"""
    (OUT / "RESULTS.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
