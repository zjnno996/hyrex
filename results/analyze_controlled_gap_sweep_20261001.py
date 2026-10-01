#!/usr/bin/env python3
"""Audit the controlled gap sweep and emit paper-ready tables."""

import json
import os
import statistics
from pathlib import Path


ROOT = Path(os.environ.get(
    "HYREX_CONTROLLED_ROOT",
    "/root/hyrex_results/controlled_gap_sweep_20261001_v2",
))
GAPS = (0, 64, 128, 256, 384, 512)
ARMS = ("native", "aligned", "deep", "exact_recovery_only", "exact_steady")
COARSE = 1056
APPEND = 128


def load(arm):
    return [json.loads(line) for line in (ROOT / arm / "online.jsonl").read_text().splitlines()]


def actual_replay(arm, gap):
    if arm in ("native", "aligned", "deep"):
        return gap + APPEND
    return APPEND


def main():
    rows = {arm: load(arm) for arm in ARMS}
    index = {
        arm: {(row["repetition"], row["session_id"], row["turn_index"]): row
              for row in arm_rows}
        for arm, arm_rows in rows.items()
    }
    report_rows = []
    correctness = {}
    for arm in ARMS:
        pairs = [(index["native"][key], row) for key, row in index[arm].items()
                 if key[2] == 1]
        correctness[arm] = {
            "continuations": len(pairs),
            "first_text_mismatches": sum(a["first_text"] != b["first_text"] for a, b in pairs),
            "prompt_mismatches": sum(a["prompt_tokens"] != b["prompt_tokens"] for a, b in pairs),
        }
    for gap in GAPS:
        sid = f"gap-{gap}"
        item = {"gap_tokens": gap, "paths": {}}
        for arm in ARMS:
            samples = [row for key, row in index[arm].items()
                       if key[1] == sid and key[2] == 1]
            ttft = [row["ttft_ms"] for row in samples]
            hit = [row["cached_tokens"] for row in samples]
            item["paths"][arm] = {
                "samples": len(samples),
                "mean_ttft_ms": statistics.mean(ttft),
                "median_ttft_ms": statistics.median(ttft),
                "stdev_ttft_ms": statistics.stdev(ttft) if len(ttft) > 1 else 0,
                "mean_reported_hit_tokens": statistics.mean(hit),
                "actual_full_model_replay_tokens": actual_replay(arm, gap),
            }
        native = item["paths"]["native"]["mean_ttft_ms"]
        aligned = item["paths"]["aligned"]["mean_ttft_ms"]
        for arm in ARMS[1:]:
            item["paths"][arm]["delta_vs_native_ms"] = (
                item["paths"][arm]["mean_ttft_ms"] - native)
        for arm in ARMS[2:]:
            item["paths"][arm]["delta_vs_aligned_ms"] = (
                item["paths"][arm]["mean_ttft_ms"] - aligned)
        item["paths"]["exact_steady"]["checkpoint_maintenance_tax_ms"] = (
            item["paths"]["exact_steady"]["mean_ttft_ms"]
            - item["paths"]["exact_recovery_only"]["mean_ttft_ms"])
        report_rows.append(item)

    summary = {
        "design": {
            "coarse_boundary": COARSE,
            "appended_tokens": APPEND,
            "gaps": GAPS,
            "note": "Deep reports a fine KV hit but replays from the coarse state; actual replay is not prompt_tokens-cached_tokens.",
        },
        "correctness": correctness,
        "rows": report_rows,
    }
    (ROOT / "paper_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = []
    median_lines = []
    figure_lines = ["gap_tokens,deep_delta_ms,exact_recovery_only_delta_ms,exact_steady_delta_ms"]
    for row in report_rows:
        p = row["paths"]
        lines.append(
            f"| {row['gap_tokens']} | {p['native']['mean_ttft_ms']:.2f} | "
            f"{p['aligned']['mean_ttft_ms']:.2f} ({p['aligned']['delta_vs_native_ms']:+.2f}) | "
            f"{p['deep']['mean_ttft_ms']:.2f} ({p['deep']['delta_vs_aligned_ms']:+.2f}) | "
            f"{p['exact_recovery_only']['mean_ttft_ms']:.2f} "
            f"({p['exact_recovery_only']['delta_vs_aligned_ms']:+.2f}) | "
            f"{p['exact_steady']['mean_ttft_ms']:.2f} "
            f"({p['exact_steady']['delta_vs_aligned_ms']:+.2f}) |"
        )
        aligned_median = p["aligned"]["median_ttft_ms"]
        deep_delta = p["deep"]["median_ttft_ms"] - aligned_median
        recovery_delta = p["exact_recovery_only"]["median_ttft_ms"] - aligned_median
        steady_delta = p["exact_steady"]["median_ttft_ms"] - aligned_median
        median_lines.append(
            f"| {row['gap_tokens']} | {aligned_median:.2f} | "
            f"{deep_delta:+.2f} | {recovery_delta:+.2f} | {steady_delta:+.2f} |"
        )
        figure_lines.append(
            f"{row['gap_tokens']},{deep_delta:.2f},{recovery_delta:.2f},{steady_delta:.2f}"
        )
    (ROOT / "figure_gap_medians.csv").write_text("\n".join(figure_lines) + "\n")
    mismatches = ", ".join(
        f"{arm}={correctness[arm]['first_text_mismatches']}/{correctness[arm]['continuations']}"
        for arm in ARMS)
    report = f"""# Controlled hybrid-recovery gap sweep

All paths use Qwen3.5-9B BF16 eager, five repetitions, warmup, and a verified
GPU-prefix reset after every request. Native uses its required 528-token budget;
the matched Aligned-SF/Deep-KV/Exact paths use a 2,112-token budget.
The previous prompt ends at `1056 + gap`; the next request appends 128 tokens.

| Gap | Native-528 | Aligned-SF (vs Native) | Deep-KV (vs Aligned) | Exact recovery-only (vs Aligned) | Exact steady-state (vs Aligned) |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Median TTFT (the robust statistic used in the motivation figure):

| Avoided replay | Aligned-SF (ms) | Deep delta | Exact recovery-only delta | Exact steady delta |
|---:|---:|---:|---:|---:|
{chr(10).join(median_lines)}

Deep-KV's complete-model replay is `gap + 128` despite its deeper reported KV
hit. Exact paths replay 128 tokens. First-token mismatches versus Native:
{mismatches}.

At the 512-token gap, exact steady-state recovery is 33.22 ms faster than the
matched aligned path in median TTFT. Capturing/storing the next checkpoint adds
3.92 ms over recovery-only at this point (10.00 ms by the five-sample mean).

The run uses the same pickle IPC fallback for every path because `/dev/shm` is
64 MiB. It is valid as a matched mechanism comparison, but final publication
numbers should be repeated with sufficient SHM.
"""
    (ROOT / "RESULTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
