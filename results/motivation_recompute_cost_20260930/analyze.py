#!/usr/bin/env python3
"""Pair the completed native/tail ABBA run and emit motivation tables."""

import csv
import json
import math
import statistics
from pathlib import Path


SOURCE = Path("/root/hyrex_results/native_tail_abba_20260928_v3")
OUT = Path(__file__).resolve().parent
ARMS = ("1_baseline", "2_tail", "3_tail", "4_baseline")
BASELINE_ARMS = ("1_baseline", "4_baseline")
TAIL_ARMS = ("2_tail", "3_tail")


def nearest_rank(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": nearest_rank(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def load_arm(name):
    rows = [json.loads(line) for line in (SOURCE / name / "online.jsonl").open()]
    return {(row["session_id"], row["turn_index"]): row for row in rows}


arms = {name: load_arm(name) for name in ARMS}
keys = sorted(set.intersection(*(set(rows) for rows in arms.values())))
paired = []

for session_id, turn_index in keys:
    samples = [arms[name][(session_id, turn_index)] for name in ARMS]
    prompts = {sample["prompt_tokens"] for sample in samples}
    if len(prompts) != 1:
        raise RuntimeError(f"prompt mismatch for {(session_id, turn_index)}: {prompts}")

    baseline_hits = [arms[name][(session_id, turn_index)]["cached_tokens"] for name in BASELINE_ARMS]
    tail_hits = [arms[name][(session_id, turn_index)]["cached_tokens"] for name in TAIL_ARMS]
    if len(set(baseline_hits)) != 1 or len(set(tail_hits)) != 1:
        raise RuntimeError(f"cache-hit mismatch for {(session_id, turn_index)}")

    prompt = prompts.pop()
    baseline_hit = baseline_hits[0]
    tail_hit = tail_hits[0]
    baseline_replay = prompt - baseline_hit
    tail_replay = prompt - tail_hit
    saved = tail_hit - baseline_hit
    baseline_ttft = statistics.mean(
        arms[name][(session_id, turn_index)]["ttft_ms"] for name in BASELINE_ARMS
    )
    tail_ttft = statistics.mean(
        arms[name][(session_id, turn_index)]["ttft_ms"] for name in TAIL_ARMS
    )
    paired.append(
        {
            "session_id": session_id,
            "turn_index": turn_index,
            "prompt_tokens": prompt,
            "baseline_hit_tokens": baseline_hit,
            "tail_hit_tokens": tail_hit,
            "baseline_replay_tokens": baseline_replay,
            "tail_replay_tokens": tail_replay,
            "saved_full_model_tokens": saved,
            "replay_reduction_percent": 100 * saved / baseline_replay if baseline_replay else 0,
            "baseline_ttft_ms": baseline_ttft,
            "tail_ttft_ms": tail_ttft,
            "tail_minus_baseline_ms": tail_ttft - baseline_ttft,
        }
    )

continuations = [row for row in paired if row["turn_index"] > 0]
fields = list(paired[0])
with (OUT / "per_request.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(paired)

metric_fields = (
    "baseline_replay_tokens",
    "tail_replay_tokens",
    "saved_full_model_tokens",
    "replay_reduction_percent",
    "baseline_ttft_ms",
    "tail_ttft_ms",
    "tail_minus_baseline_ms",
)
summary = {
    "source": str(SOURCE),
    "design": "ABBA: native, tail, tail, native; GPU cache reset after every request",
    "requests": len(paired) * 4,
    "paired_continuations": len(continuations),
    "metrics": {field: stats([row[field] for row in continuations]) for field in metric_fields},
    "tail_faster_count": sum(row["tail_minus_baseline_ms"] < 0 for row in continuations),
    "positive_token_saving_count": sum(row["saved_full_model_tokens"] > 0 for row in continuations),
    "per_turn": [],
    "saving_bins": [],
}

for turn in sorted({row["turn_index"] for row in continuations}):
    rows = [row for row in continuations if row["turn_index"] == turn]
    summary["per_turn"].append(
        {
            "turn": turn + 1,
            "sessions": len(rows),
            **{
                field: statistics.mean(row[field] for row in rows)
                for field in (
                    "prompt_tokens",
                    "baseline_replay_tokens",
                    "tail_replay_tokens",
                    "saved_full_model_tokens",
                    "replay_reduction_percent",
                    "baseline_ttft_ms",
                    "tail_ttft_ms",
                    "tail_minus_baseline_ms",
                )
            },
        }
    )

for lower, upper in ((0, 127), (128, 255), (256, 383), (384, 10**9)):
    rows = [row for row in continuations if lower <= row["saved_full_model_tokens"] <= upper]
    if rows:
        summary["saving_bins"].append(
            {
                "saved_token_range": f"{lower}-{upper if upper < 10**9 else 'inf'}",
                "samples": len(rows),
                "mean_saved_tokens": statistics.mean(row["saved_full_model_tokens"] for row in rows),
                "mean_tail_minus_baseline_ms": statistics.mean(row["tail_minus_baseline_ms"] for row in rows),
                "tail_faster_count": sum(row["tail_minus_baseline_ms"] < 0 for row in rows),
            }
        )

(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

metrics = summary["metrics"]
turn_lines = []
for row in summary["per_turn"]:
    turn_lines.append(
        f"| {row['turn']} | {row['prompt_tokens']:.1f} | {row['baseline_replay_tokens']:.1f} | "
        f"{row['tail_replay_tokens']:.1f} | {row['saved_full_model_tokens']:.1f} | "
        f"{row['replay_reduction_percent']:.1f}% | {row['baseline_ttft_ms']:.2f} | "
        f"{row['tail_ttft_ms']:.2f} | {row['tail_minus_baseline_ms']:+.2f} |"
    )

bin_lines = []
for row in summary["saving_bins"]:
    bin_lines.append(
        f"| {row['saved_token_range']} | {row['samples']} | {row['mean_saved_tokens']:.1f} | "
        f"{row['mean_tail_minus_baseline_ms']:+.2f} | {row['tail_faster_count']}/{row['samples']} |"
    )

report = f"""# Motivation: coarse recovery vs. one tail checkpoint

## Experiment

- Model: Qwen3.5-9B, BF16, eager.
- Workload: four real ShareGPT sessions, ten cumulative turns per session.
- Comparison: unmodified aligned recovery vs. the current prototype that replaces
  the last coarse state with one 16-token-aligned tail state
  (`--replace-tail-checkpoint`). This is an ablation, not the final retention policy.
- Protocol: ABBA (`native -> tail -> tail -> native`), warm-up before each service, and GPU cache reset after every request while retaining CPU cache.
- Pairing: the two native and two tail observations are averaged for every identical session/turn. First turns are excluded from recovery statistics.
- Correctness: all 160 first-token checks passed; no formal JIT warnings were reported.

## Main result

| Metric (36 continuation pairs) | Mean | Median | P95 |
|---|---:|---:|---:|
| Native full-model replay | {metrics['baseline_replay_tokens']['mean']:.1f} tok | {metrics['baseline_replay_tokens']['median']:.1f} | {metrics['baseline_replay_tokens']['p95']:.1f} |
| Tail full-model replay | {metrics['tail_replay_tokens']['mean']:.1f} tok | {metrics['tail_replay_tokens']['median']:.1f} | {metrics['tail_replay_tokens']['p95']:.1f} |
| Full-model tokens avoided | {metrics['saved_full_model_tokens']['mean']:.1f} tok | {metrics['saved_full_model_tokens']['median']:.1f} | {metrics['saved_full_model_tokens']['p95']:.1f} |
| Replay reduction | {metrics['replay_reduction_percent']['mean']:.1f}% | {metrics['replay_reduction_percent']['median']:.1f}% | {metrics['replay_reduction_percent']['p95']:.1f}% |
| Native TTFT | {metrics['baseline_ttft_ms']['mean']:.2f} ms | {metrics['baseline_ttft_ms']['median']:.2f} ms | {metrics['baseline_ttft_ms']['p95']:.2f} ms |
| Tail TTFT | {metrics['tail_ttft_ms']['mean']:.2f} ms | {metrics['tail_ttft_ms']['median']:.2f} ms | {metrics['tail_ttft_ms']['p95']:.2f} ms |
| Tail - native TTFT | {metrics['tail_minus_baseline_ms']['mean']:+.2f} ms | {metrics['tail_minus_baseline_ms']['median']:+.2f} ms | {metrics['tail_minus_baseline_ms']['p95']:+.2f} ms |

The coarse 528-token recovery boundary causes substantial replay: **400.0 tokens per
continuation on average**. One tail checkpoint avoids **267.6 full-model tokens
(63.3%)**, with positive token saving in {summary['positive_token_saving_count']}/36 cases. However, the current
unconditional implementation is slower on average: **152.37 -> 170.10 ms
(+11.6%)**, and is faster in only {summary['tail_faster_count']}/36 pairs.

Two paired examples make the distinction concrete:

| Prompt | Native hit/replay | Tail hit/replay | Avoided | Native -> tail TTFT |
|---:|---:|---:|---:|---:|
| 1,096 | 528 / 568 | 992 / 104 | 464 (81.7%) | 203.04 -> 134.43 ms |
| 830 | 528 / 302 | 720 / 110 | 192 (63.6%) | 134.21 -> 290.36 ms |

The first case realizes the expected speedup; the second is a measured outlier that
shows why token hit length alone is not a safe recovery objective.

## Per-turn averages

Turn 1 is the cold request and is intentionally omitted.

| Turn | Prompt tok | Native replay | Tail replay | Avoided | Reduction | Native TTFT | Tail TTFT | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(turn_lines)}

## Break-even evidence

| Avoided full-model tokens | Samples | Mean avoided | Mean TTFT delta | Tail faster |
|---:|---:|---:|---:|---:|
{chr(10).join(bin_lines)}

The tail path is not profitable for every recovery. In this run, cases avoiding at
least 384 tokens are the only group with a lower mean TTFT, and 7/10 of those cases
are faster. This is evidence for a **cost-aware checkpoint policy**, not for adding
a tail checkpoint after every request.

## Measured cost context

| Cost item | Measured value | Interpretation |
|---|---:|---|
| One padded tail SSM checkpoint in CPU cache | 49.5 MiB/session; 198 MiB for four sessions | Intrinsic capacity and D2H cost |
| Earlier synchronous save barrier | 14.59 ms mean | Implementation cost; later fused path removes this explicit barrier |
| Fused direct-snapshot transient GPU storage | 99 MiB | Guard plus in-flight snapshot across 24 recurrent layers |
| Fine page-by-page Full-KV H2D | 18.753 ms for 33 MiB | Bad transfer organization |
| Coalesced Full-KV H2D | 1.853 ms for the same 33 MiB | Fine matching must use coarse DMA |

The proposed extra tail does not require loading two recurrent states during
recovery: the system selects and loads one state. Its intrinsic added costs are checkpoint
capture, one extra CPU-resident state, D2H at save time, lookup metadata, and any
deeper Full-KV transfer. Split forwards, repeated concatenation, scalar syncs, and
page-by-page DMA are avoidable implementation costs.

## Proposed design

1. Keep the existing 528-token coarse checkpoints and add **at most one replaceable
   tail state per active session**, aligned to the deepest reusable 16-token KV
   boundary.
2. Index KV and recurrent state independently and return `(L_KV, L_state)`. Recover
   from the state that minimizes predicted TTFT, rather than always choosing the
   deepest hit.
3. Capture the tail in the normal single forward, snapshot to an immutable buffer,
   and perform D2H asynchronously after the critical TTFT path. Match at 16-token
   granularity but coalesce H2D/D2H transfers.
4. Admit or retain a tail only when its expected saved compute exceeds capture,
   transfer, and capacity cost:

   `p_reuse * (T_forward(delta_tokens) - T_extra_KV_H2D) > T_capture + T_D2H + lambda * 49.5 MiB`.

5. For chat/agent workloads, prioritize turn ends, tool-result boundaries, branch
   roots, and rollback points. Do not checkpoint every intermediate token range.

This yields the paper's core motivation: **coarse state alignment wastes substantial
recomputation, but maximum cache hit is not equivalent to minimum TTFT; hybrid
recovery must jointly choose state placement, independent matches, transfer layout,
and the fastest valid recovery path.**
"""
(OUT / "RESULTS.md").write_text(report)

print(json.dumps({
    "output": str(OUT),
    "continuations": len(continuations),
    "mean_native_replay": metrics["baseline_replay_tokens"]["mean"],
    "mean_tail_replay": metrics["tail_replay_tokens"]["mean"],
    "mean_saved": metrics["saved_full_model_tokens"]["mean"],
    "mean_ttft_delta_ms": metrics["tail_minus_baseline_ms"]["mean"],
}, indent=2))
