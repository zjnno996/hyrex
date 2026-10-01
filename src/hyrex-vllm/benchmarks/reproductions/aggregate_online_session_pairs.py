# SPDX-License-Identifier: Apache-2.0
"""Aggregate independently validated online baseline/HyRex result pairs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

from analyze_online_session_pair import PAIR_FIELDS, analyze_pair


T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def _last_row(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty result file: {path}")
    return rows[-1]


def aggregate_pairs(
    pairs: list[tuple[Path, Path]], *, min_repetitions: int = 3
) -> dict[str, Any]:
    if len(pairs) < min_repetitions:
        raise ValueError(
            f"need at least {min_repetitions} independent pairs, got {len(pairs)}"
        )
    all_paths = [path.resolve() for pair in pairs for path in pair]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("the same result file cannot count as multiple repetitions")

    rows = [(_last_row(reference), _last_row(candidate)) for reference, candidate in pairs]
    canonical = rows[0][0]
    canonical_names = (
        rows[0][0]["baseline"]["name"],
        rows[0][1]["baseline"]["name"],
    )
    if any(
        (
            reference["baseline"]["name"],
            candidate["baseline"]["name"],
        )
        != canonical_names
        for reference, candidate in rows[1:]
    ):
        raise ValueError("baseline names differ across repetitions")
    cross_repeat_mismatches = {
        field
        for field in PAIR_FIELDS
        for reference, candidate in rows[1:]
        for row in (reference, candidate)
        if row.get(field) != canonical.get(field)
    }
    if cross_repeat_mismatches:
        raise ValueError(
            "repetition configurations differ: "
            f"{sorted(cross_repeat_mismatches)}"
        )

    reports = [analyze_pair(reference, candidate) for reference, candidate in pairs]
    metric_names = reports[0]["improvement_percent"].keys()
    summary = {}
    for metric in metric_names:
        values = [report["improvement_percent"][metric] for report in reports]
        mean = statistics.mean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else None
        half_width = (
            T95.get(len(values), 1.96) * stddev / math.sqrt(len(values))
            if stddev is not None
            else None
        )
        summary[metric] = {
            "mean": mean,
            "sample_stddev": stddev,
            "ci95_low": mean - half_width if half_width is not None else None,
            "ci95_high": mean + half_width if half_width is not None else None,
            "repetitions": len(values),
        }
    return {
        "reference": reports[0]["reference"],
        "candidate": reports[0]["candidate"],
        "concurrency": reports[0]["concurrency"],
        "repetitions": len(reports),
        "improvement_percent": summary,
        "pairs": reports,
    }


def self_check() -> None:
    from analyze_online_session_pair import LATENCY_METRICS

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pairs = []
        for repetition, candidate_scale in enumerate((0.8, 0.7, 0.6)):
            result_paths = []
            for name, scale in (("reference", 1.0), ("candidate", candidate_scale)):
                request_path = root / f"{name}_{repetition}_requests.jsonl"
                request_path.write_text(json.dumps({
                    "arrival_index": 0,
                    "session_id": "s",
                    "turn_index": 0,
                    "prompt_tokens": 528,
                    "output_sha256": "same",
                }) + "\n")
                row = {
                    "trace": "/trace",
                    "requests": 1,
                    "concurrency": 1,
                    "arrival_model": "poisson",
                    "request_rate": 8.0,
                    "burst_size": 8,
                    "max_tokens": 2,
                    "cpu_cache_gb": 24.0,
                    "gpu_memory_utilization": 0.9,
                    "cache_state_counts": {"cpu": 1},
                    "correctness_checked": True,
                    "request_output": str(request_path),
                    "baseline": {"name": name},
                    "output_token_throughput": 1.0 / scale,
                }
                row.update({metric: 100.0 * scale for metric in LATENCY_METRICS})
                result_path = root / f"{name}_{repetition}.jsonl"
                result_path.write_text(json.dumps(row) + "\n")
                result_paths.append(result_path)
            pairs.append((result_paths[0], result_paths[1]))
        result = aggregate_pairs(pairs)
        assert result["repetitions"] == 3
        assert result["improvement_percent"]["ttft_p99_ms"]["mean"] == 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("REFERENCE", "CANDIDATE"),
        type=Path,
        required=True,
    )
    parser.add_argument("--min-repetitions", type=int, default=3)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    print(json.dumps(
        aggregate_pairs(args.pair, min_repetitions=args.min_repetitions), indent=2
    ))


if __name__ == "__main__":
    main()
