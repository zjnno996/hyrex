# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize one fair online baseline/HyRex result pair."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PAIR_FIELDS = (
    "trace",
    "trace_sha256",
    "prompt_field",
    "execution_mode",
    "workload",
    "requests",
    "concurrency",
    "arrival_model",
    "request_rate",
    "burst_size",
    "zipf_exponent",
    "zipf_max_rank",
    "request_timeout",
    "execution_timeout",
    "calibration_sha256",
    "max_tokens",
    "cpu_cache_gb",
    "gpu_memory_utilization",
    "cache_state_counts",
    "source_provenance",
)
LATENCY_METRICS = (
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "tpot_p99_ms",
)


def _last_row(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty result file: {path}")
    return rows[-1]


def _request_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {int(row["arrival_index"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate arrival_index in {path}")
    return indexed


def analyze_pair(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = _last_row(reference_path)
    candidate = _last_row(candidate_path)
    mismatches = {
        field: (reference.get(field), candidate.get(field))
        for field in PAIR_FIELDS
        if reference.get(field) != candidate.get(field)
    }
    if mismatches:
        raise ValueError(f"unpaired result configuration: {mismatches}")
    if not candidate.get("correctness_checked"):
        raise ValueError("candidate was not run with a correctness reference")

    reference_requests = _request_rows(Path(reference["request_output"]))
    candidate_requests = _request_rows(Path(candidate["request_output"]))
    if reference_requests.keys() != candidate_requests.keys():
        raise ValueError("request arrival sets differ")
    request_fields = ("session_id", "turn_index", "prompt_tokens", "output_sha256")
    incorrect = [
        index
        for index in reference_requests
        if any(
            reference_requests[index].get(field)
            != candidate_requests[index].get(field)
            for field in request_fields
        )
    ]
    if incorrect:
        raise ValueError(f"request correctness mismatch: {incorrect[:3]}")

    improvements = {
        metric: (reference[metric] - candidate[metric]) / reference[metric] * 100
        for metric in LATENCY_METRICS
    }
    improvements["output_token_throughput"] = (
        candidate["output_token_throughput"]
        - reference["output_token_throughput"]
    ) / reference["output_token_throughput"] * 100
    return {
        "reference": reference["baseline"]["name"],
        "candidate": candidate["baseline"]["name"],
        "requests": reference["requests"],
        "concurrency": reference["concurrency"],
        "cache_state_counts": reference["cache_state_counts"],
        "correctness_requests": len(reference_requests),
        "improvement_percent": improvements,
        "candidate_hyrex_decisions": candidate.get("hyrex_decisions"),
    }


def self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        request_paths = []
        for name in ("reference", "candidate"):
            path = root / f"{name}_requests.jsonl"
            path.write_text(json.dumps({
                "arrival_index": 0,
                "session_id": "s",
                "turn_index": 1,
                "prompt_tokens": 528,
                "output_sha256": "same",
            }) + "\n")
            request_paths.append(path)
        base = {
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
            "ttft_p50_ms": 10.0,
            "ttft_p95_ms": 10.0,
            "ttft_p99_ms": 10.0,
            "tpot_p50_ms": 2.0,
            "tpot_p95_ms": 2.0,
            "tpot_p99_ms": 2.0,
            "output_token_throughput": 1.0,
        }
        result_paths = []
        for name, request_path, scale in zip(
            ("reference", "candidate"), request_paths, (1.0, 0.5)
        ):
            row = dict(base)
            row["baseline"] = {"name": name}
            row["request_output"] = str(request_path)
            for metric in LATENCY_METRICS:
                row[metric] *= scale
            result_path = root / f"{name}.jsonl"
            result_path.write_text(json.dumps(row) + "\n")
            result_paths.append(result_path)
        result = analyze_pair(*result_paths)
        assert result["correctness_requests"] == 1
        assert result["improvement_percent"]["ttft_p99_ms"] == 50.0


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_pair(args.reference, args.candidate), indent=2))


if __name__ == "__main__":
    main()
