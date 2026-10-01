# SPDX-License-Identifier: Apache-2.0
"""Build a paper-ready table only from complete, valid paired E2E matrices."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from analyze_online_session_pair import LATENCY_METRICS, analyze_pair
from hybrid_baseline_config import BASELINES


def last_row(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"empty result: {path}")
    return rows[-1]


def paths(directory: Path, reference: str, concurrency: int, repetition: int) -> tuple[Path, Path]:
    if reference == "marconi":
        prefix = ""
        reference_name = reference
        candidate_name = "hyrex"
    else:
        prefix = f"{reference}_vs_"
        reference_name = reference
        candidate_name = "hyrex"
    return (
        directory / f"{prefix}{reference_name}_c{concurrency}_r{repetition}.jsonl",
        directory / f"{prefix}{candidate_name}_c{concurrency}_r{repetition}.jsonl",
    )


def pressure(row: dict[str, Any]) -> tuple[int, int]:
    backend = row["baseline"]["cache_backend"]
    if backend == "lmcache":
        counters = row.get("lmcache_l1") or {}
        return int(counters.get("write_chunks", 0)), int(counters.get("evicted_chunks", 0))
    counters = row.get("cache_pressure") or {}
    return int(counters.get("admitted_blocks", 0)), int(counters.get("evicted_blocks", 0))


def validate_cell(reference_path: Path, candidate_path: Path, reference: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = analyze_pair(reference_path, candidate_path)
    if report["reference"] != reference or report["candidate"] != "hyrex":
        raise ValueError("unexpected baseline pair")
    reference_row, candidate_row = last_row(reference_path), last_row(candidate_path)
    for name, row in ((reference, reference_row), ("hyrex", candidate_row)):
        admitted, evicted = pressure(row)
        if row["baseline"]["cache_backend"] == "none":
            continue
        if admitted <= 0 or evicted <= 0:
            raise ValueError(f"{name} lacks required cache pressure")
    return reference_row, candidate_row


def percent(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100


def build(
    directories: dict[str, Path], concurrencies: list[int], repetitions: int
) -> tuple[str, dict[str, Any]]:
    if repetitions < 3:
        raise ValueError("formal table requires at least three repetitions")
    lines = [
        "| Comparison | C | Baseline TTFT P50 | HyRex TTFT P50 | Δ | Baseline TTFT P99 | HyRex TTFT P99 | Δ | Baseline TPOT P50 | HyRex TPOT P50 | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    evidence: dict[str, Any] = {}
    for reference, directory in directories.items():
        if reference not in BASELINES or reference == "hyrex":
            raise ValueError(f"unknown reference baseline: {reference}")
        for concurrency in concurrencies:
            pairs = [
                validate_cell(*paths(directory, reference, concurrency, repetition), reference)
                for repetition in range(repetitions)
            ]
            metrics = {}
            for metric in LATENCY_METRICS:
                reference_values = [float(pair[0][metric]) for pair in pairs]
                candidate_values = [float(pair[1][metric]) for pair in pairs]
                metrics[metric] = {
                    "baseline_median": statistics.median(reference_values),
                    "hyrex_median": statistics.median(candidate_values),
                }
                metrics[metric]["improvement_percent"] = percent(
                    metrics[metric]["baseline_median"], metrics[metric]["hyrex_median"]
                )
            telemetry = {}
            for name, index in (("baseline", 0), ("hyrex", 1)):
                rows = [pair[index] for pair in pairs]
                if any("recovery_telemetry" not in row for row in rows):
                    raise ValueError(f"{reference} C{concurrency} lacks recovery telemetry")
                keys = set().union(*(row["recovery_telemetry"] for row in rows))
                telemetry[name] = {
                    key: statistics.median(
                        float(row["recovery_telemetry"].get(key, 0.0)) for row in rows
                    )
                    for key in sorted(keys)
                }
            key = f"{reference}_c{concurrency}"
            evidence[key] = {
                "repetitions": repetitions,
                "metrics": metrics,
                "recovery_telemetry": telemetry,
                "hyrex_policy_counts": {
                    policy: statistics.median(
                        float(
                            row.get("hyrex_decisions", {})
                            .get("policies", {})
                            .get(policy, 0)
                        )
                        for row in (pair[1] for pair in pairs)
                    )
                    for policy in sorted(
                        set().union(
                            *(
                                pair[1]
                                .get("hyrex_decisions", {})
                                .get("policies", {})
                                for pair in pairs
                            )
                        )
                    )
                },
            }
            columns = (
                metrics["ttft_p50_ms"], metrics["ttft_p99_ms"], metrics["tpot_p50_ms"]
            )
            lines.append(
                f"| {reference} vs HyRex | {concurrency} | "
                + " | ".join(
                    f"{column['baseline_median']:.3f} | {column['hyrex_median']:.3f} | {column['improvement_percent']:+.2f}%"
                    for column in columns
                )
                + " |"
            )
    return "\n".join(lines) + "\n", evidence


def self_check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        request = directory / "requests.jsonl"
        request.write_text(json.dumps({
            "arrival_index": 0, "session_id": "s", "turn_index": 0,
            "prompt_tokens": 1, "output_sha256": "same",
        }) + "\n")
        for repetition in range(3):
            for name, scale in (("marconi", 1.0), ("hyrex", 0.5)):
                row = {
                    "trace": "/trace", "trace_sha256": "trace", "prompt_field": "resume_prompt",
                    "execution_mode": "online", "workload": "continuous_online_multiturn",
                    "requests": 1, "concurrency": 1, "arrival_model": "poisson",
                    "request_rate": 8.0, "burst_size": 8, "zipf_exponent": 1.2,
                    "zipf_max_rank": 64, "request_timeout": 1.0, "execution_timeout": 1.0,
                    "max_tokens": 2, "cpu_cache_gb": 24.0, "gpu_memory_utilization": 0.9,
                    "cache_state_counts": {"cpu": 1}, "source_provenance": {"head": "x"},
                    "calibration_sha256": None, "correctness_checked": name == "hyrex",
                    "request_output": str(request), "baseline": {"name": name, "cache_backend": "lmcache" if name == "marconi" else "native"},
                    "output_token_throughput": 1.0 / scale,
                    "recovery_telemetry": {"retrieve_bytes": 2.0, "modeled_replay_ms": 1.0},
                    "hyrex_decisions": {
                        "policies": {"all_load": 1} if name == "hyrex" else {}
                    },
                    "lmcache_l1": {"write_chunks": 2, "evicted_chunks": 1} if name == "marconi" else None,
                    "cache_pressure": {"admitted_blocks": 2, "evicted_blocks": 1} if name == "hyrex" else None,
                }
                row.update({metric: 100.0 * scale for metric in LATENCY_METRICS})
                (directory / f"{name}_c1_r{repetition}.jsonl").write_text(json.dumps(row) + "\n")
        table, evidence = build({"marconi": directory}, [1], 3)
        assert "marconi vs HyRex" in table
        assert evidence["marconi_c1"]["metrics"]["ttft_p50_ms"]["improvement_percent"] == 50.0
        assert evidence["marconi_c1"]["hyrex_policy_counts"] == {"all_load": 1.0}


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", action="append", required=True, metavar="BASELINE=DIR",
        help="one complete matrix directory per reference baseline",
    )
    parser.add_argument("--concurrencies", default="1,4,8,16,32")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()
    directories = {}
    for item in args.comparison:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--comparison must use BASELINE=DIR")
        directories[name] = Path(raw_path)
    table, evidence = build(
        directories, [int(value) for value in args.concurrencies.split(",")], args.repetitions
    )
    args.markdown_output.write_text(table)
    args.evidence_output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
