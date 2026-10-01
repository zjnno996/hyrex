# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize Hybrid recovery JSONL cells.

The runner intentionally keeps raw rows, including failed and diagnostic
cells.  This small offline pass is the single place that turns those rows into
an oracle comparison: a row must pass the cache-state gate, and all policies
in a cell must agree on the first-token signature.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


POLICIES = ("P1", "P2", "P3", "P4", "P5", "Adaptive")
DEFAULT_CHUNK_SIZE = 528
CELL_FIELDS = (
    "backend",
    "shared_prefix_tokens",
    "suffix_tokens",
    "output_tokens",
    "requests",
    "concurrency",
    "start_index",
    "seed",
    "suffix_only",
    "retain_shared_prefix",
    "lmcache_eviction_policy",
    "lmcache_strict_layer_load",
    "p5_serialize_terminal",
    "serialize_all_load",
    "evict_gpu_cache",
    "evict_requests",
    "warmup_requests",
    "warmup_repeats",
    "warmup_settle_seconds",
    "adaptive_replay_below_tokens",
    "adaptive_max_inflight_h2d",
    "adaptive_contention_replay_below_tokens",
)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def positive_sum(value: Any) -> float:
    if isinstance(value, dict):
        return sum(float(item) for item in value.values() if float(item) > 0)
    return float(value or 0)


def actual_missing_pages(row: dict[str, Any]) -> int | None:
    pages = row.get("lmcache_missing_pages")
    if pages is None:
        needs = row.get("lmcache_need_h2d_tokens")
        if needs is None:
            needs = [
                item.get("need_h2d_tokens")
                for item in row.get("lmcache_prefix_matches", [])
            ]
        if isinstance(needs, list) and needs and all(
            isinstance(value, int)
            and value > 0
            and value % DEFAULT_CHUNK_SIZE == 0
            for value in needs
        ):
            pages = [value // DEFAULT_CHUNK_SIZE for value in needs]
    if not isinstance(pages, list) or not pages or any(
        not isinstance(value, int) or value < 1 for value in pages
    ):
        return None
    return pages[0] if len(set(pages)) == 1 else None


def cache_state_gate(row: dict[str, Any]) -> tuple[bool, str]:
    requests = row.get("requests")
    shared = row.get("shared_prefix_tokens")
    matches = row.get("lmcache_prefix_matches")
    if row.get("backend") != "lmcache":
        return False, "backend"
    if not row.get("independent_process") or not row.get("suffix_only"):
        return False, "protocol"
    if not row.get("retain_shared_prefix") or not isinstance(requests, int):
        return False, "shared-prefix protocol"
    # P2 deliberately performs no LMCache lookup.  Its cache-state gate is
    # completed by a paired Load row in summarize().
    if row.get("policy") == "P2" and not matches:
        return True, "paired control"
    if (
        actual_missing_pages(row) is None
        or not isinstance(matches, list)
        or len(matches) != requests
    ):
        return False, "non-uniform or missing page bucket"
    if any(
        item.get("local_gpu_tokens") != shared
        or item.get("cpu_tokens", 0) < shared + item.get("need_h2d_tokens", 0)
        or item.get("need_h2d_tokens", 0) <= 0
        for item in matches
    ):
        return False, "GPU-prefix/CPU-suffix gate"
    return True, "ok"


def h2d_bytes(row: dict[str, Any]) -> float:
    return max(
        float(row.get("h2d_cpu_to_gpu_bytes", 0) or 0),
        positive_sum(row.get("h2d_cpu_to_gpu_group_bytes", {})),
        positive_sum(row.get("lmcache_actual_h2d_object_group_bytes", {})),
    )


def has_h2d(row: dict[str, Any]) -> bool:
    return h2d_bytes(row) > 0 or bool(
        row.get("lmcache_actual_h2d_transferred_object_groups")
    )


def valid_row(row: dict[str, Any]) -> tuple[bool, str]:
    ok, reason = cache_state_gate(row)
    if not ok:
        return False, reason
    policy = row.get("policy")
    if policy not in POLICIES:
        return False, "policy"
    if not row.get("first_token_signature"):
        return False, "missing first-token signature"
    if policy == "P2":
        if has_h2d(row) or row.get("lmcache_h2d_retrieve_requests", 0):
            return False, "P2 has H2D"
    elif policy == "Adaptive":
        decisions = row.get("lmcache_adaptive_replays", [])
        if len(decisions) not in (0, row["requests"]):
            return False, "mixed Adaptive decisions"
        if len(decisions) == 0 and not has_h2d(row):
            return False, "Adaptive has neither replay nor H2D"
        if len(decisions) == row["requests"] and has_h2d(row):
            return False, "Adaptive replay has H2D"
    elif not has_h2d(row):
        return False, f"{policy} has no H2D"
    return True, "ok"


def cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    defaults: dict[str, Any] = {
        "p5_serialize_terminal": False,
        "serialize_all_load": False,
        "evict_gpu_cache": False,
        "evict_requests": 64,
        "warmup_requests": 72,
        "warmup_repeats": 1,
        "warmup_settle_seconds": 1.0,
    }
    return tuple(
        row.get(field) if row.get(field) is not None else defaults.get(field)
        for field in CELL_FIELDS
    )


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    rejected["invalid JSON"] += 1
                    continue
                ok, reason = valid_row(row)
                if ok:
                    rows.append(row)
                else:
                    rejected[reason] += 1
    return rows, rejected


def load_cost(row: dict[str, Any]) -> dict[str, float | None]:
    lookup = median([float(value) for value in row.get("lmcache_lookup_ms", [])])
    h2d = median([float(value) for value in row.get("lmcache_h2d_total_ms", [])])
    if h2d is None:
        wait = median([float(value) for value in row.get("lmcache_h2d_wait_ms", [])])
        sync = median([float(value) for value in row.get("lmcache_h2d_cuda_sync_ms", [])])
        h2d = (wait or 0.0) + (sync or 0.0) if wait is not None or sync is not None else None
    if h2d is None:
        h2d = float(row.get("h2d_cpu_to_gpu_service_ms", 0) or 0)
        h2d += float(row.get("h2d_cpu_to_gpu_queue_ms", 0) or 0)
        h2d = h2d or None
    return {
        "lookup_ms": lookup,
        "h2d_ms": h2d,
        "load_ms": (lookup or 0.0) + (h2d or 0.0)
        if lookup is not None or h2d is not None
        else None,
        "h2d_bytes": h2d_bytes(row),
        "h2d_queue_ms": median(
            [float(value) for value in row.get("lmcache_h2d_wait_ms", [])]
        ),
    }


def replay_cost(row: dict[str, Any]) -> float | None:
    value = float(row.get("replay_compute_time_ms", 0) or 0)
    count = float(row.get("vllm_prefill_request_count", 0) or 0)
    return value / count if value > 0 and count > 0 else None


def _median_positive(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if float(value) > 0]
    return median(numbers)


def _model_features(row: dict[str, Any]) -> tuple[int, int, float] | None:
    """Return (missing pages, concurrency, missing MiB) for one valid row."""
    pages = actual_missing_pages(row)
    concurrency = row.get("concurrency")
    if not isinstance(pages, int) or not isinstance(concurrency, int):
        return None
    bytes_value = h2d_bytes(row)
    if bytes_value <= 0 and row.get("policy") == "P2":
        # P2 intentionally has no H2D.  Its load-side feature is not used.
        bytes_value = 0.0
    return pages, concurrency, bytes_value / 2**20


def _fit_affine(points: list[tuple[float, float]]) -> dict[str, float | int | None]:
    """Fit y = intercept + slope*x using only the Python standard library."""
    if not points:
        return {"samples": 0, "intercept_ms": None, "slope_ms_per_unit": None, "rmse_ms": None}
    if len(points) == 1:
        return {
            "samples": 1,
            "intercept_ms": points[0][1],
            "slope_ms_per_unit": 0.0,
            "rmse_ms": 0.0,
        }
    x_mean = statistics.mean(point[0] for point in points)
    y_mean = statistics.mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
        if denominator > 0
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    rmse = statistics.mean(
        [(y - (intercept + slope * x)) ** 2 for x, y in points]
    ) ** 0.5
    return {
        "samples": len(points),
        "intercept_ms": intercept,
        "slope_ms_per_unit": slope,
        "rmse_ms": rmse,
    }


def fit_cost_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit transparent load/replay costs and retain exact strata when present."""
    load_points: list[tuple[float, float]] = []
    replay_points: list[tuple[float, float]] = []
    strata: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"load": [], "replay": []}
    )
    for row in rows:
        features = _model_features(row)
        if features is None:
            continue
        pages, concurrency, mib = features
        stratum = f"K{pages}:C{concurrency}"
        if row.get("policy") in {"P1", "P5"}:
            cost = load_cost(row)["load_ms"]
            if cost is not None:
                load_points.append((mib, cost))
                strata[stratum]["load"].append(cost)
        elif row.get("policy") == "P2":
            cost = replay_cost(row)
            needs = _median_positive(row.get("lmcache_need_h2d_tokens", []))
            if cost is not None and needs is not None:
                replay_points.append((needs, cost))
                strata[stratum]["replay"].append(cost)
    return {
        "load_affine_bytes_mib": _fit_affine(load_points),
        "replay_affine_tokens": _fit_affine(replay_points),
        "strata": {
            key: {
                mode: {
                    "samples": len(values),
                    "median_ms": median(values),
                }
                for mode, values in modes.items()
                if values
            }
            for key, modes in sorted(strata.items())
        },
    }


def _predict_affine(
    fit: dict[str, float | int | None], value: float
) -> float | None:
    intercept = fit.get("intercept_ms")
    slope = fit.get("slope_ms_per_unit")
    if intercept is None or slope is None:
        return None
    return max(0.0, float(intercept) + float(slope) * value)


def predict_cost(
    model: dict[str, Any], row: dict[str, Any], policy: str
) -> float | None:
    features = _model_features(row)
    if features is None:
        return None
    pages, concurrency, mib = features
    stratum = model["strata"].get(f"K{pages}:C{concurrency}", {})
    mode = "replay" if policy == "P2" else "load"
    exact = stratum.get(mode, {}).get("median_ms")
    if exact is not None:
        return float(exact)
    if mode == "replay":
        needs = _median_positive(row.get("lmcache_need_h2d_tokens", []))
        return (
            _predict_affine(model["replay_affine_tokens"], needs)
            if needs is not None
            else None
        )
    return _predict_affine(model["load_affine_bytes_mib"], mib)


def group_type_bytes(row: dict[str, Any]) -> dict[str, float]:
    """Return transferred bytes by verified object-group type."""
    labels: dict[str, str] = {}
    for key in row.get("lmcache_h2d_retrieve_group_tokens", {}):
        match = re.fullmatch(r"group(\d+):(.*)", str(key))
        if match:
            labels[match.group(1)] = match.group(2)
    result = defaultdict(float)
    for group, value in row.get("lmcache_actual_h2d_object_group_bytes", {}).items():
        group_type = labels.get(str(group), "unknown")
        if "Mamba" in group_type:
            group_type = "MambaSpec"
        elif "FullAttention" in group_type:
            group_type = "FullAttentionSpec"
        result[group_type] += float(value)
    return dict(result)


def full_output_signature(row: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    outputs = row.get("completion_texts")
    if not isinstance(outputs, list) or not all(
        isinstance(item, list) and len(item) == 2 and all(
            isinstance(value, str) for value in item
        )
        for item in outputs
    ):
        return None
    return tuple((item[0], item[1]) for item in outputs)


def architecture_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the state-size consequence of the Hybrid group layout."""
    grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        policy = row.get("policy")
        pages = actual_missing_pages(row)
        group_bytes = group_type_bytes(row)
        requests = row.get("requests")
        if policy not in {"P1", "P5"} or pages is None or not group_bytes:
            continue
        if not isinstance(requests, int) or requests < 1:
            continue
        grouped[(policy, pages)].append(
            {group: value / requests / 2**20 for group, value in group_bytes.items()}
        )
    summary = []
    for (policy, pages), samples in sorted(grouped.items()):
        group_types = sorted({group for sample in samples for group in sample})
        values = {
            group: median([sample.get(group, 0.0) for sample in samples])
            for group in group_types
        }
        summary.append(
            {
                "policy": policy,
                "missing_pages": pages,
                "repetitions": len(samples),
                "mib_per_request": values,
                "total_mib_per_request": sum(values.values()),
            }
        )
    return summary


def summarize(
    rows: list[dict[str, Any]],
    rejected: dict[str, int],
    *,
    require_full_output: bool = False,
) -> dict[str, Any]:
    cost_model = fit_cost_model(rows)
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[cell_key(row)][row["policy"]].append(row)

    cells = []
    for key, policies in sorted(grouped.items(), key=lambda item: str(item[0])):
        if not {"P1", "P2"}.issubset(policies):
            continue
        page_values = {
            page
            for row in policies["P1"]
            for page in (actual_missing_pages(row), row.get("expected_missing_pages"))
            if isinstance(page, int) and page > 0
        }
        if len(page_values) != 1:
            continue
        missing_pages = page_values.pop()
        baseline_signatures = {
            row.get("first_token_signature")
            for policy in ("P1", "P2")
            for row in policies[policy]
        }
        if len(baseline_signatures) != 1:
            continue
        baseline_signature = baseline_signatures.pop()
        full_outputs = {
            policy: {
                full_output_signature(row)
                for row in policy_rows
            }
            for policy, policy_rows in policies.items()
        }
        full_output_policies = [
            policy
            for policy in ("P1", "P2", "P5", "Adaptive")
            if policy in full_outputs
        ]
        full_output_match = (
            len(full_output_policies) >= 2
            and all(
                len(full_outputs[policy]) == 1
                and None not in full_outputs[policy]
                for policy in full_output_policies
            )
            and len(
                {
                    next(iter(full_outputs[policy]))
                    for policy in full_output_policies
                }
            )
            == 1
        )
        if require_full_output and not full_output_match:
            continue
        policy_summary: dict[str, Any] = {}
        excluded_policies: dict[str, str] = {}
        for policy, policy_rows in policies.items():
            if policy not in ("P1", "P2"):
                signatures = {row.get("first_token_signature") for row in policy_rows}
                if signatures != {baseline_signature}:
                    excluded_policies[policy] = "first-token signature"
                    continue
            ttft = median([float(row["ttft_p50_ms"]) for row in policy_rows])
            cost_rows = [load_cost(row) for row in policy_rows]
            policy_summary[policy] = {
                "repetitions": len(policy_rows),
                "ttft_p50_ms": ttft,
                "ttft_p90_ms": median(
                    [
                        float(row["ttft_p90_ms"])
                        for row in policy_rows
                        if row.get("ttft_p90_ms") is not None
                    ]
                ),
                "ttft_p99_ms": median(
                    [
                        float(row["ttft_p99_ms"])
                        for row in policy_rows
                        if row.get("ttft_p99_ms") is not None
                    ]
                ),
                "h2d_bytes": median([row["h2d_bytes"] for row in cost_rows]),
                "load_cost_ms": median(
                    [row["load_ms"] for row in cost_rows if row["load_ms"] is not None]
                ),
                "replay_cost_ms": median(
                    [cost for row in policy_rows if (cost := replay_cost(row)) is not None]
                ),
                "h2d_queue_ms": median(
                    [row["h2d_queue_ms"] for row in cost_rows if row["h2d_queue_ms"] is not None]
                ),
                "model_predicted_cost_ms": median(
                    [
                        predicted
                        for row in policy_rows
                        if (predicted := predict_cost(cost_model, row, policy))
                        is not None
                    ]
                ),
                "adaptive_replay_count": sum(
                    len(row.get("lmcache_adaptive_replays", [])) for row in policy_rows
                ),
                "adaptive_cost_load_count": sum(
                    sum(
                        decision.get("decision") == "load"
                        for decision in row.get("lmcache_adaptive_decisions", [])
                    )
                    for row in policy_rows
                ),
                "adaptive_cost_replay_count": sum(
                    sum(
                        decision.get("decision") == "replay"
                        for decision in row.get("lmcache_adaptive_decisions", [])
                    )
                    for row in policy_rows
                ),
                "adaptive_predicted_load_ms": median(
                    [
                        float(decision["predicted_load_ms"])
                        for row in policy_rows
                        for decision in row.get("lmcache_adaptive_decisions", [])
                        if "predicted_load_ms" in decision
                    ]
                ),
                "adaptive_predicted_replay_ms": median(
                    [
                        float(decision["predicted_replay_ms"])
                        for row in policy_rows
                        for decision in row.get("lmcache_adaptive_decisions", [])
                        if "predicted_replay_ms" in decision
                    ]
                ),
            }
        oracle_candidates = {
            policy: summary["ttft_p50_ms"]
            for policy, summary in policy_summary.items()
            if policy in {"P1", "P2", "P5"}
        }
        oracle = min(oracle_candidates, key=oracle_candidates.__getitem__)
        oracle_ttft = oracle_candidates[oracle]
        tail_candidates = {
            policy: summary["ttft_p99_ms"]
            for policy, summary in policy_summary.items()
            if policy in {"P1", "P2", "P5"}
            and summary["ttft_p99_ms"] is not None
        }
        tail_oracle = (
            min(tail_candidates, key=tail_candidates.__getitem__)
            if tail_candidates
            else None
        )
        p1 = policy_summary["P1"]["ttft_p50_ms"]
        p2 = policy_summary["P2"]["ttft_p50_ms"]
        cost_candidates = {
            policy: summary["model_predicted_cost_ms"]
            for policy, summary in policy_summary.items()
            if policy in {"P1", "P5"}
            and summary["model_predicted_cost_ms"] is not None
        }
        if (
            policy_summary["P2"]["model_predicted_cost_ms"] is not None
        ):
            cost_candidates["P2"] = policy_summary["P2"][
                "model_predicted_cost_ms"
            ]
        cost_model_winner = (
            min(cost_candidates, key=cost_candidates.__getitem__)
            if cost_candidates
            else None
        )
        observed_cost_candidates = {
            policy: summary["load_cost_ms"]
            for policy, summary in policy_summary.items()
            if policy in {"P1", "P5"} and summary["load_cost_ms"] is not None
        }
        if policy_summary["P2"]["replay_cost_ms"] is not None:
            observed_cost_candidates["P2"] = policy_summary["P2"][
                "replay_cost_ms"
            ]
        cell = {
            "missing_pages": missing_pages,
            "concurrency": key[CELL_FIELDS.index("concurrency")],
            "suffix_tokens": key[CELL_FIELDS.index("suffix_tokens")],
            "oracle": oracle,
            "oracle_ttft_p50_ms": oracle_ttft,
            "tail_oracle": tail_oracle,
            "oracle_ttft_p99_ms": (
                tail_candidates[tail_oracle] if tail_oracle is not None else None
            ),
            "full_output_match": full_output_match,
            "p1_p2_margin_percent": abs(p1 - p2) / min(p1, p2) * 100,
            "cost_model_winner": cost_model_winner,
            "cost_model_matches_oracle": cost_model_winner == oracle,
            "observed_cost_winner": (
                min(observed_cost_candidates, key=observed_cost_candidates.__getitem__)
                if observed_cost_candidates
                else None
            ),
            "policies": policy_summary,
            "excluded_policies": excluded_policies,
        }
        adaptive = policy_summary.get("Adaptive")
        if adaptive is not None and adaptive["ttft_p50_ms"] is not None:
            adaptive["regret_percent"] = (
                adaptive["ttft_p50_ms"] - oracle_ttft
            ) / oracle_ttft * 100
        cells.append(cell)
    return {
        "valid_rows": len(rows),
        "rejected_rows": dict(rejected),
        "cells": cells,
        "architecture": architecture_summary(rows),
        "cost_model": cost_model,
    }


def self_check() -> None:
    base = {
        "backend": "lmcache",
        "independent_process": True,
        "suffix_only": True,
        "retain_shared_prefix": True,
        "shared_prefix_tokens": 528,
        "requests": 1,
        "policy": "P1",
        "lmcache_missing_pages": [2],
        "lmcache_prefix_matches": [
            {"local_gpu_tokens": 528, "cpu_tokens": 1584, "need_h2d_tokens": 1056}
        ],
        "h2d_cpu_to_gpu_bytes": 1,
        "first_token_signature": "same",
        "ttft_p50_ms": 2,
    }
    assert valid_row(base)[0]
    replay = dict(base, policy="P2", h2d_cpu_to_gpu_bytes=1)
    assert not valid_row(replay)[0]
    replay["h2d_cpu_to_gpu_bytes"] = 0
    replay["lmcache_h2d_retrieve_requests"] = 0
    assert valid_row(replay)[0]
    p5_mismatch = dict(base, policy="P5", first_token_signature="different")
    summary = summarize([base, replay, p5_mismatch], {})
    assert len(summary["cells"]) == 1
    assert summary["cells"][0]["excluded_policies"] == {
        "P5": "first-token signature"
    }
    assert not summary["cells"][0]["full_output_match"]
    layout = dict(
        base,
        requests=1,
        lmcache_actual_h2d_object_group_bytes={
            "0": 2**20,
            "1": 2**20,
            "2": 2**20,
            "3": 2**20,
        },
        lmcache_h2d_retrieve_group_tokens={
            "group0:MambaSpec": 1,
            "group1:MambaSpec": 1,
            "group2:MambaSpec": 1,
            "group3:FullAttentionSpec": 1,
        },
    )
    layout_p5 = dict(
        layout,
        policy="P5",
        lmcache_actual_h2d_object_group_bytes={
            "0": 2**20,
            "1": 2**20,
            "2": 2**20,
            "3": 2 * 2**20,
        },
    )
    architecture = architecture_summary([layout, layout_p5])
    assert architecture[0]["mib_per_request"]["MambaSpec"] == 3.0
    assert architecture[1]["mib_per_request"]["MambaSpec"] == 3.0
    assert architecture[1]["mib_per_request"]["FullAttentionSpec"] == 2.0
    complete = dict(base, completion_texts=[["request-0", "same output"]])
    complete_replay = dict(
        replay, completion_texts=[["request-0", "same output"]]
    )
    complete_p5 = dict(
        complete, policy="P5", first_token_signature="same"
    )
    complete_summary = summarize(
        [complete, complete_replay, complete_p5], {}, require_full_output=True
    )
    assert len(complete_summary["cells"]) == 1
    assert complete_summary["cells"][0]["full_output_match"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Runner JSONL files")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Print transferred bytes by Hybrid object-group type",
    )
    parser.add_argument(
        "--require-full-output",
        action="store_true",
        help="Only keep cells with complete P1/P2(/P5) output equality",
    )
    parser.add_argument("--self-check", action="store_true", help="Run the built-in checks")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        if not args.inputs:
            return
    if not args.inputs:
        parser.error("inputs are required unless --self-check is used alone")
    rows, rejected = load_rows(args.inputs)
    result = summarize(rows, rejected, require_full_output=args.require_full_output)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"valid rows: {result['valid_rows']}; rejected: {result['rejected_rows']}")
    for cell in result["cells"]:
        policies = cell["policies"]
        line = (
            f"K={cell['missing_pages']} C={cell['concurrency']} "
            f"oracle={cell['oracle']} "
            f"p99-oracle={cell['tail_oracle']} "
            f"P1/P2-margin={cell['p1_p2_margin_percent']:.1f}%"
        )
        for policy in POLICIES:
            if policy in policies:
                value = policies[policy]["ttft_p50_ms"]
                line += f" {policy}={value:.3f}ms"
        print(line)
    if args.architecture:
        print("architecture bytes (MiB/request):")
        for item in result["architecture"]:
            values = " ".join(
                f"{group}={value:.2f}"
                for group, value in item["mib_per_request"].items()
            )
            print(
                f"{item['policy']} K={item['missing_pages']} {values} "
                f"total={item['total_mib_per_request']:.2f} "
                f"n={item['repetitions']}"
            )


if __name__ == "__main__":
    main()
