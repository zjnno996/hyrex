# SPDX-License-Identifier: Apache-2.0
"""Replay one prompt field from a ShareGPT Hybrid session trace against vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

try:
    from hybrid_baseline_config import BASELINES, baseline_config
except ImportError:
    from benchmarks.reproductions.hybrid_baseline_config import BASELINES, baseline_config


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def arrival_offsets(
    count: int,
    model: str,
    rate: float,
    seed: int,
    burst_size: int,
    zipf_exponent: float = 1.2,
    zipf_max_rank: int = 64,
) -> list[float]:
    """Return deterministic request release times in seconds."""
    if (
        count < 1
        or rate <= 0
        or burst_size < 1
        or zipf_exponent <= 0
        or zipf_max_rank < 1
    ):
        raise ValueError("arrival parameters must be positive")
    if model == "saturated":
        return [0.0] * count
    rng = random.Random(seed)
    offsets = [0.0]
    for index in range(1, count):
        if model == "uniform":
            gap = 1.0 / rate
        elif model == "poisson":
            gap = rng.expovariate(rate)
        elif model == "bursty":
            gap = burst_size / rate if index % burst_size == 0 else 0.0
        elif model == "zipf":
            # Sample a bounded discrete Zipf rank, then normalize it so the
            # mean gap remains 1/rate. This changes burstiness without silently
            # changing the offered average request rate across methods.
            ranks = range(1, zipf_max_rank + 1)
            weights = [rank ** (-zipf_exponent) for rank in ranks]
            mean_rank = sum(rank * weight for rank, weight in zip(ranks, weights)) / sum(weights)
            gap = rng.choices(tuple(ranks), weights=weights, k=1)[0] / mean_rank / rate
        else:
            raise ValueError(f"unknown arrival model: {model}")
        offsets.append(offsets[-1] + gap)
    return offsets


def validate_online_rows(rows: list[dict[str, Any]]) -> None:
    """Reject traces that can violate per-session conversation order."""
    last_turn: dict[str, int] = {}
    for expected_arrival, row in enumerate(rows):
        if row.get("arrival_index", expected_arrival) != expected_arrival:
            raise ValueError("trace arrival_index must be contiguous and ordered")
        session_id = str(row["session_id"])
        turn_index = int(row["turn_index"])
        if turn_index != last_turn.get(session_id, -1) + 1:
            raise ValueError(f"out-of-order turn for session {session_id}")
        last_turn[session_id] = turn_index


def validate_output_hashes(
    rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> None:
    references = {
        int(row["arrival_index"]): row["output_sha256"] for row in reference_rows
    }
    mismatches = [
        row["arrival_index"]
        for row in rows
        if references.get(row["arrival_index"]) != row["output_sha256"]
    ]
    if mismatches:
        raise RuntimeError(
            f"output mismatch for {len(mismatches)} requests: {mismatches[:3]}"
        )


def self_check() -> None:
    assert arrival_offsets(3, "uniform", 2.0, 0, 2) == [0.0, 0.5, 1.0]
    assert arrival_offsets(5, "bursty", 2.0, 0, 2) == [0.0, 0.0, 1.0, 1.0, 2.0]
    zipf = arrival_offsets(5, "zipf", 2.0, 0, 2, zipf_max_rank=8)
    assert zipf[0] == 0.0 and all(gap > 0 for gap in zipf[1:])
    validate_online_rows(
        [
            {"arrival_index": 0, "session_id": "a", "turn_index": 0},
            {"arrival_index": 1, "session_id": "b", "turn_index": 0},
            {"arrival_index": 2, "session_id": "a", "turn_index": 1},
        ]
    )
    validate_output_hashes(
        [{"arrival_index": 0, "output_sha256": "same"}],
        [{"arrival_index": 0, "output_sha256": "same"}],
    )


def request_metrics(
    server: str,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    kv_transfer_params: dict[str, object],
    request_id: str,
    request_timeout: float,
) -> tuple[float, list[float], str]:
    """Return latency samples and a deterministic output-content digest."""
    session = requests.Session()
    session.trust_env = False
    started = time.perf_counter()
    response = session.post(
        f"{server}/v1/completions",
        headers={"X-Request-Id": request_id},
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
            # Benchmarks need a fixed token count so an immediate EOS does not
            # erase TTFT/TPOT samples for only a subset of policies.
            "ignore_eos": True,
            "stream": True,
            # This is a first-class vLLM OpenAI field.  It reaches
            # Request.kv_transfer_params and thus the native connector.
            "kv_transfer_params": kv_transfer_params,
        },
        stream=True,
        timeout=request_timeout,
    )
    response.raise_for_status()
    token_times: list[float] = []
    output_parts: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        choices = event.get("choices", [])
        if choices and choices[0].get("text"):
            output_parts.append(choices[0]["text"])
            token_times.append((time.perf_counter() - started) * 1000)
    if not token_times:
        raise RuntimeError("stream completed without an output token")
    return (
        token_times[0],
        [later - earlier for earlier, later in zip(token_times, token_times[1:])],
        hashlib.sha256("".join(output_parts).encode()).hexdigest(),
    )


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--prompt-field", choices=("cache_prompt", "local_prompt", "resume_prompt"),
        required=True,
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", choices=tuple(BASELINES), required=True)
    parser.add_argument(
        "--kv-transfer-params-json",
        default="{}",
        help="JSON merged into the baseline's connector request metadata.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--execution-mode",
        choices=("online", "preconditioned"),
        default="online",
        help="Online replays one continuous trace; preconditioned retains the old warmup.",
    )
    parser.add_argument(
        "--arrival-model",
        choices=("saturated", "uniform", "poisson", "bursty", "zipf"),
        default="saturated",
    )
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--zipf-exponent", type=float, default=1.2)
    parser.add_argument("--zipf-max-rank", type=int, default=64)
    parser.add_argument("--request-output", type=Path, default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--correctness-reference",
        type=Path,
        default=None,
        help="Request JSONL whose arrival_index/output_sha256 must match.",
    )
    parser.add_argument(
        "--warmup-field",
        choices=("cache_prompt", "local_prompt"),
        default="cache_prompt",
        help="Send each session prefix before the measured resume request.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.max_tokens < 2
        or args.request_rate <= 0
        or args.burst_size < 1
        or args.zipf_exponent <= 0
        or args.zipf_max_rank < 1
        or args.request_timeout <= 0
        or args.limit is not None
        and args.limit < 1
    ):
        raise ValueError("concurrency and limit must be positive")

    try:
        supplied_params = json.loads(args.kv_transfer_params_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--kv-transfer-params-json must be valid JSON") from exc
    if not isinstance(supplied_params, dict):
        raise ValueError("--kv-transfer-params-json must contain an object")
    baseline = BASELINES[args.baseline]
    kv_transfer_params: dict[str, object] = {
        "hybrid_baseline": baseline.name,
        "hyrex_observe_cache": True,
        **supplied_params,
    }
    if baseline.recovery_policy is not None:
        kv_transfer_params.setdefault("hyrex_recovery_policy", baseline.recovery_policy)

    rows: list[dict[str, Any]] = [
        json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.execution_mode == "online":
        validate_online_rows(rows)
    prompts = [row[args.prompt_field] for row in rows]
    if not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"trace has no string {args.prompt_field!r} prompts")

    if args.execution_mode == "preconditioned":
        for row in rows:
            request_metrics(
                args.server,
                args.model,
                row[args.warmup_field],
                args.seed,
                2,
                kv_transfer_params,
                f"hyrex-warmup-{row['session_id']}-{row['turn_index']}",
                args.request_timeout,
            )

    ttft: list[float] = []
    tpot: list[float] = []
    request_rows: list[dict[str, Any]] = []
    offsets = arrival_offsets(
        len(rows), args.arrival_model, args.request_rate, args.seed, args.burst_size,
        args.zipf_exponent, args.zipf_max_rank,
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(prompts))) as pool:
        futures = {}
        last_session_future = {}
        for row, prompt, offset in zip(rows, prompts, offsets, strict=True):
            previous = last_session_future.get(str(row["session_id"]))
            if previous is not None:
                previous.result()
            delay = started + offset - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            future = pool.submit(
                request_metrics,
                args.server,
                args.model,
                prompt,
                args.seed,
                args.max_tokens,
                kv_transfer_params,
                f"hyrex-{row['arrival_index']}",
                args.request_timeout,
            )
            futures[future] = (row, offset, (time.perf_counter() - started))
            last_session_future[str(row["session_id"])] = future
        for future in as_completed(futures):
            first, intervals, output_sha256 = future.result()
            ttft.append(first)
            tpot.extend(intervals)
            row, scheduled, released = futures[future]
            request_rows.append(
                {
                    "arrival_index": row.get("arrival_index"),
                    "request_id": f"cmpl-hyrex-{row['arrival_index']}-0",
                    "session_id": row.get("session_id"),
                    "turn_index": row.get("turn_index"),
                    "scheduled_arrival_s": scheduled,
                    "actual_release_s": released,
                    "prompt_tokens": row.get("resume_tokens"),
                    "ttft_ms": first,
                    "tpot_p50_ms": (
                        statistics.median(intervals) if intervals else None
                    ),
                    "output_sha256": output_sha256,
                }
            )
    if not tpot:
        raise RuntimeError("no inter-token intervals; increase --max-tokens")
    correctness_checked = args.correctness_reference is not None
    if args.correctness_reference is not None:
        validate_output_hashes(
            request_rows,
            [
                json.loads(line)
                for line in args.correctness_reference.read_text().splitlines()
            ],
        )
    if args.request_output is not None:
        args.request_output.parent.mkdir(parents=True, exist_ok=True)
        with args.request_output.open("w", encoding="utf-8") as output:
            for row in sorted(request_rows, key=lambda item: item["arrival_index"]):
                output.write(json.dumps(row) + "\n")
    print(
        json.dumps(
            {
                "prompt_field": args.prompt_field,
                "baseline": baseline_config(args.baseline),
                "kv_transfer_params": kv_transfer_params,
                "requests": len(ttft),
                "concurrency": args.concurrency,
                "execution_mode": args.execution_mode,
                "arrival_model": args.arrival_model,
                "request_rate": args.request_rate,
                "burst_size": args.burst_size,
                "zipf_exponent": args.zipf_exponent,
                "zipf_max_rank": args.zipf_max_rank,
                "request_timeout": args.request_timeout,
                "request_output": str(args.request_output) if args.request_output else None,
                "warmup_field": (
                    args.warmup_field if args.execution_mode == "preconditioned" else None
                ),
                "max_tokens": args.max_tokens,
                "correctness_checked": correctness_checked,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "ttft_p50_ms": statistics.median(ttft),
                "ttft_p95_ms": percentile(ttft, 0.95),
                "ttft_p99_ms": percentile(ttft, 0.99),
                "tpot_p50_ms": statistics.median(tpot),
                "tpot_p95_ms": percentile(tpot, 0.95),
                "tpot_p99_ms": percentile(tpot, 0.99),
                "output_token_throughput": len(tpot) / ((time.perf_counter() - started)),
            }
        )
    )


if __name__ == "__main__":
    main()
