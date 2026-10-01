# SPDX-License-Identifier: Apache-2.0
"""CPU-only Marconi trace replay on real ShareGPT multi-turn sessions.

This measures cache-management behavior only.  It does not run a model and
must not be interpreted as TTFT or recovery-scheduler evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tokenizers import Tokenizer

try:
    from benchmarks.reproductions.sharegpt_data import iter_records
except ImportError:  # pragma: no cover
    from sharegpt_data import iter_records


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path("/root/dataset/ShareGPT_V3_unfiltered_cleaned_split.json")
DEFAULT_TOKENIZER = Path("/root/models/Qwen3.5-9B/tokenizer.json")


def load_index_class():
    path = ROOT / "vllm/v1/kv_offload/policies/marconi_index.py"
    spec = importlib.util.spec_from_file_location("marconi_index_trace", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Marconi index from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MarconiIndex


@dataclass(frozen=True)
class TraceEvent:
    session_id: str
    turn: int
    token_ids: tuple[int, ...]


def session_events(record: dict[str, Any], tokenizer: Tokenizer) -> tuple[TraceEvent, ...]:
    """Build exact cumulative conversation prefixes from human/gpt pairs."""
    conversation = record.get("conversations")
    if not isinstance(conversation, list):
        return ()
    history = ""
    events: list[TraceEvent] = []
    turn = 0
    index = 0
    while index + 1 < len(conversation):
        human = conversation[index]
        answer = conversation[index + 1]
        if not isinstance(human, dict) or not isinstance(answer, dict):
            index += 1
            continue
        if human.get("from") != "human" or answer.get("from") != "gpt":
            index += 1
            continue
        user_text = str(human.get("value", ""))
        answer_text = str(answer.get("value", ""))
        history += f"User: {user_text}\nAssistant: {answer_text}\n"
        tokens = tuple(tokenizer.encode(history).ids)
        if not tokens:
            index += 2
            continue
        if events and tokens[: len(events[-1].token_ids)] != events[-1].token_ids:
            # BPE boundary instability would invalidate prefix-cache semantics.
            break
        events.append(TraceEvent(str(record.get("id", "unknown")), turn, tokens))
        turn += 1
        index += 2
    return tuple(events)


def collect_events(
    dataset: Path,
    tokenizer: Tokenizer,
    sessions: int,
    max_turns: int,
    concurrency: int,
) -> tuple[TraceEvent, ...]:
    if concurrency < 1 or concurrency > sessions:
        raise ValueError("concurrency must be in [1, sessions]")
    selected: list[tuple[TraceEvent, ...]] = []
    for record in iter_records(dataset):
        events = session_events(record, tokenizer)
        if len(events) >= 2:
            selected.append(events[:max_turns])
            if len(selected) >= sessions:
                break
    if len(selected) < sessions:
        raise RuntimeError(
            f"requested {sessions} multi-turn sessions, found {len(selected)}"
        )
    # A concurrency cap changes temporal interleaving, not the CPU capacity.
    # Each wave contains at most ``concurrency`` active sessions.
    result: list[TraceEvent] = []
    for wave_start in range(0, len(selected), concurrency):
        wave = selected[wave_start : wave_start + concurrency]
        for turn in range(max_turns):
            result.extend(events[turn] for events in wave if turn < len(events))
    return tuple(result)


def replay(
    events: tuple[TraceEvent, ...],
    index_cls: Any,
    capacity_bytes: int,
    bytes_per_token: int,
    alpha: float,
) -> dict[str, Any]:
    index = index_cls()
    total_tokens = 0
    hit_tokens = 0
    hit_requests = 0
    exact_hits = 0
    matched_lengths: list[int] = []
    evictions = 0
    evicted_bytes = 0
    max_resident_bytes = 0
    for now, event in enumerate(events, start=1):
        tokens = tuple(str(token) for token in event.token_ids)
        total_tokens += len(tokens)
        matched = index.lookup(tokens)
        matched_count = len(matched)
        matched_lengths.append(matched_count)
        hit_tokens += matched_count
        if matched_count:
            hit_requests += 1
            index.touch(tokens, float(now))
        if matched_count == len(tokens):
            exact_hits += 1
        else:
            new_tokens = max(1, len(tokens) - matched_count)
            index.insert(
                tokens,
                state_kind="hybrid",
                token_count=len(tokens),
                byte_size=new_tokens * bytes_per_token,
                last_access_ms=float(now),
                compute_savings_ms=matched_count * 0.01,
            )
        overflow = index.materialized_bytes - capacity_bytes
        if overflow > 0:
            victims = index.select_evictions(
                overflow, float(now), alpha=alpha
            )
            for victim in victims:
                node = index.get(victim)
                if node is not None:
                    evicted_bytes += node.byte_size
                index.remove(victim)
                evictions += 1
        max_resident_bytes = max(max_resident_bytes, index.materialized_bytes)
    request_count = len(events)
    return {
        "events": request_count,
        "sessions": len({event.session_id for event in events}),
        "total_tokens": total_tokens,
        "hit_requests": hit_requests,
        "request_hit_rate": hit_requests / request_count if request_count else 0.0,
        "nontrivial_hit_requests": sum(length >= 16 for length in matched_lengths),
        "nontrivial_hit_rate": (
            sum(length >= 16 for length in matched_lengths) / request_count
            if request_count
            else 0.0
        ),
        "token_hit_rate": hit_tokens / total_tokens if total_tokens else 0.0,
        "mean_hit_tokens": statistics.fmean(matched_lengths) if matched_lengths else 0.0,
        "p50_hit_tokens": (
            statistics.median(matched_lengths) if matched_lengths else 0.0
        ),
        "max_hit_tokens": max(matched_lengths, default=0),
        "exact_hits": exact_hits,
        "evictions": evictions,
        "evicted_bytes": evicted_bytes,
        "resident_bytes": index.materialized_bytes,
        "resident_nodes": max(0, len(index.nodes()) - 1),
        "max_resident_bytes": max_resident_bytes,
        "capacity_bytes": capacity_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--sessions", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--concurrencies", default="1,4,8,16")
    parser.add_argument("--capacity-mib", type=float, nargs="+", default=[512.0])
    parser.add_argument("--bytes-per-token", type=int, default=131072)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("/root/marconi_sharegpt_trace.jsonl")
    )
    args = parser.parse_args()
    if args.sessions < 1 or args.max_turns < 2:
        raise ValueError("sessions must be positive and max-turns must be at least 2")
    if args.bytes_per_token < 1 or args.alpha < 0:
        raise ValueError("bytes-per-token must be positive and alpha non-negative")
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if not concurrencies or min(concurrencies) < 1 or max(concurrencies) > args.sessions:
        raise ValueError("concurrencies must be positive and no larger than sessions")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    index_cls = load_index_class()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for concurrency in concurrencies:
            events = collect_events(
                args.dataset,
                tokenizer,
                args.sessions,
                args.max_turns,
                concurrency,
            )
            for capacity_mib in args.capacity_mib:
                if capacity_mib <= 0:
                    raise ValueError("capacity-mib must be positive")
                row = {
                    "workload": "sharegpt_multi_user_multiturn",
                    "baseline": "marconi",
                    "trace_type": "real_sharegpt",
                    "max_turns": args.max_turns,
                    "concurrency": concurrency,
                    "bytes_per_token": args.bytes_per_token,
                    "capacity_mib": capacity_mib,
                    **replay(
                        events,
                        index_cls,
                        int(capacity_mib * 1024**2),
                        args.bytes_per_token,
                        args.alpha,
                    ),
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
