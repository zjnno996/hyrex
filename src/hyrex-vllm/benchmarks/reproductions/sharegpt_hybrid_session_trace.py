# SPDX-License-Identifier: Apache-2.0
"""Create a reproducible multi-session resume trace from real ShareGPT turns.

Each event contains a cached conversation history and its next real user turn.
The token check makes the history an exact prefix of the resume prompt, which
is the cache-recovery precondition required by vLLM/LMCache experiments.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    from sharegpt_data import iter_records
except ImportError:  # Imported as benchmarks.reproductions.*
    from benchmarks.reproductions.sharegpt_data import iter_records


ROLES = {
    "human": "User",
    "user": "User",
    "gpt": "Assistant",
    "chatgpt": "Assistant",
    "assistant": "Assistant",
    "bard": "Assistant",
    "bing": "Assistant",
}


def _turns(record: dict[str, Any]) -> list[str]:
    return [
        f"{ROLES[turn['from']]}: {turn['value']}"
        for turn in record.get("conversations", [])
        if turn.get("from") in ROLES
        and isinstance(turn.get("value"), str)
        and turn["value"].strip()
    ]


def _events(
    record_index: int,
    record: dict[str, Any],
    tokenizer: Any,
    *,
    min_history_tokens: int,
    max_history_tokens: int,
    min_residual_tokens: int,
    max_residual_tokens: int,
    local_prefix_tokens: int,
) -> list[dict[str, Any]]:
    turns = _turns(record)
    events: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not turn.startswith("User:") or index < 2:
            continue
        history = "\n\n".join(turns[:index])
        resume = f"{history}\n\n{turn}"
        history_ids = tokenizer.encode(history, add_special_tokens=False)
        resume_ids = tokenizer.encode(resume, add_special_tokens=False)
        residual_tokens = len(resume_ids) - len(history_ids)
        if (
            not min_history_tokens <= len(history_ids) <= max_history_tokens
            or not min_residual_tokens <= residual_tokens <= max_residual_tokens
            or resume_ids[: len(history_ids)] != history_ids
        ):
            continue
        local_ids = history_ids[:local_prefix_tokens]
        local_prompt = tokenizer.decode(local_ids, skip_special_tokens=False)
        if tokenizer.encode(local_prompt, add_special_tokens=False) != local_ids:
            continue
        events.append({
            "record_index": record_index,
            "session_id": str(record.get("id", record_index)),
            "cache_prompt": history,
            "local_prompt": local_prompt,
            "resume_prompt": resume,
            "cached_tokens": len(history_ids),
            "local_prefix_tokens": len(local_ids),
            "resume_tokens": len(resume_ids),
            "residual_tokens": residual_tokens,
            "turn_index": len(events),
        })
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-history-tokens", type=int, default=528)
    parser.add_argument("--max-history-tokens", type=int, default=4096)
    parser.add_argument("--min-residual-tokens", type=int, default=16)
    parser.add_argument("--max-residual-tokens", type=int, default=1056)
    parser.add_argument("--local-prefix-tokens", type=int, default=528)
    args = parser.parse_args()
    if (
        args.sessions < 1
        or args.max_turns < 1
        or args.min_history_tokens < 1
        or args.max_history_tokens < args.min_history_tokens
        or args.min_residual_tokens < 1
        or args.max_residual_tokens < args.min_residual_tokens
        or args.local_prefix_tokens < 1
    ):
        raise ValueError("invalid trace bounds")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    sessions: list[list[dict[str, Any]]] = []
    for record_index, record in enumerate(iter_records(args.dataset_path)):
        events = _events(
            record_index,
            record,
            tokenizer,
            min_history_tokens=args.min_history_tokens,
            max_history_tokens=args.max_history_tokens,
            min_residual_tokens=args.min_residual_tokens,
            max_residual_tokens=args.max_residual_tokens,
            local_prefix_tokens=args.local_prefix_tokens,
        )
        if events:
            sessions.append(events[: args.max_turns])
        if len(sessions) == args.sessions:
            break
    if len(sessions) != args.sessions:
        raise ValueError(f"found only {len(sessions)} eligible ShareGPT sessions")

    # Sample a random linear extension of the per-session turn order. This is
    # a realistic multi-user trace while keeping every baseline reproducible.
    rng = random.Random(args.seed)
    cursors = [0] * len(sessions)
    schedule: list[dict[str, Any]] = []
    while ready := [i for i, events in enumerate(sessions) if cursors[i] < len(events)]:
        session_index = rng.choice(ready)
        event = sessions[session_index][cursors[session_index]]
        cursors[session_index] += 1
        schedule.append(event)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for arrival_index, event in enumerate(schedule):
            output.write(
                json.dumps(
                    {**event, "arrival_index": arrival_index}, ensure_ascii=False
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "sessions": len(sessions),
                "max_turns": args.max_turns,
                "events": len(schedule),
                "seed": args.seed,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
