# SPDX-License-Identifier: Apache-2.0
"""Derive a formal recovery-cost artifact from completed native observations.

This intentionally consumes a prior diagnostic/native run instead of inventing
rates. Only decisions whose scheduler feedback was marked ``measured`` are
eligible, so configured cold-start constants cannot enter a formal matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any


MARKER = "HYREX_NATIVE_DECISION "


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        _, marker, payload = line.partition(MARKER)
        if not marker:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed HyRex decision in {path}") from error
        if not isinstance(event, dict):
            raise ValueError("HyRex decision must be a JSON object")
        events.append(event)
    return events


def derive(events: list[dict[str, Any]]) -> dict[str, float]:
    h2d = [
        float(event["h2d_gbps"])
        for event in events
        if event.get("h2d_gbps_source") == "measured"
        and float(event.get("h2d_gbps", 0)) > 0
    ]
    replay = [
        event for event in events if event.get("replay_cost_source") == "measured"
    ]
    full = [float(event["full_replay_ms_per_token"]) for event in replay]
    recurrent = [float(event["recurrent_replay_ms_per_token"]) for event in replay]
    if not h2d or not full or not recurrent:
        raise ValueError(
            "need measured H2D plus measured Full and recurrent replay decisions"
        )
    if min(*full, *recurrent) < 0:
        raise ValueError("measured replay rates must be non-negative")
    return {
        "h2d_gbps": statistics.median(h2d),
        "full_replay_ms_per_token": statistics.median(full),
        "recurrent_replay_ms_per_token": statistics.median(recurrent),
    }


def build_record(log: Path, model_name: str, source_result: Path | None) -> dict[str, Any]:
    rates = derive(decision_events(log))
    provenance: dict[str, Any] = {
        "kind": "native_scheduler_feedback",
        "server_log": str(log.resolve()),
        "server_log_sha256": sha256(log),
    }
    if source_result is not None:
        provenance["result"] = str(source_result.resolve())
        provenance["result_sha256"] = sha256(source_result)
    return {
        "schema_version": 1,
        "model_name": model_name,
        **rates,
        "provenance": provenance,
    }


def self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "server.log"
        log.write_text(
            'INFO HYREX_NATIVE_DECISION {"h2d_gbps":10.0,'
            '"h2d_gbps_source":"measured","full_replay_ms_per_token":0.2,'
            '"recurrent_replay_ms_per_token":0.4,"replay_cost_source":"measured"}\n'
            'INFO HYREX_NATIVE_DECISION {"h2d_gbps":20.0,'
            '"h2d_gbps_source":"measured","full_replay_ms_per_token":0.4,'
            '"recurrent_replay_ms_per_token":0.8,"replay_cost_source":"measured"}\n'
        )
        record = build_record(log, "model", None)
        assert record["h2d_gbps"] == 15.0
        assert abs(record["full_replay_ms_per_token"] - 0.3) < 1e-12
        assert abs(record["recurrent_replay_ms_per_token"] - 0.6) < 1e-12


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen3.5-9B")
    parser.add_argument("--source-result", type=Path, default=None)
    args = parser.parse_args()
    if not args.server_log.is_file() or (
        args.source_result is not None and not args.source_result.is_file()
    ):
        raise ValueError("server log or source result does not exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        build_record(args.server_log, args.model_name, args.source_result), indent=2
    ) + "\n")


if __name__ == "__main__":
    main()
