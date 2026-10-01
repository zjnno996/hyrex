# SPDX-License-Identifier: Apache-2.0
"""Run a bounded native feedback pass and emit its measured calibration record.

The pass uses the normal continuous multi-turn lifecycle. It is a diagnostic
workload only: its output JSONL is retained as provenance and is never added to
the formal TTFT/TPOT matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from derive_online_calibration import build_record
from run_native_independent_cells import MODEL


ROOT = Path(__file__).resolve().parents[2]
CELL = ROOT / "benchmarks/reproductions/run_online_session_cell.py"


def resumed_sessions(trace: Path, limit: int) -> int:
    rows = [json.loads(line) for line in trace.read_text().splitlines()[:limit] if line]
    return len({str(row["session_id"]) for row in rows if int(row["turn_index"]) > 0})


def command(args: argparse.Namespace, result: Path, requests: Path) -> list[str]:
    return [
        sys.executable, str(CELL),
        "--trace", str(args.trace),
        "--baseline", "hyrex",
        "--model-path", str(args.model_path),
        "--cuda-visible-devices", args.cuda_visible_devices,
        "--concurrency", str(args.concurrency),
        "--arrival-model", args.arrival_model,
        "--request-rate", str(args.request_rate),
        "--burst-size", str(args.burst_size),
        "--zipf-exponent", str(args.zipf_exponent),
        "--zipf-max-rank", str(args.zipf_max_rank),
        "--max-tokens", str(args.max_tokens),
        "--limit", str(args.limit),
        "--cpu-cache-gb", str(args.cpu_cache_gb),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--request-timeout", str(args.request_timeout),
        "--execution-timeout", str(args.execution_timeout),
        "--server-port", str(args.server_port),
        "--hyrex-h2d-gbps", str(args.bootstrap_h2d_gbps),
        "--hyrex-full-replay-ms-per-token", str(args.bootstrap_full_replay_ms_per_token),
        "--hyrex-recurrent-replay-ms-per-token", str(args.bootstrap_recurrent_replay_ms_per_token),
        "--output", str(result),
        "--request-output", str(requests),
    ]


def self_check() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.jsonl"
        trace.write_text("\n".join(json.dumps({
            "session_id": "a", "turn_index": index,
        }) for index in (0, 1)) + "\n")
        assert resumed_sessions(trace, 2) == 1
    args = argparse.Namespace(
        trace=Path("/trace"), model_path=Path("/model"), cuda_visible_devices="0",
        concurrency=1, arrival_model="poisson", request_rate=8.0, burst_size=8, max_tokens=2,
        zipf_exponent=1.2, zipf_max_rank=64,
        limit=512, cpu_cache_gb=24.0, gpu_memory_utilization=0.9,
        request_timeout=600.0, execution_timeout=1800.0, server_port=8012,
        bootstrap_h2d_gbps=19.0, bootstrap_full_replay_ms_per_token=0.0021,
        bootstrap_recurrent_replay_ms_per_token=0.0052,
    )
    assert command(args, Path("/result"), Path("/requests"))[5] == "hyrex"


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--arrival-model", default="poisson")
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--zipf-exponent", type=float, default=1.2)
    parser.add_argument("--zipf-max-rank", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--min-resumed-sessions", type=int, default=100)
    parser.add_argument("--cpu-cache-gb", type=float, default=24.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--execution-timeout", type=float, default=3600.0)
    parser.add_argument("--server-port", type=int, default=8012)
    parser.add_argument("--bootstrap-h2d-gbps", type=float, default=19.0)
    parser.add_argument("--bootstrap-full-replay-ms-per-token", type=float, default=0.0021)
    parser.add_argument("--bootstrap-recurrent-replay-ms-per-token", type=float, default=0.0052)
    args = parser.parse_args()
    if not args.trace.is_file() or not args.model_path.is_dir():
        raise ValueError("trace or model path does not exist")
    if resumed_sessions(args.trace, args.limit) < args.min_resumed_sessions:
        raise ValueError("calibration trace has too few natural resumed sessions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = args.output_dir / "native_feedback_result.jsonl"
    requests = args.output_dir / "native_feedback_requests.jsonl"
    subprocess.run(command(args, result, requests), cwd=ROOT, check=True)
    row = json.loads(result.read_text().splitlines()[-1])
    log = Path(row["server_log"])
    artifact = build_record(log, args.model_path.name, result)
    output = args.output_dir / "recovery_calibration.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"result": str(result), "calibration": str(output)}))


if __name__ == "__main__":
    main()
