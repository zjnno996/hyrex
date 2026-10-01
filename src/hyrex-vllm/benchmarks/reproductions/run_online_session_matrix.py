# SPDX-License-Identifier: Apache-2.0
"""Run a resumable, paired Marconi/HyRex online concurrency matrix."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from aggregate_online_session_pairs import aggregate_pairs
from analyze_online_session_pair import analyze_pair
from hybrid_baseline_config import BASELINES
from run_native_independent_cells import MODEL
from run_online_session_cell import cpu_cache_budget, file_sha256, source_provenance


ROOT = Path(__file__).resolve().parents[2]
CELL_RUNNER = ROOT / "benchmarks/reproductions/run_online_session_cell.py"
CALIBRATION_FIELDS = (
    "h2d_gbps",
    "full_replay_ms_per_token",
    "recurrent_replay_ms_per_token",
)


def load_calibration(path: Path) -> dict[str, Any]:
    """Load a measured recovery-cost record used by a formal matrix."""
    try:
        calibration = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid calibration record: {path}") from error
    if calibration.get("schema_version") != 1:
        raise ValueError("calibration schema_version must be 1")
    if not calibration.get("provenance"):
        raise ValueError("calibration record needs non-empty provenance")
    missing = [field for field in CALIBRATION_FIELDS if field not in calibration]
    if missing:
        raise ValueError(f"calibration record is missing: {missing}")
    if (
        float(calibration["h2d_gbps"]) <= 0
        or float(calibration["full_replay_ms_per_token"]) < 0
        or float(calibration["recurrent_replay_ms_per_token"]) < 0
    ):
        raise ValueError("calibration values must be non-negative (H2D positive)")
    return calibration


def _journal(output_dir: Path, event: dict[str, Any]) -> None:
    path = output_dir / "matrix_journal.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _failure_class(stderr: str) -> str:
    if "insufficient GPU memory" in stderr or "insufficient host RAM" in stderr:
        return "resource_gate"
    if "TimeoutExpired" in stderr or "timed out" in stderr.lower():
        return "timeout"
    if "correctness mismatch" in stderr or "output hash mismatch" in stderr:
        return "correctness"
    return "runtime"


@contextmanager
def _exclusive_lock(path: Path, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another {label} run holds {path}") from error
        yield


def _paths(
    output_dir: Path,
    reference_baseline: str,
    concurrency: int,
    repetition: int,
    baseline: str,
):
    if reference_baseline == "marconi":
        stem = f"{baseline}_c{concurrency}_r{repetition}"
    else:
        stem = f"{reference_baseline}_vs_{baseline}_c{concurrency}_r{repetition}"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}_requests.jsonl"


def _completed(
    result_path: Path,
    request_path: Path,
    expected_requests: int,
    baseline: str,
    concurrency: int,
    trace: Path,
    trace_sha256: str,
    provenance: dict[str, Any],
    cpu_cache_gb: float,
    calibration_sha256: str | None,
) -> bool:
    if not result_path.is_file() or not request_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text().splitlines()[-1])
        requests = [
            json.loads(line) for line in request_path.read_text().splitlines() if line
        ]
    except (IndexError, json.JSONDecodeError):
        return False
    return (
        result.get("requests") == expected_requests
        and len(requests) == expected_requests
        and result.get("baseline", {}).get("name") == baseline
        and result.get("concurrency") == concurrency
        and Path(result.get("trace", "")).resolve() == trace.resolve()
        and result.get("trace_sha256") == trace_sha256
        and result.get("source_provenance") == provenance
        and float(result.get("cpu_cache_gb", -1)) == cpu_cache_gb
        and result.get("calibration_sha256") == calibration_sha256
    )


def _cache_pressure(result_path: Path, baseline: str) -> dict[str, int | str]:
    result = json.loads(result_path.read_text().splitlines()[-1])
    if BASELINES[baseline].cache_backend == "lmcache":
        counters = result.get("lmcache_l1") or {}
        return {
            "unit": "lmcache_chunks",
            "admitted": int(counters.get("write_chunks", 0)),
            "evicted": int(counters.get("evicted_chunks", 0)),
            "rejected": 0,
        }
    counters = result.get("cache_pressure") or {}
    return {
        "unit": "native_blocks",
        "admitted": int(counters.get("admitted_blocks", 0)),
        "evicted": int(counters.get("evicted_blocks", 0)),
        "rejected": int(counters.get("rejected_blocks", 0)),
    }


def _validate_cache_pressure(
    result_path: Path, baseline: str, require_evictions: bool
) -> dict[str, int | str]:
    pressure = _cache_pressure(result_path, baseline)
    if BASELINES[baseline].cache_backend == "none":
        return pressure
    if pressure["admitted"] <= 0:
        raise ValueError(f"{baseline} observed no cache admission")
    if require_evictions and pressure["evicted"] <= 0:
        raise ValueError(f"{baseline} formal cell observed no cache eviction")
    return pressure


def _cell_command(
    args: argparse.Namespace,
    baseline: str,
    concurrency: int,
    repetition: int,
    correctness_reference: Path | None,
) -> list[str]:
    result_path, request_path = _paths(
        args.output_dir, args.reference_baseline, concurrency, repetition, baseline
    )
    command = [
        sys.executable,
        str(CELL_RUNNER),
        "--trace", str(args.trace),
        "--baseline", baseline,
        "--model-path", str(args.model_path),
        "--cuda-visible-devices", args.cuda_visible_devices,
        "--concurrency", str(concurrency),
        "--arrival-model", args.arrival_model,
        "--request-rate", str(args.request_rate),
        "--burst-size", str(args.burst_size),
        "--zipf-exponent", str(args.zipf_exponent),
        "--zipf-max-rank", str(args.zipf_max_rank),
        "--max-tokens", str(args.max_tokens),
        "--request-timeout", str(args.request_timeout),
        "--execution-timeout", str(args.execution_timeout),
        "--limit", str(args.limit),
        "--cpu-cache-gb", str(args.cpu_cache_gb),
        "--host-reserve-gb", str(args.host_reserve_gb),
        "--min-cpu-cache-gb", str(args.min_cpu_cache_gb),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--output", str(result_path),
        "--request-output", str(request_path),
    ]
    if args.calibration_file is not None:
        command.extend(["--calibration-file", str(args.calibration_file)])
    if baseline in {"hyrex", "request_adaptive"}:
        command.extend([
            "--hyrex-h2d-gbps", str(args.hyrex_h2d_gbps),
            "--hyrex-full-replay-ms-per-token",
            str(args.hyrex_full_replay_ms_per_token),
            "--hyrex-recurrent-replay-ms-per-token",
            str(args.hyrex_recurrent_replay_ms_per_token),
        ])
        if args.hyrex_ttft_slo_ms is not None:
            command.extend(["--hyrex-ttft-slo-ms", str(args.hyrex_ttft_slo_ms)])
        if args.hyrex_starvation_ms is not None:
            command.extend([
                "--hyrex-starvation-ms", str(args.hyrex_starvation_ms)
            ])
    elif baseline == "kvpr_hybrid":
        command.extend([
            "--kvpr-h2d-gbps", str(args.kvpr_h2d_gbps),
            "--kvpr-replay-ms-per-token", str(args.kvpr_replay_ms_per_token),
        ])
    elif baseline == "cacheflow_hybrid":
        command.extend([
            "--cacheflow-h2d-gbps", str(args.cacheflow_h2d_gbps),
            "--cacheflow-replay-ms-per-token",
            str(args.cacheflow_replay_ms_per_token),
        ])
    if correctness_reference is not None:
        command.extend(["--correctness-reference", str(correctness_reference)])
    return command


def _run_matrix_unlocked(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_baseline = args.reference_baseline
    if reference_baseline not in BASELINES or reference_baseline == "hyrex":
        raise ValueError("--reference-baseline must name a non-HyRex baseline")
    calibration: dict[str, Any] | None = None
    if args.calibration_file is not None:
        calibration = load_calibration(args.calibration_file)
        args.hyrex_h2d_gbps = float(calibration["h2d_gbps"])
        args.hyrex_full_replay_ms_per_token = float(
            calibration["full_replay_ms_per_token"]
        )
        args.hyrex_recurrent_replay_ms_per_token = float(
            calibration["recurrent_replay_ms_per_token"]
        )
        args.kvpr_h2d_gbps = float(calibration["h2d_gbps"])
        args.kvpr_replay_ms_per_token = float(calibration["full_replay_ms_per_token"])
        args.cacheflow_h2d_gbps = float(calibration["h2d_gbps"])
        args.cacheflow_replay_ms_per_token = float(
            calibration["full_replay_ms_per_token"]
        )
    elif args.profile == "formal" and not args.plan_only:
        raise ValueError("formal execution requires --calibration-file from measurement")
    calibration_sha256 = (
        file_sha256(args.calibration_file) if args.calibration_file is not None else None
    )
    cache_budget: dict[str, float | bool] = {
        "requested_gb": args.cpu_cache_gb,
        "effective_gb": args.cpu_cache_gb,
        "host_reserve_gb": args.host_reserve_gb,
        "automatic": False,
    }
    if args.auto_cpu_cache_gb and not args.plan_only:
        cache_budget = cpu_cache_budget(
            args.cpu_cache_gb,
            args.host_reserve_gb,
            args.min_cpu_cache_gb,
            automatic=True,
        )
        args.cpu_cache_gb = float(cache_budget["effective_gb"])
    if not args.plan_only:
        _journal(args.output_dir, {
            "event": "matrix_config",
            "reference_baseline": reference_baseline,
            "cpu_cache": cache_budget,
            "calibration_sha256": calibration_sha256,
        })
    trace_rows = []
    if args.trace.is_file():
        trace_rows = [
            json.loads(line) for line in args.trace.read_text().splitlines() if line
        ]
    elif not args.plan_only:
        raise ValueError(f"trace does not exist: {args.trace}")
    required_sessions = args.min_sessions
    if required_sessions is None:
        required_sessions = 512 if args.profile == "formal" else 32
    if trace_rows:
        sessions = {str(row["session_id"]) for row in trace_rows}
        if len(sessions) < required_sessions:
            raise ValueError(
                f"{args.profile} profile needs {required_sessions} sessions, "
                f"trace has {len(sessions)}"
            )
    if args.limit is None:
        args.limit = len(trace_rows) if args.profile == "formal" else 40
    if args.limit <= 0 or (trace_rows and args.limit > len(trace_rows)):
        raise ValueError(
            f"invalid request limit={args.limit} for {len(trace_rows)} trace rows"
        )
    if args.max_tokens is None:
        args.max_tokens = 16 if args.profile == "formal" else 2
    if args.execution_timeout is None:
        args.execution_timeout = 7200.0 if args.profile == "formal" else 1800.0
    if args.require_evictions is None:
        args.require_evictions = args.profile == "formal"
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if any(value <= 0 for value in concurrencies):
        raise ValueError("concurrencies must be positive")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    pair_paths: dict[int, list[tuple[Path, Path]]] = {
        concurrency: [] for concurrency in concurrencies
    }
    provenance = source_provenance()
    trace_sha256 = file_sha256(args.trace) if trace_rows else ""
    plan = []
    previous_reference: Path | None = None
    for concurrency in concurrencies:
        previous_reference = None
        for repetition in range(args.repetitions):
            reference_result, reference_requests = _paths(
                args.output_dir, reference_baseline, concurrency, repetition,
                reference_baseline,
            )
            candidate_result, candidate_requests = _paths(
                args.output_dir, reference_baseline, concurrency, repetition, "hyrex"
            )
            order = (reference_baseline, "hyrex") if repetition % 2 == 0 else (
                "hyrex", reference_baseline
            )
            for baseline in order:
                result_path, request_path = _paths(
                    args.output_dir, reference_baseline, concurrency, repetition,
                    baseline,
                )
                if _completed(
                    result_path,
                    request_path,
                    args.limit,
                    baseline,
                    concurrency,
                    args.trace,
                    trace_sha256,
                    provenance,
                    args.cpu_cache_gb,
                    calibration_sha256,
                ):
                    plan.append({"action": "reuse", "result": str(result_path)})
                    if not args.plan_only:
                        _journal(args.output_dir, {
                            "event": "cell_reused",
                            "baseline": baseline,
                            "concurrency": concurrency,
                            "repetition": repetition,
                            "result": str(result_path),
                        })
                    continue
                correctness_reference = None
                if baseline == "hyrex":
                    reference_ready = _completed(
                        reference_result,
                        reference_requests,
                        args.limit,
                        reference_baseline,
                        concurrency,
                        args.trace,
                        trace_sha256,
                        provenance,
                        args.cpu_cache_gb,
                        calibration_sha256,
                    ) or (
                        args.plan_only
                        and order.index(reference_baseline) < order.index("hyrex")
                    )
                    correctness_reference = (
                        reference_requests if reference_ready else previous_reference
                    )
                    if correctness_reference is None:
                        raise RuntimeError(
                            "HyRex-first repetition needs a completed prior baseline "
                            "reference"
                        )
                command = _cell_command(
                    args, baseline, concurrency, repetition, correctness_reference
                )
                plan.append({"action": "run", "command": command})
                if not args.plan_only:
                    _journal(args.output_dir, {
                        "event": "cell_started",
                        "baseline": baseline,
                        "concurrency": concurrency,
                        "repetition": repetition,
                        "command": command,
                        "source_provenance": provenance,
                        "trace_sha256": trace_sha256,
                    })
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as error:
                        if error.stdout:
                            print(error.stdout, end="")
                        if error.stderr:
                            print(error.stderr, end="", file=sys.stderr)
                        _journal(args.output_dir, {
                            "event": "cell_failed",
                            "baseline": baseline,
                            "concurrency": concurrency,
                            "repetition": repetition,
                            "returncode": error.returncode,
                            "failure_class": _failure_class(error.stderr or ""),
                            "stdout_tail": (error.stdout or "")[-4096:],
                            "stderr_tail": (error.stderr or "")[-4096:],
                        })
                        raise
                    if completed.stdout:
                        print(completed.stdout, end="")
                    if completed.stderr:
                        print(completed.stderr, end="", file=sys.stderr)
                    _journal(args.output_dir, {
                        "event": "cell_completed",
                        "baseline": baseline,
                        "concurrency": concurrency,
                        "repetition": repetition,
                        "result": str(result_path),
                    })
            if args.plan_only:
                previous_reference = reference_requests
                continue
            pressure = {}
            for baseline, result_path in (
                (reference_baseline, reference_result),
                ("hyrex", candidate_result),
            ):
                try:
                    pressure[baseline] = _validate_cache_pressure(
                        result_path, baseline, args.require_evictions
                    )
                except ValueError as error:
                    _journal(args.output_dir, {
                        "event": "cache_pressure_failed",
                        "baseline": baseline,
                        "concurrency": concurrency,
                        "repetition": repetition,
                        "error": str(error),
                    })
                    raise
            report = analyze_pair(reference_result, candidate_result)
            report["cache_pressure"] = pressure
            pair_prefix = "" if reference_baseline == "marconi" else f"{reference_baseline}_vs_"
            pair_report = args.output_dir / f"{pair_prefix}pair_c{concurrency}_r{repetition}.json"
            pair_report.write_text(json.dumps(report, indent=2) + "\n")
            _journal(args.output_dir, {
                "event": "pair_validated",
                "concurrency": concurrency,
                "repetition": repetition,
                "report": str(pair_report),
            })
            pair_paths[concurrency].append((reference_result, candidate_result))
            previous_reference = reference_requests

    summaries = {}
    if not args.plan_only:
        for concurrency, pairs in pair_paths.items():
            summary = aggregate_pairs(pairs, min_repetitions=args.repetitions)
            prefix = "" if reference_baseline == "marconi" else f"{reference_baseline}_vs_"
            path = args.output_dir / f"{prefix}summary_c{concurrency}.json"
            path.write_text(json.dumps(summary, indent=2) + "\n")
            summaries[str(concurrency)] = summary
    return {
        "reference_baseline": reference_baseline,
        "plan": plan,
        "summaries": summaries,
        "cpu_cache": cache_budget,
        "calibration_sha256": calibration_sha256,
    }


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan_only:
        return _run_matrix_unlocked(args)
    device_key = "_".join(
        part.strip() for part in args.cuda_visible_devices.split(",") if part.strip()
    )
    if not device_key or not device_key.replace("_", "").isdigit():
        raise ValueError("--cuda-visible-devices must contain numeric device IDs")
    with (
        _exclusive_lock(args.output_dir / ".matrix.lock", "matrix"),
        _exclusive_lock(Path(f"/tmp/hyrex_gpu_{device_key}.lock"), "GPU"),
    ):
        return _run_matrix_unlocked(args)


def self_check() -> None:
    parser = build_parser()
    with tempfile.TemporaryDirectory() as directory:
        _journal(Path(directory), {"event": "self_check"})
        journal_row = json.loads(
            (Path(directory) / "matrix_journal.jsonl").read_text()
        )
        assert journal_row["event"] == "self_check"
        assert journal_row["timestamp"].endswith("+00:00")
        assert _failure_class("insufficient GPU memory") == "resource_gate"
        assert _failure_class("TimeoutExpired") == "timeout"
        assert _failure_class("output hash mismatch") == "correctness"
        assert _failure_class("unknown") == "runtime"
        calibration_path = Path(directory) / "calibration.json"
        calibration_path.write_text(json.dumps({
            "schema_version": 1,
            "provenance": {"kind": "unit-test"},
            "h2d_gbps": 19.0,
            "full_replay_ms_per_token": 0.0021,
            "recurrent_replay_ms_per_token": 0.0052,
        }))
        assert load_calibration(calibration_path)["h2d_gbps"] == 19.0
        pressure_result = Path(directory) / "pressure.jsonl"
        pressure_result.write_text(json.dumps({
            "cache_pressure": {
                "admitted_blocks": 3,
                "evicted_blocks": 2,
                "rejected_blocks": 1,
            }
        }) + "\n")
        assert _validate_cache_pressure(pressure_result, "hyrex", True) == {
            "unit": "native_blocks",
            "admitted": 3,
            "evicted": 2,
            "rejected": 1,
        }
        lock_path = Path(directory) / "test.lock"
        with _exclusive_lock(lock_path, "test"):
            try:
                with _exclusive_lock(lock_path, "test"):
                    raise AssertionError("nested lock unexpectedly succeeded")
            except RuntimeError as error:
                assert "another test run" in str(error)
        args = parser.parse_args([
            "--trace", "/trace.jsonl",
            "--output-dir", directory,
            "--concurrencies", "1,4",
            "--repetitions", "3",
            "--limit", "40",
            "--profile", "smoke",
            "--plan-only",
        ])
        result = run_matrix(args)
        commands = [item for item in result["plan"] if item["action"] == "run"]
        assert len(commands) == 12
        assert commands[0]["command"][commands[0]["command"].index("--baseline") + 1] == "marconi"
        assert commands[2]["command"][commands[2]["command"].index("--baseline") + 1] == "hyrex"
        assert "--correctness-reference" in commands[1]["command"]
        assert "--correctness-reference" in commands[2]["command"]
        tail_args = parser.parse_args([
            "--trace", "/trace.jsonl",
            "--output-dir", directory,
            "--reference-baseline", "tail_replay",
            "--concurrencies", "1",
            "--repetitions", "1",
            "--limit", "40",
            "--profile", "smoke",
            "--plan-only",
        ])
        tail_plan = run_matrix(tail_args)["plan"]
        assert "tail_replay_vs_tail_replay" in tail_plan[0]["command"][-3]
        assert tail_plan[0]["command"][tail_plan[0]["command"].index("--baseline") + 1] == "tail_replay"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--profile", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--reference-baseline",
        choices=tuple(name for name in BASELINES if name != "hyrex"),
        default="marconi",
    )
    parser.add_argument("--min-sessions", type=int, default=None)
    parser.add_argument("--concurrencies", default="1,4,8,16,32")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arrival-model", default="poisson")
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--zipf-exponent", type=float, default=1.2)
    parser.add_argument("--zipf-max-rank", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--execution-timeout", type=float, default=None)
    parser.add_argument("--calibration-file", type=Path, default=None)
    parser.add_argument("--cpu-cache-gb", type=float, default=24.0)
    parser.add_argument("--host-reserve-gb", type=float, default=20.0)
    parser.add_argument("--min-cpu-cache-gb", type=float, default=4.0)
    parser.add_argument(
        "--auto-cpu-cache-gb",
        action="store_true",
        help="resolve one safe CPU-cache budget before the whole matrix",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--hyrex-h2d-gbps", type=float, default=19.0)
    parser.add_argument("--hyrex-full-replay-ms-per-token", type=float, default=0.0021)
    parser.add_argument(
        "--hyrex-recurrent-replay-ms-per-token", type=float, default=0.0052
    )
    parser.add_argument("--hyrex-ttft-slo-ms", type=float, default=None)
    parser.add_argument("--hyrex-starvation-ms", type=float, default=None)
    parser.add_argument("--kvpr-h2d-gbps", type=float, default=19.0)
    parser.add_argument("--kvpr-replay-ms-per-token", type=float, default=0.0021)
    parser.add_argument("--cacheflow-h2d-gbps", type=float, default=19.0)
    parser.add_argument("--cacheflow-replay-ms-per-token", type=float, default=0.0021)
    parser.add_argument(
        "--require-evictions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
        return
    parser = build_parser()
    args = parser.parse_args()
    print(json.dumps(run_matrix(args), indent=2))


if __name__ == "__main__":
    main()
