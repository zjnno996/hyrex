# SPDX-License-Identifier: Apache-2.0
"""Safely orchestrate calibration, paired matrices, and the gated main table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hybrid_baseline_config import BASELINES
from run_native_independent_cells import MODEL
from run_online_session_cell import gpu_memory, memory_available_gb


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "benchmarks/reproductions/run_online_recovery_calibration.py"
MATRIX = ROOT / "benchmarks/reproductions/run_online_session_matrix.py"
TABLE = ROOT / "benchmarks/reproductions/build_online_main_table.py"


def resource_gate(args: argparse.Namespace) -> dict[str, float]:
    available = memory_available_gb()
    total, free = gpu_memory(args.cuda_visible_devices)
    host_required = args.cpu_cache_gb + args.host_reserve_gb
    gpu_required = total * args.gpu_memory_utilization + args.gpu_reserve_gb
    if available < host_required or free < gpu_required:
        raise RuntimeError(
            "formal campaign resource gate failed: "
            f"host={available:.1f}/{host_required:.1f} GiB, "
            f"gpu={free:.1f}/{gpu_required:.1f} GiB"
        )
    return {
        "host_available_gb": available,
        "host_required_gb": host_required,
        "gpu_free_gb": free,
        "gpu_required_gb": gpu_required,
    }


def calibration_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(CALIBRATION),
        "--trace", str(args.trace),
        "--output-dir", str(args.output_dir / "calibration"),
        "--model-path", str(args.model_path),
        "--cuda-visible-devices", args.cuda_visible_devices,
        "--arrival-model", args.arrival_model,
        "--request-rate", str(args.request_rate),
        "--burst-size", str(args.burst_size),
        "--zipf-exponent", str(args.zipf_exponent),
        "--zipf-max-rank", str(args.zipf_max_rank),
        "--cpu-cache-gb", str(args.cpu_cache_gb),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]


def matrix_command(args: argparse.Namespace, baseline: str) -> list[str]:
    return [
        sys.executable, str(MATRIX),
        "--profile", "formal",
        "--reference-baseline", baseline,
        "--trace", str(args.trace),
        "--output-dir", str(args.output_dir / baseline),
        "--model-path", str(args.model_path),
        "--cuda-visible-devices", args.cuda_visible_devices,
        "--arrival-model", args.arrival_model,
        "--request-rate", str(args.request_rate),
        "--burst-size", str(args.burst_size),
        "--zipf-exponent", str(args.zipf_exponent),
        "--zipf-max-rank", str(args.zipf_max_rank),
        "--concurrencies", args.concurrencies,
        "--repetitions", str(args.repetitions),
        "--cpu-cache-gb", str(args.cpu_cache_gb),
        "--host-reserve-gb", str(args.host_reserve_gb),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--calibration-file", str(args.output_dir / "calibration/recovery_calibration.json"),
    ]


def table_command(args: argparse.Namespace, baselines: list[str]) -> list[str]:
    command = [sys.executable, str(TABLE)]
    for baseline in baselines:
        command.extend(["--comparison", f"{baseline}={args.output_dir / baseline}"])
    return command + [
        "--concurrencies", args.concurrencies,
        "--repetitions", str(args.repetitions),
        "--markdown-output", str(args.output_dir / "main_table.md"),
        "--evidence-output", str(args.output_dir / "main_table_evidence.json"),
    ]


def plan(args: argparse.Namespace) -> list[list[str]]:
    baselines = args.reference_baselines.split(",")
    if any(name not in BASELINES or name == "hyrex" for name in baselines):
        raise ValueError("reference baselines must be non-HyRex registered baselines")
    return [
        calibration_command(args),
        *(matrix_command(args, baseline) for baseline in baselines),
        table_command(args, baselines),
    ]


def self_check() -> None:
    args = argparse.Namespace(
        trace=Path("/trace"), output_dir=Path("/out"), model_path=Path("/model"),
        cuda_visible_devices="0", cpu_cache_gb=24.0, host_reserve_gb=20.0,
        gpu_memory_utilization=0.9, gpu_reserve_gb=1.0, concurrencies="1,4,8,16",
        repetitions=3, reference_baselines=(
            "full_recompute,hybrid_all_load,request_adaptive,marconi,tail_replay"
        ),
        arrival_model="poisson", request_rate=8.0, burst_size=8,
        zipf_exponent=1.2, zipf_max_rank=64,
    )
    commands = plan(args)
    assert len(commands) == 7
    assert commands[0][1] == str(CALIBRATION)
    assert "--calibration-file" in commands[1]
    assert commands[1][commands[1].index("--arrival-model") + 1] == "poisson"
    assert commands[-1][1] == str(TABLE)


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
    parser.add_argument(
        "--arrival-model",
        choices=("uniform", "poisson", "bursty", "zipf"),
        default="poisson",
    )
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--zipf-exponent", type=float, default=1.2)
    parser.add_argument("--zipf-max-rank", type=int, default=64)
    parser.add_argument("--concurrencies", default="1,4,8,16,32")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cpu-cache-gb", type=float, default=24.0)
    parser.add_argument("--host-reserve-gb", type=float, default=20.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-reserve-gb", type=float, default=1.0)
    parser.add_argument(
        "--reference-baselines",
        default=(
            "full_recompute,hybrid_all_load,request_adaptive,marconi,tail_replay"
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (
        args.request_rate <= 0
        or args.burst_size < 1
        or args.zipf_exponent <= 0
        or args.zipf_max_rank < 1
    ):
        raise ValueError("arrival parameters must be positive")
    commands = plan(args)
    if args.plan_only:
        print(json.dumps({"commands": commands}, indent=2))
        return
    if not args.trace.is_file() or not args.model_path.is_dir():
        raise ValueError("trace or model path does not exist")
    print(json.dumps({"resource_preflight": resource_gate(args)}, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        if command[1] == str(CALIBRATION):
            calibration_dir = args.output_dir / "calibration"
            if (
                (calibration_dir / "recovery_calibration.json").is_file()
                and (calibration_dir / "native_feedback_result.jsonl").is_file()
            ):
                print(json.dumps({"reused_calibration": str(calibration_dir)}))
                continue
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
