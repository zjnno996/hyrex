# SPDX-License-Identifier: Apache-2.0
"""Run one continuous online Hybrid-cache E2E cell on a single vLLM worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from hybrid_baseline_config import BASELINES, baseline_config
from run_native_independent_cells import (
    MODEL,
    remember_process_group,
    stop_server,
    wait_lmcache_ready,
)
from run_partial_prefix_cells import vllm_command, wait_ready_for_process


ROOT = Path(__file__).resolve().parents[2]
TRACE_E2E = ROOT / "benchmarks/reproductions/sharegpt_hybrid_trace_e2e.py"


def source_provenance(root: Path = ROOT) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "."],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_head": head,
        "dirty_paths": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def start_observable_lmcache(
    config: Path, port: int, http_port: int, chunk_size: int
) -> subprocess.Popen:
    values = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in config.read_text(encoding="utf-8").splitlines()
        if ":" in line and not line.lstrip().startswith("#")
    }
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--http-host", "127.0.0.1",
        "--http-port", str(http_port),
        "--chunk-size", str(chunk_size),
        "--l1-size-gb", values.get("max_local_cpu_size", "24"),
        "--eviction-policy", "LRU",
        "--separate-object-groups",
    ]
    log_file = Path(f"/tmp/lmcache_mp_{port}.log").open("w")
    process = subprocess.Popen(
        command,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    remember_process_group(process)
    process._lmcache_log_file = log_file  # type: ignore[attr-defined]
    return process


def lmcache_l1_counters(http_port: int) -> dict[str, float]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{http_port}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")
    return parse_l1_counters(text)


def parse_l1_counters(text: str) -> dict[str, float]:
    totals = {"write_chunks": 0.0, "evicted_chunks": 0.0}
    prefixes = {
        "write_chunks": "lmcache_mp_l1_write",
        "evicted_chunks": "lmcache_mp_l1_evicted",
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        bare_name = name.split("{", 1)[0]
        for key, prefix in prefixes.items():
            if bare_name.startswith(prefix) and bare_name.endswith("_total"):
                totals[key] += float(value)
    return totals


def memory_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def cpu_cache_budget(
    requested_gb: float,
    host_reserve_gb: float,
    min_cache_gb: float,
    automatic: bool,
    available_gb: float | None = None,
) -> dict[str, float | bool]:
    """Resolve one reproducible CPU-cache budget before starting a cell."""
    if requested_gb <= 0 or host_reserve_gb <= 0 or min_cache_gb <= 0:
        raise ValueError("CPU cache, host reserve, and minimum cache must be positive")
    available = memory_available_gb() if available_gb is None else available_gb
    effective = requested_gb
    if automatic:
        effective = min(requested_gb, max(0.0, available - host_reserve_gb))
    if effective < min_cache_gb:
        raise RuntimeError(
            f"insufficient host RAM for CPU cache: available={available:.1f} GiB, "
            f"reserve={host_reserve_gb:.1f} GiB, minimum_cache={min_cache_gb:.1f} GiB"
        )
    return {
        "requested_gb": requested_gb,
        "effective_gb": effective,
        "host_available_gb": available,
        "host_reserve_gb": host_reserve_gb,
        "automatic": automatic,
    }


def gpu_memory(device: str) -> tuple[float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            device.split(",", 1)[0],
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total_mib, free_mib = (float(value.strip()) for value in result.stdout.split(","))
    return total_mib / 1024, free_mib / 1024


def check_resources(args: argparse.Namespace) -> dict[str, float]:
    vllm_executable = Path(sys.executable).with_name("vllm")
    if not vllm_executable.is_file() or not os.access(vllm_executable, os.X_OK):
        raise RuntimeError(
            f"vLLM executable is unavailable next to Python: {vllm_executable}"
        )
    available = memory_available_gb()
    gpu_total, gpu_free = gpu_memory(args.cuda_visible_devices)
    required_host = args.cpu_cache_gb + args.host_reserve_gb
    required_gpu = gpu_total * args.gpu_memory_utilization + args.gpu_reserve_gb
    if available < required_host:
        raise RuntimeError(
            f"insufficient host RAM: available={available:.1f} GiB, "
            f"required={required_host:.1f} GiB "
            "(CPU cache + model/startup reserve); refusing to start"
        )
    if gpu_free < required_gpu:
        raise RuntimeError(
            f"insufficient GPU memory: free={gpu_free:.1f} GiB, "
            f"required={required_gpu:.1f} GiB; refusing to start"
        )
    return {
        "host_available_gb": available,
        "host_required_gb": required_host,
        "gpu_total_gb": gpu_total,
        "gpu_free_gb": gpu_free,
        "gpu_required_gb": required_gpu,
    }


def result_json(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"online E2E produced no JSON result: {output}")


def cache_events(log_path: Path) -> dict[str, dict[str, Any]]:
    events = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for marker in (
            "HYREX_NATIVE_DECISION ",
            "KVPR_H_NATIVE_DECISION ",
            "CACHEFLOW_H_NATIVE_DECISION ",
            "HYREX_NATIVE_DISPATCH ",
            "HYREX_NATIVE_EVICTION_EVENT ",
            "HYREX_CACHE_EVENT ",
            "HYREX_TRANSFER_EVENT ",
            "HYREX_NATIVE_CACHE_EVENT ",
            "HYREX_NATIVE_TRANSFER_EVENT ",
        ):
            if marker in line:
                event = json.loads(line.split(marker, 1)[1])
                merged = events.setdefault(event["request_id"], {})
                for key, value in event.items():
                    if key in {"retrieve_fence_completed", "store_admitted"}:
                        merged[key] = bool(merged.get(key)) or bool(value)
                    elif key in {
                        "admitted_blocks",
                        "evicted_blocks",
                        "rejected_blocks",
                    }:
                        merged[key] = int(merged.get(key, 0)) + int(value)
                    else:
                        merged[key] = value
                break
        else:
            lookup = re.search(
                r"LMCache lookup complete request=(\S+) hit_tokens=(\d+)", line
            )
            if lookup:
                request_id, hit_tokens = lookup.groups()
                events.setdefault(request_id, {}).update({
                    "request_id": request_id,
                    "cache_state": "cold" if int(hit_tokens) == 0 else "gpu",
                    "cpu_hit_tokens": int(hit_tokens),
                    "gpu_hit_tokens": int(hit_tokens),
                    "h2d_tokens": 0,
                    "recovery_action": "load_all",
                })
                continue
            prefix = re.search(
                r"LMCache prefix match request=(\S+) local_gpu_tokens=(\d+) "
                r"cpu_tokens=(\d+) need_h2d_tokens=(\d+)",
                line,
            )
            if prefix:
                request_id, gpu_tokens, cpu_tokens, h2d_tokens = prefix.groups()
                gpu_tokens, cpu_tokens, h2d_tokens = map(
                    int, (gpu_tokens, cpu_tokens, h2d_tokens)
                )
                events.setdefault(request_id, {}).update({
                    "request_id": request_id,
                    "cache_state": "cpu" if h2d_tokens else "gpu",
                    "cpu_hit_tokens": cpu_tokens,
                    "gpu_hit_tokens": gpu_tokens,
                    "h2d_tokens": h2d_tokens,
                    "recovery_action": "load_all",
                })
                continue
            retrieve = re.search(r"LMCache H2D retrieve request=(\S+)", line)
            if retrieve:
                events.setdefault(retrieve.group(1), {}).update({
                    "retrieve_fence_completed": True,
                })
    return events


def decision_summary(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decisions = [event for event in events.values() if "policy" in event]
    return {
        "count": len(decisions),
        "policies": dict(Counter(event["policy"] for event in decisions)),
        "h2d_cost_sources": dict(
            Counter(event.get("h2d_gbps_source", "unknown") for event in decisions)
        ),
        "replay_cost_sources": dict(
            Counter(event.get("replay_cost_source", "unknown") for event in decisions)
        ),
    }


def cache_pressure_summary(events: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        "admitted_blocks": sum(
            int(event.get("admitted_blocks", 0)) for event in events.values()
        ),
        "evicted_blocks": sum(
            int(event.get("evicted_blocks", 0)) for event in events.values()
        ),
        "rejected_blocks": sum(
            int(event.get("rejected_blocks", 0)) for event in events.values()
        ),
        "requests_with_eviction": sum(
            int(event.get("evicted_blocks", 0) > 0) for event in events.values()
        ),
    }


def recovery_telemetry_summary(events: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Aggregate actual native transfer feedback and modeled replay work."""
    def number(event: dict[str, Any], key: str) -> float:
        value = event.get(key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    modeled_replay = 0.0
    modeled_load = 0.0
    for event in events.values():
        policy = event.get("policy")
        if policy in {"kvpr_split", "cacheflow_split"}:
            modeled_load += number(event, "modeled_load_bytes")
            modeled_replay += number(event, "modeled_replay_ms")
        elif policy == "all_load":
            modeled_load += number(event, "full_load_bytes") + number(
                event, "recurrent_load_bytes"
            )
        elif policy == "full_load_linear_replay":
            modeled_load += number(event, "full_load_bytes")
            modeled_replay += number(event, "recurrent_replay_ms")
        elif policy == "full_replay_linear_load":
            modeled_load += number(event, "recurrent_load_bytes")
            modeled_replay += number(event, "full_replay_ms")
    return {
        "retrieve_bytes": sum(number(event, "retrieve_bytes") for event in events.values()),
        "retrieve_observed_ms": sum(
            number(event, "retrieve_observed_ms") for event in events.values()
        ),
        "retrieve_queue_ms": sum(
            number(event, "retrieve_queue_ms") for event in events.values()
        ),
        "h2d_queue_before_ms": sum(
            number(event, "h2d_queue_before_ms") for event in events.values()
        ),
        "compute_queue_before_ms": sum(
            number(event, "compute_queue_before_ms") for event in events.values()
        ),
        "h2d_tokens": sum(number(event, "h2d_tokens") for event in events.values()),
        "modeled_load_bytes": modeled_load,
        "modeled_replay_ms": modeled_replay,
    }


def join_cache_events(
    request_path: Path,
    events: dict[str, dict[str, Any]],
    *,
    cacheless: bool = False,
) -> Counter:
    rows = [json.loads(line) for line in request_path.read_text().splitlines()]
    if cacheless:
        if events:
            raise RuntimeError("cacheless baseline unexpectedly emitted cache events")
        return Counter({"cold": len(rows)})
    joined = {}
    for row in rows:
        request_id = row["request_id"]
        matches = [
            event
            for event_id, event in events.items()
            if event_id == request_id or event_id.startswith(f"{request_id}-")
        ]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous cache events for request {request_id}")
        if matches:
            joined[request_id] = matches[0]
    missing = [row["request_id"] for row in rows if row["request_id"] not in joined]
    if missing:
        raise RuntimeError(f"missing cache events for {len(missing)} requests: {missing[:3]}")
    unfenced = [
        row["request_id"]
        for row in rows
        if joined[row["request_id"]].get("h2d_tokens", 0) > 0
        and not joined[row["request_id"]].get("retrieve_fence_completed")
    ]
    if unfenced:
        raise RuntimeError(
            f"missing retrieve completion fence for {len(unfenced)} requests: "
            f"{unfenced[:3]}"
        )
    states = Counter()
    with request_path.open("w", encoding="utf-8") as output:
        for row in rows:
            event = joined[row["request_id"]]
            states[event["cache_state"]] += 1
            output.write(json.dumps({**row, **event}) + "\n")
    return states


def self_check() -> None:
    assert cpu_cache_budget(24, 20, 4, False, available_gb=44) == {
        "requested_gb": 24,
        "effective_gb": 24,
        "host_available_gb": 44,
        "host_reserve_gb": 20,
        "automatic": False,
    }
    assert cpu_cache_budget(24, 20, 4, True, available_gb=28)["effective_gb"] == 8
    try:
        cpu_cache_budget(24, 20, 4, True, available_gb=23)
    except RuntimeError:
        pass
    else:
        raise AssertionError("automatic CPU-cache budgeting accepted too little RAM")
    with tempfile.TemporaryDirectory() as directory:
        request_path = Path(directory) / "requests.jsonl"
        request_path.write_text(
            json.dumps({"request_id": "cmpl-hyrex-0-0", "ttft_ms": 1.0}) + "\n"
        )
        log_path = Path(directory) / "server.log"
        log_path.write_text(
            'INFO HYREX_NATIVE_DECISION {"request_id":"cmpl-hyrex-0-0-engine",'
            '"policy":"full_load_linear_replay","h2d_gbps_source":"measured",'
            '"replay_cost_source":"measured"}\n'
            'INFO HYREX_NATIVE_DISPATCH {"request_id":"cmpl-hyrex-0-0-engine",'
            '"job_id":7,"dispatch_rank":0,"dispatch_batch_size":2}\n'
            'INFO HYREX_NATIVE_EVICTION_EVENT '
            '{"request_id":"cmpl-hyrex-0-0-engine","admitted_blocks":3,'
            '"evicted_blocks":2,"rejected_blocks":0}\n'
            'INFO HYREX_CACHE_EVENT {"request_id":"cmpl-hyrex-0-0-engine",'
            '"cache_state":"cpu","h2d_tokens":528,"recovery_action":"load_all"}\n'
            'INFO HYREX_TRANSFER_EVENT {"request_id":"cmpl-hyrex-0-0-engine",'
            '"retrieve_observed_ms":2.5,"retrieve_fence_completed":true}\n'
            'INFO HYREX_TRANSFER_EVENT {"request_id":"cmpl-hyrex-0-0-engine",'
            '"operation":"store","retrieve_fence_completed":false,'
            '"store_admitted":true}\n'
        )
        events = cache_events(log_path)
        states = join_cache_events(request_path, events)
        row = json.loads(request_path.read_text())
        assert states == {"cpu": 1}
        assert row["h2d_tokens"] == 528
        assert row["retrieve_observed_ms"] == 2.5
        assert row["retrieve_fence_completed"] is True
        assert row["store_admitted"] is True
        assert row["recovery_action"] == "load_all"
        assert row["policy"] == "full_load_linear_replay"
        assert row["dispatch_rank"] == 0
        assert row["dispatch_batch_size"] == 2
        assert cache_pressure_summary(events) == {
            "admitted_blocks": 3,
            "evicted_blocks": 2,
            "rejected_blocks": 0,
            "requests_with_eviction": 1,
        }
        assert recovery_telemetry_summary(events) == {
            "retrieve_bytes": 0.0,
            "retrieve_observed_ms": 2.5,
            "retrieve_queue_ms": 0.0,
            "h2d_queue_before_ms": 0.0,
            "compute_queue_before_ms": 0.0,
            "h2d_tokens": 528.0,
            "modeled_load_bytes": 0.0,
            "modeled_replay_ms": 0.0,
        }
        assert decision_summary(events) == {
            "count": 1,
            "policies": {"full_load_linear_replay": 1},
            "h2d_cost_sources": {"measured": 1},
            "replay_cost_sources": {"measured": 1},
        }
        baseline_log = Path(directory) / "baseline.log"
        baseline_log.write_text(
            'INFO KVPR_H_NATIVE_DECISION {"request_id":"kvpr-0",'
            '"policy":"kvpr_split","h2d_gbps_source":"calibration",'
            '"replay_cost_source":"calibration","modeled_load_bytes":12,'
            '"modeled_replay_ms":3}\n'
            'INFO CACHEFLOW_H_NATIVE_DECISION {"request_id":"cacheflow-0",'
            '"policy":"cacheflow_split","h2d_gbps_source":"calibration",'
            '"replay_cost_source":"calibration","modeled_load_bytes":8,'
            '"modeled_replay_ms":2}\n'
        )
        baseline_events = cache_events(baseline_log)
        assert decision_summary(baseline_events)["policies"] == {
            "kvpr_split": 1,
            "cacheflow_split": 1,
        }
        assert recovery_telemetry_summary(baseline_events)["modeled_load_bytes"] == 20
        assert recovery_telemetry_summary(baseline_events)["modeled_replay_ms"] == 5
        request_path.write_text(
            json.dumps({"request_id": "cmpl-hyrex-0-0", "ttft_ms": 1.0}) + "\n"
        )
        log_path.write_text(
            "LMCache lookup complete request=cmpl-hyrex-0-0-engine "
            "hit_tokens=528 lookup_ms=1.0\n"
            "LMCache prefix match request=cmpl-hyrex-0-0-engine "
            "local_gpu_tokens=0 cpu_tokens=528 need_h2d_tokens=528\n"
            "LMCache H2D retrieve request=cmpl-hyrex-0-0-engine group=0\n"
        )
        states = join_cache_events(request_path, cache_events(log_path))
        row = json.loads(request_path.read_text())
        assert states == {"cpu": 1}
        assert row["recovery_action"] == "load_all"
        assert row["retrieve_fence_completed"] is True
        assert parse_l1_counters(
            "lmcache_mp_l1_write_total{cache_salt=\"\"} 3\n"
            "lmcache_mp_l1_evicted_total{cache_salt=\"\"} 2\n"
        ) == {"write_chunks": 3.0, "evicted_chunks": 2.0}
    native_command = vllm_command(
        "P1",
        server_port=1,
        lmcache_port=2,
        max_model_len=528,
        max_num_batched_tokens=528,
        gpu_memory_utilization=0.8,
        model_path=Path("/model"),
        model_name="model",
        attention_backend="TRITON_ATTN",
        cache_backend="native",
        cpu_cache_gb=24,
    )
    assert native_command[-4:] == [
        "--kv-offloading-backend", "native",
        "--kv-offloading-size", "24",
    ]
    assert "--kv-transfer-config" not in native_command
    hyrex_native_command = vllm_command(
        "P1",
        server_port=1,
        lmcache_port=2,
        max_model_len=528,
        max_num_batched_tokens=528,
        gpu_memory_utilization=0.8,
        model_path=Path("/model"),
        model_name="model",
        attention_backend="TRITON_ATTN",
        cache_backend="native",
        cpu_cache_gb=24,
        native_eviction_policy="hyrex",
    )
    transfer_config = json.loads(
        hyrex_native_command[hyrex_native_command.index("--kv-transfer-config") + 1]
    )
    assert transfer_config["kv_connector_extra_config"] == {
        "eviction_policy": "hyrex"
    }


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", choices=tuple(BASELINES), required=True)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL))
    parser.add_argument("--model-name", default="Qwen3.5-9B")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--arrival-model",
        choices=("saturated", "uniform", "poisson", "bursty", "zipf"),
        default="poisson",
    )
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--zipf-exponent", type=float, default=1.2)
    parser.add_argument("--zipf-max-rank", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lmcache-chunk-size", type=int, default=528)
    parser.add_argument(
        "--cpu-cache-gb",
        "--lmcache-kv-gb",
        dest="cpu_cache_gb",
        type=float,
        default=24.0,
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--host-reserve-gb", type=float, default=20.0)
    parser.add_argument("--min-cpu-cache-gb", type=float, default=4.0)
    parser.add_argument(
        "--auto-cpu-cache-gb",
        action="store_true",
        help="cap the requested CPU cache at current host-memory headroom",
    )
    parser.add_argument("--gpu-reserve-gb", type=float, default=1.0)
    parser.add_argument("--server-port", type=int, default=8012)
    parser.add_argument("--lmcache-port", type=int, default=5562)
    parser.add_argument("--lmcache-http-port", type=int, default=5563)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--execution-timeout", type=float, default=None)
    parser.add_argument("--calibration-file", type=Path, default=None)
    parser.add_argument("--hyrex-h2d-gbps", type=float, default=None)
    parser.add_argument("--hyrex-full-replay-ms-per-token", type=float, default=None)
    parser.add_argument(
        "--hyrex-recurrent-replay-ms-per-token", type=float, default=None
    )
    parser.add_argument("--hyrex-ttft-slo-ms", type=float, default=None)
    parser.add_argument("--hyrex-starvation-ms", type=float, default=None)
    parser.add_argument("--kvpr-h2d-gbps", type=float, default=None)
    parser.add_argument("--kvpr-replay-ms-per-token", type=float, default=None)
    parser.add_argument("--cacheflow-h2d-gbps", type=float, default=None)
    parser.add_argument("--cacheflow-replay-ms-per-token", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-output", type=Path, required=True)
    parser.add_argument("--correctness-reference", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.request_timeout <= 0 or (
        args.execution_timeout is not None and args.execution_timeout <= 0
    ):
        raise ValueError("request/execution timeouts must be positive")

    baseline = BASELINES[args.baseline]
    if not baseline.runtime_bound:
        raise ValueError(
            f"{args.baseline} policy={baseline.policy_plugin!r} is available, "
            "but has no runtime recovery binding"
        )
    assert baseline.recovery_policy is not None
    hyrex_calibration = {
        "hyrex_h2d_gbps": args.hyrex_h2d_gbps,
        "hyrex_concurrency": args.concurrency,
        "hyrex_full_replay_ms_per_token": args.hyrex_full_replay_ms_per_token,
        "hyrex_recurrent_replay_ms_per_token": (
            args.hyrex_recurrent_replay_ms_per_token
        ),
    }
    if args.hyrex_ttft_slo_ms is not None:
        if args.hyrex_ttft_slo_ms <= 0:
            raise ValueError("--hyrex-ttft-slo-ms must be positive")
        hyrex_calibration["hyrex_ttft_slo_ms"] = args.hyrex_ttft_slo_ms
    if args.hyrex_starvation_ms is not None:
        if args.hyrex_starvation_ms <= 0:
            raise ValueError("--hyrex-starvation-ms must be positive")
        hyrex_calibration["hyrex_starvation_ms"] = args.hyrex_starvation_ms
    if args.baseline in {"hyrex", "request_adaptive"} and (
        args.hyrex_h2d_gbps is None
        or args.hyrex_h2d_gbps <= 0
        or args.hyrex_full_replay_ms_per_token is None
        or args.hyrex_full_replay_ms_per_token < 0
        or args.hyrex_recurrent_replay_ms_per_token is None
        or args.hyrex_recurrent_replay_ms_per_token < 0
    ):
        raise ValueError(
            "HyRex-style recovery requires non-negative measured "
            "--hyrex-h2d-gbps and "
            "per-token Full/recurrent replay costs"
        )
    if args.baseline == "kvpr_hybrid" and (
        args.kvpr_h2d_gbps is None
        or args.kvpr_h2d_gbps <= 0
        or args.kvpr_replay_ms_per_token is None
        or args.kvpr_replay_ms_per_token <= 0
    ):
        raise ValueError(
            "KVPR-H requires positive measured --kvpr-h2d-gbps and "
            "--kvpr-replay-ms-per-token"
        )
    if args.baseline == "cacheflow_hybrid" and (
        args.cacheflow_h2d_gbps is None
        or args.cacheflow_h2d_gbps <= 0
        or args.cacheflow_replay_ms_per_token is None
        or args.cacheflow_replay_ms_per_token <= 0
    ):
        raise ValueError(
            "CacheFlow-H requires positive measured --cacheflow-h2d-gbps and "
            "--cacheflow-replay-ms-per-token"
        )
    if not args.trace.is_file() or not args.model_path.is_dir():
        raise ValueError("trace or model path does not exist")
    if args.calibration_file is not None and not args.calibration_file.is_file():
        raise ValueError("calibration file does not exist")
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    if not rows:
        raise ValueError("trace is empty")
    cache_budget = cpu_cache_budget(
        args.cpu_cache_gb,
        args.host_reserve_gb,
        args.min_cpu_cache_gb,
        args.auto_cpu_cache_gb,
    )
    args.cpu_cache_gb = float(cache_budget["effective_gb"])
    resources = check_resources(args)
    resources["cpu_cache_budget"] = cache_budget
    if args.check_only:
        print(json.dumps(resources))
        return

    max_model_len = (
        (
            max(int(row["resume_tokens"]) for row in rows)
            + args.max_tokens
            + 127
        )
        // 128
        * 128
    )
    config_path: Path | None = None
    if baseline.cache_backend == "lmcache":
        config_path = Path(f"/tmp/online_lmcache_{args.lmcache_port}.yaml")
        config_path.write_text(
            f"chunk_size: {args.lmcache_chunk_size}\n"
            "local_device: cpu\nlocal_cpu: true\n"
            f"max_local_cpu_size: {args.cpu_cache_gb}\n"
            # Hybrid KV/recurrent objects use the normal CPU tier. Activation
            # checkpoints are outside this experiment and must not reserve a
            # second host-memory pool.
            "enable_hidden_state_cache: false\n"
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    env.update({
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "PYTHONHASHSEED": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
        "VLLM_MOONCAKE_HYBRID_POLICY": baseline.recovery_policy,
        "VLLM_MOONCAKE_HYBRID_ACTIVATION_CHECKPOINT": "0",
        "VLLM_MOONCAKE_HYBRID_SUFFIX_ONLY": (
            "1" if args.baseline == "hyrex" else "0"
        ),
        "VLLM_HYREX_SCHEDULE_LOADS": (
            "1" if args.baseline == "hyrex" else "0"
        ),
        "LMCACHE_MP_STRICT_LAYER_LOAD": (
            "1" if args.baseline == "tail_replay" else "0"
        ),
    })
    if config_path is not None:
        env["LMCACHE_CONFIG_FILE"] = str(config_path)
    if args.baseline == "tail_replay":
        env["LMCACHE_USE_HYREX_EXTERNAL_MP"] = "1"
        env["HYREX_TAIL_COMPAT_ALL_OBJECT_RETRIEVE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    lmcache = None
    if config_path is not None:
        lmcache = start_observable_lmcache(
            config_path,
            port=args.lmcache_port,
            http_port=args.lmcache_http_port,
            chunk_size=args.lmcache_chunk_size,
        )
    worker = None
    server_log_path = Path(
        f"/tmp/online_{args.baseline}_c{args.concurrency}_{args.server_port}.log"
    )
    server_log = server_log_path.open("w")
    launch_resources = resources
    try:
        counters_before = None
        if lmcache is not None:
            wait_lmcache_ready(args.lmcache_http_port, 180)
            counters_before = lmcache_l1_counters(args.lmcache_http_port)
        server = f"http://127.0.0.1:{args.server_port}"
        # Recheck immediately before model startup. External jobs do not honor
        # the matrix GPU lock and may have claimed memory since preflight.
        launch_resources = check_resources(args)
        worker = subprocess.Popen(
            vllm_command(
                "P1",
                server_port=args.server_port,
                lmcache_port=args.lmcache_port,
                max_model_len=max_model_len,
                max_num_batched_tokens=args.lmcache_chunk_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                model_path=args.model_path,
                model_name=args.model_name,
                attention_backend="TRITON_ATTN",
                cache_backend=baseline.cache_backend,
                cpu_cache_gb=args.cpu_cache_gb,
                native_eviction_policy=(
                    "hyrex" if args.baseline == "hyrex" else None
                ),
            ),
            env=env,
            cwd=ROOT,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        remember_process_group(worker)
        wait_ready_for_process(worker, server, args.startup_timeout)
        command = [
            sys.executable,
            str(TRACE_E2E),
            "--trace", str(args.trace),
            "--prompt-field", "resume_prompt",
            "--execution-mode", "online",
            "--server", server,
            "--model", args.model_name,
            "--baseline", args.baseline,
            "--concurrency", str(args.concurrency),
            "--arrival-model", args.arrival_model,
            "--request-rate", str(args.request_rate),
            "--burst-size", str(args.burst_size),
            "--zipf-exponent", str(args.zipf_exponent),
            "--zipf-max-rank", str(args.zipf_max_rank),
            "--max-tokens", str(args.max_tokens),
            "--request-output", str(args.request_output),
            "--request-timeout", str(args.request_timeout),
        ]
        if args.baseline in {"hyrex", "request_adaptive"}:
            if args.baseline == "hyrex":
                hyrex_calibration["hyrex_enable_mixed_recovery"] = False
            command += [
                "--kv-transfer-params-json",
                json.dumps(hyrex_calibration),
            ]
        elif args.baseline == "kvpr_hybrid":
            command += [
                "--kv-transfer-params-json",
                json.dumps({
                    "kvpr_h2d_gbps": args.kvpr_h2d_gbps,
                    "kvpr_replay_ms_per_token": args.kvpr_replay_ms_per_token,
                }),
            ]
        elif args.baseline == "cacheflow_hybrid":
            command += [
                "--kv-transfer-params-json",
                json.dumps({
                    "cacheflow_h2d_gbps": args.cacheflow_h2d_gbps,
                    "cacheflow_replay_ms_per_token": (
                        args.cacheflow_replay_ms_per_token
                    ),
                }),
            ]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        if args.correctness_reference is not None:
            command += [
                "--correctness-reference",
                str(args.correctness_reference),
            ]
        try:
            completed = subprocess.run(
                command,
                env=env,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=args.execution_timeout,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr, end="")
            if exc.stderr:
                print(exc.stderr, file=sys.stderr, end="")
            raise
        server_log.flush()
        events = cache_events(server_log_path)
        states = join_cache_events(
            args.request_output,
            events,
            cacheless=baseline.cache_backend == "none",
        )
        l1_counters = None
        if counters_before is not None:
            counters_after = lmcache_l1_counters(args.lmcache_http_port)
            l1_counters = {
                key: counters_after[key] - counters_before[key]
                for key in counters_before
            }
        result = {
            **result_json(completed.stdout),
            "workload": "continuous_online_multiturn",
            "trace": str(args.trace),
            "trace_sha256": file_sha256(args.trace),
            "server_log": str(server_log_path),
            "resource_preflight": resources,
            "resources": launch_resources,
            "cpu_cache_gb": args.cpu_cache_gb,
            "cpu_cache_budget": cache_budget,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "execution_timeout": args.execution_timeout,
            "calibration_sha256": (
                file_sha256(args.calibration_file)
                if args.calibration_file is not None else None
            ),
            "source_provenance": source_provenance(),
            "baseline": baseline_config(args.baseline),
            "cache_state_counts": dict(states),
            "hyrex_decisions": decision_summary(events),
            "recovery_telemetry": recovery_telemetry_summary(events),
            "cache_pressure": cache_pressure_summary(events),
            "lmcache_l1": l1_counters,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result) + "\n")
        print(json.dumps(result))
    finally:
        if worker is not None:
            stop_server(worker)
        server_log.close()
        if lmcache is not None:
            stop_server(lmcache)
        if config_path is not None:
            config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
