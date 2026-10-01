# SPDX-License-Identifier: Apache-2.0
"""Run isolated native H2D TTFT cells with a fresh vLLM process per cell."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "benchmarks/reproductions/sharegpt_mooncake_e2e.py"
MODEL = "/root/models/Qwen3.5-9B"
DATASET = "/root/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
TOKENIZER = MODEL
POLICIES = {
    "P1": ("all_load", True),
    "P2": ("all_load", False),
    "P3": ("full_load_linear_replay", True),
    "P4": ("full_replay_linear_load", True),
    # P5: Full KV history plus only the terminal Mamba/GDN state page.
    "P5": ("terminal_linear_load", True),
    "Adaptive": ("adaptive", True),
}
SUMMARY_RE = re.compile(
    r"latency_ms_p50/p90/p99=([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
TPOT_SUMMARY_RE = re.compile(
    r"tpot_ms_p50/p90/p99=([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
OUTPUT_SIGNATURE_RE = re.compile(r"first_token_signature=([0-9a-f]{16})")
FIRST_TOKENS_RE = re.compile(r"first_tokens_json=(\[.*\])")
COMPLETION_TEXTS_RE = re.compile(r"completion_texts_json=(\[.*\])")
BACKEND_REQUEST_IDS_RE = re.compile(r"backend_request_ids_json=(\[.*\])")
REQUEST_TTFT_RE = re.compile(
    r"^request=(?P<request>\S+) latency_ms=(?P<latency>[0-9.]+)", re.MULTILINE
)
SUFFIX_RE = re.compile(
    r"Hybrid suffix-only replay .*?local_prefix=(\d+) loaded_tokens=(\d+)"
)
P2_NATIVE_PREFIX_RE = re.compile(
    r"Hybrid P2 native replay request=(\S+) local_prefix=(\d+)"
)
KV_METRIC_RE = re.compile(r"KV Transfer metrics: (?P<body>.*)")
KV_FIELD_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[0-9.eE+-]+)")
GROUP_TRANSFER_RE = re.compile(
    r"KV offload group transfer: job=(?P<job>\d+) "
    r"direction=(?P<direction>[A-Za-z_]+) group=(?P<group>\d+) "
    r"state_type=(?P<state_type>\S+) blocks=(?P<blocks>\d+) "
    r"block_index=(?P<block_index>\d+) bytes=(?P<bytes>\d+)"
)
LMCACHE_RETRIEVE_RE = re.compile(
    r"LMCache H2D retrieve request=(?P<request>\S+) "
    r"group=(?P<group>\d+) state_type=(?P<state_type>\S+) "
    r"blocks=(?P<blocks>\d+) start=(?P<start>\d+) "
    r"end=(?P<end>\d+) tokens=(?P<tokens>\d+)"
)
LMCACHE_PREFIX_RE = re.compile(
    r"LMCache prefix match request=(?P<request>\S+) "
    r"local_gpu_tokens=(?P<local>\d+) cpu_tokens=(?P<cpu>\d+) "
    r"need_h2d_tokens=(?P<need>\d+)"
)
LMCACHE_SELECTED_RE = re.compile(
    r"LMCache selective policy transferred H2D object_group=(?P<group>\d+)"
)
LMCACHE_ACTUAL_RE = re.compile(
    r"LMCache actual H2D object_group=(?P<group>\d+)"
)
LMCACHE_SKIPPED_RE = re.compile(
    r"LMCache selective policy skipped H2D object_group=(?P<group>\d+)"
)
LMCACHE_OBJECT_GROUP_BYTES_RE = re.compile(
    r"LMCache H2D object_group_bytes group=(?P<group>\d+) "
    r"objects=(?P<objects>\d+) bytes=(?P<bytes>\d+)"
)
LMCACHE_LOOKUP_TIMING_RE = re.compile(
    r"LMCache lookup complete request=(?P<request>\S+) "
    r"hit_tokens=(?P<hit_tokens>\d+) lookup_ms=(?P<lookup_ms>[0-9.eE+-]+)"
)
LMCACHE_H2D_TIMING_RE = re.compile(
    r"LMCache H2D batch complete requests=(?P<requests>\d+) "
    r"wait_ms=(?P<wait_ms>[0-9.eE+-]+) "
    r"cuda_sync_ms=(?P<cuda_sync_ms>[0-9.eE+-]+) "
    r"total_ms=(?P<total_ms>[0-9.eE+-]+)"
)
LMCACHE_RETRIEVE_BREAKDOWN_RE = re.compile(
    r"\[req_id=(?P<request>\S+)\] Retrieve breakdown "
    r"process_tokens_ms=(?P<process>[0-9.eE+-]+) "
    r"broadcast_ms=(?P<broadcast>[0-9.eE+-]+) "
    r"to_gpu_enqueue_ms=(?P<enqueue>[0-9.eE+-]+) "
    r"total_ms=(?P<total>[0-9.eE+-]+)"
)
LMCACHE_MP_RETRIEVE_BREAKDOWN_RE = re.compile(
    r"\[req_id=(?P<request>\S+)\] LMCache MP retrieve breakdown "
    r"total_ms=(?P<total>[0-9.eE+-]+) "
    r"object_wait_ms=(?P<object_wait>[0-9.eE+-]+) "
    r"to_gpu_enqueue_ms=(?P<enqueue>[0-9.eE+-]+) bytes=(?P<bytes>\d+) "
    r"object_groups=(?P<groups>\d+) chunks=(?P<chunks>\d+)"
)
LMCACHE_EOSS_SUBMIT_RE = re.compile(
    r"LMCache EOSS submitted requests=(?P<requests>\d+) "
    r"tiles_per_request=(?P<tiles>\d+) window=(?P<window>\d+)"
)
LMCACHE_EOSS_TILE_READY_RE = re.compile(
    r"LMCache EOSS tile ready tile=(?P<tile>\d+) "
    r"requests=(?P<requests>\d+) wait_ms=(?P<wait_ms>[0-9.eE+-]+)"
)
VLLM_TIMING_METRIC_RE = re.compile(
    r"^vllm:(?P<name>request_prefill_time_seconds|request_queue_time_seconds|"
    r"request_inference_time_seconds|request_decode_time_seconds|"
    r"e2e_request_latency_seconds)_(?P<kind>sum|count)"
    r"(?:\{[^}]*\})?\s+(?P<value>[0-9.eE+-]+)\s*$"
)
LMCACHE_ADAPTIVE_REPLAY_RE = re.compile(
    r"LMCache adaptive recovery request=(?P<request>\S+) decision=replay "
    r"need_h2d_tokens=(?P<need>\d+) threshold_tokens=(?P<threshold>\d+)"
    r"(?: contention_threshold_tokens=(?P<contention_threshold>\d+)"
    r" pending_h2d=(?P<pending>\d+) "
    r"max_inflight_h2d=(?P<max_inflight>\d+) reason=(?P<reason>\S+))?"
)
LMCACHE_ADAPTIVE_COST_RE = re.compile(
    r"LMCache adaptive recovery request=(?P<request>\S+) "
    r"decision=(?P<decision>load|replay) need_h2d_tokens=(?P<need>\d+) "
    r"predicted_load_ms=(?P<load>[0-9.eE+-]+) "
    r"predicted_replay_ms=(?P<replay>[0-9.eE+-]+) "
    r"pending_h2d=(?P<pending>\d+) reason=cost"
)


def is_measured_request(request_id: str, request_ids: set[str] | None) -> bool:
    """Match API IDs to EngineCore IDs, which may append request metadata."""
    return request_ids is None or any(
        request_id == measured or request_id.startswith(f"{measured}-")
        for measured in request_ids
    )


def wait_ready(
    url: str,
    timeout_s: float,
    path: str = "/health",
    process: subprocess.Popen | None = None,
) -> None:
    session = requests.Session()
    session.trust_env = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before becoming ready (exit={process.returncode})"
            )
        try:
            response = session.get(f"{url}{path}", timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    if process is not None and process.poll() is not None:
        raise RuntimeError(
            f"vLLM exited before becoming ready (exit={process.returncode})"
        )
    raise TimeoutError(f"vLLM did not become ready within {timeout_s:.0f}s")


def wait_lmcache_ready(port: int, timeout_s: float) -> None:
    wait_ready(f"http://127.0.0.1:{port}", timeout_s, path="/healthcheck")


def read_vllm_timing_metrics(url: str) -> dict[str, float]:
    """Read cumulative request timing sums/counts from the vLLM endpoint."""
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(f"{url}/metrics", timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return {}
    metrics: dict[str, float] = {}
    for line in response.text.splitlines():
        match = VLLM_TIMING_METRIC_RE.match(line)
        if match is not None:
            metrics[f"{match.group('name')}_{match.group('kind')}"] = float(
                match.group("value")
            )
    return metrics


def diff_vllm_timing_metrics(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    """Return measured-phase deltas, omitting metrics absent from either snapshot."""
    return {
        key: after[key] - before[key]
        for key in after.keys() & before.keys()
        if after[key] >= before[key]
    }


def read_h2d_metrics(log_path: Path, skip_lines: int = 0) -> dict[str, float]:
    """Aggregate native connector metrics emitted by the vLLM server log."""
    totals: dict[str, float] = {}
    if not log_path.exists():
        return totals
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[skip_lines:]:
        match = KV_METRIC_RE.search(line)
        if match is None:
            continue
        for field in KV_FIELD_RE.finditer(match.group("body")):
            key = field.group("key")
            if key.endswith(("_total_bytes", "_total_time", "_total_queue_time")):
                totals[key] = totals.get(key, 0.0) + float(field.group("value"))
    return totals


def read_group_transfer_metrics(
    log_path: Path, skip_lines: int = 0
) -> dict[str, dict[str, int]]:
    """Aggregate the per-group transfer trace emitted by native offload."""
    totals = {"bytes": {}, "blocks": {}}
    if not log_path.exists():
        return totals
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[skip_lines:]:
        match = GROUP_TRANSFER_RE.search(line)
        if match is None or match.group("direction") != "CPU_to_GPU":
            continue
        label = f"group{match.group('group')}:{match.group('state_type')}"
        totals["bytes"][label] = totals["bytes"].get(label, 0) + int(
            match.group("bytes")
        )
        totals["blocks"][label] = totals["blocks"].get(label, 0) + int(
            match.group("blocks")
        )
    return totals


def read_lmcache_retrieve_metrics(
    log_path: Path,
    skip_lines: int = 0,
    request_ids: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Read group-level LMCache retrieve submissions from the server log."""
    totals = {"blocks": {}, "tokens": {}, "requests": set()}
    seen_groups: set[tuple[str, str]] = set()
    if not log_path.exists():
        return totals
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[skip_lines:]:
        match = LMCACHE_RETRIEVE_RE.search(line)
        if match is None:
            continue
        if not is_measured_request(match.group("request"), request_ids):
            continue
        key = (match.group("request"), match.group("group"))
        if key in seen_groups:
            continue
        seen_groups.add(key)
        label = f"group{match.group('group')}:{match.group('state_type')}"
        totals["blocks"][label] = totals["blocks"].get(label, 0) + int(
            match.group("blocks")
        )
        totals["tokens"][label] = totals["tokens"].get(label, 0) + int(
            match.group("tokens")
        )
        totals["requests"].add(match.group("request"))
    return totals


def read_lmcache_prefix_matches(
    log_path: Path,
    skip_lines: int = 0,
    request_ids: set[str] | None = None,
) -> list[dict[str, int | str]]:
    """Read per-request local/CPU prefix matching from the measured phase."""
    matches_by_request: dict[str, dict[str, int | str]] = {}
    if not log_path.exists():
        return []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[skip_lines:]:
        match = LMCACHE_PREFIX_RE.search(line)
        if match is None:
            continue
        if not is_measured_request(match.group("request"), request_ids):
            continue
        request = match.group("request")
        matches_by_request[request] = {
            "request": request,
            "local_gpu_tokens": int(match.group("local")),
            "cpu_tokens": int(match.group("cpu")),
            "need_h2d_tokens": int(match.group("need")),
        }
    return list(matches_by_request.values())


def read_lmcache_adaptive_replays(
    log_path: Path,
    skip_lines: int = 0,
    request_ids: set[str] | None = None,
) -> list[dict[str, int | str]]:
    """Read request-level Adaptive decisions from the measured phase."""
    if not log_path.exists():
        return []
    decisions = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
        skip_lines:
    ]:
        match = LMCACHE_ADAPTIVE_REPLAY_RE.search(line)
        if match is not None:
            if not is_measured_request(match.group("request"), request_ids):
                continue
            decisions.append(
                {
                    "request": match.group("request"),
                    "need_h2d_tokens": int(match.group("need")),
                    "threshold_tokens": int(match.group("threshold")),
                    **(
                        {
                            "pending_h2d": int(match.group("pending")),
                            "max_inflight_h2d": int(match.group("max_inflight")),
                            "contention_threshold_tokens": int(
                                match.group("contention_threshold")
                            ),
                            "reason": match.group("reason"),
                        }
                        if match.group("pending") is not None
                        else {}
                    ),
                }
            )
    return decisions


def read_lmcache_adaptive_decisions(
    log_path: Path,
    skip_lines: int = 0,
    request_ids: set[str] | None = None,
) -> list[dict[str, int | float | str]]:
    """Read cost-model Load/Replay decisions from the measured phase."""
    if not log_path.exists():
        return []
    decisions = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
        skip_lines:
    ]:
        match = LMCACHE_ADAPTIVE_COST_RE.search(line)
        if match is None or not is_measured_request(match.group("request"), request_ids):
            continue
        decisions.append(
            {
                "request": match.group("request"),
                "decision": match.group("decision"),
                "need_h2d_tokens": int(match.group("need")),
                "predicted_load_ms": float(match.group("load")),
                "predicted_replay_ms": float(match.group("replay")),
                "pending_h2d": int(match.group("pending")),
                "reason": "cost",
            }
        )
    return decisions


def read_lmcache_policy_metrics(
    log_path: Path, skip_lines: int = 0
) -> dict[str, dict[str, int]]:
    """Count actual LMCache object-group H2D decisions from the worker log."""
    totals = {"transferred": {}, "skipped": {}, "bytes": {}}
    if not log_path.exists():
        return totals
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[skip_lines:]:
        for pattern, key in (
            (LMCACHE_ACTUAL_RE, "transferred"),
            (LMCACHE_SELECTED_RE, "transferred"),
            (LMCACHE_SKIPPED_RE, "skipped"),
        ):
            match = pattern.search(line)
            if match is not None:
                group = match.group("group")
                totals[key][group] = totals[key].get(group, 0) + 1
        bytes_match = LMCACHE_OBJECT_GROUP_BYTES_RE.search(line)
        if bytes_match is not None:
            group = bytes_match.group("group")
            totals["bytes"][group] = totals["bytes"].get(group, 0) + int(
                bytes_match.group("bytes")
            )
    return totals


def read_lmcache_timing_metrics(
    log_path: Path,
    skip_lines: int = 0,
    request_ids: set[str] | None = None,
) -> dict:
    """Read request lookup and strict H2D timing traces from the vLLM log."""
    lookup_ms: list[float] = []
    h2d_wait_ms: list[float] = []
    h2d_cuda_sync_ms: list[float] = []
    h2d_total_ms: list[float] = []
    h2d_batch_requests: list[int] = []
    if not log_path.exists():
        return {
            "lookup_ms": lookup_ms,
            "h2d_wait_ms": h2d_wait_ms,
            "h2d_cuda_sync_ms": h2d_cuda_sync_ms,
            "h2d_total_ms": h2d_total_ms,
            "h2d_batch_requests": h2d_batch_requests,
        }
    for line in log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()[skip_lines:]:
        lookup = LMCACHE_LOOKUP_TIMING_RE.search(line)
        if lookup is not None:
            if not is_measured_request(lookup.group("request"), request_ids):
                continue
            lookup_ms.append(float(lookup.group("lookup_ms")))
        h2d = LMCACHE_H2D_TIMING_RE.search(line)
        if h2d is not None:
            h2d_wait_ms.append(float(h2d.group("wait_ms")))
            h2d_cuda_sync_ms.append(float(h2d.group("cuda_sync_ms")))
            h2d_total_ms.append(float(h2d.group("total_ms")))
            h2d_batch_requests.append(int(h2d.group("requests")))
    return {
        "lookup_ms": lookup_ms,
        "h2d_wait_ms": h2d_wait_ms,
        "h2d_cuda_sync_ms": h2d_cuda_sync_ms,
        "h2d_total_ms": h2d_total_ms,
        "h2d_batch_requests": h2d_batch_requests,
    }


def read_lmcache_retrieve_breakdown(
    log_path: Path,
    skip_lines: int = 0,
    request_ids: set[str] | None = None,
) -> list[dict[str, float | str]]:
    """Read LMCache CPU/object/transfer enqueue timing per retrieve."""
    if not log_path.exists():
        return []
    result = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
        skip_lines:
    ]:
        match = LMCACHE_RETRIEVE_BREAKDOWN_RE.search(line)
        if match is None:
            match = LMCACHE_MP_RETRIEVE_BREAKDOWN_RE.search(line)
            if match is None:
                continue
            if not is_measured_request(match.group("request"), request_ids):
                continue
            result.append(
                {
                    "request": match.group("request"),
                    "total_ms": float(match.group("total")),
                    "object_wait_ms": float(match.group("object_wait")),
                    "to_gpu_enqueue_ms": float(match.group("enqueue")),
                    "bytes": int(match.group("bytes")),
                    "object_groups": int(match.group("groups")),
                    "chunks": int(match.group("chunks")),
                }
            )
            continue
        if not is_measured_request(match.group("request"), request_ids):
            continue
        result.append(
            {
                "request": match.group("request"),
                "process_tokens_ms": float(match.group("process")),
                "broadcast_ms": float(match.group("broadcast")),
                "to_gpu_enqueue_ms": float(match.group("enqueue")),
                "total_ms": float(match.group("total")),
            }
        )
    return result


def read_lmcache_eoss_metrics(log_path: Path, skip_lines: int = 0) -> dict:
    """Read per-tile EOSS submission and readiness traces."""
    if not log_path.exists():
        return {"submissions": [], "tile_wait_ms": {}}
    submissions = []
    tile_wait_ms: dict[str, list[float]] = {}
    for line in log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()[skip_lines:]:
        submitted = LMCACHE_EOSS_SUBMIT_RE.search(line)
        if submitted is not None:
            submissions.append(
                {
                    "requests": int(submitted.group("requests")),
                    "tiles_per_request": int(submitted.group("tiles")),
                    "window": int(submitted.group("window")),
                }
            )
        ready = LMCACHE_EOSS_TILE_READY_RE.search(line)
        if ready is not None:
            tile_wait_ms.setdefault(ready.group("tile"), []).append(
                float(ready.group("wait_ms"))
            )
    return {"submissions": submissions, "tile_wait_ms": tile_wait_ms}


def remember_process_group(process: subprocess.Popen) -> None:
    """Remember the session/process-group created for a child process.

    ``Popen`` can already report an exited leader by the time cleanup starts,
    while EngineCore descendants are still alive.  In that case looking up
    the PGID from ``process.pid`` fails, so retain it immediately after spawn.
    All callers use ``start_new_session=True``; therefore the leader PID is
    also a private process-group ID.
    """
    try:
        process._process_group_id = os.getpgid(process.pid)  # type: ignore[attr-defined]
    except ProcessLookupError:
        process._process_group_id = process.pid  # type: ignore[attr-defined]


def stop_server(process: subprocess.Popen) -> None:
    # vLLM starts EngineCore and multiprocessing helper processes below the
    # API server.  The server is launched with ``start_new_session=True``, so
    # signal the whole process group; signalling only the API PID can leave a
    # CUDA-owning EngineCore orphaned and keep most GPU memory allocated.
    process_group = getattr(process, "_process_group_id", None)
    if process_group is None:
        try:
            process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            # The leader may be gone while EngineCore is still in the group.
            process_group = process.pid

    def signal_group(sig: signal.Signals) -> None:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            # The group is already gone.  Do not signal an unrelated PID.
            pass

    def group_alive() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        return True

    signal_group(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_group(signal.SIGKILL)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # A child can remain in uninterruptible kernel sleep.  The
                # group kill is still the correct last resort; report the
                # timeout to the caller instead of silently continuing.
                raise
    finally:
        # The API leader can exit cleanly while EngineCore survives.  Always
        # reap any remaining descendants in the private group before the next
        # cell starts, otherwise they retain CUDA contexts and GPU memory.
        signal_group(signal.SIGKILL)
        # Do not start another GPU cell if a child is stuck in D-state (or
        # otherwise ignored the kill).  Continuing would overlap CUDA
        # contexts and recreate the original "GPU memory is still allocated"
        # failure; the caller must stop and repair the host instead.
        deadline = time.monotonic() + 5
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        group_survived = group_alive()
        log_file = getattr(process, "_lmcache_log_file", None)
        if log_file is not None:
            log_file.close()
        if group_survived:
            raise RuntimeError(
                f"process group {process_group} survived SIGKILL; "
                "refusing to start another GPU cell"
            )


def start_lmcache_server(
    config: Path,
    port: int,
    http_port: int,
    chunk_size: int,
    eviction_policy: str = "LRU",
    slru_protected_ratio: float = 0.8,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Start the real LMCache MP server used by the hybrid backend."""
    values = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in config.read_text(encoding="utf-8").splitlines()
        if ":" in line and not line.lstrip().startswith("#")
    }
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
        "--chunk-size",
        str(chunk_size),
        "--l1-size-gb",
        values.get("max_local_cpu_size", "32"),
        "--eviction-policy",
        eviction_policy,
        "--slru-protected-ratio",
        str(slru_protected_ratio),
        "--separate-object-groups",
        "--disable-observability",
    ]
    server_env = os.environ.copy() if env is None else env.copy()
    server_env["PYTHONHASHSEED"] = "0"
    server_log = Path(f"/tmp/lmcache_mp_{port}.log").open("w")
    process = subprocess.Popen(
        command,
        env=server_env,
        cwd=ROOT,
        # Keep the server log drained and inspectable.  An unread PIPE can
        # fill during startup and block worker registration.
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    remember_process_group(process)
    process._lmcache_log_file = server_log  # type: ignore[attr-defined]
    return process


def run_e2e(
    mode: str,
    server: str,
    concurrency: int,
    requests_count: int,
    *,
    shared_prefix_tokens: int = 0,
    suffix_tokens: int = 0,
    distinct_suffix_prefix_tokens: int = 0,
    prefix_only: bool = False,
    output_tokens: int = 1024,
    start_index: int = 0,
    warmup_concurrency: int = 1,
    warmup_settle_seconds: float = 1.0,
    request_timeout_s: float = 180.0,
) -> str:
    command = [
        sys.executable,
        str(E2E),
        mode,
        "--server",
        server,
        "--requests",
        str(requests_count),
        "--concurrency",
        str(concurrency),
        "--warmup-concurrency",
        str(warmup_concurrency),
        "--start-index",
        str(start_index),
        "--min-tokens",
        "784",
        "--max-tokens",
        "900",
        "--output-tokens",
        str(output_tokens),
        "--model",
        "Qwen3.5-9B",
        "--tokenizer",
        TOKENIZER,
        "--dataset-path",
        DATASET,
        "--warmup-settle-seconds",
        str(warmup_settle_seconds),
        "--request-timeout",
        str(request_timeout_s),
    ]
    if shared_prefix_tokens or suffix_tokens:
        command += [
            "--shared-prefix-tokens",
            str(shared_prefix_tokens),
            "--suffix-tokens",
            str(suffix_tokens),
        ]
        if distinct_suffix_prefix_tokens:
            command += [
                "--distinct-suffix-prefix-tokens",
                str(distinct_suffix_prefix_tokens),
            ]
        if prefix_only:
            command.append("--prefix-only")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"e2e {mode}/C{concurrency} failed (exit={exc.returncode})\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc
    return result.stdout


def reset_local_prefix_cache(server: str, timeout_s: float) -> None:
    """Clear vLLM's GPU prefix hashes while leaving the external tier intact."""
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        f"{server}/reset_prefix_cache", timeout=timeout_s
    )
    response.raise_for_status()


def request_ttft_ms(output: str) -> dict[str, float]:
    """Read the per-request streamed first-token latency emitted by E2E."""
    return {
        match.group("request"): float(match.group("latency"))
        for match in REQUEST_TTFT_RE.finditer(output)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/qwen35_lmcache_independent_cells.jsonl"),
    )
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=72)
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=1,
        help=(
            "Repeat the sequential warmup workload before measurement; useful "
            "when the first pass only stores a partial LMCache chunk set."
        ),
    )
    parser.add_argument(
        "--activation-checkpoint",
        action="store_true",
        help="Enable hybrid activation checkpoints for P3/P4 replay.",
    )
    parser.add_argument(
        "--activation-max-bytes",
        type=int,
        default=48 * 1024**3,
        help="Host-memory bound for activation checkpoints (0=unbounded).",
    )
    parser.add_argument(
        "--suffix-only",
        action="store_true",
        help=(
            "Keep a local prefix hit and replay only the externally missing "
            "suffix for mixed P3/P4 policies. Activation checkpoints are "
            "optional and should be disabled for a fair KV-only comparison."
        ),
    )
    parser.add_argument(
        "--retain-shared-prefix",
        action="store_true",
        help=(
            "After warmup, touch the shared prefix once more so it remains a "
            "local GPU prefix hit while unique suffixes are evicted."
        ),
    )
    parser.add_argument(
        "--evict-gpu-cache",
        action="store_true",
        help=(
            "Fill the GPU prefix cache with distinct partial prompts before "
            "retaining the shared prefix and measuring recovery."
        ),
    )
    parser.add_argument(
        "--reset-local-prefix-cache",
        action="store_true",
        help=(
            "Use vLLM's reset endpoint after CPU-cache population, then restore "
            "only the shared GPU prefix. Requires a suffix-only partial workload."
        ),
    )
    parser.add_argument(
        "--evict-requests",
        type=int,
        default=64,
        help="Number of short-output filler prompts used for GPU-cache eviction.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900,
        help="Server startup timeout; Qwen3.5-9B may need several minutes to load.",
    )
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=("auto", "eager", "prefetch"),
        default="auto",
        help=(
            "Checkpoint loading strategy; auto leaves vLLM prefetch disabled "
            "on overlayfs."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="vLLM GPU memory utilization target (default: 0.95).",
    )
    parser.add_argument(
        "--backend",
        choices=("native", "lmcache"),
        default="lmcache",
        help="KV backend for load cells; LMCache is the main experiment path.",
    )
    parser.add_argument(
        "--lmcache-config",
        type=Path,
        default=None,
        help="Existing LMCache YAML. If omitted, a local CPU/H2D config is created.",
    )
    parser.add_argument(
        "--lmcache-kv-gb", type=float, default=24.0,
        help="LMCache local CPU KV pool size when creating a config.",
    )
    parser.add_argument(
        "--lmcache-hidden-state-gb", type=float, default=8.0,
        help="LMCache hidden-state pool size when activation checkpoints are enabled.",
    )
    parser.add_argument(
        "--native-kv-gb",
        type=float,
        default=24.0,
        help="Native CPU offload capacity in GiB for the H2D baseline.",
    )
    parser.add_argument(
        "--lmcache-chunk-size",
        type=int,
        default=528,
        help="LMCache/vLLM unified block size (528 for Qwen3.5-9B align mode).",
    )
    parser.add_argument(
        "--lmcache-eviction-policy",
        choices=("LRU", "SLRU"),
        default="LRU",
        help="LMCache MP local-CPU eviction policy.",
    )
    parser.add_argument(
        "--slru-protected-ratio",
        type=float,
        default=0.8,
        help="Protected-segment fraction when LMCache eviction policy is SLRU.",
    )
    parser.add_argument("--concurrencies", default="1,16,32")
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Use a shared-prefix plus unique-suffix workload instead of 784-900 tokens.",
    )
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=0,
        help="Unique suffix length for the partial-prefix workload.",
    )
    parser.add_argument(
        "--expected-missing-pages",
        type=int,
        default=None,
        help="Require every measured Load request to recover exactly this many pages.",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=1024,
        help="Fixed completion length for every request.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Per-request HTTP/stream timeout in seconds.",
    )
    parser.add_argument(
        "--adaptive-replay-below-tokens",
        type=int,
        default=0,
        help=(
            "For Adaptive, replay rather than H2D-load a CPU hit whose missing "
            "interval is at most this many tokens (0 keeps all-load behavior)."
        ),
    )
    parser.add_argument(
        "--adaptive-max-inflight-h2d",
        type=int,
        default=0,
        help=(
            "For Adaptive, replay when this many other requests are already "
            "waiting for H2D (0 disables the contention proxy)."
        ),
    )
    parser.add_argument(
        "--adaptive-contention-replay-below-tokens",
        type=int,
        default=0,
        help=(
            "For Adaptive contention decisions, only replay missing intervals "
            "at most this size (0 disables contention-triggered replay)."
        ),
    )
    parser.add_argument(
        "--adaptive-h2d-gbps",
        type=float,
        default=0.0,
        help=(
            "Adaptive cost model H2D bandwidth in GB/s; paired with the replay "
            "cost below to enable dynamic TTFT Load/Replay selection."
        ),
    )
    parser.add_argument(
        "--adaptive-replay-ms-per-token",
        type=float,
        default=0.0,
        help="Adaptive cost model replay prefill milliseconds per prompt token.",
    )
    parser.add_argument(
        "--adaptive-replay-base-ms",
        type=float,
        default=0.0,
        help="Fixed Adaptive replay cost term in milliseconds.",
    )
    parser.add_argument(
        "--adaptive-use-terminal-state",
        action="store_true",
        help="Use P5 terminal Mamba materialization when Adaptive chooses Load.",
    )
    parser.add_argument(
        "--hybrid-execution-tile-size",
        type=int,
        default=0,
        help=(
            "Split Hybrid LMCache objects by this many decoder layers in "
            "execution order (0 keeps the state-family layout)."
        ),
    )
    parser.add_argument(
        "--hybrid-cache-slice-size",
        type=int,
        default=0,
        help=(
            "Split only LMCache CPU objects by decoder-layer tiles while "
            "preserving vLLM physical KV/page groups (0 disables it)."
        ),
    )
    parser.add_argument(
        "--hybrid-eoss",
        action="store_true",
        help=(
            "Use execution-order state streaming: submit one LMCache retrieve "
            "per cache-plane tile and wait only when the decoder enters it."
        ),
    )
    parser.add_argument(
        "--hybrid-eoss-window",
        type=int,
        default=2,
        help="Number of execution-order cache tiles initially kept in H2D flight.",
    )
    parser.add_argument(
        "--warmup-output-tokens",
        type=int,
        default=1,
        help="Completion length used only by warmup/prefix-retain requests.",
    )
    parser.add_argument(
        "--warmup-settle-seconds",
        type=float,
        default=3.0,
        help="Time to wait after warmup so asynchronous CPU stores become visible.",
    )
    parser.add_argument(
        "--p5-serialize-terminal",
        action="store_true",
        help="Serialize P5 Full/terminal H2D materialization per request.",
    )
    parser.add_argument(
        "--serialize-all-load",
        action="store_true",
        help="Serialize P1 All Load H2D materialization per request.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many deterministically selected ShareGPT prefixes.",
    )
    parser.add_argument(
        "--policies",
        default="P1,P2,P3,P4",
        help="Comma-separated policy names to run.",
    )
    args = parser.parse_args()
    if bool(args.shared_prefix_tokens) != bool(args.suffix_tokens):
        raise ValueError(
            "--shared-prefix-tokens and --suffix-tokens must be provided together"
        )
    if args.reset_local_prefix_cache and not args.shared_prefix_tokens:
        raise ValueError(
            "--reset-local-prefix-cache requires a partial-prefix workload"
        )
    if args.reset_local_prefix_cache and args.evict_gpu_cache:
        raise ValueError(
            "choose either --reset-local-prefix-cache or --evict-gpu-cache"
        )
    if args.output_tokens < 1:
        raise ValueError("--output-tokens must be positive")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.adaptive_replay_below_tokens < 0:
        raise ValueError("--adaptive-replay-below-tokens must be non-negative")
    if args.adaptive_max_inflight_h2d < 0:
        raise ValueError("--adaptive-max-inflight-h2d must be non-negative")
    if args.adaptive_contention_replay_below_tokens < 0:
        raise ValueError(
            "--adaptive-contention-replay-below-tokens must be non-negative"
        )
    if args.adaptive_h2d_gbps < 0 or args.adaptive_replay_ms_per_token < 0:
        raise ValueError("Adaptive cost parameters must be non-negative")
    if args.adaptive_replay_base_ms < 0:
        raise ValueError("--adaptive-replay-base-ms must be non-negative")
    if args.hybrid_execution_tile_size < 0:
        raise ValueError("--hybrid-execution-tile-size must be non-negative")
    if args.hybrid_cache_slice_size < 0:
        raise ValueError("--hybrid-cache-slice-size must be non-negative")
    if args.hybrid_execution_tile_size and args.hybrid_cache_slice_size:
        raise ValueError("choose only one Hybrid tile implementation per run")
    if args.hybrid_eoss and args.backend != "lmcache":
        raise ValueError("--hybrid-eoss currently requires --backend lmcache")
    if args.hybrid_eoss and not args.hybrid_cache_slice_size:
        raise ValueError(
            "--hybrid-eoss requires --hybrid-cache-slice-size so it does not "
            "change vLLM's physical KV/page groups"
        )
    if args.hybrid_eoss_window < 1:
        raise ValueError("--hybrid-eoss-window must be positive")
    if args.warmup_output_tokens < 1:
        raise ValueError("--warmup-output-tokens must be positive")
    if args.warmup_repeats < 1:
        raise ValueError("--warmup-repeats must be positive")
    if args.warmup_settle_seconds < 0:
        raise ValueError("--warmup-settle-seconds must be non-negative")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.expected_missing_pages is not None and args.expected_missing_pages < 1:
        raise ValueError("--expected-missing-pages must be positive")
    partial_tokens = args.shared_prefix_tokens + args.suffix_tokens
    # Token-boundary concatenation can add a small number of tokens when the
    # decoded ShareGPT prefix and suffix are re-tokenized.  Keep a safety
    # margin (and a 1K floor) so short partial workloads do not turn into
    # HTTP 400 "prompt too long" failures merely because of that boundary.
    max_model_len = (
        max(2048, ((partial_tokens + args.output_tokens + 127) // 128) * 128)
        if partial_tokens
        else 2048
    )
    if args.backend == "lmcache":
        # Qwen3.5's aligned Mamba page is 528 tokens in the current vLLM
        # runtime. LMCache hybrid recovery requires exactly one page per
        # prefill step: block_size <= max_num_batched_tokens < 2*block_size.
        max_num_batched_tokens = 528
    else:
        max_num_batched_tokens = (
            max(2 * partial_tokens - 1, 2 * args.lmcache_chunk_size - 1)
            if partial_tokens
            else 2 * args.lmcache_chunk_size - 1
        )
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if not concurrencies or min(concurrencies) < 1:
        raise ValueError("--concurrencies must contain positive integers")
    if args.requests < max(concurrencies):
        raise ValueError(
            "--requests must be at least the maximum requested concurrency; "
            "otherwise the measured concurrency is silently lower"
        )
    selected_policies = [
        value.strip() for value in args.policies.split(",") if value.strip()
    ]
    if args.hybrid_eoss and set(selected_policies) != {"P1"}:
        raise ValueError(
            "--hybrid-eoss currently validates all-load P1 only; P3/P4 "
            "remain a separate correctness task"
        )
    if "P5" in selected_policies and args.backend != "lmcache":
        raise ValueError(
            "P5 terminal-state materialization currently requires --backend lmcache"
        )
    if "P5" in selected_policies and args.hybrid_cache_slice_size:
        raise ValueError("P5 requires --hybrid-cache-slice-size 0")
    unknown_policies = set(selected_policies) - set(POLICIES)
    if unknown_policies:
        raise ValueError(f"unknown policies: {sorted(unknown_policies)}")
    if (
        args.suffix_only
        and args.shared_prefix_tokens
        and any(policy in ("P3", "P4") for policy in selected_policies)
        and not args.retain_shared_prefix
    ):
        raise ValueError(
            "--suffix-only with P3/P4 requires --retain-shared-prefix so the "
            "replay group has a verified local GPU prefix"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generated_lmcache_config = None
    if args.backend == "lmcache" and args.lmcache_config is None:
        config_fd, config_path = tempfile.mkstemp(
            prefix="vllm_lmcache_hybrid_", suffix=".yaml"
        )
        os.close(config_fd)
        generated_lmcache_config = Path(config_path)
        generated_lmcache_config.write_text(
            "\n".join(
                [
                    f"chunk_size: {args.lmcache_chunk_size}",
                    "local_device: cpu",
                    "local_cpu: true",
                    f"max_local_cpu_size: {args.lmcache_kv_gb}",
                    "enable_hidden_state_cache: "
                    f"{'true' if args.activation_checkpoint else 'false'}",
                    *(
                        [
                            "max_hidden_state_cpu_size: "
                            f"{args.lmcache_hidden_state_gb}"
                        ]
                        if args.activation_checkpoint
                        else []
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    lmcache_config = args.lmcache_config or generated_lmcache_config

    for policy in selected_policies:
        policy_env, use_offload = POLICIES[policy]
        # P2 is the no-offload/native replay control.  When the real LMCache
        # MP connector is selected, keep the connector enabled but explicitly
        # suppress every external group load so the residual is recomputed.
        if args.backend == "lmcache" and policy == "P2":
            policy_env = "all_replay"
        for concurrency in concurrencies:
            server = f"http://127.0.0.1:{args.port}"
            # Derive LMCache control ports from the vLLM port instead of using
            # the old global 5555/8080 + concurrency convention.  A previous
            # interrupted cell can leave its LMCache process alive; reusing
            # that fixed port makes the next vLLM process attach to stale
            # state and hang during KV-cache registration.
            lmcache_port = args.port + 10000 + concurrency
            lmcache_http_port = args.port + 20000 + concurrency
            lmcache_process = None
            env = os.environ.copy()
            env.update(
                {
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
                    # Keep cross-policy correctness checks reproducible.  The
                    # Qwen3.5 GDN/align path otherwise uses different CUDA
                    # reduction choices across fresh processes/batch shapes.
                    "VLLM_FLOAT32_MATMUL_PRECISION": "highest",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "VLLM_MOONCAKE_HYBRID_POLICY": policy_env,
                    # Qwen3.5-9B's aligned hybrid layout currently exposes
                    # three Mamba/GDN groups followed by one FullAttention
                    # group.  LMCache's worker uses this only to suppress the
                    # non-selected H2D object groups for P3/P4.
                    "VLLM_MOONCAKE_HYBRID_GROUP_STATE_TYPES": (
                        "MambaSpec,MambaSpec,MambaSpec,FullAttentionSpec"
                    ),
                    "VLLM_MOONCAKE_HYBRID_STATE_OBJECT_GROUPS": "1",
                    "VLLM_MOONCAKE_HYBRID_ACTIVATION_CHECKPOINT": (
                        "1" if args.activation_checkpoint else "0"
                    ),
                    "VLLM_MOONCAKE_HYBRID_ACTIVATION_MAX_BYTES": str(
                        args.activation_max_bytes
                    ),
                    "VLLM_MOONCAKE_HYBRID_SUFFIX_ONLY": (
                        "1" if args.suffix_only else "0"
                    ),
                    "VLLM_MOONCAKE_ADAPTIVE_REPLAY_BELOW_TOKENS": str(
                        args.adaptive_replay_below_tokens
                    ),
                    "VLLM_MOONCAKE_ADAPTIVE_MAX_INFLIGHT_H2D": str(
                        args.adaptive_max_inflight_h2d
                    ),
                    "VLLM_MOONCAKE_ADAPTIVE_CONTENTION_REPLAY_BELOW_TOKENS": str(
                        args.adaptive_contention_replay_below_tokens
                    ),
                    "VLLM_MOONCAKE_ADAPTIVE_H2D_GBPS": str(
                        args.adaptive_h2d_gbps
                    ),
                    "VLLM_MOONCAKE_ADAPTIVE_REPLAY_MS_PER_TOKEN": str(
                        args.adaptive_replay_ms_per_token
                    ),
                    "VLLM_MOONCAKE_ADAPTIVE_REPLAY_BASE_MS": str(
                        args.adaptive_replay_base_ms
                    ),
                    "VLLM_MOONCAKE_ADAPTIVE_USE_TERMINAL_STATE": (
                        "1" if args.adaptive_use_terminal_state else "0"
                    ),
                    "VLLM_MOONCAKE_HYBRID_EXECUTION_TILE_SIZE": str(
                        args.hybrid_execution_tile_size
                    ),
                    "VLLM_MOONCAKE_HYBRID_CACHE_SLICE_SIZE": str(
                        args.hybrid_cache_slice_size
                    ),
                    "LMCACHE_MP_EOSS": "1" if args.hybrid_eoss else "0",
                    "LMCACHE_MP_EOSS_WINDOW": str(args.hybrid_eoss_window),
                    # Emit connector transfer counters frequently enough that
                    # short TTFT cells still expose the actual H2D operation.
                    "VLLM_LOG_STATS_INTERVAL": "1",
                    "VLLM_SERVER_DEV_MODE": (
                        "1" if args.reset_local_prefix_cache else "0"
                    ),
                    # Ensure the measured all-load cell is correctness-safe:
                    # do not consume Hybrid state before its H2D future ends.
                    "LMCACHE_MP_STRICT_LAYER_LOAD": (
                        "0" if args.hybrid_eoss else "1"
                    ),
                    "LMCACHE_MP_SERIALIZE_TERMINAL": (
                        "1"
                        if args.p5_serialize_terminal
                        or env.get("LMCACHE_MP_SERIALIZE_TERMINAL") == "1"
                        else "0"
                    ),
                    "LMCACHE_MP_SERIALIZE_ALL_LOAD": (
                        "1"
                        if args.serialize_all_load
                        or env.get("LMCACHE_MP_SERIALIZE_ALL_LOAD") == "1"
                        else "0"
                    ),
                }
            )
            if args.backend == "lmcache":
                assert lmcache_config is not None
                env["LMCACHE_CONFIG_FILE"] = str(lmcache_config)
            command = [
                str(Path(sys.executable).with_name("vllm")),
                "serve",
                MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                str(max_model_len),
                "--enforce-eager",
                "--enable-prefix-caching",
                "--mamba-cache-mode",
                "align",
                "--max-num-batched-tokens",
                str(max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--seed",
                "0",
                "--served-model-name",
                "Qwen3.5-9B",
            ]
            if args.safetensors_load_strategy != "auto":
                command += [
                    "--safetensors-load-strategy",
                    args.safetensors_load_strategy,
                ]
            if args.backend == "lmcache":
                lmcache_process = start_lmcache_server(
                    lmcache_config,
                    port=lmcache_port,
                    http_port=lmcache_http_port,
                    chunk_size=args.lmcache_chunk_size,
                    eviction_policy=args.lmcache_eviction_policy,
                    slru_protected_ratio=args.slru_protected_ratio,
                    env=env,
                )
                wait_lmcache_ready(lmcache_http_port, args.startup_timeout)
                command += [
                    "--kv-transfer-config",
                    json.dumps(
                        {
                            "kv_connector": "LMCacheMPConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "lmcache.mp.host": "tcp://127.0.0.1",
                                "lmcache.mp.port": lmcache_port,
                            },
                        }
                    ),
                ]
            elif use_offload:
                command += [
                    "--kv-offloading-size",
                    str(args.native_kv_gb),
                    "--kv-offloading-backend",
                    "native",
                ]
            log_path = Path(
                f"/tmp/{args.backend}_independent_{policy}_c{concurrency}_"
                f"port{args.port}.log"
            )
            print(f"starting policy={policy} concurrency={concurrency}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                remember_process_group(process)
            try:
                wait_ready(server, args.startup_timeout, process=process)
                for warmup_repeat in range(args.warmup_repeats):
                    run_e2e(
                        "warmup",
                        server,
                        1,
                        args.warmup_requests,
                        shared_prefix_tokens=args.shared_prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        distinct_suffix_prefix_tokens=args.lmcache_chunk_size,
                        output_tokens=args.warmup_output_tokens,
                        start_index=args.start_index,
                        warmup_settle_seconds=args.warmup_settle_seconds,
                        request_timeout_s=args.request_timeout,
                    )
                if args.evict_gpu_cache:
                    if not args.shared_prefix_tokens or not args.suffix_tokens:
                        raise ValueError(
                            "--evict-gpu-cache requires a partial-prefix workload"
                        )
                    run_e2e(
                        "warmup",
                        server,
                        min(16, args.evict_requests),
                        args.evict_requests,
                        shared_prefix_tokens=args.shared_prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        distinct_suffix_prefix_tokens=args.lmcache_chunk_size,
                        output_tokens=1,
                        start_index=args.start_index + args.requests + 1,
                        warmup_concurrency=min(16, args.evict_requests),
                        warmup_settle_seconds=args.warmup_settle_seconds,
                        request_timeout_s=args.request_timeout,
                    )
                if args.reset_local_prefix_cache:
                    reset_local_prefix_cache(server, args.request_timeout)
                if args.retain_shared_prefix and args.shared_prefix_tokens:
                    run_e2e(
                        "warmup",
                        server,
                        1,
                        1,
                        shared_prefix_tokens=args.shared_prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        distinct_suffix_prefix_tokens=args.lmcache_chunk_size,
                        prefix_only=True,
                        output_tokens=args.warmup_output_tokens,
                        start_index=args.start_index,
                        warmup_settle_seconds=args.warmup_settle_seconds,
                        request_timeout_s=args.request_timeout,
                    )
                # Report only the measured load phase, not warmup, eviction,
                # or the deliberate shared-prefix retain restore.
                load_log_lines = len(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                )
                lmcache_log_path = Path(f"/tmp/lmcache_mp_{lmcache_port}.log")
                lmcache_load_log_lines = (
                    len(
                        lmcache_log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()
                    )
                    if args.backend == "lmcache" and lmcache_log_path.exists()
                    else 0
                )
                vllm_timing_before = read_vllm_timing_metrics(server)
                output = run_e2e(
                    "load",
                    server,
                    concurrency,
                    args.requests,
                    shared_prefix_tokens=args.shared_prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    distinct_suffix_prefix_tokens=args.lmcache_chunk_size,
                    output_tokens=args.output_tokens,
                    start_index=args.start_index,
                    warmup_settle_seconds=args.warmup_settle_seconds,
                    request_timeout_s=args.request_timeout,
                )
                vllm_timing_after = read_vllm_timing_metrics(server)
                vllm_timing = diff_vllm_timing_metrics(
                    vllm_timing_before, vllm_timing_after
                )
                match = SUMMARY_RE.search(output)
                if match is None:
                    raise RuntimeError(
                        f"could not parse TTFT summary for {policy}/C{concurrency}: "
                        f"{output}"
                    )
                tpot_match = TPOT_SUMMARY_RE.search(output)
                if tpot_match is None:
                    raise RuntimeError(
                        f"could not parse TPOT summary for {policy}/C{concurrency}: "
                        f"{output}"
                    )
                signature_match = OUTPUT_SIGNATURE_RE.search(output)
                if signature_match is None:
                    raise RuntimeError(
                        f"could not parse first-token signature for {policy}/C{concurrency}"
                    )
                first_tokens_match = FIRST_TOKENS_RE.search(output)
                if first_tokens_match is None:
                    raise RuntimeError(
                        f"could not parse first-token map for {policy}/C{concurrency}"
                    )
                first_tokens = json.loads(first_tokens_match.group(1))
                if len(first_tokens) != args.requests:
                    raise RuntimeError(
                        f"{policy}/C{concurrency} returned {len(first_tokens)} unique "
                        f"request tokens for {args.requests} requests"
                    )
                request_ttfts = request_ttft_ms(output)
                if len(request_ttfts) != args.requests:
                    raise RuntimeError(
                        f"{policy}/C{concurrency} returned {len(request_ttfts)} "
                        f"per-request TTFT values for {args.requests} requests"
                    )
                completion_texts_match = COMPLETION_TEXTS_RE.search(output)
                if completion_texts_match is None:
                    raise RuntimeError(
                        f"could not parse completion texts for {policy}/C{concurrency}"
                    )
                completion_texts = json.loads(completion_texts_match.group(1))
                if len(completion_texts) != args.requests:
                    raise RuntimeError(
                        f"{policy}/C{concurrency} returned "
                        f"{len(completion_texts)} completion texts for "
                        f"{args.requests} requests"
                    )
                backend_request_ids_match = BACKEND_REQUEST_IDS_RE.search(output)
                if backend_request_ids_match is None:
                    raise RuntimeError(
                        f"could not parse backend request IDs for {policy}/C{concurrency}"
                    )
                measured_request_ids = set(
                    json.loads(backend_request_ids_match.group(1))
                )
                if len(measured_request_ids) != args.requests:
                    raise RuntimeError(
                        f"{policy}/C{concurrency} returned "
                        f"{len(measured_request_ids)} backend request IDs for "
                        f"{args.requests} requests"
                    )
                # The suffix-only transition is logged by the engine process.
                # Stop it before reading the file so buffered logger output is
                # flushed and the validation fields cannot silently disappear.
                stop_server(process)
                process = None
                h2d_metrics = read_h2d_metrics(log_path, load_log_lines)
                group_metrics = read_group_transfer_metrics(log_path, load_log_lines)
                lmcache_metrics = read_lmcache_retrieve_metrics(
                    log_path, load_log_lines, measured_request_ids
                )
                lmcache_prefix_matches = read_lmcache_prefix_matches(
                    log_path, load_log_lines, measured_request_ids
                )
                lmcache_timing_metrics = read_lmcache_timing_metrics(
                    log_path, load_log_lines, measured_request_ids
                )
                lmcache_retrieve_breakdown = read_lmcache_retrieve_breakdown(
                    log_path, load_log_lines, measured_request_ids
                )
                lmcache_eoss_metrics = read_lmcache_eoss_metrics(
                    log_path, load_log_lines
                )
                if args.hybrid_eoss:
                    expected_tiles = (32 + args.hybrid_cache_slice_size - 1) // (
                        args.hybrid_cache_slice_size
                    )
                    observed_tiles = {
                        int(tile) for tile in lmcache_eoss_metrics["tile_wait_ms"]
                    }
                    if (
                        not lmcache_eoss_metrics["submissions"]
                        or observed_tiles != set(range(expected_tiles))
                    ):
                        raise RuntimeError(
                            f"{policy}/C{concurrency} did not complete the EOSS "
                            f"tile trace: submissions="
                            f"{lmcache_eoss_metrics['submissions']}, "
                            f"observed_tiles={sorted(observed_tiles)}, "
                            f"expected_tiles={expected_tiles}"
                        )
                lmcache_adaptive_replays = read_lmcache_adaptive_replays(
                    log_path, load_log_lines, measured_request_ids
                )
                lmcache_adaptive_decisions = read_lmcache_adaptive_decisions(
                    log_path, load_log_lines, measured_request_ids
                )
                lmcache_policy_metrics = (
                    read_lmcache_policy_metrics(lmcache_log_path, lmcache_load_log_lines)
                    if args.backend == "lmcache"
                    else {"transferred": {}, "skipped": {}}
                )
                if args.backend == "lmcache":
                    transferred_groups = lmcache_policy_metrics["transferred"]
                    adaptive_replay_decisions = bool(lmcache_adaptive_replays)
                    if policy == "P2" and transferred_groups:
                        raise RuntimeError(
                            f"{policy}/C{concurrency} unexpectedly transferred "
                            f"LMCache groups: {transferred_groups}"
                        )
                    if (
                        policy != "P2"
                        and args.evict_gpu_cache
                        and args.suffix_tokens >= args.lmcache_chunk_size
                        # A full Mamba page is a recurrent checkpoint after
                        # its last token.  At an exact page boundary, P1
                        # correctly replays that final page for first-token
                        # logits instead of restoring an unsafe state.
                        and not (
                            policy == "P1"
                            and args.suffix_tokens == args.lmcache_chunk_size
                        )
                        and not transferred_groups
                        and not (
                            policy == "Adaptive" and adaptive_replay_decisions
                        )
                    ):
                        raise RuntimeError(
                            f"{policy}/C{concurrency} produced no LMCache H2D after "
                            "GPU-cache eviction; refusing to record a GPU-cache hit "
                            "as CPU recovery unless Adaptive explicitly logged a "
                            "Replay decision"
                        )
                h2d_bytes = h2d_metrics.get("CPU_to_GPU_total_bytes", 0.0)
                if args.backend == "native" and use_offload and h2d_bytes <= 0:
                    raise RuntimeError(
                        f"{policy}/C{concurrency} produced no CPU_to_GPU bytes; "
                        "the request hit GPU prefix cache or did not complete an "
                        "H2D restore, so refusing to record it"
                    )
                traced_group_bytes = sum(group_metrics["bytes"].values())
                if (
                    args.backend == "native"
                    and use_offload
                    and traced_group_bytes <= 0
                ):
                    raise RuntimeError(
                        f"{policy}/C{concurrency} produced no per-group CPU_to_GPU "
                        "trace; refusing to record an unattributed restore"
                    )
                if h2d_bytes > 0 and abs(traced_group_bytes - h2d_bytes) > 1:
                    raise RuntimeError(
                        f"{policy}/C{concurrency} group trace mismatch: "
                        f"group_bytes={traced_group_bytes} h2d_bytes={h2d_bytes}"
                    )
                row = {
                    "backend": (
                        "lmcache"
                        if args.backend == "lmcache"
                        else ("native" if use_offload else "none")
                    ),
                    "policy": policy,
                    "policy_env": policy_env,
                    "independent_process": True,
                    "prefix_tokens_min": 784,
                    "prefix_tokens_max": 900,
                    "shared_prefix_tokens": args.shared_prefix_tokens,
                    "suffix_tokens": args.suffix_tokens,
                    "expected_missing_pages": args.expected_missing_pages,
                    "output_tokens": args.output_tokens,
                    "adaptive_replay_below_tokens": args.adaptive_replay_below_tokens,
                    "adaptive_max_inflight_h2d": args.adaptive_max_inflight_h2d,
                    "adaptive_contention_replay_below_tokens": (
                        args.adaptive_contention_replay_below_tokens
                    ),
                    "adaptive_h2d_gbps": args.adaptive_h2d_gbps,
                    "adaptive_replay_ms_per_token": args.adaptive_replay_ms_per_token,
                    "adaptive_replay_base_ms": args.adaptive_replay_base_ms,
                    "adaptive_use_terminal_state": args.adaptive_use_terminal_state,
                    "hybrid_execution_tile_size": args.hybrid_execution_tile_size,
                    "hybrid_cache_slice_size": args.hybrid_cache_slice_size,
                    "hybrid_eoss": args.hybrid_eoss,
                    "hybrid_eoss_window": args.hybrid_eoss_window,
                    "warmup_requests": args.warmup_requests,
                    "warmup_repeats": args.warmup_repeats,
                    "warmup_settle_seconds": args.warmup_settle_seconds,
                    "evict_gpu_cache": args.evict_gpu_cache,
                    "reset_local_prefix_cache": args.reset_local_prefix_cache,
                    "evict_requests": args.evict_requests,
                    "p5_serialize_terminal": args.p5_serialize_terminal
                    or env.get("LMCACHE_MP_SERIALIZE_TERMINAL") == "1",
                    "serialize_all_load": args.serialize_all_load
                    or env.get("LMCACHE_MP_SERIALIZE_ALL_LOAD") == "1",
                    "requests": args.requests,
                    "concurrency": concurrency,
                    "start_index": args.start_index,
                    "seed": 0,
                    "safetensors_load_strategy": args.safetensors_load_strategy,
                    "activation_checkpoint": args.activation_checkpoint,
                    "activation_max_bytes": args.activation_max_bytes,
                    "suffix_only": args.suffix_only,
                    "retain_shared_prefix": args.retain_shared_prefix,
                    "lmcache_eviction_policy": (
                        args.lmcache_eviction_policy
                        if args.backend == "lmcache"
                        else None
                    ),
                    "slru_protected_ratio": (
                        args.slru_protected_ratio if args.backend == "lmcache" else None
                    ),
                    "lmcache_strict_layer_load": (
                        args.backend == "lmcache" and not args.hybrid_eoss
                    ),
                    "ttft_p50_ms": float(match.group(1)),
                    "ttft_p90_ms": float(match.group(2)),
                    "ttft_p99_ms": float(match.group(3)),
                    "tpot_p50_ms": float(tpot_match.group(1)),
                    "tpot_p90_ms": float(tpot_match.group(2)),
                    "tpot_p99_ms": float(tpot_match.group(3)),
                    "first_token_signature": signature_match.group(1),
                    "first_tokens": first_tokens,
                    "request_ttft_ms": request_ttfts,
                    "completion_texts": completion_texts,
                }
                row["h2d_cpu_to_gpu_bytes"] = h2d_bytes
                row["h2d_cpu_to_gpu_service_ms"] = 1000.0 * h2d_metrics.get(
                    "CPU_to_GPU_total_time", 0.0
                )
                row["h2d_cpu_to_gpu_queue_ms"] = 1000.0 * h2d_metrics.get(
                    "CPU_to_GPU_total_queue_time", 0.0
                )
                row["h2d_cpu_to_gpu_group_bytes"] = group_metrics["bytes"]
                row["h2d_cpu_to_gpu_group_blocks"] = group_metrics["blocks"]
                row["lmcache_h2d_retrieve_requests"] = len(
                    lmcache_metrics["requests"]
                )
                row["lmcache_h2d_retrieve_group_blocks"] = lmcache_metrics[
                    "blocks"
                ]
                row["lmcache_h2d_retrieve_group_tokens"] = lmcache_metrics[
                    "tokens"
                ]
                row["lmcache_prefix_match_count"] = len(lmcache_prefix_matches)
                row["lmcache_prefix_matches"] = lmcache_prefix_matches
                recovery_need_tokens = [
                    item["need_h2d_tokens"] for item in lmcache_prefix_matches
                ]
                row["lmcache_need_h2d_tokens"] = recovery_need_tokens
                row["lmcache_missing_pages"] = [
                    (
                        need_tokens // args.lmcache_chunk_size
                        if args.lmcache_chunk_size > 0
                        and need_tokens % args.lmcache_chunk_size == 0
                        else None
                    )
                    for need_tokens in recovery_need_tokens
                ]
                row["lmcache_recovery_length_uniform"] = (
                    len(set(recovery_need_tokens)) <= 1
                )
                row["lmcache_lookup_ms"] = lmcache_timing_metrics["lookup_ms"]
                row["lmcache_h2d_wait_ms"] = lmcache_timing_metrics[
                    "h2d_wait_ms"
                ]
                row["lmcache_h2d_cuda_sync_ms"] = lmcache_timing_metrics[
                    "h2d_cuda_sync_ms"
                ]
                row["lmcache_h2d_total_ms"] = lmcache_timing_metrics[
                    "h2d_total_ms"
                ]
                row["lmcache_h2d_batch_requests"] = lmcache_timing_metrics[
                    "h2d_batch_requests"
                ]
                row["lmcache_retrieve_breakdown"] = lmcache_retrieve_breakdown
                row["vllm_timing_metrics"] = vllm_timing
                row["vllm_prefill_time_ms"] = 1000.0 * vllm_timing.get(
                    "request_prefill_time_seconds_sum", 0.0
                )
                row["vllm_prefill_request_count"] = vllm_timing.get(
                    "request_prefill_time_seconds_count", 0.0
                )
                row["vllm_queue_time_ms"] = 1000.0 * vllm_timing.get(
                    "request_queue_time_seconds_sum", 0.0
                )
                row["vllm_inference_time_ms"] = 1000.0 * vllm_timing.get(
                    "request_inference_time_seconds_sum", 0.0
                )
                # P2's prefill phase is the measured replay-compute proxy.
                # Adaptive requests that explicitly chose replay use the same
                # proxy; P1/P5 prefill excludes the restored prefix and is
                # retained only as a separate phase metric.
                row["replay_compute_time_ms"] = (
                    row["vllm_prefill_time_ms"]
                    if policy == "P2" or lmcache_adaptive_replays
                    else 0.0
                )
                row["adaptive_replay_count"] = len(lmcache_adaptive_replays)
                row["lmcache_eoss_submissions"] = lmcache_eoss_metrics[
                    "submissions"
                ]
                row["lmcache_eoss_tile_wait_ms"] = lmcache_eoss_metrics[
                    "tile_wait_ms"
                ]
                row["lmcache_adaptive_replays"] = lmcache_adaptive_replays
                row["lmcache_adaptive_decisions"] = lmcache_adaptive_decisions
                row["lmcache_actual_h2d_transferred_object_groups"] = (
                    lmcache_policy_metrics["transferred"]
                )
                row["lmcache_actual_h2d_skipped_object_groups"] = (
                    lmcache_policy_metrics["skipped"]
                )
                row["lmcache_actual_h2d_object_group_bytes"] = (
                    lmcache_policy_metrics["bytes"]
                )
                # Validate the intended suffix-only state transition from the
                # engine log instead of assuming that the flag was effective.
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                suffix_matches = SUFFIX_RE.findall(
                    "\n".join(log_text.splitlines()[load_log_lines:])
                )
                p2_native_prefix_matches = P2_NATIVE_PREFIX_RE.findall(
                    "\n".join(log_text.splitlines()[load_log_lines:])
                )
                row["suffix_only_replay_count"] = len(suffix_matches)
                if p2_native_prefix_matches:
                    row["p2_native_prefix_tokens"] = {
                        request_id: int(prefix_tokens)
                        for request_id, prefix_tokens in p2_native_prefix_matches
                    }
                if (
                    args.backend == "lmcache"
                    and args.suffix_only
                    and policy in ("P1", "P5", "Adaptive")
                    and args.shared_prefix_tokens
                ):
                    expected_prefix = args.shared_prefix_tokens
                    valid_prefix_matches = [
                        item
                        for item in lmcache_prefix_matches
                        if item["local_gpu_tokens"] == expected_prefix
                        and item["cpu_tokens"] >= expected_prefix
                    ]
                    invalid_matches = [
                        {
                            "local_gpu_tokens": item["local_gpu_tokens"],
                            "cpu_tokens": item["cpu_tokens"],
                            "need_h2d_tokens": item["need_h2d_tokens"],
                        }
                        for item in lmcache_prefix_matches
                        if item not in valid_prefix_matches
                    ]
                    h2d_matches = [
                        item
                        for item in valid_prefix_matches
                        if item["need_h2d_tokens"] > 0
                    ]
                    gpu_only_matches = [
                        item
                        for item in valid_prefix_matches
                        if item["need_h2d_tokens"] == 0
                    ]
                    row["lmcache_recovery_match_count"] = len(h2d_matches)
                    row["lmcache_gpu_only_match_count"] = len(gpu_only_matches)
                    # With warmup/repeated requests, a suffix may legitimately
                    # remain on GPU.  Require at least one real recovery, but
                    # do not reject those GPU-only hits as invalid requests.
                    requires_all_h2d = args.warmup_requests == 0
                    if (
                        len(lmcache_prefix_matches) != args.requests
                        or invalid_matches
                        or not h2d_matches
                        or (requires_all_h2d and gpu_only_matches)
                    ):
                        raise RuntimeError(
                            f"{policy}/C{concurrency} did not satisfy the "
                            "GPU-prefix + CPU-suffix recovery precondition: "
                            f"expected {args.requests} requests with local_prefix="
                            f"{expected_prefix}, at least one H2D recovery; got "
                            f"count={len(lmcache_prefix_matches)}, "
                            f"recovery={len(h2d_matches)}, gpu_only="
                            f"{len(gpu_only_matches)}, invalid={invalid_matches[:4]}"
                        )
                    if args.expected_missing_pages is not None:
                        observed_need_tokens = [
                            item["need_h2d_tokens"] for item in valid_prefix_matches
                        ]
                        expected_need_tokens = (
                            args.expected_missing_pages * args.lmcache_chunk_size
                        )
                        observed_pages = {
                            need_tokens // args.lmcache_chunk_size
                            for need_tokens in observed_need_tokens
                            if args.lmcache_chunk_size > 0
                            and need_tokens % args.lmcache_chunk_size == 0
                        }
                        if (
                            len(observed_need_tokens) != args.requests
                            or any(
                                need_tokens != expected_need_tokens
                                for need_tokens in observed_need_tokens
                            )
                        ):
                            raise RuntimeError(
                                f"{policy}/C{concurrency} did not satisfy the "
                                "expected actual missing-page bucket: expected "
                                f"K={args.expected_missing_pages}, observed="
                                f"{sorted(observed_pages)}, tokens="
                                f"{observed_need_tokens}"
                            )
                if (
                    args.suffix_only
                    and policy in ("P3", "P4")
                    and args.shared_prefix_tokens
                ):
                    expected_prefix = args.shared_prefix_tokens
                    invalid_matches = [
                        (int(local_prefix), int(loaded_tokens))
                        for local_prefix, loaded_tokens in suffix_matches
                        if int(local_prefix) != expected_prefix or int(loaded_tokens) <= 0
                    ]
                    if len(suffix_matches) != args.requests or invalid_matches:
                        raise RuntimeError(
                            f"{policy}/C{concurrency} did not satisfy the "
                            "suffix-only precondition: expected "
                            f"{args.requests} replay markers with "
                            f"local_prefix={expected_prefix}, got "
                            f"count={len(suffix_matches)}, "
                            f"invalid={invalid_matches[:4]}"
                        )
                if (
                    args.suffix_only
                    and policy == "P2"
                    and args.shared_prefix_tokens
                ):
                    prefixes = [
                        int(prefix_tokens)
                        for _, prefix_tokens in p2_native_prefix_matches
                    ]
                    if (
                        len(prefixes) != args.requests
                        or any(prefix != args.shared_prefix_tokens for prefix in prefixes)
                    ):
                        raise RuntimeError(
                            f"P2/C{concurrency} did not retain the expected native "
                            "GPU prefix for suffix-only replay: expected "
                            f"{args.shared_prefix_tokens}, observed={prefixes}"
                        )
                if suffix_matches:
                    row["suffix_only_local_prefix_tokens"] = int(
                        suffix_matches[-1][0]
                    )
                    row["suffix_only_loaded_tokens"] = int(suffix_matches[-1][1])
                with args.output.open("a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            finally:
                if process is not None:
                    stop_server(process)
                if lmcache_process is not None:
                    stop_server(lmcache_process)

    if generated_lmcache_config is not None:
        generated_lmcache_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
