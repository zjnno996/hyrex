# SPDX-License-Identifier: Apache-2.0
"""Run a real two-tier prefix workload against LMCache MP.

Each cell first stores real conversation histories in LMCache's CPU tier,
then restarts vLLM and submits the next real user turns. Restarting releases
the GPU KV cache, so every measured CPU hit follows the normal idle-session
life cycle rather than an artificial shared GPU-prefix construction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from run_native_independent_cells import (
    DATASET,
    MODEL,
    start_lmcache_server,
    remember_process_group,
    stop_server,
    wait_lmcache_ready,
    wait_ready,
)
import requests
from hybrid_baseline_config import BASELINES, baseline_config


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "benchmarks/reproductions/sharegpt_mooncake_e2e.py"
TRACE_E2E = ROOT / "benchmarks/reproductions/sharegpt_hybrid_trace_e2e.py"
POLICIES = {
    "P1": "all_load",
    "P2": "all_replay",
    "P3": "full_load_linear_replay",
    "P4": "full_replay_linear_load",
}
LEGACY_POLICY_BASELINES = {
    "P1": "marconi",
    "P3": "tail_replay",
}
SUMMARY_RE = re.compile(
    r"latency_ms_p50/p90/p99=([0-9.]+)/([0-9.]+)/([0-9.]+)"
)


def parse_summary(output: str) -> tuple[float, float, float]:
    match = SUMMARY_RE.search(output)
    if match is not None:
        return tuple(float(match.group(index)) for index in range(1, 4))
    for line in reversed(output.splitlines()):
        try:
            row = json.loads(line)
            return (
                row["ttft_p50_ms"],
                row.get("ttft_p95_ms") or row["ttft_p90_ms"],
                row["ttft_p99_ms"],
            )
        except (json.JSONDecodeError, KeyError):
            continue
    raise RuntimeError(f"could not parse TTFT summary: {output}")


def parse_streaming_metrics(output: str) -> dict[str, float]:
    """Extract optional streaming metrics emitted by the trace E2E driver."""
    for line in reversed(output.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        return {
            key: float(row[key])
            for key in (
                "tpot_p50_ms",
                "tpot_p95_ms",
                "tpot_p99_ms",
                "output_token_throughput",
                "wall_ms",
            )
            if key in row
        }
    return {}


def wait_ready_for_process(
    process: subprocess.Popen, url: str, timeout_s: float
) -> None:
    """Wait for vLLM, surfacing an early worker failure immediately."""
    session = requests.Session()
    session.trust_env = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before becoming ready (returncode={process.returncode})"
            )
        try:
            response = session.get(f"{url}/health", timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s:.0f}s")


def run_e2e(
    mode: str,
    server: str,
    *,
    requests: int,
    concurrency: int,
    shared_prefix_tokens: int,
    suffix_tokens: int,
    model_name: str,
    tokenizer: str,
    dataset: Path,
    prefix_only: bool = False,
    trace: Path | None = None,
    trace_prompt_field: str = "resume_prompt",
    baseline: str = "marconi",
) -> str:
    if trace is not None:
        command = [
            sys.executable,
            str(TRACE_E2E),
            "--trace",
            str(trace),
            "--prompt-field",
            trace_prompt_field,
            "--execution-mode",
            "preconditioned",
            "--server",
            server,
            "--model",
            model_name,
            "--baseline",
            baseline,
            "--concurrency",
            str(concurrency),
            "--limit",
            str(requests),
            "--max-tokens",
            "16",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    command = [
        os.environ.get("PYTHON", "python"),
        str(E2E),
        mode,
        "--server",
        server,
        "--requests",
        str(requests),
        "--concurrency",
        str(concurrency),
        "--warmup-concurrency",
        "1",
        "--model",
        model_name,
        "--tokenizer",
        tokenizer,
        "--dataset-path",
        str(dataset),
        "--shared-prefix-tokens",
        str(shared_prefix_tokens),
        "--suffix-tokens",
        str(suffix_tokens),
        "--warmup-settle-seconds",
        "2",
    ]
    if prefix_only:
        command.append("--prefix-only")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def vllm_command(
    policy: str,
    *,
    server_port: int,
    lmcache_port: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    model_path: Path,
    model_name: str,
    attention_backend: str,
    cache_backend: str = "lmcache",
    cpu_cache_gb: float = 24.0,
    native_eviction_policy: str | None = None,
) -> list[str]:
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(server_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--enforce-eager",
        # HyRex evaluates text conversation-cache recovery.  Disable Qwen's
        # unused vision tower so vLLM does not profile a FlashAttention-only
        # multimodal path on this source-only build.
        "--language-model-only",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--attention-backend",
        attention_backend,
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        "0",
        "--served-model-name",
        model_name,
    ]
    if cache_backend == "none":
        pass
    elif cache_backend == "lmcache":
        command += [
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "LMCacheMPConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "lmcache.mp.host": "tcp://127.0.0.1",
                        "lmcache.mp.port": lmcache_port,
                        "discard_partial_chunks": False,
                    },
                }
            ),
        ]
    elif cache_backend == "native":
        command += [
            "--kv-offloading-backend", "native",
            "--kv-offloading-size", str(cpu_cache_gb),
        ]
        if native_eviction_policy is not None:
            command += [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "eviction_policy": native_eviction_policy,
                        },
                    }
                ),
            ]
    else:
        raise ValueError(f"unsupported cache backend: {cache_backend}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        choices=tuple(BASELINES),
        default="marconi",
        help="Named method under test; recorded verbatim in the result artifact.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default=None,
        help="Deprecated compatibility alias. Use --baseline for new experiments.",
    )
    parser.add_argument("--model-path", type=Path, default=Path(MODEL))
    parser.add_argument("--model-name", default="Qwen3.5-9B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DATASET))
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="JSONL from sharegpt_hybrid_session_trace.py; uses real resume turns.",
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--lmcache-chunk-size", type=int, default=528)
    parser.add_argument("--lmcache-kv-gb", type=float, default=24.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--shared-prefix-tokens", type=int, default=256)
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=1312,
        help="CPU-resident residual; default makes a 256+1312=1568-token prompt.",
    )
    parser.add_argument("--server-port", type=int, default=8002)
    parser.add_argument("--lmcache-port", type=int, default=5556)
    parser.add_argument("--lmcache-http-port", type=int, default=8081)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help=(
            "GPU fraction for the two-worker test. LMCache's CUDA staging pool "
            "remains resident while the second worker starts."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/root/qwen35_partial_prefix.jsonl")
    )
    args = parser.parse_args()
    if args.policy is not None:
        try:
            requested_baseline = LEGACY_POLICY_BASELINES[args.policy]
        except KeyError as exc:
            raise ValueError(
                f"legacy policy {args.policy} has no named baseline; use --baseline"
            ) from exc
        if args.baseline != "marconi" and args.baseline != requested_baseline:
            raise ValueError("--policy and --baseline select different methods")
        args.baseline = requested_baseline
    selected_baseline = BASELINES[args.baseline]
    if selected_baseline.recovery_policy is None:
        raise ValueError(
            f"{args.baseline} has no native runtime binding yet; do not label an all-load run as it"
        )
    recovery_policy = selected_baseline.recovery_policy
    if not args.model_path.is_dir():
        raise ValueError(f"model path does not exist: {args.model_path}")
    if args.trace is None and not args.dataset_path.is_file():
        raise ValueError(f"dataset path does not exist: {args.dataset_path}")
    if args.trace is not None and not args.trace.is_file():
        raise ValueError(f"trace path does not exist: {args.trace}")
    tokenizer = args.tokenizer or str(args.model_path)
    trace_rows: list[dict[str, object]] = []
    trace_seed_requests = 0
    if args.trace is not None:
        trace_rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
        if not trace_rows:
            raise ValueError("trace is empty")
        trace_seed_requests = len({row["session_id"] for row in trace_rows})
        args.requests = len(trace_rows)
    if args.requests is None:
        args.requests = args.concurrency
    if args.requests < args.concurrency:
        raise ValueError("requests must be >= concurrency")

    total_tokens = (
        max(int(row["resume_tokens"]) for row in trace_rows)
        if trace_rows
        else args.shared_prefix_tokens + args.suffix_tokens
    )
    max_model_len = ((total_tokens + 127) // 128) * 128
    # LMCache requires exactly one Mamba/attention block per prefill step so
    # that it can checkpoint the recurrent state at every boundary.
    max_num_batched_tokens = args.lmcache_chunk_size
    config_path = Path(f"/tmp/partial_lmcache_{args.lmcache_port}.yaml")
    config_path.write_text(
        "\n".join(
            [
                f"chunk_size: {args.lmcache_chunk_size}",
                "local_device: cpu",
                "local_cpu: true",
                f"max_local_cpu_size: {args.lmcache_kv_gb}",
                "save_unfull_chunk: true",
                "enable_hidden_state_cache: true",
                f"max_hidden_state_cpu_size: {args.lmcache_kv_gb}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    # The virtualenv's editable vLLM install may point at the primary
    # worktree. Put this branch first so a baseline/HyRex experiment executes
    # the code that its artifact names, rather than silently importing another
    # branch's vLLM package.
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
            "VLLM_MOONCAKE_HYBRID_POLICY": recovery_policy,
            "VLLM_MOONCAKE_HYBRID_ACTIVATION_CHECKPOINT": "0",
            "VLLM_MOONCAKE_HYBRID_SUFFIX_ONLY": (
                # Main E2E runs model an idle session whose GPU cache was
                # released when the seed worker stopped.  There is no
                # fabricated GPU-local prefix to preserve.
                "0"
            ),
            # Tail-Replay must fence the selected Full-Attention H2D before
            # the replay forward consumes it; keep other baselines overlapped.
            "LMCACHE_MP_STRICT_LAYER_LOAD": (
                "1" if args.baseline == "tail_replay" else "0"
            ),
            "LMCACHE_CONFIG_FILE": str(config_path),
        }
    )
    # Tail-Replay uses HyRex's narrow adapter over LMCache's external
    # Hybrid-aware connector.  The adapter preserves the upstream per-group
    # block mapping and fences, and only selects the FullAttention objects.
    if args.baseline == "tail_replay":
        env["LMCACHE_USE_HYREX_EXTERNAL_MP"] = "1"
        # LMCache's current single-object-group IPC retrieve can leave its
        # device future unresolved.  This mode keeps the stable all-object
        # metadata lookup while LMCache's Hybrid policy filters the actual
        # Mamba H2D copies and vLLM replays the omitted state.  It is a
        # transport compatibility mode, not an all-load fallback.
        env["HYREX_TAIL_COMPAT_ALL_OBJECT_RETRIEVE"] = "1"
    if args.cuda_visible_devices is not None:
        # LMCache MP is spawned below and initializes CUDA during startup.
        # Set the parent environment before launching it, not only vLLM's
        # Popen environment, otherwise its CUDA context can span every GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    lmcache = start_lmcache_server(
        config_path,
        port=args.lmcache_port,
        http_port=args.lmcache_http_port,
        chunk_size=args.lmcache_chunk_size,
    )
    process = None
    try:
        wait_lmcache_ready(args.lmcache_http_port, 180)
        server = f"http://127.0.0.1:{args.server_port}"
        command = vllm_command(
            "P1",
            server_port=args.server_port,
            lmcache_port=args.lmcache_port,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            model_path=args.model_path,
            model_name=args.model_name,
            attention_backend=args.attention_backend,
        )
        seed_log = Path(f"/tmp/partial_seed_c{args.concurrency}.log").open("w")
        process = subprocess.Popen(
            command,
            env={**env, "VLLM_MOONCAKE_HYBRID_POLICY": "all_load"},
            cwd=ROOT,
            stdout=seed_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        remember_process_group(process)
        wait_ready_for_process(process, server, args.startup_timeout)
        run_e2e(
            "warmup",
            server,
            requests=trace_seed_requests or args.requests,
            concurrency=1,
            shared_prefix_tokens=args.shared_prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            model_name=args.model_name,
            tokenizer=tokenizer,
            dataset=args.dataset_path,
            trace=args.trace,
            trace_prompt_field="cache_prompt",
        )
        # LMCache stores are asynchronous; let the seed server publish all
        # object groups before its CUDA handles are torn down.
        time.sleep(5)
        stop_server(process)
        process = None
        seed_log.close()

        test_log = Path(
            f"/tmp/partial_{args.baseline}_c{args.concurrency}.log"
        ).open("w")
        process = subprocess.Popen(
            vllm_command(
                args.policy,
                server_port=args.server_port,
                lmcache_port=args.lmcache_port,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                model_path=args.model_path,
                model_name=args.model_name,
                attention_backend=args.attention_backend,
            ),
            env=env,
            cwd=ROOT,
            stdout=test_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        remember_process_group(process)
        wait_ready_for_process(process, server, args.startup_timeout)
        output = run_e2e(
            "load",
            server,
            requests=args.requests,
            concurrency=args.concurrency,
            shared_prefix_tokens=args.shared_prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            model_name=args.model_name,
            tokenizer=tokenizer,
            dataset=args.dataset_path,
            trace=args.trace,
            trace_prompt_field="resume_prompt",
            baseline=args.baseline,
        )
        # Tail-Replay is evaluated as an ordinary resumed session: the seed
        # worker has exited, so GPU KV is absent and CPU-resident history is
        # recovered directly. ``op.block_ids`` still lists every Hybrid group
        # for allocator correctness; that is not evidence of a Mamba H2D.
        if recovery_policy == "full_load_linear_replay":
            test_log.flush()
            log_text = test_log.name and Path(test_log.name).read_text(
                encoding="utf-8", errors="replace"
            )
            has_transport = (
                "HyRex Tail-Replay compatibility all-object lookup"
                in log_text
                or "HyRex Tail-Replay selected Full object groups=" in log_text
            )
            has_cpu_only_resume = bool(
                re.search(r"local_gpu_tokens=0 cpu_tokens=[1-9]", log_text)
            )
            if not has_transport or not has_cpu_only_resume:
                raise RuntimeError(
                    "Tail-Replay runtime invalid: missing Tail transport or "
                    "CPU-only resumed-session cache hit"
                )
        p50, p95, p99 = parse_summary(output)
        streaming_metrics = parse_streaming_metrics(output)
        row = {
            "workload": (
                "sharegpt_session_resume_after_gpu_release"
                if args.trace is not None
                else "cpu_resident_session_resume"
            ),
            "backend": "lmcache_mp",
            "baseline": baseline_config(args.baseline),
            "recovery_policy": recovery_policy,
            "tail_transport_mode": (
                "compat_all_object_lookup"
                if args.baseline == "tail_replay"
                else None
            ),
            "tail_replay_scope": (
                "full_history_after_gpu_release"
                if args.baseline == "tail_replay"
                else None
            ),
            "gpu_cache_lifecycle": "seed_worker_exit_before_resume",
            "shared_prefix_tokens": args.shared_prefix_tokens,
            "cpu_residual_tokens": args.suffix_tokens,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "seed": 0,
            "ttft_p50_ms": p50,
            "ttft_p95_ms": p95,
            "ttft_p99_ms": p99,
            **streaming_metrics,
        }
        if args.trace is not None:
            row.update(
                trace=str(args.trace),
                trace_sessions=trace_seed_requests,
                residual_tokens_min=min(int(r["residual_tokens"]) for r in trace_rows),
                residual_tokens_max=max(int(r["residual_tokens"]) for r in trace_rows),
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    finally:
        if process is not None:
            stop_server(process)
        stop_server(lmcache)
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
