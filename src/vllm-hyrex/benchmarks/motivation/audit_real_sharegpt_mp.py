#!/usr/bin/env python3
"""Audit stock vLLM + official LMCache MP on real ShareGPT turns."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests


ROOT = Path(__file__).resolve().parents[2]
HTTP = requests.Session()
HTTP.trust_env = False
REQUEST_HTTP = threading.local()
RESET_LOG = None


def wait_ready(process, url, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{url} exited with code {process.returncode}")
        try:
            if HTTP.get(url, timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(url)


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def start(command, env, log_path):
    log = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return process


def request(port, model, prompt, max_tokens=1, first_token_logprobs=False):
    http = getattr(REQUEST_HTTP, "session", None)
    if http is None:
        http = requests.Session()
        http.trust_env = False
        REQUEST_HTTP.session = http
    start_time = time.perf_counter()
    response = http.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0, "ignore_eos": max_tokens > 1,
              "stream": True, "stream_options": {"include_usage": True},
              **({"logprobs": 5} if first_token_logprobs else {})},
        stream=True, timeout=240,
    )
    response.raise_for_status()
    usage = None
    ttft_ms = None
    first_text = None
    first_logprobs = None
    generated_text = []
    for line in response.iter_lines(decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line or not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        generated_text.extend(choice.get("text", "") for choice in event.get("choices", []))
        if ttft_ms is None and any(choice.get("text") for choice in event.get("choices", [])):
            ttft_ms = (time.perf_counter() - start_time) * 1000
            first_text = next(choice["text"] for choice in event["choices"] if choice.get("text"))
            first_logprobs = next(
                choice.get("logprobs") for choice in event["choices"] if choice.get("text")
            )
        if event.get("usage"):
            usage = event["usage"]
    response.close()
    if usage is None or ttft_ms is None:
        raise RuntimeError("stream did not include first token and final usage")
    if first_token_logprobs:
        if first_logprobs is None:
            raise RuntimeError("requested first-token logprobs were absent")
        usage["diagnostic_first_logprobs"] = first_logprobs
        usage["diagnostic_generated_text"] = "".join(generated_text)
    return usage, ttft_ms, (time.perf_counter() - start_time) * 1000, first_text


def reset_gpu_prefix_cache(port, timeout=120):
    start = time.monotonic()
    attempts = 0
    while time.monotonic() - start < timeout:
        attempts += 1
        offset = RESET_LOG.stat().st_size if RESET_LOG else 0
        response = HTTP.post(
            f"http://127.0.0.1:{port}/reset_prefix_cache",
            params={"reset_external": "false"}, timeout=60,
        )
        response.raise_for_status()
        if response.content and response.json().get("success") is True:
            return attempts, round((time.monotonic() - start) * 1000, 2)
        if not response.content and RESET_LOG:
            # The upstream endpoint returns an empty 200 even on a failed
            # reset. Verify the engine's acknowledgement, not just HTTP status.
            with RESET_LOG.open() as log:
                log.seek(offset)
                messages = log.read()
            if "Successfully reset prefix cache" in messages and "Failed to reset prefix cache" not in messages:
                return attempts, round((time.monotonic() - start) * 1000, 2)
        time.sleep(0.5)
    raise TimeoutError("GPU prefix cache did not reset; refusing contaminated TTFT")


def main():
    global RESET_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("/root/hyrex_sharegpt_session_smoke.jsonl"))
    parser.add_argument("--limit", type=int, default=8, help="0 means all selected rows")
    parser.add_argument("--sessions", type=int, default=0, help="0 means all sessions")
    parser.add_argument("--min-session-turns", type=int, default=1)
    parser.add_argument("--mode", choices=("default", "retain_tail"), required=True)
    parser.add_argument("--workflow", choices=("paired", "online"), default="paired")
    parser.add_argument("--online-repetitions", type=int, default=1,
                        help="Repeat selected trace, clearing CPU/GPU between repetitions")
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--first-token-logprobs", action="store_true",
                        help="Online correctness diagnosis only; records top-5 first-token logprobs")
    parser.add_argument("--profile-request-index", type=int,
                        help="Diagnostic only: profile one zero-based online request; timings are not benchmark data")
    parser.add_argument("--reset-between-requests", action="store_true",
                        help="Keep LMCache CPU state while clearing GPU prefix cache after each request")
    parser.add_argument("--round-robin-sessions", action="store_true",
                        help="Interleave sessions by turn for the 4x10 CPU-offload experiment")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent requests per same-turn wave")
    parser.add_argument("--concurrency-sweep",
                        help="Comma-separated online concurrencies run in one service")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip unrelated requests before the online measurement")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reset-method", choices=("local", "restart"), default="local")
    parser.add_argument("--attach", action="store_true", help="Use already-running services; do not stop them")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prefill-budget", type=int, default=2048)
    parser.add_argument("--cpu-gb", type=int, default=4)
    parser.add_argument("--experimental-lmcache-source", type=Path,
                        help="Opt-in LMCache source checkout for independent-store smoke tests")
    parser.add_argument("--vllm-source", type=Path,
                        help="Use a separate vLLM checkout (e.g. clean baseline)")
    parser.add_argument("--experimental-state-cap-tokens", type=int,
                        help="Store recurrent checkpoints only through this boundary")
    parser.add_argument("--experimental-full-page-size", type=int,
                        help="Index Full Attention KV independently at the physical page size")
    parser.add_argument("--experimental-no-mask-full-kv", action="store_true",
                        help="Diagnostic: allow replay to overwrite preloaded Full KV")
    parser.add_argument("--experimental-no-deep-lookup", action="store_true",
                        help="Diagnostic: retain independent stores but use common-prefix lookup")
    parser.add_argument("--vllm-port", type=int, default=8565)
    parser.add_argument("--tail-probe-boundary", type=int,
                        help="Experimental single exact recurrent checkpoint")
    parser.add_argument("--replace-tail-checkpoint", action="store_true",
                        help="Isolated-pair probe: replace final coarse checkpoint with exact KV tail")
    parser.add_argument("--lmcache-port", type=int, default=5565)
    parser.add_argument("--lmcache-http-port", type=int, default=8566)
    args = parser.parse_args()
    concurrencies = ([int(value) for value in args.concurrency_sweep.split(",")]
                     if args.concurrency_sweep else [args.concurrency])
    if args.attach and args.reset_method != "local":
        parser.error("--attach requires --reset-method local")
    if args.reset_between_requests and args.workflow != "online":
        parser.error("--reset-between-requests requires --workflow online")
    if args.online_repetitions < 1:
        parser.error("online repetitions must be positive")
    if not concurrencies or any(value < 1 for value in concurrencies):
        parser.error("concurrency must be positive")
    if len(set(concurrencies)) != len(concurrencies):
        parser.error("concurrency sweep values must be unique")
    if max(concurrencies) > 1 and args.workflow != "online":
        parser.error("concurrency > 1 requires online workflow")
    if (max(concurrencies) > 1 or len(concurrencies) > 1) and args.profile_request_index is not None:
        parser.error("request profiling requires concurrency 1")
    if len(concurrencies) > 1 and args.online_repetitions != 1:
        parser.error("concurrency sweep requires one online repetition")
    if args.online_repetitions > 1 and not args.reset_between_requests:
        parser.error("repetitions require strict GPU resets")
    if args.experimental_state_cap_tokens is not None and args.experimental_lmcache_source is None:
        parser.error("experimental state cap requires experimental LMCache source")
    if args.experimental_full_page_size and args.experimental_lmcache_source is None:
        parser.error("independent Full KV pages require experimental LMCache source")

    trace_rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    counts = Counter(row["session_id"] for row in trace_rows)
    eligible = [sid for sid, count in counts.items() if count >= args.min_session_turns]
    if args.sessions:
        eligible = eligible[:args.sessions]
        if len(eligible) != args.sessions:
            raise ValueError(f"only {len(eligible)} eligible sessions in {args.trace}")
    selected = set(eligible)
    rows = [row for row in trace_rows if row["session_id"] in selected]
    if args.round_robin_sessions:
        session_order = {sid: index for index, sid in enumerate(eligible)}
        rows.sort(key=lambda row: (row["turn_index"], session_order[row["session_id"]]))
    if args.limit:
        rows = rows[:args.limit]
    assert rows and all("cache_prompt" in row and "resume_prompt" in row for row in rows)
    if args.workflow == "online":
        last_turn = {}
        for row in rows:
            sid = row["session_id"]
            if row["turn_index"] != last_turn.get(sid, -1) + 1:
                raise ValueError(f"out-of-order turn for session {sid}")
            last_turn[sid] = row["turn_index"]
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.check_only:
        turn_counts = Counter(row["turn_index"] for row in rows)
        waves = {value: sum(math.ceil(count / value) for count in turn_counts.values())
                 for value in concurrencies}
        print(json.dumps({"workflow": args.workflow,
                          "sessions": len({r["session_id"] for r in rows}),
                          "requests_per_sweep": len(rows), "concurrencies": concurrencies,
                          "waves": waves}))
        return
    free_gpu_mib = int(subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).strip())
    available_gib = psutil.virtual_memory().available / 1024**3
    if not args.attach and (free_gpu_mib < 22000 or available_gib < args.cpu_gb + 20):
        raise RuntimeError(f"resource gate: GPU={free_gpu_mib} MiB, host={available_gib:.1f} GiB")
    if not args.attach:
        for port in (args.vllm_port, args.lmcache_port, args.lmcache_http_port):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    RESET_LOG = args.output_dir / "vllm.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(args.vllm_source.resolve() if args.vllm_source else ROOT)
    if args.experimental_lmcache_source is not None:
        env["PYTHONPATH"] = str(args.experimental_lmcache_source.resolve()) + os.pathsep + env["PYTHONPATH"]
        env["LMCACHE_HYREX_SPLIT_ENGINE_GROUPS"] = "1"
        if args.experimental_state_cap_tokens is not None:
            env["LMCACHE_HYREX_STATE_STORE_CAP_TOKENS"] = str(args.experimental_state_cap_tokens)
        else:
            env.pop("LMCACHE_HYREX_STATE_STORE_CAP_TOKENS", None)
        if args.experimental_full_page_size:
            env["LMCACHE_HYREX_FULL_PAGE_SIZE"] = str(args.experimental_full_page_size)
    if args.experimental_no_mask_full_kv:
        env["VLLM_HYREX_MASK_FULL_KV"] = "0"
    if args.experimental_no_deep_lookup:
        env["LMCACHE_HYREX_DEEP_LOOKUP"] = "0"
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    if args.tail_probe_boundary:
        if not args.experimental_lmcache_source or args.experimental_full_page_size != 16:
            parser.error("tail probe requires experimental LMCache and full16")
        env["VLLM_HYREX_TAIL_PROBE"] = str(args.tail_probe_boundary)
    if args.replace_tail_checkpoint:
        if not args.experimental_lmcache_source or args.experimental_full_page_size != 16:
            parser.error("replacement requires independent Full16")
        env["VLLM_HYREX_REPLACE_TAIL"] = "1"
    if args.reset_method == "local":
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if args.mode == "retain_tail":
        env["LMCACHE_SAVE_UNFULL_CHUNK"] = "True"
    else:
        env.pop("LMCACHE_SAVE_UNFULL_CHUNK", None)

    model = "/root/models/Qwen3.5-9B"
    extra = {
        "lmcache.mp.host": "tcp://127.0.0.1",
        "lmcache.mp.port": args.lmcache_port,
    }
    if args.mode == "retain_tail":
        extra["discard_partial_chunks"] = False
    transfer = json.dumps({
        "kv_connector": "ReplaceTailConnector" if args.replace_tail_checkpoint else ("TailProbeConnector" if args.tail_probe_boundary else "LMCacheMPConnector"),
        "kv_connector_module_path": (
            "lmcache.integration.vllm.replace_tail_connector" if args.replace_tail_checkpoint else
            "lmcache.integration.vllm.tail_probe_connector" if args.tail_probe_boundary
            else "lmcache.integration.vllm.lmcache_mp_connector"),
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    })
    serve = [
        str(Path(sys.executable).with_name("vllm")), "serve", model,
        "--host", "127.0.0.1", "--port", str(args.vllm_port),
        "--served-model-name", "Qwen3.5-9B", "--dtype", "bfloat16",
        "--language-model-only", "--enable-prefix-caching",
        "--max-model-len", "4096", "--max-num-batched-tokens", str(args.prefill_budget),
        "--max-num-seqs", str(max(concurrencies)),
        "--gpu-memory-utilization", "0.8", "--enforce-eager",
        "--safetensors-load-strategy", "eager",
        "--enable-prompt-tokens-details", "--kv-transfer-config", transfer,
    ]
    cache = [
        sys.executable, "-m", "lmcache.v1.multiprocess.http_server",
        "--host", "127.0.0.1", "--port", str(args.lmcache_port),
        "--http-host", "127.0.0.1", "--http-port", str(args.lmcache_http_port),
        "--chunk-size", "528", "--l1-size-gb", str(args.cpu_gb),
        "--eviction-policy", "LRU", "--separate-object-groups",
        "--disable-observability",
    ]
    if args.profile_request_index is not None:
        serve += ["--profiler-config", json.dumps({
            "profiler": "torch", "torch_profiler_dir": str((args.output_dir / "profile").resolve()),
            "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
            "ignore_frontend": True})]
        env["VLLM_HYREX_LOG_REPLAY"] = "1"
    cache_process = None
    vllm_process = None
    try:
        if not args.attach:
            cache_process = start(cache, env, args.output_dir / "lmcache.log")
        wait_ready(cache_process, f"http://127.0.0.1:{args.lmcache_http_port}/healthcheck")
        if args.workflow == "online":
            if not args.attach:
                vllm_process = start(serve, env, args.output_dir / "vllm.log")
            wait_ready(vllm_process, f"http://127.0.0.1:{args.vllm_port}/health")
            if not args.no_warmup and not args.attach:
                with (args.output_dir / "warmup.jsonl").open("w") as warmup_out:
                    for i, length in enumerate((651, 1191, 1708, 1922)):
                        prompt = [1000 + i] + [2000 + i] * (length - 1)
                        for repeat in range(2):
                            usage, ttft_ms, _, _ = request(
                                args.vllm_port, "Qwen3.5-9B", prompt, max_tokens=32,
                                first_token_logprobs=args.first_token_logprobs)
                            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                            item = {"length": length, "repeat": repeat,
                                    "cached_tokens": cached, "ttft_ms": round(ttft_ms, 2),
                                    "completion_tokens": usage["completion_tokens"]}
                            warmup_out.write(json.dumps(item) + "\n")
                            warmup_out.flush()
                            print(json.dumps({"warmup": item}), flush=True)
                            if repeat and cached == 0:
                                raise RuntimeError(f"LMCache warmup CPU miss at length={length}")
                            reset_gpu_prefix_cache(args.vllm_port)
                            time.sleep(2)
                    # Unrelated tokens, but the same seed/resume shapes as the
                    # mechanism probe. Exercise replay/tail and decode before timing.
                    for length in (1054, 1191):
                        prompt = [3001] + [3002] * 1053 + [3003] * (length - 1054)
                        usage, ttft_ms, _, _ = request(
                            args.vllm_port, "Qwen3.5-9B", prompt, max_tokens=32,
                            first_token_logprobs=args.first_token_logprobs)
                        warmup_out.write(json.dumps({
                            "phase": "unrelated_shape_pair", "length": length,
                            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                            "completion_tokens": usage["completion_tokens"],
                            "ttft_ms": round(ttft_ms, 2)}) + "\n")
                        warmup_out.flush()
                        reset_gpu_prefix_cache(args.vllm_port)
                        time.sleep(2)
                clear = HTTP.post(
                    f"http://127.0.0.1:{args.lmcache_http_port}/cache/clear",
                    json={"tier": "l1"}, timeout=60,
                )
                clear.raise_for_status()
                if clear.json().get("status") != "ok":
                    raise RuntimeError(f"LMCache warmup clear failed: {clear.text}")
                print("Warmup GPU and LMCache CPU caches cleared", flush=True)
            with (args.output_dir / "online.jsonl").open("w") as out, \
                    (args.output_dir / "resets.jsonl").open("w") as reset_out, \
                    (args.output_dir / "concurrency_warmup.jsonl").open("w") as concurrent_warmup_out, \
                    (args.output_dir / "sweep_diagnostics.jsonl").open("w") as diagnostics_out:
                def clear_cpu_cache():
                    response = HTTP.post(
                        f"http://127.0.0.1:{args.lmcache_http_port}/cache/clear",
                        json={"tier": "l1"}, timeout=60)
                    response.raise_for_status()
                    if response.json().get("status") != "ok":
                        raise RuntimeError("CPU cache clear failed")

                def run_one(index_row):
                    index, row = index_row
                    usage, ttft_ms, elapsed_ms, first_text = request(
                        args.vllm_port, "Qwen3.5-9B", row["resume_prompt"],
                        max_tokens=row.get("max_output_tokens", args.max_output_tokens),
                        first_token_logprobs=args.first_token_logprobs,
                    )
                    return index, row, usage, ttft_ms, elapsed_ms, first_text

                request_index = 0
                global_wave_index = 0
                with ThreadPoolExecutor(max_workers=max(concurrencies)) as executor:
                    for sweep_index, concurrency in enumerate(concurrencies):
                        if sweep_index:
                            time.sleep(2)
                            reset_gpu_prefix_cache(args.vllm_port)
                            clear_cpu_cache()

                        # Warm the actual batch size and recovery path with unrelated
                        # token IDs, then remove those entries from both cache tiers.
                        lengths = (651, 903, 1191, 1708)
                        warm_prompts = [
                            [4000 + index] + [5000 + index] * (lengths[index % len(lengths)] - 1)
                            for index in range(concurrency)
                        ]
                        for phase in ("seed", "resume"):
                            futures = [executor.submit(
                                request, args.vllm_port, "Qwen3.5-9B", prompt, 1, False)
                                for prompt in warm_prompts]
                            warm_results = [future.result() for future in futures]
                            concurrent_warmup_out.write(json.dumps({
                                "concurrency": concurrency, "phase": phase,
                                "ttft_ms": [round(result[1], 2) for result in warm_results],
                            }) + "\n")
                            concurrent_warmup_out.flush()
                            time.sleep(2)
                            reset_gpu_prefix_cache(args.vllm_port)
                        clear_cpu_cache()
                        jit_before = RESET_LOG.read_text().count(
                            "Triton kernel JIT compilation during inference")
                        measured_start = time.perf_counter()

                        waves = []
                        for repetition in range(args.online_repetitions):
                            by_turn = {}
                            for row in rows:
                                by_turn.setdefault(row["turn_index"], []).append(row)
                            for turn_rows in by_turn.values():
                                for offset in range(0, len(turn_rows), concurrency):
                                    waves.append((repetition, turn_rows[offset:offset + concurrency]))

                        for sweep_wave_index, (repetition, wave_rows) in enumerate(waves):
                            if (sweep_wave_index and
                                    repetition != waves[sweep_wave_index - 1][0]):
                                clear_cpu_cache()
                            indexed_rows = list(enumerate(wave_rows, start=request_index))
                            request_index += len(wave_rows)
                            profiling = (args.profile_request_index is not None and
                                         indexed_rows[0][0] == args.profile_request_index)
                            if profiling:
                                HTTP.post(f"http://127.0.0.1:{args.vllm_port}/start_profile",
                                          timeout=120).raise_for_status()
                            results = list(executor.map(run_one, indexed_rows))
                            if profiling:
                                HTTP.post(f"http://127.0.0.1:{args.vllm_port}/stop_profile",
                                          timeout=120).raise_for_status()
                            for index, row, usage, ttft_ms, elapsed_ms, first_text in results:
                                item = {
                                    "arrival_index": row["arrival_index"],
                                    "repetition": repetition,
                                    "session_id": row["session_id"],
                                    "turn_index": row["turn_index"],
                                    "wave_index": global_wave_index,
                                    "sweep_wave_index": sweep_wave_index,
                                    "wave_size": len(wave_rows),
                                    "concurrency": concurrency,
                                    "prompt_tokens": usage["prompt_tokens"],
                                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                                    "completion_tokens": usage["completion_tokens"],
                                    "elapsed_ms": round(elapsed_ms, 2),
                                    "ttft_ms": round(ttft_ms, 2),
                                    "first_text": first_text,
                                }
                                for key in ("conversation_turn", "request_kind", "source_event_index"):
                                    if key in row:
                                        item[key] = row[key]
                                if args.profile_request_index is not None:
                                    item["diagnostic_profile_run"] = True
                                if args.first_token_logprobs:
                                    item["first_logprobs"] = usage["diagnostic_first_logprobs"]
                                    item["generated_text"] = usage["diagnostic_generated_text"]
                                out.write(json.dumps(item) + "\n")
                                out.flush()
                                print(json.dumps(item), flush=True)
                            if args.reset_between_requests and sweep_wave_index + 1 < len(waves):
                                time.sleep(2)
                                attempts, wait_ms = reset_gpu_prefix_cache(args.vllm_port)
                                reset_item = {
                                    "wave_index": global_wave_index,
                                    "sweep_wave_index": sweep_wave_index,
                                    "concurrency": concurrency,
                                    "arrival_indices": [row["arrival_index"] for row in wave_rows],
                                    "attempts": attempts, "reset_wait_ms": wait_ms,
                                    "success": True,
                                }
                                reset_out.write(json.dumps(reset_item) + "\n")
                                reset_out.flush()
                                print(f"GPU prefix reset after c={concurrency} wave {sweep_wave_index} in {attempts} attempt(s); CPU LMCache retained", flush=True)
                            global_wave_index += 1
                        jit_after = RESET_LOG.read_text().count(
                            "Triton kernel JIT compilation during inference")
                        diagnostics_out.write(json.dumps({
                            "concurrency": concurrency,
                            "measured_wall_ms": round(
                                (time.perf_counter() - measured_start) * 1000, 2),
                            "formal_jit_warnings": jit_after - jit_before,
                        }) + "\n")
                        diagnostics_out.flush()
                return
        for phase, prompt_field in (("seed", "cache_prompt"), ("resume", "resume_prompt")):
            if not args.attach and vllm_process is None:
                log_name = "vllm.log" if args.reset_method == "local" else f"vllm_{phase}.log"
                vllm_process = start(serve, env, args.output_dir / log_name)
            wait_ready(vllm_process, f"http://127.0.0.1:{args.vllm_port}/health")
            with (args.output_dir / f"{phase}.jsonl").open("w") as out:
                for row in rows:
                    usage, ttft_ms, elapsed_ms, first_text = request(args.vllm_port, "Qwen3.5-9B", row[prompt_field])
                    item = {
                        "mode": args.mode, "phase": phase, "reset_method": args.reset_method,
                        "session_id": row["session_id"],
                        "turn_index": row.get("turn_index", row.get("round")),
                        "trace_prefix_tokens": row["cached_tokens"],
                        "prompt_tokens": usage["prompt_tokens"],
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                        "elapsed_ms": round(elapsed_ms, 2),
                        "ttft_ms": round(ttft_ms, 2),
                        "first_text": first_text,
                    }
                    out.write(json.dumps(item) + "\n")
                    out.flush()
                    print(json.dumps(item), flush=True)
            if phase == "seed":
                time.sleep(5)
                if args.reset_method == "local":
                    reset_gpu_prefix_cache(args.vllm_port)
                    print("Reset local vLLM prefix cache; LMCache CPU state retained", flush=True)
                else:
                    stop(vllm_process)
                    vllm_process = None
    finally:
        stop(vllm_process)
        stop(cache_process)


if __name__ == "__main__":
    main()
