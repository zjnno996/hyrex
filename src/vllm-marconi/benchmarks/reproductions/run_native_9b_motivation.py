#!/usr/bin/env python3
"""Single-GPU Qwen3.5-9B CPU-recovery motivation run (one request at a time)."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
MODEL = "/root/models/Qwen3.5-9B"


def completion(http, port, prompt):
    start = time.perf_counter()
    response = http.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json={
            "model": "Qwen3.5-9B",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    ttft = None
    usage = None
    token = None
    for line in response.iter_lines(decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode()
        if not line or not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        if event.get("choices"):
            choice = event["choices"][0]
            if choice.get("text") and ttft is None:
                ttft = (time.perf_counter() - start) * 1000
                token = choice["text"]
        usage = event.get("usage") or usage
    response.close()
    if ttft is None or usage is None:
        raise RuntimeError("missing streamed first token or usage")
    return round(ttft, 2), usage, token


def reset(http, port, *, external):
    response = http.post(
        f"http://127.0.0.1:{port}/reset_prefix_cache",
        params={"reset_external": str(external).lower()},
        timeout=90,
    )
    response.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("all_load", "independent_full_kv"), required=True)
    parser.add_argument("--trace", type=Path, default=Path("/root/hyrex_results/motivation_sharegpt_4s10t_trace.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8567)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--recurrent-cap", type=int, default=528)
    parser.add_argument("--startup-timeout", type=int, default=5400)
    args = parser.parse_args()
    if args.recurrent_cap < 0 or args.limit < 1:
        parser.error("--recurrent-cap must be non-negative and --limit positive")

    rows = [json.loads(line) for line in args.trace.read_text().splitlines()][: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        VLLM_SERVER_DEV_MODE="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_MOONCAKE_HYBRID_POLICY=args.policy,
        VLLM_HYREX_EXPERIMENT_RECURRENT_STORE_CAP=str(args.recurrent_cap),
    )
    command = [
        str(Path(sys.executable).with_name("vllm")), "serve", MODEL,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", "Qwen3.5-9B", "--dtype", "bfloat16",
        "--language-model-only", "--enable-prefix-caching",
        "--enable-prompt-tokens-details", "--mamba-cache-mode", "align",
        "--max-model-len", "4096", "--max-num-batched-tokens", "1055",
        "--gpu-memory-utilization", "0.82", "--enforce-eager",
        "--safetensors-load-strategy", "eager",
        "--kv-offloading-size", "2", "--kv-offloading-backend", "native",
    ]
    http = requests.Session()
    http.trust_env = False
    with (args.output_dir / "vllm.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"9B server exited: {process.returncode}")
            try:
                if http.get(f"http://127.0.0.1:{args.port}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("9B server startup")

        # Distinct first token prevents overlap with ShareGPT prefixes. Cover
        # both cold prefill and CPU recovery at the measured prompt lengths.
        with (args.output_dir / "warmup.jsonl").open("w") as warmup_out:
            for i, length in enumerate((651, 1191, 1708, 1922)):
                prompt = [1000 + i] + [2000 + i] * (length - 1)
                for repeat in range(2):
                    ttft, usage, _ = completion(http, args.port, prompt)
                    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    record = {"length": length, "repeat": repeat, "ttft_ms": ttft, "cached_tokens": cached}
                    warmup_out.write(json.dumps(record) + "\n")
                    warmup_out.flush()
                    print(f"warmup {record}", flush=True)
                    if repeat and cached == 0:
                        raise RuntimeError(f"warmup CPU recovery missed at length={length}")
                    time.sleep(1)
                    reset(http, args.port, external=False)
        time.sleep(2)
        reset(http, args.port, external=True)  # remove warmup CPU/GPU objects
        time.sleep(2)

        with (args.output_dir / "online.jsonl").open("w") as out:
            for index, row in enumerate(rows):
                ttft, usage, token = completion(http, args.port, row["resume_prompt"])
                record = {
                    "arrival_index": row["arrival_index"],
                    "session_id": row["session_id"],
                    "turn_index": row["turn_index"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                    "ttft_ms": ttft,
                    "first_token": token,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if index + 1 < len(rows):
                    time.sleep(2)
                    reset(http, args.port, external=False)  # keep CPU objects
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
