#!/usr/bin/env python3
"""Fail-fast resource preflight for a single Hybrid recovery cell.

This does not start vLLM or allocate a cache.  It only checks whether an
otherwise idle GPU can honor vLLM's memory-utilization target and whether the
host has room for one LMCache CPU tier plus a safety margin.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MIB = 1024**2
GIB = 1024**3


def gpu_memory_mib(index: int) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    total, free = (int(value.strip()) for value in output.split(","))
    return total, free


def host_available_bytes() -> int:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return values["MemAvailable"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--lmcache-kv-gb", type=float, default=24.0)
    parser.add_argument("--host-margin-gb", type=float, default=4.0)
    parser.add_argument("--gpu-start-margin-mib", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if min(args.lmcache_kv_gb, args.host_margin_gb, args.gpu_start_margin_mib) < 0:
        raise SystemExit("resource sizes must be non-negative")

    total_mib, free_mib = gpu_memory_mib(args.gpu)
    target_mib = total_mib * args.gpu_memory_utilization
    needed_gpu_mib = target_mib + args.gpu_start_margin_mib
    available_host = host_available_bytes()
    needed_host = int((args.lmcache_kv_gb + args.host_margin_gb) * GIB)
    gpu_ok = free_mib >= needed_gpu_mib
    host_ok = available_host >= needed_host

    print(
        "GPU%d: free=%.2f GiB target=%.2f GiB startup_margin=%d MiB => %s"
        % (
            args.gpu,
            free_mib / 1024,
            target_mib / 1024,
            args.gpu_start_margin_mib,
            "PASS" if gpu_ok else "FAIL",
        )
    )
    print(
        "Host: available=%.2f GiB tier=%.2f GiB margin=%.2f GiB => %s"
        % (
            available_host / GIB,
            args.lmcache_kv_gb,
            args.host_margin_gb,
            "PASS" if host_ok else "FAIL",
        )
    )
    if not (gpu_ok and host_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
