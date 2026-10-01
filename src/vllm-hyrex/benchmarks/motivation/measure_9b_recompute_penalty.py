#!/usr/bin/env python3
"""Measure Qwen3.5-9B TTFT cost after a 528-aligned common boundary."""

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams, TokensPrompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen3.5-9B")
    parser.add_argument("--boundary", type=int, default=1584)
    parser.add_argument("--gaps", default="16,128,256,384,480,512")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gaps = [int(value) for value in args.gaps.split(",")]
    llm = LLM(
        model=args.model,
        language_model_only=True,
        dtype="bfloat16",
        max_model_len=args.boundary + max(gaps) + 64,
        max_num_batched_tokens=528,
        gpu_memory_utilization=0.82,
        enable_prefix_caching=True,
        disable_hybrid_kv_cache_manager=False,
    )
    params = SamplingParams(temperature=0, max_tokens=1)
    common = [100 + (i % 1000) for i in range(args.boundary)]

    # Materialize the recurrent checkpoint and Full KV at the common boundary.
    llm.generate([TokensPrompt(prompt_token_ids=common)], params, use_tqdm=False)
    rows = []
    for gap in gaps:
        samples = []
        cached_values = []
        for repetition in range(args.repetitions):
            # Different branch tokens prevent a previous repetition from becoming
            # a deeper hit while retaining the same common prefix.
            suffix = [2000 + repetition * 16 + gap // 16] * gap
            start = time.perf_counter()
            output = llm.generate(
                [TokensPrompt(prompt_token_ids=common + suffix)],
                params,
                use_tqdm=False,
            )[0]
            samples.append(time.perf_counter() - start)
            cached_values.append(output.num_cached_tokens or 0)
        row = {
            "common_boundary_tokens": args.boundary,
            "stranded_recompute_tokens": gap,
            "cached_tokens": cached_values,
            "ttft_seconds": samples,
            "ttft_median_seconds": statistics.median(samples),
            "ttft_mean_seconds": statistics.mean(samples),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
