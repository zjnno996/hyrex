#!/usr/bin/env python3
"""Measure stock vLLM+LMCache prefix loss from hybrid block alignment."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen3.5-9B")
    parser.add_argument("--lengths", default="1584,1712,1840,1968,2064,2112")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-cache-gb", type=float, default=4.0)
    args = parser.parse_args()

    os.environ["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).parent), os.environ.get("PATH", ""))
    )
    os.environ.update(
        {
            "LMCACHE_CHUNK_SIZE": "528",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.cpu_cache_gb),
            "LMCACHE_SAVE_UNFULL_CHUNK": "True",
            "LMCACHE_LOG_LEVEL": "INFO",
        }
    )

    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    lengths = [int(value) for value in args.lengths.split(",")]
    llm = LLM(
        model=args.model,
        language_model_only=True,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(lengths) + 128,
        max_num_batched_tokens=528,
        gpu_memory_utilization=0.82,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
            kv_connector_extra_config={"discard_partial_chunks": False},
        ),
    )
    params = SamplingParams(temperature=0, max_tokens=1)
    vocab_size = llm.get_tokenizer().vocab_size
    rows = []

    try:
        for case, shared_tokens in enumerate(lengths):
            # A per-case first token prevents hits from earlier sweep points.
            marker = 1000 + case
            common = [marker] + [100 + (i % 1000) for i in range(shared_tokens - 1)]
            seed = common + [2001] * 32
            resume = common + [2002] * 32
            assert max(seed + resume) < vocab_size

            llm.generate([TokensPrompt(prompt_token_ids=seed)], params, use_tqdm=False)
            if not llm.reset_prefix_cache(reset_connector=False):
                raise RuntimeError("local prefix-cache reset failed")

            start = time.perf_counter()
            output = llm.generate(
                [TokensPrompt(prompt_token_ids=resume)], params, use_tqdm=False
            )[0]
            elapsed = time.perf_counter() - start
            cached = output.num_cached_tokens or 0
            row = {
                "shared_tokens": shared_tokens,
                "cached_tokens": cached,
                "stranded_tokens": shared_tokens - cached,
                "cache_utilization": cached / shared_tokens,
                "restore_seconds": elapsed,
                "stock_alignment_tokens": 528,
                "full_attention_alignment_tokens": 16,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    finally:
        del llm
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
