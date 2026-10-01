# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise CPU-offloaded prefix recovery with real ShareGPT turns.

The same client works with either the native vLLM CPU offloader
(``--kv-offloading-backend native``) or the MooncakeStoreConnector. Run
``warmup`` (or the backwards-compatible ``populate`` alias) on a deterministic
ShareGPT subset, then run ``load`` with the identical arguments. The latter
measures HTTP/request queueing, prefix matching, scheduler admission,
connector lookup, CPU-to-GPU materialization, prefill, and first-token
latency (TTFT). For native offload, warmup and load are intentionally kept in
one vLLM process so the CPU tier remains alive; the metrics endpoint can verify
that requests actually performed CPU->GPU H2D recovery.

The dataset is streamed with ijson and only the selected prefixes are retained
in memory; the 900 MiB source JSON is never loaded as one Python list.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

try:
    from sharegpt_data import iter_records
except ImportError:  # Imported as benchmarks.reproductions.*
    from benchmarks.reproductions.sharegpt_data import iter_records


ROLE_NAMES = {
    "human": "User",
    "user": "User",
    "gpt": "Assistant",
    "chatgpt": "Assistant",
    "bing": "Assistant",
    "bard": "Assistant",
    "system": "System",
}


@dataclass(frozen=True)
class ShareGPTPrefix:
    record_index: int
    token_count: int
    prompt_token_ids: tuple[int, ...]


def prefix_cache_path(
    dataset_path: Path,
    tokenizer_path: str,
    shared_prefix_tokens: int,
    suffix_tokens: int,
    start_index: int,
    distinct_suffix_prefix_tokens: int,
) -> Path:
    key = "|".join(
        [
            str(dataset_path.resolve()),
            tokenizer_path,
            str(shared_prefix_tokens),
            str(suffix_tokens),
            str(start_index),
            str(distinct_suffix_prefix_tokens),
            "tail-window-v2",
        ]
    ).encode()
    digest = hashlib.sha256(key).hexdigest()[:16]
    return Path("/tmp") / f"sharegpt_prefix_cache_{digest}.json"


def read_prefix_cache(path: Path, count: int) -> list[ShareGPTPrefix] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    prefixes = [
        ShareGPTPrefix(
            int(item["record_index"]),
            int(item["token_count"]),
            tuple(int(token) for token in item["prompt_token_ids"]),
        )
        for item in payload
    ]
    return prefixes[:count] if len(prefixes) >= count else None


def write_prefix_cache(path: Path, prefixes: list[ShareGPTPrefix]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "record_index": prefix.record_index,
                    "token_count": prefix.token_count,
                    "prompt_token_ids": list(prefix.prompt_token_ids),
                }
                for prefix in prefixes
            ],
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def select_partial_prefixes(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    count: int,
    shared_prefix_tokens: int,
    suffix_tokens: int,
    distinct_suffix_prefix_tokens: int = 0,
    skip: int = 0,
) -> list[ShareGPTPrefix]:
    """Build a two-tier prefix workload from real ShareGPT turns.

    The first ``shared_prefix_tokens`` come from one real ShareGPT prompt and
    are identical for every request.  Each request receives a different
    suffix, also taken from a real ShareGPT prompt.  The resulting workload
    is useful for testing a local-prefix hit followed by an LMCache CPU hit:
    the local worker can retain the shared prefix while the suffix is loaded
    from the external CPU tier.
    """
    if shared_prefix_tokens < 1 or suffix_tokens < 1:
        raise ValueError("shared prefix and suffix lengths must be positive")
    if count < 1 or skip < 0:
        raise ValueError("count must be positive and skip must be non-negative")
    if distinct_suffix_prefix_tokens < 0:
        raise ValueError("distinct suffix prefix length must be non-negative")
    if distinct_suffix_prefix_tokens > suffix_tokens:
        raise ValueError("distinct suffix prefix length exceeds suffix length")

    # Use two deterministic streaming passes.  This keeps the large source
    # JSON out of memory while making the base prefix independent of suffix
    # selection.
    candidates = select_prefixes(
        records,
        tokenizer,
        count=1,
        min_tokens=shared_prefix_tokens,
        max_tokens=max(shared_prefix_tokens + 4096, shared_prefix_tokens),
    )
    base_ids = list(candidates[0].prompt_token_ids)
    suffixes = select_prefixes(
        records,
        tokenizer,
        # A single ShareGPT conversation can yield many prompts with the same
        # leading history.  Gather extra candidates, then de-duplicate their
        # suffix windows below.
        count=count * 8,
        skip=skip + 1,
        min_tokens=suffix_tokens,
        max_tokens=max(suffix_tokens + 4096, suffix_tokens),
    )

    partial: list[ShareGPTPrefix] = []
    seen_suffixes: set[tuple[int, ...]] = set()
    for suffix in suffixes:
        suffix_ids = suffix.prompt_token_ids
        suffix_window = suffix_ids[-suffix_tokens:]
        cache_identity = (
            suffix_window[:distinct_suffix_prefix_tokens]
            if distinct_suffix_prefix_tokens
            else suffix_window
        )
        if cache_identity in seen_suffixes:
            continue
        seen_suffixes.add(cache_identity)
        # Send token ids to vLLM directly. Decoding and re-encoding text at
        # this boundary can change BPE merges and silently lose the GPU hit.
        prompt_ids = tuple(
            base_ids[:shared_prefix_tokens] + list(suffix_window)
        )
        partial.append(
            ShareGPTPrefix(
                record_index=suffix.record_index,
                token_count=len(prompt_ids),
                prompt_token_ids=prompt_ids,
            )
        )
        if len(partial) == count:
            return partial
    raise ValueError(
        f"found only {len(partial)} unique suffix windows; need {count}"
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def select_prefixes(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    count: int,
    skip: int = 0,
    min_tokens: int,
    max_tokens: int,
) -> list[ShareGPTPrefix]:
    if count < 1 or skip < 0:
        raise ValueError("count must be positive and skip must be non-negative")
    prefixes: list[ShareGPTPrefix] = []
    matched = 0
    for record_index, record in enumerate(records):
        history: list[str] = []
        saw_user = False
        saw_assistant = False
        for turn in record.get("conversations", []):
            role = ROLE_NAMES.get(turn.get("from"))
            content = turn.get("value")
            if role is None or not isinstance(content, str) or not content.strip():
                continue
            if role == "User" and saw_user and saw_assistant:
                prompt = "\n\n".join([*history, f"{role}: {content}"])
                token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
                if min_tokens <= token_count <= max_tokens:
                    if matched < skip:
                        matched += 1
                        history.append(f"{role}: {content}")
                        saw_user |= role == "User"
                        saw_assistant |= role == "Assistant"
                        continue
                    prefixes.append(
                        ShareGPTPrefix(
                            record_index,
                            token_count,
                            tuple(tokenizer.encode(prompt, add_special_tokens=False)),
                        )
                    )
                    if len(prefixes) == count:
                        return prefixes
            history.append(f"{role}: {content}")
            saw_user |= role == "User"
            saw_assistant |= role == "Assistant"
    raise ValueError(
        f"found only {len(prefixes)} ShareGPT prefixes in [{min_tokens}, "
        f"{max_tokens}] tokens; need {count}"
    )


def issue_request(
    server: str,
    model: str,
    prefix: ShareGPTPrefix,
    seed: int,
    output_tokens: int,
    request_timeout_s: float,
) -> tuple[str, str | None, float, float, str, int, str]:
    started = time.perf_counter()
    # Do not route localhost TTFT measurements through the shell's proxy.
    # Otherwise a proxy error is reported as a false 502 even when vLLM
    # completed the request successfully.
    session = requests.Session()
    session.trust_env = False
    with session.post(
        f"{server}/v1/completions",
        json={
            "model": model,
            "prompt": list(prefix.prompt_token_ids),
            "max_tokens": output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "seed": seed,
        },
        timeout=request_timeout_s,
        stream=True,
    ) as response:
        response.raise_for_status()
        ttft_ms = None
        first_token = None
        backend_request_id = None
        completion_tokens = None
        completion_parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            payload = json.loads(line[6:])
            backend_request_id = payload.get("id") or backend_request_id
            usage = payload.get("usage")
            if usage is not None and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = payload.get("choices") or []
            if choices:
                completion_parts.append(
                    choices[0].get("text") or choices[0].get("delta", {}).get(
                        "content", ""
                    )
                )
            if ttft_ms is None and choices:
                ttft_ms = (time.perf_counter() - started) * 1000
                choice = choices[0]
                first_token = choice.get("text") or choice.get("delta", {}).get(
                    "content", ""
                )
        if ttft_ms is None or first_token is None:
            raise RuntimeError("stream ended before the first completion token")
    total_ms = (time.perf_counter() - started) * 1000
    if completion_tokens is None:
        completion_tokens = output_tokens
    tpot_ms = (total_ms - ttft_ms) / max(completion_tokens - 1, 1)
    request_id = (
        f"{prefix.record_index}:"
        f"{hashlib.sha256(json.dumps(prefix.prompt_token_ids).encode()).hexdigest()[:16]}"
    )
    return (
        request_id,
        backend_request_id,
        ttft_ms,
        tpot_ms,
        first_token,
        completion_tokens,
        "".join(completion_parts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["warmup", "populate", "load"])
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/data/datasets/sharegpt/sharegpt_20230401_clean_lang_split.json"
        ),
    )
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--tokenizer", default="/data/models/Qwen3.5-27B")
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--requests",
        type=int,
        default=16,
        help="Number of ShareGPT prefixes in the bounded test set.",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=1,
        help="Concurrency for cache-populating warmup; sequential is safest.",
    )
    parser.add_argument(
        "--warmup-settle-seconds",
        type=float,
        default=2.0,
        help="Wait for asynchronous Mooncake stores after warmup requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed sampling seed sent with every deterministic request.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many deterministically selected prefixes.",
    )
    parser.add_argument("--min-tokens", type=int, default=784)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=1024,
        help="Fixed completion length; TTFT is measured from the streamed first token.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Per-request HTTP/stream timeout in seconds.",
    )
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Construct a shared real-ShareGPT prefix followed by unique suffixes.",
    )
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=0,
        help="Suffix length for --shared-prefix-tokens workloads.",
    )
    parser.add_argument(
        "--distinct-suffix-prefix-tokens",
        type=int,
        default=0,
        help=(
            "Require unique suffix prefixes of this physical cache-page length; "
            "useful when measured requests run sequentially."
        ),
    )
    parser.add_argument(
        "--prefix-only",
        action="store_true",
        help="For a partial workload, send only the shared prefix (GPU warmup stage).",
    )
    args = parser.parse_args()
    if (
        args.requests < 1
        or args.concurrency < 1
        or args.warmup_concurrency < 1
        or args.start_index < 0
        or args.warmup_settle_seconds < 0
        or args.output_tokens < 1
        or args.request_timeout <= 0
    ):
        raise ValueError(
            "requests and concurrency must be positive; start-index must be non-negative"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if bool(args.shared_prefix_tokens) != bool(args.suffix_tokens):
        raise ValueError("--shared-prefix-tokens and --suffix-tokens must be used together")
    if args.distinct_suffix_prefix_tokens and not args.shared_prefix_tokens:
        raise ValueError(
            "--distinct-suffix-prefix-tokens requires a partial prefix workload"
        )
    if args.shared_prefix_tokens:
        cache_path = prefix_cache_path(
            args.dataset_path,
            args.tokenizer,
            args.shared_prefix_tokens,
            args.suffix_tokens,
            args.start_index,
            args.distinct_suffix_prefix_tokens,
        )
        prefixes = read_prefix_cache(cache_path, args.requests)
        if prefixes is None:
            prefixes = select_partial_prefixes(
                iter_records(args.dataset_path),
                tokenizer,
                count=args.requests,
                skip=args.start_index,
                shared_prefix_tokens=args.shared_prefix_tokens,
                suffix_tokens=args.suffix_tokens,
                distinct_suffix_prefix_tokens=args.distinct_suffix_prefix_tokens,
            )
            write_prefix_cache(cache_path, prefixes)
        if args.prefix_only:
            shared_ids = prefixes[0].prompt_token_ids[: args.shared_prefix_tokens]
            prefixes = [
                ShareGPTPrefix(p.record_index, len(shared_ids), shared_ids)
                for p in prefixes
            ]
    else:
        if args.prefix_only:
            raise ValueError("--prefix-only requires a partial prefix workload")
        prefixes = select_prefixes(
            iter_records(args.dataset_path),
            tokenizer,
            count=args.requests,
            skip=args.start_index,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
    token_counts = [prefix.token_count for prefix in prefixes]
    print(
        f"selected {len(prefixes)} real ShareGPT prefixes; tokens p50/p90 = "
        f"{statistics.median(token_counts):.0f}/{percentile(token_counts, 0.9):.0f}"
    )

    # ``warmup`` and ``populate`` deliberately do not report these requests as
    # measurements.  Native CPU offloading keeps the CPU tier in the same vLLM
    # process, so run ``load`` against this process after GPU-cache eviction.
    # Mooncake deployments may restart the requester while keeping the
    # Mooncake owner alive.
    if args.mode in {"warmup", "populate"}:
        policy = os.environ.get("VLLM_MOONCAKE_HYBRID_POLICY", "all_load")
        if policy != "all_load":
            print(
                f"warning: warmup is running with policy={policy!r}; for a fair "
                "four-policy comparison, warm up once with all_load so both "
                "Full and Linear groups are written to Mooncake."
            )
        with ThreadPoolExecutor(
            max_workers=min(args.warmup_concurrency, len(prefixes))
        ) as pool:
            futures = [
                pool.submit(
                    issue_request,
                    args.server,
                    args.model,
                    prefix,
                    args.seed,
                    args.output_tokens,
                    args.request_timeout,
                )
                for prefix in prefixes
            ]
            completed = futures if args.warmup_concurrency == 1 else as_completed(futures)
            for future in completed:
                request_id, _, latency_ms, _, _, _, _ = future.result()
                print(f"warmup request={request_id} latency_ms={latency_ms:.3f}")
                # A completed HTTP response does not necessarily mean that
                # LMCache's asynchronous D2H store has finished reading the
                # request's GPU pages.  With sequential warmup, wait before
                # the next request can recycle those pages.
                if args.warmup_concurrency == 1 and args.warmup_settle_seconds:
                    time.sleep(args.warmup_settle_seconds)
        if args.warmup_concurrency != 1 and args.warmup_settle_seconds:
            time.sleep(args.warmup_settle_seconds)
        print(
            f"warmup complete: {len(prefixes)} prefixes written/attempted; "
            "run load in this process after GPU-cache eviction (or restart only "
            "the Mooncake requester while keeping its owner alive)."
        )
        return

    latencies: list[float] = []
    tpots: list[float] = []
    first_tokens: dict[str, str] = {}
    completion_texts: dict[str, str] = {}
    backend_request_ids: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(prefixes))) as pool:
        futures = [
            pool.submit(
                issue_request,
                args.server,
                args.model,
                prefix,
                args.seed,
                args.output_tokens,
                args.request_timeout,
            )
            for prefix in prefixes
        ]
        for future in as_completed(futures):
            (
                request_id,
                backend_request_id,
                latency_ms,
                tpot_ms,
                first_token,
                _,
                completion_text,
            ) = future.result()
            latencies.append(latency_ms)
            tpots.append(tpot_ms)
            first_tokens[request_id] = first_token
            completion_texts[request_id] = completion_text
            if backend_request_id is not None:
                backend_request_ids.append(backend_request_id)
            print(
                f"request={request_id} latency_ms={latency_ms:.3f} "
                f"tpot_ms={tpot_ms:.3f}"
            )
    wall_ms = (time.perf_counter() - started) * 1000
    print(
        f"mode={args.mode} requests={len(latencies)} concurrency={args.concurrency} "
        f"wall_ms={wall_ms:.3f} latency_ms_p50/p90/p99="
        f"{statistics.median(latencies):.3f}/{percentile(latencies, 0.9):.3f}/"
        f"{percentile(latencies, 0.99):.3f} tpot_ms_p50/p90/p99="
        f"{statistics.median(tpots):.3f}/{percentile(tpots, 0.9):.3f}/"
        f"{percentile(tpots, 0.99):.3f} first_token_signature="
        f"{hashlib.sha256(json.dumps(sorted(first_tokens.items())).encode()).hexdigest()[:16]}"
    )
    print(f"first_tokens_json={json.dumps(sorted(first_tokens.items()))}")
    print(f"completion_texts_json={json.dumps(sorted(completion_texts.items()))}")
    print(f"backend_request_ids_json={json.dumps(sorted(backend_request_ids))}")


if __name__ == "__main__":
    main()
