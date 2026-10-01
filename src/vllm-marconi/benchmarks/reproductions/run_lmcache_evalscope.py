# SPDX-License-Identifier: Apache-2.0
"""Run a small real-ShareGPT multi-turn stress test through EvalScope."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

try:
    from sharegpt_data import iter_records
    from sharegpt_mooncake_e2e import select_prefixes
except ImportError:  # Imported as benchmarks.reproductions.*
    from benchmarks.reproductions.sharegpt_data import iter_records
    from benchmarks.reproductions.sharegpt_mooncake_e2e import select_prefixes


def build_dataset(
    source: Path,
    output: Path,
    tokenizer_path: str,
    conversations: int,
    prompt_min_tokens: int,
    prompt_max_tokens: int,
) -> None:
    """Write long real-ShareGPT task prefixes as two-user-turn JSONL rows."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prefixes = select_prefixes(
        iter_records(source),
        tokenizer,
        count=conversations,
        min_tokens=prompt_min_tokens,
        max_tokens=prompt_max_tokens,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for prefix in prefixes:
            messages = [
                {"role": "user", "content": prefix.prompt},
                {"role": "assistant", "content": "Reference response."},
                {
                    "role": "user",
                    "content": "Continue the prior task with one concrete next step.",
                },
            ]
            file.write(json.dumps(messages, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server", default="http://127.0.0.1:8002/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--tokenizer", default="/root/models/Qwen3.5-9B")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/root/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"),
    )
    parser.add_argument(
        "--generated-dataset",
        type=Path,
        default=Path("/root/evalscope_sharegpt_8x2.jsonl"),
    )
    parser.add_argument("--conversations", type=int, default=8)
    parser.add_argument("--concurrencies", default="1,8")
    parser.add_argument("--prompt-min-tokens", type=int, default=784)
    parser.add_argument("--prompt-max-tokens", type=int, default=900)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--outputs-dir", type=Path, default=Path("/root/evalscope"))
    parser.add_argument("--name", default="qwen35-9b-lmcache")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (
        args.conversations < 1
        or args.output_tokens < 1
        or args.prompt_min_tokens < 1
        or args.prompt_max_tokens < args.prompt_min_tokens
    ):
        raise ValueError("invalid conversation, output, or prompt token bounds")
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if not concurrencies or min(concurrencies) < 1:
        raise ValueError("--concurrencies must contain positive integers")
    if max(concurrencies) > args.conversations:
        raise ValueError("--conversations must be at least the largest concurrency")

    build_dataset(
        args.dataset,
        args.generated_dataset,
        args.tokenizer,
        args.conversations,
        args.prompt_min_tokens,
        args.prompt_max_tokens,
    )
    if args.prepare_only:
        print(args.generated_dataset)
        return
    command = [
        str(Path(sys.executable).with_name("evalscope")),
        "perf",
        "--model",
        args.model,
        "--url",
        args.server,
        "--api",
        "openai",
        "--dataset",
        "custom_multi_turn",
        "--dataset-path",
        str(args.generated_dataset),
        "--multi-turn",
        "--max-turns",
        "2",
        "--number",
        *([str(args.conversations)] * len(concurrencies)),
        "--parallel",
        *[str(value) for value in concurrencies],
        "--max-tokens",
        str(args.output_tokens),
        "--stream",
        "--temperature",
        "0",
        "--tokenizer-path",
        args.tokenizer,
        "--outputs-dir",
        str(args.outputs_dir),
        "--name",
        args.name,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
