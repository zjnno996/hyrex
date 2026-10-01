#!/usr/bin/env python3
"""Compute the reusable-prefix loss caused by stock 528-token alignment."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--lengths", default="1584,1712,1840,1968,2064,2111,2112")
parser.add_argument("--output", type=Path)
args = parser.parse_args()

rows = []
for prefix in map(int, args.lengths.split(",")):
    full_kv_boundary = prefix // 16 * 16
    aligned_boundary = prefix // 528 * 528
    rows.append(
        {
            "prefix_tokens": prefix,
            "independent_full_kv_boundary": full_kv_boundary,
            "stock_common_boundary": aligned_boundary,
            "stranded_full_kv_tokens": full_kv_boundary - aligned_boundary,
            "common_boundary_utilization": aligned_boundary / prefix,
        }
    )

text = "".join(json.dumps(row) + "\n" for row in rows)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
print(text, end="")
