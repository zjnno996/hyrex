"""Paired 40-request summary; keep outliers and use continuation means."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    rows = [json.loads(s) for s in (path / "online.jsonl").read_text().splitlines()]
    resets = [json.loads(s) for s in (path / "resets.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and len(resets) == 39 and all(r["success"] for r in resets)
    assert not any(r.get("diagnostic_profile_run") for r in rows)
    keyed = {(r["session_id"], r["turn_index"]): r for r in rows}
    assert len(keyed) == 40
    return keyed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shallow", type=Path)
    parser.add_argument("deep_qkv", type=Path)
    parser.add_argument("deep_qonly", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {arm: read(getattr(args, arm)) for arm in ("shallow", "deep_qkv", "deep_qonly")}
    base = runs["shallow"]
    keys = [k for k in base if k[1] > 0]
    assert len(keys) == 36 and all(r.keys() == base.keys() for r in runs.values())
    assert all(runs["deep_qkv"][k]["cached_tokens"] == runs["deep_qonly"][k]["cached_tokens"] for k in base)
    result = {"scope": "one sequential 4x10 pass per arm; matched experimental stack",
              "outliers_removed": False, "arms": {}}
    for arm, rows in runs.items():
        samples = [rows[k]["ttft_ms"] for k in keys]
        result["arms"][arm] = {
            "resume_count": len(keys), "mean_ttft_ms": statistics.mean(samples),
            "stdev_ttft_ms": statistics.stdev(samples), "max_ttft_ms": max(samples),
            "wins_vs_shallow": sum(rows[k]["ttft_ms"] < base[k]["ttft_ms"] for k in keys),
            "text_mismatches_vs_shallow": [list(k) for k in base
                if rows[k]["generated_text"] != base[k]["generated_text"]],
            "session_means_ms": {sid: statistics.mean(rows[k]["ttft_ms"] for k in keys if k[0] == sid)
                                 for sid in sorted({k[0] for k in keys})},
        }
    gaps = [runs["deep_qonly"][k]["cached_tokens"]-base[k]["cached_tokens"] for k in keys]
    result["extra_full_kv_tokens_mean"] = statistics.mean(gaps)
    result["extra_full_h2d_MiB_mean"] = statistics.mean(gaps) / 32
    a, b = (result["arms"][x]["mean_ttft_ms"] for x in ("shallow", "deep_qonly"))
    result["qonly_saving_ms"] = a-b
    result["qonly_saving_percent"] = (a-b)/a*100
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
