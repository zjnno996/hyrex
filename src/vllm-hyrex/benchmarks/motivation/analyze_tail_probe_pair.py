"""Fail-closed paired summary; exclude repetition zero for shape warmup."""
import argparse
import json
import statistics
from pathlib import Path


def read_run(path, boundary):
    assert not json.loads((path / "design.json").read_text()).get("diagnostic_profile_run"), \
        "profiler runs are not performance measurements"
    assert not json.loads((path / "design.json").read_text()).get("diagnostic_byte_audit"), \
        "byte-audit runs synchronize GPU/D2H and are not performance measurements"
    rows = [json.loads(line) for line in (path / "online.jsonl").read_text().splitlines()]
    pairs = {}
    for row in rows:
        rep = row["repetition"]
        if row["turn_index"] == 1:
            assert row["cached_tokens"] == boundary, row
        pairs.setdefault(rep, {})[row["turn_index"]] = row
    assert all(set(p) == {0, 1} for p in pairs.values())
    resets = [json.loads(line) for line in (path / "resets.jsonl").read_text().splitlines()]
    assert len(resets) == len(rows)-1 and all(r["success"] for r in resets)
    return {rep: pair for rep, pair in pairs.items() if rep > 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shallow", type=Path)
    p.add_argument("tail", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    design = json.loads((args.tail / "design.json").read_text())
    a = read_run(args.shallow, design["coarse_boundary"])
    b = read_run(args.tail, design["tail_boundary"])
    assert a and a.keys() == b.keys()
    mismatches = [dict(repetition=r, turn=t) for r in a for t in (0, 1)
                  if a[r][t]["generated_text"] != b[r][t]["generated_text"]]
    result = {"scope": "one selected real pair, not workload average",
              "repetitions": len(a), "generated_text_mismatches": mismatches,
              "saved_forward_tokens": design["tail_boundary"]-design["coarse_boundary"],
              "extra_checkpoint_MiB": 49.5}
    for name, runs in (("shallow", a), ("tail", b)):
        result[name] = {
            "seed_ttft_mean_ms": statistics.mean(x[0]["ttft_ms"] for x in runs.values()),
            "resume_ttft_mean_ms": statistics.mean(x[1]["ttft_ms"] for x in runs.values()),
            "pair_elapsed_mean_ms": statistics.mean(
                x[0]["elapsed_ms"]+x[1]["elapsed_ms"] for x in runs.values()),
            "resume_ttft_samples_ms": [x[1]["ttft_ms"] for x in runs.values()],
        }
    result["resume_ttft_saving_ms"] = result["shallow"]["resume_ttft_mean_ms"]-result["tail"]["resume_ttft_mean_ms"]
    result["seed_ttft_cost_ms"] = result["tail"]["seed_ttft_mean_ms"]-result["shallow"]["seed_ttft_mean_ms"]
    result["sampled_output_checks_passed"] = not mismatches
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
