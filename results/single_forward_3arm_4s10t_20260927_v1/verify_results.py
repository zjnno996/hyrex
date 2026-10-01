"""Read-only verification and paired, arithmetic-mean TTFT summary."""
import collections
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

data = {arm: rows(root / arm / "online.jsonl") for arm in ("baseline", "deep", "tail")}
baseline = data["baseline"]
assert len(baseline) == 120
print("arm | requests | mean_ms | continuation_mean_ms | continuation_reduction_pct")
base_mean = statistics.mean(r["ttft_ms"] for r in baseline if r["turn_index"] > 0)
for arm, measured in data.items():
    assert len(measured) == 120, (arm, len(measured))
    resets = rows(root / arm / "resets.jsonl")
    assert len(resets) == 119 and all(r["success"] for r in resets)
    assert len(rows(root / arm / "warmup.jsonl")) == 10
    for expected, actual in zip(baseline, measured):
        for key in ("session_id", "turn_index", "repetition", "prompt_tokens", "first_text"):
            assert actual[key] == expected[key], (arm, key, actual)
    continuation = [r for r in measured if r["turn_index"] > 0]
    mean = statistics.mean(r["ttft_ms"] for r in continuation)
    print(f"{arm} | 120 | {statistics.mean(r['ttft_ms'] for r in measured):.3f} | {mean:.3f} | {100*(base_mean-mean)/base_mean:.2f}")
    print("  repetition continuation means:", [round(statistics.mean(r["ttft_ms"] for r in continuation if r["repetition"] == i), 3) for i in range(3)])
    if arm != "baseline":
        log = (root / arm / "vllm.log").read_text()
        steps = re.findall(r"SINGLE_FORWARD request=(\S+) start=(\d+) count=(\d+)", log)
        by_request = collections.defaultdict(list)
        for request, start, count in steps:
            by_request[request].append((int(start), int(count)))
        formal = list(by_request.values())[-120:]
        assert len(formal) == 120 and all(len(s) == 1 for s in formal)
        for request, row in zip(formal, measured):
            start, count = request[0]
            assert start + count == row["prompt_tokens"]
        print("  continuation mean actual forward tokens:", round(statistics.mean(s[0][1] for s, r in zip(formal, measured) if r["turn_index"] > 0), 2))
        print("  checkpoint retirement log events:", (root / arm / "lmcache.log").read_text().count("CHECKPOINT_RETIRED"))
print("turn | baseline_ms | deep_ms | tail_ms")
for turn in range(10):
    print(turn + 1, *[round(statistics.mean(r["ttft_ms"] for r in measured if r["turn_index"] == turn), 2) for measured in data.values()], sep=" | ")
print("PASS: first-text agreement only; not a full numerical-equivalence proof. Budgets differ (528 vs 2048).")
