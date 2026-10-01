"""Check the selected 1054→1191 pair's model-visible state/Full KV audit."""
import argparse
import json
from pathlib import Path
import re


def verify(path):
    assert json.loads((path / "design.json").read_text())["diagnostic_byte_audit"]
    rows = [json.loads(x) for x in (path / "online.jsonl").read_text().splitlines()]
    resets = [json.loads(x) for x in (path / "resets.jsonl").read_text().splitlines()]
    assert len(resets) == len(rows)-1 and all(x["success"] for x in resets)
    requests = {}
    for line in (path / "vllm.log").read_text().splitlines():
        state = re.search(r"TAIL_MODEL_STATE request=(\S+) layer=(\S+) kind=(\d+) scheduled=(\d+) src=\d+ dst=\d+ sha256=(\w+)", line)
        full = re.search(r"TAIL_MODEL_FULL request=(\S+) layer=(\S+) scheduled=(\d+) hashes=([a-f0-9,]+)", line)
        if state:
            req, layer, kind, n, digest = state.groups()
            record = requests.setdefault(req, {"n": int(n), "state": {}, "full": {}})
            record["state"][layer, kind] = digest
        if full:
            req, layer, n, digests = full.groups()
            record = requests.setdefault(req, {"n": int(n), "state": {}, "full": {}})
            record["full"][layer] = digests.split(",")
    seed, comparisons = None, []
    for record in requests.values():
        if record["n"] == 14:
            seed = record
        if record["n"] == 151:
            assert seed is not None
            for item in (seed, record):
                assert len(item["state"]) == 48 and len(item["full"]) == 8
                assert all(len(pages) == 65 for pages in item["full"].values())
            assert seed["state"] == record["state"], "model state mismatch"
            assert seed["full"] == record["full"], "model Full KV mismatch"
            comparisons.append({"state_tensors": 48, "full_layer_pages": 520})
            seed = None
    resumes = [x for x in rows if x["turn_index"] == 1]
    assert comparisons and len(comparisons) == len(resumes) == len(rows)//2
    assert all(x["cached_tokens"] == 1040 for x in resumes)
    # Same-segmentation cold reference for this selected pair, not a general
    # demand for bit equality between different chunking algorithms.
    reference = [json.loads(x) for x in Path(
        "/root/hyrex_results/tail_probe_cold_fused_reference_20260927_v1/online.jsonl"
    ).read_text().splitlines() if json.loads(x)["turn_index"] == 1]
    assert len({x["generated_text"] for x in reference}) == 1
    assert all(x["first_logprobs"] == reference[0]["first_logprobs"]
               and x["generated_text"] == reference[0]["generated_text"] for x in resumes)
    return {"repetitions": len(resumes), "verified_gpu_resets": len(resets),
            "model_bytes_match": True, "cold_reference_sampled_output_match": True,
            "comparisons": comparisons, "performance_measurement": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(args.run), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)
