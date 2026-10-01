"""Correlate CUDA kernels to prefill aten::mm shapes in a diagnostic trace."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert json.loads((args.run / "design.json").read_text())["diagnostic_profile_run"]
    traces = list((args.run / "profile").glob("*.pt.trace.json.gz"))
    assert len(traces) == 1
    events = json.loads(gzip.decompress(traces[0].read_bytes()))["traceEvents"]
    mm = {e["args"]["External id"]: e for e in events
          if e.get("cat") == "cpu_op" and e.get("name") == "aten::mm"}
    launches = {e["args"]["correlation"]: e["args"].get("External id") for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    shapes = defaultdict(lambda: {"kernel_count": 0, "kernel_ms_sum": 0.0})
    for event in events:
        if event.get("cat") != "kernel":
            continue
        op = mm.get(launches.get(event.get("args", {}).get("correlation")))
        if op is None:
            continue
        dims = op["args"].get("Input Dims", [])
        if not dims or dims[0][0] <= 1:
            continue  # exclude the 31 decode iterations
        group = shapes[str(dims)]
        group["kernel_count"] += 1
        group["kernel_ms_sum"] += event["dur"] / 1000
    assert shapes["[[528, 4096], [4096, 24576]]"]["kernel_count"] == 32
    assert shapes["[[528, 12288], [12288, 4096]]"]["kernel_count"] == 32
    report = {"trace": str(traces[0]), "scope": "profiled kernel sums, not TTFT; profiler can perturb timing",
              "prefill_mm_shapes": dict(shapes)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
