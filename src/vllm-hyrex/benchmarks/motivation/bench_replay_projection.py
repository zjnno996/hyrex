"""Shape-matched projection microbenchmark; not model TTFT or layer latency."""
import argparse
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    x = torch.randn(528, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(10240, 4096, device="cuda", dtype=torch.bfloat16) * .02

    def fused():
        return F.linear(x, w)

    def split(replay, no_cat=False):
        q = F.linear(x, w[:8192])
        kv = torch.zeros((528, 2048), device=x.device, dtype=x.dtype)
        kv[replay:] = F.linear(x[replay:], w[8192:])
        return (q, kv) if no_cat else torch.cat((q, kv), dim=-1)

    def measure(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 50

    results = []
    for replay in (112, 264, 416, 512):
        for _ in range(20):
            fused()
            split(replay)
            split(replay, True)
        a, b, c = [], [], []
        for repeat in range(8):
            variants = [(a, fused), (b, lambda: split(replay)),
                        (c, lambda: split(replay, True))]
            for samples, fn in variants[repeat % 3:] + variants[:repeat % 3]:
                samples.append(measure(fn))
        expected, actual = fused(), split(replay)
        separate = torch.cat(split(replay, True), dim=-1)
        assert torch.equal(actual, separate), "removing cat changed projected values"
        # The model also reshapes the interleaved Q/gate heads. Check the
        # different strides produced by separate buffers through that step.
        q_separate, kv_separate = split(replay, True)
        q_old, k_old, v_old = actual.split([8192, 1024, 1024], dim=-1)
        k_new, v_new = kv_separate.split(1024, dim=-1)
        for old, new in zip(q_old.view(528, 16, 512).chunk(2, -1),
                            q_separate.view(528, 16, 512).chunk(2, -1)):
            assert torch.equal(old.reshape(528, -1), new.reshape(528, -1))
        assert torch.equal(k_old, k_new) and torch.equal(v_old, v_new)
        results.append({"tokens": 528, "replay_tokens": replay,
            "fused_ms": statistics.mean(a), "split_qonly_ms": statistics.mean(b),
            "no_cat_ms": statistics.mean(c), "no_cat_samples_ms": c,
            "fused_samples_ms": a, "split_samples_ms": b,
            "q_gate_max_abs_difference": (actual[:, :8192]-expected[:, :8192]).abs().max().item(),
            "new_kv_max_abs_difference": (actual[replay:, 8192:]-expected[replay:, 8192:]).abs().max().item()})
    report = {"scope": "one Full Attention projection only, includes split/zero/cat overhead",
              "device": torch.cuda.get_device_name(), "results": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
