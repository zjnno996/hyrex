# Exact short-tail recurrence ablation (eager, Qwen3.5-9B)

## Outcome

| Metric | Existing tail checkpoint implementation | Short recurrent tail enabled |
|---|---:|---:|
| Measured requests | 40 | 40 |
| All-request mean TTFT | 181.273 ms | 159.294 ms |
| Turns 2–10 mean TTFT | 179.430 ms | 157.149 ms |
| Formal JIT warnings | 0 | 0 |
| Successful inter-request GPU prefix resets | 39 | 39 |

Observed continuation reduction: 22.281 ms (12.42%). This is one sequential control/treatment pass, not a randomized repeated estimate or proof of a general speedup. No native baseline was rerun in this ablation; the optimized result must not be claimed to beat native based on previous runs.

Important negative control: session WnjND3T_0 turn 8 has a 1584-token prompt and no post-checkpoint tail, so the new short-tail path is not selected. Its TTFT nevertheless changes from 174.39 to 157.24 ms. One such request cannot quantify drift, but warns against attributing the entire 22.281-ms difference to this change. A reverse-order follow-up is required and has been launched separately.

## Single changed factor

Both arms run source commit `f8814fcfc` on branch `motivation/short-tail-recurrent-20260927`, with the same LMCache checkout and original eager harness configuration (GPU utilization 0.8, CPU cache 2 GiB, prefill budget 2048). Only `VLLM_HYREX_SHORT_TAIL` changes from 0 to 1. Every service startup has ten unrelated warmup requests. GPU prefix cache is cleared between each measured request, CPU retained between turns. Four real ShareGPT split sessions, ten turns each, one generated token per measured request.

The final complete-KV-boundary checkpoint is unchanged. The portion before it still uses the existing chunk algorithm. Only the last 1–15 tokens after that checkpoint use the existing fused recurrent kernel with a new opt-in final-state-only store. There is no removal of checkpoint maintenance, storage, or lookup. The short kernel uses a separate output state; it does not mutate the checkpoint.

This is not a single-kernel intermediate-checkpoint implementation: it still has a long-segment call and a short-segment call. It tests whether the second full chunk call is unnecessary overhead.

## Correctness and mechanism checks

- All lengths 1–15 tested at the model's 16 key heads / 32 value heads / 128 dimensions against an explicit FP32 recurrence. Final-state relative L2 error <1e-4, output error <0.004; initial state unchanged.
- Segmented cases 79/64, 151/144, 1054/1040, 1071/(528,1056), 1056/(528,1056) tested. Saved checkpoints match the old implementation elementwise. Final-state BF16/chunk-vs-recurrent differences were about 0.2–0.3%, not bitwise equality.
- Existing recurrent/convolution checkpoint tests pass.
- A separate real 1054→1191-token model smoke produced exactly the same 32-token generated text as the saved previous tail smoke.
- Every first-token text in both 40-request arms matches the frozen native eager reference and the other arm. Every cached-token count matches between the two arms.
- Both arms log one forward per measured request. These finite checks are not a proof of numerical equivalence for all prompts, lengths, or backends.

## Component evidence

An alternating-order warmed GPU test of the 79-token (64+15) recurrence component repeated across 24 simulated layers measured 23.824 ms for two chunk calls versus 15.287 ms for chunk+short recurrence. Includes Python submission and one synchronization per 24 calls; excludes the rest of the model and LMCache. Do not add/subtract this component timing from end-to-end TTFT as a strict decomposition.

## Per-turn mean TTFT (four sessions)

| Turn | Control ms | Short-tail ms |
|---|---:|---:|
| 1 | 197.86 | 178.59 |
| 2 | 153.75 | 140.72 |
| 3 | 193.79 | 155.63 |
| 4 | 198.92 | 150.43 |
| 5 | 173.75 | 156.05 |
| 6 | 176.08 | 142.16 |
| 7 | 182.10 | 165.53 |
| 8 | 177.63 | 171.52 |
| 9 | 166.44 | 163.44 |
| 10 | 192.39 | 168.87 |

Reproduce correctness/component test: `CUDA_VISIBLE_DEVICES=1 PYTHONSAFEPATH=1 PYTHONPATH=/root/exp-vllm-short-tail /root/hybrid-model-offloading/.venv/bin/python /root/exp-vllm-short-tail/benchmarks/motivation/test_short_tail.py`.

The model-run commands and feature switch are recorded in each arm's `command.json`. Pristine vLLM and LMCache sources were not modified.
