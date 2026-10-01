# 9B fixed-native / deep-KV / tail-state comparison, 2026-09-29

## Result

One real ShareGPT session WRAImOg_0, seed prompt 872 tokens followed by a
903-token continuation, three repetitions. All methods retain CPU cache within
a pair and reset GPU prefix cache before the continuation. CPU cache is cleared
between repetitions. Ten unrelated startup warmups per service; one generated
token per formal request. GPU 1, eager, identical CPU affinity 0-11,24-35.

| Resume metric | Fixed native | Deep KV + state replay | Deep KV + tail state |
|---|---:|---:|---:|
| Full KV hit | 528 | 864 | 864 |
| State hit | 528 | 528 | 864 |
| Whole-model forward tokens | 375 | 375 | 39 |
| Whole-model tokens saved vs native | 0 | 0 | 336 |
| Additional prefix positions avoiding K/V projection per Full layer | 0 | 336 | 336 |
| Mean TTFT ms | 125.60 | 137.14 | 155.86 |
| TTFT reduction vs native | — | -9.19% | -24.09% |

Negative reduction means slower: deep KV adds 11.54 ms; tail state adds 30.26 ms.
This run does NOT demonstrate average recovery acceleration. Deep KV replay
avoids K/V projection for 336 positions in each of 8 Full-Attention layers
(2688 token-layer projection positions), not 336 complete model forward tokens.
The tail-state variant avoids 336 whole-model forward tokens (89.6%), while
maintaining a new checkpoint at S896. These two savings columns overlap and
must not be added together.

First turns have zero cache hits and zero reused-prefix token savings for all
methods. Their scheduling differs (native budget 528, experimental 2048), so
seed TTFT is not evidence for recovery speedup.

## Every measured request

| Repeat | Turn | Prompt tokens | Native TTFT ms | Deep KV TTFT ms | Tail state TTFT ms |
|---|---|---:|---:|---:|---:|
| 1 | Seed | 872 | 198.81 | 148.32 | 174.28 |
| 1 | Resume | 903 | 135.52 | 129.00 | 166.36 |
| 2 | Seed | 872 | 192.56 | 147.66 | 173.31 |
| 2 | Resume | 903 | 103.17 | 139.57 | 149.28 |
| 3 | Seed | 872 | 188.29 | 167.20 | 133.71 |
| 3 | Resume | 903 | 138.11 | 142.85 | 151.94 |

There is material run-to-run variation; native repeat 2 is low and has NOT been
discarded. Three repetitions and one startup per method do not establish
statistical significance or a general slowdown. Order was deep, native, tail.
Tail uses the exact native observations above, not a second independent native
measurement. Changes in historical means cannot be assigned to a specific fix.

## Final audit

- Native vLLM HEAD 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 and LMCache HEAD
  140819c9d57a975dbc5678a6459a218e544cb58b remained clean before/after the run.
- Each of the three actual services completed 10 warmups and six formal requests.
  Five successful formal GPU resets per service cover every consecutive pair
  of requests; startup warmup caches are also cleared. GPU reset retains CPU.
- Deep logs contain three start=528,count=375 continuation forwards and three
  state=528,full_kv=864 hits. Tail logs contain three start=864,count=39 forwards,
  state=864,full=864 hits, and S896 checkpoint maintenance.
- Tail verified-lookup reuse is active in runtime logs; direct snapshot code is
  present in the frozen source manifest. No mid-run source changes: both
  manifests were recomputed against current files and matched after completion.
- All six paired first tokens match across all three methods. This is only a
  first-token smoke check, not proof of complete sequence/logit equivalence.
- Both runtime configs explicitly enforce eager; no CUDA Graph comparison bias.
- Processes completed and GPU 1 was released. No pure H2D DMA timing was measured;
  do not treat host retrieve-submission logs as H2D completion latency.

## Raw evidence

- Native: [online.jsonl](2_baseline/online.jsonl), [resets](2_baseline/resets.jsonl),
  [warmups](2_baseline/warmup.jsonl), [command](2_baseline/command.json).
- Deep: [online.jsonl](1_optimized/online.jsonl), [runtime](1_optimized/vllm.log),
  [resets](1_optimized/resets.jsonl), [command](1_optimized/command.json).
- Tail: [online.jsonl](../tail_verified_retest_20260929_v1/1_optimized/online.jsonl),
  [runtime](../tail_verified_retest_20260929_v1/1_optimized/vllm.log),
  [resets](../tail_verified_retest_20260929_v1/1_optimized/resets.jsonl),
  [baseline origin](../tail_verified_retest_20260929_v1/baseline_origin.txt).
