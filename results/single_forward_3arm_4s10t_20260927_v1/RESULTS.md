# 9B real-session recovery validation

4 real ShareGPT split-dataset sessions, first 10 turns each, 3 repetitions per arm: 360 sequential measured requests. The tenth prompts contain 1922, 1954, 1740 and 1947 tokens. History uses dataset reference answers, not newly generated model answers. Measurement requests generate one token.

Each service startup has 10 unrelated warmup requests with up to 32 generated tokens. Every arm has 119 successful GPU-prefix resets between its 120 measured requests; CPU LMCache is retained between turns and cleared between repetitions. Reset checks verify engine acknowledgement. GPU reset here means prefix-cache clearing, not a device reset.

| Arm | All-request mean TTFT (ms) | Turns 2–10 mean TTFT (ms) | Change vs native, turns 2–10 |
|---|---:|---:|---:|
| Pristine vLLM + LMCache | 159.527 | 155.882 | reference |
| Deeper independent KV + coarse-state replay | 160.437 | 156.767 | 0.57% slower |
| Last state moved to complete KV boundary | 176.582 | 175.919 | 12.85% slower |

Continuation means by repetition: native 157.250 / 156.461 / 153.935 ms; deep 157.419 / 152.084 / 160.798 ms; tail 176.771 / 171.271 / 179.715 ms.

| Turn | Native mean ms | Deep mean ms | Tail mean ms |
|---|---:|---:|---:|
| 1 | 192.34 | 193.46 | 182.55 |
| 2 | 140.02 | 137.25 | 165.44 |
| 3 | 139.45 | 137.12 | 177.46 |
| 4 | 175.67 | 171.41 | 172.38 |
| 5 | 149.95 | 148.54 | 174.87 |
| 6 | 141.85 | 147.92 | 174.97 |
| 7 | 155.04 | 156.26 | 177.61 |
| 8 | 183.39 | 182.36 | 173.32 |
| 9 | 147.58 | 158.29 | 185.80 |
| 10 | 169.99 | 171.75 | 181.43 |

## Mechanism and correctness checks

- All 120 first-token texts in each experimental arm match the corresponding native requests. This is not proof of full-output or numerical equivalence.
- Both experimental arms have exactly one logged model forward per measured request; start + count equals the prompt token count in every case.
- Actual continuation forward lengths average 400.03 tokens for deep and 132.47 for tail (66.9% fewer). This compares experimental arms, not an independently instrumented native-forward token count.
- Q-only Full-Attention replay activation is logged in deep. Cached K/V does not remove Q, attention, MLP, or recurrent computation needed to reconstruct missing state.
- Tail logs 252 checkpoint-retirement events, including warmup. This confirms the retirement path executes, not its timing cost or peak memory footprint.
- Native source trees remain unmodified. Source commits and exact commands are recorded in design.json and arm command.json files.

## Interpretation and limitations

On this short-context workload, deeper KV reuse does not show an aggregate TTFT gain. Tail state substantially reduces forward tokens but is slower end-to-end in this implementation. These measurements do not identify the causal time split among lookup, H2D, checkpoint capture/store, GPU computation and host scheduling; a separate diagnostic profile is needed before assigning the regression to one component.

Native prefill budget is 528, required by its connector checkpoint contract; experimental budget is 2048 with in-layer checkpoint capture. This is a combined-method comparison, not an equal-budget ablation. Arms run sequentially, not randomized; no claim of statistical significance for the 0.57% deep difference. Results apply to four short real conversation fragments, not long-context/general workloads.

Recheck with `python verify_results.py` in this directory. The verification script is the only added analysis code; no benchmark source was changed during this frozen run.
