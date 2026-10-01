# Deep KV: TTFT and token work

Two passes per arm, 4 real sessions × 10 turns; means, not medians.
Baseline forward tokens are derived from prompt minus reported cache hit; experimental forward tokens are measured from SINGLE_FORWARD logs.
K/V-only reuse is not whole-model forward skipping. Percentages use ratios of means.

Continuation mean TTFT (ms): {"baseline": 150.62458333333333, "old": 153.9675, "optimized": 163.16875}
Optimized vs native: -8.33% reduction; vs old deep: -5.98% reduction.

|Turn|Native ms|Old deep ms|Optimized ms|Gain vs native|Whole forward tokens saved|K/V projection tokens saved|
|---|---:|---:|---:|---:|---:|---:|
|1|191.62|190.25|192.41|-0.41%|0.00|0.00|
|2|128.44|142.06|146.17|-13.80%|0.00|248.00|
|3|138.64|134.02|147.14|-6.13%|0.00|308.00|
|4|163.97|169.11|176.09|-7.40%|0.00|408.00|
|5|151.19|156.26|162.42|-7.43%|0.00|252.00|
|6|139.87|136.81|150.29|-7.45%|0.00|200.00|
|7|148.63|155.04|169.17|-13.82%|0.00|296.00|
|8|181.56|183.88|191.65|-5.56%|0.00|316.00|
|9|148.03|148.34|156.89|-5.99%|0.00|108.00|
|10|155.29|160.18|168.69|-8.63%|0.00|272.00|

Exact session/turn hit boundaries, forward counts and paired mean TTFT are in per_session_turn.csv.
Native prefill budget=528; experimental=2048. Sequential crossover does not establish statistical significance or concurrency performance.
