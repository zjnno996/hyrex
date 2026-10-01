# Native versus deep KV: telemetry ABBA

Continuation mean TTFT: native 160.530 ms; deep 162.833 ms; reduction -1.43%.

160 requests: native/deep/deep/native, 4 sessions × 10 turns per arm. GPU prefix clear after every request except last; 10 unrelated warmups per launch. CPU affinity node 0, memory binding not enforced. Native sources unchanged. Budgets remain native 528, deep 2048.

NVML 100-ms sampling is device-wide and may miss short DMA bursts. RX/TX counters are throughput samples, not exact per-request bytes or H2D latency. No profiler was enabled for these TTFT measurements.

Deep forward counts verified from current SINGLE_FORWARD logs; native counts inferred from prompt minus common hit. Complete model forward token saving remains zero. Extra Full KV bytes are BF16 logical payload (32 KiB/token), not measured PCIe traffic.

|Turn|Native ms|Deep ms|Reduction %|K/V projection tokens saved|
|---|---:|---:|---:|---:|
|1|201.91|201.51|0.20|0.00|
|2|136.16|149.07|-9.48|248.00|
|3|143.39|146.05|-1.85|308.00|
|4|175.81|180.22|-2.51|408.00|
|5|162.30|159.96|1.44|252.00|
|6|148.86|153.25|-2.95|200.00|
|7|164.25|167.42|-1.93|296.00|
|8|186.54|188.81|-1.22|316.00|
|9|152.98|152.42|0.37|108.00|
|10|174.48|168.30|3.54|272.00|
