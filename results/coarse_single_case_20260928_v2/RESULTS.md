# Single real-session case

> INVALID DEEP-KV COMPARISON: experimental hit lengths were 864/528/528.
> See INVALID_DEEP_COMPARISON.md. Use the corrected v3 run instead.

Session: WRAImOg_0. Three seed/resume repetitions per arm; 10 unrelated startup warmups; GPU reset between requests; CPU reset between repetitions.

Resume mean TTFT: native 125.043 ms; deep 136.177 ms; reduction -8.90%.

First generated token agrees; this does not establish full-sequence equivalence. One startup per arm, deep then native: preliminary case, not a robust speedup claim. Native prefill budget=528; deep=2048. Raw per-request data in online.jsonl.
