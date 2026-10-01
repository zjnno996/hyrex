# One-case deep KV versus fixed native baseline

This is a new measurement, not a reuse of older timing numbers.

- Model: Qwen3.5-9B, GPU 1, eager, one request at a time.
- Real ShareGPT session: `WRAImOg_0`, first two turns, repeated three times.
- Selection criterion: largest initial fine-KV/coarse-state boundary gap among the four existing sessions. No latency-based selection.
- Fixed native sources: `/root/exp-vllm-pristine-3way` and `/root/exp-lmcache-pristine-3way`; both verified clean against HEAD before measurement.
- Experimental sources: `exp-vllm-deep-overhead` and `exp-lmcache-deep-overhead`, with coarse CPU objects and fine prefix indexing enabled.
- Ten unrelated warmup requests on each service startup. Clear GPU and CPU caches after warmup.
- Within a repetition: seed request, wait for save, reset GPU prefix cache, continuation request. Retain LMCache CPU cache. Reset both caches between repetitions.
- CPU cache capacity: 2 GiB. One generated token, streaming client TTFT. Raw first-token log probabilities retained.

## Expected boundaries, to be checked against new logs

| Continuation (903 prompt tokens) | Native | Deep KV |
|---|---:|---:|
| SSM checkpoint | 528 | 528 |
| Full KV hit | 528 | 864 |
| Tokens requiring state replay/new computation | 375 | 375 |
| Tokens requiring new K/V projection in Full Attention layers | 375 | 39 |

The 336-token difference saves K/V projections, not 336 tokens of the full model forward. Query projections, attention outputs, MLPs and recurrent computation remain necessary for exact state recovery.

Both paths use 528-token physical CPU Full-KV objects. Deep KV adds fine 16-token aliases and a masked partial-tail transfer. In this case, native Full KV is one 16.5 MiB object; deep KV is two objects (33 MiB transferred with padding, 27 MiB valid KV). State traffic is additional. Greater hit depth therefore trades extra transfer and lookup/copy work against fewer K/V projections.

## Interpretation

Report all three continuation TTFT values and their arithmetic mean, observed hit lengths, forward-token counts and first-token agreement. Do not infer throughput or multi-user benefit from this one case. One startup per arm (deep then native) is a preliminary comparison, vulnerable to order/environment effects. Native prefill budget is 528, experimental budget is 2048, following the existing fixed baseline; this is not a single-factor ablation.

No speedup is assumed. Startup disk stalls are outside the request TTFT timer but can prevent the experiment from reaching inference.
