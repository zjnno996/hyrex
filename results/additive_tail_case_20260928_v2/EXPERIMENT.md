# Additive exact-tail SSM checkpoint

Qwen3.5-9B, GPU 1, real session WRAImOg_0, two turns, three repetitions.
Native reference: ../coarse_single_case_20260928_v2/2_baseline (copied with
provenance). Both arms use prefill budget 528, ten unrelated warmup requests,
GPU reset between requests, CPU retained within a pair and cleared between pairs.

Experimental mode retains ordinary SSM checkpoints and adds a content-addressed
checkpoint at the last complete 16-token Full-KV page. The implementation
reuses ReplaceTailConnector with VLLM_HYREX_ADD_TAIL=1, but does NOT replace S528.
VLLM_HYREX_SINGLE_FORWARD=0: this initial implementation splits the continuous
forward at checkpoint boundaries and waits for tail D2H before overwriting
the running-state buffer. This is not a fused single-forward implementation.

Expected seed (872 tokens): forward 0:528, 528:864, 864:872; save S528 and S864.
Expected resume (903 tokens): restore KV0:864 plus S864; forward 864:896,
896:903; save S896 for the next turn. New-state maintenance is enabled.

One extra SSM checkpoint occupies three opaque 16.5 MiB transport objects:
49.5 MiB payload in the existing CPU pool, not an additional 49.5 MiB RSS
allocation above the pool. Actual state+convolution data is 49.125 MiB before
transport padding. Older tail entries remain until eviction or pair reset.

The v1 attempt failed startup validation because prefill budget was 2048;
no requests completed. v2 sets 528 without bypassing the native safety check.

Additional runtime source SHA256 (complements source_sha256.json):
- scheduler.py: 3c9d35296af9d5450163e2be86266abdd5dd27dbeceefa6767d2c1e33973d575
- gpu_model_runner.py: 318e5d64a9023bb180ebb48d035cd8d3d5e748af4de3f74a6a64f5e81b31e69b
