# Single real-session case

Session: WRAImOg_0. Three seed/resume repetitions per arm; 10 unrelated startup warmups; GPU reset between requests; CPU reset between repetitions.

Resume mean TTFT: native 125.043 ms; deep 140.293 ms; reduction -12.20%.

First generated token agrees; this does not establish full-sequence equivalence. One startup per arm: preliminary case, not a robust speedup claim. Baseline reused from /root/hyrex_results/coarse_single_case_20260928_v2/2_baseline (None means newly run). Native prefill budget=528; deep=2048. Raw per-request data in online.jsonl.

## Per-repetition continuation measurements

| Repetition | Native TTFT ms | Deep TTFT ms | Native Full KV hit | Deep Full KV hit |
|---|---:|---:|---:|---:|
| 1 | 135.80 | 140.18 | 528 | 864 |
| 2 | 134.62 | 141.69 | 528 | 864 |
| 3 | 104.71 | 139.01 | 528 | 864 |
| Arithmetic mean | 125.043 | 140.293 | 528 | 864 |

Observed mean difference: deep is 15.250 ms / 12.20% slower. All three deep
measurements are slower than their corresponding native repetition; native
timing spread is substantial, so this small sequential experiment is not a
statistical estimate of a general slowdown. No measured speedup in this case.

## What was actually saved

Seed length is 872 tokens; continuation length is 903 tokens.

| Continuation work | Native | Deep |
|---|---:|---:|
| SSM checkpoint boundary | 528 | 528 |
| Full KV reusable boundary | 528 | 864 |
| Forward/recurrent token count | 375 | 375 |
| Tokens needing Full-Attention K/V projections | 375 | 39 |
| Full-KV physical CPU objects loaded | 1 | 2 |

The experimental runtime logs independently confirm `state=528 full_kv=864`
and `SINGLE_FORWARD start=528 count=375` for every continuation. The 336-token
K/V-projection saving is 89.6% of the native continuation projection-token
count, not an 89.6% reduction in total model computation. Query projection,
attention outputs, output projection, MLPs and GDN replay still execute.

From the transfer layout, each physical Full-KV object is 16.5 MiB, so Full-KV
DMA payload is 16.5 MiB native versus 33 MiB deep (27 MiB valid data plus
padding). These are code/layout-derived byte counts, not profiler measurements;
SSM traffic is additional. A partial tail also takes the masked-copy path.
This run does not separately time H2D, lookup, copying and forward computation,
so the 15.25 ms TTFT difference cannot be causally assigned to any one of them.

## Fix and validation

- Fixed stale fine-prefix metadata after native CPU force-clear, through the
  existing ManagementModule.clear entry point. Four backend tests pass,
  including the clear/reseed regression test.
- The earlier v2 experimental hit sequence was 864/528/528 and is invalid for
  this comparison. See v2/INVALID_DEEP_COMPARISON.md; no cherry-picked first
  repetition is used as the final deep result.
- Here all three deep hits are 864 and all native hits are 528. Each arm has
  10 warmup records and 5 successful formal GPU resets. CPU is retained within
  seed/resume pairs and cleared between repetitions.
- First generated tokens match for all six paired requests. Log probabilities
  differ; only first-token agreement, not full-sequence numerical correctness,
  is established by this measurement.
- Both pristine native source trees were checked clean against HEAD. The
  experimental source snapshot is in source_sha256.json. Baseline provenance
  is in baseline_origin.txt; the baseline was measured immediately before the
  corrected deep arm, not rerun or changed for this result.
- All experiment processes exited and GPU 1 was released after measurement.
