# Tail checkpoint: eliminate convolution scalar readback

## Change under test

Only the experimental vLLM convolution-checkpoint path changes:

- When checkpoint offset covers convolution history (width 3 on this model),
  copy the last 3 pre-convolution projection rows directly. No old-state gather
  or CUDA boolean conversion is needed.
- For offsets 1 or 2, gather the old state with device index_select and use
  torch.where to select zero history for cold requests. No Python CUDA scalar
  readback. This also correctly masks NaN values in uninitialized storage.
- Use one contiguous clone, rather than contiguous followed by clone.

Recurrent segmentation, recurrent snapshot clones, CPU object layout, transfer,
and cache matching are unchanged. New checkpoint maintenance remains enabled.
The frozen source manifest now includes the modified GDN and core allocator files.

## Checks before model run

- test_conv_checkpoint_no_sync.py: passed CPU/CUDA, warm/cold histories,
  offsets 1/2/3/32, non-contiguous source state, snapshot ownership. CPU profiler
  sees no aten::item or aten::_local_scalar_dense in the helper; offset 32 does
  not execute index_select. This does not claim zero synchronization elsewhere.
- test_checkpoint_capture.py: passed independent prefix state checks and
  recurrent final/output relative-error checks (threshold 0.025).
- Both pristine baseline source trees were clean before launch.

## Paired model experiment

Qwen3.5-9B, GPU 1, eager, one real ShareGPT session WRAImOg_0, first two turns
(872/903 tokens), three repetitions, one output token per measured request.
Both services perform 10 unrelated warmups. GPU cache is reset between each
formal request; CPU cache is retained inside seed/resume pairs and cleared
between repetitions. Both arms inherit identical CPU affinity (0-11,24-35).
Optimized arm runs first, then a freshly launched pristine baseline; no reused
baseline. Native prefill budget 528, experimental 2048, as in the previous test.

Expected continuation: native KV/state hit 528 and forward 375 tokens;
experimental KV/state hit 864 and forward 39 tokens, while saving the next
checkpoint at 896. New first-token agreement is checked but does not establish
full-sequence equivalence. Three repetitions and one service startup per arm
are a preliminary case, not a robust statistical estimate or causal profiling.
