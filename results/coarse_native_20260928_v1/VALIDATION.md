# Coarse-object / fine-index validation

As of 2026-09-28 10:00 UTC, no 9B inference or TTFT result is available.

Passed during this work:

- 9 dependency-free prefix-index tests (previous turn).
- 3 backend tests with actual TokenHasher and native ObjectKey conversion.
- 20 lookup/protocol tests: 18 handler/layout tests and 2 process round trips.
  Initial process tests failed under slow startup; an unchanged rerun passed.
  Layout mocks were corrected to explicitly disable the experimental backend
  and provide the group-specific hasher/chunk-size interface.
- 8 GPU tail-layout round trips, FP32/BF16, valid lengths 16/224/512/528.
- 8 real pinned-CPU/native-GPU DMA round trips for the same sizes and dtypes.
  Data was exact, partial-tail padding zero, unrelated GPU pages unchanged.
- Actual serving batch-planner checks: one native call for full objects,
  separate partial tail, correct sparse store page IDs and empty-store handling.

The 9B smoke driver is live (see status.json), but its LMCache child has spent
roughly 12 minutes in dependency startup. Observed process states are D with
`blk_mq_get_tag`, `folio_wait_bit_common` or `wait_on_buffer`; read byte counts
continue increasing. No inference request has been measured yet.

Host diagnostics: I/O full pressure roughly 54–60%, around 58 GiB available RAM,
no swap, filesystem reporting 100% rounded utilization with 65 GB available.
GPU 1 has only about 18 MiB allocated. These observations support a host-I/O
startup bottleneck, not GPU OOM; they do not isolate the cause of disk pressure.
No unrelated jobs or files were stopped/deleted.

Remaining: complete the 8-request, 32-output-token 9B smoke; then run the fixed
native/coarse/coarse/native 40-request arms with unrelated startup warmup and
GPU prefix reset between requests. The runner checks outputs before reporting
TTFT and writes per-session/turn means, K/V projection token savings and whole
forward token savings separately. No speedup is claimed from component tests.
