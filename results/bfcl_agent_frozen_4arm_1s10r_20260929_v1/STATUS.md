# Status

Stopped before inference on 2026-09-29 because the source-integrity import
process blocked in `folio_wait_bit_common`. At the stop point, system-wide I/O
pressure was `full avg10=56.62%`; no vLLM server, LMCache server, warmup, or
measured request had started. This directory contains no TTFT result.

The validated input is
`/root/hyrex_results/motivation_bfcl_agent_1s10r_trace.jsonl` (one session,
ten cumulative requests). Use a new output directory when I/O pressure has
returned to normal.
