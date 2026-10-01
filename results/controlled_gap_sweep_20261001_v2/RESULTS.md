# Controlled hybrid-recovery gap sweep

All paths use Qwen3.5-9B BF16 eager, five repetitions, warmup, and a verified
GPU-prefix reset after every request. Native uses its required 528-token budget;
the matched Aligned-SF/Deep-KV/Exact paths use a 2,112-token budget.
The previous prompt ends at `1056 + gap`; the next request appends 128 tokens.

| Gap | Native-528 | Aligned-SF (vs Native) | Deep-KV (vs Aligned) | Exact recovery-only (vs Aligned) | Exact steady-state (vs Aligned) |
|---:|---:|---:|---:|---:|---:|
| 0 | 134.02 | 140.34 (+6.32) | 139.04 (-1.30) | 193.98 (+53.64) | 154.58 (+14.24) |
| 64 | 135.84 | 145.63 (+9.79) | 144.36 (-1.27) | 155.34 (+9.71) | 160.02 (+14.39) |
| 128 | 131.31 | 143.09 (+11.78) | 143.49 (+0.41) | 153.24 (+10.16) | 150.48 (+7.40) |
| 256 | 136.43 | 140.68 (+4.25) | 147.67 (+6.99) | 159.55 (+18.86) | 150.70 (+10.01) |
| 384 | 134.50 | 142.52 (+8.02) | 141.85 (-0.67) | 146.42 (+3.90) | 148.39 (+5.87) |
| 512 | 201.45 | 198.78 (-2.67) | 197.47 (-1.31) | 156.34 (-42.44) | 166.34 (-32.44) |

Median TTFT (the robust statistic used in the motivation figure):

| Avoided replay | Aligned-SF (ms) | Deep delta | Exact recovery-only delta | Exact steady delta |
|---:|---:|---:|---:|---:|
| 0 | 140.73 | +0.40 | +20.98 | +16.56 |
| 64 | 144.64 | +1.13 | +13.40 | +13.09 |
| 128 | 144.31 | -0.58 | +6.12 | +13.17 |
| 256 | 143.84 | +4.39 | +15.95 | +12.78 |
| 384 | 143.61 | +3.37 | +11.14 | +11.59 |
| 512 | 199.57 | -2.72 | -37.14 | -33.22 |

Deep-KV's complete-model replay is `gap + 128` despite its deeper reported KV
hit. Exact paths replay 128 tokens. First-token mismatches versus Native:
native=0/30, aligned=0/30, deep=0/30, exact_recovery_only=0/30, exact_steady=0/30.

At the 512-token gap, exact steady-state recovery is 33.22 ms faster than the
matched aligned path in median TTFT. Capturing/storing the next checkpoint adds
3.92 ms over recovery-only at this point (10.00 ms by the five-sample mean).

The run uses the same pickle IPC fallback for every path because `/dev/shm` is
64 MiB. It is valid as a matched mechanism comparison, but final publication
numbers should be repeated with sufficient SHM.
