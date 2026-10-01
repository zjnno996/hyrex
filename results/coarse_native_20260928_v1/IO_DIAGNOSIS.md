# Startup I/O diagnosis — 2026-09-28 10:03–10:05 UTC

## Experiment progress

9B smoke driver PID 1506227; harness PID 1506316; LMCache PID 1506319.
The LMCache process is still starting, in D state with block-I/O wait channels.
No vLLM log, warmup output or online request output exists yet. Therefore no
new TTFT measurement exists and this delay is not measured cache-recovery/H2D time.

Completed checks: 9 index tests, 3 real-key/hasher backend tests, 20 query/protocol
tests, 8 GPU layout cases, 8 real native CPU/GPU DMA cases, and batch-planner
checks. These do not establish end-to-end inference correctness or acceleration.

## Resource observations

| Metric | Observation |
|---|---:|
| Available host RAM | about 58 GiB |
| Swap | none |
| GPU 1 allocation / utilization | 18 MiB / 0% |
| Host CPU I/O-wait, two vmstat intervals | 67%, 68% |
| Host blocked tasks, same intervals | 163, 138 |
| Host I/O PSI full, 60-second average | about 63–64% |
| Container I/O PSI full, 60-second average | about 50% |
| Root overlay filesystem | rounded 100% used, 65 GB available |
| Visible cgroup io.max | no local limit entries |

PSI percentages describe stalled time, not disk bandwidth utilization. Parent
cgroup limits and host-wide workload attribution are not visible from this view.
Filesystem fullness alone is not sufficient to explain the measured delay.

## Block-device sample

Derived from /proc/diskstats deltas over 35.389 seconds; read/write await values
are aggregate completion-accounting estimates, not per-file request traces.

| Device | Type | Read MiB/s | Write MiB/s | Read IOPS | Read await ms | Write await ms | Average queue | Busy % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sda | rotational, ST16000NM000J-2T, 14.6 TiB | 4.28 | 2.42 | 476.82 | 130.65 | 118.51 | 64.17 | 98.10 |
| nvme0n1 | NVMe, 3.6 TiB | 0.08 | 22.69 | 6.73 | 1.16 | 1.87 | 0.68 | 14.37 |

sda's configured device queue depth is 32; 32 requests were in flight at both
sample endpoints. An earlier sample also showed read await ~134 ms and queue
~65. Root overlay paths are under /mnt/disk1-16/docker-storage; cgroup counters
show substantial I/O on 8:0 (sda). This strongly points to contention on the
shared rotational storage path; it is not a file-by-file physical-device trace.

The LMCache process had read about 21 MiB from storage after roughly 17 minutes
and written only 20 KiB. In a separate approximately 44-second process sample it
read 0.91 MiB and wrote zero. Other visible container processes also showed only
small deltas, while host sda traffic was much larger. Container-external activity
or host filesystem/kernel work is likely involved, but no specific offending
job can be identified with the available process visibility.

## Interpretation and next actions

The immediate bottleneck is storage request latency/queueing during startup,
not a shortage of GPU memory, host RAM, or a measured failure of the recovery
algorithm. Many serial imports, library page faults and filesystem metadata
accesses can accumulate substantial delay on this overloaded path.

Ask the host operator to identify competing sda I/O and verify disk health and
parent I/O controls. Prefer a host-provided NVMe-backed workspace for environment,
source, model and logs if available. The relatively idle NVMe shown here is not
proof it is mounted or available to this container. No files have been moved or
deleted and no unrelated jobs have been stopped. Once startup works, retain
identical warmup/reset conditions for native and coarse recovery; recheck host
pressure before treating TTFT measurements as representative.
