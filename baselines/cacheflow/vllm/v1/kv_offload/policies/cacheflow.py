# SPDX-License-Identifier: Apache-2.0
"""Paper-level CacheFlow batch recovery scheduler.

CacheFlow advances a compute pointer from the prefix head and an I/O pointer
from its tail.  After every chunk it re-prioritizes requests by their remaining
recomputation cost.  This module implements that progressive, single-GPU
two-pointer semantics; it intentionally does not claim CacheFlow's distributed
3D executor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, inf
from time import monotonic
from typing import Any

from vllm.v1.kv_offload.recovery_policy import (
    RecoveryBatch,
    RecoveryTelemetry,
    register_recovery_policy,
)


@dataclass(frozen=True, slots=True)
class CacheFlowSegment:
    request_id: str
    segment_id: str
    load_bytes: int
    replay_ms: float
    chunks: int = 1
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.load_bytes < 0 or self.replay_ms < 0 or self.chunks < 1:
            raise ValueError("CacheFlow costs must be non-negative")
        if self.load_bytes % self.chunks:
            raise ValueError("load_bytes must divide evenly into chunks")


@dataclass(frozen=True, slots=True)
class CacheFlowTask:
    segment: CacheFlowSegment
    action: str
    chunk_index: int
    start_ms: float
    finish_ms: float


@dataclass(frozen=True, slots=True)
class CacheFlowPlan:
    tasks: tuple[CacheFlowTask, ...]
    h2d_finish_ms: float
    compute_finish_ms: float

    def validate(self) -> None:
        """Reject schedules that cannot be dispatched as two-pointer jobs."""
        by_segment: dict[tuple[str, str], list[CacheFlowTask]] = {}
        resource_finish = {"load": 0.0, "replay": 0.0}
        for task in self.tasks:
            if task.action not in resource_finish:
                raise ValueError(f"unknown CacheFlow action: {task.action!r}")
            if task.start_ms < resource_finish[task.action]:
                raise ValueError(f"overlapping {task.action} tasks")
            if task.finish_ms < task.start_ms:
                raise ValueError("task finishes before it starts")
            resource_finish[task.action] = task.finish_ms
            by_segment.setdefault(
                (task.segment.request_id, task.segment.segment_id), []
            ).append(task)
        for tasks in by_segment.values():
            segment = tasks[0].segment
            if len(tasks) != segment.chunks:
                raise ValueError("segment does not restore every chunk exactly once")
            if sorted(task.chunk_index for task in tasks) != list(range(segment.chunks)):
                raise ValueError("segment has missing or duplicate chunks")
            replay = [task.chunk_index for task in tasks if task.action == "replay"]
            load = [task.chunk_index for task in tasks if task.action == "load"]
            if replay != sorted(replay) or load != sorted(load, reverse=True):
                raise ValueError("segment violates CacheFlow two-pointer order")


@dataclass(frozen=True, slots=True)
class CacheFlowExecutionCallbacks:
    """Backend hooks for dispatching an already-validated CacheFlow plan.

    The callbacks own actual CUDA streams and completion fences. Keeping them
    outside the policy makes this a faithful scheduling adapter without
    fabricating a second vLLM transfer implementation.
    """

    load: Callable[[CacheFlowTask], Any]
    replay: Callable[[CacheFlowTask], Any]


@dataclass(frozen=True, slots=True)
class CacheFlowExecutionResult:
    """Completed callback records, retained in planner dispatch order."""

    records: tuple["CacheFlowExecutionRecord", ...]

    @property
    def results(self) -> tuple[Any, ...]:
        return tuple(record.result for record in self.records)


@dataclass(frozen=True, slots=True)
class CacheFlowExecutionRecord:
    task: CacheFlowTask
    start_ms: float
    finish_ms: float
    result: Any


class CacheFlowExecutor:
    """Dispatch H2D and replay queues concurrently, serially per resource."""

    @staticmethod
    def execute(
        plan: CacheFlowPlan, callbacks: CacheFlowExecutionCallbacks
    ) -> CacheFlowExecutionResult:
        plan.validate()

        def wait_for_completion(result: Any) -> Any:
            """Turn CUDA events/futures into an actual completion boundary."""
            if hasattr(result, "synchronize"):
                result.synchronize()
            elif hasattr(result, "result"):
                result.result()
            return result

        def run(action: str) -> list[tuple[int, CacheFlowExecutionRecord]]:
            callback = callbacks.load if action == "load" else callbacks.replay
            completed = []
            for index, task in enumerate(plan.tasks):
                if task.action != action:
                    continue
                start = monotonic() * 1e3
                result = wait_for_completion(callback(task))
                completed.append(
                    (
                        index,
                        CacheFlowExecutionRecord(
                            task, start, monotonic() * 1e3, result
                        ),
                    )
                )
            return completed

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cacheflow") as pool:
            load_results = pool.submit(run, "load")
            replay_results = pool.submit(run, "replay")
            ordered = load_results.result() + replay_results.result()
        records = [None] * len(plan.tasks)
        for index, record in ordered:
            records[index] = record
        return CacheFlowExecutionResult(tuple(records))


@dataclass(frozen=True, slots=True)
class CacheFlowPolicy:
    name: str = "cacheflow"
    h2d_gbps: float = 32.0

    def __post_init__(self) -> None:
        if self.h2d_gbps <= 0:
            raise ValueError("CacheFlow H2D bandwidth must be positive")

    @staticmethod
    def _next_segment(
        segments: tuple[CacheFlowSegment, ...], remaining: dict[int, int]
    ) -> int:
        """Choose the request with the largest remaining recompute cost."""
        return min(
            (index for index, chunks in remaining.items() if chunks),
            key=lambda index: (
                -(remaining[index] * segments[index].replay_ms / segments[index].chunks),
                inf if segments[index].deadline_ms is None else segments[index].deadline_ms,
                segments[index].request_id,
                segments[index].segment_id,
                index,
            ),
        )

    def plan(
        self, batch: RecoveryBatch, telemetry: RecoveryTelemetry
    ) -> CacheFlowPlan:
        bandwidth = telemetry.h2d_gbps or self.h2d_gbps
        if bandwidth <= 0:
            raise ValueError("telemetry h2d_gbps must be positive")
        segments = tuple(batch.segments)
        if not all(isinstance(segment, CacheFlowSegment) for segment in segments):
            raise TypeError("CacheFlowPolicy requires CacheFlowSegment inputs")

        h2d_clock = telemetry.h2d_ready_ms
        compute_clock = telemetry.compute_ready_ms
        tasks: list[CacheFlowTask] = []
        remaining = {index: segment.chunks for index, segment in enumerate(segments)}
        head = {index: 0 for index in remaining}
        tail = {index: segments[index].chunks - 1 for index in remaining}
        while any(remaining.values()):
            index = self._next_segment(segments, remaining)
            segment = segments[index]
            load_ms = segment.load_bytes / segment.chunks / (bandwidth * 1e9) * 1e3
            replay_ms = segment.replay_ms / segment.chunks
            load_start, load_finish = h2d_clock, h2d_clock + load_ms
            replay_start, replay_finish = compute_clock, compute_clock + replay_ms
            if load_finish <= replay_finish:
                tasks.append(
                    CacheFlowTask(segment, "load", tail[index], load_start, load_finish)
                )
                tail[index] -= 1
                h2d_clock = load_finish
            else:
                tasks.append(
                    CacheFlowTask(segment, "replay", head[index], replay_start, replay_finish)
                )
                head[index] += 1
                compute_clock = replay_finish
            remaining[index] -= 1
        plan = CacheFlowPlan(tuple(tasks), h2d_clock, compute_clock)
        plan.validate()
        return plan


register_recovery_policy("cacheflow", CacheFlowPolicy)


@dataclass(frozen=True, slots=True)
class CacheFlow3DSegment:
    """One paper-level restoration graph over token, layer, and GPU axes."""

    request_id: str
    cached_tokens: int
    chunk_tokens: int
    layer_count: int
    stage_count: int
    compute_ms: float
    io_ms: float
    layer_crossover_tokens: int

    def __post_init__(self) -> None:
        if (
            min(
                self.cached_tokens,
                self.chunk_tokens,
                self.layer_count,
                self.stage_count,
            )
            <= 0
            or self.compute_ms < 0
            or self.io_ms < 0
            or self.layer_crossover_tokens < 0
        ):
            raise ValueError("invalid CacheFlow 3D segment")

    @property
    def axis(self) -> str:
        return (
            "token"
            if self.cached_tokens >= self.layer_crossover_tokens
            else "layer"
        )

    @property
    def units(self) -> int:
        return (
            ceil(self.cached_tokens / self.chunk_tokens)
            if self.axis == "token"
            else self.layer_count
        )


@dataclass(frozen=True, slots=True)
class CacheFlow3DTask:
    request_id: str
    axis: str
    stage: int
    action: str
    unit_index: int
    needs_boundary_activation: bool


@dataclass(frozen=True, slots=True)
class CacheFlow3DPlan:
    """Stage-local two-pointer tasks with explicit boundary-state dependency."""

    tasks: tuple[CacheFlow3DTask, ...]


@dataclass(frozen=True, slots=True)
class CacheFlow3DExecutionCallbacks:
    load: Callable[[CacheFlow3DTask], Any]
    replay: Callable[[CacheFlow3DTask], Any]
    load_boundary: Callable[[CacheFlow3DTask], Any]


@dataclass(frozen=True, slots=True)
class CacheFlow3DExecutionResult:
    results: tuple[Any, ...]


class CacheFlow3DExecutor:
    """Minimal stage-parallel executor for the paper's boundary-state model."""

    @staticmethod
    def execute(
        plan: CacheFlow3DPlan, callbacks: CacheFlow3DExecutionCallbacks
    ) -> CacheFlow3DExecutionResult:
        if not plan.tasks:
            return CacheFlow3DExecutionResult(())
        by_stage: dict[int, list[tuple[int, CacheFlow3DTask]]] = {}
        for index, task in enumerate(plan.tasks):
            by_stage.setdefault(task.stage, []).append((index, task))

        def run_stage(
            tasks: list[tuple[int, CacheFlow3DTask]]
        ) -> list[tuple[int, Any]]:
            boundaries: set[str] = set()
            completed: list[tuple[int, Any]] = []
            for index, task in tasks:
                if task.needs_boundary_activation and task.request_id not in boundaries:
                    callbacks.load_boundary(task)
                    boundaries.add(task.request_id)
                callback = callbacks.load if task.action == "load" else callbacks.replay
                completed.append((index, callback(task)))
            return completed

        with ThreadPoolExecutor(max_workers=len(by_stage)) as pool:
            futures = [pool.submit(run_stage, tasks) for tasks in by_stage.values()]
            completed = [item for future in futures for item in future.result()]
        results: list[Any] = [None] * len(plan.tasks)
        for index, result in completed:
            results[index] = result
        return CacheFlow3DExecutionResult(tuple(results))


class CacheFlow3DPlanner:
    """Clean-room implementation of CacheFlow §§3.1--3.3 planning rules."""

    @staticmethod
    def _io_units(segment: CacheFlow3DSegment) -> int:
        total = segment.compute_ms + segment.io_ms
        if total == 0:
            return 0
        # Eq. (1): the two pointers meet when compute and I/O critical paths
        # are balanced. The trailing units are loaded from storage.
        return min(segment.units, ceil(segment.units * segment.compute_ms / total))

    def plan(self, segments: tuple[CacheFlow3DSegment, ...]) -> CacheFlow3DPlan:
        remaining = {segment.request_id: segment for segment in segments}
        pointers = {
            segment.request_id: [0, segment.units - 1]
            for segment in segments
        }
        io_budget = {
            segment.request_id: self._io_units(segment)
            for segment in segments
        }
        tasks: list[CacheFlow3DTask] = []
        while remaining:
            # Algorithm 1: I/O serves the request with largest remaining work.
            request_id = max(
                remaining,
                key=lambda key: (
                    pointers[key][1] - pointers[key][0] + 1,
                    key,
                ),
            )
            segment = remaining[request_id]
            head, tail = pointers[request_id]
            if io_budget[request_id] > 0:
                action, unit_index = "load", tail
                pointers[request_id][1] -= 1
                io_budget[request_id] -= 1
            else:
                action, unit_index = "replay", head
                pointers[request_id][0] += 1
            for stage in range(segment.stage_count):
                tasks.append(
                    CacheFlow3DTask(
                        request_id,
                        segment.axis,
                        stage,
                        action,
                        unit_index,
                        stage > 0 and action == "replay",
                    )
                )
            if pointers[request_id][0] > pointers[request_id][1]:
                del remaining[request_id]
        return CacheFlow3DPlan(tuple(tasks))

__all__ = [
    "CacheFlowExecutionCallbacks",
    "CacheFlowExecutionRecord",
    "CacheFlowExecutionResult",
    "CacheFlowExecutor",
    "CacheFlow3DPlan",
    "CacheFlow3DExecutionCallbacks",
    "CacheFlow3DExecutionResult",
    "CacheFlow3DExecutor",
    "CacheFlow3DPlanner",
    "CacheFlow3DSegment",
    "CacheFlow3DTask",
    "CacheFlowPlan",
    "CacheFlowPolicy",
    "CacheFlowSegment",
    "CacheFlowTask",
]
