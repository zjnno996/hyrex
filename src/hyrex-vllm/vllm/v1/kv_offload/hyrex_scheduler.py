# SPDX-License-Identifier: Apache-2.0
"""Cost-aware batch planning for heterogeneous Hybrid-cache recovery.

This module is deliberately independent of a transport backend.  It is the
small decision layer that can be called after cache lookup and before a load
job is submitted.  A later executor can turn :class:`RecoveryPlan` into vLLM
connector jobs without changing the decision model.

Important distinction:
``source_key`` identifies a shared CPU-cache object, while
``materialization_key`` identifies an identical GPU destination.  Sharing a
source removes lookup/metadata work, but does *not* remove H2D bytes when two
requests need different GPU blocks.  Only identical materializations may
share one physical transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import inf


class StateKind(str, Enum):
    FULL_KV = "full_kv"
    RECURRENT = "recurrent"


class RecoveryAction(str, Enum):
    LOAD = "load"
    TERMINAL_LOAD = "terminal_load"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class RecoverySegment:
    """One independently decidable missing state segment.

    ``replay_ms`` is preferred when available because real prefill/replay is
    not generally linear in token count.  The fallback coefficients are useful
    during cold start and can be fitted from completed requests.
    """

    request_id: str
    segment_id: str
    state_kind: StateKind
    missing_tokens: int
    load_bytes: int
    lookup_ms: float = 0.0
    replay_ms: float | None = None
    source_ready: bool = True
    source_key: str | None = None
    # Same key means the physical H2D destination can be shared.  Leave None
    # for normal per-request GPU blocks, even when source_key is shared.
    materialization_key: str | None = None
    terminal_bytes: int | None = None
    terminal_ready: bool = False
    priority: float = 0.0
    arrival_ms: float = 0.0
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.missing_tokens <= 0:
            raise ValueError("missing_tokens must be positive")
        if self.load_bytes < 0:
            raise ValueError("load_bytes must be non-negative")
        if self.lookup_ms < 0:
            raise ValueError("lookup_ms must be non-negative")
        if self.replay_ms is not None and self.replay_ms < 0:
            raise ValueError("replay_ms must be non-negative")
        if self.terminal_bytes is not None and self.terminal_bytes < 0:
            raise ValueError("terminal_bytes must be non-negative")
        if (
            self.state_kind is not StateKind.RECURRENT
            and self.terminal_bytes is not None
        ):
            raise ValueError("terminal_bytes is only valid for recurrent state")


@dataclass(frozen=True, slots=True)
class RecoveryEstimate:
    segment: RecoverySegment
    load_ms: float
    terminal_load_ms: float | None
    replay_ms: float

    def cost(self, action: RecoveryAction) -> float:
        if action is RecoveryAction.LOAD:
            return self.load_ms
        if action is RecoveryAction.TERMINAL_LOAD:
            return inf if self.terminal_load_ms is None else self.terminal_load_ms
        return self.replay_ms


@dataclass(frozen=True, slots=True)
class PlannedRecovery:
    segment: RecoverySegment
    action: RecoveryAction
    estimated_start_ms: float
    estimated_finish_ms: float
    estimated_cost_ms: float
    source_coalesced: bool = False
    physical_transfer_coalesced: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    tasks: tuple[PlannedRecovery, ...]
    h2d_finish_ms: float
    compute_finish_ms: float

    @property
    def p99_finish_ms(self) -> float:
        finishes = sorted(task.estimated_finish_ms for task in self.tasks)
        if not finishes:
            return 0.0
        # Nearest-rank p99 is deterministic for small serving batches.
        return finishes[min(len(finishes) - 1, (99 * len(finishes) + 99) // 100 - 1)]

    @property
    def physical_load_tasks(self) -> tuple[PlannedRecovery, ...]:
        """Loads that require a physical transfer, excluding shared copies."""
        return tuple(
            task
            for task in self.tasks
            if task.action in {RecoveryAction.LOAD, RecoveryAction.TERMINAL_LOAD}
            and not task.physical_transfer_coalesced
        )

    @property
    def logical_load_tasks(self) -> tuple[PlannedRecovery, ...]:
        """All load consumers, including requests sharing one GPU copy."""
        return tuple(
            task
            for task in self.tasks
            if task.action in {RecoveryAction.LOAD, RecoveryAction.TERMINAL_LOAD}
        )


@dataclass(frozen=True, slots=True)
class HyRexCostModel:
    """Small calibrated model for the two shared recovery resources."""

    h2d_gbps: float = 20.0
    h2d_fixed_ms: float = 0.0
    replay_base_ms: float = 0.0
    replay_ms_per_token: float = 0.0

    def __post_init__(self) -> None:
        if self.h2d_gbps <= 0:
            raise ValueError("h2d_gbps must be positive")
        if self.h2d_fixed_ms < 0 or self.replay_base_ms < 0:
            raise ValueError("fixed costs must be non-negative")
        if self.replay_ms_per_token < 0:
            raise ValueError("replay_ms_per_token must be non-negative")

    def _transfer_ms(self, num_bytes: int) -> float:
        return self.h2d_fixed_ms + num_bytes / (self.h2d_gbps * 1_000_000_000) * 1_000

    def estimate(self, segment: RecoverySegment) -> RecoveryEstimate:
        replay_ms = (
            segment.replay_ms
            if segment.replay_ms is not None
            else self.replay_base_ms + segment.missing_tokens * self.replay_ms_per_token
        )
        load_ms = segment.lookup_ms + self._transfer_ms(segment.load_bytes)
        terminal_load_ms: float | None = None
        if (
            segment.state_kind is StateKind.RECURRENT
            and segment.terminal_ready
            and segment.terminal_bytes is not None
        ):
            terminal_load_ms = segment.lookup_ms + self._transfer_ms(
                segment.terminal_bytes
            )
        return RecoveryEstimate(segment, load_ms, terminal_load_ms, replay_ms)


@dataclass(slots=True)
class HyRexPlanner:
    """Greedy batch planner with explicit H2D and compute resource clocks.

    The planner is intentionally O(n log n), which is enough for a vLLM
    scheduler batch and keeps the first prototype inspectable.  The upgrade
    path for a full constrained optimizer is to replace ``_order``; the cost
    model and execution contract remain unchanged.
    """

    model: HyRexCostModel = field(default_factory=HyRexCostModel)

    def _order(self, segments: list[RecoverySegment]) -> list[RecoverySegment]:
        def key(segment: RecoverySegment) -> tuple[float, float, float, str]:
            deadline = inf if segment.deadline_ms is None else segment.deadline_ms
            # Higher priority and earlier deadlines first; arrival breaks ties.
            return (-segment.priority, deadline, segment.arrival_ms, segment.request_id)

        return sorted(segments, key=key)

    @staticmethod
    def _source_key(segment: RecoverySegment) -> str:
        return segment.source_key or f"{segment.request_id}:{segment.segment_id}"

    @staticmethod
    def _materialization_key(segment: RecoverySegment, action: RecoveryAction) -> str:
        key = segment.materialization_key
        if key is None:
            key = f"{segment.request_id}:{segment.segment_id}"
        return f"{action.value}:{key}"

    def plan(
        self,
        segments: list[RecoverySegment],
        *,
        h2d_ready_ms: float = 0.0,
        compute_ready_ms: float = 0.0,
    ) -> RecoveryPlan:
        if h2d_ready_ms < 0 or compute_ready_ms < 0:
            raise ValueError("resource-ready times must be non-negative")

        h2d_clock = h2d_ready_ms
        compute_clock = compute_ready_ms
        seen_sources: set[str] = set()
        seen_materializations: set[str] = set()
        tasks: list[PlannedRecovery] = []

        for segment in self._order(segments):
            estimate = self.model.estimate(segment)
            candidates: list[tuple[RecoveryAction, float, float, float]] = []

            if segment.source_ready:
                source_key = self._source_key(segment)
                source_coalesced = source_key in seen_sources
                materialization = self._materialization_key(
                    segment, RecoveryAction.LOAD
                )
                physical_coalesced = materialization in seen_materializations
                # Lookup is shared by a source, while H2D is shared only by an
                # identical destination.  This is the critical Hybrid/cache
                # distinction that a request-level load/replay rule misses.
                load_cost = estimate.load_ms
                if source_coalesced:
                    load_cost -= segment.lookup_ms
                if physical_coalesced:
                    load_cost = 0.0
                start = h2d_clock
                finish = start + load_cost
                candidates.append((RecoveryAction.LOAD, start, finish, load_cost))

            if estimate.terminal_load_ms is not None:
                materialization = self._materialization_key(
                    segment, RecoveryAction.TERMINAL_LOAD
                )
                physical_coalesced = materialization in seen_materializations
                load_cost = 0.0 if physical_coalesced else estimate.terminal_load_ms
                start = h2d_clock
                finish = start + load_cost
                candidates.append(
                    (RecoveryAction.TERMINAL_LOAD, start, finish, load_cost)
                )

            replay_start = compute_clock
            replay_finish = replay_start + estimate.replay_ms
            candidates.append(
                (RecoveryAction.REPLAY, replay_start, replay_finish, estimate.replay_ms)
            )

            # Choose the operation that makes this request ready first. Ties
            # prefer load because it preserves compute capacity for the batch.
            action, start, finish, cost = min(
                candidates,
                key=lambda candidate: (
                    candidate[2],
                    candidate[0] is RecoveryAction.REPLAY,
                ),
            )
            source_key = self._source_key(segment)
            materialization = self._materialization_key(segment, action)
            source_coalesced = source_key in seen_sources
            physical_coalesced = materialization in seen_materializations
            tasks.append(
                PlannedRecovery(
                    segment=segment,
                    action=action,
                    estimated_start_ms=start,
                    estimated_finish_ms=finish,
                    estimated_cost_ms=cost,
                    source_coalesced=source_coalesced,
                    physical_transfer_coalesced=physical_coalesced,
                )
            )
            if action is RecoveryAction.REPLAY:
                compute_clock = finish
            else:
                h2d_clock = finish
                seen_sources.add(source_key)
                seen_materializations.add(materialization)

        return RecoveryPlan(tuple(tasks), h2d_clock, compute_clock)


__all__ = [
    "HyRexCostModel",
    "HyRexPlanner",
    "PlannedRecovery",
    "RecoveryAction",
    "RecoveryEstimate",
    "RecoveryPlan",
    "RecoverySegment",
    "StateKind",
]
