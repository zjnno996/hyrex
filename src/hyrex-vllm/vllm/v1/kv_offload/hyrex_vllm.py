# SPDX-License-Identifier: Apache-2.0
"""vLLM-facing observation and execution glue for HyRex.

This module deliberately does not own cache storage.  The native offload path
still owns block lookup and H2D; HyRex only turns its observations into a
heterogeneous recovery plan and dispatches the selected operation.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Protocol

from vllm.v1.kv_offload.hyrex_scheduler import (
    HyRexCostModel,
    HyRexPlanner,
    PlannedRecovery,
    RecoveryAction,
    RecoveryPlan,
    RecoverySegment,
    StateKind,
)

ALL_LOAD = "all_load"
FULL_LOAD_LINEAR_REPLAY = "full_load_linear_replay"
FULL_REPLAY_LINEAR_LOAD = "full_replay_linear_load"
ALL_REPLAY = "all_replay"


@dataclass(frozen=True, slots=True)
class HyRexVLLMDecision:
    """A HyRex plan mapped to a recovery policy vLLM can execute today."""

    policy: str | None
    plan: RecoveryPlan
    requires_terminal_store: bool = False


def connector_params(
    decision: HyRexVLLMDecision,
    *,
    replay_ms: float | None = None,
    source_key: str | None = None,
    priority: float = 0.0,
) -> dict[str, object]:
    """Serialize a validated decision for vLLM's native offload connector.

    Returning no policy for a terminal-state decision is intentional: the
    connector must not skip a recurrent prefix until that state has a real
    worker-side restore implementation.
    """
    if decision.policy is None:
        raise ValueError("terminal-state restore is not bound to the connector")
    params: dict[str, object] = {
        "hyrex_recovery_policy": decision.policy,
        "hyrex_priority": priority,
    }
    if replay_ms is not None:
        if replay_ms < 0:
            raise ValueError("replay_ms must be non-negative")
        params["hyrex_replay_ms"] = replay_ms
    if source_key:
        params["hyrex_source_key"] = source_key
    return params


def plan_vllm_recovery(
    segments: list[RecoverySegment],
    *,
    planner: HyRexPlanner | None = None,
) -> HyRexVLLMDecision:
    """Map Full-KV and recurrent decisions onto existing vLLM replay paths.

    A None policy is deliberate: terminal-state load needs the P5 state store
    and must never be silently downgraded to a different policy.
    """
    return _decision_from_tasks((planner or HyRexPlanner()).plan(segments).tasks)


def select_native_policy(
    *,
    missing_tokens: int,
    full_load_bytes: int,
    recurrent_load_bytes: int,
    full_replay_ms: float,
    recurrent_replay_ms: float,
    h2d_gbps: float,
    h2d_ready_ms: float = 0.0,
    compute_ready_ms: float = 0.0,
    full_source_ready: bool = True,
    recurrent_source_ready: bool = True,
) -> str:
    """Choose the fastest policy supported by the native mixed-replay path.

    This deliberately requires measured replay and H2D inputs.
    """
    if missing_tokens <= 0:
        raise ValueError("missing_tokens must be positive")
    if min(full_load_bytes, recurrent_load_bytes) < 0:
        raise ValueError("load bytes must be non-negative")
    if min(full_replay_ms, recurrent_replay_ms, h2d_ready_ms, compute_ready_ms) < 0:
        raise ValueError("latencies must be non-negative")

    model = HyRexCostModel(h2d_gbps=h2d_gbps)
    full = model.estimate(
        RecoverySegment(
            "native",
            "full",
            StateKind.FULL_KV,
            missing_tokens,
            full_load_bytes,
            replay_ms=full_replay_ms,
        )
    )
    recurrent = model.estimate(
        RecoverySegment(
            "native",
            "recurrent",
            StateKind.RECURRENT,
            missing_tokens,
            recurrent_load_bytes,
            replay_ms=recurrent_replay_ms,
        )
    )
    candidates: dict[str, float] = {
        ALL_REPLAY: compute_ready_ms + full.replay_ms + recurrent.replay_ms
    }
    if full_source_ready and recurrent_source_ready:
        candidates[ALL_LOAD] = h2d_ready_ms + full.load_ms + recurrent.load_ms
    if full_source_ready:
        candidates[FULL_LOAD_LINEAR_REPLAY] = max(
            h2d_ready_ms + full.load_ms,
            compute_ready_ms + recurrent.replay_ms,
        )
    if recurrent_source_ready:
        candidates[FULL_REPLAY_LINEAR_LOAD] = max(
            h2d_ready_ms + recurrent.load_ms,
            compute_ready_ms + full.replay_ms,
        )
    # Dict order makes an exact tie preserve compute by preferring all-load.
    return min(candidates, key=candidates.__getitem__)


def _decision_from_tasks(
    tasks: tuple[PlannedRecovery, ...],
) -> HyRexVLLMDecision:
    """Map a request's tasks without re-planning a batch-wide schedule."""
    plan = RecoveryPlan(
        tasks,
        max(
            (
                task.estimated_finish_ms
                for task in tasks
                if task.action is not RecoveryAction.REPLAY
            ),
            default=0.0,
        ),
        max(
            (
                task.estimated_finish_ms
                for task in tasks
                if task.action is RecoveryAction.REPLAY
            ),
            default=0.0,
        ),
    )
    actions: dict[StateKind, set[RecoveryAction]] = {}
    for task in plan.tasks:
        actions.setdefault(task.segment.state_kind, set()).add(task.action)
    if any(RecoveryAction.TERMINAL_LOAD in value for value in actions.values()):
        return HyRexVLLMDecision(None, plan, requires_terminal_store=True)

    full = actions.get(StateKind.FULL_KV, {RecoveryAction.LOAD})
    recurrent = actions.get(StateKind.RECURRENT, {RecoveryAction.LOAD})
    if len(full) != 1 or len(recurrent) != 1:
        raise ValueError("one HyRex action per state kind is required by vLLM")
    full_action = next(iter(full))
    recurrent_action = next(iter(recurrent))
    policies = {
        (RecoveryAction.LOAD, RecoveryAction.LOAD): ALL_LOAD,
        (RecoveryAction.LOAD, RecoveryAction.REPLAY): FULL_LOAD_LINEAR_REPLAY,
        (RecoveryAction.REPLAY, RecoveryAction.LOAD): FULL_REPLAY_LINEAR_LOAD,
        (RecoveryAction.REPLAY, RecoveryAction.REPLAY): ALL_REPLAY,
    }
    return HyRexVLLMDecision(policies[(full_action, recurrent_action)], plan)


@dataclass(frozen=True, slots=True)
class HyRexObservation:
    """Cache-lookup facts for one missing hybrid-cache segment.

    All latency and byte fields are observed by the cache/offload backend;
    this adapter intentionally never invents a cache hit or an H2D size.
    """

    request_id: str
    segment_id: str
    layer_type: str
    missing_tokens: int
    load_bytes: int
    source_ready: bool
    source_key: str | None = None
    materialization_key: str | None = None
    lookup_ms: float = 0.0
    replay_ms: float | None = None
    terminal_bytes: int | None = None
    terminal_ready: bool = False
    priority: float = 0.0
    arrival_ms: float = 0.0
    deadline_ms: float | None = None

    def state_kind(self) -> StateKind:
        if self.layer_type == "full_attention":
            return StateKind.FULL_KV
        if self.layer_type in {"mamba", "linear_attention", "recurrent"}:
            return StateKind.RECURRENT
        raise ValueError(f"unsupported HyRex layer type: {self.layer_type}")

    def segment(self) -> RecoverySegment:
        kind = self.state_kind()
        if kind is StateKind.FULL_KV and (
            self.terminal_bytes is not None or self.terminal_ready
        ):
            raise ValueError("full-attention KV cannot use a terminal state")
        return RecoverySegment(
            request_id=self.request_id,
            segment_id=self.segment_id,
            state_kind=kind,
            missing_tokens=self.missing_tokens,
            load_bytes=self.load_bytes,
            lookup_ms=self.lookup_ms,
            replay_ms=self.replay_ms,
            source_ready=self.source_ready,
            source_key=self.source_key,
            materialization_key=self.materialization_key,
            terminal_bytes=self.terminal_bytes,
            terminal_ready=self.terminal_ready,
            priority=self.priority,
            arrival_ms=self.arrival_ms,
            deadline_ms=self.deadline_ms,
        )


def _bucket(value: int) -> int:
    return max(1, value.bit_length() - 1)


@dataclass(slots=True)
class HyRexCalibrator:
    """EWMA calibration from completed native H2D and replay operations."""

    default_model: HyRexCostModel = HyRexCostModel()
    alpha: float = 0.2
    _h2d_ms_per_byte: dict[tuple[int, int], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _h2d_global_ms_per_byte: float | None = field(
        default=None, init=False, repr=False
    )
    _replay_ms: dict[tuple[StateKind, int, int], float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def _update(self, old: float | None, value: float) -> float:
        return value if old is None else self.alpha * value + (1.0 - self.alpha) * old

    def observe_h2d(
        self, num_bytes: int, elapsed_ms: float, concurrency: int = 1
    ) -> None:
        if num_bytes <= 0 or elapsed_ms < 0 or concurrency <= 0:
            raise ValueError(
                "H2D observation requires positive bytes/concurrency and "
                "non-negative time"
            )
        value = elapsed_ms / num_bytes
        key = (_bucket(num_bytes), _bucket(concurrency))
        self._h2d_ms_per_byte[key] = self._update(
            self._h2d_ms_per_byte.get(key), value
        )
        self._h2d_global_ms_per_byte = self._update(
            self._h2d_global_ms_per_byte, value
        )

    def observe_replay(
        self,
        state_kind: StateKind,
        missing_tokens: int,
        concurrency: int,
        elapsed_ms: float,
    ) -> None:
        if missing_tokens <= 0 or concurrency <= 0 or elapsed_ms < 0:
            raise ValueError("invalid replay observation")
        key = (state_kind, _bucket(missing_tokens), _bucket(concurrency))
        self._replay_ms[key] = self._update(self._replay_ms.get(key), elapsed_ms)

    def model(self, concurrency: int = 1, num_bytes: int = 1) -> HyRexCostModel:
        if concurrency <= 0 or num_bytes <= 0:
            raise ValueError("concurrency and num_bytes must be positive")
        target = (_bucket(num_bytes), _bucket(concurrency))
        rate = self._h2d_ms_per_byte.get(target)
        if rate is None and self._h2d_ms_per_byte:
            nearest = min(
                self._h2d_ms_per_byte,
                key=lambda key: abs(key[0] - target[0]) + abs(key[1] - target[1]),
            )
            rate = self._h2d_ms_per_byte[nearest]
        if rate is None:
            rate = self._h2d_global_ms_per_byte
        if rate is None or rate == 0:
            return self.default_model
        return replace(
            self.default_model,
            h2d_gbps=1e-6 / rate,
            h2d_fixed_ms=0.0,
        )

    def enrich(self, segment: RecoverySegment, concurrency: int) -> RecoverySegment:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        key = (
            segment.state_kind,
            _bucket(segment.missing_tokens),
            _bucket(concurrency),
        )
        replay_ms = self._replay_ms.get(key, segment.replay_ms)
        return replace(segment, replay_ms=replay_ms)


@dataclass(frozen=True, slots=True)
class HyRexBatchDecision:
    """One batch-wide resource plan plus the policy selected per request."""

    plan: RecoveryPlan
    requests: dict[str, HyRexVLLMDecision]


@dataclass(slots=True)
class HyRexController:
    """The HyRex control plane between lookup and native recovery execution.

    It plans all requests together exactly once.  Per-request decisions are
    projections of that plan, not independently re-planned policies, so the
    result retains PCIe/GPU contention and cross-request ordering.
    """

    calibrator: HyRexCalibrator = field(default_factory=HyRexCalibrator)

    def plan_batch(
        self,
        observations: list[HyRexObservation],
        *,
        h2d_ready_ms: float = 0.0,
        compute_ready_ms: float = 0.0,
    ) -> HyRexBatchDecision:
        if not observations:
            empty = RecoveryPlan((), h2d_ready_ms, compute_ready_ms)
            return HyRexBatchDecision(empty, {})

        concurrency = len({observation.request_id for observation in observations})
        segments = [
            self.calibrator.enrich(observation.segment(), concurrency)
            for observation in observations
        ]
        total_load_bytes = max(1, sum(segment.load_bytes for segment in segments))
        plan = HyRexPlanner(
            self.calibrator.model(concurrency, total_load_bytes)
        ).plan(
            segments,
            h2d_ready_ms=h2d_ready_ms,
            compute_ready_ms=compute_ready_ms,
        )
        tasks_by_request: dict[str, list[PlannedRecovery]] = {}
        for task in plan.tasks:
            tasks_by_request.setdefault(task.segment.request_id, []).append(task)
        requests = {
            request_id: _decision_from_tasks(tuple(tasks))
            for request_id, tasks in tasks_by_request.items()
        }
        return HyRexBatchDecision(plan, requests)


class HyRexCallbacks(Protocol):
    """Native paths to invoke after HyRex makes a decision."""

    def load(self, task: PlannedRecovery) -> object: ...

    def terminal_load(self, task: PlannedRecovery) -> object: ...

    def replay(self, task: PlannedRecovery) -> object: ...


def _wait(result: object) -> None:
    if isinstance(result, Future):
        result.result()
    elif hasattr(result, "synchronize"):
        result.synchronize()  # type: ignore[union-attr]
    elif hasattr(result, "wait"):
        result.wait()  # type: ignore[union-attr]


def _invoke(callback: object, task: PlannedRecovery) -> None:
    _wait(callback(task))  # type: ignore[operator]


@dataclass(slots=True)
class HyRexExecutor:
    """Execute the plan with one serialized H2D and one serialized compute lane."""

    callbacks: HyRexCallbacks

    def execute(self, plan: RecoveryPlan) -> None:
        # The two workers preserve each contended resource's order while
        # allowing native H2D and replay to overlap.
        with (
            ThreadPoolExecutor(max_workers=1) as h2d_worker,
            ThreadPoolExecutor(max_workers=1) as compute_worker,
        ):
            h2d: list[Future[None]] = []
            compute: list[Future[None]] = []
            for task in plan.tasks:
                if task.action is RecoveryAction.REPLAY:
                    compute.append(
                        compute_worker.submit(_invoke, self.callbacks.replay, task)
                    )
                elif task.action is RecoveryAction.TERMINAL_LOAD:
                    h2d.append(
                        h2d_worker.submit(_invoke, self.callbacks.terminal_load, task)
                    )
                else:
                    h2d.append(h2d_worker.submit(_invoke, self.callbacks.load, task))
            for future in h2d + compute:
                future.result()


__all__ = [
    "ALL_LOAD",
    "ALL_REPLAY",
    "FULL_LOAD_LINEAR_REPLAY",
    "FULL_REPLAY_LINEAR_LOAD",
    "HyRexBatchDecision",
    "HyRexCallbacks",
    "HyRexCalibrator",
    "HyRexController",
    "HyRexExecutor",
    "HyRexObservation",
    "HyRexVLLMDecision",
    "connector_params",
    "plan_vllm_recovery",
    "select_native_policy",
]
