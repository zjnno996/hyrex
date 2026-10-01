# SPDX-License-Identifier: Apache-2.0
"""Lazy recovery-policy adapter for the HyRex planner."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.kv_offload.hyrex_scheduler import (
    HyRexPlanner,
    RecoveryPlan,
    RecoverySegment,
)
from vllm.v1.kv_offload.recovery_policy import (
    RecoveryBatch,
    RecoveryTelemetry,
    register_recovery_policy,
)


@dataclass(slots=True)
class HyRexPolicy:
    name: str = "hyrex"
    planner: HyRexPlanner = field(default_factory=HyRexPlanner)

    def plan(
        self, batch: RecoveryBatch, telemetry: RecoveryTelemetry
    ) -> RecoveryPlan:
        segments = list(batch.segments)
        if not all(isinstance(segment, RecoverySegment) for segment in segments):
            raise TypeError("HyRexPolicy requires RecoverySegment inputs")
        return self.planner.plan(
            segments,
            h2d_ready_ms=telemetry.h2d_ready_ms,
            compute_ready_ms=telemetry.compute_ready_ms,
        )


register_recovery_policy("hyrex", HyRexPolicy)

__all__ = ["HyRexPolicy"]
