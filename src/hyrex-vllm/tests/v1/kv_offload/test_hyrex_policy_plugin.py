# SPDX-License-Identifier: Apache-2.0

from vllm.v1.kv_offload.hyrex_scheduler import RecoverySegment, StateKind
from vllm.v1.kv_offload.recovery_policy import (
    RecoveryBatch,
    RecoveryTelemetry,
    load_recovery_policy,
)


def test_hyrex_plugin_uses_shared_resource_ready_times():
    policy = load_recovery_policy("hyrex")
    segment = RecoverySegment(
        "r0",
        "full",
        StateKind.FULL_KV,
        missing_tokens=528,
        load_bytes=1,
        replay_ms=10.0,
    )
    plan = policy.plan(
        RecoveryBatch((segment,)),
        RecoveryTelemetry(h2d_ready_ms=3.0, compute_ready_ms=20.0),
    )
    assert plan.tasks[0].estimated_start_ms == 3.0
    assert plan.tasks[0].action.value == "load"
