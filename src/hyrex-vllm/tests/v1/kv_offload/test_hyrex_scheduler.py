# SPDX-License-Identifier: Apache-2.0
"""Small behavioral checks for the HyRex recovery planner."""

import pytest

from vllm.v1.kv_offload.hyrex_scheduler import (
    HyRexCostModel,
    HyRexPlanner,
    RecoveryAction,
    RecoverySegment,
    StateKind,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_recurrent_terminal_state_beats_long_replay():
    segment = RecoverySegment(
        request_id="r0",
        segment_id="s0",
        state_kind=StateKind.RECURRENT,
        missing_tokens=32_000,
        load_bytes=65_000_000,
        terminal_bytes=1_000_000,
        terminal_ready=True,
        replay_ms=50.0,
    )
    task = HyRexPlanner(HyRexCostModel(h2d_gbps=20.0)).plan([segment]).tasks[0]
    assert task.action is RecoveryAction.TERMINAL_LOAD
    assert task.estimated_cost_ms < task.segment.replay_ms  # type: ignore[operator]


def test_shared_source_does_not_hide_distinct_h2d_destinations():
    common = dict(
        state_kind=StateKind.FULL_KV,
        missing_tokens=528,
        load_bytes=20_000_000,
        lookup_ms=2.0,
        replay_ms=100.0,
        source_key="conversation-prefix",
    )
    segments = [
        RecoverySegment(request_id="r0", segment_id="s0", **common),
        RecoverySegment(request_id="r1", segment_id="s0", **common),
    ]
    plan = HyRexPlanner(HyRexCostModel(h2d_gbps=20.0)).plan(segments)
    assert all(task.action is RecoveryAction.LOAD for task in plan.tasks)
    assert plan.tasks[1].source_coalesced
    assert not plan.tasks[1].physical_transfer_coalesced
    assert plan.tasks[1].estimated_cost_ms > 0.0


def test_identical_materialization_has_one_physical_load():
    common = dict(
        state_kind=StateKind.FULL_KV,
        missing_tokens=528,
        load_bytes=20_000_000,
        lookup_ms=2.0,
        replay_ms=100.0,
        source_key="conversation-prefix",
        materialization_key="gpu-blocks:42-43",
    )
    plan = HyRexPlanner(HyRexCostModel(h2d_gbps=20.0)).plan(
        [
            RecoverySegment(request_id="r0", segment_id="s0", **common),
            RecoverySegment(request_id="r1", segment_id="s0", **common),
        ]
    )

    assert len(plan.logical_load_tasks) == 2
    assert len(plan.physical_load_tasks) == 1
    assert plan.tasks[1].source_coalesced
    assert plan.tasks[1].physical_transfer_coalesced
    assert plan.tasks[1].estimated_cost_ms == 0.0


def test_readiness_removes_load_candidate():
    segment = RecoverySegment(
        request_id="r0",
        segment_id="s0",
        state_kind=StateKind.FULL_KV,
        missing_tokens=528,
        load_bytes=1,
        replay_ms=3.0,
        source_ready=False,
    )
    task = HyRexPlanner().plan([segment]).tasks[0]
    assert task.action is RecoveryAction.REPLAY
