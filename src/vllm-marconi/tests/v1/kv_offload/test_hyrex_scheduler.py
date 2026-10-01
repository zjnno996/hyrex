# SPDX-License-Identifier: Apache-2.0
"""Small behavioral checks for the HyRex recovery planner."""

from vllm.v1.kv_offload.hyrex_scheduler import (
    HyRexCostModel,
    HyRexPlanner,
    RecoveryAction,
    RecoverySegment,
    StateKind,
    match_hybrid_prefix,
)


def test_independent_kv_and_state_boundaries():
    match = match_hybrid_prefix(
        prompt_tokens=8193,
        kv_page_tokens=16,
        kv_page_hits=[True] * 512,
        state_checkpoints=[0, 5280, 6272],
    )
    assert (match.kv_tokens, match.state_tokens) == (8192, 6272)
    assert match.state_replay_tokens == 1920
    assert match.uncached_suffix_tokens == 1


def test_match_requires_contiguous_kv_and_bounded_state():
    match = match_hybrid_prefix(
        prompt_tokens=50,
        kv_page_tokens=16,
        kv_page_hits=[True, False, True],
        state_checkpoints=[0, 12, 48],
    )
    assert (match.kv_tokens, match.state_tokens) == (16, 12)


def test_last_prompt_token_is_replayed_once():
    match = match_hybrid_prefix(
        prompt_tokens=32,
        kv_page_tokens=16,
        kv_page_hits=[True, True],
        state_checkpoints=[16, 32],
    )
    assert (match.kv_tokens, match.state_tokens) == (16, 16)
    assert match.uncached_suffix_tokens == 16


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
