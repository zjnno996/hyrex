# SPDX-License-Identifier: Apache-2.0

from vllm.v1.kv_offload.policies.kvpr_hybrid import (
    KVPRHybridPolicy,
    KVPRHybridSegment,
)
from vllm.v1.kv_offload.recovery_policy import RecoveryBatch, RecoveryTelemetry


def _segment() -> KVPRHybridSegment:
    return KVPRHybridSegment(
        "r0",
        total_tokens=4096,
        full_kv_bytes_per_token=4096,
        recurrent_state_bytes=8 * 1024 * 1024,
        replay_ms_per_token=0.02,
        checkpoint_tokens=528,
    )


def test_kvpr_h_uses_checkpoint_aligned_candidates():
    policy = KVPRHybridPolicy(h2d_gbps=32.0)
    plan = policy.plan(RecoveryBatch((_segment(),)), RecoveryTelemetry())[0]
    assert plan.replay_tokens in {0, 528, 1056, 1584, 2112, 2640, 3168, 3696, 4096}
    assert plan.replay_tokens + plan.full_kv_load_tokens == 4096


def test_kvpr_h_can_choose_terminal_state_load():
    segment = KVPRHybridSegment(
        "r0", 528, 1024, 1024, replay_ms_per_token=10.0
    )
    plan = KVPRHybridPolicy(h2d_gbps=32.0).plan(
        RecoveryBatch((segment,)), RecoveryTelemetry()
    )[0]
    assert plan.recurrent_action == "terminal_load"
    assert plan.replay_tokens == 0


def test_kvpr_h_uses_replay_when_state_load_is_expensive():
    segment = KVPRHybridSegment(
        "r0", 4096, 1024, 4 * 1024 * 1024, replay_ms_per_token=0.001
    )
    plan = KVPRHybridPolicy(h2d_gbps=1.0).plan(
        RecoveryBatch((segment,)), RecoveryTelemetry()
    )[0]
    assert plan.recurrent_action == "replay"
    assert plan.replay_tokens > 0


def test_kvpr_h_responds_to_h2d_telemetry():
    segment = _segment()
    policy = KVPRHybridPolicy()
    fast = policy.plan(
        RecoveryBatch((segment,)), RecoveryTelemetry(h2d_gbps=64.0)
    )[0]
    slow = policy.plan(
        RecoveryBatch((segment,)), RecoveryTelemetry(h2d_gbps=8.0)
    )[0]
    assert slow.estimated_ms >= fast.estimated_ms


def test_contiguous_runtime_plan_charges_recurrent_checkpoint():
    segment = KVPRHybridSegment(
        "r0", 1056, 32_768, 51_511_296, replay_ms_per_token=0.01
    )
    plan = KVPRHybridPolicy(h2d_gbps=19).plan_contiguous_prefix(segment)
    expected_bytes = plan.load_tokens * 32_768
    if plan.load_tokens:
        expected_bytes += 51_511_296
    assert plan.h2d_ms == expected_bytes / 19e9 * 1e3
    assert plan.load_tokens + plan.replay_tokens == 1056
    assert plan.estimated_ms == plan.h2d_ms + plan.replay_ms
