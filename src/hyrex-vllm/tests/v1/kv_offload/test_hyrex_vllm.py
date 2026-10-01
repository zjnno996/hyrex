# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.v1.kv_offload.hyrex_scheduler import (
    HyRexCostModel,
    HyRexPlanner,
    RecoveryAction,
    RecoverySegment,
    StateKind,
)
from vllm.v1.kv_offload.hyrex_vllm import (
    ALL_LOAD,
    ALL_REPLAY,
    FULL_LOAD_LINEAR_REPLAY,
    HyRexCalibrator,
    HyRexController,
    HyRexObservation,
    connector_params,
    plan_vllm_recovery,
    select_native_policy,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_observation_keeps_hybrid_state_semantics():
    recurrent = HyRexObservation(
        "r0",
        "mamba-0",
        "mamba",
        4096,
        8_000_000,
        True,
        terminal_bytes=64_000,
        terminal_ready=True,
    ).segment()
    assert recurrent.state_kind is StateKind.RECURRENT
    assert recurrent.terminal_ready


def test_calibration_changes_path_from_measured_cost():
    segment = HyRexObservation(
        "r0",
        "kv-0",
        "full_attention",
        4096,
        20_000_000,
        True,
        replay_ms=4.0,
    ).segment()
    calibrator = HyRexCalibrator(HyRexCostModel(h2d_gbps=100.0))
    assert (
        HyRexPlanner(calibrator.model()).plan([segment]).tasks[0].action
        is RecoveryAction.LOAD
    )
    calibrator.observe_h2d(20_000_000, 20.0)
    assert (
        HyRexPlanner(calibrator.model()).plan([segment]).tasks[0].action
        is RecoveryAction.REPLAY
    )


def test_h2d_calibration_is_concurrency_and_size_aware():
    calibrator = HyRexCalibrator(HyRexCostModel(h2d_gbps=20.0))
    calibrator.observe_h2d(1_000_000, 1.0, concurrency=1)
    calibrator.observe_h2d(64_000_000, 32.0, concurrency=16)

    assert calibrator.model(1, 1_000_000).h2d_gbps == pytest.approx(1.0)
    assert calibrator.model(16, 64_000_000).h2d_gbps == pytest.approx(2.0)


def test_hyrex_maps_heterogeneous_plan_to_existing_vllm_policy():
    decision = plan_vllm_recovery(
        [
            RecoverySegment(
                "r0",
                "full",
                StateKind.FULL_KV,
                4096,
                1,
                replay_ms=100.0,
            ),
            RecoverySegment(
                "r0",
                "linear",
                StateKind.RECURRENT,
                4096,
                100_000_000,
                replay_ms=1.0,
            ),
        ],
        planner=HyRexPlanner(HyRexCostModel(h2d_gbps=20.0)),
    )
    assert decision.policy == FULL_LOAD_LINEAR_REPLAY


def test_controller_plans_the_batch_once_then_projects_request_policies():
    controller = HyRexController(HyRexCalibrator(HyRexCostModel(h2d_gbps=20.0)))
    decision = controller.plan_batch(
        [
            HyRexObservation(
                "urgent",
                "full",
                "full_attention",
                4096,
                1,
                True,
                replay_ms=100.0,
                priority=1.0,
            ),
            HyRexObservation(
                "urgent",
                "linear",
                "mamba",
                4096,
                100_000_000,
                True,
                replay_ms=1.0,
                priority=1.0,
            ),
            HyRexObservation(
                "normal", "full", "full_attention", 4096, 1, True, replay_ms=100.0
            ),
            HyRexObservation(
                "normal", "linear", "mamba", 4096, 1, True, replay_ms=100.0
            ),
        ]
    )

    assert [task.segment.request_id for task in decision.plan.tasks[:2]] == [
        "urgent",
        "urgent",
    ]
    assert decision.requests["urgent"].policy == FULL_LOAD_LINEAR_REPLAY
    assert decision.requests["normal"].policy == ALL_LOAD


def test_connector_params_only_serializes_a_worker_executable_decision():
    decision = plan_vllm_recovery(
        [
            RecoverySegment("r", "full", StateKind.FULL_KV, 8, 1, replay_ms=10),
            RecoverySegment(
                "r", "linear", StateKind.RECURRENT, 8, 1_000_000_000, replay_ms=1
            ),
        ],
        planner=HyRexPlanner(HyRexCostModel(h2d_gbps=20)),
    )
    params = connector_params(decision, replay_ms=1.0, source_key="session:r")
    assert params["hyrex_recovery_policy"] == FULL_LOAD_LINEAR_REPLAY
    assert params["hyrex_source_key"] == "session:r"


def test_native_policy_uses_actual_bytes_and_measured_replay_costs():
    assert (
        select_native_policy(
            missing_tokens=4096,
            full_load_bytes=1,
            recurrent_load_bytes=100_000_000,
            full_replay_ms=100.0,
            recurrent_replay_ms=1.0,
            h2d_gbps=20.0,
        )
        == FULL_LOAD_LINEAR_REPLAY
    )

    assert (
        select_native_policy(
            missing_tokens=4096,
            full_load_bytes=1,
            recurrent_load_bytes=1,
            full_replay_ms=100.0,
            recurrent_replay_ms=100.0,
            h2d_gbps=20.0,
        )
        == ALL_LOAD
    )
    assert (
        select_native_policy(
            missing_tokens=4096,
            full_load_bytes=1,
            recurrent_load_bytes=1,
            full_replay_ms=100.0,
            recurrent_replay_ms=1.0,
            h2d_gbps=20.0,
            full_source_ready=False,
            recurrent_source_ready=False,
        )
        == ALL_REPLAY
    )
    assert (
        select_native_policy(
            missing_tokens=4096,
            full_load_bytes=1_000_000_000,
            recurrent_load_bytes=1_000_000_000,
            full_replay_ms=1.0,
            recurrent_replay_ms=1.0,
            h2d_gbps=20.0,
        )
        == ALL_REPLAY
    )
