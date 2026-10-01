# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for the isolated CacheFlow scheduling baseline."""

from vllm.v1.kv_offload.policies.cacheflow import CacheFlowPolicy, CacheFlowSegment
from vllm.v1.kv_offload.recovery_policy import RecoveryBatch, RecoveryTelemetry


def test_cacheflow_orders_by_global_marginal_recompute_savings():
    plan = CacheFlowPolicy().plan(
        RecoveryBatch((
            CacheFlowSegment("late", "a", 1, 10.0),
            CacheFlowSegment("early", "a", 1, 100.0),
        )),
        RecoveryTelemetry(),
    )
    assert [task.segment.request_id for task in plan.tasks] == ["early", "late"]


def test_cacheflow_uses_the_resource_that_finishes_first():
    plan = CacheFlowPolicy(h2d_gbps=1.0).plan(
        RecoveryBatch((
            CacheFlowSegment("load", "a", 1_000_000, 10.0),
            CacheFlowSegment("replay", "a", 100_000_000, 9.0),
        )),
        RecoveryTelemetry(),
    )
    assert [task.action for task in plan.tasks] == ["load", "replay"]


def test_cacheflow_reprioritizes_after_each_chunk_and_moves_two_pointers():
    plan = CacheFlowPolicy(h2d_gbps=1.0).plan(
        RecoveryBatch((
            CacheFlowSegment("long", "a", 4_000_000, 40.0, chunks=4),
            CacheFlowSegment("short", "a", 1_000_000, 9.0),
        )),
        RecoveryTelemetry(),
    )
    assert [task.segment.request_id for task in plan.tasks[:4]] == [
        "long", "long", "long", "long"
    ]
    assert [task.chunk_index for task in plan.tasks[:4]] == [3, 2, 1, 0]


def test_cacheflow_plan_covers_each_chunk_in_a_dispatchable_order():
    plan = CacheFlowPolicy(h2d_gbps=1.0).plan(
        RecoveryBatch((
            CacheFlowSegment("a", "prefix", 4_000_000, 2.0, chunks=4),
            CacheFlowSegment("b", "prefix", 4_000_000, 40.0, chunks=4),
        )),
        RecoveryTelemetry(),
    )
    plan.validate()
    assert {task.action for task in plan.tasks} == {"load", "replay"}


def test_cacheflow_executor_runs_resources_concurrently():
    from threading import Event

    from vllm.v1.kv_offload.policies.cacheflow import (
        CacheFlowExecutionCallbacks,
        CacheFlowExecutor,
    )

    plan = CacheFlowPolicy(h2d_gbps=1.0).plan(
        RecoveryBatch((
            CacheFlowSegment("load", "a", 1_000_000, 10.0),
            CacheFlowSegment("replay", "a", 100_000_000, 9.0),
        )),
        RecoveryTelemetry(),
    )
    load_started, replay_started = Event(), Event()

    def load(task):
        load_started.set()
        assert replay_started.wait(timeout=1)
        return f"load:{task.segment.request_id}"

    def replay(task):
        replay_started.set()
        assert load_started.wait(timeout=1)
        return f"replay:{task.segment.request_id}"

    result = CacheFlowExecutor.execute(
        plan, CacheFlowExecutionCallbacks(load=load, replay=replay)
    )
    assert result.results == ("load:load", "replay:replay")
    assert all(record.finish_ms >= record.start_ms for record in result.records)


def test_cacheflow_executor_waits_for_cuda_style_completion_events():
    from vllm.v1.kv_offload.policies.cacheflow import (
        CacheFlowExecutionCallbacks,
        CacheFlowExecutor,
    )

    class Event:
        def __init__(self):
            self.waited = False

        def synchronize(self):
            self.waited = True

    plan = CacheFlowPolicy(h2d_gbps=1.0).plan(
        RecoveryBatch((CacheFlowSegment("load", "a", 1_000_000, 10.0),)),
        RecoveryTelemetry(),
    )
    event = Event()
    result = CacheFlowExecutor.execute(
        plan, CacheFlowExecutionCallbacks(load=lambda _: event, replay=lambda _: event)
    )
    assert result.results == (event,)
    assert event.waited


def test_cacheflow_dispatcher_uses_vllm_worker_and_completes_load():
    from vllm.v1.kv_offload.policies.cacheflow import CacheFlowTask
    from vllm.v1.kv_offload.policies.cacheflow_vllm import (
        CacheFlowLoadBinding,
        CacheFlowVLLMDispatcher,
    )

    calls = []

    class Worker:
        def transfer_async(self, job_id, spec):
            calls.append(("submit", job_id, spec))
            return True

        def wait(self, job_ids):
            calls.append(("wait", job_ids))

    segment = CacheFlowSegment("r0", "prefix", 1, 1)
    task = CacheFlowTask(segment, "load", 0, 0, 1)
    dispatcher = CacheFlowVLLMDispatcher(
        Worker(),
        {
            ("r0", "prefix", 0): CacheFlowLoadBinding(
                7, "spec", lambda: calls.append(("done",))
            )
        },
    )
    callbacks = dispatcher.callbacks(lambda _: "replayed")
    assert callbacks.load(task).job_id == 7
    assert callbacks.replay(task) == "replayed"
    assert calls == [("submit", 7, "spec"), ("wait", {7}), ("done",)]


def test_cacheflow_3d_plan_chooses_axis_and_marks_stage_boundaries():
    from vllm.v1.kv_offload.policies.cacheflow import (
        CacheFlow3DPlanner,
        CacheFlow3DSegment,
    )

    short = CacheFlow3DSegment("short", 256, 128, 4, 2, 4.0, 2.0, 512)
    long = CacheFlow3DSegment("long", 1024, 512, 4, 2, 4.0, 2.0, 512)
    plan = CacheFlow3DPlanner().plan((short, long))

    assert {task.axis for task in plan.tasks if task.request_id == "short"} == {
        "layer"
    }
    assert {task.axis for task in plan.tasks if task.request_id == "long"} == {
        "token"
    }
    assert any(task.needs_boundary_activation for task in plan.tasks)


def test_cacheflow_3d_executor_loads_each_stage_boundary_once():
    from vllm.v1.kv_offload.policies.cacheflow import (
        CacheFlow3DExecutionCallbacks,
        CacheFlow3DExecutor,
        CacheFlow3DPlan,
        CacheFlow3DTask,
    )

    plan = CacheFlow3DPlan((
        CacheFlow3DTask("r0", "token", 0, "replay", 0, False),
        CacheFlow3DTask("r0", "token", 1, "replay", 0, True),
        CacheFlow3DTask("r0", "token", 1, "replay", 1, True),
    ))
    boundaries = []
    result = CacheFlow3DExecutor.execute(
        plan,
        CacheFlow3DExecutionCallbacks(
            load=lambda task: ("load", task.stage),
            replay=lambda task: ("replay", task.stage, task.unit_index),
            load_boundary=lambda task: boundaries.append((task.request_id, task.stage)),
        ),
    )
    assert boundaries == [("r0", 1)]
    assert result.results == (("replay", 0, 0), ("replay", 1, 0), ("replay", 1, 1))
