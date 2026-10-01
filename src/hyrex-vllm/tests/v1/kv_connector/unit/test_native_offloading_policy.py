# SPDX-License-Identifier: Apache-2.0
"""Unit tests for group selection in native CPU/H2D hybrid recovery."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import TransferJob
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    _hyrex_bucket,
    native_cache_event,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_offload.base import GPULoadStoreSpec

pytestmark = pytest.mark.skip_global_cleanup


def _scheduler_with_full_and_linear_groups() -> OffloadingConnectorScheduler:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(kv_group_configs=(object(), object()))
    scheduler._group_specs = (
        object(),
        MambaSpec(
            block_size=16,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
        ),
    )
    return scheduler


def _gpu_load_spec() -> GPULoadStoreSpec:
    return GPULoadStoreSpec([1, 2], group_sizes=(1, 1), block_indices=(0, 0))


def _set_hyrex_layout(
    scheduler: OffloadingConnectorScheduler, *, page_size_padded: int | None = None
) -> None:
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(gpu_block_size=16),
            SimpleNamespace(gpu_block_size=16),
        )
    )
    scheduler._group_specs = tuple(
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=page_size_padded,
        )
        for _ in range(2)
    )


def test_native_offload_loads_all_groups_by_default(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    monkeypatch.setenv("VLLM_MOONCAKE_HYBRID_POLICY", "all_load")
    assert scheduler._get_loaded_group_indices() == (0, 1)


def test_native_offload_p4_loads_only_linear_group(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    monkeypatch.setenv("VLLM_MOONCAKE_HYBRID_POLICY", "full_replay_linear_load")
    assert scheduler._get_loaded_group_indices() == (1,)


def test_native_offload_p3_loads_only_full_group(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
        ),
        MambaSpec(
            block_size=16,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
        ),
    )
    monkeypatch.setenv("VLLM_MOONCAKE_HYBRID_POLICY", "full_load_linear_replay")
    assert scheduler._get_loaded_group_indices() == (0,)


def test_request_local_hyrex_policy_does_not_change_the_default(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
        ),
        scheduler._group_specs[1],
    )
    monkeypatch.setenv("VLLM_MOONCAKE_HYBRID_POLICY", "all_load")

    assert scheduler._get_loaded_group_indices(
        {"hyrex_recovery_policy": "full_load_linear_replay"}
    ) == (0,)
    assert scheduler._get_loaded_group_indices() == (0, 1)


def test_hyrex_store_admission_is_state_aware():
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(gpu_block_size=528),
            SimpleNamespace(gpu_block_size=528),
        )
    )
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=528,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=100_000_000,
        ),
        MambaSpec(
            block_size=528,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=16,
        ),
    )
    scheduler._group_layer_counts = (1, 1)
    scheduler._hyrex_measured_h2d_gbps = None
    request = SimpleNamespace(
        num_tokens=528,
        kv_transfer_params={
            "hybrid_baseline": "hyrex",
            "hyrex_h2d_gbps": 1.0,
            "hyrex_full_replay_ms_per_token": 0.001,
            "hyrex_recurrent_replay_ms_per_token": 0.001,
        },
    )

    assert scheduler._get_store_group_indices(request) == (1,)


def test_hyrex_store_admission_uses_concurrency_profile():
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(kv_group_configs=(
        SimpleNamespace(gpu_block_size=100),
        SimpleNamespace(gpu_block_size=100),
    ))
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=100,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=10_000_000,
        ),
        MambaSpec(
            block_size=100,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=10_000_000,
        ),
    )
    scheduler._group_layer_counts = (1, 1)
    scheduler._hyrex_measured_h2d_gbps = None
    scheduler._hyrex_measured_h2d_gbps_by_bucket = {
        (_hyrex_bucket(10_000_000), _hyrex_bucket(1)): 10.0,
        (_hyrex_bucket(10_000_000), _hyrex_bucket(16)): 1.0,
    }
    scheduler._hyrex_measured_replay_ms_per_token = {}
    scheduler._hyrex_measured_replay_by_bucket = {}
    request = SimpleNamespace(
        num_tokens=100,
        kv_transfer_params={
            "hybrid_baseline": "hyrex",
            "hyrex_concurrency": 16,
            "hyrex_h2d_gbps": 10.0,
            "hyrex_full_replay_ms_per_token": 0.02,
            "hyrex_recurrent_replay_ms_per_token": 0.02,
        },
    )

    assert scheduler._get_store_group_indices(request) == ()


def test_hyrex_lookup_keeps_one_state_type_hit(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
        ),
        scheduler._group_specs[1],
    )
    scheduler._full_attention_groups = (0,)
    scheduler._sliding_window_groups = (1,)
    status = SimpleNamespace(
        lookup_groups=(0, 1), loaded_group_indices=(0, 1)
    )

    def lookup(req_status):
        return 0 if req_status.lookup_groups == (0,) else 528

    monkeypatch.setattr(scheduler, "_lookup", lookup)
    hit_tokens, available = scheduler._lookup_hyrex_groups(status)

    assert hit_tokens == 528
    assert available == (1,)
    assert status.lookup_groups == (0, 1)
    assert status.loaded_group_indices == (0, 1)


def test_hyrex_lookup_uses_common_boundary_and_propagates_defer(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
        ),
        scheduler._group_specs[1],
    )
    scheduler._full_attention_groups = (0,)
    scheduler._sliding_window_groups = (1,)
    status = SimpleNamespace(lookup_groups=(0, 1), loaded_group_indices=(0, 1))
    hits = {(0,): 1056, (1,): 528}
    monkeypatch.setattr(scheduler, "_lookup", lambda state: hits[state.lookup_groups])

    assert scheduler._lookup_hyrex_groups(status) == (528, (0, 1))
    assert status.hyrex_hit_tokens_by_kind == {"full": 1056, "recurrent": 528}

    hits[(1,)] = None
    assert scheduler._lookup_hyrex_groups(status) == (None, ())


def test_hyrex_uses_longer_independent_full_boundary():
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(kv_group_configs=(
        SimpleNamespace(gpu_block_size=528),
        SimpleNamespace(gpu_block_size=528),
    ))
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=528,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=40_000,
        ),
        MambaSpec(
            block_size=528,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=100_000_000,
        ),
    )
    scheduler._group_layer_counts = (1, 1)
    scheduler._full_attention_groups = (0,)
    scheduler._sliding_window_groups = (1,)
    params = {
        "hybrid_baseline": "hyrex",
        "hyrex_h2d_gbps": 1.0,
        "hyrex_full_replay_ms_per_token": 0.1,
        "hyrex_recurrent_replay_ms_per_token": 0.001,
    }
    status = SimpleNamespace(
        req=SimpleNamespace(kv_transfer_params=params),
        hybrid_policy="all_load",
        loaded_group_indices=(0, 1),
        store_group_indices=(0, 1),
        lookup_groups=(0, 1),
        hyrex_hit_tokens_by_kind={"full": 1056, "recurrent": 528},
    )

    selected = scheduler._bind_hyrex_policy(status, 528, (0, 1))

    assert selected == 1056
    assert status.hybrid_policy == "full_load_linear_replay"
    assert status.loaded_group_indices == (0,)


def test_hyrex_finalizes_batch_before_allocation(monkeypatch):
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._hyrex_pending_lookups = {
        "relaxed": (528, (0, 1)),
        "urgent": (528, (0, 1)),
    }
    scheduler._hyrex_preplanned = {}
    scheduler._req_status = {
        request_id: SimpleNamespace(
            req=SimpleNamespace(
                request_id=request_id,
                priority=0,
                arrival_time=1.0,
                kv_transfer_params={
                    "hyrex_priority": priority,
                    "hyrex_deadline_ms": deadline,
                },
            ),
            hybrid_policy="all_load",
        )
        for request_id, priority, deadline in (
            ("relaxed", 0.0, 100.0),
            ("urgent", 1.0, 5.0),
        )
    }
    seen = []

    def bind(status, hit_tokens, available):
        seen.append(status.req.request_id)
        status.hybrid_policy = "all_load"
        return hit_tokens

    monkeypatch.setattr(scheduler, "_bind_hyrex_policy", bind)
    scheduler._finalize_hyrex_batch()

    assert seen == ["urgent", "relaxed"]
    assert scheduler._hyrex_pending_lookups == {}
    assert set(scheduler._hyrex_preplanned) == {"urgent", "relaxed"}
    assert all(
        state.req.kv_transfer_params["hyrex_concurrency"] == 2
        for state in scheduler._req_status.values()
    )


def test_hyrex_binds_policy_after_real_lookup(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(gpu_block_size=528),
            SimpleNamespace(gpu_block_size=528),
        )
    )
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=528,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=2_162_688,
        ),
        MambaSpec(
            block_size=528,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=2_146_304,
        ),
    )
    scheduler._group_layer_counts = (8, 24)
    params = {
        "hybrid_baseline": "hyrex",
        "hyrex_h2d_gbps": 20.0,
        "hyrex_full_replay_ms_per_token": 0.0021,
        "hyrex_recurrent_replay_ms_per_token": 0.0052,
    }
    status = SimpleNamespace(
        req=SimpleNamespace(kv_transfer_params=params),
        hybrid_policy="all_load",
        loaded_group_indices=(0, 1),
        store_group_indices=(0, 1),
        lookup_groups=(0, 1),
    )

    scheduler._full_attention_groups = (0,)
    scheduler._sliding_window_groups = (1,)
    scheduler._bind_hyrex_policy(status, 528)

    assert status.hybrid_policy == "full_replay_linear_load"
    assert status.loaded_group_indices == (1,)
    assert status.store_group_indices == (0, 1)
    assert params["hyrex_recovery_policy"] == status.hybrid_policy


def test_hyrex_changes_path_when_h2d_queue_is_busy(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(gpu_block_size=528),
            SimpleNamespace(gpu_block_size=528),
        )
    )
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=528,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=10_000_000,
        ),
        MambaSpec(
            block_size=528,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=10_000_000,
        ),
    )
    scheduler._group_layer_counts = (1, 1)
    scheduler._full_attention_groups = (0,)
    scheduler._sliding_window_groups = (1,)
    scheduler._hyrex_h2d_ready_ms = 0.0
    scheduler._hyrex_compute_ready_ms = 0.0
    params = {
        "hybrid_baseline": "hyrex",
        "hyrex_h2d_gbps": 1.0,
        "hyrex_full_replay_ms_per_token": 30 / 528,
        "hyrex_recurrent_replay_ms_per_token": 30 / 528,
    }
    status = SimpleNamespace(
        req=SimpleNamespace(request_id="idle", kv_transfer_params=params),
        hybrid_policy="all_load",
        loaded_group_indices=(0, 1),
        lookup_groups=(0, 1),
    )
    scheduler._bind_hyrex_policy(status, 528)
    assert status.hybrid_policy == "all_load"

    scheduler._hyrex_h2d_ready_ms = 25.0
    scheduler._hyrex_compute_ready_ms = 0.0
    params["hyrex_recovery_policy"] = "all_load"
    scheduler._bind_hyrex_policy(status, 528)
    assert status.hybrid_policy == "full_load_linear_replay"
    assert scheduler._hyrex_h2d_ready_ms == 35.0
    assert scheduler._hyrex_compute_ready_ms == 30.0


def test_hyrex_calibrates_h2d_from_completed_native_loads():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._hyrex_measured_h2d_gbps = None
    scheduler._hyrex_measured_h2d_gbps_by_bucket = {}
    scheduler._observe_hyrex_h2d(20_000_000, 2.0, concurrency=1)
    assert scheduler._hyrex_measured_h2d_gbps == 10.0
    scheduler._observe_hyrex_h2d(20_000_000, 1.0, concurrency=1)
    assert scheduler._hyrex_measured_h2d_gbps == 12.0
    scheduler._observe_hyrex_h2d(20_000_000, 4.0, concurrency=16)
    assert len(scheduler._hyrex_measured_h2d_gbps_by_bucket) == 2


def test_hyrex_queue_debt_decays_across_scheduler_steps():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._hyrex_h2d_ready_ms = 25.0
    scheduler._hyrex_compute_ready_ms = 10.0
    scheduler._hyrex_queue_clock_ms = 100.0

    scheduler._decay_hyrex_queue_debt(now_ms=106.0)

    assert scheduler._hyrex_h2d_ready_ms == 19.0
    assert scheduler._hyrex_compute_ready_ms == 4.0
    assert scheduler._hyrex_queue_clock_ms == 106.0

    scheduler._decay_hyrex_queue_debt(now_ms=120.0)
    assert scheduler._hyrex_h2d_ready_ms == 5.0
    assert scheduler._hyrex_compute_ready_ms == 0.0


def test_hyrex_calibrates_replay_by_replayed_state_kind():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._hyrex_measured_replay_ms_per_token = {}
    scheduler.observe_hyrex_replay({
        "linear": ("full_load_linear_replay", 100, 20.0),
        "full": ("full_replay_linear_load", 100, 40.0),
        "ignored": ("all_load", 100, 1.0),
    })
    assert scheduler._hyrex_measured_replay_ms_per_token == {
        "recurrent": 0.2,
        "full": 0.4,
    }
    scheduler.observe_hyrex_replay({
        "linear": ("full_load_linear_replay", 100, 30.0),
    })
    assert scheduler._hyrex_measured_replay_ms_per_token["recurrent"] == pytest.approx(
        0.22
    )


def test_kvpr_hybrid_caps_native_load_at_checkpoint_split():
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(gpu_block_size=528, offloaded_block_size=528),
            SimpleNamespace(gpu_block_size=528, offloaded_block_size=528),
        )
    )
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=528,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=2_162_688,
        ),
        MambaSpec(
            block_size=528,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=2_146_304,
        ),
    )
    scheduler._group_layer_counts = (8, 24)
    params = {
        "hybrid_baseline": "kvpr_hybrid",
        "kvpr_h2d_gbps": 1.0,
        "kvpr_replay_ms_per_token": 0.0001,
    }
    status = SimpleNamespace(
        req=SimpleNamespace(request_id="r0", kv_transfer_params=params)
    )

    selected = scheduler._bind_kvpr_hybrid(status, 1056)

    assert selected in {0, 528, 1056}
    assert selected < 1056
    assert params["kvpr_load_tokens"] == selected
    assert params["kvpr_replay_tokens"] + selected == 1056

    event = native_cache_event(
        "r0",
        0,
        selected,
        "all_load",
        available_external_tokens=1056,
        recovery_action="kvpr_split",
    )
    assert event["cache_state"] == "cpu"
    assert event["cpu_hit_tokens"] == 1056
    assert event["h2d_tokens"] == selected


def test_cacheflow_hybrid_caps_native_load_by_chunks():
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(gpu_block_size=528, offloaded_block_size=528),
            SimpleNamespace(gpu_block_size=528, offloaded_block_size=528),
        )
    )
    scheduler._group_specs = (
        FullAttentionSpec(
            block_size=528,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            page_size_padded=2_162_688,
        ),
        MambaSpec(
            block_size=528,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=2_146_304,
        ),
    )
    scheduler._group_layer_counts = (8, 24)
    params = {
        "hybrid_baseline": "cacheflow_hybrid",
        "cacheflow_h2d_gbps": 1.0,
        "cacheflow_replay_ms_per_token": 0.0001,
    }
    status = SimpleNamespace(
        req=SimpleNamespace(request_id="r0", kv_transfer_params=params)
    )

    selected = scheduler._bind_cacheflow_hybrid(status, 1056)

    assert selected in {0, 528, 1056}
    assert selected < 1056
    assert params["cacheflow_load_tokens"] == selected
    assert params["cacheflow_replay_tokens"] + selected == 1056


def test_cacheflow_orders_native_jobs_by_remaining_replay_work():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._current_batch_load_jobs = {
        1: TransferJob("short", (object(), _gpu_load_spec())),
        2: TransferJob("long", (object(), _gpu_load_spec())),
    }
    scheduler._req_status = {
        request_id: SimpleNamespace(
            req=SimpleNamespace(kv_transfer_params={
                "hybrid_baseline": "cacheflow_hybrid",
                "cacheflow_replay_ms": replay_ms,
            })
        )
        for request_id, replay_ms in (("short", 1.0), ("long", 10.0))
    }

    scheduler._schedule_cacheflow_load_jobs()

    assert list(scheduler._current_batch_load_jobs) == [2, 1]


def test_hyrex_orders_native_h2d_jobs_by_priority(monkeypatch):
    scheduler = object.__new__(OffloadingConnectorScheduler)
    _set_hyrex_layout(scheduler)
    scheduler._current_batch_load_jobs = {
        1: TransferJob("slow", (object(), _gpu_load_spec())),
        2: TransferJob("urgent", (object(), _gpu_load_spec())),
    }
    scheduler._req_status = {
        request_id: SimpleNamespace(
            req=SimpleNamespace(
                kv_transfer_params={
                    "hyrex_replay_ms": 10.0,
                    "hyrex_priority": priority,
                }
            )
        )
        for request_id, priority in (("slow", 0.0), ("urgent", 1.0))
    }
    monkeypatch.setenv("VLLM_HYREX_SCHEDULE_LOADS", "1")

    scheduler._schedule_hyrex_load_jobs()

    assert list(scheduler._current_batch_load_jobs) == [2, 1]


def test_hyrex_preserves_arrival_order_instead_of_request_id_order(monkeypatch):
    scheduler = object.__new__(OffloadingConnectorScheduler)
    _set_hyrex_layout(scheduler)
    scheduler._current_batch_load_jobs = {
        1: TransferJob("request-10", (object(), _gpu_load_spec())),
        2: TransferJob("request-2", (object(), _gpu_load_spec())),
    }
    scheduler._req_status = {
        request_id: SimpleNamespace(
            req=SimpleNamespace(
                priority=0,
                arrival_time=arrival_time,
                kv_transfer_params={"hyrex_replay_ms": 10.0},
            )
        )
        for request_id, arrival_time in (
            ("request-10", 2.0),
            ("request-2", 1.0),
        )
    }
    monkeypatch.setenv("VLLM_HYREX_SCHEDULE_LOADS", "1")

    scheduler._schedule_hyrex_load_jobs()

    assert list(scheduler._current_batch_load_jobs) == [2, 1]


def test_hyrex_orders_native_h2d_jobs_by_remaining_slo(monkeypatch):
    scheduler = object.__new__(OffloadingConnectorScheduler)
    _set_hyrex_layout(scheduler)
    scheduler._current_batch_load_jobs = {
        1: TransferJob("relaxed", (object(), _gpu_load_spec())),
        2: TransferJob("urgent", (object(), _gpu_load_spec())),
    }
    scheduler._req_status = {
        request_id: SimpleNamespace(
            req=SimpleNamespace(
                priority=0,
                arrival_time=1.0,
                kv_transfer_params={
                    "hyrex_replay_ms": 10.0,
                    "hyrex_deadline_ms": deadline_ms,
                },
            )
        )
        for request_id, deadline_ms in (("relaxed", 100.0), ("urgent", 5.0))
    }
    monkeypatch.setenv("VLLM_HYREX_SCHEDULE_LOADS", "1")

    scheduler._schedule_hyrex_load_jobs()

    assert list(scheduler._current_batch_load_jobs) == [2, 1]


def test_hyrex_promotes_starved_native_h2d_job(monkeypatch):
    scheduler = object.__new__(OffloadingConnectorScheduler)
    _set_hyrex_layout(scheduler)
    scheduler._current_batch_load_jobs = {
        1: TransferJob("starved", (object(), _gpu_load_spec())),
        2: TransferJob("urgent", (object(), _gpu_load_spec())),
    }
    scheduler._req_status = {
        "starved": SimpleNamespace(
            req=SimpleNamespace(
                priority=0,
                kv_transfer_params={
                    "hyrex_replay_ms": 10.0,
                    "hyrex_priority": 0.0,
                    "hyrex_arrival_ms": 0.0,
                    "hyrex_starvation_ms": 1.0,
                },
            )
        ),
        "urgent": SimpleNamespace(
            req=SimpleNamespace(
                priority=0,
                kv_transfer_params={
                    "hyrex_replay_ms": 10.0,
                    "hyrex_priority": 1.0,
                },
            )
        ),
    }
    monkeypatch.setenv("VLLM_HYREX_SCHEDULE_LOADS", "1")

    scheduler._schedule_hyrex_load_jobs()

    assert list(scheduler._current_batch_load_jobs) == [1, 2]


def test_hyrex_never_changes_a_native_load_to_replay(monkeypatch):
    scheduler = object.__new__(OffloadingConnectorScheduler)
    _set_hyrex_layout(scheduler, page_size_padded=20_000_000_000)
    scheduler._current_batch_load_jobs = {
        1: TransferJob("r0", (object(), _gpu_load_spec())),
        2: TransferJob("r1", (object(), _gpu_load_spec())),
    }
    scheduler._req_status = {
        request_id: SimpleNamespace(
            req=SimpleNamespace(
                kv_transfer_params={
                    "hyrex_replay_ms": 0.1,
                }
            )
        )
        for request_id in ("r0", "r1")
    }
    monkeypatch.setenv("VLLM_HYREX_SCHEDULE_LOADS", "1")

    scheduler._schedule_hyrex_load_jobs()

    assert list(scheduler._current_batch_load_jobs) == [1, 2]
