# SPDX-License-Identifier: Apache-2.0
"""Unit tests for group selection in native CPU/H2D hybrid recovery."""

from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


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


def test_native_offload_loads_all_groups_by_default(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    monkeypatch.setenv("VLLM_MOONCAKE_HYBRID_POLICY", "all_load")
    assert scheduler._get_loaded_group_indices() == (0, 1)


def test_native_offload_p4_loads_only_linear_group(monkeypatch):
    scheduler = _scheduler_with_full_and_linear_groups()
    monkeypatch.setenv(
        "VLLM_MOONCAKE_HYBRID_POLICY", "full_replay_linear_load"
    )
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
    monkeypatch.setenv(
        "VLLM_MOONCAKE_HYBRID_POLICY", "full_load_linear_replay"
    )
    assert scheduler._get_loaded_group_indices() == (0,)


def test_native_offload_replays_final_mamba_page():
    scheduler = _scheduler_with_full_and_linear_groups()
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(offloaded_block_size=528),
            SimpleNamespace(offloaded_block_size=528),
        )
    )
    scheduler._loaded_group_indices = (0, 1)

    assert scheduler._max_hit_before_last_token_replay(1056) == 528
    assert scheduler._max_hit_before_last_token_replay(1312) == 1056


def test_independent_cpu_lookup_keeps_deeper_full_kv(monkeypatch):
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
    scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(offloaded_block_size=528),
            SimpleNamespace(offloaded_block_size=528),
        )
    )
    request = SimpleNamespace(
        request_id="r0",
        hybrid_independent_stage=0,
        hybrid_independent_target_tokens=0,
        hybrid_local_tokens=0,
    )
    groups = [SimpleNamespace(block_ids=[]) for _ in range(2)]
    state = SimpleNamespace(
        group_states=groups,
        req=request,
        update_offload_keys=lambda: None,
        update_num_hit_blocks=lambda _: None,
    )
    scheduler._req_status = {"r0": state}
    scheduler._lookup = lambda _: 528
    scheduler._independent_full_depth = lambda _: 1056
    scheduler._touch = lambda _: None
    monkeypatch.setenv("VLLM_MOONCAKE_HYBRID_POLICY", "independent_full_kv")

    assert scheduler.get_num_new_matched_tokens(request, 0) == (528, True)
    assert request.hybrid_independent_stage == 0
    assert request.hybrid_independent_target_tokens == 1056
    request.hybrid_independent_stage = 1  # phase-one transfer completed
    groups[0].block_ids.append(7)
    assert scheduler.get_num_new_matched_tokens(request, 528) == (528, True)
    assert groups[0].block_ids == [7]
    assert request.hybrid_independent_stage == 1
    assert request.hybrid_local_tokens == 528
    assert state.lookup_groups == (0,)
