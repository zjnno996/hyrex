# SPDX-License-Identifier: Apache-2.0
"""Checks for the Marconi adapter at the native vLLM block-policy boundary."""

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.marconi import MarconiCachePolicy


def _key(value: int):
    return make_offload_key(bytes([value]) * 32, 0)


def test_marconi_adapter_preserves_block_eviction_contract():
    policy = MarconiCachePolicy(cache_capacity=2)
    stale = _key(1)
    recent = _key(2)
    policy.insert(stale, BlockStatus(0))
    policy.insert(recent, BlockStatus(1))
    policy.observe_metadata(
        stale,
        state_kind="full_kv",
        token_count=128,
        byte_size=1,
        compute_savings_ms=1.0,
    )
    policy.observe_metadata(
        recent,
        state_kind="recurrent",
        token_count=528,
        byte_size=1,
        compute_savings_ms=100.0,
    )
    policy.touch((recent,))

    evicted = policy.evict(1, set())
    assert evicted is not None
    assert [key for key, _ in evicted] == [stale]
    assert policy.get(recent) is not None


def test_marconi_adapter_protects_inflight_blocks():
    policy = MarconiCachePolicy(cache_capacity=2)
    key = _key(3)
    block = BlockStatus(0)
    block.ref_cnt = 1
    policy.insert(key, block)
    assert policy.evict(1, set()) is None


def test_marconi_adapter_returns_exact_physical_block_count():
    policy = MarconiCachePolicy(cache_capacity=3)
    keys = (_key(4), _key(5), _key(6))
    for block_id, key in enumerate(keys):
        policy.insert(key, BlockStatus(block_id))
        policy.observe_metadata(
            key,
            state_kind="full_kv",
            token_count=528,
            byte_size=100 + block_id,
            compute_savings_ms=float(block_id + 1),
        )

    evicted = policy.evict(2, set())
    assert evicted is not None
    assert len(evicted) == 2
