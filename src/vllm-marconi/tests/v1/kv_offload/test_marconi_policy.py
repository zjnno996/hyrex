# SPDX-License-Identifier: Apache-2.0
"""Clean-room behavioral checks for the Marconi utility policy."""

from vllm.v1.kv_offload.policies.marconi import MarconiPolicy
from vllm.v1.kv_offload.recovery_policy import CacheEntry


def test_marconi_eviction_prefers_low_utility_entry():
    entries = (
        CacheEntry(
            key="recent-expensive",
            state_kind="full_kv",
            token_count=1024,
            byte_size=100,
            last_access_ms=99,
            compute_savings_ms=100,
        ),
        CacheEntry(
            key="stale-cheap",
            state_kind="full_kv",
            token_count=128,
            byte_size=100,
            last_access_ms=1,
            compute_savings_ms=1,
        ),
    )
    assert MarconiPolicy(alpha=1.0).select_evictions(entries, 100, 100) == (
        "stale-cheap",
    )


def test_marconi_eviction_is_deterministic_and_capacity_aware():
    entries = tuple(
        CacheEntry(
            key=f"entry-{idx}",
            state_kind="recurrent",
            token_count=528,
            byte_size=60,
            last_access_ms=idx,
            compute_savings_ms=float(idx),
        )
        for idx in range(3)
    )
    evicted = MarconiPolicy(alpha=1.0).select_evictions(entries, 120, 10)
    assert len(evicted) == 2
    assert evicted == MarconiPolicy(alpha=1.0).select_evictions(entries, 120, 10)
