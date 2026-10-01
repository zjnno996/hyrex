# SPDX-License-Identifier: Apache-2.0
"""Checks for isolated cache admission/eviction registration."""

import pytest

from vllm.v1.kv_offload.recovery_policy import (
    CacheEntry,
    load_cache_policy,
    register_cache_policy,
    registered_cache_policies,
)


class _UnitCachePolicy:
    name = "unit_cache_registry"

    def score(self, entry: CacheEntry, now_ms: float) -> float:
        return entry.compute_savings_ms

    def select_evictions(self, entries, bytes_needed, now_ms):
        return tuple(entry.key for entry in entries)


def test_cache_policy_registry_loads_registered_factory():
    register_cache_policy("unit_cache_registry", _UnitCachePolicy)
    policy = load_cache_policy("unit_cache_registry")
    assert policy.name == "unit_cache_registry"
    assert "unit_cache_registry" in registered_cache_policies()


def test_cache_policy_registry_rejects_duplicate_and_unsafe_names():
    with pytest.raises(ValueError):
        register_cache_policy("unit_cache_registry", _UnitCachePolicy)
    with pytest.raises(ValueError):
        load_cache_policy("../escape")
