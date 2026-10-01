# SPDX-License-Identifier: Apache-2.0
"""Clean-room Marconi-style Hybrid cache admission/eviction policy.

This is a small vLLM adapter of the policy described in Marconi, not a copy of
the CC BY-NC artifact.  The paper's radix-tree integration remains an adapter
concern; this module operates on the generic cache-entry view exposed by the
core registry and preserves the defining utility rule:

    normalized recency + alpha * normalized compute-savings/bytes

The policy is intentionally state-type agnostic.  Hybrid-specific byte and
compute estimates are supplied by the vLLM observation layer.
"""

from __future__ import annotations

from vllm.v1.kv_offload.recovery_policy import (
    CacheAdmissionPolicy,
    CacheEntry,
    register_cache_policy,
)
from vllm.v1.kv_offload.policies.marconi_index import MarconiIndex, MarconiNode


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


class MarconiPolicy(CacheAdmissionPolicy):
    """Marconi's recency/FLOP-efficiency retention utility."""

    name = "marconi"

    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha

    def score(self, entry: CacheEntry, now_ms: float) -> float:
        age = max(0.0, now_ms - entry.last_access_ms)
        recency = 1.0 / (age + 1.0)
        efficiency = entry.compute_savings_ms / max(entry.byte_size, 1)
        # For a standalone score, use the unnormalized positive utility.  The
        # batch eviction path below applies the paper's min-max normalization.
        return recency + self.alpha * efficiency

    def select_evictions(
        self,
        entries: tuple[CacheEntry, ...],
        bytes_needed: int,
        now_ms: float,
    ) -> tuple[str, ...]:
        if bytes_needed <= 0:
            return ()
        if not entries:
            return ()

        ages = [max(0.0, now_ms - entry.last_access_ms) for entry in entries]
        recencies = _normalize([1.0 / (age + 1.0) for age in ages])
        efficiencies = _normalize(
            [
                entry.compute_savings_ms / max(entry.byte_size, 1)
                for entry in entries
            ]
        )
        utilities = [
            recency + self.alpha * efficiency
            for recency, efficiency in zip(recencies, efficiencies)
        ]

        # Lowest utility is evicted first.  Ties are deterministic and larger
        # entries are preferred so the requested capacity is freed quickly.
        ranked = sorted(
            zip(entries, utilities),
            key=lambda pair: (pair[1], -pair[0].byte_size, pair[0].key),
        )
        freed = 0
        evicted: list[str] = []
        for entry, _ in ranked:
            evicted.append(entry.key)
            freed += entry.byte_size
            if freed >= bytes_needed:
                break
        return tuple(evicted)


register_cache_policy("marconi", MarconiPolicy)


__all__ = ["MarconiIndex", "MarconiNode", "MarconiPolicy"]
