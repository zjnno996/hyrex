# SPDX-License-Identifier: Apache-2.0
"""Marconi policy adapter for vLLM's native block offloader.

The native manager evicts a number of physical blocks, while Marconi scores
logical cache entries.  This adapter keeps that contract intact and exposes
``observe_metadata`` for a future scheduler/index bridge.  Until such metadata
is supplied, every block has unit size and zero compute-savings metadata, so
the behavior is a deterministic recency fallback rather than a fabricated
Hybrid estimate.
"""

from __future__ import annotations

from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    OffloadKey,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.policies.marconi import MarconiPolicy
from vllm.v1.kv_offload.recovery_policy import CacheEntry


class MarconiCachePolicy(CachePolicy):
    """Block-compatible Marconi adapter with optional Hybrid observations."""

    def __init__(self, cache_capacity: int, alpha: float = 1.0):
        del cache_capacity  # the manager owns physical capacity accounting
        self._blocks: dict[OffloadKey, BlockStatus] = {}
        self._entries: dict[OffloadKey, CacheEntry] = {}
        self._clock = 0.0
        self._policy = MarconiPolicy(alpha=alpha)

    @property
    def blocks(self) -> dict[OffloadKey, BlockStatus]:
        """Expose the same inspection surface as the native LRU policy."""
        return self._blocks

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self._blocks.get(key)

    def observe_metadata(
        self,
        key: OffloadKey,
        *,
        state_kind: str,
        token_count: int,
        byte_size: int,
        compute_savings_ms: float,
    ) -> None:
        """Attach measured/estimated Hybrid metadata to one cached block."""
        if token_count < 0 or byte_size < 1 or compute_savings_ms < 0:
            raise ValueError("invalid Marconi block metadata")
        previous = self._entries.get(key)
        last_access_ms = self._clock if previous is None else previous.last_access_ms
        self._entries[key] = CacheEntry(
            key=key.hex(),
            state_kind=state_kind,
            token_count=token_count,
            byte_size=byte_size,
            last_access_ms=last_access_ms,
            compute_savings_ms=compute_savings_ms,
        )

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self._blocks[key] = block
        if key not in self._entries:
            self._entries[key] = CacheEntry(
                key=key.hex(),
                state_kind=f"group:{get_offload_group_idx(key)}",
                token_count=1,
                byte_size=1,
                last_access_ms=self._clock,
            )

    @override
    def remove(self, key: OffloadKey) -> None:
        del self._blocks[key]
        self._entries.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        self._clock += 1.0
        for key in keys:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries[key] = CacheEntry(
                    key=entry.key,
                    state_kind=entry.state_kind,
                    token_count=entry.token_count,
                    byte_size=entry.byte_size,
                    last_access_ms=self._clock,
                    compute_savings_ms=entry.compute_savings_ms,
                )

    @override
    def clear(self) -> None:
        self._blocks.clear()
        self._entries.clear()
        self._clock = 0.0

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        candidates = tuple(
            # The manager asks for an exact number of physical blocks.  Keep
            # Marconi's logical efficiency (savings / bytes) but normalize the
            # capacity unit to one block so a large logical object cannot make
            # the adapter evict more than ``n`` blocks.
            CacheEntry(
                key=entry.key,
                state_kind=entry.state_kind,
                token_count=entry.token_count,
                byte_size=1,
                last_access_ms=entry.last_access_ms,
                estimated_reuse_probability=entry.estimated_reuse_probability,
                compute_savings_ms=entry.compute_savings_ms
                / max(entry.byte_size, 1),
            )
            for key, entry in self._entries.items()
            if key in self._blocks
            and key not in protected
            and self._blocks[key].ref_cnt == 0
        )
        selected = self._policy.select_evictions(
            candidates,
            bytes_needed=n,
            now_ms=self._clock,
        )
        if len(selected) != n:
            return None
        selected_keys = {bytes.fromhex(key) for key in selected}
        result = [(key, self._blocks[key]) for key in self._blocks if key in selected_keys]
        for key, _ in result:
            del self._blocks[key]
            del self._entries[key]
        return result


__all__ = ["MarconiCachePolicy"]
