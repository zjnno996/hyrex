# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class SLRUCachePolicy(CachePolicy):
    """Segmented LRU with probationary and protected segments.

    New blocks enter the probationary segment.  A touched block is promoted
    to the protected segment; when that segment is full, its LRU block is
    demoted back to probationary.  Eviction scans probationary before
    protected, while still skipping in-flight and explicitly protected
    blocks.
    """

    def __init__(self, cache_capacity: int, protected_ratio: float = 0.8):
        if not 0.0 <= protected_ratio <= 1.0:
            raise ValueError("SLRU protected_ratio must be in [0, 1]")
        self.cache_capacity = cache_capacity
        self.protected_capacity = min(
            cache_capacity, int(cache_capacity * protected_ratio)
        )
        self.probationary: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.protected: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.probationary.get(key) or self.protected.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.probationary[key] = block
        self.protected.pop(key, None)

    @override
    def remove(self, key: OffloadKey) -> None:
        if self.probationary.pop(key, None) is None:
            self.protected.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.probationary:
                block = self.probationary.pop(key)
                if self.protected_capacity == 0:
                    self.probationary[key] = block
                    continue
                if len(self.protected) >= self.protected_capacity:
                    demoted_key, demoted_block = self.protected.popitem(last=False)
                    self.probationary[demoted_key] = demoted_block
                self.protected[key] = block
            elif key in self.protected:
                self.protected.move_to_end(key)

    @override
    def clear(self) -> None:
        self.probationary.clear()
        self.protected.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus, bool]] = []
        for segment, is_probationary in (
            (self.probationary, True),
            (self.protected, False),
        ):
            for key, block in segment.items():
                if block.ref_cnt == 0 and key not in protected:
                    candidates.append((key, block, is_probationary))
                    if len(candidates) == n:
                        break
            if len(candidates) == n:
                break

        if len(candidates) < n:
            return None

        result: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block, is_probationary in candidates:
            (self.probationary if is_probationary else self.protected).pop(key)
            result.append((key, block))
        return result
