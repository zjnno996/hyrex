# SPDX-License-Identifier: Apache-2.0
"""Recovery-benefit eviction for heterogeneous Hybrid-model state."""

from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class HyRexCachePolicy(CachePolicy):
    """Evict the least recovery benefit, using recency to break ties."""

    def __init__(self, cache_capacity: int):
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.utilities: dict[OffloadKey, float] = {}

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.utilities.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    @override
    def set_utility(self, key: OffloadKey, utility: float) -> None:
        self.utilities[key] = utility

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        candidates = [
            (key, block)
            for key, block in self.blocks.items()
            if block.ref_cnt == 0 and key not in protected
        ]
        if len(candidates) < n:
            return None
        # OrderedDict iteration preserves LRU order for equal utility.
        selected = sorted(
            candidates, key=lambda item: self.utilities.get(item[0], 0.0)
        )[:n]
        for key, _ in selected:
            del self.blocks[key]
            self.utilities.pop(key, None)
        return selected

    @override
    def clear(self) -> None:
        self.blocks.clear()
        self.utilities.clear()
