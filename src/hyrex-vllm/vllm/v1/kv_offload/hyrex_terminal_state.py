# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU storage for Hybrid recurrent terminal states."""

from __future__ import annotations

from collections import OrderedDict

import torch

from vllm.utils.platform_utils import is_pin_memory_available


class HyRexTerminalStateStore:
    """Store the last recurrent state for one prefix and KV-cache group."""

    def __init__(self, max_bytes: int = 0) -> None:
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple[str, int], tuple[torch.Tensor, ...]] = (
            OrderedDict()
        )
        self._bytes = 0
        self._pin_memory = is_pin_memory_available() and torch.cuda.is_available()

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def _evict(self) -> None:
        while self.max_bytes > 0 and self._bytes > self.max_bytes and self._entries:
            _, tensors = self._entries.popitem(last=False)
            self._bytes -= sum(tensor.nbytes for tensor in tensors)

    def store(
        self, prefix_key: str, group_idx: int, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        if group_idx < 0 or not tensors:
            raise ValueError("group_idx must be non-negative and tensors non-empty")
        key = (prefix_key, group_idx)
        old = self._entries.pop(key, ())
        self._bytes -= sum(tensor.nbytes for tensor in old)
        copied = tuple(
            tensor.detach().to(device="cpu").contiguous().pin_memory()
            if self._pin_memory
            else tensor.detach().to(device="cpu").contiguous()
            for tensor in tensors
        )
        self._entries[key] = copied
        self._bytes += sum(tensor.nbytes for tensor in copied)
        self._evict()

    def restore(
        self, prefix_key: str, group_idx: int, device: torch.device
    ) -> tuple[torch.Tensor, ...] | None:
        key = (prefix_key, group_idx)
        tensors = self._entries.get(key)
        if tensors is None:
            return None
        self._entries.move_to_end(key)
        return tuple(
            tensor.to(device=device, non_blocking=tensor.is_pinned())
            for tensor in tensors
        )

    def store_group(
        self,
        prefix_key: str,
        group_idx: int,
        layer_names: tuple[str, ...],
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        block_id: int,
    ) -> None:
        """Snapshot the terminal Mamba/GDN state for one vLLM KV group."""
        if block_id < 0 or not layer_names:
            raise ValueError("terminal state requires a block and layer names")
        tensors: list[torch.Tensor] = []
        for layer_name in layer_names:
            states = kv_caches.get(layer_name)
            if not isinstance(states, list):
                raise TypeError(f"{layer_name} is not a recurrent KV cache")
            tensors.extend(state[block_id] for state in states)
        self.store(prefix_key, group_idx, tuple(tensors))

    def restore_group(
        self,
        prefix_key: str,
        group_idx: int,
        layer_names: tuple[str, ...],
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        block_id: int,
    ) -> bool:
        """Restore a terminal state into the scheduler-allocated KV block."""
        if block_id < 0 or not layer_names:
            raise ValueError("terminal state requires a block and layer names")
        destinations: list[torch.Tensor] = []
        for layer_name in layer_names:
            states = kv_caches.get(layer_name)
            if not isinstance(states, list):
                raise TypeError(f"{layer_name} is not a recurrent KV cache")
            destinations.extend(state[block_id] for state in states)
        if not destinations:
            return False
        restored = self.restore(prefix_key, group_idx, destinations[0].device)
        if restored is None:
            return False
        if len(restored) != len(destinations):
            raise ValueError("terminal state layout mismatch")
        for destination, source in zip(destinations, restored):
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError("terminal state tensor mismatch")
            destination.copy_(source, non_blocking=source.is_pinned())
        return True
