# SPDX-License-Identifier: Apache-2.0
"""Bridge CacheFlow load tasks to vLLM's native offloading worker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_offload.policies.cacheflow import (
    CacheFlowExecutionCallbacks,
    CacheFlowTask,
)


@dataclass(frozen=True, slots=True)
class CacheFlowLoadBinding:
    """One pre-allocated vLLM transfer for a scheduled CacheFlow chunk."""

    job_id: int
    transfer_spec: Any
    complete: Callable[[], None]


@dataclass(slots=True)
class CacheFlowVLLMDispatcher:
    """Use vLLM's load worker; CacheFlow only chooses the dispatch order."""

    offload_worker: Any
    load_bindings: dict[tuple[str, str, int], CacheFlowLoadBinding]

    @staticmethod
    def _key(task: CacheFlowTask) -> tuple[str, str, int]:
        return task.segment.request_id, task.segment.segment_id, task.chunk_index

    def load(self, task: CacheFlowTask) -> CacheFlowLoadBinding:
        try:
            binding = self.load_bindings[self._key(task)]
        except KeyError as exc:
            raise KeyError("CacheFlow task has no vLLM load binding") from exc
        if not self.offload_worker.transfer_async(
            binding.job_id, binding.transfer_spec
        ):
            raise RuntimeError(f"CacheFlow failed to submit load {binding.job_id}")
        self.offload_worker.wait({binding.job_id})
        binding.complete()
        return binding

    def callbacks(
        self, replay: Callable[[CacheFlowTask], Any]
    ) -> CacheFlowExecutionCallbacks:
        """Return the executor callbacks for this worker and a compute path."""
        return CacheFlowExecutionCallbacks(load=self.load, replay=replay)


__all__ = ["CacheFlowLoadBinding", "CacheFlowVLLMDispatcher"]
