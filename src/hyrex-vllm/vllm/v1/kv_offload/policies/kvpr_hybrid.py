# SPDX-License-Identifier: Apache-2.0
"""Hybrid-aware KVPR baseline.

The original KVPR split assumes every cached object is token-indexed KV.  A
Hybrid model also has a recurrent state, so this adapter only replays from a
checkpoint-aligned boundary and treats the recurrent terminal state as one
independent object.  Full-attention KV for the complementary tail is loaded
over H2D; replay and that load share the critical path.
"""

from __future__ import annotations

from dataclasses import dataclass
from vllm.v1.kv_offload.recovery_policy import (
    RecoveryBatch,
    RecoveryTelemetry,
    register_recovery_policy,
)


@dataclass(frozen=True, slots=True)
class KVPRHybridSegment:
    """One Hybrid restoration range after a cached recurrent checkpoint."""

    request_id: str
    total_tokens: int
    full_kv_bytes_per_token: int
    recurrent_state_bytes: int
    replay_ms_per_token: float
    checkpoint_tokens: int = 528
    allow_terminal_load: bool = True

    def __post_init__(self) -> None:
        if min(
            self.total_tokens,
            self.full_kv_bytes_per_token,
            self.recurrent_state_bytes,
            self.checkpoint_tokens,
        ) <= 0:
            raise ValueError("KVPR-H sizes must be positive")
        if self.replay_ms_per_token <= 0:
            raise ValueError("KVPR-H replay cost must be positive")


@dataclass(frozen=True, slots=True)
class KVPRHybridPlan:
    request_id: str
    replay_tokens: int
    full_kv_load_tokens: int
    recurrent_action: str
    full_kv_load_ms: float
    recurrent_restore_ms: float
    estimated_ms: float


@dataclass(frozen=True, slots=True)
class KVPRContiguousPlan:
    """Executable vLLM split: restore a Hybrid prefix, replay its suffix."""

    request_id: str
    load_tokens: int
    replay_tokens: int
    h2d_ms: float
    replay_ms: float
    estimated_ms: float


@dataclass(frozen=True, slots=True)
class KVPRHybridPolicy:
    """Profile-based KVPR-H split for heterogeneous Hybrid cache state."""

    name: str = "kvpr_hybrid"
    h2d_gbps: float = 32.0

    def __post_init__(self) -> None:
        if self.h2d_gbps <= 0:
            raise ValueError("KVPR-H H2D bandwidth must be positive")

    @staticmethod
    def _replay_candidates(
        total_tokens: int, checkpoint_tokens: int
    ) -> tuple[int, ...]:
        aligned = range(0, total_tokens + 1, checkpoint_tokens)
        values = list(aligned)
        if not values or values[-1] != total_tokens:
            values.append(total_tokens)
        return tuple(values)

    def _plan_one(
        self, segment: KVPRHybridSegment, h2d_gbps: float
    ) -> KVPRHybridPlan:
        best: KVPRHybridPlan | None = None
        for replay_tokens in self._replay_candidates(
            segment.total_tokens, segment.checkpoint_tokens
        ):
            if replay_tokens == 0 and not segment.allow_terminal_load:
                continue
            load_tokens = segment.total_tokens - replay_tokens
            full_kv_load_ms = (
                load_tokens
                * segment.full_kv_bytes_per_token
                / (h2d_gbps * 1e9)
                * 1e3
            )
            if replay_tokens == 0 and segment.allow_terminal_load:
                # Both objects use the same H2D resource, so they add rather
                # than overlap.
                recurrent_action = "terminal_load"
                recurrent_restore_ms = (
                    segment.recurrent_state_bytes
                    / (h2d_gbps * 1e9)
                    * 1e3
                )
                estimated_ms = full_kv_load_ms + recurrent_restore_ms
            else:
                recurrent_action = "replay"
                recurrent_restore_ms = replay_tokens * segment.replay_ms_per_token
                estimated_ms = max(full_kv_load_ms, recurrent_restore_ms)
            plan = KVPRHybridPlan(
                segment.request_id,
                replay_tokens,
                load_tokens,
                recurrent_action,
                full_kv_load_ms,
                recurrent_restore_ms,
                estimated_ms,
            )
            if best is None or plan.estimated_ms < best.estimated_ms:
                best = plan
        assert best is not None
        return best

    def plan(
        self, batch: RecoveryBatch, telemetry: RecoveryTelemetry
    ) -> tuple[KVPRHybridPlan, ...]:
        bandwidth = telemetry.h2d_gbps or self.h2d_gbps
        if bandwidth <= 0:
            raise ValueError("telemetry h2d_gbps must be positive")
        segments = tuple(batch.segments)
        if not all(isinstance(segment, KVPRHybridSegment) for segment in segments):
            raise TypeError("KVPRHybridPolicy requires KVPRHybridSegment inputs")
        return tuple(self._plan_one(segment, bandwidth) for segment in segments)

    def plan_contiguous_prefix(
        self, segment: KVPRHybridSegment, h2d_gbps: float | None = None
    ) -> KVPRContiguousPlan:
        """Plan the serialized prefix-load path executable by native vLLM."""
        bandwidth = self.h2d_gbps if h2d_gbps is None else h2d_gbps
        if bandwidth <= 0:
            raise ValueError("h2d_gbps must be positive")
        best: KVPRContiguousPlan | None = None
        for replay_tokens in self._replay_candidates(
            segment.total_tokens, segment.checkpoint_tokens
        ):
            load_tokens = segment.total_tokens - replay_tokens
            load_bytes = load_tokens * segment.full_kv_bytes_per_token
            if load_tokens:
                load_bytes += segment.recurrent_state_bytes
            h2d_ms = load_bytes / (bandwidth * 1e9) * 1e3
            replay_ms = replay_tokens * segment.replay_ms_per_token
            plan = KVPRContiguousPlan(
                segment.request_id,
                load_tokens,
                replay_tokens,
                h2d_ms,
                replay_ms,
                h2d_ms + replay_ms,
            )
            if best is None or plan.estimated_ms < best.estimated_ms:
                best = plan
        assert best is not None
        return best


register_recovery_policy("kvpr_hybrid", KVPRHybridPolicy)

__all__ = [
    "KVPRContiguousPlan",
    "KVPRHybridPlan",
    "KVPRHybridPolicy",
    "KVPRHybridSegment",
]
