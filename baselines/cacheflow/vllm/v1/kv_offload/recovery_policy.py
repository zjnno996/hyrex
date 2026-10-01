# SPDX-License-Identifier: Apache-2.0
"""Stable extension point for isolated Hybrid-cache recovery policies.

The core vLLM path owns observation and execution.  A policy plugin only
turns the observed batch into a plan.  Keeping registration here means each
baseline can live on a branch that adds one module under ``policies/`` without
editing a shared registry or the scheduler's dispatch table.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True, slots=True)
class RecoveryBatch:
    """Backend-neutral snapshot handed to a recovery policy."""

    segments: tuple[Any, ...]
    now_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class RecoveryTelemetry:
    """Shared-resource state visible to a policy at one scheduling point."""

    h2d_ready_ms: float = 0.0
    compute_ready_ms: float = 0.0
    d2d_ready_ms: float = 0.0
    h2d_gbps: float | None = None
    concurrency: int = 0


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Backend-neutral metadata exposed to admission/eviction plugins."""

    key: str
    state_kind: str
    token_count: int
    byte_size: int
    last_access_ms: float = 0.0
    estimated_reuse_probability: float = 0.0
    compute_savings_ms: float = 0.0


class CacheAdmissionPolicy(Protocol):
    """Protocol for cache admission and eviction strategies."""

    name: str

    def score(self, entry: CacheEntry, now_ms: float) -> float:
        """Return a higher-is-better retention score."""

    def select_evictions(
        self,
        entries: tuple[CacheEntry, ...],
        bytes_needed: int,
        now_ms: float,
    ) -> tuple[str, ...]:
        """Return keys to evict, or an empty tuple if space is unavailable."""


class RecoveryPolicy(Protocol):
    """Protocol implemented by one isolated recovery strategy."""

    name: str

    def plan(
        self, batch: RecoveryBatch, telemetry: RecoveryTelemetry
    ) -> Any:
        """Return a backend-specific execution plan."""


PolicyFactory = Callable[[], RecoveryPolicy]
CachePolicyFactory = Callable[[], CacheAdmissionPolicy]
_POLICY_MODULE_PREFIX = "vllm.v1.kv_offload.policies."
_POLICY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_POLICY_FACTORIES: dict[str, PolicyFactory] = {}
_CACHE_POLICY_FACTORIES: dict[str, CachePolicyFactory] = {}


def register_recovery_policy(name: str, factory: PolicyFactory) -> None:
    """Register one policy plugin without modifying core dispatch code."""
    if not _POLICY_NAME.fullmatch(name):
        raise ValueError(f"invalid recovery policy name: {name!r}")
    if name in _POLICY_FACTORIES:
        raise ValueError(f"recovery policy already registered: {name!r}")
    _POLICY_FACTORIES[name] = factory


def load_recovery_policy(name: str) -> RecoveryPolicy:
    """Load a policy module lazily and return a fresh policy instance."""
    if not _POLICY_NAME.fullmatch(name):
        raise ValueError(f"invalid recovery policy name: {name!r}")
    if name not in _POLICY_FACTORIES:
        importlib.import_module(_POLICY_MODULE_PREFIX + name)
    try:
        return _POLICY_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"policy module {name!r} did not register itself"
        ) from exc


def registered_recovery_policies() -> tuple[str, ...]:
    """Return registered names in deterministic order for CLI/help output."""
    return tuple(sorted(_POLICY_FACTORIES))


def register_cache_policy(name: str, factory: CachePolicyFactory) -> None:
    """Register an admission/eviction plugin without shared dispatch edits."""
    if not _POLICY_NAME.fullmatch(name):
        raise ValueError(f"invalid cache policy name: {name!r}")
    if name in _CACHE_POLICY_FACTORIES:
        raise ValueError(f"cache policy already registered: {name!r}")
    _CACHE_POLICY_FACTORIES[name] = factory


def load_cache_policy(name: str) -> CacheAdmissionPolicy:
    """Load a cache policy module lazily and return a fresh instance."""
    if not _POLICY_NAME.fullmatch(name):
        raise ValueError(f"invalid cache policy name: {name!r}")
    if name not in _CACHE_POLICY_FACTORIES:
        importlib.import_module(_POLICY_MODULE_PREFIX + name)
    try:
        return _CACHE_POLICY_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"cache policy module {name!r} did not register itself"
        ) from exc


def registered_cache_policies() -> tuple[str, ...]:
    """Return cache policy names in deterministic order."""
    return tuple(sorted(_CACHE_POLICY_FACTORIES))


__all__ = [
    "RecoveryBatch",
    "CacheAdmissionPolicy",
    "CacheEntry",
    "RecoveryPolicy",
    "RecoveryTelemetry",
    "load_recovery_policy",
    "load_cache_policy",
    "register_cache_policy",
    "register_recovery_policy",
    "registered_cache_policies",
    "registered_recovery_policies",
]
