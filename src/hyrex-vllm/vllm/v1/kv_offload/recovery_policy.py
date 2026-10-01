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


class RecoveryPolicy(Protocol):
    """Protocol implemented by one isolated recovery strategy."""

    name: str

    def plan(
        self, batch: RecoveryBatch, telemetry: RecoveryTelemetry
    ) -> Any:
        """Return a backend-specific execution plan."""


PolicyFactory = Callable[[], RecoveryPolicy]
_POLICY_MODULE_PREFIX = "vllm.v1.kv_offload.policies."
_POLICY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_POLICY_FACTORIES: dict[str, PolicyFactory] = {}


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


__all__ = [
    "RecoveryBatch",
    "RecoveryPolicy",
    "RecoveryTelemetry",
    "load_recovery_policy",
    "register_recovery_policy",
    "registered_recovery_policies",
]
