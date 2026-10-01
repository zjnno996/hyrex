# SPDX-License-Identifier: Apache-2.0
"""Checks for isolated recovery-policy registration."""

import pytest

from vllm.v1.kv_offload.recovery_policy import (
    load_recovery_policy,
    register_recovery_policy,
    registered_recovery_policies,
)


class _UnitPolicy:
    name = "unit_registry"

    def plan(self, batch, telemetry):
        return (batch, telemetry)


def test_policy_registry_loads_registered_factory():
    register_recovery_policy("unit_registry", _UnitPolicy)
    policy = load_recovery_policy("unit_registry")
    assert policy.name == "unit_registry"
    assert "unit_registry" in registered_recovery_policies()


def test_policy_registry_rejects_duplicate_and_unsafe_names():
    with pytest.raises(ValueError):
        register_recovery_policy("unit_registry", _UnitPolicy)
    with pytest.raises(ValueError):
        load_recovery_policy("../escape")
