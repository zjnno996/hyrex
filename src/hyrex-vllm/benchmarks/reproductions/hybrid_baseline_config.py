# SPDX-License-Identifier: Apache-2.0
"""Single evaluation contract for HyRex and its Hybrid-serving baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class HybridBaseline:
    name: str
    branch: str
    kind: str
    native_hybrid: bool
    cache_backend: str
    policy_plugin: str | None
    recovery_policy: str | None
    runtime_bound: bool

    def __post_init__(self) -> None:
        if self.cache_backend not in {"none", "lmcache", "native"}:
            raise ValueError(f"unsupported cache backend: {self.cache_backend}")


BASELINES = {
    "full_recompute": HybridBaseline(
        "full_recompute", "hyrec/hyrex", "no_external_cache", True,
        "none", None, "all_replay", True
    ),
    "hybrid_all_load": HybridBaseline(
        "hybrid_all_load", "hyrec/hyrex", "fixed_recovery", True,
        "native", None, "all_load", True
    ),
    "request_adaptive": HybridBaseline(
        "request_adaptive", "hyrec/hyrex", "request_adaptive", True,
        "native", "request_adaptive", "all_load", True
    ),
    "marconi": HybridBaseline(
        "marconi", "hyrec/baseline-marconi", "cache_management", True,
        "lmcache", None, "all_load", True
    ),
    "tail_replay": HybridBaseline(
        "tail_replay",
        "hyrec/baseline-tail-replay",
        "fixed_recovery",
        True,
        "lmcache",
        None,
        "full_load_linear_replay",
        True,
    ),
    "kvpr_hybrid": HybridBaseline(
        "kvpr_hybrid",
        "hyrec/baseline-kvpr-hybrid",
        "hybrid_extension",
        False,
        "native",
        "kvpr_hybrid",
        "all_load",
        True,
    ),
    "cacheflow_hybrid": HybridBaseline(
        "cacheflow_hybrid",
        "hyrec/baseline-cacheflow",
        "hybrid_extension",
        False,
        "native",
        "cacheflow",
        "all_load",
        True,
    ),
    "hyrex": HybridBaseline(
        "hyrex", "hyrec/hyrex", "proposed", True, "native", "hyrex", "all_load",
        True
    ),
}


def baseline_config(name: str) -> dict[str, object]:
    try:
        return asdict(BASELINES[name])
    except KeyError as exc:
        raise ValueError(f"unknown Hybrid baseline: {name}") from exc


__all__ = ["BASELINES", "HybridBaseline", "baseline_config"]
