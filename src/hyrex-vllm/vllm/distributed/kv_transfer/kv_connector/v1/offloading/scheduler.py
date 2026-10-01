# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
import json
from math import inf
import time
from typing import Any, NamedTuple

import vllm.envs as envs
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.hyrex_vllm import select_native_policy
from vllm.v1.kv_offload.policies.cacheflow import CacheFlowPolicy, CacheFlowSegment
from vllm.v1.kv_offload.policies.kvpr_hybrid import (
    KVPRHybridPolicy,
    KVPRHybridSegment,
)
from vllm.v1.kv_offload.recovery_policy import RecoveryBatch, RecoveryTelemetry
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


def _hyrex_bucket(value: int) -> int:
    return max(1, value.bit_length() - 1)


def _nearest_profile(
    profiles: dict[tuple[int, int], float], target: tuple[int, int]
) -> float | None:
    if not profiles:
        return None
    key = min(
        profiles,
        key=lambda item: abs(item[0] - target[0]) + abs(item[1] - target[1]),
    )
    return profiles[key]


def native_cache_event(
    request_id: str,
    gpu_hit_tokens: int,
    external_hit_tokens: int,
    policy: str,
    *,
    available_external_tokens: int | None = None,
    recovery_action: str | None = None,
) -> dict[str, str | int]:
    selected_action = recovery_action or {
        "all_load": "load_all",
        "all_replay": "recompute",
        "full_load_linear_replay": "load_full_replay_linear",
        "full_replay_linear_load": "replay_full_load_linear",
    }[policy]
    available_external_tokens = (
        external_hit_tokens
        if available_external_tokens is None
        else available_external_tokens
    )
    cache_state = (
        "partial"
        if gpu_hit_tokens > 0 and available_external_tokens > 0
        else "cpu"
        if available_external_tokens > 0
        else "gpu"
        if gpu_hit_tokens > 0
        else "cold"
    )
    return {
        "request_id": request_id,
        "gpu_hit_tokens": gpu_hit_tokens,
        "cpu_hit_tokens": gpu_hit_tokens + available_external_tokens,
        "h2d_tokens": external_hit_tokens,
        "cache_state": cache_state,
        "recovery_action": selected_action,
    }


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # None below means full attention
    sliding_window_size_in_blocks: int | None
    # Number of this group's offloaded blocks per full-attention alignment
    # segment. Used to skip storing SWA blocks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_block_count: int | None = None


def get_sliding_window_size_in_blocks(
    kv_cache_spec: KVCacheSpec, offloaded_block_size: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, offloaded_block_size)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int
    offload_prompt_only: bool

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the offloaded_block_size of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_offloaded_block_sizes: set[int] = set()
        for idx, gpu_block_size in enumerate(spec.gpu_block_size):
            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_blocks(
                kv_spec, gpu_block_size * spec.block_size_factor
            )
            if sw is None:
                full_attn_offloaded_block_sizes.add(
                    gpu_block_size * spec.block_size_factor
                )

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_offloaded_block_sizes) == 1:
            alignment_tokens = full_attn_offloaded_block_sizes.pop()

        def _alignment_block_count(
            offloaded_block_size: int,
            sliding_window_size_in_blocks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_blocks is None:
                return None
            if alignment_tokens <= offloaded_block_size:
                return None
            per_segment = alignment_tokens // offloaded_block_size
            if sliding_window_size_in_blocks >= per_segment:
                return None
            return per_segment

        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                    sliding_window_size_in_blocks=(
                        sw := get_sliding_window_size_in_blocks(
                            spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            gpu_block_size * spec.block_size_factor,
                        )
                    ),
                    alignment_block_count=_alignment_block_count(
                        gpu_block_size * spec.block_size_factor, sw
                    ),
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
            offload_prompt_only=spec.offload_prompt_only,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    loaded_group_indices: tuple[int, ...]
    store_group_indices: tuple[int, ...]
    lookup_groups: tuple[int, ...]
    hybrid_policy: str
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    hyrex_hit_tokens_by_kind: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


def _create_req_context(req: Request) -> ReqContext:
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=req.kv_transfer_params,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()
        self._group_specs = tuple(
            group.kv_cache_spec for group in spec.kv_cache_config.kv_cache_groups
        )
        self._group_layer_counts = tuple(
            len(group.layer_names) for group in spec.kv_cache_config.kv_cache_groups
        )

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._full_attention_groups: tuple[int, ...] = tuple(full_attention_groups)
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)

        # Native CPU offloading normally transfers every KV group.  The
        # hybrid recovery experiment can select one attention type for P3/P4;
        # the main Scheduler then performs one replay pass for the missing
        # attention type.
        default_loaded_groups = self._get_loaded_group_indices()
        logger.info(
            "Native CPU offload hybrid policy=%s; loaded KV groups=%s/%s",
            envs.VLLM_MOONCAKE_HYBRID_POLICY,
            default_loaded_groups,
            len(self.config.kv_group_configs),
        )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._hyrex_h2d_ready_ms = 0.0
        self._hyrex_compute_ready_ms = 0.0
        self._hyrex_queue_clock_ms = time.monotonic() * 1000
        self._hyrex_step_clock_advanced = False
        self._hyrex_measured_h2d_gbps: float | None = None
        self._hyrex_measured_h2d_gbps_by_bucket: dict[tuple[int, int], float] = {}
        self._hyrex_measured_replay_ms_per_token: dict[str, float] = {}
        self._hyrex_measured_replay_by_bucket: dict[
            str, dict[tuple[int, int], float]
        ] = {}
        # HyRex uses one scheduler step as a lookup barrier: all waiting
        # requests expose their real CPU hits, then actions are chosen together
        # before any request allocates destination GPU blocks.
        self._hyrex_pending_lookups: dict[
            str, tuple[int, tuple[int, ...]]
        ] = {}
        self._hyrex_preplanned: dict[str, tuple[str, int]] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

    def _get_hybrid_policy(self, params: dict[str, Any] | None = None) -> str:
        """Return a validated request-local policy, or the legacy default."""
        policy = (params or {}).get("hyrex_recovery_policy")
        if policy in {
            "all_load",
            "all_replay",
            "full_load_linear_replay",
            "full_replay_linear_load",
        }:
            return policy
        if policy is not None:
            logger.warning("Ignoring invalid hyrex_recovery_policy=%r", policy)
        return envs.VLLM_MOONCAKE_HYBRID_POLICY

    def _get_loaded_group_indices(
        self, params: dict[str, Any] | None = None
    ) -> tuple[int, ...]:
        """Return KV groups materialized from CPU for one request."""
        policy = self._get_hybrid_policy(params)
        if policy == "all_replay":
            return ()
        if policy not in {
            "full_load_linear_replay",
            "full_replay_linear_load",
        }:
            return tuple(range(len(self.config.kv_group_configs)))

        # Qwen3.5 GDN groups use MambaSpec.  Keep this type check explicit so
        # a future hybrid model does not silently classify an unknown group as
        # Linear state.
        if policy == "full_load_linear_replay":
            return tuple(
                group_idx
                for group_idx in range(len(self.config.kv_group_configs))
                if isinstance(self._group_specs[group_idx], FullAttentionSpec)
            )
        return tuple(
            group_idx
            for group_idx in range(len(self.config.kv_group_configs))
            if isinstance(self._group_specs[group_idx], MambaSpec)
        )

    def _get_lookup_groups(
        self, loaded_group_indices: tuple[int, ...]
    ) -> tuple[int, ...]:
        selected = set(loaded_group_indices)
        return tuple(
            group_idx
            for group_idx in (
                *self._full_attention_groups,
                *self._sliding_window_groups,
            )
            if group_idx in selected
        )

    def _get_store_group_indices(self, request: Request) -> tuple[int, ...]:
        """Admit state whose expected reload is cheaper than replay."""
        params = request.kv_transfer_params or {}
        all_groups = tuple(range(len(self.config.kv_group_configs)))
        if (
            params.get("hybrid_baseline") != "hyrex"
            or not params.get("hyrex_state_aware_store", True)
        ):
            return all_groups
        required = (
            "hyrex_h2d_gbps",
            "hyrex_full_replay_ms_per_token",
            "hyrex_recurrent_replay_ms_per_token",
        )
        if any(name not in params for name in required):
            return all_groups

        tokens = max(1, request.num_tokens)
        concurrency = max(1, int(params.get("hyrex_concurrency", 1)))
        concurrency_bucket = _hyrex_bucket(concurrency)
        h2d_profiles = getattr(self, "_hyrex_measured_h2d_gbps_by_bucket", {})
        fallback_h2d_gbps = getattr(
            self, "_hyrex_measured_h2d_gbps", None
        ) or float(params["hyrex_h2d_gbps"])
        replay_profiles = getattr(self, "_hyrex_measured_replay_by_bucket", {})
        measured_replay = getattr(
            self, "_hyrex_measured_replay_ms_per_token", {}
        )
        bytes_by_kind = {"full": 0, "recurrent": 0}
        for group_config, spec, layer_count in zip(
            self.config.kv_group_configs,
            self._group_specs,
            self._group_layer_counts,
        ):
            num_pages = cdiv(tokens, group_config.gpu_block_size)
            num_bytes = num_pages * spec.page_size_bytes * layer_count
            if isinstance(spec, FullAttentionSpec):
                kind = "full"
            elif isinstance(spec, MambaSpec):
                kind = "recurrent"
            else:
                return all_groups
            bytes_by_kind[kind] += num_bytes
        profitable = {}
        utility_by_kind: dict[str, float] = {}
        token_bucket = _hyrex_bucket(tokens)
        for kind, param_name in (
            ("full", "hyrex_full_replay_ms_per_token"),
            ("recurrent", "hyrex_recurrent_replay_ms_per_token"),
        ):
            h2d_gbps = _nearest_profile(
                h2d_profiles,
                (_hyrex_bucket(max(1, bytes_by_kind[kind])), concurrency_bucket),
            ) or fallback_h2d_gbps
            replay_rate = _nearest_profile(
                replay_profiles.get(kind, {}),
                (token_bucket, concurrency_bucket),
            ) or measured_replay.get(kind, float(params[param_name]))
            profitable[kind] = (
                bytes_by_kind[kind] / (h2d_gbps * 1e9) * 1e3
                <= tokens * replay_rate
            )
            load_ms = bytes_by_kind[kind] / (h2d_gbps * 1e9) * 1e3
            replay_ms = tokens * replay_rate
            utility_by_kind[kind] = max(0.0, replay_ms - load_ms) / max(
                1, bytes_by_kind[kind]
            )
        params["hyrex_eviction_utility_by_group"] = {
            index: utility_by_kind[
                "full" if isinstance(spec, FullAttentionSpec) else "recurrent"
            ]
            for index, spec in enumerate(self._group_specs)
            if isinstance(spec, (FullAttentionSpec, MambaSpec))
        }
        return tuple(
            index
            for index, spec in enumerate(self._group_specs)
            if (
                isinstance(spec, FullAttentionSpec)
                and profitable["full"]
            )
            or (isinstance(spec, MambaSpec) and profitable["recurrent"])
        )

    def _bind_hyrex_policy(
        self,
        req_status: RequestOffloadState,
        num_hit_tokens: int,
        available_group_indices: tuple[int, ...] | None = None,
    ) -> int:
        """Select a request policy after lookup and before GPU allocation."""
        params = req_status.req.kv_transfer_params or {}
        baseline = params.get("hybrid_baseline")
        if baseline not in {"hyrex", "request_adaptive"} or num_hit_tokens <= 0:
            return num_hit_tokens
        required = (
            "hyrex_h2d_gbps",
            "hyrex_full_replay_ms_per_token",
            "hyrex_recurrent_replay_ms_per_token",
        )
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(f"HyRex runtime calibration is missing: {missing}")

        if not getattr(self, "_hyrex_step_clock_advanced", False):
            self._decay_hyrex_queue_debt()
            self._hyrex_step_clock_advanced = True

        hits = getattr(req_status, "hyrex_hit_tokens_by_kind", {})
        full_hit_tokens = hits.get("full", num_hit_tokens)
        recurrent_hit_tokens = hits.get("recurrent", num_hit_tokens)

        def load_bytes(kind: str, tokens: int) -> int:
            total = 0
            for group_config, spec, layer_count in zip(
                self.config.kv_group_configs,
                self._group_specs,
                self._group_layer_counts,
            ):
                matches = (
                    isinstance(spec, FullAttentionSpec)
                    if kind == "full"
                    else isinstance(spec, MambaSpec)
                )
                if matches:
                    total += (
                        cdiv(tokens, group_config.gpu_block_size)
                        * spec.page_size_bytes
                        * layer_count
                    )
                elif not isinstance(spec, (FullAttentionSpec, MambaSpec)):
                    raise ValueError(
                        f"unsupported HyRex KV group: {type(spec).__name__}"
                    )
            return total

        available = set(
            range(len(self.config.kv_group_configs))
            if available_group_indices is None
            else available_group_indices
        )
        full_source_ready = any(
            index in available and isinstance(spec, FullAttentionSpec)
            for index, spec in enumerate(self._group_specs)
        )
        recurrent_source_ready = any(
            index in available and isinstance(spec, MambaSpec)
            for index, spec in enumerate(self._group_specs)
        )

        configured_h2d_gbps = float(params["hyrex_h2d_gbps"])
        concurrency = max(1, int(params.get("hyrex_concurrency", 1)))
        profile_tokens = max(full_hit_tokens, recurrent_hit_tokens, num_hit_tokens)
        full_bytes = load_bytes("full", max(1, full_hit_tokens))
        recurrent_bytes = load_bytes("recurrent", max(1, recurrent_hit_tokens))
        measured_h2d_gbps = _nearest_profile(
            getattr(self, "_hyrex_measured_h2d_gbps_by_bucket", {}),
            (
                _hyrex_bucket(max(1, full_bytes + recurrent_bytes)),
                _hyrex_bucket(concurrency),
            ),
        ) or getattr(self, "_hyrex_measured_h2d_gbps", None)
        h2d_gbps = measured_h2d_gbps or configured_h2d_gbps
        measured_replay = getattr(
            self, "_hyrex_measured_replay_ms_per_token", {}
        )
        replay_profiles = getattr(self, "_hyrex_measured_replay_by_bucket", {})
        replay_target = (_hyrex_bucket(profile_tokens), _hyrex_bucket(concurrency))
        full_replay_rate = _nearest_profile(
            replay_profiles.get("full", {}), replay_target
        ) or measured_replay.get(
            "full", float(params["hyrex_full_replay_ms_per_token"])
        )
        recurrent_replay_rate = _nearest_profile(
            replay_profiles.get("recurrent", {}), replay_target
        ) or measured_replay.get(
            "recurrent", float(params["hyrex_recurrent_replay_ms_per_token"])
        )
        full_replay_ms = full_replay_rate * profile_tokens
        recurrent_replay_ms = recurrent_replay_rate * profile_tokens
        h2d_ready_ms = max(
            getattr(self, "_hyrex_h2d_ready_ms", 0.0),
            float(params.get("hyrex_h2d_ready_ms", 0.0)),
        )
        compute_ready_ms = max(
            getattr(self, "_hyrex_compute_ready_ms", 0.0),
            float(params.get("hyrex_compute_ready_ms", 0.0)),
        )
        h2d_queue_before_ms = h2d_ready_ms
        compute_queue_before_ms = compute_ready_ms
        candidates: dict[str, tuple[float, int]] = {
            "all_replay": (
                compute_ready_ms
                + (full_replay_rate + recurrent_replay_rate) * profile_tokens,
                0,
            )
        }
        if full_source_ready and recurrent_source_ready:
            common = min(full_hit_tokens, recurrent_hit_tokens)
            common_full_bytes = load_bytes("full", common)
            common_recurrent_bytes = load_bytes("recurrent", common)
            candidates["all_load"] = (
                h2d_ready_ms
                + (common_full_bytes + common_recurrent_bytes)
                / (h2d_gbps * 1e9)
                * 1e3,
                common,
            )
        enable_mixed = bool(params.get("hyrex_enable_mixed_recovery", True))
        if full_source_ready and enable_mixed:
            candidates["full_load_linear_replay"] = (
                max(
                    h2d_ready_ms + full_bytes / (h2d_gbps * 1e9) * 1e3,
                    compute_ready_ms + recurrent_replay_rate * full_hit_tokens,
                ),
                full_hit_tokens,
            )
        if recurrent_source_ready and enable_mixed:
            candidates["full_replay_linear_load"] = (
                max(
                    h2d_ready_ms + recurrent_bytes / (h2d_gbps * 1e9) * 1e3,
                    compute_ready_ms + full_replay_rate * recurrent_hit_tokens,
                ),
                recurrent_hit_tokens,
            )
        if baseline == "request_adaptive":
            candidates = {
                name: value
                for name, value in candidates.items()
                if name in {"all_load", "all_replay"}
            }
        policy, (_, selected_hit_tokens) = min(
            candidates.items(), key=lambda item: item[1][0]
        )
        full_bytes = load_bytes("full", selected_hit_tokens)
        recurrent_bytes = load_bytes("recurrent", selected_hit_tokens)
        replay_tokens = (
            profile_tokens if policy == "all_replay" else selected_hit_tokens
        )
        full_replay_ms = full_replay_rate * replay_tokens
        recurrent_replay_ms = recurrent_replay_rate * replay_tokens
        full_load_ms = full_bytes / (h2d_gbps * 1e9) * 1e3
        recurrent_load_ms = recurrent_bytes / (h2d_gbps * 1e9) * 1e3
        if policy == "all_replay":
            compute_ready_ms += full_replay_ms + recurrent_replay_ms
        elif policy == "all_load":
            h2d_ready_ms += full_load_ms + recurrent_load_ms
        elif policy == "full_load_linear_replay":
            h2d_ready_ms += full_load_ms
            compute_ready_ms += recurrent_replay_ms
        else:
            h2d_ready_ms += recurrent_load_ms
            compute_ready_ms += full_replay_ms
        self._hyrex_h2d_ready_ms = h2d_ready_ms
        self._hyrex_compute_ready_ms = compute_ready_ms
        params["hyrex_recovery_policy"] = policy
        req_status.hybrid_policy = policy
        req_status.loaded_group_indices = self._get_loaded_group_indices(params)
        req_status.lookup_groups = self._get_lookup_groups(
            req_status.loaded_group_indices
        )
        if params.get("hyrex_state_aware_store", True):
            # Admission describes the value of a future recovery, independent
            # of the action selected for this request.  In particular, an
            # all-replay decision at token zero must not disable all stores.
            admission_tokens = max(1, profile_tokens)
            admission_full_load_ms = (
                load_bytes("full", admission_tokens) / (h2d_gbps * 1e9) * 1e3
            )
            admission_recurrent_load_ms = (
                load_bytes("recurrent", admission_tokens)
                / (h2d_gbps * 1e9)
                * 1e3
            )
            store_full = admission_full_load_ms <= (
                full_replay_rate * admission_tokens
            )
            store_recurrent = admission_recurrent_load_ms <= (
                recurrent_replay_rate * admission_tokens
            )
            req_status.store_group_indices = tuple(
                index
                for index, spec in enumerate(self._group_specs)
                if (isinstance(spec, FullAttentionSpec) and store_full)
                or (isinstance(spec, MambaSpec) and store_recurrent)
            )
        if params.get("hyrex_observe_cache"):
            logger.info(
                "HYREX_NATIVE_DECISION %s",
                json.dumps(
                    {
                        "request_id": req_status.req.request_id,
                        "missing_tokens": num_hit_tokens,
                        "selected_hit_tokens": selected_hit_tokens,
                        "full_hit_tokens": full_hit_tokens,
                        "recurrent_hit_tokens": recurrent_hit_tokens,
                        "full_load_bytes": full_bytes,
                        "recurrent_load_bytes": recurrent_bytes,
                        "full_replay_ms": full_replay_ms,
                        "recurrent_replay_ms": recurrent_replay_ms,
                        "h2d_queue_before_ms": h2d_queue_before_ms,
                        "compute_queue_before_ms": compute_queue_before_ms,
                        "h2d_ready_after_ms": h2d_ready_ms,
                        "compute_ready_after_ms": compute_ready_ms,
                        "h2d_gbps": h2d_gbps,
                        "concurrency": concurrency,
                        "h2d_gbps_source": (
                            "measured"
                            if measured_h2d_gbps is not None
                            else "configured"
                        ),
                        "full_replay_ms_per_token": full_replay_rate,
                        "recurrent_replay_ms_per_token": recurrent_replay_rate,
                        "replay_cost_source": (
                            "measured" if measured_replay else "configured"
                        ),
                        "policy": policy,
                        "available_group_indices": sorted(available),
                        "store_group_indices": req_status.store_group_indices,
                    },
                    sort_keys=True,
                ),
            )
        return selected_hit_tokens

    def _finalize_hyrex_batch(self) -> None:
        pending = getattr(self, "_hyrex_pending_lookups", {})
        if not pending:
            return

        def order(item: tuple[str, tuple[int, tuple[int, ...]]]):
            request_id, _ = item
            request = self._req_status[request_id].req
            params = request.kv_transfer_params or {}
            deadline = params.get("hyrex_deadline_ms")
            return (
                -float(params.get("hyrex_priority", -getattr(request, "priority", 0))),
                inf if deadline is None else float(deadline),
                float(params.get("hyrex_arrival_ms", request.arrival_time * 1000)),
                request_id,
            )

        concurrency = len(pending)
        preplanned = getattr(self, "_hyrex_preplanned", {})
        for request_id, (hit_tokens, available) in sorted(
            pending.items(), key=order
        ):
            status = self._req_status[request_id]
            params = status.req.kv_transfer_params or {}
            params["hyrex_concurrency"] = concurrency
            selected = self._bind_hyrex_policy(status, hit_tokens, available)
            preplanned[request_id] = (status.hybrid_policy, selected)
        self._hyrex_preplanned = preplanned
        pending.clear()

    def _apply_preplanned_hyrex(
        self,
        req_status: RequestOffloadState,
        num_hit_tokens: int,
        available: tuple[int, ...],
    ) -> int:
        planned = getattr(self, "_hyrex_preplanned", {}).pop(
            req_status.req.request_id, None
        )
        if planned is None:
            return self._bind_hyrex_policy(req_status, num_hit_tokens, available)
        policy, selected = planned
        hits = req_status.hyrex_hit_tokens_by_kind
        required = {
            "all_load": min(hits.get("full", 0), hits.get("recurrent", 0)),
            "full_load_linear_replay": hits.get("full", 0),
            "full_replay_linear_load": hits.get("recurrent", 0),
            "all_replay": 0,
        }[policy]
        # A source may have been evicted during the barrier. Re-plan instead
        # of trusting stale metadata.
        if required < selected:
            return self._bind_hyrex_policy(req_status, num_hit_tokens, available)
        params = req_status.req.kv_transfer_params or {}
        params["hyrex_recovery_policy"] = policy
        req_status.hybrid_policy = policy
        req_status.loaded_group_indices = self._get_loaded_group_indices(params)
        req_status.lookup_groups = self._get_lookup_groups(
            req_status.loaded_group_indices
        )
        return selected

    def _bind_kvpr_hybrid(
        self, req_status: RequestOffloadState, available_tokens: int
    ) -> int:
        """Cap native restoration at KVPR-H's checkpoint-aligned split.

        vLLM restores the selected Hybrid prefix and recomputes the suffix.
        H2D/recompute overlap remains outside this conservative binding.
        """
        params = req_status.req.kv_transfer_params or {}
        if params.get("hybrid_baseline") != "kvpr_hybrid" or available_tokens <= 0:
            return available_tokens
        required = ("kvpr_h2d_gbps", "kvpr_replay_ms_per_token")
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(f"KVPR-H runtime calibration is missing: {missing}")

        full_bytes = recurrent_bytes = 0
        checkpoint_tokens = min(
            config.offloaded_block_size for config in self.config.kv_group_configs
        )
        for config, spec, layer_count in zip(
            self.config.kv_group_configs,
            self._group_specs,
            self._group_layer_counts,
        ):
            bytes_per_page = spec.page_size_bytes * layer_count
            if isinstance(spec, FullAttentionSpec):
                full_bytes += bytes_per_page
            elif isinstance(spec, MambaSpec):
                recurrent_bytes += bytes_per_page
            else:
                raise ValueError(f"unsupported KVPR-H group: {type(spec).__name__}")
        if full_bytes <= 0 or recurrent_bytes <= 0:
            raise ValueError("KVPR-H requires Full-Attention and recurrent groups")

        policy = KVPRHybridPolicy(h2d_gbps=float(params["kvpr_h2d_gbps"]))
        plan = policy.plan_contiguous_prefix(KVPRHybridSegment(
            req_status.req.request_id,
            available_tokens,
            max(1, full_bytes // checkpoint_tokens),
            recurrent_bytes,
            float(params["kvpr_replay_ms_per_token"]),
            checkpoint_tokens,
        ))
        selected = plan.load_tokens
        params["kvpr_available_tokens"] = available_tokens
        params["kvpr_load_tokens"] = selected
        params["kvpr_replay_tokens"] = plan.replay_tokens
        if params.get("hyrex_observe_cache"):
            logger.info(
                "KVPR_H_NATIVE_DECISION %s",
                json.dumps(
                    {
                        "request_id": req_status.req.request_id,
                        "available_tokens": available_tokens,
                        "load_tokens": selected,
                        "replay_tokens": plan.replay_tokens,
                        "estimated_ms": plan.estimated_ms,
                        "policy": "kvpr_split",
                        "h2d_gbps": float(params["kvpr_h2d_gbps"]),
                        "h2d_gbps_source": "calibration",
                        "replay_cost_source": "calibration",
                        "modeled_load_bytes": (
                            (full_bytes + recurrent_bytes)
                            * selected / available_tokens
                        ),
                        "modeled_replay_ms": (
                            plan.replay_tokens
                            * float(params["kvpr_replay_ms_per_token"])
                        ),
                        "overlap_bound": False,
                    },
                    sort_keys=True,
                ),
            )
        return selected

    def _bind_cacheflow_hybrid(
        self, req_status: RequestOffloadState, available_tokens: int
    ) -> int:
        """Bind CacheFlow's chunk choices to a native contiguous prefix."""
        params = req_status.req.kv_transfer_params or {}
        if (
            params.get("hybrid_baseline") != "cacheflow_hybrid"
            or available_tokens <= 0
        ):
            return available_tokens
        required = ("cacheflow_h2d_gbps", "cacheflow_replay_ms_per_token")
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(f"CacheFlow-H runtime calibration is missing: {missing}")

        chunk_tokens = min(
            config.offloaded_block_size for config in self.config.kv_group_configs
        )
        if available_tokens % chunk_tokens:
            raise ValueError("CacheFlow-H CPU hit must be checkpoint aligned")
        chunks = available_tokens // chunk_tokens
        load_bytes = 0
        for config, spec, layer_count in zip(
            self.config.kv_group_configs,
            self._group_specs,
            self._group_layer_counts,
        ):
            pages = cdiv(available_tokens, config.gpu_block_size)
            load_bytes += pages * spec.page_size_bytes * layer_count
        replay_ms = (
            available_tokens * float(params["cacheflow_replay_ms_per_token"])
        )
        plan = CacheFlowPolicy(
            h2d_gbps=float(params["cacheflow_h2d_gbps"])
        ).plan(
            RecoveryBatch((CacheFlowSegment(
                req_status.req.request_id,
                "hybrid_prefix",
                load_bytes,
                replay_ms,
                chunks,
            ),)),
            RecoveryTelemetry(h2d_gbps=float(params["cacheflow_h2d_gbps"])),
        )
        load_chunks = sum(task.action == "load" for task in plan.tasks)
        selected = load_chunks * chunk_tokens
        params["cacheflow_available_tokens"] = available_tokens
        params["cacheflow_load_tokens"] = selected
        params["cacheflow_replay_tokens"] = available_tokens - selected
        params["cacheflow_replay_ms"] = replay_ms
        if params.get("hyrex_observe_cache"):
            logger.info(
                "CACHEFLOW_H_NATIVE_DECISION %s",
                json.dumps(
                    {
                        "request_id": req_status.req.request_id,
                        "available_tokens": available_tokens,
                        "load_tokens": selected,
                        "replay_tokens": available_tokens - selected,
                        "load_chunks": load_chunks,
                        "chunks": chunks,
                        "policy": "cacheflow_split",
                        "h2d_gbps": float(params["cacheflow_h2d_gbps"]),
                        "h2d_gbps_source": "calibration",
                        "replay_cost_source": "calibration",
                        "modeled_load_bytes": load_bytes * load_chunks / chunks,
                        "modeled_replay_ms": (
                            (available_tokens - selected)
                            * float(params["cacheflow_replay_ms_per_token"])
                        ),
                        "contiguous_prefix_adapter": True,
                        "overlap_bound": False,
                    },
                    sort_keys=True,
                ),
            )
        return selected

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _observe_hyrex_h2d(
        self, num_bytes: int, service_ms: float, concurrency: int = 1
    ) -> None:
        if num_bytes <= 0 or service_ms <= 0:
            return
        concurrency = max(1, concurrency)
        measured = num_bytes / (service_ms / 1e3) / 1e9
        previous = getattr(self, "_hyrex_measured_h2d_gbps", None)
        self._hyrex_measured_h2d_gbps = (
            measured if previous is None else 0.2 * measured + 0.8 * previous
        )
        profiles = getattr(self, "_hyrex_measured_h2d_gbps_by_bucket", {})
        key = (_hyrex_bucket(num_bytes), _hyrex_bucket(concurrency))
        previous = profiles.get(key)
        profiles[key] = (
            measured if previous is None else 0.2 * measured + 0.8 * previous
        )
        self._hyrex_measured_h2d_gbps_by_bucket = profiles

    def _decay_hyrex_queue_debt(self, now_ms: float | None = None) -> None:
        """Carry predicted recovery contention across scheduler steps."""
        now_ms = time.monotonic() * 1000 if now_ms is None else now_ms
        previous_ms = getattr(self, "_hyrex_queue_clock_ms", None)
        self._hyrex_queue_clock_ms = now_ms
        if previous_ms is None:
            return
        elapsed_ms = max(0.0, now_ms - previous_ms)
        self._hyrex_h2d_ready_ms = max(
            0.0, getattr(self, "_hyrex_h2d_ready_ms", 0.0) - elapsed_ms
        )
        self._hyrex_compute_ready_ms = max(
            0.0, getattr(self, "_hyrex_compute_ready_ms", 0.0) - elapsed_ms
        )

    def observe_hyrex_replay(
        self, observations: dict[str, tuple[str, int, float]]
    ) -> None:
        rates = getattr(self, "_hyrex_measured_replay_ms_per_token", {})
        profiles = getattr(self, "_hyrex_measured_replay_by_bucket", {})
        concurrency = max(1, len(observations))
        for policy, tokens, elapsed_ms in observations.values():
            if tokens <= 0 or elapsed_ms < 0:
                continue
            state_kind = (
                "recurrent"
                if policy == "full_load_linear_replay"
                else "full"
                if policy == "full_replay_linear_load"
                else None
            )
            if state_kind is None:
                continue
            measured = elapsed_ms / tokens
            previous = rates.get(state_kind)
            rates[state_kind] = (
                measured if previous is None else 0.2 * measured + 0.8 * previous
            )
            state_profiles = profiles.setdefault(state_kind, {})
            key = (_hyrex_bucket(tokens), _hyrex_bucket(concurrency))
            previous = state_profiles.get(key)
            state_profiles[key] = (
                measured if previous is None else 0.2 * measured + 0.8 * previous
            )
        self._hyrex_measured_replay_ms_per_token = rates
        self._hyrex_measured_replay_by_bucket = profiles

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            result = self.manager.lookup(key, req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a miss is detected)
                result = True
            if not result:
                break
            hit_count += 1
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            result = self.manager.lookup(keys[idx], req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a hit is detected)
                result = False
            if not result:
                consecutive_hits = 0
            else:
                consecutive_hits += 1
                if consecutive_hits == sliding_window_size:
                    return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_idx, (group_config, group_state) in enumerate(
            zip(self.config.kv_group_configs, req_status.group_states)
        ):
            if group_idx not in req_status.loaded_group_indices:
                continue
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups and not (
            envs.VLLM_MOONCAKE_HYBRID_SUFFIX_ONLY
            and req_status.hybrid_policy
            in {"full_load_linear_replay", "full_replay_linear_load"}
        ):
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = req_status.lookup_groups
        suffix_only = envs.VLLM_MOONCAKE_HYBRID_SUFFIX_ONLY
        mixed_policy = req_status.hybrid_policy in {
            "full_load_linear_replay",
            "full_replay_linear_load",
        }
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys)
                    >= req_status.req.num_tokens // offloaded_block_size
                )

                # Constrain to block-aligned boundary for this group
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                num_blocks = min(
                    cdiv(max_hit_size_tokens, offloaded_block_size), len(offload_keys)
                )
                # The selected group is not represented by the local GPU
                # prefix in suffix-only mode.  Its CPU object therefore has
                # to be looked up from token zero (Full KV needs the prefix
                # keys/values to attend to the suffix; a Linear state is a
                # whole-prefix recurrent checkpoint).  The returned hit
                # count remains relative to ``num_computed_tokens`` so the
                # main scheduler still sees only newly materialized tokens.
                load_from_zero = suffix_only and mixed_policy
                lookup_start_tokens = 0 if load_from_zero else num_computed_tokens
                start_block_idx = lookup_start_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_blocks: int | None
                if sliding_window_size_in_blocks is None:
                    num_hit_blocks = self._maximal_prefix_lookup(
                        offload_keys, req_status.req_context
                    )
                else:
                    num_hit_blocks = self._sliding_window_lookup(
                        offload_keys,
                        sliding_window_size_in_blocks,
                        req_status.req_context,
                    )
                if num_hit_blocks == 0:
                    return 0

                if num_hit_blocks is None:
                    defer_lookup = True
                else:
                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        offloaded_block_size * (start_block_idx + num_hit_blocks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = req_status.lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._blocks_being_loaded:
            for group_idx, (group_config, group_state) in enumerate(
                zip(self.config.kv_group_configs, req_status.group_states)
            ):
                if group_idx not in req_status.loaded_group_indices:
                    continue
                offloaded_block_size = group_config.offloaded_block_size
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                load_from_zero = suffix_only and mixed_policy
                lookup_start_tokens = 0 if load_from_zero else num_computed_tokens
                start_block_idx = lookup_start_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def _lookup_hyrex_groups(
        self, req_status: RequestOffloadState
    ) -> tuple[int | None, tuple[int, ...]]:
        """Find independently recoverable Full and recurrent state groups."""
        saved_lookup = req_status.lookup_groups
        saved_loaded = req_status.loaded_group_indices
        hits: list[tuple[tuple[int, ...], int]] = []
        try:
            for spec_type in (FullAttentionSpec, MambaSpec):
                groups = tuple(
                    index
                    for index, spec in enumerate(self._group_specs)
                    if isinstance(spec, spec_type)
                )
                if not groups:
                    continue
                req_status.lookup_groups = self._get_lookup_groups(groups)
                req_status.loaded_group_indices = groups
                hit = self._lookup(req_status)
                if hit is None:
                    return None, ()
                if hit > 0:
                    hits.append((groups, hit))
        finally:
            req_status.lookup_groups = saved_lookup
            req_status.loaded_group_indices = saved_loaded

        if not hits:
            req_status.hyrex_hit_tokens_by_kind = {}
            return 0, ()
        req_status.hyrex_hit_tokens_by_kind = {
            (
                "full"
                if isinstance(self._group_specs[groups[0]], FullAttentionSpec)
                else "recurrent"
            ): hit
            for groups, hit in hits
        }
        available = tuple(index for groups, _ in hits for index in groups)
        # Both state types must agree on a common boundary for all-load. A
        # single available type can use its longer independent mixed hit.
        hit_tokens = min(hit for _, hit in hits) if len(hits) > 1 else hits[0][1]
        return hit_tokens, available

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        loaded_group_indices = self._get_loaded_group_indices(
            request.kv_transfer_params
        )
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
            loaded_group_indices=loaded_group_indices,
            store_group_indices=self._get_store_group_indices(request),
            lookup_groups=self._get_lookup_groups(loaded_group_indices),
            hybrid_policy=self._get_hybrid_policy(request.kv_transfer_params),
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        params = request.kv_transfer_params or {}
        available_group_indices: tuple[int, ...] | None = None
        if params.get("hybrid_baseline") == "hyrex":
            num_hit_tokens, available_group_indices = self._lookup_hyrex_groups(
                req_status
            )
        else:
            num_hit_tokens = self._lookup(req_status)
        available_tokens = num_hit_tokens
        if (
            params.get("hybrid_baseline") == "hyrex"
            and req_status.hyrex_hit_tokens_by_kind
        ):
            available_tokens = max(req_status.hyrex_hit_tokens_by_kind.values())
        if num_hit_tokens is not None:
            if (
                params.get("hybrid_baseline") == "hyrex"
                and params.get("hyrex_batch_wide", True)
                and request.request_id
                not in getattr(self, "_hyrex_preplanned", {})
                and num_hit_tokens > 0
            ):
                pending = getattr(self, "_hyrex_pending_lookups", {})
                pending[request.request_id] = (
                    num_hit_tokens,
                    available_group_indices or (),
                )
                self._hyrex_pending_lookups = pending
                return None, False
            if params.get("hybrid_baseline") == "hyrex":
                num_hit_tokens = self._apply_preplanned_hyrex(
                    req_status,
                    num_hit_tokens,
                    available_group_indices or (),
                )
            else:
                num_hit_tokens = self._bind_hyrex_policy(
                    req_status, num_hit_tokens, available_group_indices
                )
            num_hit_tokens = self._bind_kvpr_hybrid(req_status, num_hit_tokens)
            num_hit_tokens = self._bind_cacheflow_hybrid(req_status, num_hit_tokens)
        if num_hit_tokens is not None and bool(
            (request.kv_transfer_params or {}).get("hyrex_observe_cache")
        ):
            logger.info(
                "HYREX_NATIVE_CACHE_EVENT %s",
                json.dumps(
                    native_cache_event(
                        request.request_id,
                        num_computed_tokens,
                        num_hit_tokens,
                        req_status.hybrid_policy,
                        available_external_tokens=available_tokens,
                        recovery_action=(
                            "kvpr_split"
                            if (request.kv_transfer_params or {}).get(
                                "hybrid_baseline"
                            ) == "kvpr_hybrid"
                            else "cacheflow_split"
                            if (request.kv_transfer_params or {}).get(
                                "hybrid_baseline"
                            ) == "cacheflow_hybrid"
                            else None
                        ),
                    ),
                    sort_keys=True,
                ),
            )
        req_status.update_num_hit_blocks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_idx, (group_config, group_state, group_blocks) in enumerate(
            zip(self.config.kv_group_configs, req_status.group_states, blocks.blocks)
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            if group_idx not in req_status.loaded_group_indices:
                # Keep the group position in GPULoadStoreSpec, but do not
                # allocate/load any CPU block for a replay-only group.
                group_sizes.append(0)
                block_indices.append(0)
                continue

            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            if num_locally_computed_tokens > (
                num_locally_computed_gpu_blocks * gpu_block_size
            ):
                # Suffix-only hybrid recovery deliberately uses the replay
                # group's GPU prefix as the scheduler boundary while the
                # selected group is restored from CPU starting at block zero
                # (Full KV or a single Linear/Mamba state).  That selected
                # group has no local prefix blocks by design; its transfer is
                # still valid and must not be rejected by the ordinary
                # all-groups-local invariant.
                suffix_only = envs.VLLM_MOONCAKE_HYBRID_SUFFIX_ONLY
                mixed_policy = req_status.hybrid_policy in {
                    "full_load_linear_replay",
                    "full_replay_linear_load",
                }
                if not (suffix_only and mixed_policy):
                    raise AssertionError(
                        "locally computed tokens exceed local GPU blocks"
                    )
                num_locally_computed_gpu_blocks = 0
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_blocks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_blocks
                    * self.config.block_size_factor
                )

            num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
            assert len(offload_keys) >= num_blocks
            if num_pending_gpu_blocks:
                start_block_idx = (
                    num_locally_computed_gpu_blocks // self.config.block_size_factor
                )
                keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_block_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_block_idx = num_blocks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            transfer_spec=(src_spec, dst_spec),
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_block_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            block_size_factor = self.config.block_size_factor
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_block_idx * block_size_factor
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
            max_offload_tokens = req_status.max_offload_tokens
            if max_offload_tokens is not None:
                num_offloadable_tokens = min(num_offloadable_tokens, max_offload_tokens)

            # Skip decode-phase blocks: clamp to the prompt length so only
            # prefill (prompt) blocks become eligible for store. next_stored_idx
            # never advances past this boundary, so decode blocks are never
            # queued in this or any later step.
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req.num_prompt_tokens
                )

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_idx, (group_config, group_state) in enumerate(
                zip(self.config.kv_group_configs, req_status.group_states)
            ):
                if group_idx not in req_status.store_group_indices:
                    continue
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)

                alignment_block_count = group_config.alignment_block_count
                tail = group_config.sliding_window_size_in_blocks

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%.
                    if alignment_block_count is not None:
                        assert tail is not None
                        abs_block_idx = start_block_idx + key_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                if (req.kv_transfer_params or {}).get("hyrex_observe_cache"):
                    logger.info(
                        "HYREX_NATIVE_EVICTION_EVENT %s",
                        json.dumps({
                            "request_id": req_id,
                            "admitted_blocks": 0,
                            "evicted_blocks": 0,
                            "rejected_blocks": len(new_offload_keys),
                        }, sort_keys=True),
                    )
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            if (req.kv_transfer_params or {}).get("hyrex_observe_cache"):
                logger.info(
                    "HYREX_NATIVE_EVICTION_EVENT %s",
                    json.dumps({
                        "request_id": req_id,
                        "admitted_blocks": len(store_output.keys_to_store),
                        "evicted_blocks": len(store_output.evicted_keys),
                        "rejected_blocks": 0,
                    }, sort_keys=True),
                )

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_idx, (group_config, group_state) in enumerate(
                zip(self.config.kv_group_configs, req_status.group_states)
            ):
                if group_idx not in req_status.store_group_indices:
                    group_sizes.append(0)
                    block_indices.append(0)
                    continue
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    offloaded_block_idx = start_block_idx + idx
                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_block_idx = num_blocks

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, transfer_spec=(src_spec, dst_spec)
            )

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs

    def _schedule_hyrex_load_jobs(self) -> None:
        """Order native H2D jobs using measured HyRex recovery metadata.

        A connector reaches this point only after allocating external KV
        blocks, so it is not safe to turn a load into replay here. We apply
        the plan only when every selected action remains LOAD; otherwise the
        original order is retained and the caller must choose replay before
        requesting external tokens.
        """
        if not envs.VLLM_HYREX_SCHEDULE_LOADS or len(self._current_batch_load_jobs) < 2:
            return

        from vllm.v1.kv_offload.hyrex_scheduler import (
            HyRexPlanner,
            RecoveryAction,
            RecoverySegment,
            StateKind,
        )

        segments: list[RecoverySegment] = []
        now_ms = time.time() * 1000
        for job_id, job in self._current_batch_load_jobs.items():
            req_status = self._req_status.get(job.req_id)
            params = (
                req_status.req.kv_transfer_params if req_status is not None else None
            )
            if req_status is None:
                return
            _, dst_spec = job.transfer_spec
            if not isinstance(dst_spec, GPULoadStoreSpec):
                return

            # Derive the measurement from the native allocation, not client
            # metadata. Each group size is in GPU blocks, whose page size is
            # defined by the corresponding KVCacheSpec.
            if len(dst_spec.group_sizes) != len(self._group_specs):
                return
            loaded_groups = [
                index for index, count in enumerate(dst_spec.group_sizes) if count
            ]
            if not loaded_groups:
                return
            load_bytes = sum(
                dst_spec.group_sizes[index] * self._group_specs[index].page_size_bytes
                for index in loaded_groups
            )
            missing_tokens = max(
                dst_spec.group_sizes[index]
                * self.config.kv_group_configs[index].gpu_block_size
                for index in loaded_groups
            )
            params = params or {}
            replay_ms = params.get("hyrex_replay_ms", inf)
            if not isinstance(replay_ms, (int, float)) or replay_ms < 0:
                return
            state_kind = (
                StateKind.RECURRENT
                if all(
                    isinstance(self._group_specs[index], MambaSpec)
                    for index in loaded_groups
                )
                else StateKind.FULL_KV
            )
            arrival_ms = float(
                params.get(
                    "hyrex_arrival_ms",
                    getattr(req_status.req, "arrival_time", 0.0) * 1000,
                )
            )
            deadline_ms = params.get("hyrex_deadline_ms")
            if deadline_ms is None and params.get("hyrex_ttft_slo_ms") is not None:
                waited_ms = max(0.0, now_ms - arrival_ms)
                deadline_ms = max(
                    0.0, float(params["hyrex_ttft_slo_ms"]) - waited_ms
                )
            else:
                waited_ms = max(0.0, now_ms - arrival_ms)
            starvation_ms = float(params.get("hyrex_starvation_ms", 0.0))
            starvation_promoted = starvation_ms > 0 and waited_ms >= starvation_ms
            priority = float(
                params.get("hyrex_priority", -getattr(req_status.req, "priority", 0))
            )
            if starvation_promoted:
                # A promoted job outranks ordinary request priorities within
                # this transfer batch, making the waiting bound observable.
                priority += float(params.get("hyrex_starvation_boost", 1_000_000.0))
            segments.append(
                RecoverySegment(
                    request_id=job.req_id,
                    segment_id=str(job_id),
                    state_kind=state_kind,
                    missing_tokens=missing_tokens,
                    load_bytes=load_bytes,
                    replay_ms=float(replay_ms),
                    source_ready=True,
                    source_key=params.get("hyrex_source_key"),
                    materialization_key=params.get("hyrex_materialization_key"),
                    priority=priority,
                    arrival_ms=arrival_ms,
                    deadline_ms=deadline_ms,
                )
            )

        plan = HyRexPlanner().plan(segments)
        if any(task.action is not RecoveryAction.LOAD for task in plan.tasks):
            return
        ordered_ids = [int(task.segment.segment_id) for task in plan.tasks]
        planned_tasks = {
            int(task.segment.segment_id): task for task in plan.tasks
        }
        self._current_batch_load_jobs = {
            job_id: self._current_batch_load_jobs[job_id] for job_id in ordered_ids
        }
        for rank, job_id in enumerate(ordered_ids):
            job = self._current_batch_load_jobs[job_id]
            params = self._req_status[job.req_id].req.kv_transfer_params or {}
            if params.get("hyrex_observe_cache"):
                task = planned_tasks[job_id]
                logger.info(
                    "HYREX_NATIVE_DISPATCH %s",
                    json.dumps({
                        "request_id": job.req_id,
                        "job_id": job_id,
                        "dispatch_rank": rank,
                        "dispatch_batch_size": len(ordered_ids),
                        "estimated_finish_ms": task.estimated_finish_ms,
                        "deadline_ms": task.segment.deadline_ms,
                        "predicted_slo_miss": (
                            task.segment.deadline_ms is not None
                            and task.estimated_finish_ms > task.segment.deadline_ms
                        ),
                        "waited_ms": max(0.0, now_ms - task.segment.arrival_ms),
                        "starvation_promoted": bool(
                            params.get("hyrex_starvation_ms", 0.0)
                            and max(0.0, now_ms - task.segment.arrival_ms)
                            >= float(params["hyrex_starvation_ms"])
                        ),
                    }, sort_keys=True),
                )

    def _schedule_cacheflow_load_jobs(self) -> None:
        """Prioritize real H2D jobs by remaining replay work."""
        def priority(item: tuple[int, TransferJob]) -> tuple[float, int, str, int]:
            job_id, job = item
            params = self._req_status[job.req_id].req.kv_transfer_params or {}
            replay_ms = params.get("cacheflow_replay_ms", 0.0)
            if not isinstance(replay_ms, (int, float)) or replay_ms < 0:
                replay_ms = 0.0
            _, dst = job.transfer_spec
            return -float(replay_ms), -len(dst.block_ids), job.req_id, job_id

        if any(
            (self._req_status[job.req_id].req.kv_transfer_params or {}).get(
                "hybrid_baseline"
            ) == "cacheflow_hybrid"
            for job in self._current_batch_load_jobs.values()
        ):
            self._current_batch_load_jobs = dict(
                sorted(self._current_batch_load_jobs.items(), key=priority)
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._finalize_hyrex_batch()
        self._update_req_states(scheduler_output)
        self.manager.on_schedule_end()

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        # If all tracked requests are finished, flush all pending jobs
        # (both store and load) - there might not be a future scheduler
        # step to trigger their completion.
        if self._req_status and all(
            rs.req.is_finished() for rs in self._req_status.values()
        ):
            self._current_batch_jobs_to_flush.update(self._jobs.keys())

        self._schedule_hyrex_load_jobs()
        self._schedule_cacheflow_load_jobs()
        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=self._build_store_jobs(scheduler_output),
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        if not getattr(self, "_hyrex_step_clock_advanced", False):
            self._decay_hyrex_queue_debt()
        else:
            self._hyrex_queue_clock_ms = time.monotonic() * 1000
        self._hyrex_step_clock_advanced = False
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                observation = meta.transfer_observations.get(job_id)
                if observation is not None:
                    num_bytes, service_ms, _ = observation
                    params = req_status.req.kv_transfer_params or {}
                    self._observe_hyrex_h2d(
                        num_bytes,
                        service_ms,
                        int(params.get("hyrex_concurrency", 1)),
                    )
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.non_sliding_window_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if not req_status.transfer_jobs and req_status.req.is_finished():
                del self._req_status[job_status.req_id]

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        getattr(self, "_hyrex_pending_lookups", {}).pop(request.request_id, None)
        getattr(self, "_hyrex_preplanned", {}).pop(request.request_id, None)
        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        req_status = self._req_status.get(request.request_id)

        req_context = (
            req_status.req_context if req_status else _create_req_context(request)
        )
        self.manager.on_request_finished(req_context)

        if req_status is None:
            return False, None
        if not req_status.transfer_jobs:
            del self._req_status[request.request_id]
            return False, None
        # Pending stores will outlive the request's block ownership.
        # Register them so future block reuse triggers a flush.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            block_hashes = [get_offload_block_hash(key) for key in event.keys]
            if event.removed:
                yield BlockRemoved(block_hashes=block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=0,
                    medium=event.medium,
                    lora_name=None,
                )

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored blocks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from block 0
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_block_idx = 0

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
