# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal-aware cache-key token ids in the MP connector.

vLLM emits identical placeholder token ids for every image, so the MP
connector must overwrite placeholder spans with mm_hash-derived values
before token ids are used for key derivation (lookup, store, retrieve,
lock release). These tests exercise the public tracker/metadata
interfaces of ``lmcache_mp_connector``.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    LMCacheMPConnector,
)
from lmcache.integration.vllm.utils import hex_hash_to_int16  # noqa: E402

IMAGE_PLACEHOLDER_ID = 99


@pytest.mark.parametrize("shallow", [False, True])
def test_load_boundary_ablation_keeps_deep_cpu_storage(shallow):
    from vllm.v1.request import RequestStatus

    req = _FakeRequest(list(range(1740)))
    req.status = RequestStatus.WAITING
    tracker = LMCacheMPRequestTracker(req)
    adapter = Mock()
    adapter.lmcache_tokens_per_chunk = 528
    adapter.check_lookup_result.side_effect = [1056, 1472]
    connector = SimpleNamespace(
        _get_or_create_request_tracker=lambda _: tracker,
        scheduler_adapter=adapter,
        _hyrex_page_size=16,
        _hyrex_deep_lookup=True,
        _hyrex_full_group_id=3,
        _group_tokens_per_block=[528] * 4,
        _hit_alignment_tokens=528,
        _hyrex_last_state=True,
        _hyrex_full_load_to_state=shallow,
    )
    result = LMCacheMPConnector.get_num_new_matched_tokens(connector, req, 0)
    assert result == (1056 if shallow else 1472, True)
    assert tracker.num_full_stored_tokens == 1472
    assert tracker.num_full_hit_tokens == (1056 if shallow else 1472)
    assert tracker.num_lmcache_hit_tokens == 1056
    if shallow:
        adapter.free_lookup_locks.assert_called_once()
        assert adapter.free_lookup_locks.call_args.kwargs["start"] == 1056
        assert adapter.free_lookup_locks.call_args.kwargs["end"] == 1472
        assert adapter.free_lookup_locks.call_args.kwargs["object_group_ids"] == [3]
    else:
        adapter.free_lookup_locks.assert_not_called()


@dataclass
class _FakePlaceholder:
    offset: int
    length: int


@dataclass
class _FakeMMFeature:
    identifier: str
    mm_position: _FakePlaceholder


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(
        self,
        prompt_token_ids: list[int],
        mm_features: list[_FakeMMFeature] | None = None,
        cache_salt: str = "",
    ):
        self.request_id = "req-0"
        self.cache_salt = cache_salt
        self.prompt_token_ids = list(prompt_token_ids)
        self._live_token_ids = list(prompt_token_ids)
        self.all_token_ids = ConstantList(self._live_token_ids)
        self.mm_features = mm_features or []

    def append_decode_token(self, token_id: int):
        """Simulate vLLM appending a decode token to the live token list."""
        self._live_token_ids.append(token_id)


def _make_mm_request(
    prompt_token_ids: list[int],
    identifier: str,
    offset: int,
    length: int,
) -> _FakeRequest:
    mm_features = [_FakeMMFeature(identifier, _FakePlaceholder(offset, length))]
    return _FakeRequest(prompt_token_ids, mm_features=mm_features)


def test_text_only_request_uses_raw_token_ids():
    prompt = list(range(100, 108))
    tracker = LMCacheMPRequestTracker(_FakeRequest(prompt))
    assert tracker.get_token_ids() == prompt


def test_text_only_request_returns_mutable_copy():
    prompt = list(range(100, 108))
    tracker = LMCacheMPRequestTracker(_FakeRequest(prompt))
    token_ids = tracker.get_token_ids()
    token_ids[0] = -1
    assert tracker.get_token_ids() == prompt


def test_mm_request_overwrites_placeholder_span():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 3 + [3, 4, 5]
    tracker = LMCacheMPRequestTracker(
        _make_mm_request(prompt, identifier="0xabcd", offset=2, length=3)
    )
    fill = hex_hash_to_int16("0xabcd")
    assert tracker.get_token_ids() == [1, 2, fill, fill, fill, 3, 4, 5]


def test_different_images_produce_different_key_tokens():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 3 + [3, 4, 5]
    tracker_a = LMCacheMPRequestTracker(
        _make_mm_request(prompt, identifier="0xaaaa", offset=2, length=3)
    )
    tracker_b = LMCacheMPRequestTracker(
        _make_mm_request(prompt, identifier="0xbbbb", offset=2, length=3)
    )
    assert tracker_a.get_token_ids() != tracker_b.get_token_ids()


def test_decode_tokens_appended_unchanged():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3]
    request = _make_mm_request(prompt, identifier="0xabcd", offset=2, length=2)
    tracker = LMCacheMPRequestTracker(request)
    request.append_decode_token(500)
    request.append_decode_token(501)
    fill = hex_hash_to_int16("0xabcd")
    assert tracker.get_token_ids() == [1, 2, fill, fill, 3, 500, 501]


def _prepare_storable_tracker(request: _FakeRequest) -> LMCacheMPRequestTracker:
    """Give the tracker enough scheduled tokens and blocks to emit one
    store op covering the whole 8-token prompt (chunk size 4)."""
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_scheduled_tokens = 8
    return tracker


def test_store_metadata_uses_mm_adjusted_token_ids():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
    request = _make_mm_request(prompt, identifier="0xabcd", offset=2, length=2)
    tracker = _prepare_storable_tracker(request)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
    )

    assert metadata is not None
    fill = hex_hash_to_int16("0xabcd")
    assert metadata.op.token_ids == [1, 2, fill, fill, 3, 4, 5, 6]
    assert metadata.op.start == 0
    assert metadata.op.end == 8


def test_independent_full_store_keeps_deeper_chunks():
    tracker = LMCacheMPRequestTracker(_FakeRequest(list(range(12))))
    tracker.allocated_block_ids = {
        0: [10, 11, 12],
        1: [20, 21, 22],
    }
    tracker.num_scheduled_tokens = 12
    common = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, 4, [4, 4], max_tokens=4
    )
    full = LMCacheMPRequestMetadata.GetIndependentFullStoreMetadata(
        tracker, 4, [4, 4], full_group_id=1
    )
    assert common is not None and (common.op.start, common.op.end) == (0, 4)
    assert full is not None and (full.op.start, full.op.end) == (4, 12)
    assert full.op.object_group_ids == [1]
    assert full.op.block_ids == [[11, 12], [21, 22]]
    assert tracker.num_stored_tokens == 4
    assert tracker.num_full_stored_tokens == 12
    assert LMCacheMPRequestMetadata.GetIndependentFullStoreMetadata(
        tracker, 4, [4, 4], full_group_id=1
    ) is None


def test_full_kv_uses_physical_16_token_pages_independently():
    tracker = LMCacheMPRequestTracker(_FakeRequest(list(range(1600))))
    tracker.allocated_block_ids = {0: [0, 1, 2, 3], 1: [7, 5, 9, 1]}
    tracker.num_scheduled_tokens = 1600
    state = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, 528, [528, 528], max_tokens=528, object_group_ids=[0]
    )
    full = LMCacheMPRequestMetadata.GetIndependentFullStoreMetadata(
        tracker, 528, [528, 528], full_group_id=1, page_size=16
    )
    assert state is not None and (state.op.start, state.op.end) == (0, 528)
    assert full is not None and (full.op.start, full.op.end) == (0, 1600)
    assert full.op.block_ids[0] == []
    assert len(full.op.block_ids[1]) == 100
    assert full.op.block_ids[1][0] == 7 * 33
    assert full.op.block_ids[1][33] == 5 * 33
    assert full.op.block_ids[1][-1] == 1 * 33

    tracker.num_lmcache_hit_tokens = 528
    tracker.num_full_hit_tokens = 1600
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    reload = LMCacheMPRequestMetadata.GetIndependentFullRetrieveMetadata(
        tracker, 528, [528, 528], full_group_id=1, page_size=16
    )
    assert reload is not None
    assert (reload.op.start, reload.op.end) == (0, 1600)
    assert reload.op.block_ids == full.op.block_ids

    terminal_state = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, 528, [528, 528], object_group_ids=[0],
        last_checkpoint_only=True,
    )
    assert terminal_state is not None
    assert (terminal_state.op.start, terminal_state.op.end) == (0, 528)
    tracker.num_lmcache_hit_tokens = 1584
    terminal_state = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, 528, [528, 528], object_group_ids=[0],
        last_checkpoint_only=True,
    )
    assert terminal_state is not None
    assert (terminal_state.op.start, terminal_state.op.end) == (1056, 1584)
    assert terminal_state.op.block_ids[0] == [2]
    assert terminal_state.op.skip_first_n_tokens == 0

def test_retrieve_metadata_uses_mm_adjusted_token_ids():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
    request = _make_mm_request(prompt, identifier="0xabcd", offset=2, length=2)
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_lmcache_hit_tokens = 8
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
    )

    assert metadata is not None
    fill = hex_hash_to_int16("0xabcd")
    assert metadata.op.token_ids == [1, 2, fill, fill, 3, 4, 5, 6]
    assert metadata.op.start == 0
    assert metadata.op.end == 8
