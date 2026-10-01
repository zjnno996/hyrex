"""Single-boundary exact-state experiment, not a production cache policy.

Keep regular checkpoints and one explicitly selected tail. State objects use
the existing opaque 528-token transport geometry, under a separate namespace
keyed by the SHA256 of the complete *real* prefix. The virtual transport key
does not change the model input or claim 528 tokens of recurrent history.
"""
import hashlib
import json
import os
import time

from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector, LMCacheMPRequestMetadata, LoadStoreOp, logger,
)


def tail_salt(tokens, boundary, salt):
    if boundary <= 0 or len(tokens) < boundary:
        raise ValueError("checkpoint requires the entire prefix")
    digest = hashlib.sha256(json.dumps(tokens[:boundary]).encode()).hexdigest()
    return f"{salt}:exact-tail-v1:{boundary}:{digest}"


class TailProbeConnector(LMCacheMPConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail = int(os.environ["VLLM_HYREX_TAIL_PROBE"])
        if not self.tail or self.tail % 16 or self.tail % 528 == 0:
            raise ValueError("probe tail must align to 16, but not 528")
        if self._hyrex_page_size != 16 or not self._hyrex_last_state:
            raise ValueError("probe requires full16 and last-state-only")

    def _salt(self, tracker):
        return tail_salt(tracker.get_token_ids(), self.tail, tracker.cache_salt)

    def _groups(self):
        return [i for i in range(len(self._group_tokens_per_block))
                if i != self._hyrex_full_group_id]

    def _tail_meta(self, tracker, direction):
        ids = [[] for _ in self._group_tokens_per_block]
        for group in self._groups():
            idx = (self.tail - 1) // self._group_tokens_per_block[group]
            block = tracker.allocated_block_ids[group][idx]
            if block == 0:
                raise RuntimeError("tail checkpoint must not use null block")
            ids[group] = [block]
        return LMCacheMPRequestMetadata(
            request_id=tracker.request_id, direction=direction,
            op=LoadStoreOp(token_ids=tracker.get_token_ids(), block_ids=ids,
                           start=0, end=528, object_group_ids=self._groups()),
            cache_salt=self._salt(tracker),
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if num_computed_tokens:
            raise RuntimeError("tail probe requires a verified GPU reset")
        tracker = self._get_or_create_request_tracker(request)
        if os.getenv("LMCACHE_TAIL_COLD_FUSED_REFERENCE") == "1":
            return 0, False
        if len(tracker.all_token_ids) <= self.tail:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        lookup_id = request.request_id + ":tail-probe"
        self.scheduler_adapter.maybe_submit_lookup_request(
            lookup_id, token_ids=tracker.get_token_ids()[:528], cache_salt=self._salt(tracker),
            object_group_ids=self._groups(),
        )
        hit = self.scheduler_adapter.check_lookup_result(lookup_id)
        if hit is None:
            return None, True
        cap = self._hyrex_full_load_to_state
        if hit:
            self._hyrex_full_load_to_state = False
        try:
            result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        finally:
            self._hyrex_full_load_to_state = cap
        if result[0] is None:
            return result
        if hit:
            if tracker.num_full_hit_tokens < self.tail:
                raise RuntimeError("tail state hit without complete Full KV prefix")
            if tracker.num_lmcache_hit_tokens >= self.tail:
                self.scheduler_adapter.free_lookup_locks(
                    token_ids=tracker.get_token_ids()[:528], start=0, end=528, request_id=lookup_id,
                    cache_salt=self._salt(tracker), object_group_ids=self._groups(),
                )
                return result
            tracker.tail_base_hit = tracker.num_lmcache_hit_tokens
            tracker.tail_loaded = True
            tracker.tail_skip_new_states = os.getenv("LMCACHE_TAIL_SKIP_NEW_STATES") == "1"
            request.hyrex_tail_skip_new_states = tracker.tail_skip_new_states
            tracker.num_lmcache_hit_tokens = self.tail
            request.hyrex_state_boundary = self.tail
            request.hyrex_full_boundary = tracker.num_full_hit_tokens
            request.hyrex_state_load_start = self.tail // 528 * 528
            logger.info("TAIL_PROBE HIT request=%s state=%d full=%d base=%d",
                        request.request_id, self.tail, tracker.num_full_hit_tokens,
                        tracker.tail_base_hit)
        return result

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        tracker = self._get_request_tracker(request.request_id)
        loaded = getattr(tracker, "tail_loaded", False)
        if loaded and not getattr(tracker, "tail_locks_released", False):
            base = tracker.tail_base_hit
            if base:
                self.scheduler_adapter.free_lookup_locks(
                    token_ids=tracker.get_token_ids(), start=0, end=base,
                    request_id=request.request_id, cache_salt=tracker.cache_salt,
                    object_group_ids=self._groups(),
                )
            tracker.tail_locks_released = True
        actual = tracker.num_lmcache_hit_tokens
        if loaded:
            tracker.num_lmcache_hit_tokens = 0
        try:
            super().update_state_after_alloc(request, blocks, num_external_tokens)
        finally:
            tracker.num_lmcache_hit_tokens = actual
        self.scheduler_adapter.cleanup_lookup_result(request.request_id + ":tail-probe")

    def _process_retrieve_requests(self, metadata):
        tails = [t for t in self.request_trackers.values()
                 if getattr(t, "tail_loaded", False) and t.is_ready_for_retrieving()]
        for tracker in tails:
            tracker.num_lmcache_hit_tokens = 0
        try:
            super()._process_retrieve_requests(metadata)
        finally:
            for tracker in tails:
                tracker.num_lmcache_hit_tokens = self.tail
        for tracker in tails:
            metadata.add_request_metadata(self._tail_meta(tracker, "RETRIEVE"))

    def _append_store_metadata(self, tracker, chunk_size, metadata):
        computed = tracker.num_scheduled_tokens + max(
            tracker.num_vllm_hit_tokens, tracker.num_lmcache_hit_tokens)
        if os.getenv("LMCACHE_TAIL_COLD_FUSED_REFERENCE") == "1" and computed > self.tail:
            tracker.tail_skip_new_states = True
        cap = self._hyrex_state_store_cap
        if getattr(tracker, "tail_skip_new_states", False):
            self._hyrex_state_store_cap = 0
        try:
            super()._append_store_metadata(tracker, chunk_size, metadata)
        finally:
            self._hyrex_state_store_cap = cap
        computed = tracker.num_scheduled_tokens + max(
            tracker.num_vllm_hit_tokens, tracker.num_lmcache_hit_tokens)
        previous = getattr(tracker, "tail_previous_scheduled", 0)
        scheduled = tracker.num_scheduled_tokens - previous
        tracker.tail_previous_scheduled = tracker.num_scheduled_tokens
        if scheduled > 1:
            logger.info("TAIL_PROBE STEP request=%s start=%d count=%d end=%d skip_new_states=%s",
                        tracker.request_id, computed-scheduled, scheduled, computed,
                        getattr(tracker, "tail_skip_new_states", False))
        if computed == self.tail and not getattr(tracker, "tail_loaded", False):
            metadata.add_request_metadata(self._tail_meta(tracker, "STORE"))
            logger.info("TAIL_PROBE STORE request=%s boundary=%d", tracker.request_id, self.tail)

    def wait_for_save(self):
        metadata = self._get_connector_metadata()
        tail_store = any(m.direction == "STORE" and ":exact-tail-v1:" in m.cache_salt
                         for m in metadata.requests)
        start = time.perf_counter()
        super().wait_for_save()
        if tail_store:
            # The partial running-state block will be overwritten next step.
            # Wait for this request's real LMCache STORE before allowing it.
            from lmcache import torch_dev
            for request_id in {m.request_id for m in metadata.requests
                               if m.direction == "STORE"}:
                futures = self.worker_adapter.store_futures.get(request_id, [])
                if not futures:
                    raise RuntimeError("tail STORE was not submitted")
                for future in futures:
                    if not future.result(timeout=60):
                        raise RuntimeError("tail STORE failed")
            torch_dev.synchronize()
            logger.info("TAIL_PROBE SAVE_BARRIER_MS %.3f", (time.perf_counter()-start)*1000)

    def request_finished(self, request, block_ids):
        self.scheduler_adapter.end_session(request.request_id + ":tail-probe")
        return super().request_finished(request, block_ids)
