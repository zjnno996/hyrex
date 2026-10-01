"""Exact last-checkpoint replacement for isolated seed/resume experiments.

Not a continuous-session eviction policy: CPU must be cleared before each
seed/resume pair. Old tail objects are retained until that pair-level clear.
"""
from contextlib import contextmanager
import os

from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.tail_probe_connector import TailProbeConnector
from lmcache.integration.vllm.checkpoint_index import CheckpointIndex, checkpoint_salt


def replacement_positions(length):
    coarse = length // 528 * 528
    tail = length // 16 * 16
    return (coarse - 528, tail) if coarse and tail != coarse else (coarse, 0)


class ReplaceTailConnector(TailProbeConnector):
    def __init__(self, *args, **kwargs):
        LMCacheMPConnector.__init__(self, *args, **kwargs)
        self.tail = 0
        self.checkpoint_index = CheckpointIndex()
        if self._hyrex_page_size != 16 or not self._hyrex_last_state:
            raise ValueError("replacement requires Full16 and last-state-only")

    @contextmanager
    def boundary(self, value):
        previous = self.tail
        self.tail = value
        try:
            yield
        finally:
            self.tail = previous

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if num_computed_tokens:
            raise RuntimeError("replacement probe requires GPU reset")
        tracker = self._get_or_create_request_tracker(request)
        tracker.replace_cap, tracker.replace_store = replacement_positions(request.num_prompt_tokens)
        tracker.replace_prompt_length = request.num_prompt_tokens
        lookup_id = request.request_id + ":hyrex-full"
        self.scheduler_adapter.maybe_submit_lookup_request(
            lookup_id, token_ids=tracker.get_token_ids(), cache_salt=tracker.cache_salt,
            object_group_ids=[self._hyrex_full_group_id], chunk_size=16)
        full = self.scheduler_adapter.check_lookup_result(lookup_id)
        if full is None:
            return None, True
        candidate = full
        if os.getenv("VLLM_HYREX_SINGLE_FORWARD") == "1":
            candidates = self.checkpoint_index.candidates(tracker.get_token_ids(), full, tracker.cache_salt)
            candidate = candidates[0] if candidates else 0
        tracker.replace_load = candidate
        if not candidate or (candidate % 528 == 0 and os.getenv("VLLM_HYREX_SINGLE_FORWARD") != "1"):
            return LMCacheMPConnector.get_num_new_matched_tokens(self, request, num_computed_tokens)
        with self.boundary(candidate):
            result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if getattr(tracker, "tail_loaded", False):
            # Do not manufacture historical checkpoints skipped by tail load.
            tracker.num_stored_tokens = max(tracker.num_stored_tokens, candidate // 528 * 528)
            request.hyrex_state_load_start = (candidate - 1) // 528 * 528
            if os.getenv("LMCACHE_HYREX_RECOVERY_ONLY") == "1":
                tracker.tail_skip_new_states = True
                request.hyrex_tail_skip_new_states = True
        return result

    def _salt(self, tracker):
        if os.getenv("VLLM_HYREX_SINGLE_FORWARD") == "1":
            return checkpoint_salt(tracker.get_token_ids(), self.tail, tracker.cache_salt)
        return super()._salt(tracker)

    def _tail_meta(self, tracker, direction):
        with self.boundary(tracker.replace_load if direction == "RETRIEVE" else tracker.replace_store):
            meta = super()._tail_meta(tracker, direction)
        if direction == "STORE" and os.getenv("VLLM_HYREX_SINGLE_FORWARD") == "1":
            # The checkpoint occupies the replaced coarse slot; the final
            # running state lives in the following slot and can keep decoding.
            index = tracker.replace_store // 528 - 1
            for group in self._groups():
                block = tracker.allocated_block_ids[group][index]
                if block == 0:
                    raise RuntimeError("tail snapshot destination is null")
                meta.op.block_ids[group] = [block]
        return meta

    def _process_retrieve_requests(self, metadata):
        # The parent temporarily zeros the state hit for the ordinary retrieve.
        # Restore each tracker's actual (request-specific) boundary afterward.
        tails = [t for t in self.request_trackers.values()
                 if getattr(t, "tail_loaded", False) and t.is_ready_for_retrieving()]
        super()._process_retrieve_requests(metadata)
        for tracker in tails:
            tracker.num_lmcache_hit_tokens = tracker.replace_load

    def _append_store_metadata(self, tracker, chunk_size, metadata):
        if (getattr(tracker, "tail_loaded", False)
                and os.getenv("LMCACHE_HYREX_RECOVERY_ONLY") == "1"):
            return
        cap = self._hyrex_state_store_cap
        loaded = getattr(tracker, "tail_loaded", False)
        self._hyrex_state_store_cap = tracker.replace_cap
        try:
            if os.getenv("VLLM_HYREX_SINGLE_FORWARD") == "1":
                # Every state snapshot uses an independent content-addressed
                # slot. No contiguous coarse-state lookup across moved holes.
                self._hyrex_state_store_cap = 0
                LMCacheMPConnector._append_store_metadata(self, tracker, chunk_size, metadata)
                computed = tracker.num_scheduled_tokens + max(tracker.num_vllm_hit_tokens, tracker.num_lmcache_hit_tokens)
                previous = getattr(tracker, "replace_previous", max(tracker.num_vllm_hit_tokens, tracker.num_lmcache_hit_tokens))
                coarse = tracker.replace_prompt_length // 528 * 528
                positions = list(range(528, coarse + 1, 528))
                original = tracker.replace_store
                if original:
                    positions[-1] = original
                for position in positions:
                    if previous < position <= computed:
                        tracker.replace_store = position
                        metadata.add_request_metadata(self._tail_meta(tracker, "STORE"))
                        self.checkpoint_index.record(tracker.get_token_ids(), position, tracker.cache_salt)
                tracker.replace_store = original
                tracker.replace_previous = computed
                return
            if not tracker.replace_store:
                return LMCacheMPConnector._append_store_metadata(self, tracker, chunk_size, metadata)
            # Loading an earlier tail must not suppress this request's new tail.
            tracker.tail_loaded = tracker.replace_load == tracker.replace_store and loaded
            with self.boundary(tracker.replace_store):
                return super()._append_store_metadata(tracker, chunk_size, metadata)
        finally:
            tracker.tail_loaded = loaded
            self._hyrex_state_store_cap = cap

    def wait_for_save(self):
        if os.getenv("VLLM_HYREX_SINGLE_FORWARD") == "1":
            # Snapshots occupy separate blocks held until request STORE futures
            # complete; the model's running state no longer aliases them.
            return LMCacheMPConnector.wait_for_save(self)
        return super().wait_for_save()
