from types import SimpleNamespace

import pytest

from lmcache.integration.vllm.tail_probe_connector import tail_salt, TailProbeConnector
from vllm.v1.core.sched.scheduler import Scheduler


def test_tail_key_includes_entire_prefix():
    a = list(range(1040))
    assert tail_salt(a, 1040, "a") == tail_salt(a + [1], 1040, "a")
    assert tail_salt(a, 1040, "a") != tail_salt([99] + a[1:], 1040, "a")
    assert tail_salt(a, 1040, "a") != tail_salt(a, 1040, "b")
    with pytest.raises(ValueError):
        tail_salt(a, 1056, "a")


def test_tail_capture_and_next_coarse_checkpoint(monkeypatch):
    monkeypatch.setenv("VLLM_HYREX_TAIL_PROBE", "1040")
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=528), use_eagle=False)
    req = SimpleNamespace(num_computed_tokens=528, num_prompt_tokens=1054, num_tokens=1054)
    split = Scheduler._mamba_block_aligned_split
    assert split(scheduler, req, 526) == 512
    req.num_computed_tokens = 1040
    req.num_tokens = req.num_prompt_tokens = 1800
    assert split(scheduler, req, 528) == 16


def test_tail_transport_uses_partial_state_slot():
    probe = object.__new__(TailProbeConnector)
    probe.tail = 1040
    probe._group_tokens_per_block = [528] * 4
    probe._hyrex_full_group_id = 3
    tracker = SimpleNamespace(request_id="r", cache_salt="", get_token_ids=lambda: list(range(1054)),
                              allocated_block_ids={i: [11, 12] for i in range(4)})
    meta = probe._tail_meta(tracker, "STORE")
    assert meta.op.block_ids == [[12], [12], [12], []]
    assert meta.op.object_group_ids == [0, 1, 2]
    # STORE shares the real request's memoized prefix hashes. Using synthetic
    # zero tokens here silently disagrees with a fresh LOOKUP session.
    assert meta.op.token_ids == tracker.get_token_ids()
    tracker.allocated_block_ids[1][1] = 0
    with pytest.raises(RuntimeError, match="null block"):
        probe._tail_meta(tracker, "STORE")


def test_skipped_checkpoints_do_not_split_or_publish(monkeypatch):
    from unittest.mock import patch
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SingleTypeKVCacheManager
    req = SimpleNamespace(request_id="r", num_computed_tokens=1040,
                          num_prompt_tokens=1191, num_tokens=1191,
                          hyrex_tail_skip_new_states=True)
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=528), use_eagle=False)
    assert Scheduler._mamba_block_aligned_split(scheduler, req, 151) == 151
    manager = SimpleNamespace(block_size=528, num_cached_block={})
    with patch.object(SingleTypeKVCacheManager, "cache_blocks") as publish:
        MambaManager.cache_blocks(manager, req, 1191)
        publish.assert_not_called()
    assert manager.num_cached_block["r"] == 2


def test_skipped_checkpoints_disable_state_store_but_keep_full(monkeypatch):
    from unittest.mock import patch
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    probe = object.__new__(TailProbeConnector)
    probe.tail = 1040
    probe._hyrex_state_store_cap = None
    tracker = SimpleNamespace(request_id="r", tail_skip_new_states=True,
                              tail_loaded=True, num_scheduled_tokens=151,
                              num_vllm_hit_tokens=0, num_lmcache_hit_tokens=1040)
    def store(self, tracker, chunk, metadata):
        assert self._hyrex_state_store_cap == 0
        metadata.append("full-store-still-enabled")
    metadata = []
    with patch.object(LMCacheMPConnector, "_append_store_metadata", store):
        probe._append_store_metadata(tracker, 528, metadata)
    assert metadata == ["full-store-still-enabled"]
    assert probe._hyrex_state_store_cap is None


@pytest.mark.parametrize("corrupt", [False, True])
def test_model_visible_state_audit_detects_bad_copy(monkeypatch, corrupt):
    import torch
    from vllm.v1.worker import mamba_utils as m
    monkeypatch.setenv("LMCACHE_TAIL_VERIFY_MODEL_STATE", "1")
    monkeypatch.setenv("VLLM_HYREX_TAIL_PROBE", "1040")
    state = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    monkeypatch.setattr(m, "collect_mamba_copy_meta", lambda *args: None)
    monkeypatch.setattr(m, "do_mamba_copy_block",
                        lambda _: None if corrupt else state[3].copy_(state[2]))
    scheduler = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"r": 151})
    kwargs = dict(scheduler_output=scheduler,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer"], kv_cache_spec=None)]),
        cache_config=SimpleNamespace(enable_prefix_caching=True), mamba_state_idx={},
        input_batch=SimpleNamespace(req_ids=["r"], num_accepted_tokens_cpu=[1]),
        requests={"r": SimpleNamespace(num_computed_tokens=1040, block_ids=[[1, 2, 3]])},
        forward_context={"layer": SimpleNamespace(kv_cache=[state])},
        mamba_state_copy_funcs=(), copy_bufs=SimpleNamespace(mamba_group_ids=[0],
            mamba_spec=SimpleNamespace(num_speculative_blocks=0, block_size=528)))
    if corrupt:
        with pytest.raises(RuntimeError, match="Model state copy mismatch"):
            m.preprocess_mamba(**kwargs)
    else:
        m.preprocess_mamba(**kwargs)
