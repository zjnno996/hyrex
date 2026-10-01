# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.kv_offload.hyrex_terminal_state import HyRexTerminalStateStore

pytestmark = pytest.mark.skip_global_cleanup


def test_terminal_state_round_trip_and_lru_eviction():
    state = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    store = HyRexTerminalStateStore(max_bytes=state.nbytes)
    store.store("prefix-a", 1, (state,))
    restored = store.restore("prefix-a", 1, torch.device("cpu"))
    assert restored is not None
    assert torch.equal(restored[0], state)

    store.store("prefix-b", 1, (state,))
    assert store.restore("prefix-a", 1, torch.device("cpu")) is None


def test_terminal_state_restores_a_vllm_recurrent_group_block():
    cache = {"linear": [torch.zeros((3, 2)), torch.zeros((3, 4))]}
    cache["linear"][0][1].fill_(3)
    cache["linear"][1][1].fill_(7)
    store = HyRexTerminalStateStore()
    store.store_group("prefix", 0, ("linear",), cache, 1)
    cache["linear"][0][2].zero_()
    cache["linear"][1][2].zero_()

    assert store.restore_group("prefix", 0, ("linear",), cache, 2)
    assert torch.equal(cache["linear"][0][2], torch.full((2,), 3.0))
    assert torch.equal(cache["linear"][1][2], torch.full((4,), 7.0))
