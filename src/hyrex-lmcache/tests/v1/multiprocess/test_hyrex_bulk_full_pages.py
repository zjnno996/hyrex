"""Check page ordering and layer layout across multiple staging-buffer runs."""

from types import SimpleNamespace

import torch
import pytest

import lmcache.c_ops as ops
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    _transfer_contiguous_full_pages,
    _verify_single_state_checkpoint,
)


@pytest.mark.parametrize("corrupt", [False, True])
def test_bulk_full_pages_scatter_preserves_layout(monkeypatch, corrupt):
    monkeypatch.setenv("LMCACHE_HYREX_VERIFY_FULL_H2D", "1")
    if corrupt:
        original_copy = torch.Tensor.index_copy_

        def corrupt_copy(self, dim, index, source):
            return original_copy(self, dim, index, torch.zeros_like(source))

        monkeypatch.setattr(torch.Tensor, "index_copy_", corrupt_copy)
    count, nl, page_size, nh, hs = 6, 3, 16, 1, 2
    source = torch.arange(count * 2 * nl * page_size * nh * hs).float().reshape(
        count, 2, nl, page_size, nh, hs
    )
    page_bytes = source[0].numel() * source.element_size()
    objects = [SimpleNamespace(
        raw_tensor=source[i].view(torch.uint8).flatten(),
        get_size=lambda: page_bytes,
    ) for i in range(count)]
    targets = [torch.full((8, 2, page_size, nh, hs), -1.) for _ in range(nl)]
    buffer = torch.empty(2 * page_bytes, dtype=torch.uint8)
    ids = torch.tensor([4, 1, 6, 0, 3, 7])
    context = SimpleNamespace(
        kv_layer_groups_manager=SimpleNamespace(
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            kernel_groups=[SimpleNamespace(layer_indices=list(range(nl)))],
        ),
        kv_caches_=targets,
        get_temp_object_group_buffer=lambda *_: buffer,
        get_kernel_group_kv_pointers=lambda *_: None,
        get_shape_desc=lambda *_: SimpleNamespace(nh=nh, hs=hs),
        get_engine_kv_format=lambda *_: ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
    )
    if corrupt:
        with pytest.raises(RuntimeError, match="Full KV H2D mismatch"):
            _transfer_contiguous_full_pages(context, [ids], objects, 0, page_size)
        return
    assert _transfer_contiguous_full_pages(context, [ids], objects, 0, page_size)
    for i, target in enumerate(targets):
        torch.testing.assert_close(target[ids], source[:, :, i])
        assert (target[[2, 5]] == -1).all()


@pytest.mark.parametrize("corrupt", [False, True])
def test_opaque_state_audit(corrupt):
    source = torch.arange(2*3*16*2).float().reshape(2, 3, 16, 1, 2)
    targets = [torch.zeros(4, 2, 16, 1, 2) for _ in range(3)]
    for i, target in enumerate(targets):
        target[2].copy_(source[:, i])
    if corrupt:
        targets[1][2, 0, 0, 0, 0] += 1
    context = SimpleNamespace(
        kv_layer_groups_manager=SimpleNamespace(
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            kernel_groups=[SimpleNamespace(layer_indices=[0, 1, 2])]),
        kv_caches_=targets,
        get_engine_kv_format=lambda _: ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS)
    obj = SimpleNamespace(raw_tensor=source.view(torch.uint8).flatten(),
                          get_size=lambda: source.numel()*source.element_size())
    if corrupt:
        with pytest.raises(RuntimeError, match="State byte mismatch"):
            _verify_single_state_checkpoint(context, [torch.tensor([2])], [obj], 0)
    else:
        _verify_single_state_checkpoint(context, [torch.tensor([2])], [obj], 0)
