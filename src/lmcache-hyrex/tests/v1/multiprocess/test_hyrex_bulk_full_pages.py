"""Check page ordering and layer layout across multiple staging-buffer runs."""

from types import SimpleNamespace

import torch
import pytest

import lmcache.c_ops as ops
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    _transfer_batched_full_pages,
    _transfer_contiguous_full_pages,
    _verify_single_state_checkpoint,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("capacity", [2, 3])
def test_batched_roundtrip(device, fragmented, capacity):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    count, nl, page, nh, hs = 7, 3, 16, 1, 2
    shape = (count, 2, nl, page, nh, hs)
    expected = torch.arange(count*2*nl*page*nh*hs, dtype=torch.float32).reshape(shape)
    slab = torch.full((count+2, 2, nl, page, nh, hs), -7., pin_memory=device == "cuda")
    hosts = [slab[i+1] for i in range(count)]
    if fragmented:
        hosts = [torch.full_like(hosts[0], -7., pin_memory=device == "cuda") for _ in hosts]
    size = expected[0].numel()*4
    objects = [SimpleNamespace(raw_tensor=t.view(torch.uint8).flatten(),
                               get_size=lambda: size, parent=lambda: None) for t in hosts]
    ids = torch.tensor([8, 1, 6, 0, 3, 7, 4], device=device)
    sources = [torch.full((9, 2, page, nh, hs), -1., device=device) for _ in range(nl)]
    targets = [torch.full_like(t, -2.) for t in sources]
    for i, t in enumerate(sources):
        t.index_copy_(0, ids, expected[:, :, i].to(device))
    context = SimpleNamespace(
        kv_layer_groups_manager=SimpleNamespace(
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            kernel_groups=[SimpleNamespace(layer_indices=list(range(nl)))]),
        kv_caches_=sources,
        get_temp_object_group_buffer=lambda *_: buffer,
        get_shape_desc=lambda *_: SimpleNamespace(nh=nh, hs=hs),
        get_engine_kv_format=lambda *_: ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS)
    buffer = torch.empty(capacity*size, dtype=torch.uint8, device=device)

    def transfer():
        assert _transfer_batched_full_pages(context, [ids], objects, 0, page, ops.TransferDirection.D2H)
        context.kv_caches_ = targets
        assert _transfer_batched_full_pages(context, [ids], objects, 0, page, ops.TransferDirection.H2D)

    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000000)
            transfer()
        stream.synchronize()
    else:
        transfer()
    for i, t in enumerate(targets):
        assert torch.equal(t[ids].cpu(), expected[:, :, i])
        assert (t[[2, 5]] == -2).all()
    assert torch.equal(torch.stack(hosts), expected)
    assert (slab[[0, -1]] == -7).all()
    # Partially reserved stores must fall back before touching any buffers.
    assert not _transfer_batched_full_pages(context, [ids], [None]+objects[1:], 0, page, ops.TransferDirection.D2H)


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
