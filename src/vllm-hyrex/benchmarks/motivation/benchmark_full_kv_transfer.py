"""Equal-byte transfer diagnostic; this measures transfer paths, not TTFT."""

import json
import statistics
import time
from types import SimpleNamespace

import torch

import lmcache.c_ops as ops
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    _transfer_contiguous_full_pages,
)
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer


def main():
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    nl, tokens, page, nh, hs = 8, 1056, 16, 4, 256
    npages = tokens // page
    source = torch.randn(2, nl, tokens, nh, hs, dtype=torch.bfloat16)
    coarse = [source[:, :, i:i + 528].contiguous().pin_memory()
              for i in range(0, tokens, 528)]
    fine = source.reshape(2, nl, npages, page, nh, hs).permute(
        2, 0, 1, 3, 4, 5
    ).contiguous().pin_memory()
    targets = [torch.empty(npages, 2, page, nh, hs, dtype=source.dtype, device=device)
               for _ in range(nl)]
    ids = torch.arange(npages, device=device)
    desc = SimpleNamespace(bs=page, nl=nl, nb=npages, nh=nh, hs=hs)
    fmt = ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
    page_bytes = fine[0].numel() * fine.element_size()
    buffer = torch.empty(page_bytes * 33, dtype=torch.uint8, device=device)
    objects = [SimpleNamespace(raw_tensor=t.view(torch.uint8).flatten(),
                               get_size=lambda: page_bytes) for t in fine]
    context = SimpleNamespace(
        kv_caches_=targets,
        kv_layer_groups_manager=SimpleNamespace(
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            kernel_groups=[SimpleNamespace(layer_indices=list(range(nl)))],
        ),
        get_temp_object_group_buffer=lambda *_: buffer,
        get_kernel_group_kv_pointers=lambda *_: None,
        get_shape_desc=lambda *_: desc,
        get_engine_kv_format=lambda *_: fmt,
    )

    def fallback(objects, chunk):
        multi_layer_block_kv_transfer(
            targets, objects, ids, device, ops.TransferDirection.H2D,
            desc, chunk, fmt, 0,
        )

    variants = {
        "coarse_528": lambda: fallback(coarse, 528),
        "fine_16": lambda: fallback(list(fine), 16),
        "fine_16_bulk_scatter": lambda: _transfer_contiguous_full_pages(
            context, [ids], objects, 0, page
        ),
    }
    for name, run in variants.items():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            run()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        for i, target in enumerate(targets):
            expected = source[:, i].reshape(2, npages, page, nh, hs).transpose(0, 1)
            torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)
        print(json.dumps({"variant": name, "tokens": tokens,
                          "payload_mib": source.numel() * source.element_size() / 2**20,
                          "median_ms": statistics.median(samples),
                          "samples_ms": samples, "bitwise_correct": True}), flush=True)


if __name__ == "__main__":
    main()
