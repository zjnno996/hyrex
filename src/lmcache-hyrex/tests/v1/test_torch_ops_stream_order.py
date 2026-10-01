"""CUDA fallback copies must obey producers/consumers on a non-default stream."""
import pytest
import torch

from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.ops_types import TransferDirection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
def test_pointer_copy_obeys_current_stream(direction):
    gpu = torch.zeros(16384, device="cuda", dtype=torch.uint8)
    cpu = torch.full_like(gpu, 29, device="cpu", pin_memory=True)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        gpu.fill_(17)
        dest, source = (cpu, gpu) if direction == TransferDirection.D2H else (gpu, cpu)
        torch_ops.lmcache_memcpy_async(
            dest.data_ptr(), source.data_ptr(), source.numel(), direction, 2048, 4096)
    stream.synchronize()
    expected = 17 if direction == TransferDirection.D2H else 29
    assert torch.all(dest.cpu() == expected)
