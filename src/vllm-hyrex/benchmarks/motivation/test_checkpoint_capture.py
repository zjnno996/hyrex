"""GPU mathematical check; no model or LMCache server required."""
import torch
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
from vllm.model_executor.layers.mamba.gdn.checkpoint_capture import (
    conv_checkpoint, recurrent_with_checkpoints,
)

torch.manual_seed(7)
for length, offsets in [(1054, (528,)), (1054, (1040,)), (151, (144,)), (663, (528,))]:
    shape = (1, length, 2, 128)
    q, k = [torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, length, 2, device="cuda") * 0.1
    beta = torch.rand(1, length, 2, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(1, 2, 128, 128, device="cuda") * 0.1
    args = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial,
                use_qk_l2norm_in_kernel=False)
    output, final, checkpoints = recurrent_with_checkpoints(
        chunk_gated_delta_rule, offsets=offsets, **args)
    reference, ref_final = chunk_gated_delta_rule(output_final_state=True, **args)
    # BF16 chunk regrouping need not be bit-identical. Check relative L2 error
    # as well as independently recomputed prefix states, not only final output.
    def check(a, b):
        error = torch.linalg.vector_norm(a.float()-b.float()) / torch.linalg.vector_norm(b.float()).clamp_min(1e-6)
        assert error < 0.025, error.item()
        return error.item()
    errors = [check(output, reference), check(final, ref_final)]
    for offset in offsets:
        prefix = {name: tensor[:, :offset].contiguous() for name, tensor in
                  dict(q=q, k=k, v=v, g=g, beta=beta).items()}
        _, expected = chunk_gated_delta_rule(**prefix, initial_state=initial,
                    output_final_state=True, use_qk_l2norm_in_kernel=False)
        errors.append(check(checkpoints[offset], expected))
        assert checkpoints[offset].data_ptr() != final.data_ptr()
    print(length, offsets, "relative L2 errors", errors, flush=True)

raw = torch.arange(30).reshape(10, 3)
previous = torch.arange(-12, 0).reshape(3, 4)
for offset in (1, 3, 4, 9, 10):
    expected = torch.cat((previous.T, raw[:offset]), dim=0)[-4:].T
    assert torch.equal(conv_checkpoint(raw, previous, offset), expected)
print("GDN recurrent and convolution checkpoint tests passed")
