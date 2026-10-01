"""Single-request GDN snapshots within one transformer forward.

Only the recurrent operator is segmented. Projections, Full Attention, MLPs
and the model scheduler need not run once per checkpoint. Experimental eager
path; callers own snapshot allocation and lifetime until CPU STORE completes.
"""
import torch


def recurrent_with_checkpoints(kernel, *, q, k, v, g, beta, initial_state,
                               offsets, **kwargs):
    if q.shape[0] != 1:
        raise ValueError("checkpoint capture currently supports one sequence")
    length = q.shape[1]
    offsets = tuple(offsets)
    if offsets != tuple(sorted(set(offsets))) or any(p <= 0 or p > length for p in offsets):
        raise ValueError("checkpoint offsets must be unique, increasing, and in sequence")
    forbidden = {"cu_seqlens", "chunk_indices", "chunk_offsets", "output_final_state", "core_attn_out"}
    if forbidden.intersection(kwargs):
        raise ValueError("segment metadata must be regenerated, not reused")
    outputs, snapshots = [], {}
    state = initial_state
    start = 0
    ends = offsets if offsets and offsets[-1] == length else (*offsets, length)
    for end in ends:
        output, state = kernel(
            q=q[:, start:end].contiguous(), k=k[:, start:end].contiguous(),
            v=v[:, start:end].contiguous(), g=g[:, start:end].contiguous(),
            beta=beta[:, start:end].contiguous(), initial_state=state,
            output_final_state=True, **kwargs)
        outputs.append(output)
        if end in offsets:
            snapshots[end] = state.clone()
        start = end
    return torch.cat(outputs, dim=1), state, snapshots


def conv_checkpoint(raw_projection, initial_conv_state, offset):
    """Copy the *pre-convolution* history at offset, including initial history.

    raw_projection: [tokens, channels]; initial state: [channels, width].
    Never derive this from activated convolution output or final conv state.
    """
    if not 0 < offset <= raw_projection.shape[0]:
        raise ValueError("invalid convolution checkpoint offset")
    width = initial_conv_state.shape[-1]
    if raw_projection.shape[1] != initial_conv_state.shape[0]:
        raise ValueError("convolution channel mismatch")
    if offset >= width:
        return raw_projection[offset-width:offset].T.contiguous().clone()
    return torch.cat((initial_conv_state[:, offset:], raw_projection[:offset].T), dim=1)
