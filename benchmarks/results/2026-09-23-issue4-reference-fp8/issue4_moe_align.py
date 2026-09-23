"""Diagnostic canonical Marlin grouping for the issue-4 TP-only reference.

Keep the original expert counts, block sizes and padding. Within each expert,
place flattened (token, top-k slot) IDs in ascending order. No routing decision,
weight or arithmetic kernel is changed. Expert parallel maps are outside this
control's supported scope.
"""
import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size as _original


def moe_align_block_size(topk_ids,block_size,num_experts,expert_map=None,
                         pad_sorted_ids=False,ignore_invalid_experts=False):
    if expert_map is not None:
        raise NotImplementedError('Issue4 canonical grouping control requires TP without expert parallel mapping')
    sorted_ids,experts,total=_original(topk_ids,block_size,num_experts,None,
        pad_sorted_ids=pad_sorted_ids,ignore_invalid_experts=ignore_invalid_experts)
    count=topk_ids.numel()
    slots=torch.arange(sorted_ids.numel(),device=sorted_ids.device,dtype=torch.int64)
    valid=slots<total
    # Do not sort or interpret uninitialized capacity beyond total.
    owner=experts.repeat_interleave(block_size)[:sorted_ids.numel()].to(torch.int64)
    owner=torch.where(valid,owner,num_experts)
    token=torch.where(valid,sorted_ids.to(torch.int64),count)
    keys=owner*(count+1)+token
    canonical=torch.sort(keys).values.remainder(count+1).to(torch.int32)
    return canonical,experts,total
