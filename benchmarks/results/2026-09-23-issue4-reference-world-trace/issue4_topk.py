"""Diagnostic exact lexicographic top-k, with index-ordered output.

Float scores retain every bit; lower column index wins exact ties. The signed
64-bit key encodes ordered float32 bits followed by the inverse column index.
Invisible columns are excluded, including their uninitialized score storage.
NaNs are outside the QSA score contract. This is a reference control, not a
proposed production kernel.
"""
import torch


def select(logits, lengths, k):
    assert logits.dtype == torch.float32 and logits.shape[1] >= k
    width = logits.shape[1]
    index = torch.arange(width, device=logits.device, dtype=torch.int64)
    bits = logits.view(torch.int32)
    # Canonicalize signed zero and map negative floats into signed-order keys.
    bits = torch.where(logits == 0, 0, bits)
    ordered = torch.where(bits < 0, (~bits) ^ -2147483648, bits)
    keys = (ordered.to(torch.int64) << 32) | (4294967295 - index)
    keys.masked_fill_(index[None, :] >= lengths[:, None], -9223372036854775808)
    chosen = torch.topk(keys, k, dim=1, sorted=False).indices
    # Keep padding after all live blocks: expansion consumes the live prefix.
    chosen = torch.where(chosen < lengths[:, None], chosen, width)
    chosen = chosen.sort(dim=1).values
    return torch.where(chosen < width, chosen, -1).to(torch.int32)
