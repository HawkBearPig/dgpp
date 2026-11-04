"""Independent mathematical reference for MiniMax's published Q8KV4 attention.

The precision boundaries follow SGLang 58484246 and MSA c8e1013a. This uses
contiguous tensors and FP32 PyTorch products; it does not import DGPP kernels
or reproduce their reductions. MSA has distinct prefill and decode formulas:
prefill combines BF16 page outputs, decode keeps an online FP32 accumulator.
Both quantize unnormalized probabilities to E4M3 with a factor of 448.
See docs/minimax_native_precision.md for the pinned sources and limitations.
"""

import math

import torch

from minimax_reference_quantization import e2m1


def queries(x):
    """The published BF16 norm/RoPE output -> E4M3, with unit outer scale."""
    if x.dtype != torch.bfloat16 or x.shape[-1] != 128:
        raise ValueError('queries require BF16 128-wide heads')
    return x.to(torch.float8_e4m3fn)


def pack_cache(x):
    """BF16 heads -> packed E2M1 and E4M3 scales, one scale per 16 values.

    The cache's outer scale is one. This is different from the routed-expert
    activation quantizer, which chooses a dynamic global scale for each row.
    """
    if x.dtype != torch.bfloat16 or x.shape[-1] != 128:
        raise ValueError('cache quantization requires BF16 128-wide heads')
    groups = x.float().reshape(*x.shape[:-1], 8, 16)
    maximum = groups.abs().amax(-1, keepdim=True)
    scales = (maximum / 6).to(torch.float8_e4m3fn)
    inverse = torch.where(maximum != 0, scales.float().reciprocal(), 0)
    normalized = groups * inverse
    values = e2m1(normalized)
    # PTX's saturating E2M1 conversion maps NaN to +6 (e.g. 0 * inf in a
    # nonzero group whose scale underflows). The zero scale still decodes to 0.
    values = torch.where(torch.isnan(normalized), 6.0, values)
    lut = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6), device=x.device)
    codes = torch.searchsorted(lut, values.abs().contiguous()).to(torch.uint8)
    codes |= torch.signbit(values).to(torch.uint8) << 3
    codes = codes.reshape_as(x)
    return codes[..., ::2] | (codes[..., 1::2] << 4), scales.squeeze(-1)


def unpack_cache(packed, scales):
    """MSA QMUL4 semantics: E2M1 * E4M3, then saturating E4M3 rounding."""
    if packed.dtype != torch.uint8 or packed.shape[-1] != 64:
        raise ValueError('cache payload requires uint8 [..., 64]')
    if scales.dtype != torch.float8_e4m3fn or scales.shape != (*packed.shape[:-1], 8):
        raise ValueError('cache scales require E4M3 [..., 8]')
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    lut = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6,
                        -0.0, -.5, -1, -1.5, -2, -3, -4, -6), device=packed.device)
    values = lut[codes.long()] * scales.float().repeat_interleave(16, dim=-1)
    return values.clamp(-448, 448).to(torch.float8_e4m3fn)


def select_blocks(q, k, positions, topk=16, query_rows=32, score_fn=None):
    """FP32 index dots over FP8 queries/keys; causal page maxima and top-k.

    Force the current page into the selection. Equal page scores prefer the
    smaller page ID; output order is increasing page ID, as in DGPP's contract.
    Backend tie-breaking must be checked separately against upstream kernels.
    """
    if (q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn
            or q.ndim != 3 or q.shape[-1] != 128 or k.shape[1:] != (1, 128)
            or positions.shape != (len(q),) or not 1 <= topk <= 16 or query_rows < 1):
        raise ValueError('Q8 index geometry mismatch')
    result = torch.full((*q.shape[:2], topk), -1, device=q.device, dtype=torch.int32)
    blocks = math.ceil(len(k) / 128)
    key_positions = torch.arange(len(k), device=q.device)
    for start in range(0, len(q), query_rows):
        end = min(start + query_rows, len(q))
        scores = (torch.einsum('qhd,td->qht', q[start:end].float(), k[:, 0].float())
                  if score_fn is None else score_fn(q[start:end], k[:, 0]))
        visible = key_positions[None, :] <= positions[start:end, None]
        scores.masked_fill_(~visible[:, None, :], -torch.inf)
        scores = torch.nn.functional.pad(scores, (0, blocks * 128 - len(k)), value=-torch.inf)
        maxima = scores.reshape(end - start, q.shape[1], blocks, 128).amax(-1)
        current = (positions[start:end] // 128)[:, None, None].expand(-1, q.shape[1], 1)
        maxima.scatter_(2, current, torch.inf)
        chosen = maxima.argsort(dim=-1, descending=True, stable=True)[..., :topk].sort(-1).values
        chosen = torch.where(chosen * 128 <= positions[start:end, None, None], chosen, -1)
        result[start:end, :, :chosen.shape[-1]] = chosen.to(torch.int32)
    return result


def attention(q, k, v, selected, positions, phase='prefill', query_rows=8):
    """Q8KV4-world sparse attention after cache dequantization to E4M3.

    `phase` selects the upstream precision formula, independently of query row
    count. Accumulations are FP32 with TF32 disabled by the caller. This is a
    formula oracle, not a claim of bitwise agreement with upstream GPU GEMMs.
    """
    if (phase not in ('prefill', 'decode') or query_rows < 1
            or any(x.dtype != torch.float8_e4m3fn for x in (q, k, v))
            or q.ndim != 3 or k.ndim != 3 or q.shape[-1] != 128 or k.shape[-1] != 128
            or k.shape != v.shape or q.shape[1] % k.shape[1]
            or selected.shape[:2] != (len(q), k.shape[1])
            or not 1 <= selected.shape[-1] <= 16 or positions.shape != (len(q),)):
        raise ValueError('Q8KV4 attention geometry mismatch')
    output = torch.empty_like(q, dtype=torch.bfloat16)
    heads_per_kv = q.shape[1] // k.shape[1]
    offsets = torch.arange(128, device=q.device)
    k, v = k.float(), v.float()
    for start in range(0, len(q), query_rows):
        end = min(start + query_rows, len(q))
        for kv in range(k.shape[1]):
            head = slice(kv * heads_per_kv, (kv + 1) * heads_per_kv)
            pages = selected[start:end, kv].long()
            token_ids = pages[:, :, None] * 128 + offsets
            visible = ((pages >= 0)[:, :, None] & (token_ids < len(k))
                       & (token_ids <= positions[start:end, None, None]))
            safe = token_ids.clamp(0, len(k) - 1)
            keys, values = k[safe, kv], v[safe, kv]
            scores = torch.einsum('qhd,qptd->qhpt', q[start:end, head].float(), keys) * (128 ** -.5)
            scores.masked_fill_(~visible[:, None], -torch.inf)
            if phase == 'prefill':
                maxima = scores.amax(-1)
                finite = torch.isfinite(maxima)
                probability = torch.exp(scores - torch.where(finite, maxima, 0)[..., None])
                denominator = probability.sum(-1)
                probability8 = (probability * 448).to(torch.float8_e4m3fn).float() / 448
                partial = torch.einsum('qhpt,qptd->qhpd', probability8, values)
                partial = (partial / denominator.clamp_min(1)[..., None]).to(torch.bfloat16).float()
                lse = (maxima + denominator.clamp_min(1).log()).masked_fill(~finite, -torch.inf)
                weights = torch.exp(lse - torch.logsumexp(lse, dim=-1, keepdim=True))
                output[start:end, head] = torch.einsum('qhp,qhpd->qhd', weights, partial).to(torch.bfloat16)
            else:
                maximum = torch.full(scores.shape[:2], -torch.inf, device=q.device)
                denominator = torch.zeros_like(maximum)
                acc = torch.zeros((*maximum.shape, 128), device=q.device)
                for page in range(pages.shape[-1]):
                    new_maximum = torch.maximum(maximum, scores[:, :, page].amax(-1))
                    safe_maximum = torch.where(torch.isfinite(new_maximum), new_maximum, 0)
                    correction = torch.exp(maximum - safe_maximum)
                    probability = torch.exp(scores[:, :, page] - safe_maximum[..., None]) * 448
                    probability8 = probability.to(torch.float8_e4m3fn).float()
                    acc = acc * correction[..., None] + torch.bmm(probability8, values[:, page])
                    denominator = denominator * correction + probability.sum(-1)
                    maximum = new_maximum
                output[start:end, head] = (acc / denominator.clamp_min(1)[..., None]).to(torch.bfloat16)
    return output
