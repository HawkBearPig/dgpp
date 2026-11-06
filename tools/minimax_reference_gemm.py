"""Fixed-reduction BF16 projection for the independent MiniMax reference.

Every matrix shape uses the same 16-term tensor-core chain, without split-K
or a cuBLAS algorithm choice. MXFP8 projections may permute each 64-column
group to the physical order of dgpp's streaming projection. This changes
only accumulation order: the independently decoded weights and BF16 inputs
are multiplied without activation quantization. The ordinary order is used
for the grouped NVFP4 expert projections.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _projection(x, weight, result, rows, columns: tl.constexpr, inner: tl.constexpr,
                x_stride: tl.constexpr, weight_stride: tl.constexpr,
                streaming_order: tl.constexpr):
    m = tl.program_id(0)*16+tl.arange(0,16)
    n = tl.program_id(1)*32+tl.arange(0,32)
    lane = tl.arange(0,16)
    acc = tl.zeros((16,32),tl.float32)
    for block in range(tl.cdiv(inner,16)):
        if streaming_order:
            # The streaming MMA's four consecutive instructions cover one
            # 64-column group. The lane quad owns four 16-column runs.
            k = (block//4)*64+(block%4)*4+(lane%8//2)*16+lane%2+(lane//8)*2
        else:
            k = block*16+lane
        a = tl.load(x+m[:,None]*x_stride+k[None,:],
                    (m[:,None]<rows)&(k[None,:]<inner),other=0)
        b = tl.load(weight+n[None,:]*weight_stride+k[:,None],
                    (n[None,:]<columns)&(k[:,None]<inner),other=0)
        acc = tl.dot(a,b,acc)
    tl.store(result+m[:,None]*columns+n[None,:],acc,
              (m[:,None]<rows)&(n[None,:]<columns))


def projection(x, weight, streaming_order=False):
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError('fixed projection expects decoded BF16 inputs and weights')
    if x.stride(1) != 1 or weight.stride(1) != 1 or x.shape[1] != weight.shape[1]:
        raise ValueError('fixed projection requires contiguous inner dimensions')
    if streaming_order and x.shape[1]%64:
        raise ValueError('streaming projection requires a multiple of 64 columns')
    out = torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=torch.float32)
    _projection[(triton.cdiv(x.shape[0],16),triton.cdiv(weight.shape[0],32))](
        x,weight,out,x.shape[0],weight.shape[0],x.shape[1],x.stride(0),weight.stride(0),
        streaming_order,num_warps=4,enable_fp_fusion=False)
    return out


def self_test():
    torch.manual_seed(9017)
    x = torch.randn((35,6144),device='cuda',dtype=torch.bfloat16)
    weight = torch.randn((65,6144),device='cuda',dtype=torch.bfloat16)
    expected = x.double()@weight.double().T
    # A forward-error bound for a sequential FP32 dot; cancellation can make
    # relative error arbitrarily large, so the oracle uses each dot's L1 sum.
    unit_roundoff = torch.finfo(torch.float32).eps/2
    gamma = x.shape[1]*unit_roundoff/(1-x.shape[1]*unit_roundoff)
    bound = gamma*(x.double().abs()@weight.double().abs().T)
    basis = torch.zeros((64,6144),device='cuda',dtype=torch.bfloat16)
    indices = torch.arange(64,device='cuda')*96
    basis[torch.arange(64,device='cuda'),indices] = 1
    for order in (False,True):
        whole = projection(x,weight,order)
        rms = ((whole.double()-expected).norm()/expected.norm()).item()
        if not torch.all((whole.double()-expected).abs() <= bound):
            raise AssertionError(f'FP64 dot exceeds FP32 accumulation bound: {rms}')
        if not torch.equal(projection(basis,weight,order),weight[:,indices].T.float()):
            raise AssertionError('basis projection changed the weight/activation column mapping')
        for rows in (1,8,17,33):
            split = torch.cat([projection(part,weight,order) for part in x.split(rows)])
            if not torch.equal(whole,split):
                raise AssertionError(f'projection changed with {rows} rows per call')
        print(f'Fixed projection order={order}: FP64 relative RMS {rms:.6g}, exact row geometry',flush=True)


if __name__=='__main__':
    self_test()
