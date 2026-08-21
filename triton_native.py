import torch
import triton
import triton.language as tl
import time

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offests = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offests < N
    a = tl.load(a_ptr + offests, mask=mask)
    b = tl.load(b_ptr + offests, mask=mask)
    c = a + b
    tl.store(c_ptr + offests, c)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.is_contiguous() and y.is_contiguous()
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, N)
    return out

def main():
    N = 1 <<27
    a = torch.rand(N, device='cuda', dtype=torch.float32)
    b = torch.rand(N, device='cuda', dtype=torch.float32)
    c = add(a,b)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    c = add(a,b)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Time: {end - start:.6f} seconds.")

if __name__ == "__main__":
    main()