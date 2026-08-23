import torch
import triton
import triton.language as tl

@triton.jit
def reduce_kernel(
        intput_ptr, partial_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input = tl.load(intput_ptr + offsets, mask = mask)
    score = tl.sum(input, axis = 0)
    tl.atomic_add(partial_ptr, score)

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    output.zero_()
    reduce_kernel[grid](input, output, N, BLOCK_SIZE)
    return output
