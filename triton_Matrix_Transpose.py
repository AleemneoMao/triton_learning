import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,
                            BLOCK_SIZE: tl.constexpr):
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    offs_r = pid_r * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_c = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows_2d = offs_r[:, None]
    cols_2d = offs_c[None, :]
    mask = (rows_2d < rows) & (cols_2d < cols)

    input_d = tl.load(input + rows_2d * stride_ir + cols_2d * stride_ic, mask=mask)
    output_d = tl.store(output + cols_2d * stride_or + rows_2d * stride_oc, input_d, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1
    BLOCK_SIZE = 32
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE),)
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, BLOCK_SIZE
    )
