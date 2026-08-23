import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(
    image_ptr,
    width,
    height,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    rx = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    ry = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # rx[:, None] 是 (BLOCK_SIZE_X, 1)，ry[None, :] 是 (1, BLOCK_SIZE_Y)
    mask = (rx[:, None] < height) & (ry[None, :] < width)

    # 计算 1D 数组中的基础偏移量：(row * width + col) * 4（因为是RGBA）
    # 这里利用了广播机制
    base_offsets = (rx[:, None] * width + ry[None, :]) * 4

    for i in range(3):
        ptr = image_ptr + base_offsets + i
        val = tl.load(ptr, mask=mask)
        tl.store(ptr, 255 - val, mask=mask)

def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32

    # 计算网格大小
    grid = (
        triton.cdiv(height, BLOCK_SIZE_X),
        triton.cdiv(width, BLOCK_SIZE_Y)
    )

    # 启动核函数
    invert_kernel[grid](
        image,
        width,
        height,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        num_warps=8
    )

    return image