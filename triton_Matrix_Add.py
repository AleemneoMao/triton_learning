from nt import set_handle_inheritable

import torch
import triton
import triton.language as tl
import time
import numpy

# 方案 1：直接加法
@triton.jit
def matrix_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask = mask)
    b = tl.load(b_ptr + offsets, mask = mask)

    c = a + b
    tl.store(c_ptr + offsets, c, mask = mask)

def solve_triton_native(a:torch.tensor, b:torch.tensor, c:torch.tensor, N:int):
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)

# 方案 2：Triton 一维向量 + Autotune
@triton.autotune(
    configs = [
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 1}, num_warps = 4),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 2}, num_warps = 4),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_WIDTH': 2}, num_warps = 8),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_WIDTH': 4}, num_warps = 8),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_WIDTH': 8}, num_warps = 16),
    ],
    key = ['n_elements'],
)

@triton.jit
def matrix_add_kernel_1d(a_ptr, b_ptr, c_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr, VEC_WIDTH:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VEC_WIDTH

    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_WIDTH +tl.arange(0, VEC_WIDTH)[None, :]
    offsets = tl.reshape(offsets, (BLOCK_SIZE*VEC_WIDTH))
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask = mask, other = 0.0)
    b = tl.load(b_ptr + offsets, mask = mask, other = 0.0)

    c = a + b
    tl.store(c_ptr + offsets, c, mask = mask)

def solve_triton_1d(a:torch.tensor, b:torch.tensor, c:torch.tensor, N:int):
    n_elements = N * N
    grid =  lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VEC_WIDTH']),)
    matrix_add_kernel_1d[grid](a, b, c, n_elements)

# 方案 3：Triton 二维块指针 + Autotune
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def matrix_add_kernel_2d(a_ptr, b_ptr, c_ptr, N, BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    a_block_ptr = tl.make_block_ptr(
        base = a_ptr,
        shape = (N, N),
        strides = (N, 1),
        offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape = (BLOCK_M, BLOCK_N),
        order = (1,0),
    )

    b_block_ptr = tl.make_block_ptr(
        base = b_ptr,
        shape = (N, N),
        strides = (N, 1),
        offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape = (BLOCK_M, BLOCK_N),
        order = (1,0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(N, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    a = tl.load(a_block_ptr,boundary_check = (0,1))
    b = tl.load(b_block_ptr,boundary_check = (0,1))
    c = a + b
    tl.store(c_block_ptr, c, boundary_check = (0,1))

def solve_triton_2d(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    matrix_add_kernel_2d[grid](A, B, C, N)

# ------------------------------------------------------------
# 性能测试工具
# ------------------------------------------------------------

def benchmark(func, A, B, C, N, warmup = 10, repeat = 1000):

    # 返回平均耗时
    for _ in range(warmup):
        func(A, B, C, N)
        torch.cuda.synchronize()

    # 计时
    start = time.perf_counter()
    for i in range(repeat):
        func(A, B, C, N)
        torch.cuda.synchronize()
        end = time.perf_counter()
        avg_time_ms = (end - start) / repeat * 1000
        return avg_time_ms

def verify_results(C_triton_native, C_triton_1D, C_triton_2D):
        if torch.allclose(C_triton_native, C_triton_1D, atol=1e-5):
            print("✅ Naive Triton 与 Triton 1D 结果一致")
        else:
            print("❌ Naive Triton 与 Triton 1D 结果不一致")

        if torch.allclose(C_triton_native, C_triton_2D, atol=1e-5):
            print("✅ Naive Triton 与 Triton 2D 结果一致")
        else:
            print("❌ Naive Triton 与 Triton 2D 结果不一致")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用')
    device = torch.device('cuda')
    print(f"运行设备：{torch.cuda.get_device_name(device)}")

    N = 4096

    A = torch.randn(N, N,device=device,dtype=torch.float32)
    B = torch.randn(N, N, device=device,dtype=torch.float32)
    C_triton_native = torch.empty_like(A)
    C_triton_1d = torch.empty_like(A)
    C_triton_2d = torch.empty_like(A)

    print(f"\n矩阵大小：{N} * {N} {N*N}个元素")

    solve_triton_native(A, B, C_triton_native, N)
    solve_triton_1d(A, B, C_triton_1d, N)
    solve_triton_2d(A, B, C_triton_2d, N)
    verify_results(C_triton_native, C_triton_1d, C_triton_2d)

    print("\n开始性能测试 (预热 10 次，计时 100 次取平均)...\n")

    time_pytorch = benchmark(solve_triton_native, A, B, C_triton_native, N)
    time_triton_1d = benchmark(solve_triton_1d, A, B, C_triton_1d, N)
    time_triton_2d = benchmark(solve_triton_2d, A, B, C_triton_2d, N)

    print(f"Triton 直接加法:      {time_pytorch:.4f} ms")
    print(f"Triton 1D 向量化:      {time_triton_1d:.4f} ms")
    print(f"Triton 2D 块指针:      {time_triton_2d:.4f} ms")

    # 计算加速比
    baseline = time_pytorch
    print(f"\n相对于 Naive 的加速比:")
    print(f"  Triton 1D: {baseline / time_triton_1d:.2f}x")
    print(f"  Triton 2D: {baseline / time_triton_2d:.2f}x")

    # 计算内存带宽
    bytes_per_element = 4  # float32
    total_bytes = 3 * N * N * bytes_per_element  # A读 + B读 + C写
    bw_pytorch = total_bytes / (time_pytorch / 1000) / 1e9
    bw_triton_1d = total_bytes / (time_triton_1d / 1000) / 1e9
    bw_triton_2d = total_bytes / (time_triton_2d / 1000) / 1e9

    print(f"\n估算内存带宽 (GB/s):")
    print(f"  Triton Naive:  {bw_pytorch:.2f} GB/s")
    print(f"  Triton 1D: {bw_triton_1d:.2f} GB/s")
    print(f"  Triton 2D: {bw_triton_2d:.2f} GB/s")


if __name__ == "__main__":
    main()