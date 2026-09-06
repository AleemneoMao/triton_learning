import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 32},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 32},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_N": 32},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 32},
            num_warps=8, num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matrix_multiplication_kernel(
    a, b, c,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_an: tl.constexpr,
    stride_bn: tl.constexpr, stride_bk: tl.constexpr,
    stride_cm: tl.constexpr, stride_ck: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    num_m = tl.cdiv(M, BLOCK_M)
    num_k = tl.cdiv(K, BLOCK_K)

    # 分组调度，提高相邻计算块对 B 的缓存复用率。
    GROUP_M: tl.constexpr = 8
    group_size = GROUP_M * num_k
    group_id = pid // group_size
    first_m = group_id * GROUP_M
    actual_group_m = tl.minimum(num_m - first_m, GROUP_M)

    pid_in_group = pid % group_size
    pid_m = first_m + pid_in_group % actual_group_m
    pid_k = pid_in_group // actual_group_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    reduction = tl.arange(0, BLOCK_N)

    a_ptrs = (
        a
        + rows[:, None] * stride_am
        + reduction[None, :] * stride_an
    )
    b_ptrs = (
        b
        + reduction[:, None] * stride_bn
        + cols[None, :] * stride_bk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for block in range(tl.cdiv(N, BLOCK_N)):
        valid_n = block * BLOCK_N + reduction < N

        a_tile = tl.load(
            a_ptrs,
            mask=(rows[:, None] < M) & valid_n[None, :],
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=valid_n[:, None] & (cols[None, :] < K),
            other=0.0,
        )

        acc = tl.dot(a_tile, b_tile, acc, input_precision="tf32x3")

        a_ptrs += BLOCK_N * stride_an
        b_ptrs += BLOCK_N * stride_bn

    c_ptrs = (
        c
        + rows[:, None] * stride_cm
        + cols[None, :] * stride_ck
    )
    tl.store(
        c_ptrs,
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < K),
    )


def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"])
        * triton.cdiv(K, meta["BLOCK_K"]),
    )

    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
    )