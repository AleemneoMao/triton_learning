import triton
import triton.language as tl
import torch


@triton.jit
def dot_product_atomic_kernel(x_ptr, y_ptr, output_ptr, n_elements,
                              BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x * y)
    tl.atomic_add(output_ptr, partial)


def dot_product_atomic(x: torch.Tensor, y: torch.Tensor,
                       BLOCK_SIZE: int = 1024) -> torch.Tensor:
    n = x.numel()
    output = torch.zeros(1, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    dot_product_atomic_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def dot_product_partial_kernel(x_ptr, y_ptr, partial_out_ptr, n_elements,
                               BLOCK_SIZE: tl.constexpr):
    """Stage 1: 计算局部点积, 写入 per-block 缓冲区."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x * y)
    tl.store(partial_out_ptr + pid, partial)


@triton.jit
def reduce_final_kernel(partial_ptr, output_ptr, n_partials,
                        BLOCK_SIZE: tl.constexpr):
    """Stage 2: 将 partial 缓冲区归约为最终标量."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials

    vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(vals)
    tl.atomic_add(output_ptr, partial)


def dot_product_two_stage(x: torch.Tensor, y: torch.Tensor,
                          BLOCK_SIZE: int = 1024,
                          REDUCE_BLOCK: int = 1024) -> torch.Tensor:
    n = x.numel()
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    # Stage 1: 局部点积
    partial = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    grid1 = (num_blocks,)
    dot_product_partial_kernel[grid1](x, y, partial, n, BLOCK_SIZE=BLOCK_SIZE)

    # Stage 2: Triton 归约
    output = torch.zeros(1, device=x.device, dtype=torch.float32)
    num_reduce = triton.cdiv(num_blocks, REDUCE_BLOCK)
    grid2 = (num_reduce,)
    reduce_final_kernel[grid2](partial, output, num_blocks, BLOCK_SIZE=REDUCE_BLOCK)

    return output


def dot_product_pytorch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x, y)


def dot_product_pytorch_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y)


def benchmark(fn, x, y, ref, name, iters=100, warmup=10):
    for _ in range(warmup):
        fn(x, y)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(x, y)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters

    result = fn(x, y)
    result_val = result.item()
    ref_val = ref.item() if isinstance(ref, torch.Tensor) else ref

    error = abs(result_val - ref_val)
    rel_error = error / (abs(ref_val) + 1e-10)
    passed = rel_error < 1e-4

    bytes_read = 2 * x.numel() * x.element_size()
    bw = bytes_read / (elapsed_ms * 1e6)

    tag = "PASS" if passed else "FAIL"
    print(f"  [{name}]")
    print(f"    Time: {elapsed_ms:.4f} ms | BW: {bw:.2f} GB/s | {tag}")
    if not passed:
        print(f"    Debug: Ref={ref_val:.6f}, Result={result_val:.6f}, "
              f"RelError={rel_error:.6e}")

    return elapsed_ms, bw, passed


def main():
    configs = [
        ("32M",     33554432),
    ]

    for name, N in configs:
        print(f"--- N = {name} ({N:,} elements) ---")

        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.randn(N, device='cuda', dtype=torch.float32)
        ref = torch.dot(x.cpu().double(), y.cpu().double())

        benchmark(dot_product_atomic, x, y, ref,
                  "Atomic")
        benchmark(dot_product_two_stage, x, y, ref,
                  "Two-Stage")
        benchmark(dot_product_pytorch_dot, x, y, ref,
                  "PyTorch dot")
        benchmark(dot_product_pytorch_sum, x, y, ref,
                  "PyTorch sum(a*b)")
        print()


if __name__ == "__main__":
    main()