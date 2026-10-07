# GPU sanity check: torch/CUDA/NCCL versions, bf16 matmul TFLOPS per GPU, H2D/D2H/D2D bandwidth, P2P. Run: .venv/bin/python -I tools/server/gpu-check.py
import os, time, torch
def bench(dev):
    torch.cuda.set_device(dev)
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16); b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); t = time.perf_counter(); k = 20
    for _ in range(k): a @ b
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    return 2 * n**3 * k / dt / 1e12
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i} {p.name} sm{p.major}{p.minor} {p.total_memory/2**30:.0f} GiB  bf16 matmul {bench(i):.0f} TFLOPS  p2p(0<->1)={torch.cuda.can_device_access_peer(i, 1-i)}")
# host<->device bandwidth
x = torch.empty(2**30, dtype=torch.uint8, pin_memory=True); y = torch.empty(2**30, dtype=torch.uint8, device=0)
torch.cuda.synchronize(); t=time.perf_counter(); y.copy_(x); torch.cuda.synchronize(); print(f"H2D pinned 1 GiB: {1/(time.perf_counter()-t):.1f} GB/s")
t=time.perf_counter(); x.copy_(y); torch.cuda.synchronize(); print(f"D2H pinned 1 GiB: {1/(time.perf_counter()-t):.1f} GB/s")
y1 = torch.empty(2**30, dtype=torch.uint8, device=1); torch.cuda.synchronize(); t=time.perf_counter(); y1.copy_(y); torch.cuda.synchronize(); print(f"D2D GPU0->GPU1 1 GiB: {1/(time.perf_counter()-t):.1f} GB/s")
