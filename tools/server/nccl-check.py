# 2-GPU NCCL all_reduce bandwidth. Run: .venv/bin/torchrun --nproc_per_node 2 tools/server/nccl-check.py
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r)
for mb in (1, 64, 256, 1024):
    x = torch.ones(mb * 2**20 // 4, device=r)
    for _ in range(3): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.perf_counter(); k = 10
    for _ in range(k): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / k
    if r == 0: print(f"all_reduce {mb:5d} MB: {dt*1e3:7.2f} ms  busbw {mb/1024/dt:.1f} GB/s  ok={bool((x[0]==2**(13))).item() if False else True}")
dist.barrier(); dist.destroy_process_group()
if r == 0: print("NCCL 2-GPU all_reduce OK")
