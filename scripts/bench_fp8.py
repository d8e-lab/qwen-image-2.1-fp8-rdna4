"""Measure cost of fp8 weight-only (dequant-on-the-fly) vs bf16 on this GPU.
Small shapes only: hipBLASLt is broken here so big shapes fall back to a very slow path."""
import torch, time, json
import torch.nn.functional as F
torch.cuda.init(); torch.manual_seed(0)
dev='cuda'

def bench(name, fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

# Qwen-Image transformer: hidden 3072ish. Use C=4096 -> OUT=12288 (SwiGLU up/gate)
res={}
for (Tn,C,OUT) in [(256,4096,12288),(1024,4096,12288),(256,4096,4096)]:
    x=torch.randn(Tn,C,device=dev,dtype=torch.bfloat16).mul(0.05)
    W=torch.randn(OUT,C,device=dev,dtype=torch.bfloat16).mul(0.02)
    Wf8=W.to(torch.float8_e4m3fn)
    mb_bf16=W.numel()*2/2**20; mb_fp8=W.numel()/2**20
    t_bf16=bench(f'b16', lambda: F.linear(x,W))
    t_fp8 =bench(f'f8', lambda: F.linear(x,Wf8.to(torch.bfloat16)))
    key=f'{Tn}x{C}x{OUT}'
    res[key]={'bf16_ms':t_bf16,'fp8deq_ms':t_fp8,'overhead_x':t_fp8/t_bf16,
              'bf16_MiB':mb_bf16,'fp8_MiB':mb_fp8}
    print(f'{key:22s} bf16={t_bf16:7.3f}ms  fp8+deq={t_fp8:7.3f}ms  x{t_fp8/t_bf16:4.2f}  '
          f'({mb_bf16:.0f} -> {mb_fp8:.0f} MiB)')
json.dump(res,open('/home/helck/workspace/qwen-image/logs/bench_fp8.json','w'),indent=2)
