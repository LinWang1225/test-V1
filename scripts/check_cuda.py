import json
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(),
                  "capability": torch.cuda.get_device_capability(),
                  "matmul": "passed"}, indent=2))
