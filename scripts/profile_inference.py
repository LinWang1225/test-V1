"""Run warmed, same-process vLLM profiling; traces are diagnostic, not throughput."""
import argparse
import json
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--method', choices=['fp16', 'kvarn'], required=True)
p.add_argument('--prompts', type=Path, required=True, help='JSONL with prompt field')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--max-tokens', type=int, default=256)
p.add_argument('--max-model-len', type=int, default=32768)
p.add_argument('--profile', action='store_true')
a = p.parse_args()
import torch
from vllm import LLM, SamplingParams

if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: no benchmark performed')
prompts = [json.loads(line)['prompt'] for line in a.prompts.read_text().splitlines() if line.strip()]
a.out.mkdir(parents=True, exist_ok=True)
llm = LLM(model=a.model, dtype='float16', kv_cache_dtype='auto' if a.method == 'fp16' else 'kvarn_k4v2_g128', block_size=128, max_model_len=a.max_model_len, max_num_seqs=1, enable_prefix_caching=False, enforce_eager=a.profile, distributed_executor_backend='uni', seed=42)
params = SamplingParams(temperature=0, max_tokens=a.max_tokens)
llm.generate(['Warm up.'] * 2, SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
def run():
    rows = []
    for i, prompt in enumerate(prompts):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = llm.generate([prompt], params, use_tqdm=False)[0]
        torch.cuda.synchronize()
        rows.append(dict(index=i, elapsed_s=time.perf_counter()-start, input_tokens=len(result.prompt_token_ids), output_tokens=len(result.outputs[0].token_ids), text=result.outputs[0].text, finish_reason=result.outputs[0].finish_reason))
    return rows
if a.profile:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        rows = run()
    prof.export_chrome_trace(str(a.out / 'trace.json'))
    (a.out / 'kernels.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=100))
else:
    rows = run()
(a.out / 'generations.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
(a.out / 'metadata.json').write_text(json.dumps(dict(method=a.method, profile=a.profile, model=a.model, max_tokens=a.max_tokens, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), mode='serial batch=1', eager=a.profile), indent=2))
