"""Small environment check; results are not a quality or performance benchmark."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("fp16", "kvarn"), required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/experiment.json").read_text())
from vllm import LLM, SamplingParams

llm = LLM(model=config["model_default"], dtype="float16",
          kv_cache_dtype=config["methods"][args.method], block_size=128,
          max_model_len=2048, max_num_seqs=1, gpu_memory_utilization=0.7,
          enforce_eager=True, enable_prefix_caching=False,
          distributed_executor_backend="uni", seed=42)
outputs = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
result = outputs[0].outputs[0]
assert result.token_ids, "No tokens generated"
print(json.dumps({"method": args.method, "tokens": len(result.token_ids),
                  "text": result.text, "check": "passed"}, ensure_ascii=False))
