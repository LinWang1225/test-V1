#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
INSTALL_PID="${1:?installation process PID required}"
while kill -0 "$INSTALL_PID" 2>/dev/null; do sleep 10; done
if [[ ! -s locks/installed.freeze.txt ]]; then
  echo 'Installation did not finish successfully; see logs/install-environment.log.'
  exit 1
fi
export PATH="$ROOT/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
.venv/bin/uv pip check --python .venv/bin/python
.venv/bin/python scripts/check_cuda.py > logs/check-cuda.json 2>&1
.venv/bin/python scripts/smoke_inference.py --method fp16 > logs/smoke-fp16.log 2>&1
.venv/bin/python scripts/smoke_inference.py --method kvarn > logs/smoke-kvarn.log 2>&1
printf 'CUDA, FP16 generation, and KVarN generation passed.\n' > results/environment-verification.txt
